# /// script
# dependencies = ["torch>=2.4","transformers>=4.46","datasets>=3.0","accelerate>=1.0"]
# ///
"""
SmolLM2 schedule-matched Q9 -> Q3 control.

Tests the v9-selected global LR schedule on the established v7 Q9 -> Q3
protocol without any Q9-specific hyperparameter search.

Global LR schedule:
- steps 1-100: linear warmup to 1e-3;
- steps 101-1200: cosine decay to 1e-4.

Quantization schedule:
- global steps 1-300: Q9;
- transition: discard learned Q9 scales, restore original Q3 scales, fresh Adam;
- global steps 301-1200: Q3, continuing the same global LR curve.

Source helper lineage: v7 equal-compute geometry.

Original v7 description follows for provenance.

SmolLM2 ternary experiment v7: equal-compute preparation geometry.

Question:
After the same first 300 training chunks, does Q9 preparation produce
(a) a better immediate Q3 checkpoint than direct Q3 preparation,
(b) a master-weight geometry that is more trainable during the next 900
    ternary steps, or
(c) merely the same benefit as an unconstrained FP32-master warmup?

Preparation arms:
A: 300 updates with Q3 forward weights
B: 300 updates with Q9 forward weights
C: 300 updates with no weight quantization (FP32 masters; common autocast compute)

At step 300 all arms are projected through the SAME original Q3 scales and
scored on the SAME diagnostic set. Then every arm enters a 900-step Q3
continuation with fresh Adam and those same original scales. Thus only the
prepared FP32 master weights cross the boundary.

Additional diagnostics:
- exact projected-Q3 code Hamming/overlap among the three step-300 states;
- distance of master weights to the nearest future Q3 decision threshold;
- native preparation-space diagnostics (Q3, Q9, or unquantized);
- fixed held-out final evaluator and generation prompts.
"""
import gc, json, math, time, hashlib, os, tempfile
from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import parametrize
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

MODEL_ID="HuggingFaceTB/SmolLM2-360M-Instruct"
MODEL_REVISION="a10cc1512eabd3dde888204e902eca88bddb4951"
WIKITEXT_REVISION="b08601e04326c79dfdd32d625aee71d232d685c3"
DEVICE="cuda" if torch.cuda.is_available() else "cpu"
SEED=190027
SEQ=128
TRAIN_CHUNKS=1200
LRVAL_CHUNKS=24
EVAL_CHUNKS=64
LR_CANDIDATES=[2e-5,5e-5,1e-4]
SWEEP_STEPS=100
FLIP_EVERY=50
CE_W=0.35
KL_W=0.65

torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

PROMPTS=[
    "In two sentences, explain why a baseline matters in an experiment.",
    "What is 17 + 28? Give a short answer.",
    "Write one calm sentence about a dog waiting outside a library.",
    "What is one possible benefit of reducing numerical precision gradually?",
]

tok=AutoTokenizer.from_pretrained(MODEL_ID,revision=MODEL_REVISION)
if tok.pad_token_id is None:
    tok.pad_token=tok.eos_token

def stream_chunks(split,n):
    ds=load_dataset("Salesforce/wikitext","wikitext-2-raw-v1",split=split,revision=WIKITEXT_REVISION)
    buf=[]; out=[]
    for row in ds:
        t=row["text"]
        if not t or not t.strip():
            continue
        z=tok.encode(t,add_special_tokens=False)
        if not z:
            continue
        buf.extend(z+[tok.eos_token_id])
        while len(buf)>=SEQ+1 and len(out)<n:
            out.append(torch.tensor(buf[:SEQ+1],dtype=torch.long))
            buf=buf[SEQ+1:]
        if len(out)>=n:
            break
    if len(out)<n:
        raise RuntimeError((split,len(out),n))
    return out

all_train=stream_chunks("train",TRAIN_CHUNKS+LRVAL_CHUNKS)
train_chunks=all_train[:TRAIN_CHUNKS]
lrval_chunks=all_train[TRAIN_CHUNKS:]
# Confirmatory replication: seed controls training-order permutation.
_order_g=torch.Generator().manual_seed(SEED)
_order=torch.randperm(len(train_chunks),generator=_order_g).tolist()
train_chunks=[train_chunks[i] for i in _order]
eval_chunks=stream_chunks("test",EVAL_CHUNKS)
validation_chunks=stream_chunks("validation",64)

def load_model(dtype=torch.float32):
    m=AutoModelForCausalLM.from_pretrained(
        MODEL_ID,dtype=dtype,low_cpu_mem_usage=True,revision=MODEL_REVISION
    ).to(DEVICE)
    m.config.use_cache=False
    return m

@torch.no_grad()
def bf16_roundtrip_(m):
    for p in m.parameters():
        if p.is_floating_point():
            p.copy_(p.to(torch.bfloat16).to(p.dtype))

def target_linears(m):
    for name,mod in m.named_modules():
        if isinstance(mod,nn.Linear) and name!="lm_head":
            yield name,mod

def inv_softplus(x):
    return torch.where(x>20,x,torch.log(torch.expm1(x.clamp_min(1e-7))))

class SharedScaleQuant(nn.Module):
    def __init__(self,w,levels=3):
        super().__init__()
        with torch.no_grad():
            meanabs=w.detach().abs().mean(dim=1,keepdim=True).float().clamp_min(1e-7)
            init_alpha=1.5*meanabs
        self.raw_alpha=nn.Parameter(inv_softplus(init_alpha))
        self.levels=int(levels)
        self.q9_mode="W"

    def alpha(self):
        return F.softplus(self.raw_alpha)+1e-7

    def _n(self):
        if self.levels==3:
            return 1.5
        if self.levels==9:
            return 4.0
        raise ValueError(self.levels)

    def hard_code(self,w):
        a=self.alpha().to(device=w.device,dtype=w.dtype)
        n=self._n()
        z=(w/a).clamp(-0.99,0.99)
        # integer code: -1..1 for ternary, -4..4 for 9-state.
        return torch.round(z*n).to(torch.int8)

    def forward(self,w):
        a=self.alpha().to(dtype=w.dtype)
        n=self._n()
        z=(w/a).clamp(-0.99,0.99)
        k=torch.round(z*n)
        hard=k/n
        if self.levels==9 and self.q9_mode in ("N","H"):
            hard=k/6.0
            if self.q9_mode=="H":
                # Restore only the outermost +/-4 reconstruction magnitudes.
                hard=hard+torch.where(k.abs()==4,k.sign()/3.0,torch.zeros_like(k))
        # STE: forward sees hard grid, backward sees clipped z.
        qcode=z+(hard-z).detach()
        return qcode*a

def attach_quantizers(m,levels):
    # Freeze everything first. This prevents embeddings/lm_head/norms from
    # quietly compensating for a frozen ternary backbone.
    for p in m.parameters():
        p.requires_grad_(False)
    qs=[]
    trainable=[]
    for name,mod in target_linears(m):
        q=SharedScaleQuant(mod.weight,levels=levels)
        parametrize.register_parametrization(mod,"weight",q)
        shadow=mod.parametrizations.weight.original
        shadow.requires_grad_(True)
        q.raw_alpha.requires_grad_(True)
        trainable.extend([shadow,q.raw_alpha])
        qs.append((name,mod,q))
    return qs,trainable

def set_levels(qs,levels):
    for _,_,q in qs:
        q.levels=int(levels)

@torch.no_grad()
def code_hist(qs):
    counts=defaultdict(int); total=0
    for _,mod,q in qs:
        w=mod.parametrizations.weight.original
        c=q.hard_code(w).reshape(-1).cpu().to(torch.int16)
        for val,n in zip(*torch.unique(c,return_counts=True)):
            counts[int(val.item())]+=int(n.item())
        total+=c.numel()
    return {str(k):counts[k]/total for k in sorted(counts)}

@torch.no_grad()
def snapshot_codes(qs):
    out={}
    for name,mod,q in qs:
        w=mod.parametrizations.weight.original
        out[name]=q.hard_code(w).detach().cpu().clone()
    return out

@torch.no_grad()
def flip_stats(qs,prev,start):
    changed=0; from_start=0; total=0; layer_rows=[]
    current={}
    for name,mod,q in qs:
        w=mod.parametrizations.weight.original
        c=q.hard_code(w).detach().cpu()
        p=prev[name]; s=start[name]
        ch=int((c!=p).sum().item())
        cs=int((c!=s).sum().item())
        n=c.numel()
        changed+=ch; from_start+=cs; total+=n
        layer_rows.append((ch/max(1,n),name,cs/max(1,n)))
        current[name]=c.clone()
    layer_rows.sort(reverse=True)
    return current,{
        "flip_since_last":changed/max(1,total),
        "different_from_stage_start":from_start/max(1,total),
        "top_layers":[
            {"name":name,"flip_since_last":fr,"different_from_start":fs}
            for fr,name,fs in layer_rows[:5]
        ]
    }

@torch.no_grad()
def scale_stats(qs):
    vals=torch.cat([q.alpha().detach().float().cpu().reshape(-1) for _,_,q in qs])
    return {
        "mean":vals.mean().item(),"std":vals.std().item(),
        "min":vals.min().item(),"max":vals.max().item()
    }

@torch.no_grad()
def score_on(m,teacher,chunks_):
    m.eval(); teacher.eval()
    nll=0.; nt=0; agree=0; an=0; kls=0.; kn=0
    for ids_cpu in chunks_:
        x=ids_cpu[:-1].unsqueeze(0).to(DEVICE)
        y=ids_cpu[1:].unsqueeze(0).to(DEVICE)
        with torch.autocast("cuda",dtype=torch.float16,enabled=(DEVICE=="cuda")):
            s=m(x).logits.float()
            t=teacher(x).logits.float()
        ce=F.cross_entropy(s.reshape(-1,s.shape[-1]),y.reshape(-1),reduction="sum")
        nll+=ce.item(); nt+=y.numel()
        sp=s.argmax(-1); tp=t.argmax(-1)
        agree+=(sp==tp).sum().item(); an+=tp.numel()
        kl=F.kl_div(
            F.log_softmax(s,dim=-1),F.softmax(t,dim=-1),reduction="none"
        ).sum(-1)
        kls+=kl.sum().item(); kn+=kl.numel()
    loss=nll/nt
    return {
        "loss":loss,"ppl":math.exp(min(loss,20)),
        "top1_agreement":agree/an,"kl_to_teacher":kls/kn,
        "tokens":nt
    }

@torch.no_grad()
def generate(m):
    m.eval(); m.config.use_cache=True; out=[]
    for p in PROMPTS:
        text=tok.apply_chat_template(
            [{"role":"user","content":p}],
            tokenize=False,add_generation_prompt=True
        )
        e=tok(text,return_tensors="pt").to(DEVICE)
        g=m.generate(**e,max_new_tokens=48,do_sample=False,
                     pad_token_id=tok.eos_token_id)
        out.append(tok.decode(g[0,e.input_ids.shape[1]:],skip_special_tokens=True))
    m.config.use_cache=False
    return out

def make_optimizer(trainable,lr):
    return torch.optim.AdamW(trainable,lr=lr,betas=(0.9,0.95),weight_decay=0.0)

def one_step(m,teacher,opt,scaler,ids_cpu):
    x=ids_cpu[:-1].unsqueeze(0).to(DEVICE)
    y=ids_cpu[1:].unsqueeze(0).to(DEVICE)
    opt.zero_grad(set_to_none=True)
    with torch.no_grad():
        with torch.autocast("cuda",dtype=torch.float16,enabled=(DEVICE=="cuda")):
            t=teacher(x).logits
    with torch.autocast("cuda",dtype=torch.float16,enabled=(DEVICE=="cuda")):
        s=m(x).logits
        ce=F.cross_entropy(s.reshape(-1,s.shape[-1]),y.reshape(-1))
        kl=F.kl_div(
            F.log_softmax(s.float(),dim=-1),
            F.softmax(t.float(),dim=-1),
            reduction="none"
        ).sum(-1).mean()
        loss=CE_W*ce+KL_W*kl
    scaler.scale(loss).backward()
    scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(
        [p for group in opt.param_groups for p in group["params"]],1.0
    )
    scaler.step(opt); scaler.update()
    return float(loss.detach().cpu()),float(ce.detach().cpu()),float(kl.detach().cpu())

def cleanup(*objs):
    for x in objs:
        del x
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def lr_sweep(teacher):
    rows=[]
    for lr in LR_CANDIDATES:
        m=load_model(torch.float32); bf16_roundtrip_(m)
        qs,trainable=attach_quantizers(m,3)
        initial_hist=code_hist(qs)
        stage_start=snapshot_codes(qs); prev=stage_start
        opt=make_optimizer(trainable,lr)
        scaler=torch.amp.GradScaler("cuda",enabled=(DEVICE=="cuda"))
        losses=[]; flips=[]
        for step in range(SWEEP_STEPS):
            loss,ce,kl=one_step(m,teacher,opt,scaler,train_chunks[step])
            losses.append(loss)
            if (step+1)%25==0:
                prev,fs=flip_stats(qs,prev,stage_start)
                fs["step"]=step+1
                flips.append(fs)
                print(json.dumps({
                    "event":"lr_sweep","lr":lr,"step":step+1,
                    "loss":loss,"ce":ce,"kl":kl,**fs
                }),flush=True)
        val=score_on(m,teacher,lrval_chunks)
        row={
            "lr":lr,"first_loss":losses[0],"last_loss":losses[-1],
            "val":val,"initial_hist":initial_hist,"final_hist":code_hist(qs),
            "scale":scale_stats(qs),"flips":flips
        }
        rows.append(row)
        del opt,scaler,m,qs,trainable,prev,stage_start
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    # Pick lowest validation loss. Flip rates remain explicit diagnostics, so a
    # suspicious near-zero-flip winner will be obvious in the record.
    best=min(rows,key=lambda r:r["val"]["loss"])
    return rows,best["lr"]

def run_main(label,schedule,lr,teacher):
    m=load_model(torch.float32); bf16_roundtrip_(m)
    first_level=schedule[0][0]
    qs,trainable=attach_quantizers(m,first_level)
    opt=make_optimizer(trainable,lr)
    scaler=torch.amp.GradScaler("cuda",enabled=(DEVICE=="cuda"))
    all_trace=[]; global_step=0; stage_summaries=[]
    start_time=time.time()

    for stage_i,(levels,steps) in enumerate(schedule):
        set_levels(qs,levels)
        # Do NOT recreate optimizer: momentum/state persists across 9 -> 3.
        stage_start=snapshot_codes(qs)
        prev=stage_start
        start_hist=code_hist(qs)
        stage_trace=[]
        for local in range(steps):
            ids=train_chunks[global_step%len(train_chunks)]
            loss,ce,kl=one_step(m,teacher,opt,scaler,ids)
            global_step+=1
            if local==0 or (local+1)%FLIP_EVERY==0 or local+1==steps:
                prev,fs=flip_stats(qs,prev,stage_start)
                row={
                    "global_step":global_step,"stage_step":local+1,
                    "levels":levels,"loss":loss,"ce":ce,"kl":kl,
                    **fs
                }
                stage_trace.append(row); all_trace.append(row)
                print(json.dumps({"event":"main","label":label,**row}),flush=True)
        stage_summaries.append({
            "levels":levels,"steps":steps,
            "start_hist":start_hist,"end_hist":code_hist(qs),
            "scale":scale_stats(qs),"trace":stage_trace
        })
        del prev,stage_start
        gc.collect()

    metrics=score_on(m,teacher,eval_chunks)
    metrics["gen"]=generate(m)
    metrics["final_hist"]=code_hist(qs)
    metrics["scale"]=scale_stats(qs)
    metrics["stages"]=stage_summaries
    metrics["seconds"]=time.time()-start_time

    del opt,scaler,m,qs,trainable
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return metrics



@torch.no_grad()
def capture_scale_state(qs):
    out={}
    for name,mod,q in qs:
        out[name]={
            "raw_alpha":q.raw_alpha.detach().cpu().clone(),
            "alpha":q.alpha().detach().cpu().clone(),
        }
    return out

@torch.no_grad()
def capture_quantized_masters(qs):
    return {
        name:mod.parametrizations.weight.original.detach().cpu().clone()
        for name,mod,q in qs
    }

@torch.no_grad()
def capture_plain_masters(m):
    return {
        name:mod.weight.detach().cpu().clone()
        for name,mod in target_linears(m)
    }

@torch.no_grad()
def load_masters_and_scales(qs,masters,scale_state):
    for name,mod,q in qs:
        w=mod.parametrizations.weight.original
        w.copy_(masters[name].to(device=w.device,dtype=w.dtype))
        raw=scale_state[name]["raw_alpha"]
        q.raw_alpha.copy_(raw.to(device=q.raw_alpha.device,dtype=q.raw_alpha.dtype))
        q.levels=3

@torch.no_grad()
def threshold_margin_stats(qs):
    # Q3 boundaries in normalized z=w/alpha coordinates are +/- 1/3 because
    # round(1.5*z) changes integer code at +/-0.5.
    epses=[0.005,0.01,0.02,0.05,0.10]
    counts={str(e):0 for e in epses}
    total=0
    margin_sum=0.0
    layer_rows=[]
    for name,mod,q in qs:
        w=mod.parametrizations.weight.original.detach().float()
        a=q.alpha().detach().to(device=w.device,dtype=w.dtype)
        z=w/a
        margin=torch.minimum((z-(1.0/3.0)).abs(),(z+(1.0/3.0)).abs())
        n=margin.numel()
        total+=n
        margin_sum+=margin.sum().item()
        layer_counts={}
        for e in epses:
            c=int((margin<e).sum().item())
            counts[str(e)]+=c
            layer_counts[str(e)]=c/max(1,n)
        layer_rows.append((layer_counts["0.02"],name,layer_counts))
    layer_rows.sort(reverse=True)
    return {
        "mean_normalized_distance":margin_sum/max(1,total),
        "fraction_within":{k:v/max(1,total) for k,v in counts.items()},
        "top_layers_within_0.02":[
            {"name":name,"fraction_within":fracs}
            for _,name,fracs in layer_rows[:8]
        ]
    }

@torch.no_grad()
def hamming_codes(a,b):
    changed=0; total=0; rows=[]
    for name in a:
        x=a[name]; y=b[name]
        ch=int((x!=y).sum().item()); n=x.numel()
        changed+=ch; total+=n
        rows.append((ch/max(1,n),name))
    rows.sort(reverse=True)
    return {
        "fraction":changed/max(1,total),
        "top_layers":[{"name":name,"fraction":fr} for fr,name in rows[:8]]
    }

@torch.no_grad()
def code_overlap_against_initial(initial,a,b):
    total=0
    ca=cb=both=union=same_final_both=0
    for name in initial:
        i=initial[name]; x=a[name]; y=b[name]
        ma=(x!=i); mb=(y!=i)
        n=i.numel(); total+=n
        ca+=int(ma.sum().item()); cb+=int(mb.sum().item())
        bi=ma & mb
        both+=int(bi.sum().item())
        union+=int((ma|mb).sum().item())
        same_final_both+=int((bi & (x==y)).sum().item())
    return {
        "a_changed_fraction":ca/max(1,total),
        "b_changed_fraction":cb/max(1,total),
        "changed_set_intersection_fraction":both/max(1,total),
        "changed_set_union_fraction":union/max(1,total),
        "changed_set_jaccard":both/max(1,union),
        "same_final_code_among_both_changed":same_final_both/max(1,both),
        "pairwise_hamming":hamming_codes(a,b)["fraction"]
    }

def initial_q3_reference(teacher):
    m=load_model(torch.float32); bf16_roundtrip_(m)
    qs,trainable=attach_quantizers(m,3)
    scales=capture_scale_state(qs)
    codes=snapshot_codes(qs)
    out={
        "diag":score_on(m,teacher,lrval_chunks),
        "hist":code_hist(qs),
        "scale":scale_stats(qs),
        "margin":threshold_margin_stats(qs)
    }
    del m,qs,trainable
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return scales,codes,out

def prepare_quantized(label,levels,teacher,lr):
    m=load_model(torch.float32); bf16_roundtrip_(m)
    qs,trainable=attach_quantizers(m,levels)
    opt=make_optimizer(trainable,lr)
    scaler=torch.amp.GradScaler("cuda",enabled=(DEVICE=="cuda"))
    trace=[]
    for step in range(300):
        loss,ce,kl=one_step(m,teacher,opt,scaler,train_chunks[step])
        if step==0 or (step+1)%50==0 or step+1==300:
            row={"step":step+1,"loss":loss,"ce":ce,"kl":kl}
            trace.append(row)
            print(json.dumps({"event":"v7_prep","label":label,**row}),flush=True)
    native_diag=score_on(m,teacher,lrval_chunks)
    masters=capture_quantized_masters(qs)
    out={
        "native_space":"q3" if levels==3 else "q9",
        "native_diag":native_diag,
        "learned_scale":scale_stats(qs),
        "trace":trace
    }
    del opt,scaler,m,qs,trainable
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return masters,out

def prepare_fp32(label,teacher,lr):
    m=load_model(torch.float32); bf16_roundtrip_(m)
    for p in m.parameters():
        p.requires_grad_(False)
    trainable=[]
    for name,mod in target_linears(m):
        mod.weight.requires_grad_(True)
        trainable.append(mod.weight)
    opt=make_optimizer(trainable,lr)
    scaler=torch.amp.GradScaler("cuda",enabled=(DEVICE=="cuda"))
    trace=[]
    for step in range(300):
        loss,ce,kl=one_step(m,teacher,opt,scaler,train_chunks[step])
        if step==0 or (step+1)%50==0 or step+1==300:
            row={"step":step+1,"loss":loss,"ce":ce,"kl":kl}
            trace.append(row)
            print(json.dumps({"event":"v7_prep","label":label,**row}),flush=True)
    native_diag=score_on(m,teacher,lrval_chunks)
    masters=capture_plain_masters(m)
    out={
        "native_space":"unquantized",
        "native_diag":native_diag,
        "trace":trace
    }
    del opt,scaler,m,trainable
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return masters,out

def project_and_continue(label,masters,reference_scales,initial_codes,teacher,lr):
    m=load_model(torch.float32); bf16_roundtrip_(m)
    qs,trainable=attach_quantizers(m,3)
    load_masters_and_scales(qs,masters,reference_scales)

    start_codes=snapshot_codes(qs)
    pre={
        "diag":score_on(m,teacher,lrval_chunks),
        "hist":code_hist(qs),
        "margin":threshold_margin_stats(qs),
        "vs_initial_codes":hamming_codes(initial_codes,start_codes)
    }

    # All arms receive the same fresh optimizer and same original scales here.
    opt=make_optimizer(trainable,lr)
    scaler=torch.amp.GradScaler("cuda",enabled=(DEVICE=="cuda"))
    stage_start=start_codes
    prev={k:v.clone() for k,v in stage_start.items()}
    trace=[]
    start_time=time.time()
    for local in range(900):
        loss,ce,kl=one_step(m,teacher,opt,scaler,train_chunks[300+local])
        if local==0 or (local+1)%FLIP_EVERY==0 or local+1==900:
            prev,fs=flip_stats(qs,prev,stage_start)
            row={"step":local+1,"loss":loss,"ce":ce,"kl":kl,**fs}
            trace.append(row)
            print(json.dumps({"event":"v7_continue","label":label,**row}),flush=True)

    final=score_on(m,teacher,eval_chunks)
    final["gen"]=generate(m)
    final["final_hist"]=code_hist(qs)
    final["scale"]=scale_stats(qs)
    final["trace"]=trace
    final["seconds"]=time.time()-start_time

    del opt,scaler,m,qs,trainable,prev,stage_start
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return pre,final,start_codes

PEAK_LR=1e-3
MIN_LR=1e-4
WARMUP_STEPS=100
FULL_STEPS=1200
PREP_STEPS=300
CONT_STEPS=900

def matched_lr(global_step0):
    s=global_step0+1
    if s<=WARMUP_STEPS:
        return PEAK_LR*s/WARMUP_STEPS
    frac=(s-WARMUP_STEPS)/max(1,FULL_STEPS-WARMUP_STEPS)
    frac=min(max(frac,0.0),1.0)
    return MIN_LR+0.5*(PEAK_LR-MIN_LR)*(1.0+math.cos(math.pi*frac))

def set_optimizer_lr(opt,lr):
    for group in opt.param_groups:
        group["lr"]=float(lr)

def prepare_q9_matched(teacher):
    m=load_model(torch.float32); bf16_roundtrip_(m)
    qs,trainable=attach_quantizers(m,9)
    opt=make_optimizer(trainable,matched_lr(0))
    scaler=torch.amp.GradScaler("cuda",enabled=(DEVICE=="cuda"))
    trace=[]
    stage_start=snapshot_codes(qs)
    prev={k:v.clone() for k,v in stage_start.items()}
    start_time=time.time()

    for global_step0 in range(PREP_STEPS):
        lr=matched_lr(global_step0)
        set_optimizer_lr(opt,lr)
        loss,ce,kl=one_step(m,teacher,opt,scaler,train_chunks[global_step0])
        if global_step0==0 or (global_step0+1)%50==0 or global_step0+1==PREP_STEPS:
            prev,fs=flip_stats(qs,prev,stage_start)
            row={
                "global_step":global_step0+1,"levels":9,"lr":lr,
                "loss":loss,"ce":ce,"kl":kl,**fs
            }
            trace.append(row)
            print(json.dumps({"event":"matched_q9_prep",**row}),flush=True)

    masters=capture_quantized_masters(qs)
    out={
        "native_diag":score_on(m,teacher,lrval_chunks),
        "learned_q9_scale":scale_stats(qs),
        "hist":code_hist(qs),
        "trace":trace,
        "seconds":time.time()-start_time
    }
    del opt,scaler,m,qs,trainable,prev,stage_start
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return masters,out

def continue_q3_matched(masters,reference_scales,initial_codes,teacher):
    m=load_model(torch.float32); bf16_roundtrip_(m)
    qs,trainable=attach_quantizers(m,3)
    load_masters_and_scales(qs,masters,reference_scales)

    start_codes=snapshot_codes(qs)
    pre={
        "diag":score_on(m,teacher,lrval_chunks),
        "hist":code_hist(qs),
        "margin":threshold_margin_stats(qs),
        "vs_initial_codes":hamming_codes(initial_codes,start_codes),
        "first_q3_lr":matched_lr(PREP_STEPS)
    }

    # v7 transition semantics: fresh Adam + original Q3 scales.
    # Only the global LR curve continues; optimizer moments do not.
    opt=make_optimizer(trainable,matched_lr(PREP_STEPS))
    scaler=torch.amp.GradScaler("cuda",enabled=(DEVICE=="cuda"))
    stage_start=start_codes
    prev={k:v.clone() for k,v in stage_start.items()}
    trace=[]
    start_time=time.time()

    for local in range(CONT_STEPS):
        global_step0=PREP_STEPS+local
        lr=matched_lr(global_step0)
        set_optimizer_lr(opt,lr)
        loss,ce,kl=one_step(m,teacher,opt,scaler,train_chunks[global_step0])
        if local==0 or (local+1)%50==0 or local+1==CONT_STEPS:
            prev,fs=flip_stats(qs,prev,stage_start)
            row={
                "global_step":global_step0+1,"stage_step":local+1,
                "levels":3,"lr":lr,"loss":loss,"ce":ce,"kl":kl,**fs
            }
            trace.append(row)
            print(json.dumps({"event":"matched_q3_continue",**row}),flush=True)

    final=score_on(m,teacher,eval_chunks)
    final["gen"]=generate(m)
    final["final_hist"]=code_hist(qs)
    final["scale"]=scale_stats(qs)
    final["trace"]=trace
    final["seconds"]=time.time()-start_time

    final_codes=snapshot_codes(qs)
    del opt,scaler,m,qs,trainable,prev,stage_start
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return pre,final,start_codes,final_codes


R5_PROTOCOL="ternary_pet/research_log/r6_imdb_two_seed_prereg_2026-10-10.md"
R5_FINEWEB_REV="e628166"
R5_BUCKET=7
R5_SOURCE_MIN_ROW=1
R5_CKPT_REPO="codeflash85/ternary-pet-r5-checkpoints"

def r5_select_new_documents():
    import hashlib
    source=load_dataset("stanfordnlp/imdb",split="test",streaming=True,revision=R5_FINEWEB_REV)
    docs=[];seen=set();scanned=0;eligible=0;too_short=0
    for rownum,item in enumerate(source,1):
        scanned=rownum
        if rownum<R5_SOURCE_MIN_ROW:continue
        if rownum>25000:break
        docid=str(rownum)+":"+hashlib.sha256(str(item["text"]).encode("utf-8")).hexdigest()
        digest=hashlib.sha256(docid.encode("utf-8")).hexdigest()
        if digest in seen:continue
        seen.add(digest)
        if int.from_bytes(bytes.fromhex(digest)[:8],"big")%20!=R5_BUCKET:continue
        eligible+=1
        text=item.get("text","")
        if not text or not text.strip():too_short+=1;continue
        tokens=tok.encode(text,add_special_tokens=False)
        if len(tokens)<516:too_short+=1;continue
        windows=[torch.tensor(tokens[k*129:(k+1)*129],dtype=torch.long) for k in range(4)]
        assert all(len(w)==129 for w in windows)
        docs.append({"sha256":digest,"source_row":rownum,"windows":windows})
        if len(docs)==32:break
    assert len(docs)==32,("R6_imdb_doc_shortfall",scanned,len(docs))
    assert len(set(x["sha256"] for x in docs))==32
    audit={"dataset":"stanfordnlp/imdb","dataset_revision":R5_FINEWEB_REV,
       "minimum_row":R5_SOURCE_MIN_ROW,"hash_bucket":7,
       "selected_docs":len(docs),"scanned_rows":scanned,
       "qualifying_bucket_docs":eligible,"short_docs":too_short,
       "target_tokens":16384,
       "source_rows":[x["source_row"] for x in docs],
       "document_sha256":[x["sha256"] for x in docs]}
    print(json.dumps({"event":"r5_fresh_doc_selector","model":MODEL_ID,"audit":audit}),flush=True)
    return docs,audit

R5_FRESH_DOCS,R5_FRESH_AUDIT=r5_select_new_documents()

@torch.no_grad()
def r5_score_new_fineweb(m):
    m.eval();per=[];total=0.0;n=0
    for item in R5_FRESH_DOCS:
        sm=0.0;count=0
        for ids in item["windows"]:
            x=ids[:-1].unsqueeze(0).to(DEVICE)
            y=ids[1:].unsqueeze(0).to(DEVICE)
            with torch.autocast("cuda",dtype=torch.float16,enabled=(DEVICE=="cuda")):
                z=m(x).logits.float()
            ce=F.cross_entropy(z.reshape(-1,z.shape[-1]),y.reshape(-1),reduction="sum")
            sm+=float(ce.item());count+=int(y.numel())
        assert count==512 and math.isfinite(sm)
        per.append({"doc_sha256":item["sha256"],"row":item["source_row"],
                    "tokens":count,"nll_sum":sm,"nll":sm/count})
        total+=sm;n+=count
    assert len(per)==32 and n==16384
    return {"nll":total/n,"ppl":math.exp(min(total/n,20)),
            "tokens":n,"document_count":32,"per_document":per}

@torch.no_grad()
def r5_checkpoint_h(qs,arm):
    if arm!="H":return {"attempted":False,"retained":False}
    import numpy as np
    from huggingface_hub import HfApi
    payload={};meta=[];total=0
    for i,(name,mod,q) in enumerate(qs):
        code=q.hard_code(mod.parametrizations.weight.original).cpu().numpy().astype(np.int8)
        assert np.all(np.isin(code,[-1,0,1]))
        flat=np.where(code.reshape(-1)==-1,2,code.reshape(-1)).astype(np.uint8)
        pad=(-flat.size)%4
        if pad:flat=np.pad(flat,(0,pad))
        packed=(flat[0::4] | (flat[1::4]<<2) | (flat[2::4]<<4) | (flat[3::4]<<6)).astype(np.uint8)
        key=f"{i:03d}"
        payload["codes_"+key]=packed
        payload["alpha_"+key]=q.alpha().detach().float().cpu().numpy()
        meta.append({"key":key,"module":name,"shape":list(code.shape),"count":int(code.size),
                     "packing":"2 bits/code: 0->0, +1->1, -1->2; 4 codes/byte, least-significant first",
                     "alpha":"code/1.5 multiplied by alpha, normal Q3 inference"})
        total+=code.size
    manifest={"kind":"r5_restorable_inference_quantized_linears_only",
         "seed":SEED,"family":MODEL_ID,"model_revision":MODEL_REVISION,
         "source_roundtrip":"BF16-rounded source, unchanged nonquantized tensors from pinned base model",
         "note":"No FP32 master weights, optimizer moments or restartable training state.",
         "layers":meta,"quantized_weights":total}
    payload["manifest_utf8"]=np.frombuffer(json.dumps(manifest).encode("utf-8"),dtype=np.uint8)
    outpath=os.path.join(tempfile.gettempdir(),f"r5_smol_seed{SEED}_H_final_compact.npz")
    np.savez_compressed(outpath,**payload)
    with open(outpath,"rb") as f:digest=hashlib.sha256(f.read()).hexdigest()
    size=os.path.getsize(outpath)
    ans={"attempted":True,"retained":False,"local_size":size,"sha256":digest,
         "repo_id":R5_CKPT_REPO,"remote_path":f"smol_seed{SEED}/H_final_compact.npz",
         "layers":len(meta),"quantized_weights":total}
    try:
        token=os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        if not token:
            raise RuntimeError("Missing HF_TOKEN write credential in HF Job secrets; compact snapshot not retained")
        api=HfApi(token=token)
        api.create_repo(repo_id=R5_CKPT_REPO,repo_type="dataset",private=True,exist_ok=True)
        info=api.repo_info(repo_id=R5_CKPT_REPO,repo_type="dataset")
        if not getattr(info,"private",False):raise RuntimeError("Refusing to upload into nonprivate repo")
        api.upload_file(path_or_fileobj=outpath,path_in_repo=ans["remote_path"],
            repo_id=R5_CKPT_REPO,repo_type="dataset",
            commit_message=f"R5 Smol seed{SEED} compact H inference snapshot")
        ans["retained"]=True
        print(json.dumps({"event":"r5_snapshot_uploaded","summary":ans}),flush=True)
    except Exception as exc:
        ans["error"]=f"{type(exc).__name__}: {str(exc)[:600]}"
        print(json.dumps({"event":"r5_snapshot_upload_failed","summary":ans}),flush=True)
    finally:
        try:os.remove(outpath)
        except OSError:pass
    return ans

R4_PROTOCOL=R5_PROTOCOL
ARM_ORDER=("D","W","N","H")
CODEBOOKS={"W":[0.0,1/4,2/4,3/4,1.0],
           "N":[0.0,1/6,2/6,3/6,4/6],
           "H":[0.0,1/6,2/6,3/6,1.0]}
R4_HIST={}

def reset_r4_rng():
    torch.manual_seed(SEED)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(SEED)

@torch.no_grad()
def prep_grid_diag(qs):
    total=0;outer=0;states=set()
    for _,mod,q in qs:
        assert q.levels==9
        c=q.hard_code(mod.parametrizations.weight.original)
        total+=int(c.numel())
        outer+=int((c.abs()==4).sum().item())
        states.update(int(x.item()) for x in torch.unique(c))
    return {"weights":total,"extreme_fraction":outer/total,
            "integer_states":sorted(states),
            "mode":qs[0][2].q9_mode,
            "max_abs_output_normalized":max(CODEBOOKS[qs[0][2].q9_mode])}

@torch.no_grad()
def eval_wikitext_per_chunk(m):
    m.eval()
    sums=[];counts=[]
    for ids in validation_chunks:
        x=ids[:-1].unsqueeze(0).to(DEVICE)
        y=ids[1:].unsqueeze(0).to(DEVICE)
        with torch.autocast("cuda",dtype=torch.float16,enabled=(DEVICE=="cuda")):
            z=m(x).logits.float()
        loss=F.cross_entropy(z.reshape(-1,z.shape[-1]),y.reshape(-1),reduction="sum")
        sums.append(float(loss.item()));counts.append(int(y.numel()))
    nll=sum(sums)/sum(counts)
    return {"nll":nll,"ppl":math.exp(min(nll,20)),"tokens":sum(counts),
        "chunk_count":len(sums),"per_chunk_nll_sum":sums,
        "per_chunk_token_count":counts}

def train_r4_arm(name,teacher,original_scales):
    assert name in ARM_ORDER
    reset_r4_rng()
    t0=time.time()
    m=load_model(torch.float32)
    bf16_roundtrip_(m)
    staged=(name!="D")
    qs,trainable=attach_quantizers(m,9 if staged else 3)
    if staged:
        for _,_,q in qs:q.q9_mode=name
    opt=make_optimizer(trainable,matched_lr(0))
    scaler=torch.amp.GradScaler("cuda",enabled=(DEVICE=="cuda"))
    trace=[];validation_dev={};prep=None;switch=[];num_update=0;num_skip=0
    nquant=sum(int(mod.parametrizations.weight.original.numel()) for _,mod,_ in qs)
    assert nquant==314572800

    for i in range(1200):
        if staged and i==PREP_STEPS:
            native=score_on(m,teacher,lrval_chunks)
            prep={"native_train_dev":native,"code_hist":code_hist(qs),
                  "grid":prep_grid_diag(qs),"scale_stats":scale_stats(qs)}
            masters=capture_quantized_masters(qs)
            del opt,scaler,qs,trainable,m
            gc.collect()
            if torch.cuda.is_available():torch.cuda.empty_cache()
            reset_r4_rng()
            m=load_model(torch.float32)
            bf16_roundtrip_(m)
            qs,trainable=attach_quantizers(m,3)
            load_masters_and_scales(qs,masters,original_scales)
            del masters
            # Historical v10: learned Q9 scale is discarded, FP32 masters preserved,
            # AdamW and GradScaler reset; global LR is NOT reset at 301.
            proj=score_on(m,teacher,lrval_chunks)
            switch.append({"at_global_step":300,
                "native_dev":native,"original_scale_projected_Q3_dev":proj,
                "projection_shock_nats":proj["loss"]-native["loss"]})
            opt=make_optimizer(trainable,matched_lr(i))
            scaler=torch.amp.GradScaler("cuda",enabled=(DEVICE=="cuda"))
            print(json.dumps({"event":"r4_smol_switch","arm":name,
                "shock":switch[-1]["projection_shock_nats"]}),flush=True)
        lr=matched_lr(i);set_optimizer_lr(opt,lr)
        before=scaler.get_scale()
        loss,ce,kl=one_step(m,teacher,opt,scaler,train_chunks[i])
        skip=int(scaler.get_scale()<before)
        num_skip+=skip;num_update+=1-skip
        step=i+1
        if not math.isfinite(loss):raise RuntimeError(("nonfinite_train",name,step))
        if step==1 or step%100==0:
            row={"step":step,"lr":lr,"loss":loss,"ce":ce,"kl":kl,
                 "effective":num_update,"amp_skips":num_skip,"levels":qs[0][2].levels}
            trace.append(row)
            print(json.dumps({"event":"r4_smol_train","arm":name,**row}),flush=True)
        if step in (300,600,900,1200):
            validation_dev[str(step)]=score_on(m,teacher,lrval_chunks)
    assert num_update+num_skip==1200 and all(q.levels==3 for _,_,q in qs)
    fresh=eval_wikitext_per_chunk(m)
    old=score_on(m,teacher,eval_chunks)
    independent=r5_score_new_fineweb(m)
    snapshot=r5_checkpoint_h(qs,name)
    result={"arm":name,"scheduled":1200,"updates":num_update,"skips":num_skip,
       "prep":prep,"switches":switch,"train_dev":validation_dev,"trace":trace,
       "fresh_wikitext_validation":fresh,"original_wikitext_test":old,
       "fresh_fineweb":independent,"snapshot":snapshot,
       "final_q3_code_hist":code_hist(qs),"final_scale":scale_stats(qs),
       "quantized_master_weights":nquant,"elapsed_sec":time.time()-t0}
    print(json.dumps({"event":"r4_smol_arm_final","arm":name,
      "old_test":old["loss"],"val_nll":fresh["nll"],"fineweb_nll":independent["nll"],
      "updates":num_update,"skips":num_skip}),flush=True)
    del opt,scaler,m,qs,trainable;gc.collect()
    if torch.cuda.is_available():torch.cuda.empty_cache()
    return result

print(json.dumps({"event":"r4_smol_start","seed":SEED,"model":MODEL_ID,
"model_revision":MODEL_REVISION,"wikitext_revision":WIKITEXT_REVISION,
"global_LR":{"peak":PEAK_LR,"floor":MIN_LR,"warmup":WARMUP_STEPS,
            "horizon":FULL_STEPS},
"schedule":{"D":"Q3 1200 continuous Adam",
            "WNH":"Q9 300 then Q3 900, original scales+new Adam/GradScaler"},
"architecture_precision":"BF16-rounded Smol source, FP16 student/teacher compute, FP32 masters",
"arms":ARM_ORDER,"protocol":R4_PROTOCOL,"order_head":_order[:16],
"gpu":torch.cuda.get_device_name(0) if DEVICE=="cuda" else None}),flush=True)
teacher=load_model(torch.float16 if DEVICE=="cuda" else torch.float32)
teacher.requires_grad_(False);teacher.eval()
original_scales,initial_codes,initial_ref=initial_q3_reference(teacher)
del initial_codes
arms={}
for name in ARM_ORDER:
    arms[name]=train_r4_arm(name,teacher,original_scales)
old={k:arms[k]["original_wikitext_test"]["loss"] for k in ARM_ORDER}
fresh={k:arms[k]["fresh_wikitext_validation"]["nll"] for k in ARM_ORDER}
primary={}
for x,y in (("N","H"),("H","W"),("D","H"),("D","W"),("D","N")):
    aa=arms[x]["fresh_wikitext_validation"]["per_chunk_nll_sum"]
    bb=arms[y]["fresh_wikitext_validation"]["per_chunk_nll_sum"]
    assert len(aa)==len(bb)==64
    delta=sorted((u-v)/SEQ for u,v in zip(aa,bb))
    primary[x+"_minus_"+y]={"aggregate_gap":fresh[x]-fresh[y],
      "positive_chunks":sum(v>0 for v in delta),
      "median_chunk_gap":(delta[31]+delta[32])/2,
      "min_chunk_gap":delta[0],"max_chunk_gap":delta[-1]}

new={k:arms[k]["fresh_fineweb"]["nll"] for k in ARM_ORDER}
paired={}
for left,right in (("N","H"),("H","W"),("D","H"),("D","W"),("D","N")):
    left_docs=arms[left]["fresh_fineweb"]["per_document"]
    right_docs=arms[right]["fresh_fineweb"]["per_document"]
    assert [x["doc_sha256"] for x in left_docs]==[x["doc_sha256"] for x in right_docs]
    delta=sorted(a["nll"]-b["nll"] for a,b in zip(left_docs,right_docs))
    paired[left+"_minus_"+right]={
      "aggregate_gap":new[left]-new[right],
      "positive_documents":sum(x>0 for x in delta),
      "median_doc_gap":(delta[15]+delta[16])/2,
      "minimum_doc_gap":delta[0],"maximum_doc_gap":delta[-1]}
checks={
 "four_arms":set(arms)==set(ARM_ORDER),
 "all_1200_opportunities":all(a["scheduled"]==1200 and a["updates"]+a["skips"]==1200 for a in arms.values()),
 "D_continuous":len(arms["D"]["switches"])==0,
 "all_staged_switch300":all(len(arms[k]["switches"])==1 and
    arms[k]["switches"][0]["at_global_step"]==300 for k in ("W","N","H")),
 "fresh32doc_16384tokens":R5_FRESH_AUDIT["selected_docs"]==32
    and len(set(R5_FRESH_AUDIT["document_sha256"]))==32
    and all(x>=1 for x in R5_FRESH_AUDIT["source_rows"])
    and all(a["fresh_fineweb"]["document_count"]==32
       and a["fresh_fineweb"]["tokens"]==16384 for a in arms.values()),
 "fresh_all_arms_paired_docs":all([d["doc_sha256"] for d in arms[k]["fresh_fineweb"]["per_document"]]==R5_FRESH_AUDIT["document_sha256"] for k in ARM_ORDER),
 "new_seed_190027":SEED==130363,
 "model_pin":MODEL_REVISION=="a10cc1512eabd3dde888204e902eca88bddb4951",
 "wikitext_pin":WIKITEXT_REVISION=="b08601e04326c79dfdd32d625aee71d232d685c3",
 "fineweb_pin":R5_FINEWEB_REV=="e628166",
 "all_q3_quantized_master_weights":all(a["quantized_master_weights"]==314572800 and
      set(a["final_q3_code_hist"])=={"-1","0","1"} and
       abs(sum(a["final_q3_code_hist"].values())-1.0)<1e-6 for a in arms.values()),
 "all_Q9_9_states_at300":all(arms[k]["prep"]["grid"]["integer_states"]==list(range(-4,5)) for k in ("W","N","H")),
 "all_expected_Q9_amplitudes":all(abs(arms[k]["prep"]["grid"]["max_abs_output_normalized"]-
    (2/3 if k=="N" else 1.0))<1e-6 for k in ("W","N","H")),
 "validation8192":all(a["fresh_wikitext_validation"]["tokens"]==8192 for a in arms.values()),
 "all_finite":all(math.isfinite(v) for v in (*new.values(),*old.values(),*fresh.values())),
 "lr_schedule":matched_lr(99)==1e-3 and abs(matched_lr(1199)-1e-4)<1e-12,
 "snapshot_H_attempted":arms["H"]["snapshot"]["attempted"]}
import importlib.metadata as md
versions={}
for pkg in ("torch","transformers","datasets","accelerate","huggingface_hub","numpy"):
    try:versions[pkg]=md.version(pkg)
    except Exception:versions[pkg]="unknown"
out={"kind":"r6_smol_imdb_hybrid","seed":SEED,
 "protocol":R5_PROTOCOL,"model_revision":MODEL_REVISION,
 "fineweb_revision":R5_FINEWEB_REV,"wikitext_revision":WIKITEXT_REVISION,
 "fineweb_new_doc_audit":R5_FRESH_AUDIT,"train_order_head":_order[:16],
 "dependency_versions":versions,"arms":arms,"fresh_fineweb_nll":new,
 "primary_paired_documents":paired,"secondary_wikitext_validation_nll":fresh,
 "legacy_wikitext_test_nll":old,"checks":checks,
 "valid_for_science":all(checks.values()),
 "checkpoint_status":arms["H"]["snapshot"],
 "caveat":"One new deterministic seed, new-QAT FineWeb docs may have appeared in pretrained model data; inference-only snapshot if uploaded, not training restart."}
print("FINAL_JSON_BEGIN",flush=True)
print(json.dumps(out),flush=True)
print("FINAL_JSON_END",flush=True)
if not out["valid_for_science"]:raise RuntimeError(("R5_INVALID",checks))
