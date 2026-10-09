# /// script
# dependencies = ["torch>=2.4","transformers>=4.46","datasets>=3.0","accelerate>=1.0"]
# ///

# Phase C / C1: Q9-to-Q3 switching after 250 or 300 globally matched steps.
# One exploratory seed, preregistered before this GPU job.
# Based on v11 exact optimizer/teacher/quantizer/loader semantics.
# Original v11 docstring below is preserved as helper provenance, not this experiment's design.
"""
SmolLM2 v11: matched-schedule four-arm hybrid-master factorial.

Purpose:
Causally localize the schedule-matched Q9 -> Q3 trainability advantage.

Build two equal-compute step-300 FP32 master states under the v10 global LR
curve:
  D = 300 direct-Q3 updates
  S = 300 Q9 updates

Both are projected through the same ORIGINAL Q3 scales. Define mask M at exact
weight positions where projected Q3 codes differ between D and S.

Construct four master states without changing any other parameter:
  00: D on M, D on ~M = D everywhere
  10: S on M, D on ~M
  01: D on M, S on ~M
  11: S on M, S on ~M = S everywhere

By construction:
  Q3(00) == Q3(01)
  Q3(10) == Q3(11)

All arms then receive original Q3 scales, fresh Adam, and the same global LR
curve from step 301 through 1200.

This is a causal partition of the actual D-vs-S prepared-master difference into
code-disagreement positions M versus same-code continuous geometry ~M.
"""
import gc, json, math, time
from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import parametrize
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

MODEL_ID="HuggingFaceTB/SmolLM2-360M-Instruct"
DEVICE="cuda" if torch.cuda.is_available() else "cpu"
SEED=1729
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

tok=AutoTokenizer.from_pretrained(MODEL_ID)
if tok.pad_token_id is None:
    tok.pad_token=tok.eos_token

def stream_chunks(split,n):
    ds=load_dataset("Salesforce/wikitext","wikitext-2-raw-v1",split=split)
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

def load_model(dtype=torch.float32):
    m=AutoModelForCausalLM.from_pretrained(
        MODEL_ID,dtype=dtype,low_cpu_mem_usage=True
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
        hard=torch.round(z*n)/n
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


def reset_rng():
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

@torch.no_grad()
def projected_q3_codes_from_masters(masters, reference_scales):
    out={}
    for name,w in masters.items():
        a=reference_scales[name]["alpha"].float()
        z=(w.float()/a).clamp(-0.99,0.99)
        out[name]=torch.round(z*1.5).to(torch.int8)
    return out

@torch.no_grad()
def build_diff_mask(d_codes,s_codes):
    mask={}
    changed=0; total=0; layer_rows=[]
    for name in d_codes:
        m=(d_codes[name]!=s_codes[name])
        n=m.numel(); ch=int(m.sum().item())
        mask[name]=m
        changed+=ch; total+=n
        layer_rows.append((ch/max(1,n),name,ch,n))
    layer_rows.sort(reverse=True)
    return mask,{
        "fraction":changed/max(1,total),
        "changed_count":changed,
        "total":total,
        "top_layers":[
            {"name":name,"fraction":fr,"changed_count":ch,"total":n}
            for fr,name,ch,n in layer_rows[:10]
        ]
    }

def prepare_matched(label,levels,teacher):
    reset_rng()
    m=load_model(torch.float32); bf16_roundtrip_(m)
    qs,trainable=attach_quantizers(m,levels)
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
                "global_step":global_step0+1,
                "levels":levels,"lr":lr,
                "loss":loss,"ce":ce,"kl":kl,**fs
            }
            trace.append(row)
            print(json.dumps({"event":"v11_prep","label":label,**row}),flush=True)

    masters=capture_quantized_masters(qs)
    out={
        "label":label,
        "levels":levels,
        "native_diag":score_on(m,teacher,lrval_chunks),
        "learned_scale":scale_stats(qs),
        "native_hist":code_hist(qs),
        "trace":trace,
        "seconds":time.time()-start_time
    }
    del opt,scaler,m,qs,trainable,prev,stage_start
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return masters,out

@torch.no_grad()
def load_hybrid_masters(qs,d_masters,s_masters,diff_mask,arm):
    if arm not in ("00","10","01","11"):
        raise ValueError(arm)
    use_s_on_m=(arm[0]=="1")
    use_s_off_m=(arm[1]=="1")
    for name,mod,q in qs:
        d=d_masters[name]
        s=s_masters[name]
        mask=diff_mask[name]
        if use_s_on_m and use_s_off_m:
            h=s
        elif (not use_s_on_m) and (not use_s_off_m):
            h=d
        elif use_s_on_m:
            h=torch.where(mask,s,d)
        else:
            h=torch.where(mask,d,s)
        w=mod.parametrizations.weight.original
        w.copy_(h.to(device=w.device,dtype=w.dtype))

@torch.no_grad()
def set_reference_scales(qs,reference_scales):
    for name,mod,q in qs:
        raw=reference_scales[name]["raw_alpha"]
        q.raw_alpha.copy_(raw.to(device=q.raw_alpha.device,dtype=q.raw_alpha.dtype))
        q.levels=3

def instantiate_hybrid(arm,d_masters,s_masters,diff_mask,reference_scales,teacher):
    m=load_model(torch.float32); bf16_roundtrip_(m)
    qs,trainable=attach_quantizers(m,3)
    load_hybrid_masters(qs,d_masters,s_masters,diff_mask,arm)
    set_reference_scales(qs,reference_scales)
    codes=snapshot_codes(qs)
    diag=score_on(m,teacher,lrval_chunks)
    return m,qs,trainable,codes,diag

def continue_hybrid(arm,d_masters,s_masters,diff_mask,reference_scales,teacher):
    reset_rng()
    m,qs,trainable,start_codes,pre_diag=instantiate_hybrid(
        arm,d_masters,s_masters,diff_mask,reference_scales,teacher
    )
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
                "lr":lr,"loss":loss,"ce":ce,"kl":kl,**fs
            }
            trace.append(row)
            print(json.dumps({"event":"v11_continue","arm":arm,**row}),flush=True)

    final=score_on(m,teacher,eval_chunks)
    final["gen"]=generate(m)
    final["final_hist"]=code_hist(qs)
    final["scale"]=scale_stats(qs)
    final["trace"]=trace
    final["seconds"]=time.time()-start_time
    final_codes=snapshot_codes(qs)

    out={
        "pre_diag":pre_diag,
        "start_hist":code_hist(qs) if False else None,
        "final":final,
        "start_codes_vs_final":hamming_codes(start_codes,final_codes)
    }

    del opt,scaler,m,qs,trainable,prev,stage_start,start_codes,final_codes
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return out

def diag_close(a,b,tol=1e-5):
    keys=["loss","ppl","top1_agreement","kl_to_teacher"]
    return all(abs(float(a[k])-float(b[k]))<=tol for k in keys)


C1_ARMS=(("D",0),("S250",250),("S300",300))
GLOBAL_VALIDATION_STEPS=(250,300,600,900,1200)
HISTORIC_TEST_D=5.595722187310457
HISTORIC_TEST_S300=4.9009853675961494
REPRO_TOL=0.06
PRESERVE_LOSS_TOL=0.07
PRESERVE_GAIN_FRAC=0.90

@torch.no_grad()
def capture_original_scaled_q3_codes(qs,ref_scales):
    codes={}
    for name,mod,q in qs:
        w=mod.parametrizations.weight.original
        a=ref_scales[name]["alpha"].to(device=w.device,dtype=w.dtype)
        codes[name]=torch.round((w/a).clamp(-0.99,0.99)*1.5).to(torch.int8).detach().cpu()
    return codes

@torch.no_grad()
def switch_to_reference_q3(qs,ref_scales):
    for name,mod,q in qs:
        raw=ref_scales[name]["raw_alpha"].to(q.raw_alpha.device,dtype=q.raw_alpha.dtype)
        q.raw_alpha.copy_(raw)
        q.levels=3

def run_arm(label,q9_steps,teacher,ref_scales,initial_codes):
    reset_rng()
    model=load_model(torch.float32)
    bf16_roundtrip_(model)
    quantizers,trainable=attach_quantizers(model,9 if q9_steps>0 else 3)
    before_codes=capture_original_scaled_q3_codes(quantizers,ref_scales)
    initial_ham=hamming_codes(initial_codes,before_codes)["fraction"]
    assert initial_ham==0.0,("source Q3 code mismatch",label,initial_ham)
    del before_codes
    optim=make_optimizer(trainable,matched_lr(0))
    amp=torch.amp.GradScaler("cuda",enabled=(DEVICE=="cuda"))
    trace=[]
    dev={}
    switches=[]
    skipped=0
    successes=0
    t0=time.time()
    prev_codes=snapshot_codes(quantizers)
    stage_codes=prev_codes

    print(json.dumps({"event":"c1_arm_start","arm":label,
        "q9_prep_steps":q9_steps,"q3_continuation_steps":1200-q9_steps,
        "order_head":_order[:16],"init_code_hamming":initial_ham}),flush=True)

    for global_step0 in range(FULL_STEPS):
        step=global_step0+1
        if q9_steps>0 and global_step0==q9_steps:
            # Perform switch immediately before the NEXT optimizer update.
            # Q9 prep steps 1..q9_steps are complete; original Q3 scale restored.
            pre_native=score_on(model,teacher,lrval_chunks)
            before_switch_codes=capture_original_scaled_q3_codes(quantizers,ref_scales)
            code_movement=hamming_codes(initial_codes,before_switch_codes)
            del before_switch_codes
            switch_to_reference_q3(quantizers,ref_scales)
            pre_projection=score_on(model,teacher,lrval_chunks)
            if not(math.isfinite(pre_native["loss"]) and math.isfinite(pre_projection["loss"])):
                raise RuntimeError(("nonfinite_switch_diag",label,q9_steps))
            switches.append({
                "global_step":q9_steps,
                "native_q9_dev":pre_native,
                "immediate_original_scale_q3_dev":pre_projection,
                "transition_loss_shock":pre_projection["loss"]-pre_native["loss"],
                "codes_changed_vs_initial":code_movement,
                "initial_q3_lr":matched_lr(global_step0)
            })
            print(json.dumps({"event":"c1_switch","arm":label,**switches[-1]}),flush=True)
            # Reset Adam and scaler at the switch exactly as v10.
            del optim,amp,prev_codes,stage_codes
            gc.collect()
            optim=make_optimizer(trainable,matched_lr(global_step0))
            amp=torch.amp.GradScaler("cuda",enabled=(DEVICE=="cuda"))
            prev_codes=snapshot_codes(quantizers)
            stage_codes=prev_codes
            model.train()

        lr=matched_lr(global_step0)
        set_optimizer_lr(optim,lr)
        before_scale=amp.get_scale() if DEVICE=="cuda" else 1.0
        loss,ce,kl=one_step(model,teacher,optim,amp,train_chunks[global_step0])
        after_scale=amp.get_scale() if DEVICE=="cuda" else 1.0
        skipped_now=bool(after_scale<before_scale)
        skipped+=int(skipped_now)
        successes+=int(not skipped_now)

        if step==1 or step%100==0 or step==q9_steps or step==q9_steps+1:
            prev_codes,flip=flip_stats(quantizers,prev_codes,stage_codes)
            row={
                "global_step":step,"arm":label,"levels":quantizers[0][2].levels,
                "lr":lr,"loss":loss,"ce":ce,"kl":kl,
                "amp_scale_before":before_scale,"amp_scale_after":after_scale,
                "skipped_this_step":skipped_now,
                "effective_optimizer_updates_total":successes,
                **flip
            }
            trace.append(row)
            print(json.dumps({"event":"c1_train_trace",**row}),flush=True)

        if step in GLOBAL_VALIDATION_STEPS:
            diag=score_on(model,teacher,lrval_chunks)
            dev[str(step)]={"active_levels":quantizers[0][2].levels,
                           "train_split_validation":diag,
                           "effective_optimizer_updates":successes}
            print(json.dumps({"event":"c1_dev","arm":label,"global_step":step,
                **dev[str(step)]}),flush=True)
            model.train()
        if not math.isfinite(loss):
            raise RuntimeError(("nonfinite_training_loss",label,step))

    assert len(switches)==(1 if q9_steps>0 else 0)
    assert successes+skipped==FULL_STEPS, (label,successes,skipped)
    assert all(q.levels==3 for _,_,q in quantizers)

    final=score_on(model,teacher,eval_chunks)
    final_dev=dev["1200"]["train_split_validation"]
    final_codes=snapshot_codes(quantizers)
    assert all(bool(torch.all((c>=-1)&(c<=1))) for c in final_codes.values())
    final["code_change_vs_initial"]=hamming_codes(initial_codes,final_codes)
    final["hist"]=code_hist(quantizers)
    final["scale"]=scale_stats(quantizers)
    final["train_split_validation"]=final_dev
    final["elapsed_seconds"]=time.time()-t0
    del optim,amp,model,quantizers,trainable,prev_codes,stage_codes,final_codes
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    result={"arm":label,"scheduled_steps":FULL_STEPS,
            "q9_prep_steps":q9_steps,"q3_continuation_steps":FULL_STEPS-q9_steps,
            "effective_optimizer_updates":successes,"amp_skipped_opportunities":skipped,
            "transitions":switches,"dev_checkpoints":dev,"trace":trace,"final":final}
    print(json.dumps({"event":"c1_arm_final","arm":label,
        "test_loss":final["loss"],"test_ppl":final["ppl"],
        "top1":final["top1_agreement"],"test_kl":final["kl_to_teacher"],
        "effective_optimizer_updates":successes,"amp_skips":skipped,
        "train_split_final_loss":final_dev["loss"]}),flush=True)
    return result

print(json.dumps({
    "event":"c1_start",
    "model":MODEL_ID,"seed":SEED,
    "gpu":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "code_lineage":"v11 matched Q9 preparation and v10 reset/reference-scale Q3 continuation",
    "arms":C1_ARMS,"lr":{"warmup":WARMUP_STEPS,"peak":PEAK_LR,"floor":MIN_LR,
                         "full_steps":FULL_STEPS},
    "data_order_head":_order[:16],
    "prereg":"ternary_pet/research_log/phase_c_q9_switch250_vs300_seed1729_prereg_2026-10-08.md",
    "exploratory":True
}),flush=True)

teacher=load_model(torch.float16 if DEVICE=="cuda" else torch.float32)
teacher.requires_grad_(False)
teacher.eval()
reference_scales,initial_codes,initial_reference=initial_q3_reference(teacher)

results={"kind":"c1_smollm2_q9_switch250_vs300_seed1729",
         "model":MODEL_ID,"seed":SEED,"order_head":_order[:16],
         "total_global_step_opportunities":FULL_STEPS,
         "prereg":"ternary_pet/research_log/phase_c_q9_switch250_vs300_seed1729_prereg_2026-10-08.md",
         "initial_reference_train_validation":initial_reference["diag"],
         "arms":{}}
for label,prep in C1_ARMS:
    results["arms"][label]=run_arm(label,prep,teacher,reference_scales,initial_codes)

LD=results["arms"]["D"]["final"]["loss"]
L250=results["arms"]["S250"]["final"]["loss"]
L300=results["arms"]["S300"]["final"]["loss"]
delta=L250-L300
full_gain=LD-L300
short_gain=LD-L250
fraction=short_gain/full_gain if full_gain>0 else None
checks={
    "all_arms_1200_scheduled":all(x["scheduled_steps"]==FULL_STEPS for x in results["arms"].values()),
    "same_order":True,
    "D_loss_versus_historical":LD-HISTORIC_TEST_D,
    "S300_loss_versus_historical":L300-HISTORIC_TEST_S300,
    "D_reproduces_within_0p06":abs(LD-HISTORIC_TEST_D)<=REPRO_TOL,
    "S300_reproduces_within_0p06":abs(L300-HISTORIC_TEST_S300)<=REPRO_TOL,
}
results["comparison"]={
    "D_loss":LD,"S250_loss":L250,"S300_loss":L300,
    "S250_minus_S300_loss":delta,
    "S300_gain_vs_D":full_gain,
    "S250_gain_vs_D":short_gain,
    "fraction_of_S300_gain_retained":fraction,
    "pass_loss_margin_0p07":delta<=PRESERVE_LOSS_TOL,
    "pass_gain_recovery_0p90":fraction is not None and fraction>=PRESERVE_GAIN_FRAC,
    "preserves_benefit_exploratory":delta<=PRESERVE_LOSS_TOL and fraction is not None and fraction>=PRESERVE_GAIN_FRAC
}
results["checks"]=checks
results["valid_for_science"]=all(bool(v) for k,v in checks.items() if isinstance(v,bool))
results["exploratory_only"]=True
print("FINAL_JSON_BEGIN",flush=True)
print(json.dumps(results),flush=True)
print("FINAL_JSON_END",flush=True)
if not results["valid_for_science"]:
    raise RuntimeError(("C1_TECHNICAL_REPRODUCIBILITY_DISCREPANCY",checks))
