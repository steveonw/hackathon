# /// script
# dependencies = ["torch>=2.4","transformers>=4.46","datasets>=3.0","accelerate>=1.0"]
# ///
"""
SmolLM2 v13: closing depth-sweep and firmness controls.

Seed/order 1729 first. Reuses the v12 preparation and continuation setup
exactly through the step-300 direct-Q3 (D) and Q9 (S) states.

M is the projected-Q3 disagreement mask between D and S under the original Q3
scales alpha0.

Depth arms put Q9's selected code on M at normalized depth
d in {0.03,0.25,0.50,0.75,1.00} between the Q3 decision boundary and the
canonical reconstruction prototype. d=0.03 re-anchors v12 minimal crossing;
d=1.00 re-anchors v12 prototype.

B1 snaps DIRECT's own codes to their prototypes on M.
B2 snaps DIRECT's own codes to their prototypes on D's own step-300 changed set
(relative to the initial Q3 projection).

All arms use D masters elsewhere, original Q3 scales, fresh Adam, and the same
global LR continuation for steps 301-1200.

For depth arms, code survival of Q9's selected code on M is measured after
100, 300, and 900 continuation updates under both:
  (a) the arm's current learned Q3 scale, and
  (b) the fixed original alpha0 used to define depth.
Survival is also split by Q9 target code zero vs nonzero.
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



V13_DEPTHS=[0.03,0.25,0.50,0.75,1.00]
V13_DEPTH_ARMS={
    "d003":0.03,
    "d025":0.25,
    "d050":0.50,
    "d075":0.75,
    "d100":1.00,
}
V13_ARMS=["D","d003","d025","d050","d075","d100"]
SURVIVAL_STEPS={100,300,900}

@torch.no_grad()
def set_reference_scales(qs,reference_scales):
    for name,mod,q in qs:
        raw=reference_scales[name]["raw_alpha"]
        q.raw_alpha.copy_(raw.to(device=q.raw_alpha.device,dtype=q.raw_alpha.dtype))
        q.levels=3

@torch.no_grad()
def q3_prototype(alpha,code):
    return (2.0/3.0)*alpha.float()*code.to(torch.float32)

@torch.no_grad()
def depth_value(alpha,source_code,target_code,depth):
    """
    Depth coordinate in u=w/alpha0.
      target +/-1: u=c*(1/3 + d*(2/3-1/3))
      target 0:    u=sign(u_D)*(1/3)*(1-d)
    d=0 is conceptual edge only; no d=0 arm is instantiated because exact
    half-integer rounding is tie-sensitive.
    """
    d=float(depth)
    tc=target_code.to(torch.int8)
    sc=source_code.to(torch.int8)
    z=torch.empty_like(alpha,dtype=torch.float32)

    nz=(tc!=0)
    z[nz]=tc[nz].to(torch.float32)*((1.0/3.0)+d*(1.0/3.0))

    to0=(tc==0)
    bad=to0 & (sc==0)
    if bool(bad.any().item()):
        raise RuntimeError("depth target-zero received source-zero on disagreement mask")
    z[to0]=sc[to0].to(torch.float32)*(1.0/3.0)*(1.0-d)
    return alpha.float()*z

@torch.no_grad()
def mask_overlap_stats(a,b):
    total=0; ca=0; cb=0; both=0; union=0
    for name in a:
        xa=a[name].bool(); yb=b[name].bool()
        n=xa.numel(); total+=n
        ca+=int(xa.sum().item()); cb+=int(yb.sum().item())
        both+=int((xa & yb).sum().item())
        union+=int((xa | yb).sum().item())
    return {
        "a_fraction":ca/max(1,total),
        "b_fraction":cb/max(1,total),
        "intersection_fraction":both/max(1,total),
        "union_fraction":union/max(1,total),
        "jaccard":both/max(1,union),
    }

@torch.no_grad()
def transition_counts(src_codes,tgt_codes,mask):
    out=defaultdict(int)
    for name in src_codes:
        s=src_codes[name].reshape(-1)
        t=tgt_codes[name].reshape(-1)
        m=mask[name].reshape(-1)
        for a in (-1,0,1):
            for b in (-1,0,1):
                if a==b: continue
                n=int((m & (s==a) & (t==b)).sum().item())
                if n: out[f"{a}->{b}"]+=n
    return dict(out)

@torch.no_grad()
def load_v13_arm(qs,arm,d_masters,d_codes,s_codes,diff_mask,d_changed_mask,
                 reference_scales):
    for name,mod,q in qs:
        h=d_masters[name].clone()
        a=reference_scales[name]["alpha"].float().expand_as(h)
        dc=d_codes[name]
        sc=s_codes[name]

        if arm=="D":
            pass
        elif arm in V13_DEPTH_ARMS:
            m=diff_mask[name]
            h[m]=depth_value(a[m],dc[m],sc[m],V13_DEPTH_ARMS[arm])
        elif arm=="B1":
            m=diff_mask[name]
            h[m]=q3_prototype(a[m],dc[m])
        elif arm=="B2":
            m=d_changed_mask[name]
            h[m]=q3_prototype(a[m],dc[m])
        else:
            raise ValueError(arm)

        w=mod.parametrizations.weight.original
        w.copy_(h.to(device=w.device,dtype=w.dtype))

def instantiate_v13(arm,d_masters,d_codes,s_codes,diff_mask,d_changed_mask,
                    reference_scales,teacher):
    m=load_model(torch.float32); bf16_roundtrip_(m)
    qs,trainable=attach_quantizers(m,3)
    load_v13_arm(qs,arm,d_masters,d_codes,s_codes,diff_mask,d_changed_mask,
                 reference_scales)
    set_reference_scales(qs,reference_scales)
    codes=snapshot_codes(qs)
    diag=score_on(m,teacher,lrval_chunks)
    return m,qs,trainable,codes,diag

@torch.no_grad()
def q9_code_survival(qs,diff_mask,s_codes,reference_scales):
    groups={
        "all":{"n":0,"current":0,"fixed_alpha0":0},
        "target_zero":{"n":0,"current":0,"fixed_alpha0":0},
        "target_nonzero":{"n":0,"current":0,"fixed_alpha0":0},
    }
    for name,mod,q in qs:
        w=mod.parametrizations.weight.original
        mask=diff_mask[name].to(device=w.device)
        if not bool(mask.any().item()):
            continue
        target=s_codes[name].to(device=w.device)
        current=q.hard_code(w)
        a0=reference_scales[name]["alpha"].to(device=w.device,dtype=w.dtype)
        z=(w/a0).clamp(-0.99,0.99)
        fixed=torch.round(z*1.5).to(torch.int8)

        for key,sel in [
            ("all",mask),
            ("target_zero",mask & (target==0)),
            ("target_nonzero",mask & (target!=0)),
        ]:
            n=int(sel.sum().item())
            groups[key]["n"]+=n
            if n:
                groups[key]["current"]+=int(((current==target)&sel).sum().item())
                groups[key]["fixed_alpha0"]+=int(((fixed==target)&sel).sum().item())

    out={}
    for key,g in groups.items():
        n=max(1,g["n"])
        out[key]={
            "n":g["n"],
            "current_scale_fraction":g["current"]/n,
            "fixed_alpha0_fraction":g["fixed_alpha0"]/n,
        }
    return out

def continue_v13(arm,d_masters,d_codes,s_codes,diff_mask,d_changed_mask,
                 reference_scales,teacher):
    reset_rng()
    m,qs,trainable,start_codes,pre_diag=instantiate_v13(
        arm,d_masters,d_codes,s_codes,diff_mask,d_changed_mask,
        reference_scales,teacher
    )
    opt=make_optimizer(trainable,matched_lr(PREP_STEPS))
    scaler=torch.amp.GradScaler("cuda",enabled=(DEVICE=="cuda"))
    stage_start=start_codes
    prev={k:v.clone() for k,v in stage_start.items()}
    trace=[]
    survival={}
    start_time=time.time()

    for local in range(CONT_STEPS):
        global_step0=PREP_STEPS+local
        lr=matched_lr(global_step0)
        set_optimizer_lr(opt,lr)
        loss,ce,kl=one_step(m,teacher,opt,scaler,train_chunks[global_step0])
        step=local+1

        if arm in V13_DEPTH_ARMS and step in SURVIVAL_STEPS:
            sv=q9_code_survival(qs,diff_mask,s_codes,reference_scales)
            survival[str(step)]=sv
            print(json.dumps({
                "event":"v13_survival","arm":arm,"stage_step":step,"survival":sv
            }),flush=True)

        if local==0 or step%50==0 or step==CONT_STEPS:
            prev,fs=flip_stats(qs,prev,stage_start)
            row={
                "global_step":global_step0+1,"stage_step":step,
                "lr":lr,"loss":loss,"ce":ce,"kl":kl,**fs
            }
            trace.append(row)
            print(json.dumps({"event":"v13_continue","arm":arm,**row}),flush=True)

    final=score_on(m,teacher,eval_chunks)
    final["gen"]=generate(m)
    final["final_hist"]=code_hist(qs)
    final["scale"]=scale_stats(qs)
    final["trace"]=trace
    final["seconds"]=time.time()-start_time
    final_codes=snapshot_codes(qs)

    out={
        "pre_diag":pre_diag,
        "final":final,
        "start_codes_vs_final":hamming_codes(start_codes,final_codes),
        "q9_code_survival":survival,
    }

    del opt,scaler,m,qs,trainable,prev,stage_start,start_codes,final_codes
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return out

def diag_close(a,b,tol=1e-5):
    keys=["loss","ppl","top1_agreement","kl_to_teacher"]
    return all(abs(float(a[k])-float(b[k]))<=tol for k in keys)


# G1: depth-by-learnable-scale factorial; frozen preregistration 2026-10-09
G1_ARMS=[
    ("d003_learned","d003",False),
    ("d050_learned","d050",False),
    ("d003_frozen","d003",True),
    ("d050_frozen","d050",True),
]
G1_HIST_LOSSES={"d003_learned":5.4948,"d050_learned":4.8850}
G1_PREREG="ternary_pet/research_log/g1_frozen_vs_learned_q3_scale_depth_prereg_2026-10-09.md"

def g1_continue(label,depth_arm,frozen,d_masters,d_codes,s_codes,diff_mask,d_changed_mask,ref,teacher):
    reset_rng()
    m,qs,trainable,start_codes,pre_diag=instantiate_v13(
        depth_arm,d_masters,d_codes,s_codes,diff_mask,d_changed_mask,ref,teacher)
    assert all(q.levels==3 for _,_,q in qs)
    if frozen:
        for name,mod,q in qs:
            q.raw_alpha.requires_grad_(False)
        # Train only FP32 masters; scale parameters are neither trainable nor in Adam.
        params=[mod.parametrizations.weight.original for _,mod,q in qs]
        assert all(p.requires_grad for p in params)
        assert not any(q.raw_alpha.requires_grad for _,_,q in qs)
    else:
        params=trainable
        assert all(q.raw_alpha.requires_grad for _,_,q in qs)
    opt=make_optimizer(params,matched_lr(PREP_STEPS))
    if frozen:
        for name,mod,q in qs:
            assert all(q.raw_alpha is not p for group in opt.param_groups for p in group["params"])
    amp=torch.amp.GradScaler("cuda",enabled=(DEVICE=="cuda"))
    saved_scale={name:q.raw_alpha.detach().cpu().clone() for name,mod,q in qs}
    trace=[]; survival={}; effective=0;skips=0
    start=time.time()
    for local in range(CONT_STEPS):
        global_step0=PREP_STEPS+local
        lr=matched_lr(global_step0)
        set_optimizer_lr(opt,lr)
        old=amp.get_scale()
        loss,ce,kl=one_step(m,teacher,opt,amp,train_chunks[global_step0])
        skip=amp.get_scale()<old
        skips+=int(skip); effective+=int(not skip)
        step=local+1
        if not math.isfinite(loss):
            raise RuntimeError(("G1_nonfinite",label,step))
        if step in SURVIVAL_STEPS:
            survival[str(step)]=q9_code_survival(qs,diff_mask,s_codes,ref)
            print(json.dumps({"event":"g1_survival","arm":label,"q3_step":step,
                              "data":survival[str(step)]}),flush=True)
        if step==1 or step%100==0:
            row={"q3_step":step,"loss":loss,"ce":ce,"kl":kl,"lr":lr,
                 "effective_updates":effective,"amp_skips":skips}
            trace.append(row)
            print(json.dumps({"event":"g1_train","arm":label,**row}),flush=True)
    assert effective+skips==CONT_STEPS
    final=score_on(m,teacher,eval_chunks)
    final_codes=snapshot_codes(qs)
    assert all(q.levels==3 for _,_,q in qs)
    scale_max_delta=0.0
    for name,mod,q in qs:
        delta=float((q.raw_alpha.detach().cpu()-saved_scale[name]).abs().max())
        scale_max_delta=max(scale_max_delta,delta)
        if frozen:
            assert delta==0,("G1_frozen_scale_changed",label,name,delta)
            assert q.raw_alpha.grad is None,("G1_frozen_scale_grad",label,name)
    fin={"arm":label,"depth":V13_DEPTH_ARMS[depth_arm],"scales_frozen":frozen,
         "starting_train_dev":pre_diag,"final":final,"trace":trace,
         "survival":survival,"scale_raw_alpha_max_abs_delta":scale_max_delta,
         "scheduled_q3":CONT_STEPS,"q3_effective_updates":effective,
         "q3_amp_skips":skips,
         "start_to_end_hamming":hamming_codes(start_codes,final_codes),
         "final_scale_stats":scale_stats(qs),
         "seconds":time.time()-start}
    print(json.dumps({"event":"g1_arm_final","arm":label,"loss":final["loss"],
                      "frozen":frozen,"scale_delta":scale_max_delta,
                      "updates":effective,"skips":skips}),flush=True)
    del m,qs,opt,amp,params,trainable,final_codes,start_codes
    gc.collect()
    if torch.cuda.is_available():torch.cuda.empty_cache()
    return fin

print(json.dumps({"event":"g1_start","model":MODEL_ID,"seed":SEED,
    "gpu":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "prereg":G1_PREREG,"design":G1_ARMS,"order_head":_order[:16]}),flush=True)
teacher=load_model(torch.float16 if DEVICE=="cuda" else torch.float32)
teacher.requires_grad_(False);teacher.eval()
ref,initial_codes,initial_reference=initial_q3_reference(teacher)
d_masters,d_prep=prepare_matched("D_direct_q3_300",3,teacher)
s_masters,s_prep=prepare_matched("S_q9_300",9,teacher)
d_codes=projected_q3_codes_from_masters(d_masters,ref)
s_codes=projected_q3_codes_from_masters(s_masters,ref)
diff_mask,mask_stats=build_diff_mask(d_codes,s_codes)
d_changed_mask,d_changed_stats=build_diff_mask(initial_codes,d_codes)
assert mask_stats["changed_count"]>0
start_checks={}
start_details={}
anchor_code=None;anchor_diag=None
for label,depth_arm,frozen in G1_ARMS:
    m,qs,params,codes,diag=instantiate_v13(
        depth_arm,d_masters,d_codes,s_codes,diff_mask,d_changed_mask,ref,teacher)
    if anchor_code is None:
        anchor_code=codes;anchor_diag=diag
    ham=hamming_codes(codes,anchor_code)
    ham_target=hamming_codes(codes,s_codes)
    start_checks[label]={"vs_anchor_hamming":ham["fraction"],
                         "vs_target_q9_hamming":ham_target["fraction"],
                         "same_start_dev":diag_close(diag,anchor_diag)}
    start_details[label]=diag
    assert ham["fraction"]==0 and ham_target["fraction"]==0 and diag_close(diag,anchor_diag)
    del m,qs,params,codes
    gc.collect()
    if torch.cuda.is_available():torch.cuda.empty_cache()
del anchor_code
arms={}
for label,depth_arm,frozen in G1_ARMS:
    arms[label]=g1_continue(label,depth_arm,frozen,d_masters,d_codes,s_codes,diff_mask,d_changed_mask,ref,teacher)
L={key:arms[key]["final"]["loss"] for key in arms}
gain_learned=L["d003_learned"]-L["d050_learned"]
gain_frozen=L["d003_frozen"]-L["d050_frozen"]
checks={
    "all_start_code_and_dev_equal":all(z["vs_anchor_hamming"]==0.0 and z["vs_target_q9_hamming"]==0.0 and z["same_start_dev"] for z in start_checks.values()),
    "all_four_900_q3_steps":all(z["scheduled_q3"]==CONT_STEPS and z["q3_effective_updates"]+z["q3_amp_skips"]==CONT_STEPS for z in arms.values()),
    "both_frozen_scale_unchanged":all(arms[k]["scale_raw_alpha_max_abs_delta"]==0.0 for k in ("d003_frozen","d050_frozen")),
    "all_finite":all(math.isfinite(z) for z in L.values()),
    "learned_d003_reproduced":abs(L["d003_learned"]-G1_HIST_LOSSES["d003_learned"])<=0.06,
    "learned_d050_reproduced":abs(L["d050_learned"]-G1_HIST_LOSSES["d050_learned"])<=0.06
}
result={"kind":"g1_depth_vs_scale_freeze_seed1729","seed":SEED,"prereg":G1_PREREG,
        "original_v13_depth_sweep":"ternary_pet/replications/smollm2_v13A_depth_sweep_seed1729.py",
        "geometry":{"Q9_vs_D_mask":mask_stats,"D_vs_original_changed_mask":d_changed_stats},
        "start_checks":start_checks,"start_train_dev":start_details,
        "arms":arms,
        "effects":{"learned_scale_depth_gain_shallow_minus_deep":gain_learned,
                   "frozen_scale_depth_gain_shallow_minus_deep":gain_frozen,
                   "primary_frozen_minus_learned_depth_gain":gain_frozen-gain_learned},
        "checks":checks,"valid_for_science":all(checks.values()),
        "exploratory_same_seed_and_historical_test":True}
print("FINAL_JSON_BEGIN",flush=True)
print(json.dumps(result),flush=True)
print("FINAL_JSON_END",flush=True)
if not result["valid_for_science"]:
    raise RuntimeError(("G1_TECHNICAL_CHECK_FAILED",checks))
