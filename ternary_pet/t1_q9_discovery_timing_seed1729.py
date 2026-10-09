# /// script
# dependencies = ["torch>=2.4","transformers>=4.46","datasets>=3.0","accelerate>=1.0"]
# ///

# Phase A / T1 experimental pilot: Q9 assignment discovery timing.
# Preregistered at ternary_pet/research_log/phase_a_q9_timing_seed1729_prereg_2026-10-08.md
# Only diagnostic checkpoints; no heldout test, no Q3 continuation, no Gaussian, no gridward.
# Parent code: ternary_pet/smollm2_v11_hybrid_factorial.py, reused unchanged for training path.
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
eval_chunks=[]  # Phase A: heldout test deliberately untouched

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


import numpy as np

PHASE_A_STEPS=(0,50,100,150,200,250,300)
HISTORICAL_D_NATIVE_300=5.9880944689114886
HISTORICAL_S_NATIVE_300=5.182718833287557
HISTORICAL_D_PROJECTED_300=5.9880944689114886
HISTORICAL_S_PROJECTED_300=6.542291978995006
HISTORICAL_MASK_300=0.06458076477050781

def pack_array_codes(codes):
    """Pack ternary -1/0/+1 as two-bit values 0/1/2, four codes per byte."""
    c=np.asarray(codes,dtype=np.int8).reshape(-1)
    if not bool(np.all((c>=-1)&(c<=1))):
        raise RuntimeError("non-ternary Q3 code encountered")
    v=(c+1).astype(np.uint8,copy=False)
    n=int(v.size)
    if n%4:
        v=np.pad(v,(0,4-(n%4)),constant_values=1)
    packed=(v[0::4] | (v[1::4]<<2) | (v[2::4]<<4) | (v[3::4]<<6)).copy()
    if n and not np.array_equal(unpack_array_codes((packed,n)),c):
        raise RuntimeError("two-bit code roundtrip failed")
    return (packed,n)

def unpack_array_codes(record):
    b,n=record
    out=np.empty(len(b)*4,dtype=np.uint8)
    out[0::4]=b&3
    out[1::4]=(b>>2)&3
    out[2::4]=(b>>4)&3
    out[3::4]=(b>>6)&3
    return out[:n].astype(np.int8)-1

@torch.no_grad()
def snapshot_fixed_q3(qs,reference_scales):
    packed={}
    for name,mod,q in qs:
        w=mod.parametrizations.weight.original.detach()
        a=reference_scales[name]["alpha"].to(device=w.device,dtype=w.dtype)
        codes=torch.round((w/a).clamp(-0.99,0.99)*1.5).to(torch.int8)
        packed[name]=pack_array_codes(codes.cpu().numpy())
    return packed

def pack_source_q3(reference_codes):
    return {name:pack_array_codes(c.cpu().numpy()) for name,c in reference_codes.items()}

@torch.no_grad()
def score_fixed_original_q3(m,qs,reference_scales,teacher):
    originals={}
    try:
        for name,mod,q in qs:
            originals[name]=(q.raw_alpha.detach().clone(),q.levels)
            q.raw_alpha.copy_(reference_scales[name]["raw_alpha"].to(q.raw_alpha.device))
            q.levels=3
        return score_on(m,teacher,lrval_chunks)
    finally:
        for name,mod,q in qs:
            raw,levels=originals[name]
            q.raw_alpha.copy_(raw)
            q.levels=levels

def run_preparation_with_snapshots(label,levels,teacher,reference_scales):
    reset_rng()
    m=load_model(torch.float32)
    bf16_roundtrip_(m)
    qs,trainable=attach_quantizers(m,levels)
    opt=make_optimizer(trainable,matched_lr(0))
    scaler=torch.amp.GradScaler("cuda",enabled=(DEVICE=="cuda"))
    snapshots={0:snapshot_fixed_q3(qs,reference_scales)}
    native_stage_start=snapshot_codes(qs)
    prev={name:c.clone() for name,c in native_stage_start.items()}
    trace=[]
    start=time.time()
    for global_step0 in range(300):
        step=global_step0+1
        lr=matched_lr(global_step0)
        set_optimizer_lr(opt,lr)
        loss,ce,kl=one_step(m,teacher,opt,scaler,train_chunks[global_step0])
        if step in PHASE_A_STEPS:
            snapshots[step]=snapshot_fixed_q3(qs,reference_scales)
        if global_step0==0 or step%50==0:
            prev,fs=flip_stats(qs,prev,native_stage_start)
            row={"global_step":step,"levels":levels,"lr":lr,"loss":loss,
                 "ce":ce,"kl":kl,**fs}
            trace.append(row)
            print(json.dumps({"event":"t1_native_trace","arm":label,**row}),flush=True)
        if step%50==0:
            nbytes=sum(len(c[0]) for c in snapshots[step].values())
            print(json.dumps({"event":"t1_snapshot","arm":label,"step":step,
                "tensor_count":len(snapshots[step]),"packed_bytes":nbytes}),flush=True)
    native_val=score_on(m,teacher,lrval_chunks)
    projected_val=score_fixed_original_q3(m,qs,reference_scales,teacher)
    result={"native_validation_step300":native_val,
            "fixed_q3_validation_step300":projected_val,
            "trace":trace,"elapsed_seconds":time.time()-start}
    del m,qs,trainable,opt,scaler,prev,native_stage_start
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return snapshots,result

def collect_timing(D,S,source):
    assert set(D)==set(S)==set(PHASE_A_STEPS)
    names=set(source)
    for step in PHASE_A_STEPS:
        assert names==set(D[step])==set(S[step])
    totals={step:{"step":step,"n":0,"D_changed":0,"S_changed":0,
                  "Q9_only":0,"D_only":0,"both_changed_different":0,
                  "disagreement":0,"intersect_M300":0,
                  "Q9_code_matches_S300_on_M300":0,
                  "Q9_only_intersect_final":0,"D_only_intersect_final":0,
                  "M300_size":0,
                  "Q9_only_300_size":0,"D_only_300_size":0,
                  "full_step_layer_rows":[]} for step in PHASE_A_STEPS}
    for name in sorted(source):
        c0=unpack_array_codes(source[name])
        d300=unpack_array_codes(D[300][name])
        s300=unpack_array_codes(S[300][name])
        final_m=(d300!=s300)
        final_q=(s300!=c0)&(d300==c0)
        final_d=(d300!=c0)&(s300==c0)
        for step in PHASE_A_STEPS:
            d=unpack_array_codes(D[step][name])
            s=unpack_array_codes(S[step][name])
            if not(c0.shape==d.shape==s.shape):
                raise AssertionError(("tensor_length_mismatch",name,step))
            dchange=d!=c0; schange=s!=c0
            qonly=schange & ~dchange
            donly=dchange & ~schange
            bothdiff=dchange & schange & (d!=s)
            m=d!=s
            if not np.array_equal(m,qonly|donly|bothdiff):
                raise AssertionError(("mask_partition",name,step))
            z=totals[step]
            z["n"]+=int(len(c0))
            z["D_changed"]+=int(np.count_nonzero(dchange))
            z["S_changed"]+=int(np.count_nonzero(schange))
            z["Q9_only"]+=int(np.count_nonzero(qonly))
            z["D_only"]+=int(np.count_nonzero(donly))
            z["both_changed_different"]+=int(np.count_nonzero(bothdiff))
            z["disagreement"]+=int(np.count_nonzero(m))
            z["intersect_M300"]+=int(np.count_nonzero(m & final_m))
            z["Q9_code_matches_S300_on_M300"]+=int(np.count_nonzero((s==s300)&final_m))
            z["Q9_only_intersect_final"]+=int(np.count_nonzero(qonly & final_q))
            z["D_only_intersect_final"]+=int(np.count_nonzero(donly & final_d))
            z["M300_size"]+=int(np.count_nonzero(final_m))
            z["Q9_only_300_size"]+=int(np.count_nonzero(final_q))
            z["D_only_300_size"]+=int(np.count_nonzero(final_d))
            if step==300:
                z["full_step_layer_rows"].append({
                    "name":name,"n":int(len(c0)),
                    "q9_only":int(np.count_nonzero(qonly)),
                    "direct_only":int(np.count_nonzero(donly)),
                    "both_changed_different":int(np.count_nonzero(bothdiff)),
                    "disagreement":int(np.count_nonzero(m))})
    result=[]
    for step in PHASE_A_STEPS:
        z=totals[step]
        n=z["n"]
        if n!=314572800:
            raise AssertionError(("unexpected_total_quantized_weights",step,n))
        if z["disagreement"]!=z["Q9_only"]+z["D_only"]+z["both_changed_different"]:
            raise AssertionError(("disagreement_count_mismatch",step))
        m=z["disagreement"]; mf=z["M300_size"]; both=z["intersect_M300"]
        result.append({
            "step":step,"total_quantized_weights":n,
            "D_changed_fraction":z["D_changed"]/n,
            "S_changed_fraction":z["S_changed"]/n,
            "Q9_only_fraction":z["Q9_only"]/n,
            "D_only_fraction":z["D_only"]/n,
            "both_changed_different_fraction":z["both_changed_different"]/n,
            "disagreement_fraction":m/n,
            "overlap_with_step300":{
                "intersection":both,
                "precision":both/m if m else None,
                "recall":both/mf if mf else None,
                "jaccard":both/(m+mf-both) if (m+mf-both) else None,
                "S300_code_identity_recall_on_M300":z["Q9_code_matches_S300_on_M300"]/mf if mf else None,
                "Q9_only_mask_recall":z["Q9_only_intersect_final"]/z["Q9_only_300_size"] if z["Q9_only_300_size"] else None,
                "D_only_mask_recall":z["D_only_intersect_final"]/z["D_only_300_size"] if z["D_only_300_size"] else None
            },
            **({"top_layers_by_disagreement":sorted(z["full_step_layer_rows"],
                key=lambda p:p["disagreement"]/p["n"],reverse=True)[:16]} if step==300 else {})
        })
    assert result[0]["disagreement_fraction"]==0.0
    assert result[0]["D_changed_fraction"]==0.0
    assert result[0]["S_changed_fraction"]==0.0
    if abs(result[-1]["disagreement_fraction"]-HISTORICAL_MASK_300)>0.002:
        raise AssertionError(("step300_mask_not_reproduced",result[-1]["disagreement_fraction"]))
    return result

print(json.dumps({
    "event":"t1_start","seed":SEED,"model":MODEL_ID,
    "gpu":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "checkpoints":PHASE_A_STEPS,
    "design":"matched Q3/Q9 300-step-only projected Q3 timing; train-split validation only",
    "heldout_test_loaded":False,"no_q3_continuation":True,
    "order_head":_order[:16],
    "source":"v11 matched preparation; original rowwise Q3 scales"
}),flush=True)
teacher=load_model(torch.float16 if DEVICE=="cuda" else torch.float32)
teacher.requires_grad_(False)
teacher.eval()
reference_scales,initial_codes,reference=initial_q3_reference(teacher)
source=pack_source_q3(initial_codes)
del initial_codes
gc.collect()

D,d_result=run_preparation_with_snapshots("D",3,teacher,reference_scales)
S,s_result=run_preparation_with_snapshots("S",9,teacher,reference_scales)
timing=collect_timing(D,S,source)

expected_diag={
    "D_native":(d_result["native_validation_step300"]["loss"],HISTORICAL_D_NATIVE_300),
    "S_native":(s_result["native_validation_step300"]["loss"],HISTORICAL_S_NATIVE_300),
    "D_fixed_q3":(d_result["fixed_q3_validation_step300"]["loss"],HISTORICAL_D_PROJECTED_300),
    "S_fixed_q3":(s_result["fixed_q3_validation_step300"]["loss"],HISTORICAL_S_PROJECTED_300)}
checks={k:{"observed":float(a),"reference":float(b),
           "difference":float(a-b),"pass":abs(a-b)<0.035}
        for k,(a,b) in expected_diag.items()}
checks["step300_mask_reproduced"]=abs(
    timing[-1]["disagreement_fraction"]-HISTORICAL_MASK_300)<=0.002
checks["step0_matched"]=timing[0]["disagreement_fraction"]==0.0

results={
    "kind":"t1_q9_discovery_timing_seed1729",
    "prereg":"ternary_pet/research_log/phase_a_q9_timing_seed1729_prereg_2026-10-08.md",
    "seed":SEED,"model":MODEL_ID,
    "train_only":True,"test_not_evaluated":True,
    "global_schedule":{"warmup_steps":WARMUP_STEPS,"peak_lr":PEAK_LR,
        "min_lr":MIN_LR,"full_steps":FULL_STEPS,"pilot_steps":300},
    "training_order_head":_order[:16],
    "initial_reference_validation":reference["diag"],
    "D":d_result,"S":s_result,
    "timing":timing,
    "checks":checks,
    "valid_for_science":all(v["pass"] for k,v in checks.items() if isinstance(v,dict)) and checks["step300_mask_reproduced"] and checks["step0_matched"]
}
print("FINAL_JSON_BEGIN",flush=True)
print(json.dumps(results),flush=True)
print("FINAL_JSON_END",flush=True)
if not results["valid_for_science"]:
    raise RuntimeError(("TECHNICAL_REPRODUCIBILITY_DISCREPANCY",checks))
