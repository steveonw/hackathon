# /// script
# dependencies = ["torch>=2.4","transformers>=4.46","datasets>=3.0","accelerate>=1.0"]
# ///
"""
SmolLM2 v12: code identity vs continuous position on the replicated Q9 mask.

Seed/order 1729 first.

Rebuild matched-schedule step-300 direct-Q3 masters D and Q9 masters S.
Define M where their projections through the same original Q3 scales choose
different ternary codes.

Continuation arms, all D outside the modified positions:
  D       : D everywhere.
  exact   : exact S masters on the true mask M (v11 arm-10 positive control).
  proto   : Q3 reconstruction prototype for S's code on M.
  minimal : just inside S's Q3 code region on M (epsilon=0.01 in w/alpha).
  random  : layer/source/target-transition-matched random reassignment outside M,
            using Q3 reconstruction prototypes.

exact/proto/minimal are required to have exactly the same projected Q3 codes
and identical pre-continuation forward diagnostics. They therefore isolate
hidden continuous master position while holding the ternary forward model fixed.

The random control preserves, per layer, the observed direct-code -> Q9-code
transition counts but applies them to randomly selected positions outside M.
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
SEED=271828
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


V12_EPS=0.01
V12_ARMS=["D","exact","proto","minimal","random"]

@torch.no_grad()
def set_reference_scales(qs,reference_scales):
    for name,mod,q in qs:
        raw=reference_scales[name]["raw_alpha"]
        q.raw_alpha.copy_(raw.to(device=q.raw_alpha.device,dtype=q.raw_alpha.dtype))
        q.levels=3

@torch.no_grad()
def transition_summary(d_codes,s_codes,diff_mask,d_masters,s_masters,reference_scales):
    global_counts=defaultdict(int)
    target_counts=defaultdict(int)
    layer_counts={}
    target_margin={
        "target_zero":{"n":0,"D_sum":0.0,"S_sum":0.0},
        "target_nonzero":{"n":0,"D_sum":0.0,"S_sum":0.0},
    }
    for name in d_codes:
        d=d_codes[name].reshape(-1)
        s=s_codes[name].reshape(-1)
        m=diff_mask[name].reshape(-1)
        ld=defaultdict(int)
        for cd in (-1,0,1):
            for cs in (-1,0,1):
                if cd==cs:
                    continue
                n=int((m & (d==cd) & (s==cs)).sum().item())
                if n:
                    key=f"{cd}->{cs}"
                    global_counts[key]+=n
                    ld[key]+=n
                    target_counts[str(cs)]+=n
        layer_counts[name]=dict(ld)

        idx=m.nonzero(as_tuple=False).flatten()
        if idx.numel():
            a=reference_scales[name]["alpha"].float().expand_as(d_masters[name]).reshape(-1)
            zd=(d_masters[name].float().reshape(-1)/a)[idx]
            zs=(s_masters[name].float().reshape(-1)/a)[idx]
            ss=s[idx]
            md=torch.minimum((zd-1/3).abs(),(zd+1/3).abs())
            ms=torch.minimum((zs-1/3).abs(),(zs+1/3).abs())
            for key,sel in [("target_zero",ss==0),("target_nonzero",ss!=0)]:
                n=int(sel.sum().item())
                if n:
                    target_margin[key]["n"]+=n
                    target_margin[key]["D_sum"]+=float(md[sel].sum().item())
                    target_margin[key]["S_sum"]+=float(ms[sel].sum().item())

    for key,v in target_margin.items():
        n=max(1,v["n"])
        v["D_mean_nearest_boundary_distance"]=v.pop("D_sum")/n
        v["S_mean_nearest_boundary_distance"]=v.pop("S_sum")/n

    return {
        "transition_counts":dict(global_counts),
        "target_code_counts":dict(target_counts),
        "per_layer_transition_counts":layer_counts,
        "boundary_distance_by_target":target_margin,
    }

@torch.no_grad()
def build_matched_random_plan(d_codes,s_codes,diff_mask):
    plan={}
    audit={}
    total_selected=0
    for li,name in enumerate(d_codes):
        d=d_codes[name].reshape(-1)
        s=s_codes[name].reshape(-1)
        m=diff_mask[name].reshape(-1)
        idx_parts=[]; tgt_parts=[]
        layer_audit={}
        for cd in (-1,0,1):
            transitions=[]
            need_total=0
            for cs in (-1,0,1):
                if cs==cd:
                    continue
                n=int((m & (d==cd) & (s==cs)).sum().item())
                if n:
                    transitions.append((cs,n))
                    need_total+=n
            if need_total==0:
                continue

            cand=((~m) & (d==cd)).nonzero(as_tuple=False).flatten()
            if cand.numel()<need_total:
                raise RuntimeError(("insufficient_random_candidates",name,cd,
                                    int(cand.numel()),need_total))
            g=torch.Generator().manual_seed(SEED*1000003 + 12012 + li*97 + (cd+1)*17)
            perm=torch.randperm(cand.numel(),generator=g)[:need_total]
            chosen=cand[perm]
            offset=0
            for cs,n in sorted(transitions):
                part=chosen[offset:offset+n]
                idx_parts.append(part.to(torch.int32))
                tgt_parts.append(torch.full((n,),cs,dtype=torch.int8))
                layer_audit[f"{cd}->{cs}"]=n
                offset+=n
            assert offset==need_total

        if idx_parts:
            idx=torch.cat(idx_parts)
            tgt=torch.cat(tgt_parts)
        else:
            idx=torch.empty(0,dtype=torch.int32)
            tgt=torch.empty(0,dtype=torch.int8)
        if idx.numel()!=torch.unique(idx).numel():
            raise RuntimeError(("duplicate_random_positions",name))
        if idx.numel() and bool(m[idx.long()].any().item()):
            raise RuntimeError(("random_overlap_true_mask",name))
        plan[name]={"idx":idx,"target":tgt}
        audit[name]=layer_audit
        total_selected+=int(idx.numel())

    return plan,{"total_selected":total_selected,
                 "per_layer_transition_counts":audit}

@torch.no_grad()
def prototype_value(alpha,target_code):
    return (2.0/3.0)*alpha*target_code.to(alpha.dtype)

@torch.no_grad()
def minimal_value(alpha,source_code,target_code):
    eps=V12_EPS
    z=torch.empty_like(alpha,dtype=torch.float32)
    tc=target_code.to(torch.int8)
    sc=source_code.to(torch.int8)

    z[tc==1]=(1.0/3.0)+eps
    z[tc==-1]=-(1.0/3.0)-eps
    sel=(tc==0) & (sc==1)
    z[sel]=(1.0/3.0)-eps
    sel=(tc==0) & (sc==-1)
    z[sel]=-(1.0/3.0)+eps

    bad=(tc==0) & (sc==0)
    if bool(bad.any().item()):
        raise RuntimeError("target-zero minimal construction received same-code position")
    return alpha.float()*z

@torch.no_grad()
def load_v12_arm(qs,arm,d_masters,s_masters,d_codes,s_codes,diff_mask,
                 reference_scales,random_plan):
    for name,mod,q in qs:
        base=d_masters[name].clone()
        a=reference_scales[name]["alpha"].float().expand_as(base)
        dcode=d_codes[name]
        scode=s_codes[name]
        m=diff_mask[name]

        if arm=="D":
            h=base
        elif arm=="exact":
            h=torch.where(m,s_masters[name],base)
        elif arm=="proto":
            h=base
            h[m]=prototype_value(a[m],scode[m])
        elif arm=="minimal":
            h=base
            h[m]=minimal_value(a[m],dcode[m],scode[m])
        elif arm=="random":
            h=base
            rp=random_plan[name]
            idx=rp["idx"].long()
            if idx.numel():
                flat=h.reshape(-1)
                af=a.reshape(-1)
                flat[idx]=prototype_value(af[idx],rp["target"])
        else:
            raise ValueError(arm)

        w=mod.parametrizations.weight.original
        w.copy_(h.to(device=w.device,dtype=w.dtype))

@torch.no_grad()
def projected_codes_for_random_plan(d_codes,random_plan):
    out={name:c.clone() for name,c in d_codes.items()}
    for name,c in out.items():
        rp=random_plan[name]
        idx=rp["idx"].long()
        if idx.numel():
            c.reshape(-1)[idx]=rp["target"]
    return out

def instantiate_v12(arm,d_masters,s_masters,d_codes,s_codes,diff_mask,
                    reference_scales,random_plan,teacher):
    m=load_model(torch.float32); bf16_roundtrip_(m)
    qs,trainable=attach_quantizers(m,3)
    load_v12_arm(qs,arm,d_masters,s_masters,d_codes,s_codes,diff_mask,
                 reference_scales,random_plan)
    set_reference_scales(qs,reference_scales)
    codes=snapshot_codes(qs)
    diag=score_on(m,teacher,lrval_chunks)
    return m,qs,trainable,codes,diag

def continue_v12(arm,d_masters,s_masters,d_codes,s_codes,diff_mask,
                 reference_scales,random_plan,teacher):
    reset_rng()
    m,qs,trainable,start_codes,pre_diag=instantiate_v12(
        arm,d_masters,s_masters,d_codes,s_codes,diff_mask,
        reference_scales,random_plan,teacher
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
            print(json.dumps({"event":"v12_continue","arm":arm,**row}),flush=True)

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
        "start_codes_vs_final":hamming_codes(start_codes,final_codes)
    }

    del opt,scaler,m,qs,trainable,prev,stage_start,start_codes,final_codes
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return out

def diag_close(a,b,tol=1e-5):
    keys=["loss","ppl","top1_agreement","kl_to_teacher"]
    return all(abs(float(a[k])-float(b[k]))<=tol for k in keys)

print(json.dumps({
    "event":"v12_start",
    "seed":SEED,"model":MODEL_ID,"device":DEVICE,
    "gpu":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "epsilon_normalized":V12_EPS,
    "arms":V12_ARMS,
    "global_schedule":{
        "lr_step_300":matched_lr(299),
        "lr_step_301":matched_lr(300)
    },
    "order_head":_order[:16]
}),flush=True)

teacher=load_model(torch.float16 if DEVICE=="cuda" else torch.float32)
teacher.requires_grad_(False)

reference_scales,initial_codes,initial_reference=initial_q3_reference(teacher)
d_masters,d_prep=prepare_matched("D_direct_q3_300",3,teacher)
s_masters,s_prep=prepare_matched("S_q9_300",9,teacher)

d_codes=projected_q3_codes_from_masters(d_masters,reference_scales)
s_codes=projected_q3_codes_from_masters(s_masters,reference_scales)
diff_mask,mask_stats=build_diff_mask(d_codes,s_codes)
transitions=transition_summary(
    d_codes,s_codes,diff_mask,d_masters,s_masters,reference_scales
)
random_plan,random_audit=build_matched_random_plan(d_codes,s_codes,diff_mask)
random_expected_codes=projected_codes_for_random_plan(d_codes,random_plan)

print(json.dumps({
    "event":"v12_mask",
    "mask_stats":mask_stats,
    "transition_counts":transitions["transition_counts"],
    "target_code_counts":transitions["target_code_counts"],
    "boundary_distance_by_target":transitions["boundary_distance_by_target"],
    "random_total_selected":random_audit["total_selected"]
}),flush=True)

pre_codes={}
pre_diags={}
for arm in V12_ARMS:
    m,qs,trainable,codes,diag=instantiate_v12(
        arm,d_masters,s_masters,d_codes,s_codes,diff_mask,
        reference_scales,random_plan,teacher
    )
    pre_codes[arm]=codes
    pre_diags[arm]=diag
    print(json.dumps({"event":"v12_pre_diag","arm":arm,"diag":diag}),flush=True)
    del m,qs,trainable
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

checks={
    "exact_vs_proto_hamming":hamming_codes(pre_codes["exact"],pre_codes["proto"]),
    "exact_vs_minimal_hamming":hamming_codes(pre_codes["exact"],pre_codes["minimal"]),
    "exact_vs_proto_diag_close":diag_close(pre_diags["exact"],pre_diags["proto"]),
    "exact_vs_minimal_diag_close":diag_close(pre_diags["exact"],pre_diags["minimal"]),
    "exact_vs_proto_loss_absdiff":abs(pre_diags["exact"]["loss"]-pre_diags["proto"]["loss"]),
    "exact_vs_minimal_loss_absdiff":abs(pre_diags["exact"]["loss"]-pre_diags["minimal"]["loss"]),
    "D_vs_exact_hamming":hamming_codes(pre_codes["D"],pre_codes["exact"]),
    "D_vs_random_hamming":hamming_codes(pre_codes["D"],pre_codes["random"]),
    "random_vs_expected_hamming":hamming_codes(pre_codes["random"],random_expected_codes),
}
assert checks["exact_vs_proto_hamming"]["fraction"]==0.0, checks
assert checks["exact_vs_minimal_hamming"]["fraction"]==0.0, checks
assert checks["exact_vs_proto_diag_close"], checks
assert checks["exact_vs_minimal_diag_close"], checks
assert checks["random_vs_expected_hamming"]["fraction"]==0.0, checks
assert abs(checks["D_vs_random_hamming"]["fraction"]-mask_stats["fraction"])<1e-12, checks
assert random_audit["total_selected"]==mask_stats["changed_count"], (random_audit,mask_stats)
print(json.dumps({"event":"v12_assertions_passed","checks":checks}),flush=True)

del pre_codes,random_expected_codes
gc.collect()

arms={}
for arm in V12_ARMS:
    arms[arm]=continue_v12(
        arm,d_masters,s_masters,d_codes,s_codes,diff_mask,
        reference_scales,random_plan,teacher
    )
    print(json.dumps({
        "event":"v12_arm_final","arm":arm,
        "final":arms[arm]["final"]
    }),flush=True)

L={arm:arms[arm]["final"]["loss"] for arm in V12_ARMS}
exact_gain=L["D"]-L["exact"]
effects={
    "D_loss":L["D"],
    "exact_loss":L["exact"],
    "prototype_loss":L["proto"],
    "minimal_loss":L["minimal"],
    "random_loss":L["random"],
    "exact_gain_vs_D":exact_gain,
    "prototype_gain_vs_D":L["D"]-L["proto"],
    "minimal_gain_vs_D":L["D"]-L["minimal"],
    "random_gain_vs_D":L["D"]-L["random"],
    "prototype_recovery_of_exact":
        (L["D"]-L["proto"])/exact_gain if abs(exact_gain)>1e-12 else float("nan"),
    "minimal_recovery_of_exact":
        (L["D"]-L["minimal"])/exact_gain if abs(exact_gain)>1e-12 else float("nan"),
    "random_recovery_of_exact":
        (L["D"]-L["random"])/exact_gain if abs(exact_gain)>1e-12 else float("nan"),
    "exact_minus_prototype":L["proto"]-L["exact"],
    "exact_minus_minimal":L["minimal"]-L["exact"],
}

results={
    "v12_seed":SEED,
    "model":MODEL_ID,
    "training_order_head":_order[:16],
    "epsilon_normalized":V12_EPS,
    "global_schedule":{
        "warmup_steps":WARMUP_STEPS,"peak_lr":PEAK_LR,
        "min_lr":MIN_LR,"full_steps":FULL_STEPS,
        "lr_step_1":matched_lr(0),"lr_step_100":matched_lr(99),
        "lr_step_300":matched_lr(299),"lr_step_301":matched_lr(300),
        "lr_step_1200":matched_lr(1199)
    },
    "design":{
        "D":"direct-Q3 masters everywhere",
        "exact":"exact Q9 masters on true disagreement mask M; D elsewhere",
        "proto":"Q3 reconstruction prototype of S code on M; D elsewhere",
        "minimal":"epsilon-inside S Q3 code region on M; D elsewhere",
        "random":"per-layer/source/target-transition-matched random reassignment outside M using prototypes",
        "all_continuations":"original Q3 scales + fresh Adam + same global LR steps 301-1200"
    },
    "initial_q3_reference":initial_reference,
    "D_prep":d_prep,
    "S_prep":s_prep,
    "prep_geometry":{
        "D_vs_initial":hamming_codes(initial_codes,d_codes),
        "S_vs_initial":hamming_codes(initial_codes,s_codes),
        "D_vs_S":hamming_codes(d_codes,s_codes),
        "mask_stats":mask_stats,
        "overlap_vs_initial":code_overlap_against_initial(initial_codes,d_codes,s_codes),
        "transition_summary":transitions,
        "random_audit":random_audit,
    },
    "pre_diags":pre_diags,
    "checks":checks,
    "arms":arms,
    "effects":effects
}

print("FINAL_JSON_BEGIN",flush=True)
print(json.dumps(results),flush=True)
print("FINAL_JSON_END",flush=True)
