# /// script
# dependencies = ["torch>=2.4","transformers>=4.57.1","datasets>=3.0","accelerate>=1.0"]
# ///
"""
G1-1: Granite 4.0 350M direct-Q3 schedule calibration.

Validation-only direct-Q3 screen at seed/order 1729.
No Q9 arm is present and the held-out test split is never evaluated.

Candidates:
A) constant 1e-4
B) warmup 100 -> 3e-4, cosine -> 1e-4 through step 1200
C) warmup 100 -> 1e-3, cosine -> 1e-4 through step 1200

Each candidate is trained for the same first 300 shuffled training chunks and
scored only on the existing 24-chunk LR-validation split. Lowest validation
loss wins and becomes frozen for G1-2.
"""
import gc, json, math, time
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import parametrize
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID="ibm-granite/granite-4.0-350m"
DEVICE="cuda" if torch.cuda.is_available() else "cpu"
SEED=1729
SEQ=128
TRAIN_CHUNKS=1200
LRVAL_CHUNKS=24
SCREEN_STEPS=300
FULL_HORIZON=1200
CE_W=0.35
KL_W=0.65
CLIP_NORM=1.0

CANDIDATES=[
    {"name":"const_1e-4","kind":"constant","peak_lr":1e-4,"min_lr":1e-4},
    {"name":"warm100_cosine_3e-4","kind":"warmup_cosine","peak_lr":3e-4,"min_lr":1e-4,"warmup_steps":100},
    {"name":"warm100_cosine_1e-3","kind":"warmup_cosine","peak_lr":1e-3,"min_lr":1e-4,"warmup_steps":100},
]

torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

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
g=torch.Generator().manual_seed(SEED)
order=torch.randperm(len(train_chunks),generator=g).tolist()
train_chunks=[train_chunks[i] for i in order]

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
    def __init__(self,w):
        super().__init__()
        with torch.no_grad():
            meanabs=w.detach().abs().mean(dim=1,keepdim=True).float().clamp_min(1e-7)
            self_init=1.5*meanabs
        self.raw_alpha=nn.Parameter(inv_softplus(self_init))
    def alpha(self):
        return F.softplus(self.raw_alpha)+1e-7
    def hard_code(self,w):
        a=self.alpha().to(device=w.device,dtype=w.dtype)
        z=(w/a).clamp(-0.99,0.99)
        return torch.round(z*1.5).to(torch.int8)
    def forward(self,w):
        a=self.alpha().to(dtype=w.dtype)
        z=(w/a).clamp(-0.99,0.99)
        hard=torch.round(z*1.5)/1.5
        return (z+(hard-z).detach())*a

def attach_q3(m):
    for p in m.parameters():
        p.requires_grad_(False)
    qs=[]; trainable=[]
    for name,mod in target_linears(m):
        q=SharedScaleQuant(mod.weight)
        parametrize.register_parametrization(mod,"weight",q)
        shadow=mod.parametrizations.weight.original
        shadow.requires_grad_(True)
        q.raw_alpha.requires_grad_(True)
        qs.append((name,mod,q))
        trainable.extend([shadow,q.raw_alpha])
    return qs,trainable

@torch.no_grad()
def snapshot_codes(qs):
    return {
        name:q.hard_code(mod.parametrizations.weight.original).detach().cpu().clone()
        for name,mod,q in qs
    }

@torch.no_grad()
def code_hist(qs):
    counts=defaultdict(int); total=0
    for _,mod,q in qs:
        c=q.hard_code(mod.parametrizations.weight.original).reshape(-1).cpu()
        for v,n in zip(*torch.unique(c,return_counts=True)):
            counts[int(v.item())]+=int(n.item())
        total+=c.numel()
    return {
        "fractions":{str(k):counts[k]/max(1,total) for k in sorted(counts)},
        "zero_fraction":counts.get(0,0)/max(1,total),
        "total":total
    }

@torch.no_grad()
def diff_fraction(qs,start):
    changed=0; total=0
    for name,mod,q in qs:
        c=q.hard_code(mod.parametrizations.weight.original).detach().cpu()
        s=start[name]
        changed+=int((c!=s).sum().item())
        total+=c.numel()
    return changed/max(1,total)

def lr_for(cfg,step0):
    if cfg["kind"]=="constant":
        return float(cfg["peak_lr"])
    s=step0+1
    warm=int(cfg["warmup_steps"])
    peak=float(cfg["peak_lr"])
    floor=float(cfg["min_lr"])
    if s<=warm:
        return peak*s/max(1,warm)
    frac=(s-warm)/max(1,FULL_HORIZON-warm)
    frac=min(max(frac,0.0),1.0)
    return floor+0.5*(peak-floor)*(1.0+math.cos(math.pi*frac))

def set_lr(opt,lr):
    for group in opt.param_groups:
        group["lr"]=float(lr)

@torch.no_grad()
def score_on(m,teacher,chunks):
    m.eval(); teacher.eval()
    nll=0.0; nt=0; agree=0; an=0; kls=0.0; kn=0
    for ids_cpu in chunks:
        x=ids_cpu[:-1].unsqueeze(0).to(DEVICE)
        y=ids_cpu[1:].unsqueeze(0).to(DEVICE)
        with torch.autocast("cuda",dtype=torch.bfloat16,enabled=(DEVICE=="cuda")):
            s=m(x).logits.float()
            t=teacher(x).logits.float()
        if not torch.isfinite(s).all() or not torch.isfinite(t).all():
            raise RuntimeError("non-finite logits during BF16 validation")
        ce=F.cross_entropy(s.reshape(-1,s.shape[-1]),y.reshape(-1),reduction="sum")
        kl=F.kl_div(
            F.log_softmax(s,dim=-1),
            F.softmax(t,dim=-1),
            reduction="none"
        ).sum(-1)
        nll+=float(ce.item()); nt+=y.numel()
        agree+=int((s.argmax(-1)==t.argmax(-1)).sum().item()); an+=t.shape[0]*t.shape[1]
        kls+=float(kl.sum().item()); kn+=kl.numel()
    loss=nll/nt
    return {
        "loss":loss,
        "ppl":math.exp(min(loss,20)),
        "top1_agreement":agree/an,
        "kl_to_teacher":kls/kn,
        "tokens":nt
    }

def one_step(m,teacher,opt,trainable,ids_cpu):
    x=ids_cpu[:-1].unsqueeze(0).to(DEVICE)
    y=ids_cpu[1:].unsqueeze(0).to(DEVICE)
    opt.zero_grad(set_to_none=True)
    with torch.no_grad():
        with torch.autocast("cuda",dtype=torch.bfloat16,enabled=(DEVICE=="cuda")):
            t=teacher(x).logits
    with torch.autocast("cuda",dtype=torch.bfloat16,enabled=(DEVICE=="cuda")):
        s=m(x).logits
        ce=F.cross_entropy(s.reshape(-1,s.shape[-1]),y.reshape(-1))
        kl=F.kl_div(
            F.log_softmax(s.float(),dim=-1),
            F.softmax(t.float(),dim=-1),
            reduction="none"
        ).sum(-1).mean()
        loss=CE_W*ce+KL_W*kl
    loss.backward()
    grad_norm=torch.nn.utils.clip_grad_norm_(trainable,CLIP_NORM)
    finite=bool(torch.isfinite(loss).item()) and bool(torch.isfinite(grad_norm).item())
    if finite:
        finite=all(
            p.grad is None or bool(torch.isfinite(p.grad).all().item())
            for p in trainable
        )
    if not finite:
        raise RuntimeError({
            "reason":"nonfinite_training_step",
            "loss":float(loss.detach().cpu()),
            "ce":float(ce.detach().cpu()),
            "kl":float(kl.detach().cpu()),
            "grad_norm":float(grad_norm.detach().cpu())
        })
    opt.step()
    return {
        "loss":float(loss.detach().cpu()),
        "ce":float(ce.detach().cpu()),
        "kl":float(kl.detach().cpu()),
        "grad_norm_preclip":float(grad_norm.detach().cpu())
    }

def cleanup(*xs):
    for x in xs:
        del x
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def reset_rng():
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

def run_candidate(cfg,teacher):
    reset_rng()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    m=load_model(torch.float32)
    bf16_roundtrip_(m)
    qs,trainable=attach_q3(m)
    start_codes=snapshot_codes(qs)
    opt=torch.optim.AdamW(
        trainable,
        lr=float(cfg["peak_lr"]),
        betas=(0.9,0.95),
        weight_decay=0.0
    )
    trace=[]; t0=time.time()
    for step in range(SCREEN_STEPS):
        lr=lr_for(cfg,step)
        set_lr(opt,lr)
        row=one_step(m,teacher,opt,trainable,train_chunks[step])
        if step==0 or (step+1)%100==0 or step+1==SCREEN_STEPS:
            log={
                "step":step+1,
                "lr":lr,
                **row,
                "different_from_start":diff_fraction(qs,start_codes)
            }
            trace.append(log)
            print(json.dumps({"event":"g1_1_train","candidate":cfg["name"],**log}),flush=True)
    val=score_on(m,teacher,lrval_chunks)
    out={
        "candidate":cfg,
        "validation":val,
        "trace":trace,
        "final_hist":code_hist(qs),
        "different_from_start":diff_fraction(qs,start_codes),
        "peak_cuda_gib":(
            torch.cuda.max_memory_allocated()/(1024**3)
            if torch.cuda.is_available() else None
        ),
        "seconds":time.time()-t0
    }
    print(json.dumps({"event":"g1_1_candidate_result",**out}),flush=True)
    cleanup(opt,m,qs,trainable,start_codes)
    return out

print(json.dumps({
    "event":"g1_1_start",
    "model":MODEL_ID,
    "seed":SEED,
    "device":DEVICE,
    "gpu":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "precision":"BF16 autocast + BF16 teacher; FP32 masters",
    "screen_steps":SCREEN_STEPS,
    "full_horizon_for_lr_curve":FULL_HORIZON,
    "selection_split":"24 lrval chunks only",
    "heldout_test_used":False,
    "candidates":CANDIDATES,
    "order_head":order[:16]
}),flush=True)

teacher=load_model(torch.bfloat16 if DEVICE=="cuda" else torch.float32)
teacher.requires_grad_(False)
teacher.eval()

source=load_model(torch.float32)
bf16_roundtrip_(source)
source_val=score_on(source,teacher,lrval_chunks)
cleanup(source)

rows=[run_candidate(cfg,teacher) for cfg in CANDIDATES]
ranked=sorted(rows,key=lambda r:r["validation"]["loss"])
winner=ranked[0]["candidate"]

final={
    "kind":"g1_1_granite350m_direct_q3_schedule_calibration",
    "model":MODEL_ID,
    "seed":SEED,
    "precision":"bf16",
    "scientific_result":False,
    "selection_rule":"lowest validation loss after 300 direct-Q3 updates; no Q9 and no held-out test",
    "source_validation":source_val,
    "candidates":rows,
    "ranking":[
        {
            "name":r["candidate"]["name"],
            "validation_loss":r["validation"]["loss"],
            "validation_ppl":r["validation"]["ppl"],
            "different_from_start":r["different_from_start"]
        }
        for r in ranked
    ],
    "winner":winner,
    "training_order_head":order[:16]
}

print("FINAL_JSON_BEGIN",flush=True)
print(json.dumps(final,indent=2),flush=True)
print("FINAL_JSON_END",flush=True)
