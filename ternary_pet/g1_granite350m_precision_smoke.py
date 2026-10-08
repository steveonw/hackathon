# /// script
# dependencies = ["torch>=2.4","transformers>=4.57.1","datasets>=3.0","accelerate>=1.0"]
# ///
"""
G1-0b: Granite 4.0 350M precision diagnostic.

Technical follow-up to G1-0. No scientific D-vs-S comparison.
The first smoke showed NaNs in the unquantized source evaluation and teacher KL
under the inherited FP16 compute path. This script compares FP32, BF16, and
FP16 source numerics, then tests one Q3 and one Q9 update using BF16 compute.
"""
import gc, json, math
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
CE_W=0.35
KL_W=0.65
LR=1e-4

torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

tok=AutoTokenizer.from_pretrained(MODEL_ID)
if tok.pad_token_id is None:
    tok.pad_token=tok.eos_token

def chunks(split,n):
    ds=load_dataset("Salesforce/wikitext","wikitext-2-raw-v1",split=split)
    buf=[]; out=[]
    for row in ds:
        t=row["text"]
        if not t or not t.strip():
            continue
        ids=tok.encode(t,add_special_tokens=False)
        if not ids:
            continue
        buf.extend(ids+[tok.eos_token_id])
        while len(buf)>=SEQ+1 and len(out)<n:
            out.append(torch.tensor(buf[:SEQ+1],dtype=torch.long))
            buf=buf[SEQ+1:]
        if len(out)>=n:
            break
    if len(out)<n:
        raise RuntimeError((split,len(out),n))
    return out

train=chunks("train",2)
test=chunks("test",2)

def load(dtype=torch.float32):
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

def cleanup(*xs):
    for x in xs:
        del x
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

@torch.no_grad()
def logits_probe(m,ids_cpu,mode):
    x=ids_cpu[:-1].unsqueeze(0).to(DEVICE)
    y=ids_cpu[1:].unsqueeze(0).to(DEVICE)
    enabled=(DEVICE=="cuda" and mode!="fp32")
    dtype=torch.bfloat16 if mode=="bf16" else torch.float16
    with torch.autocast("cuda",dtype=dtype,enabled=enabled):
        z=m(x).logits
    zf=z.float()
    finite=torch.isfinite(zf)
    finvals=zf[finite]
    ce=F.cross_entropy(zf.reshape(-1,zf.shape[-1]),y.reshape(-1))
    return {
        "mode":mode,
        "logits_dtype":str(z.dtype),
        "finite_fraction":float(finite.float().mean().item()),
        "nan_count":int(torch.isnan(zf).sum().item()),
        "inf_count":int(torch.isinf(zf).sum().item()),
        "max_abs_finite":float(finvals.abs().max().item()) if finvals.numel() else None,
        "ce":float(ce.item()),
        "ce_finite":bool(torch.isfinite(ce).item()),
    }

source=load(torch.float32)
bf16_roundtrip_(source)
source_probes={}
for mode in ["fp32","bf16","fp16"]:
    source_probes[mode]=logits_probe(source,test[0],mode)
    print(json.dumps({"event":"source_probe",**source_probes[mode]}),flush=True)
cleanup(source)

# Direct-dtype teacher probe to match G1-0's teacher construction.
teacher_probes={}
for label,dtype,mode in [
    ("teacher_bf16",torch.bfloat16,"bf16"),
    ("teacher_fp16",torch.float16,"fp16"),
]:
    m=load(dtype)
    teacher_probes[label]=logits_probe(m,test[0],mode)
    print(json.dumps({"event":"teacher_probe","label":label,**teacher_probes[label]}),flush=True)
    cleanup(m)

def inv_softplus(x):
    return torch.where(x>20,x,torch.log(torch.expm1(x.clamp_min(1e-7))))

class SharedScaleQuant(nn.Module):
    def __init__(self,w,levels):
        super().__init__()
        with torch.no_grad():
            ma=w.detach().abs().mean(dim=1,keepdim=True).float().clamp_min(1e-7)
            a=1.5*ma
        self.raw_alpha=nn.Parameter(inv_softplus(a))
        self.levels=int(levels)
    def alpha(self):
        return F.softplus(self.raw_alpha)+1e-7
    def n(self):
        return 1.5 if self.levels==3 else 4.0
    def hard_code(self,w):
        a=self.alpha().to(device=w.device,dtype=w.dtype)
        z=(w/a).clamp(-0.99,0.99)
        return torch.round(z*self.n()).to(torch.int8)
    def forward(self,w):
        a=self.alpha().to(dtype=w.dtype)
        n=self.n()
        z=(w/a).clamp(-0.99,0.99)
        hard=torch.round(z*n)/n
        return (z+(hard-z).detach())*a

def targets(m):
    for name,mod in m.named_modules():
        if isinstance(mod,nn.Linear) and name!="lm_head":
            yield name,mod

def attach(m,levels):
    for p in m.parameters():
        p.requires_grad_(False)
    qs=[]; trainable=[]
    for name,mod in targets(m):
        q=SharedScaleQuant(mod.weight,levels)
        parametrize.register_parametrization(mod,"weight",q)
        w=mod.parametrizations.weight.original
        w.requires_grad_(True)
        q.raw_alpha.requires_grad_(True)
        qs.append((name,mod,q))
        trainable.extend([w,q.raw_alpha])
    return qs,trainable

@torch.no_grad()
def hist(qs):
    counts=defaultdict(int); total=0
    for _,mod,q in qs:
        c=q.hard_code(mod.parametrizations.weight.original).reshape(-1).cpu()
        for v,n in zip(*torch.unique(c,return_counts=True)):
            counts[int(v.item())]+=int(n.item())
        total+=c.numel()
    return {
        "zero_fraction":counts.get(0,0)/max(1,total),
        "total":total
    }

def bf16_step(levels,teacher,ids_cpu):
    m=load(torch.float32)
    bf16_roundtrip_(m)
    qs,trainable=attach(m,levels)
    opt=torch.optim.AdamW(trainable,lr=LR,betas=(0.9,0.95),weight_decay=0.0)
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
    grad_norm=torch.nn.utils.clip_grad_norm_(trainable,1.0)
    finite_grads=all(
        p.grad is None or bool(torch.isfinite(p.grad).all().item())
        for p in trainable
    )
    opt.step()
    out={
        "levels":levels,
        "ce":float(ce.detach().cpu()),
        "kl":float(kl.detach().cpu()),
        "loss":float(loss.detach().cpu()),
        "ce_finite":bool(torch.isfinite(ce).item()),
        "kl_finite":bool(torch.isfinite(kl).item()),
        "loss_finite":bool(torch.isfinite(loss).item()),
        "grad_norm":float(grad_norm.detach().cpu()),
        "grad_norm_finite":bool(torch.isfinite(grad_norm).item()),
        "all_trainable_grads_finite":finite_grads,
        "post_step_hist":hist(qs),
        "peak_cuda_gib":(
            torch.cuda.max_memory_allocated()/(1024**3)
            if torch.cuda.is_available() else None
        )
    }
    cleanup(opt,m,qs,trainable)
    return out

if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()

teacher=load(torch.bfloat16)
teacher.requires_grad_(False)
teacher.eval()

bf16_steps={}
for levels,idx in [(3,0),(9,1)]:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    bf16_steps[f"q{levels}"]=bf16_step(levels,teacher,train[idx])
    print(json.dumps({"event":"bf16_step",**bf16_steps[f"q{levels}"]}),flush=True)

decision={
    "source_fp32_finite":source_probes["fp32"]["ce_finite"],
    "source_bf16_finite":source_probes["bf16"]["ce_finite"],
    "source_fp16_finite":source_probes["fp16"]["ce_finite"],
    "teacher_bf16_finite":teacher_probes["teacher_bf16"]["ce_finite"],
    "teacher_fp16_finite":teacher_probes["teacher_fp16"]["ce_finite"],
    "q3_bf16_step_finite":bf16_steps["q3"]["loss_finite"] and bf16_steps["q3"]["all_trainable_grads_finite"],
    "q9_bf16_step_finite":bf16_steps["q9"]["loss_finite"] and bf16_steps["q9"]["all_trainable_grads_finite"],
}
decision["bf16_path_pass"] = all([
    decision["source_fp32_finite"],
    decision["source_bf16_finite"],
    decision["teacher_bf16_finite"],
    decision["q3_bf16_step_finite"],
    decision["q9_bf16_step_finite"],
])

final={
    "kind":"g1_0b_granite350m_precision_smoke",
    "model":MODEL_ID,
    "seed":SEED,
    "device":DEVICE,
    "gpu":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "source_probes":source_probes,
    "teacher_probes":teacher_probes,
    "bf16_steps":bf16_steps,
    "decision":decision,
    "scientific_result":False
}
print("FINAL_JSON_BEGIN",flush=True)
print(json.dumps(final,indent=2),flush=True)
print("FINAL_JSON_END",flush=True)
