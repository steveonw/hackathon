# /// script
# dependencies = [
#   "torch>=2.4",
#   "transformers>=4.46",
#   "accelerate>=1.0",
#   "datasets>=3.0",
# ]
# ///
"""
SmolLM2 nested balanced-ternary staircase v2.

Strict hierarchy:
  final ternary magnitude = alpha
  inherited base step s = alpha / 9

  27 states: n*s, n=-13..13
   9 states: p*3s, p=-4..4
   3 states: a*9s, a=-1..1

Each three adjacent child states have exactly one fixed parent.
The per-row alpha is estimated once from the original FP weights and then
held fixed for every stage and every control.

Comparisons:
  - baseline
  - direct 3-level PTQ
  - strict nested 27->9->3 PTQ
  - direct 3-level QAT (90 steps)
  - nested QAT equal-total-compute (30+30+30)
  - nested QAT equal-final-ternary-budget (30+30+90)

Recovery uses a diverse WikiText-2 calibration stream and teacher-logit
distillation against the untouched original model.
"""
import gc, json, math, time
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
CAL_CHUNKS=96
EVAL_CHUNKS=24
DIRECT_STEPS=90
PRE_STEPS=30
LR=8e-6
CE_WEIGHT=0.30
KL_WEIGHT=0.70

torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

PROMPTS=[
    "In two sentences, explain why a baseline matters in an experiment.",
    "What is 17 + 28? Give a short answer.",
    "Write one calm sentence about a dog waiting outside a library.",
    "What is one possible benefit of reducing numerical precision gradually?",
]

def load_model(dtype=torch.float32):
    m=AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(DEVICE)
    m.config.use_cache=False
    return m

tok=AutoTokenizer.from_pretrained(MODEL_ID)
if tok.pad_token_id is None:
    tok.pad_token=tok.eos_token

def stream_chunks(split, n_chunks):
    ds=load_dataset("Salesforce/wikitext","wikitext-2-raw-v1",split=split)
    buf=[]
    out=[]
    for row in ds:
        text=row["text"]
        if not text or not text.strip():
            continue
        ids=tok.encode(text,add_special_tokens=False)
        if not ids:
            continue
        buf.extend(ids+[tok.eos_token_id])
        while len(buf)>=SEQ+1 and len(out)<n_chunks:
            out.append(torch.tensor(buf[:SEQ+1],dtype=torch.long))
            buf=buf[SEQ+1:]
        if len(out)>=n_chunks:
            break
    if len(out)<n_chunks:
        raise RuntimeError(f"Only built {len(out)} chunks for {split}")
    return out

cal_chunks=stream_chunks("train",CAL_CHUNKS)
eval_chunks=stream_chunks("test",EVAL_CHUNKS)

def qlinears(m):
    for name,mod in m.named_modules():
        if isinstance(mod,nn.Linear) and name!="lm_head":
            yield name,mod

@torch.no_grad()
def ternary_alpha(w):
    # Rowwise symmetric Lloyd-style centroid for {-alpha,0,+alpha}.
    a=w.abs().mean(dim=1,keepdim=True).clamp_min(1e-8)
    aw=w.abs()
    for _ in range(6):
        mask=aw >= (a*0.5)
        num=(aw*mask).sum(dim=1,keepdim=True)
        den=mask.sum(dim=1,keepdim=True).clamp_min(1)
        a=(num/den).clamp_min(1e-8)
    return a

@torch.no_grad()
def make_alphas(m):
    return {name:ternary_alpha(mod.weight.detach()).cpu()
            for name,mod in qlinears(m)}

def stage_spec(levels):
    if levels==27:
        return 13,9   # codes -13..13, unit alpha/9
    if levels==9:
        return 4,3    # codes -4..4, unit alpha/3
    if levels==3:
        return 1,1    # codes -1..1, unit alpha
    raise ValueError(levels)

def quantize_tensor(w,alpha,levels):
    k,den=stage_spec(levels)
    unit=alpha.to(w.device,dtype=w.dtype)/den
    code=torch.round(w/unit).clamp(-k,k)
    return code*unit

def code_index_from_exact(w,alpha,levels):
    k,den=stage_spec(levels)
    unit=alpha.to(w.device,dtype=w.dtype)/den
    code=torch.round(w/unit).clamp(-k,k).to(torch.int64)
    return code+k

def hard_quantize(m,alphas,levels):
    with torch.no_grad():
        for name,mod in qlinears(m):
            mod.weight.copy_(quantize_tensor(mod.weight,alphas[name],levels))

def hard_parent_transition(m,alphas,child_levels,parent_levels):
    assert child_levels//3==parent_levels
    pk,pden=stage_spec(parent_levels)
    with torch.no_grad():
        for name,mod in qlinears(m):
            idx=code_index_from_exact(mod.weight,alphas[name],child_levels)
            pidx=torch.div(idx,3,rounding_mode="floor")
            pcode=pidx-pk
            unit=alphas[name].to(mod.weight.device,dtype=mod.weight.dtype)/pden
            mod.weight.copy_(pcode*unit)

class FixedNestedQuant(nn.Module):
    def __init__(self,alpha,levels):
        super().__init__()
        self.register_buffer("alpha",alpha)
        self.levels=levels
    def forward(self,w):
        q=quantize_tensor(w,self.alpha,self.levels)
        return w+(q-w).detach()

def attach_qat(m,alphas,levels):
    params=[]
    for name,mod in qlinears(m):
        if parametrize.is_parametrized(mod,"weight"):
            raise RuntimeError("weight already parametrized")
        a=alphas[name].to(mod.weight.device,dtype=mod.weight.dtype)
        parametrize.register_parametrization(
            mod,"weight",FixedNestedQuant(a,levels)
        )
        params.append(mod.parametrizations.weight.original)
    return params

def commit_qat(m):
    for _,mod in list(qlinears(m)):
        if parametrize.is_parametrized(mod,"weight"):
            parametrize.remove_parametrizations(
                mod,"weight",leave_parametrized=True
            )

@torch.no_grad()
def score(m,teacher):
    m.eval(); teacher.eval()
    nll=0.0; ntok=0; agree=0; agree_n=0; kl_sum=0.0; kl_n=0
    for ids_cpu in eval_chunks:
        x=ids_cpu[:-1].unsqueeze(0).to(DEVICE)
        y=ids_cpu[1:].unsqueeze(0).to(DEVICE)
        with torch.autocast("cuda",dtype=torch.float16,enabled=(DEVICE=="cuda")):
            s=m(x).logits.float()
            t=teacher(x).logits.float()
        ce=F.cross_entropy(s.reshape(-1,s.shape[-1]),y.reshape(-1),reduction="sum")
        nll+=ce.item(); ntok+=y.numel()
        sp=s.argmax(-1); tp=t.argmax(-1)
        agree+=(sp==tp).sum().item(); agree_n+=tp.numel()
        kl=F.kl_div(
            F.log_softmax(s,dim=-1),
            F.softmax(t,dim=-1),
            reduction="none"
        ).sum(-1)
        kl_sum+=kl.sum().item(); kl_n+=kl.numel()
    loss=nll/ntok
    return {
        "loss":loss,
        "ppl":math.exp(min(loss,20)),
        "top1_agreement":agree/agree_n,
        "kl_to_teacher":kl_sum/kl_n,
        "eval_tokens":ntok,
    }

@torch.no_grad()
def generate(m):
    m.eval(); m.config.use_cache=True
    out=[]
    for p in PROMPTS:
        text=tok.apply_chat_template(
            [{"role":"user","content":p}],
            tokenize=False,
            add_generation_prompt=True,
        )
        e=tok(text,return_tensors="pt").to(DEVICE)
        g=m.generate(
            **e,max_new_tokens=48,do_sample=False,
            pad_token_id=tok.eos_token_id
        )
        out.append(tok.decode(
            g[0,e.input_ids.shape[1]:],skip_special_tokens=True
        ))
    m.config.use_cache=False
    return out

def train_stage(m,teacher,alphas,levels,steps,label):
    params=attach_qat(m,alphas,levels)
    opt=torch.optim.AdamW(params,lr=LR,betas=(0.9,0.95),weight_decay=0.0)
    scaler=torch.amp.GradScaler("cuda",enabled=(DEVICE=="cuda"))
    losses=[]; ces=[]; kls=[]; start=time.time()
    m.train(); teacher.eval()
    for step in range(steps):
        ids_cpu=cal_chunks[step%len(cal_chunks)]
        x=ids_cpu[:-1].unsqueeze(0).to(DEVICE)
        y=ids_cpu[1:].unsqueeze(0).to(DEVICE)
        opt.zero_grad(set_to_none=True)
        with torch.no_grad():
            with torch.autocast("cuda",dtype=torch.float16,enabled=(DEVICE=="cuda")):
                t=teacher(x).logits
        with torch.autocast("cuda",dtype=torch.float16,enabled=(DEVICE=="cuda")):
            s=m(x).logits
            ce=F.cross_entropy(
                s.reshape(-1,s.shape[-1]),
                y.reshape(-1)
            )
            kl=F.kl_div(
                F.log_softmax(s.float(),dim=-1),
                F.softmax(t.float(),dim=-1),
                reduction="none"
            ).sum(-1).mean()
            loss=CE_WEIGHT*ce+KL_WEIGHT*kl
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params,1.0)
        scaler.step(opt); scaler.update()
        losses.append(float(loss.detach().cpu()))
        ces.append(float(ce.detach().cpu()))
        kls.append(float(kl.detach().cpu()))
        if step==0 or (step+1)%10==0 or step+1==steps:
            print(json.dumps({
                "event":"train","label":label,"levels":levels,
                "step":step+1,"steps":steps,
                "loss":losses[-1],"ce":ces[-1],"kl":kls[-1],
                "elapsed_s":round(time.time()-start,2)
            }),flush=True)
    commit_qat(m)
    del opt,scaler
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return {
        "levels":levels,"steps":steps,
        "first_loss":losses[0],"last_loss":losses[-1],
        "first_ce":ces[0],"last_ce":ces[-1],
        "first_kl":kls[0],"last_kl":kls[-1],
        "seconds":time.time()-start,
    }

def state_hist(m,alphas,levels=3):
    k,_=stage_spec(levels)
    counts=[0]*(2*k+1)
    total=0
    with torch.no_grad():
        for name,mod in qlinears(m):
            idx=code_index_from_exact(mod.weight,alphas[name],levels)
            bc=torch.bincount(idx.reshape(-1).cpu(),minlength=2*k+1)
            counts=[a+int(b) for a,b in zip(counts,bc.tolist())]
            total+=idx.numel()
    return {
        str(i-k):counts[i]/total for i in range(2*k+1)
    }

def cleanup(m):
    del m
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

print(json.dumps({
    "event":"start","model":MODEL_ID,"device":DEVICE,
    "gpu":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "seq":SEQ,"cal_chunks":CAL_CHUNKS,"eval_chunks":EVAL_CHUNKS,
    "direct_steps":DIRECT_STEPS,"pre_steps":PRE_STEPS,
    "lr":LR,"ce_weight":CE_WEIGHT,"kl_weight":KL_WEIGHT
}),flush=True)

teacher=load_model(torch.float16 if DEVICE=="cuda" else torch.float32)
teacher.requires_grad_(False)
base_for_scale=load_model(torch.float32)
alphas=make_alphas(base_for_scale)

results={}
results["baseline"]=score(teacher,teacher)
results["baseline"]["gen"]=generate(teacher)
cleanup(base_for_scale)

# Direct raw ternary, same final codebook as every staged path.
m=load_model(torch.float32)
hard_quantize(m,alphas,3)
results["direct_ptq_3"]=score(m,teacher)
results["direct_ptq_3"]["gen"]=generate(m)
results["direct_ptq_3"]["hist"]=state_hist(m,alphas,3)
cleanup(m)

# Strict raw lineage 27 -> 9 -> 3.
m=load_model(torch.float32)
hard_quantize(m,alphas,27)
hard_parent_transition(m,alphas,27,9)
hard_parent_transition(m,alphas,9,3)
results["nested_ptq_27_9_3"]=score(m,teacher)
results["nested_ptq_27_9_3"]["gen"]=generate(m)
results["nested_ptq_27_9_3"]["hist"]=state_hist(m,alphas,3)
cleanup(m)

# Direct QAT: 90 final-ternary steps.
m=load_model(torch.float32)
direct_train=train_stage(m,teacher,alphas,3,DIRECT_STEPS,"direct_3")
results["direct_qat_3"]=score(m,teacher)
results["direct_qat_3"]["train"]=[direct_train]
results["direct_qat_3"]["gen"]=generate(m)
results["direct_qat_3"]["hist"]=state_hist(m,alphas,3)
cleanup(m)

# Nested QAT equal total compute: 30+30+30 = 90.
m=load_model(torch.float32)
nested_equal=[]
nested_equal.append(train_stage(m,teacher,alphas,27,PRE_STEPS,"nested_equal_27"))
hard_parent_transition(m,alphas,27,9)
nested_equal.append(train_stage(m,teacher,alphas,9,PRE_STEPS,"nested_equal_9"))
hard_parent_transition(m,alphas,9,3)
nested_equal.append(train_stage(m,teacher,alphas,3,PRE_STEPS,"nested_equal_3"))
results["nested_qat_equal_total"]=score(m,teacher)
results["nested_qat_equal_total"]["train"]=nested_equal
results["nested_qat_equal_total"]["gen"]=generate(m)
results["nested_qat_equal_total"]["hist"]=state_hist(m,alphas,3)
cleanup(m)

# Nested QAT with same final ternary budget as direct:
# 30 at 27 + 30 at 9 + 90 at 3.
m=load_model(torch.float32)
nested_final=[]
nested_final.append(train_stage(m,teacher,alphas,27,PRE_STEPS,"nested_final_27"))
hard_parent_transition(m,alphas,27,9)
nested_final.append(train_stage(m,teacher,alphas,9,PRE_STEPS,"nested_final_9"))
hard_parent_transition(m,alphas,9,3)
nested_final.append(train_stage(m,teacher,alphas,3,DIRECT_STEPS,"nested_final_3"))
results["nested_qat_equal_final"]=score(m,teacher)
results["nested_qat_equal_final"]["train"]=nested_final
results["nested_qat_equal_final"]["gen"]=generate(m)
results["nested_qat_equal_final"]["hist"]=state_hist(m,alphas,3)
cleanup(m)

print("FINAL_JSON_BEGIN",flush=True)
print(json.dumps(results),flush=True)
print("FINAL_JSON_END",flush=True)
