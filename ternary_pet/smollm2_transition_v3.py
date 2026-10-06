# /// script
# dependencies = ["torch>=2.4","transformers>=4.46","datasets>=3.0","accelerate>=1.0"]
# ///
"""
SmolLM2 ternary transition experiment v3.

Research question:
Can a pretrained 16-bit checkpoint survive ternarization better if we
(a) phase quantization in softly, or
(b) first promote the 16-bit checkpoint to FP32 master weights, allow a
    full-precision adaptation phase, then switch to ternary QAT?

All paths:
- start from the same BF16-rounded checkpoint
- keep FP32 trainable master/shadow weights
- use a learnable per-output-row ternary scale (ParetoQ/LSQ-inspired)
- optimize the same CE + teacher-KL objective
- evaluate on the same WikiText-2 slice

Schedules:
direct_600:
    600 steps lambda=1 (ternary forward from step 1)

soft_200_400:
    200-step smooth lambda ramp 0->1, then 400 steps lambda=1
    (600 total; literature-inspired gradual quantization strength)

updown_equal_total_200_400:
    200 steps lambda=0 (full-precision forward with FP32 master weights),
    then 400 steps lambda=1, same optimizer state
    (600 total)

updown_equal_final_200_600:
    200 steps lambda=0, then 600 steps lambda=1
    (same final ternary budget as direct; 800 total)

Note: merely BF16->FP32 casting cannot restore information. The "up" hypothesis
being tested is whether updates accumulated at FP32 master precision before the
ternary switch create a better launch point.
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
SEED=314159
SEQ=128
CAL_CHUNKS=640
EVAL_CHUNKS=64
LR=2e-5
CE_W=0.35
KL_W=0.65
PRE=200
FINAL=600
TOTAL=600

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

def chunks(split,n):
    ds=load_dataset("Salesforce/wikitext","wikitext-2-raw-v1",split=split)
    buf=[]; out=[]
    for row in ds:
        t=row["text"]
        if not t or not t.strip(): continue
        z=tok.encode(t,add_special_tokens=False)
        if not z: continue
        buf.extend(z+[tok.eos_token_id])
        while len(buf)>=SEQ+1 and len(out)<n:
            out.append(torch.tensor(buf[:SEQ+1],dtype=torch.long))
            buf=buf[SEQ+1:]
        if len(out)>=n: break
    if len(out)<n: raise RuntimeError((split,len(out),n))
    return out

cal=chunks("train",CAL_CHUNKS)
ev=chunks("test",EVAL_CHUNKS)

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

def target_linears(m):
    for name,mod in m.named_modules():
        if isinstance(mod,nn.Linear) and name!="lm_head":
            yield name,mod

def inv_softplus(x):
    # x positive; stable inverse softplus.
    return torch.where(x>20,x,torch.log(torch.expm1(x.clamp_min(1e-6))))

class LearnableTernary(nn.Module):
    """
    Three-state quantizer with a learnable per-output-row span.
    Forward grid is {-2/3*alpha, 0, +2/3*alpha}, matching the ternary
    Stretched Elastic Quant geometry used by ParetoQ for its ternary setting.
    """
    def __init__(self,w):
        super().__init__()
        with torch.no_grad():
            init=w.detach().abs().amax(dim=1,keepdim=True).float().clamp_min(1e-4)
        self.raw_alpha=nn.Parameter(inv_softplus(init))
        self.lam=1.0

    def alpha(self):
        return F.softplus(self.raw_alpha)+1e-6

    def ternary(self,w):
        a=self.alpha().to(dtype=w.dtype)
        z=w/a
        zc=z.clamp(-0.99,0.99)
        hard=torch.round(zc*1.5)/1.5
        code=zc+(hard-zc).detach()
        return code*a

    def hard_code(self,w):
        a=self.alpha().to(device=w.device,dtype=w.dtype)
        z=(w/a).clamp(-0.99,0.99)
        return torch.round(z*1.5).clamp(-1,1).to(torch.int64)

    def forward(self,w):
        if self.lam<=0.0:
            return w
        q=self.ternary(w)
        if self.lam>=1.0:
            return q
        return (1.0-self.lam)*w+self.lam*q

def attach(m):
    qs=[]
    for name,mod in target_linears(m):
        q=LearnableTernary(mod.weight)
        parametrize.register_parametrization(mod,"weight",q)
        qs.append((name,mod,q))
    return qs

def set_lambda(qs,lam):
    for _,_,q in qs: q.lam=float(lam)

@torch.no_grad()
def score(m,teacher):
    m.eval(); teacher.eval()
    nll=0.; nt=0; agree=0; an=0; kls=0.; kn=0
    for ids_cpu in ev:
        x=ids_cpu[:-1].unsqueeze(0).to(DEVICE)
        y=ids_cpu[1:].unsqueeze(0).to(DEVICE)
        with torch.autocast("cuda",dtype=torch.float16,enabled=(DEVICE=="cuda")):
            s=m(x).logits.float()
            t=teacher(x).logits.float()
        ce=F.cross_entropy(s.reshape(-1,s.shape[-1]),y.reshape(-1),reduction="sum")
        nll+=ce.item(); nt+=y.numel()
        sp=s.argmax(-1); tp=t.argmax(-1)
        agree+=(sp==tp).sum().item(); an+=tp.numel()
        kl=F.kl_div(F.log_softmax(s,dim=-1),F.softmax(t,dim=-1),reduction="none").sum(-1)
        kls+=kl.sum().item(); kn+=kl.numel()
    loss=nll/nt
    return {
        "loss":loss,"ppl":math.exp(min(loss,20)),
        "top1_agreement":agree/an,"kl_to_teacher":kls/kn,
        "eval_tokens":nt
    }

@torch.no_grad()
def gen(m):
    m.eval(); m.config.use_cache=True; out=[]
    for p in PROMPTS:
        text=tok.apply_chat_template(
            [{"role":"user","content":p}],
            tokenize=False,add_generation_prompt=True
        )
        e=tok(text,return_tensors="pt").to(DEVICE)
        g=m.generate(**e,max_new_tokens=48,do_sample=False,pad_token_id=tok.eos_token_id)
        out.append(tok.decode(g[0,e.input_ids.shape[1]:],skip_special_tokens=True))
    m.config.use_cache=False
    return out

@torch.no_grad()
def hist(qs):
    c=[0,0,0]; total=0
    for _,mod,q in qs:
        w=mod.parametrizations.weight.original
        code=q.hard_code(w).reshape(-1).cpu()
        bc=torch.bincount(code+1,minlength=3)
        c=[a+int(b) for a,b in zip(c,bc.tolist())]
        total+=code.numel()
    return {"-1":c[0]/total,"0":c[1]/total,"1":c[2]/total}

@torch.no_grad()
def alpha_stats(qs):
    vals=torch.cat([q.alpha().detach().float().cpu().reshape(-1) for _,_,q in qs])
    return {"mean":vals.mean().item(),"std":vals.std().item(),
            "min":vals.min().item(),"max":vals.max().item()}

def lam_for(kind,step,total):
    if kind=="direct": return 1.0
    if kind=="soft":
        if step>=PRE: return 1.0
        # Literature-inspired 2*sigmoid(5t/T)-1 schedule.
        x=5.0*(step+1)/PRE
        return float(2.0/(1.0+math.exp(-x))-1.0)
    if kind=="updown":
        return 0.0 if step<PRE else 1.0
    raise ValueError(kind)

def run_variant(label,kind,total_steps,teacher):
    m=load(torch.float32)
    bf16_roundtrip_(m)  # every variant begins at the identical 16-bit source.
    qs=attach(m)
    set_lambda(qs,lam_for(kind,0,total_steps))
    opt=torch.optim.AdamW(m.parameters(),lr=LR,betas=(0.9,0.95),weight_decay=0.0)
    scaler=torch.amp.GradScaler("cuda",enabled=(DEVICE=="cuda"))
    first_post_switch=None
    trace=[]
    start=time.time()
    m.train(); teacher.eval()
    for step in range(total_steps):
        lam=lam_for(kind,step,total_steps)
        set_lambda(qs,lam)
        ids=cal[step%len(cal)]
        x=ids[:-1].unsqueeze(0).to(DEVICE); y=ids[1:].unsqueeze(0).to(DEVICE)
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
        torch.nn.utils.clip_grad_norm_(m.parameters(),1.0)
        scaler.step(opt); scaler.update()
        if kind=="updown" and step==PRE:
            first_post_switch=float(loss.detach().cpu())
        if step in (0,PRE-1,PRE,total_steps-1) or (step+1)%100==0:
            row={"step":step+1,"lambda":lam,"loss":float(loss.detach().cpu()),
                 "ce":float(ce.detach().cpu()),"kl":float(kl.detach().cpu())}
            trace.append(row)
            print(json.dumps({"event":"train","label":label,**row}),flush=True)
    set_lambda(qs,1.0)
    metrics=score(m,teacher)
    metrics["gen"]=gen(m)
    metrics["hist"]=hist(qs)
    metrics["alpha"]=alpha_stats(qs)
    metrics["trace"]=trace
    metrics["seconds"]=time.time()-start
    metrics["first_post_switch_loss"]=first_post_switch
    del opt,scaler,m,qs
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return metrics

print(json.dumps({
    "event":"start","model":MODEL_ID,"device":DEVICE,
    "gpu":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "seq":SEQ,"cal_chunks":CAL_CHUNKS,"eval_chunks":EVAL_CHUNKS,
    "lr":LR,"pre":PRE,"final":FINAL
}),flush=True)

teacher=load(torch.float16 if DEVICE=="cuda" else torch.float32)
teacher.requires_grad_(False)

# Baseline is the explicitly BF16-rounded source, evaluated against untouched teacher.
base=load(torch.float32); bf16_roundtrip_(base)
results={"bf16_source":score(base,teacher)}
results["bf16_source"]["gen"]=gen(base)
del base; gc.collect()
if torch.cuda.is_available(): torch.cuda.empty_cache()

results["direct_600"]=run_variant("direct_600","direct",600,teacher)
results["soft_200_400"]=run_variant("soft_200_400","soft",600,teacher)
results["updown_equal_total_200_400"]=run_variant(
    "updown_equal_total_200_400","updown",600,teacher
)
results["updown_equal_final_200_600"]=run_variant(
    "updown_equal_final_200_600","updown",800,teacher
)

print("FINAL_JSON_BEGIN",flush=True)
print(json.dumps(results),flush=True)
print("FINAL_JSON_END",flush=True)
