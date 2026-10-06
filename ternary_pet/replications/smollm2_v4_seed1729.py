# /// script
# dependencies = ["torch>=2.4","transformers>=4.46","datasets>=3.0","accelerate>=1.0"]
# ///
"""
SmolLM2 ternary experiment v4: make the codes move, then test 9 -> 3.

Design learned from v1-v3:
- fixed WikiText-2 held-out slice (8192 tokens)
- identical BF16-rounded source checkpoint for every run
- persistent FP32 shadow/master weights
- NO hard commits between 9 and 3
- SAME optimizer state across the 9 -> 3 transition
- only quantized linear shadow weights + quantizer scales are trainable
- BitNet-like absmean initialization for ternary scale
- exact discrete-code flip-rate logging
- short LR diagnostic sweep before the main comparison

Main comparison at selected LR:
A direct_1200:         1200 ternary steps
B staged_9_300_3_900: 300 nine-state + 900 ternary steps
C direct_900:           900 ternary steps

Thus B vs A is equal-total-compute (1200 vs 1200), while B's final ternary
phase vs C gives the same 900 ternary-step budget.

The shared learnable span alpha is initialized as 1.5 * rowwise mean(|w|).
For 3 states:
    q = round(clamp(w/alpha,-.99,.99)*1.5)/1.5 * alpha
so the nonzero magnitude begins at mean(|w|), matching BitNet-style absmean,
and the zero threshold begins at 0.5 * mean(|w|).

For 9 states:
    q = round(clamp(w/alpha,-.99,.99)*4)/4 * alpha
giving {-alpha, -.75alpha, ..., +alpha}.
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

print(json.dumps({
    "event":"start","model":MODEL_ID,"device":DEVICE,
    "gpu":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "seed":SEED,"train_chunks":TRAIN_CHUNKS,"lrval_chunks":LRVAL_CHUNKS,
    "eval_chunks":EVAL_CHUNKS,"seq":SEQ
}),flush=True)

teacher=load_model(torch.float16 if DEVICE=="cuda" else torch.float32)
teacher.requires_grad_(False)

base=load_model(torch.float32); bf16_roundtrip_(base)
results={"bf16_source":score_on(base,teacher,eval_chunks)}
results["bf16_source"]["gen"]=generate(base)
del base; gc.collect()
if torch.cuda.is_available(): torch.cuda.empty_cache()

sweep,chosen_lr=lr_sweep(teacher)
results["lr_sweep"]=sweep
results["chosen_lr"]=chosen_lr
print(json.dumps({"event":"lr_chosen","lr":chosen_lr}),flush=True)

results["direct_1200"]=run_main(
    "direct_1200",[(3,1200)],chosen_lr,teacher
)
results["staged_9_300_3_900"]=run_main(
    "staged_9_300_3_900",[(9,300),(3,900)],chosen_lr,teacher
)
results["direct_900"]=run_main(
    "direct_900",[(3,900)],chosen_lr,teacher
)

print("FINAL_JSON_BEGIN",flush=True)
print(json.dumps(results),flush=True)
print("FINAL_JSON_END",flush=True)
