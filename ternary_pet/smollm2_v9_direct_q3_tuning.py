# /// script
# dependencies = ["torch>=2.4","transformers>=4.46","datasets>=3.0","accelerate>=1.0"]
# ///
"""
SmolLM2 ternary experiment v9: direct-Q3 baseline tuning.

Purpose:
Stress-test the direct ternary baseline before further mechanism claims.

Protocol:
1. Screen a compact set of higher constant learning rates and warmup+cosine
   schedules for 300 direct-Q3 updates using ONLY the existing 24-chunk
   validation split.
2. Select the two non-reference candidates with lowest validation loss.
3. Retrain from the identical BF16-rounded source for the full 1200 Q3 updates:
   - original constant 1e-4 reference;
   - the best two validation-selected alternatives.
4. Use the held-out 8192-token test evaluator only after candidates are fixed.

The first 300 schedule values for a warmup+cosine candidate are identical
between screening and its full 1200-step run (the schedule horizon is always
1200). Non-quantized parameters remain frozen.
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
FLIP_EVERY=100
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


CANDIDATES=[
    {"name":"const_1e-4","kind":"constant","peak_lr":1e-4},
    {"name":"const_3e-4","kind":"constant","peak_lr":3e-4},
    {"name":"const_5e-4","kind":"constant","peak_lr":5e-4},
    {"name":"const_1e-3","kind":"constant","peak_lr":1e-3},
    {"name":"const_3e-3","kind":"constant","peak_lr":3e-3},
    {"name":"warm100_cosine_3e-4","kind":"warmup_cosine","peak_lr":3e-4,"warmup_steps":100,"min_lr":3e-5},
    {"name":"warm100_cosine_1e-3","kind":"warmup_cosine","peak_lr":1e-3,"warmup_steps":100,"min_lr":1e-4},
]
REFERENCE_NAME="const_1e-4"
SCREEN_STEPS=300
FULL_STEPS=1200

def lr_for(cfg,step0):
    if cfg["kind"]=="constant":
        return float(cfg["peak_lr"])
    if cfg["kind"]=="warmup_cosine":
        warm=int(cfg["warmup_steps"])
        peak=float(cfg["peak_lr"]); floor=float(cfg["min_lr"])
        s=step0+1
        if s<=warm:
            return peak*s/max(1,warm)
        frac=(s-warm)/max(1,FULL_STEPS-warm)
        frac=min(max(frac,0.0),1.0)
        return floor+0.5*(peak-floor)*(1.0+math.cos(math.pi*frac))
    raise ValueError(cfg)

def set_optimizer_lr(opt,lr):
    for g in opt.param_groups:
        g["lr"]=float(lr)

def reset_rng():
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

def direct_run(cfg,teacher,steps,score_test=False):
    reset_rng()
    m=load_model(torch.float32); bf16_roundtrip_(m)
    qs,trainable=attach_quantizers(m,3)
    stage_start=snapshot_codes(qs)
    prev={k:v.clone() for k,v in stage_start.items()}
    opt=make_optimizer(trainable,float(cfg["peak_lr"]))
    scaler=torch.amp.GradScaler("cuda",enabled=(DEVICE=="cuda"))
    trace=[]; diverged=False; start=time.time()

    for step in range(steps):
        lr=lr_for(cfg,step)
        set_optimizer_lr(opt,lr)
        loss,ce,kl=one_step(m,teacher,opt,scaler,train_chunks[step])
        if not (math.isfinite(loss) and math.isfinite(ce) and math.isfinite(kl)):
            diverged=True
            print(json.dumps({
                "event":"v9_nonfinite","name":cfg["name"],"step":step+1,
                "lr":lr,"loss":loss,"ce":ce,"kl":kl
            }),flush=True)
            break
        if step==0 or (step+1)%100==0 or step+1==steps:
            prev,fs=flip_stats(qs,prev,stage_start)
            val=score_on(m,teacher,lrval_chunks) if (step+1 in [300,600,900,1200]) else None
            row={
                "step":step+1,"lr":lr,"loss":loss,"ce":ce,"kl":kl,
                "validation":val,**fs
            }
            trace.append(row)
            print(json.dumps({"event":"v9_train","name":cfg["name"],**row}),flush=True)

    if diverged:
        val={"loss":float("inf"),"ppl":float("inf"),"top1_agreement":0.0,
             "kl_to_teacher":float("inf"),"tokens":len(lrval_chunks)*SEQ}
        test=None
    else:
        val=score_on(m,teacher,lrval_chunks)
        test=score_on(m,teacher,eval_chunks) if score_test else None
        if test is not None:
            test["gen"]=generate(m)
            test["final_hist"]=code_hist(qs)
            test["scale"]=scale_stats(qs)

    out={
        "config":cfg,
        "steps_completed":0 if diverged and not trace else (trace[-1]["step"] if trace else 0),
        "diverged":diverged,
        "validation":val,
        "test":test,
        "trace":trace,
        "final_hist":None if diverged else code_hist(qs),
        "scale":None if diverged else scale_stats(qs),
        "seconds":time.time()-start
    }

    del opt,scaler,m,qs,trainable,prev,stage_start
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return out

print(json.dumps({
    "event":"v9_start","model":MODEL_ID,"seed":SEED,"device":DEVICE,
    "gpu":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "screen_steps":SCREEN_STEPS,"full_steps":FULL_STEPS,
    "selection_split":"lrval_chunks_only",
    "heldout_test_used_for_selection":False,
    "candidates":CANDIDATES,
    "order_head":_order[:16]
}),flush=True)

teacher=load_model(torch.float16 if DEVICE=="cuda" else torch.float32)
teacher.requires_grad_(False)

base=load_model(torch.float32); bf16_roundtrip_(base)
bf16_test=score_on(base,teacher,eval_chunks)
del base; gc.collect()
if torch.cuda.is_available(): torch.cuda.empty_cache()

screen=[]
for cfg in CANDIDATES:
    row=direct_run(cfg,teacher,SCREEN_STEPS,score_test=False)
    screen.append(row)
    print(json.dumps({
        "event":"v9_screen_result","name":cfg["name"],
        "diverged":row["diverged"],"validation":row["validation"]
    }),flush=True)

ranked=sorted(
    [r for r in screen if not r["diverged"]],
    key=lambda r:r["validation"]["loss"]
)
nonref=[r for r in ranked if r["config"]["name"]!=REFERENCE_NAME]
selected_names=[REFERENCE_NAME]+[r["config"]["name"] for r in nonref[:2]]
selected_cfg=[next(c for c in CANDIDATES if c["name"]==n) for n in selected_names]

print(json.dumps({
    "event":"v9_selected",
    "rule":"reference const_1e-4 plus two lowest validation-loss non-reference candidates after 300 updates",
    "selected":selected_names,
    "screen_ranking":[
        {"name":r["config"]["name"],"validation_loss":r["validation"]["loss"]}
        for r in ranked
    ]
}),flush=True)

full={}
for cfg in selected_cfg:
    full[cfg["name"]]=direct_run(cfg,teacher,FULL_STEPS,score_test=True)
    print(json.dumps({
        "event":"v9_full_result","name":cfg["name"],
        "validation":full[cfg["name"]]["validation"],
        "test":full[cfg["name"]]["test"]
    }),flush=True)

results={
    "v9_seed":SEED,
    "model":MODEL_ID,
    "training_order_head":_order[:16],
    "selection_rule":"validation-only 300-step screen; full test only after selection",
    "candidates":CANDIDATES,
    "screen":screen,
    "selected_names":selected_names,
    "full":full,
    "bf16_source_test":bf16_test,
    "historical_q9_seed1729_reference":{
        "source":"v7 seed1729; external comparison only, never used for v9 selection",
        "loss":5.193768184632063,
        "ppl":180.14609933317848,
        "top1_agreement":0.337890625,
        "kl_to_teacher":2.255982033908367
    }
}

print("FINAL_JSON_BEGIN",flush=True)
print(json.dumps(results),flush=True)
print("FINAL_JSON_END",flush=True)
