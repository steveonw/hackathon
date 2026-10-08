# /// script
# dependencies = ["torch>=2.4","transformers>=4.57.1","datasets>=3.0","accelerate>=1.0"]
# ///
"""
G1-2: Granite 4.0 350M cross-family staging gate.

Scientific question:
Does the SmolLM2 Q9 -> Q3 trainability effect reproduce on a second model
family under a direct-selected, frozen schedule?

Preregistered arms, seed/order 1729:
  D: 1200 direct-Q3 updates
  S: 300 Q9 updates -> restore original Q3 scales + fresh Adam -> 900 Q3 updates

Both use the G1-1-selected constant LR 1e-4, the same shuffled training chunks,
the same CE35 + teacher-KL65 objective, FP32 masters, and the G1-0b-approved
BF16 compute path.

At equal compute (step 300), D and S masters are separately projected through
the SAME original Q3 scales and scored on the SAME 24-chunk diagnostic split.
The held-out 64-chunk test evaluator is used only for final D/S endpoints.
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
EVAL_CHUNKS=64
LR=1e-4
PREP_STEPS=300
CONT_STEPS=900
CE_W=0.35
KL_W=0.65
CLIP_NORM=1.0
TRACE_EVERY=100

PROMPTS=[
    "In two sentences, explain why a baseline matters in an experiment.",
    "What is 17 + 28? Give a short answer.",
    "Write one calm sentence about a dog waiting outside a library.",
    "What is one possible benefit of reducing numerical precision gradually?",
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
    def __init__(self,w,levels):
        super().__init__()
        with torch.no_grad():
            meanabs=w.detach().abs().mean(dim=1,keepdim=True).float().clamp_min(1e-7)
            init_alpha=1.5*meanabs
        self.raw_alpha=nn.Parameter(inv_softplus(init_alpha))
        self.levels=int(levels)

    def alpha(self):
        return F.softplus(self.raw_alpha)+1e-7

    def n(self):
        if self.levels==3:
            return 1.5
        if self.levels==9:
            return 4.0
        raise ValueError(self.levels)

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

def attach_quantizers(m,levels):
    for p in m.parameters():
        p.requires_grad_(False)
    qs=[]; trainable=[]
    for name,mod in target_linears(m):
        q=SharedScaleQuant(mod.weight,levels)
        parametrize.register_parametrization(mod,"weight",q)
        shadow=mod.parametrizations.weight.original
        shadow.requires_grad_(True)
        q.raw_alpha.requires_grad_(True)
        qs.append((name,mod,q))
        trainable.extend([shadow,q.raw_alpha])
    return qs,trainable

@torch.no_grad()
def capture_scale_state(qs):
    return {
        name:{
            "raw_alpha":q.raw_alpha.detach().cpu().clone(),
            "alpha":q.alpha().detach().cpu().clone(),
        }
        for name,_,q in qs
    }

@torch.no_grad()
def capture_masters(qs):
    return {
        name:mod.parametrizations.weight.original.detach().cpu().clone()
        for name,mod,_ in qs
    }

@torch.no_grad()
def load_masters_scales(qs,masters,scales):
    for name,mod,q in qs:
        w=mod.parametrizations.weight.original
        w.copy_(masters[name].to(device=w.device,dtype=w.dtype))
        q.raw_alpha.copy_(
            scales[name]["raw_alpha"].to(
                device=q.raw_alpha.device,dtype=q.raw_alpha.dtype
            )
        )
        q.levels=3

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
def scale_stats(qs):
    vals=torch.cat([
        q.alpha().detach().float().cpu().reshape(-1) for _,_,q in qs
    ])
    return {
        "mean":float(vals.mean().item()),
        "std":float(vals.std().item()),
        "min":float(vals.min().item()),
        "max":float(vals.max().item())
    }

@torch.no_grad()
def hamming_codes(a,b):
    changed=0; total=0; rows=[]
    for name in a:
        x=a[name]; y=b[name]
        ch=int((x!=y).sum().item()); n=x.numel()
        changed+=ch; total+=n
        rows.append((ch/max(1,n),name,ch,n))
    rows.sort(reverse=True)
    return {
        "fraction":changed/max(1,total),
        "changed":changed,
        "total":total,
        "top_layers":[
            {"name":name,"fraction":fr,"changed":ch,"total":n}
            for fr,name,ch,n in rows[:20]
        ]
    }

@torch.no_grad()
def changed_overlap(initial,a,b):
    ca=cb=inter=union=same_final_both=total=0
    for name in initial:
        i=initial[name]; x=a[name]; y=b[name]
        ma=x!=i; mb=y!=i
        total+=i.numel()
        ca+=int(ma.sum().item()); cb+=int(mb.sum().item())
        both=ma & mb
        inter+=int(both.sum().item())
        union+=int((ma|mb).sum().item())
        same_final_both+=int((both & (x==y)).sum().item())
    return {
        "d_changed_fraction":ca/max(1,total),
        "s_changed_fraction":cb/max(1,total),
        "intersection_fraction":inter/max(1,total),
        "union_fraction":union/max(1,total),
        "jaccard":inter/max(1,union),
        "same_final_code_among_both_changed":same_final_both/max(1,inter)
    }

@torch.no_grad()
def disagreement_transitions(d_codes,s_codes):
    total=0; disagree=0
    global_counts=defaultdict(int)
    layer_rows=[]
    for name in d_codes:
        d=d_codes[name].reshape(-1).to(torch.int16)
        s=s_codes[name].reshape(-1).to(torch.int16)
        mask=d!=s
        n=d.numel()
        ch=int(mask.sum().item())
        total+=n; disagree+=ch
        trans={}
        if ch:
            dm=d[mask]; sm=s[mask]
            pairs=torch.stack([dm,sm],dim=1)
            uniq,cnt=torch.unique(pairs,dim=0,return_counts=True)
            for pair,k in zip(uniq,cnt):
                key=f"{int(pair[0].item())}->{int(pair[1].item())}"
                val=int(k.item())
                trans[key]=val
                global_counts[key]+=val
        layer_rows.append({
            "name":name,
            "disagreement_fraction":ch/max(1,n),
            "disagreement_count":ch,
            "total":n,
            "transition_counts":trans
        })
    layer_rows.sort(key=lambda r:r["disagreement_fraction"],reverse=True)
    return {
        "fraction":disagree/max(1,total),
        "count":disagree,
        "total":total,
        "global_transition_counts":dict(sorted(global_counts.items())),
        "layers":layer_rows
    }

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
            raise RuntimeError("non-finite BF16 logits during evaluation")
        ce=F.cross_entropy(
            s.reshape(-1,s.shape[-1]),y.reshape(-1),reduction="sum"
        )
        kl=F.kl_div(
            F.log_softmax(s,dim=-1),
            F.softmax(t,dim=-1),
            reduction="none"
        ).sum(-1)
        nll+=float(ce.item()); nt+=y.numel()
        agree+=int((s.argmax(-1)==t.argmax(-1)).sum().item())
        an+=t.shape[0]*t.shape[1]
        kls+=float(kl.sum().item()); kn+=kl.numel()
    loss=nll/nt
    return {
        "loss":loss,
        "ppl":math.exp(min(loss,20)),
        "top1_agreement":agree/an,
        "kl_to_teacher":kls/kn,
        "tokens":nt
    }

@torch.no_grad()
def generate(m):
    m.eval(); m.config.use_cache=True
    out=[]
    for p in PROMPTS:
        text=tok.apply_chat_template(
            [{"role":"user","content":p}],
            tokenize=False,add_generation_prompt=True
        )
        e=tok(text,return_tensors="pt").to(DEVICE)
        with torch.autocast("cuda",dtype=torch.bfloat16,enabled=(DEVICE=="cuda")):
            g=m.generate(
                **e,max_new_tokens=48,do_sample=False,
                pad_token_id=tok.eos_token_id
            )
        out.append(tok.decode(
            g[0,e.input_ids.shape[1]:],skip_special_tokens=True
        ))
    m.config.use_cache=False
    return out

def make_optimizer(trainable):
    return torch.optim.AdamW(
        trainable,lr=LR,betas=(0.9,0.95),weight_decay=0.0
    )

def one_step(m,teacher,opt,trainable,ids_cpu):
    m.train()
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

def direct_arm(teacher):
    m=load_model(torch.float32)
    bf16_roundtrip_(m)
    qs,trainable=attach_quantizers(m,3)

    reference_scales=capture_scale_state(qs)
    initial_codes=snapshot_codes(qs)
    initial_hist=code_hist(qs)

    opt=make_optimizer(trainable)
    trace=[]
    step300={}
    t0=time.time()

    for step0 in range(TRAIN_CHUNKS):
        row=one_step(m,teacher,opt,trainable,train_chunks[step0])
        step=step0+1
        if step==1 or step%TRACE_EVERY==0:
            log={"step":step,**row}
            trace.append(log)
            print(json.dumps({"event":"g1_2_direct","step":step,**row}),flush=True)
        if step==PREP_STEPS:
            step300={
                "masters":capture_masters(qs),
                "native_diag":score_on(m,teacher,lrval_chunks),
                "native_hist":code_hist(qs),
                "learned_scale":scale_stats(qs),
                "learned_codes":snapshot_codes(qs)
            }

    final=score_on(m,teacher,eval_chunks)
    final["gen"]=generate(m)
    final["final_hist"]=code_hist(qs)
    final["scale"]=scale_stats(qs)
    final["trace"]=trace
    final["seconds"]=time.time()-t0
    final["final_codes_vs_initial"]=hamming_codes(initial_codes,snapshot_codes(qs))

    del opt,m,qs,trainable
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return reference_scales,initial_codes,initial_hist,step300,final

def q9_prepare(teacher):
    m=load_model(torch.float32)
    bf16_roundtrip_(m)
    qs,trainable=attach_quantizers(m,9)
    opt=make_optimizer(trainable)
    trace=[]
    t0=time.time()

    for step0 in range(PREP_STEPS):
        row=one_step(m,teacher,opt,trainable,train_chunks[step0])
        step=step0+1
        if step==1 or step%TRACE_EVERY==0:
            log={"step":step,**row}
            trace.append(log)
            print(json.dumps({"event":"g1_2_q9_prep","step":step,**row}),flush=True)

    out={
        "masters":capture_masters(qs),
        "native_diag":score_on(m,teacher,lrval_chunks),
        "native_hist":code_hist(qs),
        "learned_scale":scale_stats(qs),
        "trace":trace,
        "seconds":time.time()-t0
    }

    del opt,m,qs,trainable
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out

def project_q3(masters,reference_scales,teacher):
    m=load_model(torch.float32)
    bf16_roundtrip_(m)
    qs,trainable=attach_quantizers(m,3)
    load_masters_scales(qs,masters,reference_scales)
    out={
        "diag":score_on(m,teacher,lrval_chunks),
        "hist":code_hist(qs),
        "codes":snapshot_codes(qs),
        "scale":scale_stats(qs)
    }
    del m,qs,trainable
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out

def staged_continuation(masters,reference_scales,initial_codes,teacher):
    m=load_model(torch.float32)
    bf16_roundtrip_(m)
    qs,trainable=attach_quantizers(m,3)
    load_masters_scales(qs,masters,reference_scales)

    start_codes=snapshot_codes(qs)
    start_diag=score_on(m,teacher,lrval_chunks)
    opt=make_optimizer(trainable)  # fresh Adam by preregistration
    trace=[]
    t0=time.time()

    for local0 in range(CONT_STEPS):
        global0=PREP_STEPS+local0
        row=one_step(m,teacher,opt,trainable,train_chunks[global0])
        stage_step=local0+1
        if stage_step==1 or stage_step%TRACE_EVERY==0:
            log={
                "global_step":global0+1,
                "stage_step":stage_step,
                **row
            }
            trace.append(log)
            print(json.dumps({"event":"g1_2_staged_q3","stage_step":stage_step,**log}),flush=True)

    final=score_on(m,teacher,eval_chunks)
    final["gen"]=generate(m)
    final["final_hist"]=code_hist(qs)
    final["scale"]=scale_stats(qs)
    final["trace"]=trace
    final["seconds"]=time.time()-t0
    final["start_codes_vs_initial"]=hamming_codes(initial_codes,start_codes)
    final["final_codes_vs_initial"]=hamming_codes(initial_codes,snapshot_codes(qs))

    del opt,m,qs,trainable
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return start_diag,final,start_codes

print(json.dumps({
    "event":"g1_2_start",
    "model":MODEL_ID,
    "seed":SEED,
    "gpu":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "precision":"FP32 masters + BF16-rounded source + BF16 autocast/BF16 teacher",
    "direct_schedule":{"q3_steps":1200,"lr":LR},
    "staged_schedule":{"q9_steps":300,"q3_steps":900,"lr":LR},
    "transition":"keep FP32 masters; restore original Q3 scales; fresh Adam",
    "order_head":order[:16]
}),flush=True)

teacher=load_model(torch.bfloat16 if DEVICE=="cuda" else torch.float32)
teacher.requires_grad_(False)
teacher.eval()

# Descriptive source baseline only; no treatment selection uses held-out test.
source=load_model(torch.float32)
bf16_roundtrip_(source)
source_test=score_on(source,teacher,eval_chunks)
cleanup(source)

reference_scales,initial_codes,initial_hist,d300,d_final=direct_arm(teacher)
s300=q9_prepare(teacher)

d_fixed=project_q3(d300["masters"],reference_scales,teacher)
s_fixed=project_q3(s300["masters"],reference_scales,teacher)

geometry={
    "d_vs_s_hamming":hamming_codes(d_fixed["codes"],s_fixed["codes"]),
    "d_vs_initial":hamming_codes(initial_codes,d_fixed["codes"]),
    "s_vs_initial":hamming_codes(initial_codes,s_fixed["codes"]),
    "changed_overlap":changed_overlap(
        initial_codes,d_fixed["codes"],s_fixed["codes"]
    ),
    "disagreement_transitions":disagreement_transitions(
        d_fixed["codes"],s_fixed["codes"]
    )
}

print(json.dumps({
    "event":"g1_2_step300_gate",
    "d_fixed_q3":d_fixed["diag"],
    "s_native_q9":s300["native_diag"],
    "s_fixed_q3":s_fixed["diag"],
    "d_vs_s_hamming":geometry["d_vs_s_hamming"]["fraction"],
    "d_vs_initial":geometry["d_vs_initial"]["fraction"],
    "s_vs_initial":geometry["s_vs_initial"]["fraction"]
}),flush=True)

s_start_diag,s_final,s_start_codes=staged_continuation(
    s300["masters"],reference_scales,initial_codes,teacher
)

# Exact audit: S continuation starts from the same fixed-Q3 projection measured
# above. The diagnostics should agree to floating evaluation tolerance.
if abs(s_start_diag["loss"]-s_fixed["diag"]["loss"])>1e-6:
    raise RuntimeError(("S projection/continuation mismatch",
                        s_start_diag["loss"],s_fixed["diag"]["loss"]))
if hamming_codes(s_start_codes,s_fixed["codes"])["fraction"]!=0.0:
    raise RuntimeError("S projection/continuation code mismatch")

final={
    "kind":"g1_2_granite350m_staging_gate",
    "model":MODEL_ID,
    "seed":SEED,
    "precision":"bf16",
    "lr":LR,
    "source_test":source_test,
    "direct":{
        "step300_native":{
            "diag":d300["native_diag"],
            "hist":d300["native_hist"],
            "learned_scale":d300["learned_scale"]
        },
        "step300_fixed_q3":{
            "diag":d_fixed["diag"],
            "hist":d_fixed["hist"]
        },
        "final":d_final
    },
    "staged":{
        "step300_native_q9":{
            "diag":s300["native_diag"],
            "hist":s300["native_hist"],
            "learned_scale":s300["learned_scale"],
            "trace":s300["trace"]
        },
        "step300_fixed_q3":{
            "diag":s_fixed["diag"],
            "hist":s_fixed["hist"]
        },
        "transition":{
            "restore_original_q3_scales":True,
            "fresh_adam":True,
            "lr":LR
        },
        "final":s_final
    },
    "step300_geometry":geometry,
    "final_effect":{
        "loss_gain_d_minus_s":d_final["loss"]-s_final["loss"],
        "ppl_reduction_fraction":1.0-(s_final["ppl"]/d_final["ppl"]),
        "top1_gain":s_final["top1_agreement"]-d_final["top1_agreement"],
        "kl_gain_d_minus_s":d_final["kl_to_teacher"]-s_final["kl_to_teacher"]
    },
    "training_order_head":order[:16]
}

print("FINAL_JSON_BEGIN",flush=True)
print(json.dumps(final,indent=2),flush=True)
print("FINAL_JSON_END",flush=True)
