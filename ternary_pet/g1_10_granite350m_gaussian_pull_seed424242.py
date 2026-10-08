# /// script
# dependencies = ["torch>=2.4","transformers>=4.57.1","datasets>=3.0","accelerate>=1.0"]
# ///
"""
G1-9: Granite 350M direct ternary QAT: Gaussian noise x gridward pull.

Pre-registered four-arm 2x2 WinQ-inspired design on development seed 271828:
D = clean STE; G = Gaussian perturbation of FP32 master prior to hard
quantization during training; P = gridward convex interpolation after selected
optimizer steps; GP = both. All arms always use three-state Q3, retain FP32
masters, BF16 student/teacher and constant LR 1e-4. At global step300 original
Q3 row scales are restored and a fresh Adam is used in all arms.

Validation and heldout inference always use deterministic hard Q3(W), never
the perturbed forward. Noise and pull hyperparameters are frozen in
EXPERIMENT.md before this script was created or any GPU job launched.

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
SEED=424242
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
NOISE_SIGMA_U=0.04
NOISE_PLATEAU_STEPS=900
NOISE_DECAY_STEPS=300
PULL_LAMBDA=0.10
PULL_INTERVAL=100
PULL_LAST_GLOBAL_STEP=900
ARMS=["D","G","P","GP"]
MONITOR_LAYERS=16
MONITOR_SAMPLES_PER_LAYER=2048

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
        self.noise_sigma_u=0.0

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
        if self.training and self.noise_sigma_u>0.0:
            # WinQ-inspired: hard Q3(W + alpha*Gaussian), same STE d/dW.
            with torch.no_grad():
                noisy_z=(z.detach()+self.noise_sigma_u*torch.randn_like(z)).clamp(-0.99,0.99)
                hard=torch.round(noisy_z*n)/n
        else:
            hard=torch.round(z*n)/n
        # Forward equals discrete hard code; gradient remains latent STE.
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


def reset_rng():
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

@torch.no_grad()
def initial_q3_reference(teacher):
    reset_rng()
    m=load_model(torch.float32)
    bf16_roundtrip_(m)
    qs,trainable=attach_quantizers(m,3)
    scales=capture_scale_state(qs)
    codes=snapshot_codes(qs)
    out={
        "diag":score_on(m,teacher,lrval_chunks),
        "hist":code_hist(qs),
        "scale":scale_stats(qs)
    }
    cleanup(m,qs,trainable)
    return scales,codes,out


# === G1-9 factorial harness: add to unchanged Granite BF16 CE/KL/Q3 utilities ===

def noise_sigma_u(global_step):
    """Global step is one-indexed; normalized per-row Gaussian standard deviation."""
    assert 1 <= global_step <= TRAIN_CHUNKS
    if global_step<=NOISE_PLATEAU_STEPS:
        return NOISE_SIGMA_U
    remaining=TRAIN_CHUNKS-global_step
    return NOISE_SIGMA_U*max(0.0,remaining)/NOISE_DECAY_STEPS


def configure_noise(qs,arm,global_step):
    sigma=noise_sigma_u(global_step) if arm in ("G","GP") else 0.0
    for _,_,q in qs:
        q.noise_sigma_u=sigma
    return sigma


@torch.no_grad()
def gridward_pull_(qs,lam):
    """Convex move towards CURRENT hard-Q3 point; no code/Adam reset."""
    assert 0.0 < lam < 1.0
    for _,mod,q in qs:
        w=mod.parametrizations.weight.original
        alpha=q.alpha().detach().to(w.device,dtype=w.dtype)
        z=(w/alpha).clamp(-0.99,0.99)
        quant=(torch.round(z*1.5)/1.5)*alpha
        # Nearest-code convex interpolation stays in same quantization cell.
        w.lerp_(quant,lam)


@torch.no_grad()
def ref_projected_codes(masters,reference_scales):
    return {
        name:torch.round(
            (w.float()/reference_scales[name]["alpha"].float()).clamp(-0.99,0.99)*1.5
        ).to(torch.int8)
        for name,w in masters.items()
    }


@torch.no_grad()
def assert_q3_grid(qs,arm,stage):
    """Small representative functional check; full tensor quantizer is unchanged."""
    for name,mod,q in qs[:3]:
        assert q.levels==3,(arm,stage,name,"not ternary")
        w=mod.parametrizations.weight.original
        assert w.dtype==torch.float32,(arm,stage,name,"master not fp32")
        assert torch.isfinite(q.alpha()).all(),(arm,stage,name,"bad scales")


class SampledCodeMonitor:
    """
    Deterministic layer-stratified sample, independent of torch global RNG.
    Updates are noise-free hard codes of actual masters, never noisy training
    forward samples. Tracks per-step flips and immediate A->B->A reversals.
    """
    @torch.no_grad()
    def __init__(self,qs):
        self.samples=[]
        generator=torch.Generator(device="cpu").manual_seed(SEED+80809)
        keys=sorted(set(round(i*(len(qs)-1)/(MONITOR_LAYERS-1))
                        for i in range(MONITOR_LAYERS)))
        for li in keys:
            _,mod,q=qs[li]
            w=mod.parametrizations.weight.original
            idx_cpu=torch.randint(w.numel(),(MONITOR_SAMPLES_PER_LAYER,),generator=generator)
            idx=idx_cpu.to(device=w.device)
            row=(idx//w.shape[1]).long()
            self.samples.append((mod,q,idx,row))
        self.n_sampled=sum(s[2].numel() for s in self.samples)
        self.prev=[self._code(s) for s in self.samples]
        self.prevprev=[x.clone() for x in self.prev]
        device=self.samples[0][2].device
        self.buffer_flips=torch.zeros((),device=device,dtype=torch.int64)
        self.buffer_reversals=torch.zeros((),device=device,dtype=torch.int64)
        self.total_flips=torch.zeros((),device=device,dtype=torch.int64)
        self.total_reversals=torch.zeros((),device=device,dtype=torch.int64)
        self.buffer_steps=0
        self.steps=0

    @torch.no_grad()
    def _code(self,s):
        mod,q,idx,row=s
        w=mod.parametrizations.weight.original.reshape(-1)
        a=q.alpha().reshape(-1)
        return torch.round((w[idx]/a[row]).clamp(-0.99,0.99)*1.5).to(torch.int8)

    @torch.no_grad()
    def observe(self):
        for i,s in enumerate(self.samples):
            current=self._code(s)
            before=self.prev[i]
            before2=self.prevprev[i]
            flipped=(current!=before)
            reversed_immediately=(current==before2)&(before!=before2)
            new_flips=flipped.sum()
            new_reversals=reversed_immediately.sum()
            self.buffer_flips+=new_flips
            self.buffer_reversals+=new_reversals
            self.total_flips+=new_flips
            self.total_reversals+=new_reversals
            self.prevprev[i]=before
            self.prev[i]=current
        self.buffer_steps+=1
        self.steps+=1

    @torch.no_grad()
    def interval(self):
        flips=int(self.buffer_flips.item())
        reversals=int(self.buffer_reversals.item())
        seen=self.buffer_steps*self.n_sampled
        out={
            "sampled_weights":self.n_sampled,
            "sampled_updates":self.buffer_steps,
            "flip_rate_per_sampled_update":flips/max(1,seen),
            "immediate_reversals_per_flip":reversals/max(1,flips),
            "flips":flips,
            "immediate_reversals":reversals
        }
        self.buffer_flips.zero_()
        self.buffer_reversals.zero_()
        self.buffer_steps=0
        return out

    @torch.no_grad()
    def aggregate(self):
        flips=int(self.total_flips.item())
        reversals=int(self.total_reversals.item())
        return {
            "sampled_weights":self.n_sampled,
            "steps":self.steps,
            "flips":flips,
            "immediate_reversals":reversals,
            "flip_rate_per_sampled_update":flips/max(1,self.n_sampled*self.steps),
            "immediate_reversals_per_flip":reversals/max(1,flips)
        }


def take_training_step(arm,m,qs,teacher,opt,trainable,monitor,global_step):
    sigma=configure_noise(qs,arm,global_step)
    row=one_step(m,teacher,opt,trainable,train_chunks[global_step-1])
    pulled=(arm in ("P","GP") and
            global_step<=PULL_LAST_GLOBAL_STEP and
            global_step%PULL_INTERVAL==0)
    if pulled:
        gridward_pull_(qs,PULL_LAMBDA)
    monitor.observe()
    return {"global_step":global_step,"noise_sigma_u":sigma,
            "gridward_pull":pulled,**row}


def training_segment(arm,m,qs,teacher,opt,trainable,monitor,start,end,label):
    trace=[]
    pulls=0
    for global_step in range(start,end+1):
        row=take_training_step(
            arm,m,qs,teacher,opt,trainable,monitor,global_step
        )
        pulls+=int(row["gridward_pull"])
        if global_step==start or global_step%TRACE_EVERY==0 or global_step==end:
            row["sampled_transitions"]=monitor.interval()
            trace.append(row)
            print(json.dumps({
                "event":"g1_9_train",
                "arm":arm,"phase":label,**row
            }),flush=True)
    return {"trace":trace,"pulls":pulls,"sampled_transitions":monitor.aggregate()}


def run_arm(arm,teacher,reference_scales,initial_codes):
    assert arm in ARMS
    t0=time.time()
    reset_rng()
    m=load_model(torch.float32)
    bf16_roundtrip_(m)
    qs,trainable=attach_quantizers(m,3)
    assert_q3_grid(qs,arm,"initial")
    initial_arm_codes=snapshot_codes(qs)
    initial_check=hamming_codes(initial_codes,initial_arm_codes)
    assert initial_check["fraction"]==0.0,(arm,"source mismatch",initial_check)
    del initial_arm_codes,initial_check
    opt=make_optimizer(trainable)
    monitor=SampledCodeMonitor(qs)

    prep=training_segment(
        arm,m,qs,teacher,opt,trainable,monitor,
        1,PREP_STEPS,"prepare"
    )
    prep_native=score_on(m,teacher,lrval_chunks)
    prep_scales=scale_stats(qs)
    masters=capture_masters(qs)
    prep_projected=ref_projected_codes(masters,reference_scales)
    prep_move=hamming_codes(initial_codes,prep_projected)
    del prep_projected
    prep["native_validation"]=prep_native
    prep["learned_scales"]=prep_scales
    prep["projected_move_vs_initial"]=prep_move
    print(json.dumps({
        "event":"g1_9_prep_finished","arm":arm,
        "native_validation":prep_native,
        "projected_move_vs_initial":prep_move["fraction"]
    }),flush=True)
    del monitor,opt,qs,trainable,m
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Exactly the historical Granite G1 mechanism D branch reset:
    # restore original Q3 scales, preserve FP32 masters, fresh optimizer.
    reset_rng()
    m=load_model(torch.float32)
    bf16_roundtrip_(m)
    qs,trainable=attach_quantizers(m,3)
    load_masters_scales(qs,masters,reference_scales)
    assert_q3_grid(qs,arm,"after_reset")
    del masters
    stage_start_codes=snapshot_codes(qs)
    stage_diag=score_on(m,teacher,lrval_chunks)
    print(json.dumps({
        "event":"g1_9_continuation_start","arm":arm,
        "deterministic_val_after_scale_reset":stage_diag
    }),flush=True)
    opt=make_optimizer(trainable)
    monitor=SampledCodeMonitor(qs)
    cont=training_segment(
        arm,m,qs,teacher,opt,trainable,monitor,
        PREP_STEPS+1,TRAIN_CHUNKS,"continue"
    )
    final=score_on(m,teacher,eval_chunks)
    final_codes=snapshot_codes(qs)
    move=hamming_codes(initial_codes,final_codes)
    late=hamming_codes(stage_start_codes,final_codes)
    out={
        "arm":arm,"final":final,"prep":prep,
        "pre_continuation_validation":stage_diag,
        "continuation":cont,
        "final_movement_vs_initial":move,
        "continuation_start_vs_final":late,
        "seconds":time.time()-t0
    }
    print(json.dumps({
        "event":"g1_9_arm_final","arm":arm,"final":final,
        "final_code_movement":move["fraction"],
        "prep_pulls":prep["pulls"],"continuation_pulls":cont["pulls"],
        "prep_flip_rate":prep["sampled_transitions"]["flip_rate_per_sampled_update"],
        "continuation_flip_rate":cont["sampled_transitions"]["flip_rate_per_sampled_update"]
    }),flush=True)
    del final_codes,stage_start_codes,monitor,opt,qs,trainable,m
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


@torch.no_grad()
def self_test_quantizer():
    """CPU-only smoke checks before loading Granite or using heldout data."""
    w=torch.tensor([[0.0,0.1,-0.1,0.9,-0.9]],dtype=torch.float32)
    q=SharedScaleQuant(w,3)
    q.eval()
    clean=q(w)
    alpha=q.alpha()
    grid=(torch.round((w/alpha).clamp(-0.99,0.99)*1.5)/1.5)*alpha
    assert torch.allclose(clean,grid,atol=1e-7),("Q3 output differs",clean,grid)
    q.train()
    q.noise_sigma_u=0.0
    assert torch.allclose(q(w),clean,atol=1e-7),"Zero-noise train altered Q3"
    q.noise_sigma_u=0.04
    noisy=q(w)
    assert torch.isfinite(noisy).all(),"Nonfinite noisy quantizer"
    q.eval()
    assert torch.allclose(q(w),clean,atol=1e-7),"Eval uses noise"
    assert noise_sigma_u(1)==NOISE_SIGMA_U
    assert noise_sigma_u(900)==NOISE_SIGMA_U
    assert noise_sigma_u(1200)==0.0
    assert noise_sigma_u(901)<NOISE_SIGMA_U


self_test_quantizer()
print(json.dumps({
    "event":"g1_9_start","kind":"g1_9_granite350m_gaussian_pull_factorial",
    "model":MODEL_ID,"seed":SEED,
    "device":DEVICE,
    "gpu":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "arms":ARMS,"lr":LR,"train_steps":TRAIN_CHUNKS,
    "prep_steps":PREP_STEPS,"continuation_steps":CONT_STEPS,
    "noise_sigma_u":NOISE_SIGMA_U,
    "noise_plateau_through":NOISE_PLATEAU_STEPS,
    "noise_anneal_end":TRAIN_CHUNKS,
    "gridward_lambda":PULL_LAMBDA,
    "gridward_interval":PULL_INTERVAL,
    "gridward_last_step":PULL_LAST_GLOBAL_STEP,
    "monitor_layers":MONITOR_LAYERS,
    "monitor_samples_per_layer":MONITOR_SAMPLES_PER_LAYER,
    "precision":"FP32 masters, BF16-rounded source, BF16 compute, BF16 teacher",
    "optimizer_reset_at_300":True,
    "order_head":order[:16],
    "heldout_for_selection":False
}),flush=True)

teacher=load_model(torch.bfloat16 if DEVICE=="cuda" else torch.float32)
teacher.requires_grad_(False)
teacher.eval()
reference_scales,initial_codes,initial_ref=initial_q3_reference(teacher)
assert initial_ref["hist"]["total"]==249561088,initial_ref["hist"]["total"]
print(json.dumps({
    "event":"g1_9_source","initial_reference_validation":initial_ref["diag"],
    "initial_hist":initial_ref["hist"],
    "initial_scales":initial_ref["scale"]
}),flush=True)

results={}
for arm in ARMS:
    results[arm]=run_arm(arm,teacher,reference_scales,initial_codes)

effects={
    "D_loss":results["D"]["final"]["loss"],
    "G_loss":results["G"]["final"]["loss"],
    "P_loss":results["P"]["final"]["loss"],
    "GP_loss":results["GP"]["final"]["loss"],
    "noise_gain_D_minus_G":results["D"]["final"]["loss"]-results["G"]["final"]["loss"],
    "pull_gain_D_minus_P":results["D"]["final"]["loss"]-results["P"]["final"]["loss"],
    "combined_gain_D_minus_GP":results["D"]["final"]["loss"]-results["GP"]["final"]["loss"],
    "factorial_interaction":(
        results["G"]["final"]["loss"]+results["P"]["final"]["loss"]
        -results["D"]["final"]["loss"]-results["GP"]["final"]["loss"]
    ),
    "historic_D_loss_271828":5.726725168526173,
    "D_minus_historic_D_loss":results["D"]["final"]["loss"]-5.726725168526173
}

checks={
    "all_four_arms_present":set(results)==set(ARMS),
    "all_arms_have_300_plus_900_updates":all(
        a["prep"]["trace"][-1]["global_step"]==300 and
        a["continuation"]["trace"][-1]["global_step"]==1200 and
        a["prep"]["sampled_transitions"]["steps"]==300 and
        a["continuation"]["sampled_transitions"]["steps"]==900
        for a in results.values()
    ),
    "pulled_arms_have_nine_gridward_updates":all(
        (results[a]["prep"]["pulls"]+results[a]["continuation"]["pulls"])==
        (9 if a in ("P","GP") else 0) for a in ARMS
    ),
    "all_final_losses_finite":all(
        math.isfinite(a["final"]["loss"]) for a in results.values()
    ),
    "initial_target_weights":initial_ref["hist"]["total"]==249561088
}
assert all(checks.values()),checks
final={
    "kind":"g1_9_granite350m_gaussian_pull_factorial",
    "model":MODEL_ID,"seed":SEED,
    "precision":"BF16 compute + FP32 masters",
    "lr":LR,"train_steps":TRAIN_CHUNKS,
    "prep_steps":PREP_STEPS,"continuation_steps":CONT_STEPS,
    "noise":{
        "type":"per_row_scale_normal_gaussian_in_hard_quantizer",
        "normalized_sigma_u":NOISE_SIGMA_U,
        "plateau_through":NOISE_PLATEAU_STEPS,
        "anneal_to_zero_at":TRAIN_CHUNKS,
        "deterministic_validation_and_inference":True
    },
    "pull":{
        "type":"master_gridward_interpolation",
        "lambda":PULL_LAMBDA,
        "every_steps":PULL_INTERVAL,
        "last_step":PULL_LAST_GLOBAL_STEP,
        "resets_optimizer_state":False
    },
    "arms":results,"effects":effects,"checks":checks,
    "initial_reference":initial_ref,
    "training_order_head":order[:16]
}
print(json.dumps({"event":"g1_9_all_final","effects":effects,"checks":checks}),flush=True)
print("FINAL_JSON_BEGIN",flush=True)
print(json.dumps(final),flush=True)
print("FINAL_JSON_END",flush=True)
