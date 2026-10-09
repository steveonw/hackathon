# /// script
# dependencies = ["torch>=2.4","transformers>=4.46","datasets>=3.0","accelerate>=1.0"]
# ///
"""
S1-1 — cross-family validation of Granite gridward master-weight interpolation.

Two-arm direct ternary QAT factorial on SmolLM2-360M-Instruct, seed 1729:
D: canonical v9 tuned direct Q3.
P: identical direct Q3 + nine post-optimizer 10% gridward master pulls.

THE LR AND DIRECT TRAINING BASELINE COME FROM SMOL v9, NOT GRANITE.
The source helper functions/data/quantizer/training step are exact pinned v9.
No optimizer or scale reset at step300; direct 1200-step continuous AdamW.
Frozen settings in EXPERIMENT.md before script creation and GPU launch.
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



# === S1-1 preregistered direct-Q3 vs gridward-pull cross-model transfer ===
# Keep the entire canonical v9 tuned-direct source/data/helper code above.
# In particular, DO NOT import Granite's constant LR or step300 Adam reset.

S1_KIND="s1_1_smol360m_direct_q3_gridward_seed1729"
ARMS=["D","P"]
PULL_LAMBDA=0.10
PULL_INTERVAL=100
PULL_LAST_GLOBAL_STEP=900
MONITOR_LAYERS=16
MONITOR_SAMPLES_PER_LAYER=2048
HISTORICAL_V9_DIRECT_1729_TEST_LOSS=5.595722187310457
SELECTED_DIRECT_CFG=next(
    x for x in CANDIDATES if x["name"]=="warm100_cosine_1e-3"
)
assert SELECTED_DIRECT_CFG=={
    "name":"warm100_cosine_1e-3",
    "kind":"warmup_cosine",
    "peak_lr":1e-3,
    "warmup_steps":100,
    "min_lr":1e-4
}
assert FULL_STEPS==1200 and SEED==1729


@torch.no_grad()
def gridward_pull_(qs,lam=PULL_LAMBDA):
    """Pull FP32 master weights toward CURRENT hard ternary reconstruction."""
    assert 0.0<lam<1.0
    for name,mod,q in qs:
        assert q.levels==3,(name,q.levels)
        w=mod.parametrizations.weight.original
        assert w.dtype==torch.float32,(name,w.dtype)
        a=q.alpha().detach().to(device=w.device,dtype=w.dtype)
        z=(w/a).clamp(-0.99,0.99)
        quant=(torch.round(z*1.5)/1.5)*a
        w.lerp_(quant,lam)


@torch.no_grad()
def smoke_test_gridward():
    # Deterministic, CPU only, no consumption of global RNG.
    tmp=nn.Module()
    tmp.register_parameter(
        "weight",
        nn.Parameter(torch.tensor([[-0.8,-0.2,0.0,0.36,0.75]],dtype=torch.float32))
    )
    q=SharedScaleQuant(tmp.weight,levels=3)
    parametrize.register_parametrization(tmp,"weight",q)
    w=tmp.parametrizations.weight.original
    before=w.detach().clone()
    a0=q.alpha().detach().clone()
    code0=q.hard_code(w).clone()
    target0=tmp.weight.detach().clone()
    gridward_pull_([("smoke",tmp,q)])
    after=w.detach().clone()
    assert torch.equal(q.hard_code(w),code0),"pull changed hard ternary codes"
    assert torch.allclose(tmp.weight,target0,atol=1e-7),"pull altered hard Q3 forward"
    assert torch.equal(a0,q.alpha().detach()),"pull altered row scales"
    delta0=(before-target0).abs()
    delta1=(after-target0).abs()
    assert bool((delta1<=delta0+1e-7).all()),"pull increased distance to own code"
    assert bool((delta1<delta0-1e-5).any()),"pull did not move the master"
    return {"codes_preserved":True,"scales_preserved":True,
            "pull_distance_reduced":True}


class SampledCodeMonitor:
    """Identical layer-stratified clean-code monitoring rule used in Granite."""
    @torch.no_grad()
    def __init__(self,qs):
        if len(qs)<MONITOR_LAYERS:
            raise RuntimeError(("too_few_target_layers",len(qs)))
        gen=torch.Generator(device="cpu").manual_seed(SEED+80809)
        keys=sorted(set(round(i*(len(qs)-1)/(MONITOR_LAYERS-1))
                        for i in range(MONITOR_LAYERS)))
        self.samples=[]
        for i in keys:
            _,mod,q=qs[i]
            w=mod.parametrizations.weight.original
            ix_cpu=torch.randint(w.numel(),(MONITOR_SAMPLES_PER_LAYER,),generator=gen)
            ix=ix_cpu.to(device=w.device)
            row=(ix//w.shape[1]).long()
            self.samples.append((mod,q,ix,row))
        self.n_sampled=sum(z[2].numel() for z in self.samples)
        self.prev=[self._code(x) for x in self.samples]
        self.prevprev=[t.clone() for t in self.prev]
        device=self.samples[0][2].device
        self.buffer_flips=torch.zeros((),dtype=torch.int64,device=device)
        self.buffer_reversals=torch.zeros((),dtype=torch.int64,device=device)
        self.total_flips=torch.zeros((),dtype=torch.int64,device=device)
        self.total_reversals=torch.zeros((),dtype=torch.int64,device=device)
        self.steps=0
        self.buffer_steps=0

    @torch.no_grad()
    def _code(self,s):
        mod,q,ix,row=s
        flat=mod.parametrizations.weight.original.reshape(-1)
        alpha=q.alpha().reshape(-1)
        return torch.round((flat[ix]/alpha[row]).clamp(-0.99,0.99)*1.5).to(torch.int8)

    @torch.no_grad()
    def observe(self):
        for i,s in enumerate(self.samples):
            cur=self._code(s)
            prev=self.prev[i]
            before_prev=self.prevprev[i]
            f=(cur!=prev)
            rev=(cur==before_prev)&(prev!=before_prev)
            nf=f.sum()
            nr=rev.sum()
            self.buffer_flips+=nf
            self.buffer_reversals+=nr
            self.total_flips+=nf
            self.total_reversals+=nr
            self.prevprev[i]=prev
            self.prev[i]=cur
        self.steps+=1
        self.buffer_steps+=1

    @torch.no_grad()
    def interval(self):
        flips=int(self.buffer_flips.item())
        reversals=int(self.buffer_reversals.item())
        denom=max(1,self.n_sampled*self.buffer_steps)
        out={
            "sampled_weights":self.n_sampled,
            "steps":self.buffer_steps,
            "flips":flips,
            "reversals":reversals,
            "flip_rate_per_weight_update":flips/denom,
            "immediate_reversals_per_flip":reversals/max(1,flips)
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
            "flip_rate_per_weight_update":flips/max(1,self.n_sampled*self.steps),
            "immediate_reversals_per_flip":reversals/max(1,flips)
        }


@torch.no_grad()
def code_movement_from_start(qs,initial):
    changed=total=0
    per_layer=[]
    for name,mod,q in qs:
        current=q.hard_code(mod.parametrizations.weight.original).cpu()
        initial_code=initial[name]
        n=current.numel()
        ch=int((current!=initial_code).sum().item())
        changed+=ch
        total+=n
        per_layer.append((ch/max(1,n),name,ch,n))
    per_layer.sort(reverse=True)
    return {
        "changed":changed,
        "total":total,
        "fraction":changed/max(1,total),
        "top_layers":[{"name":name,"changed":count,"total":size,"fraction":rate}
                      for rate,name,count,size in per_layer[:12]]
    }


def run_transfer_arm(arm,teacher):
    """Parent v9 direct_run training logic with a gridward-only branch."""
    assert arm in ARMS
    reset_rng()
    m=load_model(torch.float32)
    bf16_roundtrip_(m)
    qs,trainable=attach_quantizers(m,3)
    assert all(q.levels==3 and
               mod.parametrizations.weight.original.dtype==torch.float32
               for _,mod,q in qs)
    stage_start=snapshot_codes(qs)
    prev={k:v.clone() for k,v in stage_start.items()}
    opt=make_optimizer(trainable,float(SELECTED_DIRECT_CFG["peak_lr"]))
    scaler=torch.amp.GradScaler("cuda",enabled=(DEVICE=="cuda"))
    monitor=SampledCodeMonitor(qs)
    trace=[]
    pulls=0
    skipped=0
    start=time.time()

    for step in range(FULL_STEPS):
        global_step=step+1
        lr=lr_for(SELECTED_DIRECT_CFG,step)
        set_optimizer_lr(opt,lr)
        amp_before=scaler.get_scale()
        # Same parent v9 one_step, teacher, autocast, clipping and optimizer.
        loss,ce,kl=one_step(m,teacher,opt,scaler,train_chunks[step])
        amp_after=scaler.get_scale()
        did_skip=(amp_after<amp_before)
        skipped+=int(did_skip)
        if not all(math.isfinite(x) for x in (loss,ce,kl)):
            raise RuntimeError({
                "reason":"nonfinite_training_step","arm":arm,
                "step":global_step,"loss":loss,"ce":ce,"kl":kl
            })

        scheduled_pull=(arm=="P" and
                        global_step<=PULL_LAST_GLOBAL_STEP and
                        global_step%PULL_INTERVAL==0)
        if scheduled_pull:
            if did_skip:
                raise RuntimeError({
                    "reason":"scheduled_pull_without_optimizer_step",
                    "arm":arm,"step":global_step
                })
            gridward_pull_(qs,PULL_LAMBDA)
            pulls+=1

        monitor.observe()
        if step==0 or global_step%100==0 or global_step==FULL_STEPS:
            # Preserve parent v9 eval/flip_stats ordering and fixed checkpoints.
            prev,fs=flip_stats(qs,prev,stage_start)
            val=(score_on(m,teacher,lrval_chunks)
                 if global_step in (300,600,900,1200) else None)
            row={
                "step":global_step,
                "lr":lr,
                "loss":loss,"ce":ce,"kl":kl,
                "validation":val,
                "gridward_pull":scheduled_pull,
                "amp_skipped_step":did_skip,
                "amp_scale_after":amp_after,
                "sampled_transitions":monitor.interval(),
                **fs
            }
            trace.append(row)
            print(json.dumps({
                "event":"s1_1_train","arm":arm,**row
            }),flush=True)

    # Historical v9 endpoint order: validation then test then generation
    val=score_on(m,teacher,lrval_chunks)
    final=score_on(m,teacher,eval_chunks)
    final["gen"]=generate(m)
    final["final_hist"]=code_hist(qs)
    final["scale"]=scale_stats(qs)
    movement=code_movement_from_start(qs,stage_start)
    monitor_totals=monitor.aggregate()
    out={
        "arm":arm,
        "steps_completed":FULL_STEPS,
        "amp_skipped_step_opportunities":skipped,
        "effective_optimizer_updates":FULL_STEPS-skipped,
        "gridward_pulls":pulls,
        "initial_target_weights":movement["total"],
        "final":final,
        "validation":val,
        "code_movement_from_initial":movement,
        "sampled_transitions":monitor_totals,
        "trace":trace,
        "seconds":time.time()-start
    }
    print(json.dumps({
        "event":"s1_1_arm_final","arm":arm,
        "final":{k:v for k,v in final.items()
                 if k in ("loss","ppl","top1_agreement","kl_to_teacher","tokens")},
        "steps":FULL_STEPS,"pulls":pulls,"skipped":skipped,
        "code_movement":movement["fraction"],
        "sampled_transition_rate":monitor_totals["flip_rate_per_weight_update"]
    }),flush=True)
    del opt,scaler,m,qs,trainable,prev,stage_start,monitor
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return out


SMOKE=smoke_test_gridward()
print(json.dumps({
    "event":"s1_1_start",
    "kind":S1_KIND,"seed":SEED,"model":MODEL_ID,
    "device":DEVICE,
    "gpu":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "arms":ARMS,
    "source":"pinned Smol v9 tuned direct Q3 helper and forward/optimizer",
    "historical_direct_v9_1729_loss":HISTORICAL_V9_DIRECT_1729_TEST_LOSS,
    "schedule":SELECTED_DIRECT_CFG,
    "lr_step_1":lr_for(SELECTED_DIRECT_CFG,0),
    "lr_step_100":lr_for(SELECTED_DIRECT_CFG,99),
    "lr_step_300":lr_for(SELECTED_DIRECT_CFG,299),
    "lr_step_301":lr_for(SELECTED_DIRECT_CFG,300),
    "lr_step_1200":lr_for(SELECTED_DIRECT_CFG,1199),
    "optimizer_reset_at_300":False,
    "pull_steps":list(range(100,901,100)),
    "pull_lambda":PULL_LAMBDA,
    "amp_compute":"Smol v9 FP16, GradScaler and FP32 masters",
    "training_order_head":_order[:16],
    "smoke":SMOKE
}),flush=True)

teacher=load_model(torch.float16 if DEVICE=="cuda" else torch.float32)
teacher.requires_grad_(False)

base=load_model(torch.float32)
bf16_roundtrip_(base)
bf16_test=score_on(base,teacher,eval_chunks)
del base
gc.collect()
if torch.cuda.is_available(): torch.cuda.empty_cache()

results={}
for arm in ARMS:
    results[arm]=run_transfer_arm(arm,teacher)

d=results["D"]; p=results["P"]
checks={
    "expected_arms":set(results)==set(ARMS),
    "full_step_opportunities":all(
        v["steps_completed"]==FULL_STEPS for v in results.values()
    ),
    "pull_count":d["gridward_pulls"]==0 and p["gridward_pulls"]==9,
    "same_total_target_weights":(
        d["initial_target_weights"]==p["initial_target_weights"] and
        d["initial_target_weights"]>0
    ),
    "finite_test_loss":all(math.isfinite(v["final"]["loss"]) for v in results.values()),
    "heldout_test_tokens":all(v["final"]["tokens"]==8192 for v in results.values()),
    "ternary_grid_smoke":all(SMOKE.values()),
    "schedule_endpoints":(
        abs(lr_for(SELECTED_DIRECT_CFG,99)-1e-3)<1e-12 and
        abs(lr_for(SELECTED_DIRECT_CFG,1199)-1e-4)<1e-12
    ),
    "complete_transition_monitor":all(
        v["sampled_transitions"]["steps"]==FULL_STEPS for v in results.values()
    )
}
assert all(checks.values()),checks
effect={
    "D_loss":d["final"]["loss"],
    "P_loss":p["final"]["loss"],
    "P_gain_vs_D_nats":d["final"]["loss"]-p["final"]["loss"],
    "D_ppl":d["final"]["ppl"],
    "P_ppl":p["final"]["ppl"],
    "P_ppl_relative_reduction":(
        (d["final"]["ppl"]-p["final"]["ppl"])/d["final"]["ppl"]
    ),
    "P_minus_D_validation_loss":p["validation"]["loss"]-d["validation"]["loss"],
    "P_minus_D_sampled_flip_rate":(
        p["sampled_transitions"]["flip_rate_per_weight_update"]
        -d["sampled_transitions"]["flip_rate_per_weight_update"]
    ),
    "historical_D_loss":HISTORICAL_V9_DIRECT_1729_TEST_LOSS,
    "within_run_D_minus_historical_D":(
        d["final"]["loss"]-HISTORICAL_V9_DIRECT_1729_TEST_LOSS
    )
}
final={
    "kind":S1_KIND,
    "seed":SEED,"model":MODEL_ID,
    "precision":"BF16-rounded FP32 masters; Smol v9 FP16 autocast+GradScaler",
    "source_parent_v9_pin":"5c088e58539b2dede93df57ac3f72dbe0480a028",
    "schedule":SELECTED_DIRECT_CFG,
    "steps":FULL_STEPS,
    "optimizer_reset_at_300":False,
    "pull":{
        "lambda":PULL_LAMBDA,"every_n_steps":PULL_INTERVAL,
        "last_global_step":PULL_LAST_GLOBAL_STEP,
        "post_optimizer":True,
        "scales_unchanged_directly":True
    },
    "bf16_source_test":bf16_test,
    "training_order_head":_order[:16],
    "arms":results,"checks":checks,"effects":effect,
    "smoke":SMOKE,
    "no_gaussian_and_no_q9":True,
    "heldout_used_for_selection":False
}
print(json.dumps({
    "event":"s1_1_all_final","effects":effect,"checks":checks
}),flush=True)
print("FINAL_JSON_BEGIN",flush=True)
print(json.dumps(final),flush=True)
print("FINAL_JSON_END",flush=True)
