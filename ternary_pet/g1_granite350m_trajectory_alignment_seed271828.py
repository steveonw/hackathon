# /// script
# dependencies = ["torch>=2.4","transformers>=4.57.1","datasets>=3.0","accelerate>=1.0"]
# ///
"""
G1-3: Granite 4.0 350M compressed causal mechanism test.

Scientific question:
Does Granite reproduce the SmolLM2 causal mechanism: a small D-vs-S projected
Q3 disagreement set carrying the staging benefit, with useful interior
placement at d=0.5 and a matched-random specificity control?

Preregistered seed/order 1729 arms:
  D          direct-prepared masters everywhere
  S          Q9-prepared masters everywhere
  M-exact    S masters only on the true D-vs-S Q3 disagreement mask
  M-d50      S-selected Q3 code on true M at normalized depth d=0.5
  Random-d50 per-layer/source->target matched random positions outside M

Every arm receives original Q3 scales, fresh Adam, constant LR 1e-4, and the
same continuation chunks 301-1200. No held-out result is used to construct M
or the random control.
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
SEED=271828
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
SELECTOR_WINDOW=24

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

def prepare_arm(label,levels,teacher):
    reset_rng()
    m=load_model(torch.float32)
    bf16_roundtrip_(m)
    qs,trainable=attach_quantizers(m,levels)
    opt=make_optimizer(trainable)
    trace=[]
    t0=time.time()

    for step0 in range(PREP_STEPS):
        row=one_step(m,teacher,opt,trainable,train_chunks[step0])
        step=step0+1
        if step==1 or step%TRACE_EVERY==0 or step==PREP_STEPS:
            log={"step":step,**row}
            trace.append(log)
            print(json.dumps({"event":"g1_3_prep","label":label,**log}),flush=True)

    masters=capture_masters(qs)
    out={
        "label":label,
        "levels":levels,
        "native_diag":score_on(m,teacher,lrval_chunks),
        "native_hist":code_hist(qs),
        "learned_scale":scale_stats(qs),
        "trace":trace,
        "seconds":time.time()-t0
    }
    cleanup(opt,m,qs,trainable)
    return masters,out

@torch.no_grad()
def projected_q3_codes_from_masters(masters,reference_scales):
    out={}
    for name,w in masters.items():
        a=reference_scales[name]["alpha"].float()
        z=(w.float()/a).clamp(-0.99,0.99)
        out[name]=torch.round(z*1.5).to(torch.int8)
    return out

@torch.no_grad()
def build_diff_mask(d_codes,s_codes):
    mask={}
    changed=0
    total=0
    layer_rows=[]
    for name in d_codes:
        m=(d_codes[name]!=s_codes[name])
        ch=int(m.sum().item())
        n=m.numel()
        mask[name]=m
        changed+=ch
        total+=n
        layer_rows.append((ch/max(1,n),name,ch,n))
    layer_rows.sort(reverse=True)
    return mask,{
        "fraction":changed/max(1,total),
        "changed_count":changed,
        "total":total,
        "top_layers":[
            {"name":name,"fraction":fr,"changed_count":ch,"total":n}
            for fr,name,ch,n in layer_rows[:20]
        ]
    }

@torch.no_grad()
def transition_counts_for_mask(src_codes,tgt_codes,mask):
    global_counts=defaultdict(int)
    per_layer={}
    for name in src_codes:
        src=src_codes[name].reshape(-1)
        tgt=tgt_codes[name].reshape(-1)
        m=mask[name].reshape(-1)
        layer=defaultdict(int)
        for a in (-1,0,1):
            for b in (-1,0,1):
                if a==b:
                    continue
                n=int((m & (src==a) & (tgt==b)).sum().item())
                if n:
                    key=f"{a}->{b}"
                    layer[key]+=n
                    global_counts[key]+=n
        per_layer[name]=dict(sorted(layer.items()))
    return {
        "global":dict(sorted(global_counts.items())),
        "per_layer":per_layer
    }

@torch.no_grad()
def build_matched_random_plan(d_codes,s_codes,diff_mask):
    plan={}
    per_layer={}
    total_selected=0
    for li,name in enumerate(d_codes):
        d=d_codes[name].reshape(-1)
        s=s_codes[name].reshape(-1)
        m=diff_mask[name].reshape(-1)
        idx_parts=[]
        tgt_parts=[]
        audit=defaultdict(int)

        for cd in (-1,0,1):
            needs=[]
            need_total=0
            for cs in (-1,0,1):
                if cs==cd:
                    continue
                n=int((m & (d==cd) & (s==cs)).sum().item())
                if n:
                    needs.append((cs,n))
                    need_total+=n
            if need_total==0:
                continue

            candidates=((~m) & (d==cd)).nonzero(as_tuple=False).flatten()
            if candidates.numel()<need_total:
                raise RuntimeError(
                    ("insufficient_random_candidates",name,cd,
                     int(candidates.numel()),need_total)
                )

            g=torch.Generator().manual_seed(
                SEED*1000003 + 31003 + li*97 + (cd+1)*17
            )
            chosen=candidates[
                torch.randperm(candidates.numel(),generator=g)[:need_total]
            ]
            offset=0
            for cs,n in sorted(needs):
                part=chosen[offset:offset+n]
                idx_parts.append(part.to(torch.int32))
                tgt_parts.append(torch.full((n,),cs,dtype=torch.int8))
                audit[f"{cd}->{cs}"]+=n
                offset+=n
            if offset!=need_total:
                raise RuntimeError(("random_plan_offset",name,offset,need_total))

        if idx_parts:
            idx=torch.cat(idx_parts)
            target=torch.cat(tgt_parts)
        else:
            idx=torch.empty(0,dtype=torch.int32)
            target=torch.empty(0,dtype=torch.int8)

        if idx.numel()!=torch.unique(idx).numel():
            raise RuntimeError(("duplicate_random_positions",name))
        if idx.numel() and bool(m[idx.long()].any().item()):
            raise RuntimeError(("random_overlaps_true_mask",name))

        plan[name]={"idx":idx,"target":target}
        per_layer[name]=dict(sorted(audit.items()))
        total_selected+=int(idx.numel())

    return plan,{
        "total_selected":total_selected,
        "per_layer_transition_counts":per_layer
    }

@torch.no_grad()
def random_expected_codes(d_codes,random_plan):
    out={name:c.clone() for name,c in d_codes.items()}
    for name,c in out.items():
        rp=random_plan[name]
        idx=rp["idx"].long()
        if idx.numel():
            c.reshape(-1)[idx]=rp["target"]
    return out

@torch.no_grad()
def depth_value(alpha,source_code,target_code,depth=0.5):
    d=float(depth)
    tc=target_code.to(torch.int8)
    sc=source_code.to(torch.int8)
    z=torch.empty_like(alpha,dtype=torch.float32)

    nz=(tc!=0)
    z[nz]=tc[nz].to(torch.float32)*((1.0/3.0)+d*(1.0/3.0))

    to0=(tc==0)
    bad=to0 & (sc==0)
    if bool(bad.any().item()):
        raise RuntimeError("d50 target-zero received source-zero on disagreement position")
    z[to0]=sc[to0].to(torch.float32)*(1.0/3.0)*(1.0-d)
    return alpha.float()*z

@torch.no_grad()
def set_reference_scales(qs,reference_scales):
    for name,mod,q in qs:
        q.raw_alpha.copy_(
            reference_scales[name]["raw_alpha"].to(
                device=q.raw_alpha.device,dtype=q.raw_alpha.dtype
            )
        )
        q.levels=3

@torch.no_grad()
def load_g1_3_arm(
    qs,arm,d_masters,s_masters,d_codes,s_codes,diff_mask,
    reference_scales,random_plan
):
    for name,mod,q in qs:
        base=d_masters[name].clone()
        a=reference_scales[name]["alpha"].float().expand_as(base)
        dc=d_codes[name]
        sc=s_codes[name]
        m=diff_mask[name]

        if arm=="D":
            h=base
        elif arm=="S":
            h=s_masters[name].clone()
        elif arm=="M-exact":
            h=torch.where(m,s_masters[name],base)
        elif arm=="M-d50":
            h=base
            h[m]=depth_value(a[m],dc[m],sc[m],0.5)
        elif arm=="Random-d50":
            h=base
            rp=random_plan[name]
            idx=rp["idx"].long()
            if idx.numel():
                flat=h.reshape(-1)
                af=a.reshape(-1)
                dflat=dc.reshape(-1)
                flat[idx]=depth_value(
                    af[idx],dflat[idx],rp["target"],0.5
                )
        else:
            raise ValueError(arm)

        w=mod.parametrizations.weight.original
        w.copy_(h.to(device=w.device,dtype=w.dtype))

@torch.no_grad()
def q9_code_survival(qs,diff_mask,s_codes,reference_scales):
    groups={
        "all":{"n":0,"current":0,"fixed_alpha0":0},
        "target_zero":{"n":0,"current":0,"fixed_alpha0":0},
        "target_nonzero":{"n":0,"current":0,"fixed_alpha0":0},
    }
    for name,mod,q in qs:
        w=mod.parametrizations.weight.original
        mask=diff_mask[name].to(device=w.device)
        if not bool(mask.any().item()):
            continue
        target=s_codes[name].to(device=w.device)
        current=q.hard_code(w)
        a0=reference_scales[name]["alpha"].to(device=w.device,dtype=w.dtype)
        fixed=torch.round((w/a0).clamp(-0.99,0.99)*1.5).to(torch.int8)

        for key,sel in [
            ("all",mask),
            ("target_zero",mask & (target==0)),
            ("target_nonzero",mask & (target!=0)),
        ]:
            n=int(sel.sum().item())
            groups[key]["n"]+=n
            if n:
                groups[key]["current"]+=int(((current==target)&sel).sum().item())
                groups[key]["fixed_alpha0"]+=int(((fixed==target)&sel).sum().item())

    out={}
    for key,g in groups.items():
        n=max(1,g["n"])
        out[key]={
            "n":g["n"],
            "current_scale_fraction":g["current"]/n,
            "fixed_alpha0_fraction":g["fixed_alpha0"]/n
        }
    return out

def instantiate_g1_3(
    arm,d_masters,s_masters,d_codes,s_codes,diff_mask,
    reference_scales,random_plan,teacher
):
    reset_rng()
    m=load_model(torch.float32)
    bf16_roundtrip_(m)
    qs,trainable=attach_quantizers(m,3)
    load_g1_3_arm(
        qs,arm,d_masters,s_masters,d_codes,s_codes,diff_mask,
        reference_scales,random_plan
    )
    set_reference_scales(qs,reference_scales)
    codes=snapshot_codes(qs)
    diag=score_on(m,teacher,lrval_chunks)
    return m,qs,trainable,codes,diag

def diag_close(a,b,tol=1e-5):
    keys=["loss","ppl","top1_agreement","kl_to_teacher"]
    return all(abs(float(a[k])-float(b[k]))<=tol for k in keys)

def continue_g1_3(
    arm,d_masters,s_masters,d_codes,s_codes,diff_mask,
    reference_scales,random_plan,teacher
):
    m,qs,trainable,start_codes,pre_diag=instantiate_g1_3(
        arm,d_masters,s_masters,d_codes,s_codes,diff_mask,
        reference_scales,random_plan,teacher
    )
    opt=make_optimizer(trainable)
    trace=[]
    survival={}
    t0=time.time()

    for local0 in range(CONT_STEPS):
        global0=PREP_STEPS+local0
        row=one_step(m,teacher,opt,trainable,train_chunks[global0])
        step=local0+1

        if arm=="M-d50" and step in (100,300,900):
            sv=q9_code_survival(qs,diff_mask,s_codes,reference_scales)
            survival[str(step)]=sv
            print(json.dumps({
                "event":"g1_3_survival",
                "arm":arm,"stage_step":step,"survival":sv
            }),flush=True)

        if step==1 or step%TRACE_EVERY==0 or step==CONT_STEPS:
            log={
                "global_step":global0+1,
                "stage_step":step,
                **row
            }
            trace.append(log)
            print(json.dumps({
                "event":"g1_3_continue","arm":arm,**log
            }),flush=True)

    final=score_on(m,teacher,eval_chunks)
    final_codes=snapshot_codes(qs)
    out={
        "pre_diag":pre_diag,
        "final":final,
        "trace":trace,
        "seconds":time.time()-t0,
        "start_codes_vs_final":hamming_codes(start_codes,final_codes),
        "q9_code_survival":survival
    }
    cleanup(opt,m,qs,trainable,start_codes,final_codes)
    return out


ARMS=["D","All-d50","AlignTop50-d50","AlignBottom50-d50"]

def selector_backward(m,teacher,chunks):
    m.train()
    m.zero_grad(set_to_none=True)
    rows=[]
    loss_sum=ce_sum=kl_sum=0.0

    for j,ids_cpu in enumerate(chunks,1):
        x=ids_cpu[:-1].unsqueeze(0).to(DEVICE)
        y=ids_cpu[1:].unsqueeze(0).to(DEVICE)
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
        if not bool(torch.isfinite(loss).item()):
            raise RuntimeError(("nonfinite_selector_loss",j,float(loss.detach().cpu())))
        loss.backward()
        loss_sum+=float(loss.detach().cpu())
        ce_sum+=float(ce.detach().cpu())
        kl_sum+=float(kl.detach().cpu())
        rows.append({
            "selector_step":j,
            "loss":float(loss.detach().cpu()),
            "ce":float(ce.detach().cpu()),
            "kl":float(kl.detach().cpu())
        })

    grad_sq=0.0
    finite=True
    for _,mod in target_linears(m):
        # target_linears yields parametrized Linear modules too.
        w=mod.parametrizations.weight.original
        if w.grad is not None:
            g=w.grad.detach().float()
            finite=finite and bool(torch.isfinite(g).all().item())
            grad_sq+=float((g*g).sum().item())
    if not finite:
        raise RuntimeError("nonfinite selector master gradient")

    n=max(1,len(chunks))
    return {
        "chunks":len(chunks),
        "chunk_range_1indexed":[PREP_STEPS-SELECTOR_WINDOW+1,PREP_STEPS],
        "mean_loss":loss_sum/n,
        "mean_ce":ce_sum/n,
        "mean_kl":kl_sum/n,
        "accum_master_grad_norm":math.sqrt(grad_sq),
        "trace":rows
    }

@torch.no_grad()
def mask_count(mask):
    return sum(int(v.sum().item()) for v in mask.values())

@torch.no_grad()
def mask_overlap_count(a,b):
    return sum(int((a[name] & b[name]).sum().item()) for name in a)

@torch.no_grad()
def mask_outside_count(a,parent):
    return sum(int((a[name] & (~parent[name])).sum().item()) for name in a)

@torch.no_grad()
def expected_codes_for_mask(d_codes,s_codes,mask):
    out={name:c.clone() for name,c in d_codes.items()}
    for name in out:
        out[name][mask[name]]=s_codes[name][mask[name]]
    return out

def build_alignment_halves(
    d_masters,d_codes,s_codes,diff_mask,reference_scales,teacher
):
    reset_rng()
    m=load_model(torch.float32)
    bf16_roundtrip_(m)
    qs,trainable=attach_quantizers(m,3)
    load_masters_scales(qs,d_masters,reference_scales)
    set_reference_scales(qs,reference_scales)

    selector_chunks=train_chunks[PREP_STEPS-SELECTOR_WINDOW:PREP_STEPS]
    selector_info=selector_backward(m,teacher,selector_chunks)

    top_mask={}
    bottom_mask={}
    per_layer_top={}
    per_layer_bottom={}
    score_n=score_pos=score_neg=score_zero=0
    score_sum=score_abs_sum=0.0
    top_score_sum=bottom_score_sum=0.0
    top_n=bottom_n=dropped=0

    qmap={name:(mod,q) for name,mod,q in qs}

    for name in d_codes:
        mod,q=qmap[name]
        w=mod.parametrizations.weight.original
        if w.grad is None:
            raise RuntimeError(("missing_selector_grad",name))

        grad=w.grad.detach().float().reshape(-1)
        base=w.detach().float().reshape(-1)
        dc=d_codes[name].to(device=w.device).reshape(-1)
        sc=s_codes[name].to(device=w.device).reshape(-1)
        mm=diff_mask[name].to(device=w.device).reshape(-1)
        alpha_rows=reference_scales[name]["alpha"].to(
            device=w.device,dtype=torch.float32
        ).reshape(-1)
        cols=w.shape[1]

        tflat=torch.zeros(mm.numel(),dtype=torch.bool)
        bflat=torch.zeros(mm.numel(),dtype=torch.bool)
        taudit=defaultdict(int)
        baudit=defaultdict(int)

        for cd in (-1,0,1):
            for cs in (-1,0,1):
                if cd==cs:
                    continue
                idx=(mm & (dc==cd) & (sc==cs)).nonzero(
                    as_tuple=False
                ).flatten()
                n=int(idx.numel())
                if n==0:
                    continue

                rows=(idx//cols).long()
                alpha=alpha_rows[rows]
                target=depth_value(
                    alpha,dc[idx].to(torch.int8),sc[idx].to(torch.int8),0.5
                )
                delta=target-base[idx]
                vals=-(grad[idx]*delta)

                if not bool(torch.isfinite(vals).all().item()):
                    raise RuntimeError(("nonfinite_alignment_score",name,cd,cs))

                score_n+=n
                score_pos+=int((vals>0).sum().item())
                score_neg+=int((vals<0).sum().item())
                score_zero+=int((vals==0).sum().item())
                score_sum+=float(vals.sum().item())
                score_abs_sum+=float(vals.abs().sum().item())

                k=n//2
                if k==0:
                    dropped+=n
                    continue

                order_idx=torch.argsort(vals)
                low=idx[order_idx[:k]].detach().cpu().long()
                high=idx[order_idx[-k:]].detach().cpu().long()

                bflat[low]=True
                tflat[high]=True
                key=f"{cd}->{cs}"
                taudit[key]+=k
                baudit[key]+=k
                top_n+=k
                bottom_n+=k
                dropped+=n-2*k
                top_score_sum+=float(vals[order_idx[-k:]].sum().item())
                bottom_score_sum+=float(vals[order_idx[:k]].sum().item())

        top_mask[name]=tflat.reshape(d_codes[name].shape)
        bottom_mask[name]=bflat.reshape(d_codes[name].shape)
        per_layer_top[name]=dict(sorted(taudit.items()))
        per_layer_bottom[name]=dict(sorted(baudit.items()))

    m.zero_grad(set_to_none=True)
    cleanup(m,qs,trainable)

    if score_n!=mask_count(diff_mask):
        raise RuntimeError(("selector_score_count_mismatch",score_n,mask_count(diff_mask)))

    stats={
        "definition":"score = - accumulated_Q3_grad * (w_d50 - w_D)",
        "selector":selector_info,
        "score_count":score_n,
        "positive_fraction":score_pos/max(1,score_n),
        "negative_fraction":score_neg/max(1,score_n),
        "zero_fraction":score_zero/max(1,score_n),
        "mean_score":score_sum/max(1,score_n),
        "mean_abs_score":score_abs_sum/max(1,score_n),
        "top_count":top_n,
        "bottom_count":bottom_n,
        "dropped_middle_count":dropped,
        "top_mean_score":top_score_sum/max(1,top_n),
        "bottom_mean_score":bottom_score_sum/max(1,bottom_n),
        "top_per_layer_transition_counts":per_layer_top,
        "bottom_per_layer_transition_counts":per_layer_bottom,
    }
    return top_mask,bottom_mask,stats

@torch.no_grad()
def load_alignment_arm(
    qs,arm,d_masters,d_codes,s_codes,diff_mask,top_mask,bottom_mask,
    reference_scales
):
    if arm=="D":
        selected=None
    elif arm=="All-d50":
        selected=diff_mask
    elif arm=="AlignTop50-d50":
        selected=top_mask
    elif arm=="AlignBottom50-d50":
        selected=bottom_mask
    else:
        raise ValueError(arm)

    for name,mod,q in qs:
        base=d_masters[name].clone()
        if selected is not None:
            sel=selected[name]
            if bool(sel.any().item()):
                a=reference_scales[name]["alpha"].float().expand_as(base)
                dc=d_codes[name]
                sc=s_codes[name]
                base[sel]=depth_value(a[sel],dc[sel],sc[sel],0.5)
        w=mod.parametrizations.weight.original
        w.copy_(base.to(device=w.device,dtype=w.dtype))

def selected_mask_for_arm(arm,diff_mask,top_mask,bottom_mask):
    if arm=="All-d50":
        return diff_mask
    if arm=="AlignTop50-d50":
        return top_mask
    if arm=="AlignBottom50-d50":
        return bottom_mask
    return None

def instantiate_alignment(
    arm,d_masters,d_codes,s_codes,diff_mask,top_mask,bottom_mask,
    reference_scales,teacher
):
    reset_rng()
    m=load_model(torch.float32)
    bf16_roundtrip_(m)
    qs,trainable=attach_quantizers(m,3)
    load_alignment_arm(
        qs,arm,d_masters,d_codes,s_codes,diff_mask,top_mask,bottom_mask,
        reference_scales
    )
    set_reference_scales(qs,reference_scales)
    codes=snapshot_codes(qs)
    diag=score_on(m,teacher,lrval_chunks)
    return m,qs,trainable,codes,diag

def continue_alignment(
    arm,d_masters,d_codes,s_codes,diff_mask,top_mask,bottom_mask,
    reference_scales,teacher
):
    m,qs,trainable,start_codes,pre_diag=instantiate_alignment(
        arm,d_masters,d_codes,s_codes,diff_mask,top_mask,bottom_mask,
        reference_scales,teacher
    )
    opt=make_optimizer(trainable)
    trace=[]
    survival={}
    selected=selected_mask_for_arm(arm,diff_mask,top_mask,bottom_mask)
    t0=time.time()

    for local0 in range(CONT_STEPS):
        global0=PREP_STEPS+local0
        row=one_step(m,teacher,opt,trainable,train_chunks[global0])
        step=local0+1

        if selected is not None and step in (100,300,900):
            sv=q9_code_survival(qs,selected,s_codes,reference_scales)
            survival[str(step)]=sv
            print(json.dumps({
                "event":"g1_6_survival",
                "arm":arm,"stage_step":step,"survival":sv
            }),flush=True)

        if step==1 or step%TRACE_EVERY==0 or step==CONT_STEPS:
            log={"global_step":global0+1,"stage_step":step,**row}
            trace.append(log)
            print(json.dumps({
                "event":"g1_6_continue","arm":arm,**log
            }),flush=True)

    final=score_on(m,teacher,eval_chunks)
    final_codes=snapshot_codes(qs)
    out={
        "pre_diag":pre_diag,
        "final":final,
        "trace":trace,
        "seconds":time.time()-t0,
        "start_codes_vs_final":hamming_codes(start_codes,final_codes),
        "q9_code_survival":survival
    }
    cleanup(opt,m,qs,trainable,start_codes,final_codes)
    return out

print(json.dumps({
    "event":"g1_6_start",
    "model":MODEL_ID,
    "seed":SEED,
    "gpu":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "precision":"FP32 masters + BF16-rounded source + BF16 autocast/BF16 teacher",
    "lr":LR,
    "prep_steps":PREP_STEPS,
    "continuation_steps":CONT_STEPS,
    "selector_window":SELECTOR_WINDOW,
    "selector_chunks_1indexed":[PREP_STEPS-SELECTOR_WINDOW+1,PREP_STEPS],
    "arms":ARMS,
    "alignment_definition":"-grad * (w_d50 - w_D)",
    "split":"within each layer and D->S transition class: top half vs bottom half",
    "order_head":order[:16]
}),flush=True)

teacher=load_model(torch.bfloat16 if DEVICE=="cuda" else torch.float32)
teacher.requires_grad_(False)
teacher.eval()

source=load_model(torch.float32)
bf16_roundtrip_(source)
source_test=score_on(source,teacher,eval_chunks)
cleanup(source)

reference_scales,initial_codes,initial_ref=initial_q3_reference(teacher)
d_masters,d_prep=prepare_arm("D_direct_q3_300",3,teacher)
s_masters,s_prep=prepare_arm("S_q9_300",9,teacher)

d_codes=projected_q3_codes_from_masters(d_masters,reference_scales)
s_codes=projected_q3_codes_from_masters(s_masters,reference_scales)
diff_mask,mask_stats=build_diff_mask(d_codes,s_codes)
true_transitions=transition_counts_for_mask(d_codes,s_codes,diff_mask)

top_mask,bottom_mask,selector_stats=build_alignment_halves(
    d_masters,d_codes,s_codes,diff_mask,reference_scales,teacher
)

top_transitions=transition_counts_for_mask(d_codes,s_codes,top_mask)
bottom_transitions=transition_counts_for_mask(d_codes,s_codes,bottom_mask)

print(json.dumps({
    "event":"g1_6_selector",
    "mask_stats":mask_stats,
    "selector_stats":{
        k:v for k,v in selector_stats.items()
        if k not in ("top_per_layer_transition_counts","bottom_per_layer_transition_counts")
    },
    "top_fraction_of_M":mask_count(top_mask)/max(1,mask_stats["changed_count"]),
    "bottom_fraction_of_M":mask_count(bottom_mask)/max(1,mask_stats["changed_count"])
}),flush=True)

expected_all=s_codes
expected_top=expected_codes_for_mask(d_codes,s_codes,top_mask)
expected_bottom=expected_codes_for_mask(d_codes,s_codes,bottom_mask)

pre_codes={}
pre_diags={}
for arm in ARMS:
    m,qs,trainable,codes,diag=instantiate_alignment(
        arm,d_masters,d_codes,s_codes,diff_mask,top_mask,bottom_mask,
        reference_scales,teacher
    )
    pre_codes[arm]=codes
    pre_diags[arm]=diag
    print(json.dumps({
        "event":"g1_6_pre_diag","arm":arm,"diag":diag
    }),flush=True)
    cleanup(m,qs,trainable)

top_n=mask_count(top_mask)
bottom_n=mask_count(bottom_mask)
checks={
    "top_count":top_n,
    "bottom_count":bottom_n,
    "counts_equal":top_n==bottom_n,
    "top_bottom_overlap":mask_overlap_count(top_mask,bottom_mask),
    "top_outside_M":mask_outside_count(top_mask,diff_mask),
    "bottom_outside_M":mask_outside_count(bottom_mask,diff_mask),
    "per_layer_transition_match":(
        selector_stats["top_per_layer_transition_counts"]
        ==selector_stats["bottom_per_layer_transition_counts"]
    ),
    "global_transition_match":(
        top_transitions["global"]==bottom_transitions["global"]
    ),
    "all_codes_match_S":hamming_codes(pre_codes["All-d50"],expected_all),
    "top_codes_match_expected":hamming_codes(
        pre_codes["AlignTop50-d50"],expected_top
    ),
    "bottom_codes_match_expected":hamming_codes(
        pre_codes["AlignBottom50-d50"],expected_bottom
    )
}

assert checks["counts_equal"], checks
assert checks["top_bottom_overlap"]==0, checks
assert checks["top_outside_M"]==0, checks
assert checks["bottom_outside_M"]==0, checks
assert checks["per_layer_transition_match"], checks
assert checks["global_transition_match"], checks
assert checks["all_codes_match_S"]["fraction"]==0.0, checks
assert checks["top_codes_match_expected"]["fraction"]==0.0, checks
assert checks["bottom_codes_match_expected"]["fraction"]==0.0, checks

print(json.dumps({
    "event":"g1_6_assertions_passed","checks":checks
}),flush=True)

arms={}
for arm in ARMS:
    arms[arm]=continue_alignment(
        arm,d_masters,d_codes,s_codes,diff_mask,top_mask,bottom_mask,
        reference_scales,teacher
    )
    print(json.dumps({
        "event":"g1_6_arm_final",
        "arm":arm,
        "final":arms[arm]["final"],
        "survival":arms[arm]["q9_code_survival"]
    }),flush=True)

L={arm:arms[arm]["final"]["loss"] for arm in ARMS}
effects={
    "D_loss":L["D"],
    "All_d50_loss":L["All-d50"],
    "AlignTop50_d50_loss":L["AlignTop50-d50"],
    "AlignBottom50_d50_loss":L["AlignBottom50-d50"],
    "All_d50_gain_vs_D":L["D"]-L["All-d50"],
    "AlignTop50_gain_vs_D":L["D"]-L["AlignTop50-d50"],
    "AlignBottom50_gain_vs_D":L["D"]-L["AlignBottom50-d50"],
    "Top_advantage_vs_All":L["All-d50"]-L["AlignTop50-d50"],
    "Top_advantage_vs_Bottom":L["AlignBottom50-d50"]-L["AlignTop50-d50"],
    "Bottom_advantage_vs_All":L["All-d50"]-L["AlignBottom50-d50"]
}

final={
    "kind":"g1_6_granite350m_trajectory_alignment",
    "model":MODEL_ID,
    "seed":SEED,
    "precision":"bf16",
    "lr":LR,
    "source_test":source_test,
    "initial_q3_reference":initial_ref,
    "D_prep":d_prep,
    "S_prep":s_prep,
    "geometry":{
        "mask_stats":mask_stats,
        "D_vs_initial":hamming_codes(initial_codes,d_codes),
        "S_vs_initial":hamming_codes(initial_codes,s_codes),
        "D_vs_S":hamming_codes(d_codes,s_codes),
        "true_transitions":true_transitions,
        "top_transitions":top_transitions,
        "bottom_transitions":bottom_transitions
    },
    "selector":selector_stats,
    "pre_diags":pre_diags,
    "checks":checks,
    "arms":arms,
    "effects":effects,
    "training_order_head":order[:16]
}

print("FINAL_JSON_BEGIN",flush=True)
print(json.dumps(final),flush=True)
print("FINAL_JSON_END",flush=True)
