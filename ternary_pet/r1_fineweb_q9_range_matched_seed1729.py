# /// script
# dependencies = ["torch>=2.4","transformers>=4.46","datasets>=3.0","accelerate>=1.0"]
# ///
"""
SmolLM2 v11: matched-schedule four-arm hybrid-master factorial.

Purpose:
Causally localize the schedule-matched Q9 -> Q3 trainability advantage.

Build two equal-compute step-300 FP32 master states under the v10 global LR
curve:
  D = 300 direct-Q3 updates
  S = 300 Q9 updates

Both are projected through the same ORIGINAL Q3 scales. Define mask M at exact
weight positions where projected Q3 codes differ between D and S.

Construct four master states without changing any other parameter:
  00: D on M, D on ~M = D everywhere
  10: S on M, D on ~M
  01: D on M, S on ~M
  11: S on M, S on ~M = S everywhere

By construction:
  Q3(00) == Q3(01)
  Q3(10) == Q3(11)

All arms then receive original Q3 scales, fresh Adam, and the same global LR
curve from step 301 through 1200.

This is a causal partition of the actual D-vs-S prepared-master difference into
code-disagreement positions M versus same-code continuous geometry ~M.
"""
import gc, json, math, time, hashlib
from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import parametrize
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

MODEL_ID="HuggingFaceTB/SmolLM2-360M-Instruct"
MODEL_REVISION="a10cc1512eabd3dde888204e902eca88bddb4951"
FINEWEB_REVISION="87f09149ef4734204d70ed1d046ddc9ca3f2b8f9"
WIKITEXT_REVISION="b08601e04326c79dfdd32d625aee71d232d685c3"
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

tok=AutoTokenizer.from_pretrained(MODEL_ID,revision=MODEL_REVISION)
if tok.pad_token_id is None:
    tok.pad_token=tok.eos_token

def stream_chunks(split,n):
    ds=load_dataset("Salesforce/wikitext","wikitext-2-raw-v1",split=split,revision=WIKITEXT_REVISION)
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

def load_partitioned_fineweb():
    required={"train":1200,"dev":24,"eval":128}
    result={x:[] for x in required}
    buf={x:[] for x in required}
    seen={x:set() for x in required}
    counts={x:0 for x in required}
    stream=load_dataset("HuggingFaceFW/fineweb-edu",name="sample-10BT",
                        split="train",streaming=True,revision=FINEWEB_REVISION)
    read=0
    for row in stream:
        read+=1
        doc_id=str(row["id"])
        bucket=int.from_bytes(hashlib.sha256(doc_id.encode("utf-8")).digest()[:8],"big")%20
        key="eval" if bucket==0 else "dev" if bucket==1 else "train"
        if len(result[key])>=required[key] or doc_id in seen[key]:
            continue
        seen[key].add(doc_id)
        text=row.get("text","")
        if not text or not text.strip():
            continue
        tokens=tok.encode(text,add_special_tokens=False)
        if not tokens:
            continue
        buf[key].extend(tokens+[tok.eos_token_id])
        while len(buf[key])>=SEQ+1 and len(result[key])<required[key]:
            result[key].append(torch.tensor(buf[key][:SEQ+1],dtype=torch.long))
            del buf[key][:SEQ+1]
        counts[key]+=1
        if read%1000==0:
            print(json.dumps({"event":"f1_load_progress","docs_seen":read,
               "chunks":{k:len(result[k]) for k in required}}),flush=True)
        if all(len(result[k])>=required[k] for k in required):
            break
        if read>=15000:
            break
    assert all(len(result[k])==required[k] for k in required),(
         "fineweb_insufficient",read,{k:len(result[k]) for k in required})
    assert not(seen["train"]&seen["dev"] or seen["train"]&seen["eval"] or seen["dev"]&seen["eval"])
    for k in result:
        assert all(int(x.shape[0])==SEQ+1 for x in result[k])
    info={"dataset":"HuggingFaceFW/fineweb-edu","config":"sample-10BT",
       "revision":FINEWEB_REVISION,"partition":"sha256(doc_id) 8-byte leading digest mod20",
       "partition_buckets":{"eval":[0],"dev":[1],"train":list(range(2,20))},
       "docs_scanned":read,"docs_used":{k:len(seen[k]) for k in required},
       "chunks":{k:len(result[k]) for k in required},
       "first_hashed_id_digests":{k:sorted(hashlib.sha256(doc.encode()).hexdigest()[:16]
                                    for doc in seen[k])[:5] for k in required},
       "document_id_partition_disjoint":True}
    return result,info

_fineweb_sets,_data_audit=load_partitioned_fineweb()
train_chunks=_fineweb_sets["train"]
lrval_chunks=_fineweb_sets["dev"]
eval_chunks=_fineweb_sets["eval"]
_order_g=torch.Generator().manual_seed(SEED)
_order=torch.randperm(1200,generator=_order_g).tolist()
train_chunks=[train_chunks[i] for i in _order]
wiki_validation_chunks=stream_chunks("validation",128)


def load_model(dtype=torch.float32):
    m=AutoModelForCausalLM.from_pretrained(
        MODEL_ID,dtype=dtype,low_cpu_mem_usage=True,revision=MODEL_REVISION
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
        # R1: only if Q9 nine-state PREPARATION; Q3 always uses original grid.
        self.range_matched_q9=False

    def alpha(self):
        return F.softplus(self.raw_alpha)+1e-7

    def _n(self):
        if self.levels==3:
            return 1.5
        if self.levels==9:
            return 6.0 if self.range_matched_q9 else 4.0
        raise ValueError(self.levels)

    def hard_code(self,w):
        a=self.alpha().to(device=w.device,dtype=w.dtype)
        n=self._n()
        z=(w/a).clamp(-0.99,0.99)
        # Nine-state matched-range: round six levels per alpha, then
        # clamp INTEGER code to ±4. Do not change z's outer STE clamp.
        code=torch.round(z*n)
        if self.levels==9 and self.range_matched_q9:
            code=code.clamp(-4,4)
        return code.to(torch.int8)

    def forward(self,w):
        a=self.alpha().to(dtype=w.dtype)
        n=self._n()
        z=(w/a).clamp(-0.99,0.99)
        code=torch.round(z*n)
        if self.levels==9 and self.range_matched_q9:
            code=code.clamp(-4,4)
        hard=code/n
        # STE: same surrogate and z.clamp(-.99,.99) for both Q9 grids.
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



@torch.no_grad()
def capture_scale_state(qs):
    out={}
    for name,mod,q in qs:
        out[name]={
            "raw_alpha":q.raw_alpha.detach().cpu().clone(),
            "alpha":q.alpha().detach().cpu().clone(),
        }
    return out

@torch.no_grad()
def capture_quantized_masters(qs):
    return {
        name:mod.parametrizations.weight.original.detach().cpu().clone()
        for name,mod,q in qs
    }

@torch.no_grad()
def capture_plain_masters(m):
    return {
        name:mod.weight.detach().cpu().clone()
        for name,mod in target_linears(m)
    }

@torch.no_grad()
def load_masters_and_scales(qs,masters,scale_state):
    for name,mod,q in qs:
        w=mod.parametrizations.weight.original
        w.copy_(masters[name].to(device=w.device,dtype=w.dtype))
        raw=scale_state[name]["raw_alpha"]
        q.raw_alpha.copy_(raw.to(device=q.raw_alpha.device,dtype=q.raw_alpha.dtype))
        q.levels=3

@torch.no_grad()
def threshold_margin_stats(qs):
    # Q3 boundaries in normalized z=w/alpha coordinates are +/- 1/3 because
    # round(1.5*z) changes integer code at +/-0.5.
    epses=[0.005,0.01,0.02,0.05,0.10]
    counts={str(e):0 for e in epses}
    total=0
    margin_sum=0.0
    layer_rows=[]
    for name,mod,q in qs:
        w=mod.parametrizations.weight.original.detach().float()
        a=q.alpha().detach().to(device=w.device,dtype=w.dtype)
        z=w/a
        margin=torch.minimum((z-(1.0/3.0)).abs(),(z+(1.0/3.0)).abs())
        n=margin.numel()
        total+=n
        margin_sum+=margin.sum().item()
        layer_counts={}
        for e in epses:
            c=int((margin<e).sum().item())
            counts[str(e)]+=c
            layer_counts[str(e)]=c/max(1,n)
        layer_rows.append((layer_counts["0.02"],name,layer_counts))
    layer_rows.sort(reverse=True)
    return {
        "mean_normalized_distance":margin_sum/max(1,total),
        "fraction_within":{k:v/max(1,total) for k,v in counts.items()},
        "top_layers_within_0.02":[
            {"name":name,"fraction_within":fracs}
            for _,name,fracs in layer_rows[:8]
        ]
    }

@torch.no_grad()
def hamming_codes(a,b):
    changed=0; total=0; rows=[]
    for name in a:
        x=a[name]; y=b[name]
        ch=int((x!=y).sum().item()); n=x.numel()
        changed+=ch; total+=n
        rows.append((ch/max(1,n),name))
    rows.sort(reverse=True)
    return {
        "fraction":changed/max(1,total),
        "top_layers":[{"name":name,"fraction":fr} for fr,name in rows[:8]]
    }

@torch.no_grad()
def code_overlap_against_initial(initial,a,b):
    total=0
    ca=cb=both=union=same_final_both=0
    for name in initial:
        i=initial[name]; x=a[name]; y=b[name]
        ma=(x!=i); mb=(y!=i)
        n=i.numel(); total+=n
        ca+=int(ma.sum().item()); cb+=int(mb.sum().item())
        bi=ma & mb
        both+=int(bi.sum().item())
        union+=int((ma|mb).sum().item())
        same_final_both+=int((bi & (x==y)).sum().item())
    return {
        "a_changed_fraction":ca/max(1,total),
        "b_changed_fraction":cb/max(1,total),
        "changed_set_intersection_fraction":both/max(1,total),
        "changed_set_union_fraction":union/max(1,total),
        "changed_set_jaccard":both/max(1,union),
        "same_final_code_among_both_changed":same_final_both/max(1,both),
        "pairwise_hamming":hamming_codes(a,b)["fraction"]
    }

def initial_q3_reference(teacher):
    m=load_model(torch.float32); bf16_roundtrip_(m)
    qs,trainable=attach_quantizers(m,3)
    scales=capture_scale_state(qs)
    codes=snapshot_codes(qs)
    out={
        "diag":score_on(m,teacher,lrval_chunks),
        "hist":code_hist(qs),
        "scale":scale_stats(qs),
        "margin":threshold_margin_stats(qs)
    }
    del m,qs,trainable
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return scales,codes,out

def prepare_quantized(label,levels,teacher,lr):
    m=load_model(torch.float32); bf16_roundtrip_(m)
    qs,trainable=attach_quantizers(m,levels)
    opt=make_optimizer(trainable,lr)
    scaler=torch.amp.GradScaler("cuda",enabled=(DEVICE=="cuda"))
    trace=[]
    for step in range(300):
        loss,ce,kl=one_step(m,teacher,opt,scaler,train_chunks[step])
        if step==0 or (step+1)%50==0 or step+1==300:
            row={"step":step+1,"loss":loss,"ce":ce,"kl":kl}
            trace.append(row)
            print(json.dumps({"event":"v7_prep","label":label,**row}),flush=True)
    native_diag=score_on(m,teacher,lrval_chunks)
    masters=capture_quantized_masters(qs)
    out={
        "native_space":"q3" if levels==3 else "q9",
        "native_diag":native_diag,
        "learned_scale":scale_stats(qs),
        "trace":trace
    }
    del opt,scaler,m,qs,trainable
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return masters,out

def prepare_fp32(label,teacher,lr):
    m=load_model(torch.float32); bf16_roundtrip_(m)
    for p in m.parameters():
        p.requires_grad_(False)
    trainable=[]
    for name,mod in target_linears(m):
        mod.weight.requires_grad_(True)
        trainable.append(mod.weight)
    opt=make_optimizer(trainable,lr)
    scaler=torch.amp.GradScaler("cuda",enabled=(DEVICE=="cuda"))
    trace=[]
    for step in range(300):
        loss,ce,kl=one_step(m,teacher,opt,scaler,train_chunks[step])
        if step==0 or (step+1)%50==0 or step+1==300:
            row={"step":step+1,"loss":loss,"ce":ce,"kl":kl}
            trace.append(row)
            print(json.dumps({"event":"v7_prep","label":label,**row}),flush=True)
    native_diag=score_on(m,teacher,lrval_chunks)
    masters=capture_plain_masters(m)
    out={
        "native_space":"unquantized",
        "native_diag":native_diag,
        "trace":trace
    }
    del opt,scaler,m,trainable
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return masters,out

def project_and_continue(label,masters,reference_scales,initial_codes,teacher,lr):
    m=load_model(torch.float32); bf16_roundtrip_(m)
    qs,trainable=attach_quantizers(m,3)
    load_masters_and_scales(qs,masters,reference_scales)

    start_codes=snapshot_codes(qs)
    pre={
        "diag":score_on(m,teacher,lrval_chunks),
        "hist":code_hist(qs),
        "margin":threshold_margin_stats(qs),
        "vs_initial_codes":hamming_codes(initial_codes,start_codes)
    }

    # All arms receive the same fresh optimizer and same original scales here.
    opt=make_optimizer(trainable,lr)
    scaler=torch.amp.GradScaler("cuda",enabled=(DEVICE=="cuda"))
    stage_start=start_codes
    prev={k:v.clone() for k,v in stage_start.items()}
    trace=[]
    start_time=time.time()
    for local in range(900):
        loss,ce,kl=one_step(m,teacher,opt,scaler,train_chunks[300+local])
        if local==0 or (local+1)%FLIP_EVERY==0 or local+1==900:
            prev,fs=flip_stats(qs,prev,stage_start)
            row={"step":local+1,"loss":loss,"ce":ce,"kl":kl,**fs}
            trace.append(row)
            print(json.dumps({"event":"v7_continue","label":label,**row}),flush=True)

    final=score_on(m,teacher,eval_chunks)
    final["gen"]=generate(m)
    final["final_hist"]=code_hist(qs)
    final["scale"]=scale_stats(qs)
    final["trace"]=trace
    final["seconds"]=time.time()-start_time

    del opt,scaler,m,qs,trainable,prev,stage_start
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return pre,final,start_codes

PEAK_LR=1e-3
MIN_LR=1e-4
WARMUP_STEPS=100
FULL_STEPS=1200
PREP_STEPS=300
CONT_STEPS=900

def matched_lr(global_step0):
    s=global_step0+1
    if s<=WARMUP_STEPS:
        return PEAK_LR*s/WARMUP_STEPS
    frac=(s-WARMUP_STEPS)/max(1,FULL_STEPS-WARMUP_STEPS)
    frac=min(max(frac,0.0),1.0)
    return MIN_LR+0.5*(PEAK_LR-MIN_LR)*(1.0+math.cos(math.pi*frac))

def set_optimizer_lr(opt,lr):
    for group in opt.param_groups:
        group["lr"]=float(lr)

def prepare_q9_matched(teacher):
    m=load_model(torch.float32); bf16_roundtrip_(m)
    qs,trainable=attach_quantizers(m,9)
    opt=make_optimizer(trainable,matched_lr(0))
    scaler=torch.amp.GradScaler("cuda",enabled=(DEVICE=="cuda"))
    trace=[]
    stage_start=snapshot_codes(qs)
    prev={k:v.clone() for k,v in stage_start.items()}
    start_time=time.time()

    for global_step0 in range(PREP_STEPS):
        lr=matched_lr(global_step0)
        set_optimizer_lr(opt,lr)
        loss,ce,kl=one_step(m,teacher,opt,scaler,train_chunks[global_step0])
        if global_step0==0 or (global_step0+1)%50==0 or global_step0+1==PREP_STEPS:
            prev,fs=flip_stats(qs,prev,stage_start)
            row={
                "global_step":global_step0+1,"levels":9,"lr":lr,
                "loss":loss,"ce":ce,"kl":kl,**fs
            }
            trace.append(row)
            print(json.dumps({"event":"matched_q9_prep",**row}),flush=True)

    masters=capture_quantized_masters(qs)
    out={
        "native_diag":score_on(m,teacher,lrval_chunks),
        "learned_q9_scale":scale_stats(qs),
        "hist":code_hist(qs),
        "trace":trace,
        "seconds":time.time()-start_time
    }
    del opt,scaler,m,qs,trainable,prev,stage_start
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return masters,out

def continue_q3_matched(masters,reference_scales,initial_codes,teacher):
    m=load_model(torch.float32); bf16_roundtrip_(m)
    qs,trainable=attach_quantizers(m,3)
    load_masters_and_scales(qs,masters,reference_scales)

    start_codes=snapshot_codes(qs)
    pre={
        "diag":score_on(m,teacher,lrval_chunks),
        "hist":code_hist(qs),
        "margin":threshold_margin_stats(qs),
        "vs_initial_codes":hamming_codes(initial_codes,start_codes),
        "first_q3_lr":matched_lr(PREP_STEPS)
    }

    # v7 transition semantics: fresh Adam + original Q3 scales.
    # Only the global LR curve continues; optimizer moments do not.
    opt=make_optimizer(trainable,matched_lr(PREP_STEPS))
    scaler=torch.amp.GradScaler("cuda",enabled=(DEVICE=="cuda"))
    stage_start=start_codes
    prev={k:v.clone() for k,v in stage_start.items()}
    trace=[]
    start_time=time.time()

    for local in range(CONT_STEPS):
        global_step0=PREP_STEPS+local
        lr=matched_lr(global_step0)
        set_optimizer_lr(opt,lr)
        loss,ce,kl=one_step(m,teacher,opt,scaler,train_chunks[global_step0])
        if local==0 or (local+1)%50==0 or local+1==CONT_STEPS:
            prev,fs=flip_stats(qs,prev,stage_start)
            row={
                "global_step":global_step0+1,"stage_step":local+1,
                "levels":3,"lr":lr,"loss":loss,"ce":ce,"kl":kl,**fs
            }
            trace.append(row)
            print(json.dumps({"event":"matched_q3_continue",**row}),flush=True)

    final=score_on(m,teacher,eval_chunks)
    final["gen"]=generate(m)
    final["final_hist"]=code_hist(qs)
    final["scale"]=scale_stats(qs)
    final["trace"]=trace
    final["seconds"]=time.time()-start_time

    final_codes=snapshot_codes(qs)
    del opt,scaler,m,qs,trainable,prev,stage_start
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return pre,final,start_codes,final_codes


def reset_rng():
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

@torch.no_grad()
def projected_q3_codes_from_masters(masters, reference_scales):
    out={}
    for name,w in masters.items():
        a=reference_scales[name]["alpha"].float()
        z=(w.float()/a).clamp(-0.99,0.99)
        out[name]=torch.round(z*1.5).to(torch.int8)
    return out

@torch.no_grad()
def build_diff_mask(d_codes,s_codes):
    mask={}
    changed=0; total=0; layer_rows=[]
    for name in d_codes:
        m=(d_codes[name]!=s_codes[name])
        n=m.numel(); ch=int(m.sum().item())
        mask[name]=m
        changed+=ch; total+=n
        layer_rows.append((ch/max(1,n),name,ch,n))
    layer_rows.sort(reverse=True)
    return mask,{
        "fraction":changed/max(1,total),
        "changed_count":changed,
        "total":total,
        "top_layers":[
            {"name":name,"fraction":fr,"changed_count":ch,"total":n}
            for fr,name,ch,n in layer_rows[:10]
        ]
    }

def prepare_matched(label,levels,teacher):
    reset_rng()
    m=load_model(torch.float32); bf16_roundtrip_(m)
    qs,trainable=attach_quantizers(m,levels)
    opt=make_optimizer(trainable,matched_lr(0))
    scaler=torch.amp.GradScaler("cuda",enabled=(DEVICE=="cuda"))
    trace=[]
    stage_start=snapshot_codes(qs)
    prev={k:v.clone() for k,v in stage_start.items()}
    start_time=time.time()

    for global_step0 in range(PREP_STEPS):
        lr=matched_lr(global_step0)
        set_optimizer_lr(opt,lr)
        loss,ce,kl=one_step(m,teacher,opt,scaler,train_chunks[global_step0])
        if global_step0==0 or (global_step0+1)%50==0 or global_step0+1==PREP_STEPS:
            prev,fs=flip_stats(qs,prev,stage_start)
            row={
                "global_step":global_step0+1,
                "levels":levels,"lr":lr,
                "loss":loss,"ce":ce,"kl":kl,**fs
            }
            trace.append(row)
            print(json.dumps({"event":"v11_prep","label":label,**row}),flush=True)

    masters=capture_quantized_masters(qs)
    out={
        "label":label,
        "levels":levels,
        "native_diag":score_on(m,teacher,lrval_chunks),
        "learned_scale":scale_stats(qs),
        "native_hist":code_hist(qs),
        "trace":trace,
        "seconds":time.time()-start_time
    }
    del opt,scaler,m,qs,trainable,prev,stage_start
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return masters,out

@torch.no_grad()
def load_hybrid_masters(qs,d_masters,s_masters,diff_mask,arm):
    if arm not in ("00","10","01","11"):
        raise ValueError(arm)
    use_s_on_m=(arm[0]=="1")
    use_s_off_m=(arm[1]=="1")
    for name,mod,q in qs:
        d=d_masters[name]
        s=s_masters[name]
        mask=diff_mask[name]
        if use_s_on_m and use_s_off_m:
            h=s
        elif (not use_s_on_m) and (not use_s_off_m):
            h=d
        elif use_s_on_m:
            h=torch.where(mask,s,d)
        else:
            h=torch.where(mask,d,s)
        w=mod.parametrizations.weight.original
        w.copy_(h.to(device=w.device,dtype=w.dtype))

@torch.no_grad()
def set_reference_scales(qs,reference_scales):
    for name,mod,q in qs:
        raw=reference_scales[name]["raw_alpha"]
        q.raw_alpha.copy_(raw.to(device=q.raw_alpha.device,dtype=q.raw_alpha.dtype))
        q.levels=3

def instantiate_hybrid(arm,d_masters,s_masters,diff_mask,reference_scales,teacher):
    m=load_model(torch.float32); bf16_roundtrip_(m)
    qs,trainable=attach_quantizers(m,3)
    load_hybrid_masters(qs,d_masters,s_masters,diff_mask,arm)
    set_reference_scales(qs,reference_scales)
    codes=snapshot_codes(qs)
    diag=score_on(m,teacher,lrval_chunks)
    return m,qs,trainable,codes,diag

def continue_hybrid(arm,d_masters,s_masters,diff_mask,reference_scales,teacher):
    reset_rng()
    m,qs,trainable,start_codes,pre_diag=instantiate_hybrid(
        arm,d_masters,s_masters,diff_mask,reference_scales,teacher
    )
    opt=make_optimizer(trainable,matched_lr(PREP_STEPS))
    scaler=torch.amp.GradScaler("cuda",enabled=(DEVICE=="cuda"))
    stage_start=start_codes
    prev={k:v.clone() for k,v in stage_start.items()}
    trace=[]
    start_time=time.time()

    for local in range(CONT_STEPS):
        global_step0=PREP_STEPS+local
        lr=matched_lr(global_step0)
        set_optimizer_lr(opt,lr)
        loss,ce,kl=one_step(m,teacher,opt,scaler,train_chunks[global_step0])
        if local==0 or (local+1)%50==0 or local+1==CONT_STEPS:
            prev,fs=flip_stats(qs,prev,stage_start)
            row={
                "global_step":global_step0+1,"stage_step":local+1,
                "lr":lr,"loss":loss,"ce":ce,"kl":kl,**fs
            }
            trace.append(row)
            print(json.dumps({"event":"v11_continue","arm":arm,**row}),flush=True)

    final=score_on(m,teacher,eval_chunks)
    final["gen"]=generate(m)
    final["final_hist"]=code_hist(qs)
    final["scale"]=scale_stats(qs)
    final["trace"]=trace
    final["seconds"]=time.time()-start_time
    final_codes=snapshot_codes(qs)

    out={
        "pre_diag":pre_diag,
        "start_hist":code_hist(qs) if False else None,
        "final":final,
        "start_codes_vs_final":hamming_codes(start_codes,final_codes)
    }

    del opt,scaler,m,qs,trainable,prev,stage_start,start_codes,final_codes
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return out

def diag_close(a,b,tol=1e-5):
    keys=["loss","ppl","top1_agreement","kl_to_teacher"]
    return all(abs(float(a[k])-float(b[k]))<=tol for k in keys)



R1_PREREG="ternary_pet/research_log/r1_range_matched_q9_fineweb_seed1729_prereg_2026-10-09.md"
R1_HIST_F1={
 "D_fineweb":5.7666598074138165,
 "wide_fineweb":4.996255073696375,
 "D_wikitext":6.558238908648491,
 "wide_wikitext":5.695225466042757
}
R1_EXPECTED_ORDER=[221,850,89,747,685,1055,781,170,233,62,421,806,1031,1187,683,619]

@torch.no_grad()
def r1_q9_grid_diagnostic(qs):
    counts=defaultdict(int)
    num=0
    range_gt_q3=0
    outer_clip=0
    unclamped_code_over4=0
    code_at_abs4=0
    max_effective_output_ratio=0.0
    for name,mod,q in qs:
        assert q.levels==9
        w=mod.parametrizations.weight.original.detach()
        a=q.alpha().to(device=w.device,dtype=w.dtype)
        zz=w/a
        z=zz.clamp(-0.99,0.99)
        n=q._n()
        rawcode=torch.round(z*n)
        code=rawcode.clamp(-4,4) if q.range_matched_q9 else rawcode
        assert int(code.abs().max().item())<=4
        num+=w.numel()
        range_gt_q3+=int((zz.abs()>(2.0/3.0)).sum().item())
        outer_clip+=int((zz.abs()>=0.99).sum().item())
        unclamped_code_over4+=int((rawcode.abs()>4).sum().item())
        code_at_abs4+=int((code.abs()==4).sum().item())
        mx=float((code.abs()/n).max().item())
        max_effective_output_ratio=max(mx,max_effective_output_ratio)
    return {"weights":num,"fp32_master_above_q3_range_fraction":range_gt_q3/num,
            "outer_ste_clipped_fraction":outer_clip/num,
            "would_be_integer_code_clipped_fraction":unclamped_code_over4/num,
            "abs4_code_fraction":code_at_abs4/num,
            "max_abs_output_over_alpha":max_effective_output_ratio}

def r1_run_arm(label,prep_mode,teacher,source_q3_scales,source_codes):
    assert prep_mode in ("direct","wide","range"),prep_mode
    reset_rng()
    m=load_model(torch.float32)
    bf16_roundtrip_(m)
    uses_q9=(prep_mode!="direct")
    qs,params=attach_quantizers(m,9 if uses_q9 else 3)
    if uses_q9:
        for name,mod,q in qs:
            q.range_matched_q9=(prep_mode=="range")
            assert q.levels==9
    optimizer=make_optimizer(params,matched_lr(0))
    amp=torch.amp.GradScaler("cuda",enabled=(DEVICE=="cuda"))
    effective=0;skipped=0
    trace=[];dev={};switches=[];q9_snapshot=None
    started=time.time()
    m.train()
    for i in range(1200):
        if uses_q9 and i==300:
            native=score_on(m,teacher,lrval_chunks)
            q9_snapshot={"grid":r1_q9_grid_diagnostic(qs),
                         "native_train_dev":native,"hard_hist":code_hist(qs),
                         "learned_scale_stats":scale_stats(qs),
                         "mode":prep_mode}
            print(json.dumps({"event":"r1_q9_prep","arm":label,**q9_snapshot}),flush=True)
            for name,mod,q in qs:
                q.raw_alpha.data.copy_(source_q3_scales[name]["raw_alpha"].to(q.raw_alpha.device))
                q.levels=3
                q.range_matched_q9=False
            projected=score_on(m,teacher,lrval_chunks)
            switches.append({"at_step":i,"q9_native_dev":native,
                             "original_scale_q3_dev":projected,
                             "shock":projected["loss"]-native["loss"]})
            del optimizer,amp
            gc.collect()
            optimizer=make_optimizer(params,matched_lr(i))
            amp=torch.amp.GradScaler("cuda",enabled=(DEVICE=="cuda"))
            m.train()
            print(json.dumps({"event":"r1_switch","arm":label,**switches[-1]}),flush=True)
        lr=matched_lr(i)
        set_optimizer_lr(optimizer,lr)
        old=amp.get_scale()
        loss,ce,kl=one_step(m,teacher,optimizer,amp,train_chunks[i])
        skip=bool(amp.get_scale()<old)
        skipped+=int(skip);effective+=int(not skip)
        step=i+1
        if not math.isfinite(loss):
            raise RuntimeError(("R1_nonfinite",label,step))
        if step==1 or step%100==0:
            row={"step":step,"loss":loss,"ce":ce,"kl":kl,"lr":lr,
                 "updates":effective,"skips":skipped,"levels":qs[0][2].levels}
            trace.append(row)
            print(json.dumps({"event":"r1_train","arm":label,**row}),flush=True)
        if step in (300,600,900,1200):
            dev[str(step)]=score_on(m,teacher,lrval_chunks)
            m.train()
    assert effective+skipped==1200
    assert all(q.levels==3 for name,mod,q in qs)
    assert all(not q.range_matched_q9 for name,mod,q in qs)
    ff=score_on(m,teacher,eval_chunks)
    fw=score_on(m,teacher,wiki_validation_chunks)
    out={"arm":label,"q9_prep_grid":prep_mode,"q9_prep_steps":300 if uses_q9 else 0,
         "scheduled_steps":1200,"effective_updates":effective,"amp_skipped":skipped,
         "native_q9_at_switch":q9_snapshot,"switches":switches,
         "train_split_dev":dev,"trace":trace,"fineweb_heldout":ff,"wikitext_validation":fw,
         "final_source_code_movement":hamming_codes(source_codes,snapshot_codes(qs)),
         "final_code_hist":code_hist(qs),"final_scale_stats":scale_stats(qs),
         "elapsed_seconds":time.time()-started}
    print(json.dumps({"event":"r1_arm_final","arm":label,"fineweb_loss":ff["loss"],
                      "wiki_val_loss":fw["loss"],"updates":effective,"skips":skipped}),flush=True)
    del m,qs,params,optimizer,amp
    gc.collect()
    if torch.cuda.is_available():torch.cuda.empty_cache()
    return out

print(json.dumps({"event":"r1_start","seed":SEED,"gpu":torch.cuda.get_device_name(0),
                  "prereg":R1_PREREG,"model_revision":MODEL_REVISION,
                  "fineweb_revision":FINEWEB_REVISION,"order_head":_order[:16],
                  "dataset_audit":_data_audit,
                  "arms":["D","S_wide","S_range"]}),flush=True)
teacher=load_model(torch.float16 if DEVICE=="cuda" else torch.float32)
teacher.requires_grad_(False);teacher.eval()
ref_scales,source_codes,source_diag=initial_q3_reference(teacher)
arms={}
for key,mode in (("D","direct"),("S_wide","wide"),("S_range","range")):
    arms[key]=r1_run_arm(key,mode,teacher,ref_scales,source_codes)
D=arms["D"];W=arms["S_wide"];R=arms["S_range"]
checks={
 "all_1200_same_opportunities":all(x["scheduled_steps"]==1200 and
       x["effective_updates"]+x["amp_skipped"]==1200 for x in arms.values()),
 "all_final_q3":all(len(x["switches"])==(0 if key=="D" else 1) for key,x in arms.items()),
 "fixed_fineweb_document_split":_data_audit["document_id_partition_disjoint"] and
       _data_audit["chunks"]=={"train":1200,"dev":24,"eval":128} and
       _data_audit["docs_used"]=={"train":180,"dev":3,"eval":21},
 "seed1729_training_order":_order[:16]==R1_EXPECTED_ORDER,
 "D_fineweb_reproduction":abs(D["fineweb_heldout"]["loss"]-R1_HIST_F1["D_fineweb"])<0.03,
 "S_wide_fineweb_reproduction":abs(W["fineweb_heldout"]["loss"]-R1_HIST_F1["wide_fineweb"])<0.03,
 "D_wikitext_reproduction":abs(D["wikitext_validation"]["loss"]-R1_HIST_F1["D_wikitext"])<0.03,
 "S_wide_wikitext_reproduction":abs(W["wikitext_validation"]["loss"]-R1_HIST_F1["wide_wikitext"])<0.03,
 "all_finite":all(math.isfinite(x[k]["loss"]) for x in arms.values() for k in ("fineweb_heldout","wikitext_validation")),
 "Q9_native_range_match":abs(R["native_q9_at_switch"]["grid"]["max_abs_output_over_alpha"]-(2/3))<1e-6,
 "Q9_wide_native_range":abs(W["native_q9_at_switch"]["grid"]["max_abs_output_over_alpha"]-1.0)<1e-6,
 "same_prep_target_weights":R["native_q9_at_switch"]["grid"]["weights"]==W["native_q9_at_switch"]["grid"]["weights"]==314572800
}
effects={
 "fineweb_range_minus_wide":R["fineweb_heldout"]["loss"]-W["fineweb_heldout"]["loss"],
 "fineweb_direct_minus_range":D["fineweb_heldout"]["loss"]-R["fineweb_heldout"]["loss"],
 "fineweb_direct_minus_wide":D["fineweb_heldout"]["loss"]-W["fineweb_heldout"]["loss"],
 "wikitext_range_minus_wide":R["wikitext_validation"]["loss"]-W["wikitext_validation"]["loss"],
 "wikitext_direct_minus_range":D["wikitext_validation"]["loss"]-R["wikitext_validation"]["loss"],
 "wikitext_direct_minus_wide":D["wikitext_validation"]["loss"]-W["wikitext_validation"]["loss"]
}
import importlib.metadata as md
versions={}
for dep in ("torch","transformers","datasets","accelerate","huggingface_hub"):
    try:versions[dep]=md.version(dep)
    except Exception:versions[dep]="unavailable"
result={"kind":"r1_q9_range_matched_fineweb_seed1729",
        "prereg":R1_PREREG,"seed":SEED,
        "model_revision":MODEL_REVISION,"fineweb_revision":FINEWEB_REVISION,
        "wikitext_revision":WIKITEXT_REVISION,"dataset_audit":_data_audit,
        "training_order_head":_order[:16],"dependency_versions":versions,
        "arms":arms,"effects":effects,"checks":checks,
        "valid_for_science":all(checks.values()),
        "exploratory_same_seed_and_heldout":True,
        "warning":"Range and spacing differ by construction, so not pure range-only ablation."}
print("FINAL_JSON_BEGIN",flush=True)
print(json.dumps(result),flush=True)
print("FINAL_JSON_END",flush=True)
if not result["valid_for_science"]:
    raise RuntimeError(("R1_TECHNICAL_REPRO_MISMATCH",checks))
