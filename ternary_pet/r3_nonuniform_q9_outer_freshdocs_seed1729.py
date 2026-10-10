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


# R3 independent and document-linked heldout: original F1's source loader
# stopped after reading the first 340 records. Only later bucket0 records
# with enough tokens are eligible. Each doc is evaluated separately.
R3_FRESH_DOCS=32
R3_CHUNKS_PER_DOC=4
assert _data_audit["docs_scanned"]==340

def load_r3_fresh_eval():
    seen=set(); selected=set(); docs=[]; scanned=0;eligible=0;short=0
    stream=load_dataset("HuggingFaceFW/fineweb-edu",name="sample-10BT",
         split="train",streaming=True,revision=FINEWEB_REVISION)
    for pos,row in enumerate(stream,1):
        scanned=pos
        digest=hashlib.sha256(str(row["id"]).encode("utf-8")).hexdigest()
        if pos<=340:
            seen.add(digest);continue
        if digest in seen or digest in selected:continue
        bucket=int.from_bytes(bytes.fromhex(digest)[:8],"big")%20
        if bucket!=0:continue
        eligible+=1
        content=row.get("text","")
        if not content or not content.strip():
            short+=1;continue
        tokens=tok.encode(content,add_special_tokens=False)
        if len(tokens)<R3_CHUNKS_PER_DOC*(SEQ+1):
            short+=1;continue
        chunks=[torch.tensor(tokens[j*(SEQ+1):(j+1)*(SEQ+1)],dtype=torch.long)
                for j in range(R3_CHUNKS_PER_DOC)]
        docs.append({"sha256":digest,"row":pos,"chunks":chunks})
        selected.add(digest)
        if len(docs)==R3_FRESH_DOCS:break
        if pos>=20000:break
    assert len(docs)==R3_FRESH_DOCS,("fresh_doc_shortfall",len(docs),scanned)
    assert len(selected)==32 and not (seen & selected)
    audit={"count":32,"chunks_per_doc":4,"tokens_per_doc":512,
      "total_tokens":16384,"scan_rows":scanned,
      "old_first340_distinct_ids":len(seen),"disjoint_all_first340":True,
      "eligible_bucket0_candidates":eligible,"short_candidates":short,
      "selection":"bucket0, rows>340, first32 distinct docs at least516 tokens",
      "document_sha256":[d["sha256"] for d in docs],
      "document_row":[d["row"] for d in docs],
      "dataset_revision":FINEWEB_REVISION}
    print(json.dumps({"event":"r3_fresh_eval_selected","audit":audit}),flush=True)
    return docs,audit

fresh_docs,fresh_audit=load_r3_fresh_eval()

@torch.no_grad()
def score_r3_fresh(model):
    model.eval()
    per=[];total=0.0;nt=0
    for d in fresh_docs:
        sm=0.0;n=0
        for ids in d["chunks"]:
            x=ids[:-1].unsqueeze(0).to(DEVICE)
            y=ids[1:].unsqueeze(0).to(DEVICE)
            with torch.autocast("cuda",dtype=torch.float16,enabled=(DEVICE=="cuda")):
                logits=model(x).logits.float()
            loss=F.cross_entropy(logits.reshape(-1,logits.shape[-1]),
                   y.reshape(-1),reduction="sum")
            sm+=float(loss.item());n+=int(y.numel())
        assert n==512
        total+=sm;nt+=n
        per.append({"doc_sha256":d["sha256"],"source_row":d["row"],
                    "tokens":n,"nll_sum":sm,"nll":sm/n})
    assert nt==16384 and len(per)==32
    return {"nll":total/nt,"ppl":math.exp(min(total/nt,20)),
            "tokens":nt,"docs":len(per),"per_document":per}

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
        # R2 independent nine-state codebook factors; ignored for final Q3.
        self.q9_threshold_multiplier=4.0
        self.q9_output_divisor=4.0
        self.q9_mode="W"

    def alpha(self):
        return F.softplus(self.raw_alpha)+1e-7

    def _n(self):
        if self.levels==3:
            return 1.5
        if self.levels==9:
            return self.q9_threshold_multiplier
        raise ValueError(self.levels)

    def _output_n(self):
        if self.levels==3:return 1.5
        if self.levels==9:return self.q9_output_divisor
        raise ValueError(self.levels)

    def hard_code(self,w):
        a=self.alpha().to(device=w.device,dtype=w.dtype)
        n=self._n()
        z=(w/a).clamp(-0.99,0.99)
        # Nine-state matched-range: round six levels per alpha, then
        # clamp INTEGER code to ±4. Do not change z's outer STE clamp.
        code=torch.round(z*n)
        if self.levels==9:
            code=code.clamp(-4,4)
        return code.to(torch.int8)

    def forward(self,w):
        a=self.alpha().to(dtype=w.dtype)
        n=self._n()
        z=(w/a).clamp(-0.99,0.99)
        code=torch.round(z*n)
        if self.levels==9:
            code=code.clamp(-4,4)
        hard=code/self._output_n()
        if self.levels==9 and self.q9_mode=="H":
            hard=hard+torch.where(code.abs()==4,code.sign()/3.0,torch.zeros_like(code))
        # Identical clipped-z STE surrogate for both output grids.
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




R3_PREREG="ternary_pet/research_log/r3_nonuniform_q9_outer_recovery_freshdoc_prereg_2026-10-09.md"
R3_MODES={
 "W":{"V":4.0,"codes":[0.0,0.25,0.5,0.75,1.0]},
 "N":{"V":6.0,"codes":[0.0,1/6,2/6,3/6,4/6]},
 "H":{"V":6.0,"codes":[0.0,1/6,2/6,3/6,1.0]}
}
R3_HIST={
 "D":(5.7666598074138165,6.558238908648491),
 "W":(4.996255073696375,5.695225466042757),
 "N":(5.780473280698061,6.593468256294727)}
R3_ORDER=[221,850,89,747,685,1055,781,170,233,62,421,806,1031,1187,683,619]

@torch.no_grad()
def prep_diag_r3(qs):
    n=0;outer=0;clipped=0;mx=0.0;codes_seen=set()
    for _,mod,q in qs:
        w=mod.parametrizations.weight.original.detach()
        a=q.alpha().to(w.device,w.dtype)
        z=w/a
        k=torch.round(z.clamp(-.99,.99)*4).clamp(-4,4)
        output=k/q.q9_output_divisor
        if q.q9_mode=="H":
            output=output+torch.where(k.abs()==4,k.sign()/3,torch.zeros_like(k))
        n+=w.numel();outer+=(k.abs()==4).sum().item()
        clipped+=(z.abs()>=.99).sum().item()
        mx=max(mx,float(output.abs().max().item()))
        codes_seen.update(int(t.item()) for t in torch.unique(k.to(torch.int8)))
    return {"weights":n,"extreme_code_fraction":outer/n,
            "outer_z_clipped_fraction":clipped/n,
            "max_abs_output_over_alpha":mx,"reachable_codes":sorted(codes_seen)}

def run_r3_arm(name,teacher,ref_scales,source_codes):
    assert name=="D" or name in R3_MODES
    reset_rng()
    m=load_model(torch.float32);bf16_roundtrip_(m)
    staged=name!="D"
    qs,params=attach_quantizers(m,9 if staged else 3)
    if staged:
        for _,_,q in qs:
            q.q9_mode=name;q.q9_threshold_multiplier=4.0
            q.q9_output_divisor=R3_MODES[name]["V"]
    opt=make_optimizer(params,matched_lr(0))
    scaler=torch.amp.GradScaler("cuda",enabled=(DEVICE=="cuda"))
    upd=0;skips=0;switch=[];prep_state=None;trace=[];dev={}
    started=time.time();m.train()
    for i in range(1200):
        if staged and i==300:
            native=score_on(m,teacher,lrval_chunks)
            prep_state={"native_dev":native,"grid":prep_diag_r3(qs),
                        "scales":scale_stats(qs)}
            for nm,mod,q in qs:
                q.raw_alpha.data.copy_(ref_scales[nm]["raw_alpha"].to(q.raw_alpha.device))
                q.levels=3
            projected=score_on(m,teacher,lrval_chunks)
            switch.append({"at_step":300,"native_dev":native,
                           "projected_dev":projected,
                           "shock":projected["loss"]-native["loss"]})
            del opt,scaler;gc.collect()
            opt=make_optimizer(params,matched_lr(i))
            scaler=torch.amp.GradScaler("cuda",enabled=(DEVICE=="cuda"))
            m.train()
            print(json.dumps({"event":"r3_switch","arm":name,"shock":switch[-1]["shock"]}),flush=True)
        lr=matched_lr(i);set_optimizer_lr(opt,lr)
        prev=scaler.get_scale()
        loss,ce,kl=one_step(m,teacher,opt,scaler,train_chunks[i])
        skipped=int(scaler.get_scale()<prev)
        skips+=skipped;upd+=1-skipped
        step=i+1
        if not math.isfinite(loss):raise RuntimeError(("r3_training_nonfinite",name,step))
        if step==1 or step%100==0:
            t={"step":step,"lr":lr,"loss":loss,"ce":ce,"kl":kl,
               "effective_updates":upd,"skips":skips}
            trace.append(t)
            print(json.dumps({"event":"r3_train","arm":name,**t}),flush=True)
        if step in (300,600,900,1200):
            dev[str(step)]=score_on(m,teacher,lrval_chunks);m.train()
    assert upd+skips==1200 and all(q.levels==3 for _,_,q in qs)
    fresh=score_r3_fresh(m)
    old=score_on(m,teacher,eval_chunks)
    wiki=score_on(m,teacher,wiki_validation_chunks)
    out={"arm":name,"scheduled_steps":1200,"effective_updates":upd,
         "amp_skips":skips,"prep_diag":prep_state,"switches":switch,
         "train_dev":dev,"trace":trace,"fresh":fresh,"historical_fineweb":old,
         "historical_wikitext":wiki,
         "q3_code_movement":hamming_codes(source_codes,snapshot_codes(qs)),
         "final_hist":code_hist(qs),"final_scale":scale_stats(qs),
         "seconds":time.time()-started}
    print(json.dumps({"event":"r3_arm_final","arm":name,
         "fresh_nll":fresh["nll"],"old_fineweb":old["loss"],
         "wiki":wiki["loss"],"skips":skips}),flush=True)
    del m,qs,opt,scaler,params;gc.collect()
    if torch.cuda.is_available():torch.cuda.empty_cache()
    return out

print(json.dumps({"event":"r3_start","prereg":R3_PREREG,"seed":SEED,
   "source_audit":_data_audit,"fresh_audit":fresh_audit,
   "order_head":_order[:16],"model_revision":MODEL_REVISION,
   "fineweb_revision":FINEWEB_REVISION}),flush=True)
teacher=load_model(torch.float16 if DEVICE=="cuda" else torch.float32)
teacher.requires_grad_(False);teacher.eval()
ref_scales,source_codes,source_diag=initial_q3_reference(teacher)
arms={}
for name in ("D","W","N","H"):
    arms[name]=run_r3_arm(name,teacher,ref_scales,source_codes)
fresh={k:arms[k]["fresh"]["nll"] for k in arms}
old={k:arms[k]["historical_fineweb"]["loss"] for k in arms}
wiki={k:arms[k]["historical_wikitext"]["loss"] for k in arms}
comparisons={}
for x,y in (("N","H"),("H","W"),("D","H"),("D","W"),("D","N")):
    u=arms[x]["fresh"]["per_document"]
    v=arms[y]["fresh"]["per_document"]
    assert [z["doc_sha256"] for z in u]==[z["doc_sha256"] for z in v]
    gaps=sorted(a["nll"]-b["nll"] for a,b in zip(u,v))
    comparisons[x+"_minus_"+y]={"aggregate_gap":fresh[x]-fresh[y],
        "mean_doc_gap":sum(gaps)/32,"positive_doc_count":sum(x>0 for x in gaps),
        "median_doc_gap":(gaps[15]+gaps[16])/2,
        "min":gaps[0],"max":gaps[-1]}
checks={
 "four_1200_opportunities":len(arms)==4 and all(x["scheduled_steps"]==1200
       and x["effective_updates"]+x["amp_skips"]==1200 for x in arms.values()),
 "three_Q9_switches_at300":all(len(arms[k]["switches"])==1
       and arms[k]["switches"][0]["at_step"]==300 for k in ("W","N","H")),
 "D_no_switch":len(arms["D"]["switches"])==0,
 "same_train_docs_chunks":_data_audit["document_id_partition_disjoint"]
       and _data_audit["docs_used"]=={"train":180,"dev":3,"eval":21}
       and _data_audit["chunks"]=={"train":1200,"dev":24,"eval":128},
 "exact_training_order":_order[:16]==R3_ORDER,
 "new32_docs_disjoint_old_scan":fresh_audit["count"]==32
       and fresh_audit["disjoint_all_first340"]
       and len(set(fresh_audit["document_sha256"]))==32,
 "new_docs_16384_paired_tokens":all(x["fresh"]["tokens"]==16384
       and x["fresh"]["docs"]==32
       and [y["doc_sha256"] for y in x["fresh"]["per_document"]]==fresh_audit["document_sha256"]
       for x in arms.values()),
 "finite_new_and_old":all(math.isfinite(z) for z in (*fresh.values(),*old.values(),*wiki.values())),
 "all_nine_preparation_codes":all(arms[k]["prep_diag"]["grid"]["reachable_codes"]==list(range(-4,5)) for k in ("W","N","H")),
 "actual_preparation_ranges":all(abs(arms[k]["prep_diag"]["grid"]["max_abs_output_over_alpha"]-
       (2/3 if k=="N" else 1.0))<1e-6 for k in ("W","N","H")),
 "reproduce_D_old_fineweb":abs(old["D"]-R3_HIST["D"][0])<.03,
 "reproduce_W_old_fineweb":abs(old["W"]-R3_HIST["W"][0])<.03,
 "reproduce_N_old_fineweb":abs(old["N"]-R3_HIST["N"][0])<.03,
 "reproduce_D_wikitext":abs(wiki["D"]-R3_HIST["D"][1])<.03,
 "reproduce_W_wikitext":abs(wiki["W"]-R3_HIST["W"][1])<.03,
 "reproduce_N_wikitext":abs(wiki["N"]-R3_HIST["N"][1])<.03}
import importlib.metadata as md
versions={}
for p in ("torch","transformers","datasets","accelerate","huggingface_hub"):
    try:versions[p]=md.version(p)
    except Exception:versions[p]="unavailable"
result={"kind":"r3_q9_nonuniform_outer_recovery_freshdocs_seed1729",
  "prereg":R3_PREREG,"seed":SEED,"model_revision":MODEL_REVISION,
  "fineweb_revision":FINEWEB_REVISION,"wikitext_revision":WIKITEXT_REVISION,
  "source_data_audit":_data_audit,"new_data_audit":fresh_audit,
  "codebooks":R3_MODES,"dependency_versions":versions,"arms":arms,
  "fresh_doc_primary":comparisons,"fresh_aggregate_nll":fresh,
  "old_fineweb_nll":old,"old_wikitext_nll":wiki,"checks":checks,
  "valid_for_science":all(checks.values()),
  "limitations":"New to QAT evaluation not guaranteed absent from original source model pretraining. Single seed/model; nonuniform codebook also modifies learned scale gradients."}
print("FINAL_JSON_BEGIN",flush=True)
print(json.dumps(result),flush=True)
print("FINAL_JSON_END",flush=True)
if not result["valid_for_science"]:
    raise RuntimeError(("R3_TECHNICAL_CHECK_FAILURE",checks))
