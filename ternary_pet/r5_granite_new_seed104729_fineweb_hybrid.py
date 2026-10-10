# /// script
# dependencies = ["torch>=2.4","transformers>=4.57.1","datasets>=3.0","accelerate>=1.0"]
# ///
"""
R4-G: Granite 350M R3-style outer reconstruction cross-seed robustness; preregistered easy/hard.

Scientific question:
Does Granite reproduce the SmolLM2 causal mechanism: a small D-vs-S projected
Q3 disagreement set carrying the staging benefit, with useful interior
placement at d=0.5 and a matched-random specificity control?

Preregistered seed/order 271828 arms:
  D          direct-prepared masters everywhere
  S          Q9-prepared masters everywhere
  M-exact    S masters only on the true D-vs-S Q3 disagreement mask
  M-d50      S-selected Q3 code on true M at normalized depth d=0.5
  Random-d50 per-layer/source->target matched random positions outside M

Every arm receives original Q3 scales, fresh Adam, constant LR 1e-4, and the
same continuation chunks 301-1200. No held-out result is used to construct M
or the random control.
"""
import gc, json, math, time, hashlib, os, tempfile
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import parametrize
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID="ibm-granite/granite-4.0-350m"
MODEL_REVISION="bd8a1497065c0d6ba1ef19af6b0d2b14bacf71c2"
WIKITEXT_REVISION="b08601e04326c79dfdd32d625aee71d232d685c3"
DEVICE="cuda" if torch.cuda.is_available() else "cpu"
SEED=104729
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

all_train=stream_chunks("train",TRAIN_CHUNKS+LRVAL_CHUNKS)
train_chunks=all_train[:TRAIN_CHUNKS]
lrval_chunks=all_train[TRAIN_CHUNKS:]
g=torch.Generator().manual_seed(SEED)
order=torch.randperm(len(train_chunks),generator=g).tolist()
train_chunks=[train_chunks[i] for i in order]
eval_chunks=stream_chunks("test",EVAL_CHUNKS)
validation_chunks=stream_chunks("validation",64)

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
    def __init__(self,w,levels):
        super().__init__()
        with torch.no_grad():
            meanabs=w.detach().abs().mean(dim=1,keepdim=True).float().clamp_min(1e-7)
            init_alpha=1.5*meanabs
        self.raw_alpha=nn.Parameter(inv_softplus(init_alpha))
        self.levels=int(levels)
        self.q9_mode="W"

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
        k=torch.round(z*n)
        hard=k/n
        if self.levels==9 and self.q9_mode in ("N","H"):
            hard=k/6.0
            if self.q9_mode=="H":
                # Only the two extreme codebook levels regain wide-Q9 amplitude.
                hard=hard+torch.where(k.abs()==4,k.sign()/3.0,torch.zeros_like(k))
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



R5_PROTOCOL="ternary_pet/research_log/r5_independent_seed_new_fineweb_validation_prereg_2026-10-09.md"
R5_FINEWEB_REV="87f09149ef4734204d70ed1d046ddc9ca3f2b8f9"
R5_BUCKET=7
R5_SOURCE_MIN_ROW=1223
R5_CKPT_REPO="codeflash85/ternary-pet-r5-checkpoints"

def r5_select_new_documents():
    import hashlib
    source=load_dataset("HuggingFaceFW/fineweb-edu",name="sample-10BT",
        split="train",streaming=True,revision=R5_FINEWEB_REV)
    docs=[];seen=set();scanned=0;eligible=0;too_short=0
    for rownum,item in enumerate(source,1):
        scanned=rownum
        if rownum<R5_SOURCE_MIN_ROW:continue
        if rownum>20000:break
        docid=str(item["id"])
        digest=hashlib.sha256(docid.encode("utf-8")).hexdigest()
        if digest in seen:continue
        seen.add(digest)
        if int.from_bytes(bytes.fromhex(digest)[:8],"big")%20!=R5_BUCKET:continue
        eligible+=1
        text=item.get("text","")
        if not text or not text.strip():too_short+=1;continue
        tokens=tok.encode(text,add_special_tokens=False)
        if len(tokens)<516:too_short+=1;continue
        windows=[torch.tensor(tokens[k*129:(k+1)*129],dtype=torch.long) for k in range(4)]
        assert all(len(w)==129 for w in windows)
        docs.append({"sha256":digest,"source_row":rownum,"windows":windows})
        if len(docs)==32:break
    assert len(docs)==32,("R5_fresh_doc_shortfall",scanned,len(docs))
    assert len(set(x["sha256"] for x in docs))==32
    audit={"dataset":"HuggingFaceFW/fineweb-edu","dataset_revision":R5_FINEWEB_REV,
       "minimum_row":R5_SOURCE_MIN_ROW,"hash_bucket":7,
       "selected_docs":len(docs),"scanned_rows":scanned,
       "qualifying_bucket_docs":eligible,"short_docs":too_short,
       "target_tokens":16384,
       "source_rows":[x["source_row"] for x in docs],
       "document_sha256":[x["sha256"] for x in docs]}
    print(json.dumps({"event":"r5_fresh_doc_selector","model":MODEL_ID,"audit":audit}),flush=True)
    return docs,audit

R5_FRESH_DOCS,R5_FRESH_AUDIT=r5_select_new_documents()

@torch.no_grad()
def r5_score_new_fineweb(m):
    m.eval();per=[];total=0.0;n=0
    for item in R5_FRESH_DOCS:
        sm=0.0;count=0
        for ids in item["windows"]:
            x=ids[:-1].unsqueeze(0).to(DEVICE)
            y=ids[1:].unsqueeze(0).to(DEVICE)
            with torch.autocast("cuda",dtype=torch.bfloat16,enabled=(DEVICE=="cuda")):
                z=m(x).logits.float()
            ce=F.cross_entropy(z.reshape(-1,z.shape[-1]),y.reshape(-1),reduction="sum")
            sm+=float(ce.item());count+=int(y.numel())
        assert count==512 and math.isfinite(sm)
        per.append({"doc_sha256":item["sha256"],"row":item["source_row"],
                    "tokens":count,"nll_sum":sm,"nll":sm/count})
        total+=sm;n+=count
    assert len(per)==32 and n==16384
    return {"nll":total/n,"ppl":math.exp(min(total/n,20)),
            "tokens":n,"document_count":32,"per_document":per}

@torch.no_grad()
def r5_checkpoint_h(qs,arm):
    if arm!="H":return {"attempted":False,"retained":False}
    import numpy as np
    from huggingface_hub import HfApi
    payload={};meta=[];total=0
    for i,(name,mod,q) in enumerate(qs):
        code=q.hard_code(mod.parametrizations.weight.original).cpu().numpy().astype(np.int8)
        assert np.all(np.isin(code,[-1,0,1]))
        flat=np.where(code.reshape(-1)==-1,2,code.reshape(-1)).astype(np.uint8)
        pad=(-flat.size)%4
        if pad:flat=np.pad(flat,(0,pad))
        packed=(flat[0::4] | (flat[1::4]<<2) | (flat[2::4]<<4) | (flat[3::4]<<6)).astype(np.uint8)
        key=f"{i:03d}"
        payload["codes_"+key]=packed
        payload["alpha_"+key]=q.alpha().detach().float().cpu().numpy()
        meta.append({"key":key,"module":name,"shape":list(code.shape),"count":int(code.size),
                     "packing":"2 bits/code: 0->0, +1->1, -1->2; 4 codes/byte, least-significant first",
                     "alpha":"code/1.5 multiplied by alpha, normal Q3 inference"})
        total+=code.size
    manifest={"kind":"r5_restorable_inference_quantized_linears_only",
         "seed":SEED,"family":MODEL_ID,"model_revision":MODEL_REVISION,
         "source_roundtrip":"BF16-rounded source, unchanged nonquantized tensors from pinned base model",
         "note":"No FP32 master weights, optimizer moments or restartable training state.",
         "layers":meta,"quantized_weights":total}
    payload["manifest_utf8"]=np.frombuffer(json.dumps(manifest).encode("utf-8"),dtype=np.uint8)
    outpath=os.path.join(tempfile.gettempdir(),f"r5_granite_seed{SEED}_H_final_compact.npz")
    np.savez_compressed(outpath,**payload)
    with open(outpath,"rb") as f:digest=hashlib.sha256(f.read()).hexdigest()
    size=os.path.getsize(outpath)
    ans={"attempted":True,"retained":False,"local_size":size,"sha256":digest,
         "repo_id":R5_CKPT_REPO,"remote_path":f"granite_seed{SEED}/H_final_compact.npz",
         "layers":len(meta),"quantized_weights":total}
    try:
        token=os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        if not token:
            raise RuntimeError("Missing HF_TOKEN write credential in HF Job secrets; compact snapshot not retained")
        api=HfApi(token=token)
        api.create_repo(repo_id=R5_CKPT_REPO,repo_type="dataset",private=True,exist_ok=True)
        info=api.repo_info(repo_id=R5_CKPT_REPO,repo_type="dataset")
        if not getattr(info,"private",False):raise RuntimeError("Refusing to upload into nonprivate repo")
        api.upload_file(path_or_fileobj=outpath,path_in_repo=ans["remote_path"],
            repo_id=R5_CKPT_REPO,repo_type="dataset",
            commit_message=f"R5 Granite seed{SEED} compact H inference snapshot")
        ans["retained"]=True
        print(json.dumps({"event":"r5_snapshot_uploaded","summary":ans}),flush=True)
    except Exception as exc:
        ans["error"]=f"{type(exc).__name__}: {str(exc)[:600]}"
        print(json.dumps({"event":"r5_snapshot_upload_failed","summary":ans}),flush=True)
    finally:
        try:os.remove(outpath)
        except OSError:pass
    return ans

R4_PROTOCOL=R5_PROTOCOL
R4_LABEL="UNSEEN_PREREG_SEED"
R4_HIST={}
R4_ORDER_EXPECTED={} # Filled with deterministic torch.randperm(seed) self-check in this job.
ARM_ORDER=("D","W","N","H")
CODEBOOKS={"W":[0.,.25,.5,.75,1.],"N":[0.,1/6,2/6,3/6,4/6],
           "H":[0.,1/6,2/6,3/6,1.]}

@torch.no_grad()
def prep_grid_diagnostics(qs):
    count=0;extreme=0;code_seen=set();max_output=0.0
    for name,mod,q in qs:
        assert q.levels==9 and q.q9_mode in CODEBOOKS
        w=mod.parametrizations.weight.original
        a=q.alpha().detach().to(w.device,w.dtype)
        k=q.hard_code(w)
        count+=int(k.numel())
        extreme+=int((k.abs()==4).sum().item())
        code_seen.update(int(x.item()) for x in torch.unique(k))
        grid=CODEBOOKS[q.q9_mode]
        max_output=max(max_output,float(max(grid)))
    return {"quantized_weight_count":count,"outer_code_fraction":extreme/count,
            "codes_observed":sorted(code_seen),"max_output_alpha":max_output}

@torch.no_grad()
def heldout_chunk_losses(m,chunks):
    m.eval();sums=[];counts=[]
    for ids_cpu in chunks:
        x=ids_cpu[:-1].unsqueeze(0).to(DEVICE)
        y=ids_cpu[1:].unsqueeze(0).to(DEVICE)
        with torch.autocast("cuda",dtype=torch.bfloat16,enabled=(DEVICE=="cuda")):
            logits=m(x).logits.float()
        q=F.cross_entropy(logits.reshape(-1,logits.shape[-1]),
                         y.reshape(-1),reduction="sum")
        assert math.isfinite(float(q.item()))
        sums.append(float(q.item()));counts.append(int(y.numel()))
    return {"nll":sum(sums)/sum(counts),
            "ppl":math.exp(min(sum(sums)/sum(counts),20)),
            "tokens":sum(counts),"chunk_count":len(sums),
            "per_chunk_nll_sum":sums,"per_chunk_token_count":counts}

def run_r4_arm(arm,teacher,reference_scales,source_codes):
    assert arm in ARM_ORDER
    reset_rng()
    m=load_model(torch.float32)
    bf16_roundtrip_(m)
    qs,trainable=attach_quantizers(m,3 if arm=="D" else 9)
    if arm!="D":
        for nm,mod,q in qs:
            q.q9_mode=arm
    opt=make_optimizer(trainable)
    prep_trace=[];continued_trace=[]
    t0=time.time()
    for step0 in range(PREP_STEPS):
        row=one_step(m,teacher,opt,trainable,train_chunks[step0])
        step=step0+1
        if step in (1,100,200,300):
            prep_trace.append({"step":step,**row})
            print(json.dumps({"event":"r4_train","arm":arm,"phase":"prep","step":step,**row}),flush=True)
    prep_diag=score_on(m,teacher,lrval_chunks)
    prep_codes=code_hist(qs)
    prep_grid=prep_grid_diagnostics(qs) if arm!="D" else None
    masters=capture_masters(qs)
    del m,qs,trainable,opt
    gc.collect()
    if torch.cuda.is_available():torch.cuda.empty_cache()

    # Exactly historical Granite continuation: reinitialize BF16-rounded
    # model, restore FP32 masters + original Q3 scales, use fresh AdamW.
    reset_rng()
    m=load_model(torch.float32)
    bf16_roundtrip_(m)
    qs,trainable=attach_quantizers(m,3)
    load_masters_scales(qs,masters,reference_scales)
    del masters
    q3_start=score_on(m,teacher,lrval_chunks)
    opt=make_optimizer(trainable)
    for step0 in range(PREP_STEPS,TRAIN_CHUNKS):
        row=one_step(m,teacher,opt,trainable,train_chunks[step0])
        step=step0+1
        if step in (301,400,600,900,1200):
            continued_trace.append({"step":step,**row})
            print(json.dumps({"event":"r4_train","arm":arm,"phase":"continue","step":step,**row}),flush=True)
    historical=score_on(m,teacher,eval_chunks)
    fresh=heldout_chunk_losses(m,validation_chunks)
    fresh_documents=r5_score_new_fineweb(m)
    snapshot=r5_checkpoint_h(qs,arm)
    final_codes=code_hist(qs)
    movement=hamming_codes(source_codes,snapshot_codes(qs))
    result={"arm":arm,"train_scheduled":1200,"prep_steps":300,"q3_continue_steps":900,
            "prep_trace":prep_trace,"continuation_trace":continued_trace,
            "prep_native_train_dev":prep_diag,"prep_code_hist":prep_codes,
            "prep_grid":prep_grid,"projected_q3_at_switch":q3_start,
            "final_old_wikitext_test":historical,"final_wikitext_validation":fresh,
            "fresh_fineweb":fresh_documents,"snapshot":snapshot,
            "final_q3_code_hist":final_codes,"final_code_change_from_source":movement,
            "run_seconds":time.time()-t0}
    print(json.dumps({"event":"r4_arm_final","arm":arm,
      "old_nll":historical["loss"],"fresh_val_nll":fresh["nll"],"fresh_fineweb_nll":fresh_documents["nll"]}),flush=True)
    del m,qs,opt,trainable
    gc.collect()
    if torch.cuda.is_available():torch.cuda.empty_cache()
    return result

print(json.dumps({"event":"r4_start","seed":SEED,"known_seed_category":R4_LABEL,
"model":MODEL_ID,"model_revision":MODEL_REVISION,
"dataset_revision":WIKITEXT_REVISION,"precision":"Granite BF16 teacher and autocast, FP32 masters, no GradScaler",
"lr":LR,"train_steps":TRAIN_CHUNKS,"prep300":PREP_STEPS,
"arm_order":ARM_ORDER,"protocol":R4_PROTOCOL,
"training_order_first16":order[:16],"gpu":torch.cuda.get_device_name(0) if DEVICE=="cuda" else None}),flush=True)
teacher=load_model(torch.bfloat16 if DEVICE=="cuda" else torch.float32)
teacher.requires_grad_(False);teacher.eval()
reference_scales,initial_codes,reference=initial_q3_reference(teacher)
arms={}
for arm in ARM_ORDER:
    arms[arm]=run_r4_arm(arm,teacher,reference_scales,initial_codes)

old={k:arms[k]["final_old_wikitext_test"]["loss"] for k in ARM_ORDER}
fresh={k:arms[k]["final_wikitext_validation"]["nll"] for k in ARM_ORDER}
new_comparisons={}
for left,right in (("N","H"),("H","W"),("D","H"),("D","W"),("D","N")):
    ll=arms[left]["final_wikitext_validation"]["per_chunk_nll_sum"]
    rr=arms[right]["final_wikitext_validation"]["per_chunk_nll_sum"]
    assert len(ll)==len(rr)==64
    deltas=sorted((u-v)/SEQ for u,v in zip(ll,rr))
    new_comparisons[left+"_minus_"+right]={
      "aggregate_gap":fresh[left]-fresh[right],
      "positive_chunks":sum(x>0 for x in deltas),
      "median_chunk_gap":(deltas[31]+deltas[32])/2,
      "minimum_chunk_gap":deltas[0],"maximum_chunk_gap":deltas[-1]}

new={k:arms[k]["fresh_fineweb"]["nll"] for k in ARM_ORDER}
paired={}
for left,right in (("N","H"),("H","W"),("D","H"),("D","W"),("D","N")):
    left_docs=arms[left]["fresh_fineweb"]["per_document"]
    right_docs=arms[right]["fresh_fineweb"]["per_document"]
    assert [x["doc_sha256"] for x in left_docs]==[x["doc_sha256"] for x in right_docs]
    diff=sorted(a["nll"]-b["nll"] for a,b in zip(left_docs,right_docs))
    paired[left+"_minus_"+right]={
        "aggregate_gap":new[left]-new[right],
        "positive_documents":sum(x>0 for x in diff),
        "median_doc_gap":(diff[15]+diff[16])/2,
        "min_doc_gap":diff[0],"max_doc_gap":diff[-1]}
checks={
 "four_arms":set(arms)==set(ARM_ORDER),
 "all_1200_steps":all(a["train_scheduled"]==1200 and a["prep_steps"]==300 and
         a["q3_continue_steps"]==900 for a in arms.values()),
 "new_seed_104729":SEED==104729,
 "base_model_pin":MODEL_REVISION=="bd8a1497065c0d6ba1ef19af6b0d2b14bacf71c2",
 "wikitext_pin":WIKITEXT_REVISION=="b08601e04326c79dfdd32d625aee71d232d685c3",
 "new_fineweb_pin":R5_FINEWEB_REV=="87f09149ef4734204d70ed1d046ddc9ca3f2b8f9",
 "fresh_32_docs_16384_tokens":R5_FRESH_AUDIT["selected_docs"]==32
   and len(R5_FRESH_AUDIT["document_sha256"])==32
   and len(set(R5_FRESH_AUDIT["document_sha256"]))==32
   and all(x>=1223 for x in R5_FRESH_AUDIT["source_rows"])
   and all(a["fresh_fineweb"]["document_count"]==32 and
       a["fresh_fineweb"]["tokens"]==16384 for a in arms.values()),
 "fresh_fineweb_paired":all([d["doc_sha256"] for d in arms[k]["fresh_fineweb"]["per_document"]]==R5_FRESH_AUDIT["document_sha256"] for k in ARM_ORDER),
 "all_final_q3_and_expected_count":all(a["final_q3_code_hist"]["total"]==249561088 for a in arms.values()),
 "source_q3_count":reference["hist"]["total"]==249561088,
 "9_states_at_prep":all(a["prep_grid"]["codes_observed"]==list(range(-4,5)) for k,a in arms.items() if k!="D"),
 "Q9_expected_ranges":all(abs(arms[k]["prep_grid"]["max_output_alpha"]-
       (2/3 if k=="N" else 1.0))<1e-6 for k in ("W","N","H")),
 "all_finite":all(math.isfinite(v) for v in [*new.values(),*old.values(),*fresh.values()]),
 "Granite_constant_BF16_LR":LR==1e-4,
 "snapshot_H_attempted":arms["H"]["snapshot"]["attempted"]}
import importlib.metadata as md
versions={}
for pkg in ("torch","transformers","datasets","accelerate","huggingface_hub","numpy"):
    try:versions[pkg]=md.version(pkg)
    except Exception:versions[pkg]="unknown"
out={"kind":"r5_granite_new_seed_cross_corpus_hybrid","seed":SEED,
 "protocol":R5_PROTOCOL,"model_revision":MODEL_REVISION,
 "fineweb_revision":R5_FINEWEB_REV,"wikitext_revision":WIKITEXT_REVISION,
 "fineweb_new_doc_audit":R5_FRESH_AUDIT,"training_order_first16":order[:16],
 "dependency_versions":versions,"arms":arms,"fresh_fineweb_nll":new,
 "primary_paired_documents":paired,"secondary_wikitext_validation_nll":fresh,
 "legacy_wikitext_test_nll":old,"checks":checks,
 "valid_for_science":all(checks.values()),
 "checkpoint_status":arms["H"]["snapshot"],
 "caveat":"1 new deterministic seed, FineWeb documents new to QAT research not guaranteed outside original model pretraining, different architecture training recipes, snapshot is inference-only."}
print("FINAL_JSON_BEGIN",flush=True)
print(json.dumps(out),flush=True)
print("FINAL_JSON_END",flush=True)
if not out["valid_for_science"]:raise RuntimeError(("R5_INVALID",checks))
