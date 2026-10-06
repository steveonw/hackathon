# /// script
# dependencies = ["torch>=2.4","transformers>=4.46","accelerate>=1.0"]
# ///
"""
SmolLM2-360M-Instruct ternary staircase experiment.

Run v1 compares:
  1) full-precision baseline
  2) direct 3-level PTQ
  3) staged 27 -> 9 -> 3 PTQ
  4) direct 3-level QAT/recovery for 30 steps
  5) staged 27 -> 9 -> 3 QAT/recovery for 10 steps per stage

The direct and staged QAT paths therefore receive the same total number
of optimizer steps (30). This is the script mirrored from Hugging Face
Job 6ac52475404719ba37661c8b.
"""
import torch, math, json, gc, time
import torch.nn as nn
from torch.nn.utils import parametrize
from transformers import AutoModelForCausalLM, AutoTokenizer

MID="HuggingFaceTB/SmolLM2-360M-Instruct"
DEV="cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(1729)

cal=[
"A careful experiment keeps a baseline for comparison.",
"Gradual changes can sometimes preserve behavior better than abrupt changes.",
"Quantization reduces the number of representable weight values.",
"The scientist recorded the result before interpreting it.",
"A map removes detail while preserving useful relationships.",
"Plants convert light energy into chemical energy.",
"Seventeen plus twenty eight equals forty five.",
"The dog waited quietly outside the library.",
"Compression is useful when the smaller model retains enough function.",
"The engineer measured memory, latency, and accuracy after every optimization.",
"A telescope gathers light and focuses it onto a detector.",
"Language models predict the next token from context.",
]
ev=[
"A reliable benchmark uses the same inputs and scoring procedure for every model.",
"The moon reflects sunlight rather than producing its own visible light.",
"Neural network quantization trades numerical precision for lower memory use.",
"The hikers checked their map and returned before dark.",
"A control experiment provides a reference for measuring change.",
]
prompts=[
"In two sentences, explain why a baseline matters in an experiment.",
"What is 17 + 28? Give a short answer.",
"Write one calm sentence about a dog waiting outside a library.",
"What is one possible benefit of reducing numerical precision gradually?",
]

class Q(nn.Module):
    def __init__(self,L):
        super().__init__()
        self.L=L
        self.k=(L-1)//2
    def forward(self,w):
        s=w.detach().abs().amax(
            dim=tuple(range(1,w.ndim)),keepdim=True
        ).clamp_min(1e-8)/self.k
        z=w/s
        q=z.round().clamp(-self.k,self.k)
        return (z+(q-z).detach())*s

def linears(m):
    for n,x in m.named_modules():
        if isinstance(x,nn.Linear) and n!="lm_head":
            yield n,x

def ptq(m,L):
    k=(L-1)//2
    with torch.no_grad():
        for _,x in linears(m):
            w=x.weight
            s=w.abs().amax(dim=1,keepdim=True).clamp_min(1e-8)/k
            w.copy_((w/s).round().clamp(-k,k)*s)

def attach(m,L):
    n=0
    for _,x in linears(m):
        if not parametrize.is_parametrized(x,"weight"):
            parametrize.register_parametrization(x,"weight",Q(L))
            n+=x.parametrizations.weight.original.numel()
    return n

def commit(m):
    for _,x in list(linears(m)):
        if parametrize.is_parametrized(x,"weight"):
            parametrize.remove_parametrizations(
                x,"weight",leave_parametrized=True
            )

def load():
    m=AutoModelForCausalLM.from_pretrained(
        MID,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True
    ).to(DEV)
    m.config.use_cache=False
    return m

tok=AutoTokenizer.from_pretrained(MID)
if tok.pad_token_id is None:
    tok.pad_token=tok.eos_token

def ids(s):
    return tok(
        s,
        return_tensors="pt",
        truncation=True,
        max_length=128
    ).input_ids.to(DEV)

eval_ids=[ids(s) for s in ev]

@torch.no_grad()
def refs(m):
    return [
        m(x).logits[:,:-1].argmax(-1).cpu()
        for x in eval_ids
    ]

@torch.no_grad()
def score(m,R):
    m.eval()
    nll=n=ag=an=0
    for i,x in enumerate(eval_ids):
        z=m(x).logits[:,:-1].float()
        y=x[:,1:]
        nll+=torch.nn.functional.cross_entropy(
            z.reshape(-1,z.shape[-1]),
            y.reshape(-1),
            reduction="sum"
        ).item()
        n+=y.numel()
        p=z.argmax(-1).cpu()
        ag+=(p==R[i]).sum().item()
        an+=p.numel()
    loss=nll/n
    return {
        "loss":loss,
        "ppl":math.exp(min(loss,20)),
        "top1_agreement":ag/an
    }

@torch.no_grad()
def gen(m):
    m.eval()
    m.config.use_cache=True
    out=[]
    for p in prompts:
        t=tok.apply_chat_template(
            [{"role":"user","content":p}],
            tokenize=False,
            add_generation_prompt=True
        )
        e=tok(t,return_tensors="pt").to(DEV)
        g=m.generate(
            **e,
            max_new_tokens=40,
            do_sample=False,
            pad_token_id=tok.eos_token_id
        )
        out.append(tok.decode(
            g[0,e.input_ids.shape[1]:],
            skip_special_tokens=True
        ))
    m.config.use_cache=False
    return out

def train(m,L,steps):
    attach(m,L)
    m.train()
    o=torch.optim.AdamW(
        m.parameters(),
        lr=2e-5,
        weight_decay=0
    )
    losses=[]
    t=time.time()
    for i in range(steps):
        x=ids(cal[i%len(cal)])
        o.zero_grad(set_to_none=True)
        with torch.autocast(
            "cuda",
            dtype=torch.float16,
            enabled=(DEV=="cuda")
        ):
            loss=m(x,labels=x).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(),1.0)
        o.step()
        losses.append(float(loss.detach().cpu()))
        if i in (0,steps-1) or (i+1)%5==0:
            print(json.dumps({
                "train_levels":L,
                "step":i+1,
                "loss":losses[-1]
            }),flush=True)
    commit(m)
    del o
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "first":losses[0],
        "last":losses[-1],
        "seconds":time.time()-t
    }

print(json.dumps({
    "device":DEV,
    "gpu":torch.cuda.get_device_name(0)
        if torch.cuda.is_available() else None
}),flush=True)

res={}

m=load()
R=refs(m)
res["baseline"]=score(m,R)
res["baseline"]["gen"]=gen(m)
del m
gc.collect()
if torch.cuda.is_available(): torch.cuda.empty_cache()

m=load()
ptq(m,3)
res["direct_ptq"]=score(m,R)
res["direct_ptq"]["gen"]=gen(m)
del m
gc.collect()
if torch.cuda.is_available(): torch.cuda.empty_cache()

m=load()
for L in (27,9,3):
    ptq(m,L)
res["staged_ptq"]=score(m,R)
res["staged_ptq"]["gen"]=gen(m)
del m
gc.collect()
if torch.cuda.is_available(): torch.cuda.empty_cache()

m=load()
res["direct_qat_train"]=train(m,3,30)
res["direct_qat"]=score(m,R)
res["direct_qat"]["gen"]=gen(m)
del m
gc.collect()
if torch.cuda.is_available(): torch.cuda.empty_cache()

m=load()
sts=[]
for L in (27,9,3):
    sts.append({"levels":L,**train(m,L,10)})
res["staged_qat_train"]=sts
res["staged_qat"]=score(m,R)
res["staged_qat"]["gen"]=gen(m)

print("FINAL_JSON="+json.dumps(res),flush=True)
