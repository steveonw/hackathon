# /// script
# dependencies = ["torch>=2.4","transformers>=4.57.1","accelerate>=1.0","huggingface_hub>=0.30","numpy>=1.26"]
# ///
"""R8 initial Q9 step-zero quantization diagnostic. No training, private snapshots or GPU."""
import argparse,hashlib,json,os,re,urllib.request
import numpy as np,torch
from transformers import AutoTokenizer,AutoModelForCausalLM
from huggingface_hub import hf_hub_download

REPO="codeflash85/ternary-pet-r5-checkpoints"
PROMPTS="https://raw.githubusercontent.com/steveonw/hackathon/7956c5ee8a067efbab7a34ecca84dd287cf5cab9/ternary_pet/research_log/r7_capability_baseline_prompts_2026-10-10.json"
CASES={
"granite":{"folder":"granite_seed170141","model":"ibm-granite/granite-4.0-350m","rev":"bd8a1497065c0d6ba1ef19af6b0d2b14bacf71c2","layers":168,"weights":249561088,"sha":["02cf83d461ccf33ffa595956b3445ffb5c651303a952883d809e69387d3237db","52ab128a5ad2c6d3afe4f88497bdf1a343ced3bd92a33e300e7b4e8732dffecf","9a19dca77ef91de8791123167fbad0bdb9334772509f2922ec49e90a80a1714a","f38d054414c0ac99850cf8130ef65aabc0c73b6f317edf778064bc2037c26f4a"]},
"smol":{"folder":"smol_seed190027","model":"HuggingFaceTB/SmolLM2-360M-Instruct","rev":"a10cc1512eabd3dde888204e902eca88bddb4951","layers":224,"weights":314572800,"sha":["b6be6a5d924fe95122d64cdee05fa3277fea4c509ce74e654045615b3c44722a","6df947c00994e7d9a0322db26a6f77a966f1c4c9da30d28e26f82e7751018574","7f89c6215c266af58d90fd3c671147edfdcab3a502a2a4285465e8e2fab9dbf4","468af19ee4a45535664703285e257372b2d3b81dfd2e74e2fafe319626bfc522"]}}
STAGES=["C1_Q9_step300","C2_projected_Q3_step300","C3_Q3_step600","C4_Q3_step1200"]
def decode(packed,n,levels):
    if levels==9:
        assert packed.size==(n+1)//2
        v=np.empty(packed.size*2,dtype=np.uint8)
        v[0::2]=packed&15;v[1::2]=(packed>>4)&15
        codes=v[:n].astype(np.int16)-4
        assert np.all((-4<=codes)&(codes<=4))
        # H uses narrow interiors, wide outermost codes.
        coeff=codes.astype(np.float32)/6.0
        coeff[np.abs(codes)==4]=np.sign(codes[np.abs(codes)==4]).astype(np.float32)
    else:
        assert levels==3 and packed.size==(n+3)//4
        v=np.empty(packed.size*4,dtype=np.uint8)
        for j,shift in enumerate((0,2,4,6)):v[j::4]=(packed>>shift)&3
        codes=v[:n].astype(np.int16)-1
        assert np.all((-1<=codes)&(codes<=1))
        coeff=codes.astype(np.float32)/1.5
    return coeff
def tests():
    q9=np.array([0x40,0x48],dtype=np.uint8) # -4,0,+4,0
    assert np.allclose(decode(q9,4,9),[-1,0,1,0])
    q3=np.array([0b00011000],dtype=np.uint8) # -1,1,0,-1
    assert np.allclose(decode(q3,4,3),[-2/3,2/3,0,-2/3])
def norm(s):return re.sub(r"\s+"," ",s.strip()).strip(" .,:;!?").casefold()
def grade(p,out):
    if p["category"]=="continuation":return None
    value=norm(out);target=norm(p["expected"])
    strict=value==target
    if p["category"]=="arithmetic":
        numbers=re.findall(r"(?<!\w)[+-]?\d+(?!\w)",out)
        permissive=target in numbers
    else:
        permissive=strict or bool(re.search(r"(?<!\w)"+re.escape(target)+r"(?!\w)",value))
    return {"strict":strict,"permissive":permissive}
@torch.no_grad()
def main(family):
    tests();torch.set_num_threads(4)
    cfg=CASES[family]
    raw=urllib.request.urlopen(PROMPTS,timeout=30).read();items=json.loads(raw)["prompts"]
    assert len(items)==40 and len(set(p["id"] for p in items))==40
    tokenizer=AutoTokenizer.from_pretrained(cfg["model"],revision=cfg["rev"])
    def new_model():
        m=AutoModelForCausalLM.from_pretrained(cfg["model"],revision=cfg["rev"],torch_dtype=torch.float32,low_cpu_mem_usage=True).cpu().eval()
        for param in m.parameters():
            if param.is_floating_point():param.copy_(param.bfloat16().float())
        return m
    base=new_model()
    records=[];manifest_checks=[]
    def evaluate(m,stage):
        for p in items:
            if family=="smol" and p["category"]!="continuation":
                ids=tokenizer.apply_chat_template([{"role":"user","content":p["prompt"]}],tokenize=True,add_generation_prompt=True,return_tensors="pt")
                if not torch.is_tensor(ids):
                    ids=ids["input_ids"] if hasattr(ids,"get") and ids.get("input_ids") is not None else ids
                    if not torch.is_tensor(ids):ids=torch.tensor(ids,dtype=torch.long)
                if ids.ndim==1:ids=ids.unsqueeze(0)
            else:
                prompt=p["prompt"]
                if family=="granite" and p["category"]!="continuation":prompt="Question: "+prompt+"\nAnswer:"
                ids=tokenizer(prompt,return_tensors="pt").input_ids
            assert ids.ndim==2
            out=m.generate(ids,max_new_tokens=64,do_sample=False,pad_token_id=tokenizer.eos_token_id)
            new=out[0,ids.shape[1]:].tolist();text=tokenizer.decode(new,skip_special_tokens=True)
            bg=list(zip(new,new[1:]))
            record={"family":family,"stage":stage,"id":p["id"],"category":p["category"],"output":text,"score":grade(p,text),
                "unique_bigram_ratio":len(set(bg))/len(bg) if bg else 1.0,"tokens":len(new)}
            records.append(record);print("R8_EVAL_RECORD "+json.dumps(record),flush=True)
    evaluate(base,"C0_base")
    q9=new_model();modified=0;number=0
    for name,mod in q9.named_modules():
        if not isinstance(mod,torch.nn.Linear) or name=="lm_head":continue
        w=mod.weight
        meanabs=w.detach().abs().mean(dim=1,keepdim=True).float().clamp_min(1e-7)
        init_alpha=1.5*meanabs
        raw_alpha=torch.where(init_alpha>20,init_alpha,torch.log(torch.expm1(init_alpha.clamp_min(1e-7))))
        alpha=torch.nn.functional.softplus(raw_alpha)+1e-7
        z=(w/alpha.to(dtype=w.dtype)).clamp(-0.99,0.99)
        k=torch.round(z*4)
        hard=k/6.0+torch.where(k.abs()==4,k.sign()/3.0,torch.zeros_like(k))
        weight=hard*alpha.to(dtype=w.dtype)
        assert torch.isfinite(weight).all()
        w.copy_(weight)
        modified+=1;number+=w.numel()
    assert modified==cfg["layers"] and number==cfg["weights"]
    print("R8_STEP0_Q9_READY "+json.dumps({"family":family,"layers":modified,"weights":number}),flush=True)
    evaluate(q9,"Q9_step0")
    assert len(records)==80
    summary={}
    for stage in ["C0_base","Q9_step0"]:
        rows=[r for r in records if r["stage"]==stage]
        summary[stage]={cat:{"strict":sum(bool(r["score"]["strict"]) for r in rows if r["category"]==cat),
                    "permissive":sum(bool(r["score"]["permissive"]) for r in rows if r["category"]==cat)}
                    for cat in ("arithmetic","instruction","extraction")}
        cont=[r["unique_bigram_ratio"] for r in rows if r["category"]=="continuation"]
        summary[stage]["continuation_unique_bigram_mean"]=sum(cont)/len(cont)
    print("R8_EVAL_FINAL "+json.dumps({"valid":True,"family":family,"records":len(records),
        "prompt_sha256":hashlib.sha256(raw).hexdigest(),"checkpoints":[],"step0_q9_layers":modified,"summary":summary}),flush=True)
if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--family",choices=sorted(CASES),required=True)
    main(ap.parse_args().family)
