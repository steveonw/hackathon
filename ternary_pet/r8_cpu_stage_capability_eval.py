# /// script
# dependencies = ["torch>=2.4","transformers>=4.57.1","accelerate>=1.0","huggingface_hub>=0.30","numpy>=1.26"]
# ///
"""R8 CPU stage-localization inference: pinned base and four verified R8 stages."""
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
    q9=np.array([0x04,0x08],dtype=np.uint8) # -4,0,+4,0
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
    token=os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:raise RuntimeError("private HF_TOKEN secret required")
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
    for i,stage in enumerate(STAGES):
        remote=f'r8_stage_localization/{cfg["folder"]}/R8_H_{stage}.npz'
        file=hf_hub_download(repo_id=REPO,repo_type="dataset",filename=remote,token=token)
        with open(file,"rb") as f:sha=hashlib.file_digest(f,"sha256").hexdigest()
        assert sha==cfg["sha"][i],(stage,"sha mismatch")
        m=new_model();modules=dict(m.named_modules());total=0
        with np.load(file,allow_pickle=False) as z:
            mf=json.loads(z["manifest_utf8"].tobytes())
            level=9 if i==0 else 3
            assert mf["levels"]==level and mf["stage"]==stage
            assert mf["family"]==cfg["model"] and mf["model_revision"]==cfg["rev"]
            assert len(mf["layers"])==cfg["layers"] and mf["quantized_weights"]==cfg["weights"]
            for layer in mf["layers"]:
                mod=modules[layer["module"]];shape=tuple(layer["shape"]);n=int(layer["count"])
                assert tuple(mod.weight.shape)==shape and layer["levels"]==level
                coeff=decode(z["codes_"+layer["key"]],n,level).reshape(shape)
                alpha=np.asarray(z["alpha_"+layer["key"]],dtype=np.float32)
                assert np.isfinite(alpha).all() and (alpha>0).all()
                mod.weight.copy_(torch.from_numpy(coeff)*torch.from_numpy(alpha))
                total+=n
            assert total==cfg["weights"]
        manifest_checks.append({"stage":stage,"levels":level,"sha256":sha,"weights":total})
        print("R8_EVAL_RECONSTRUCTED "+json.dumps(manifest_checks[-1]),flush=True)
        evaluate(m,stage)
        del m
    assert len(records)==200
    summary={}
    for stage in ["C0_base"]+STAGES:
        rows=[r for r in records if r["stage"]==stage]
        summary[stage]={cat:{"strict":sum(bool(r["score"]["strict"]) for r in rows if r["category"]==cat),
                    "permissive":sum(bool(r["score"]["permissive"]) for r in rows if r["category"]==cat)}
                    for cat in ("arithmetic","instruction","extraction")}
        cont=[r["unique_bigram_ratio"] for r in rows if r["category"]=="continuation"]
        summary[stage]["continuation_unique_bigram_mean"]=sum(cont)/len(cont)
    print("R8_EVAL_FINAL "+json.dumps({"valid":True,"family":family,"records":len(records),
        "prompt_sha256":hashlib.sha256(raw).hexdigest(),"checkpoints":manifest_checks,"summary":summary}),flush=True)
if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--family",choices=sorted(CASES),required=True)
    main(ap.parse_args().family)
