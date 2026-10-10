# /// script
# dependencies = ["torch>=2.4", "transformers>=4.46", "accelerate>=1.0", "huggingface_hub>=0.30", "numpy>=1.26"]
# ///
"""CPU-only R6 pinned base vs compact hybrid generation controls.
Requires private HF_TOKEN secret; downloads weights and archives, no training.
"""
import argparse, hashlib, json, os
import numpy as np
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO="codeflash85/ternary-pet-r5-checkpoints"
CASES={
 "granite":("granite_seed170141/H_final_compact.npz","3c30ab36a6518606db74a45411b6e96639171c063f0dd2b0f6cea384581f0888",50895933,170141,168,249561088),
 "smol":("smol_seed190027/H_final_compact.npz","5d41638430f7a54fb553f2ad32d861386e5f38e42901c916fe8926172447fbfe",64195086,190027,224,314572800),
}
PROMPT="A small dog waited patiently outside the library."
TASKS=[("continuation","The scientist opened the notebook and wrote"),("explanation","Explain in one short sentence why a baseline is useful in an experiment."),("arithmetic","What is 17 + 28? Answer with the number.")]

def ternary_decode(values):
    # NumPy 2 rejects negative Python integers in unsigned np.where branches.
    signed=values.astype(np.int8)
    return np.where(signed==2,-1,signed).astype(np.float32)

def test_ternary_decode():
    sample=np.array([0,1,2,0,2,1],dtype=np.uint8)
    assert ternary_decode(sample).tolist()==[0.0,1.0,-1.0,0.0,-1.0,1.0]

@torch.no_grad()
def reconstruct(family, token):
    remote,expected_sha,size,seed,expected_layers,expected_count=CASES[family]
    filename=hf_hub_download(repo_id=REPO,repo_type="dataset",filename=remote,token=token)
    with open(filename,"rb") as f:
        checksum=hashlib.file_digest(f,"sha256").hexdigest()
    assert checksum==expected_sha and os.path.getsize(filename)==size
    with np.load(filename,allow_pickle=False) as data:
        manifest=json.loads(data["manifest_utf8"].tobytes().decode())
        assert manifest["seed"]==seed
        assert len(manifest["layers"])==expected_layers
        assert manifest["quantized_weights"]==expected_count
        model_id=manifest["family"]; revision=manifest["model_revision"]
        print("R6_INFERENCE_LOADING",family,model_id,flush=True)
        tokenizer=AutoTokenizer.from_pretrained(model_id,revision=revision)
        model=AutoModelForCausalLM.from_pretrained(
            model_id,revision=revision,torch_dtype=torch.float32,low_cpu_mem_usage=True
        ).cpu().eval()
        # Both R6 families BF16-rounded FP32 base weights before QAT.
        # Untouched nonquantized parameters require the same source roundtrip.
        for param in model.parameters():
            if param.is_floating_point():
                param.copy_(param.to(torch.bfloat16).float())
        modules=dict(model.named_modules())
        restored=0; counts={-1:0,0:0,1:0}
        for entry in manifest["layers"]:
            name=entry["module"]; key=entry["key"]
            module=modules[name]
            assert isinstance(module,nn.Linear) and name!="lm_head",name
            shape=tuple(entry["shape"]); num=int(entry["count"])
            assert tuple(module.weight.shape)==shape and num==module.weight.numel()
            packed=data["codes_"+key]
            assert len(packed)==(num+3)//4
            values=np.empty(len(packed)*4,dtype=np.uint8)
            for i,shift in enumerate((0,2,4,6)):
                values[i::4]=(packed>>shift)&3
            codes=values[:num]
            assert not np.any(codes==3),name
            for k,n in ((-1,2),(0,0),(1,1)):
                counts[k]+=int(np.count_nonzero(codes==n))
            decoded=ternary_decode(codes).reshape(shape)
            alpha=np.asarray(data["alpha_"+key],dtype=np.float32)
            assert np.all(np.isfinite(alpha)) and np.all(alpha>0)
            # Q3 original forward: hard_code / 1.5 multiplied by alpha.
            target=torch.from_numpy(decoded)*torch.from_numpy(alpha)/1.5
            module.weight.copy_(target)
            restored+=num
        assert restored==expected_count
        assert sum(counts.values())==restored
        base=AutoModelForCausalLM.from_pretrained(model_id,revision=revision,torch_dtype=torch.float32,low_cpu_mem_usage=True).cpu().eval()
        for p in base.parameters():
            if p.is_floating_point():p.copy_(p.to(torch.bfloat16).float())
        for task,prompt in TASKS:
            if family=="smol" and task!="continuation":
                ids=tokenizer.apply_chat_template([{"role":"user","content":prompt}],tokenize=True,add_generation_prompt=True,return_tensors="pt")
                fmt="chat_template"
            else:
                ids=tokenizer(prompt,return_tensors="pt").input_ids
                fmt="plain"
            for name,m in (("base",base),("hybrid",model)):
                m.config.use_cache=True
                logits=m(ids).logits[0,-1].float()
                assert torch.isfinite(logits).all()
                probs=torch.softmax(logits,dim=-1)
                for method in ("greedy","sample"):
                    args={"max_new_tokens":24,"do_sample":method=="sample","pad_token_id":tokenizer.eos_token_id}
                    if method=="sample":
                        torch.manual_seed(20261010)
                        args.update({"temperature":0.8,"top_p":0.9})
                    output=m.generate(ids,**args)
                    print("R6_GENERATION_CONTROL "+json.dumps({"family":family,"task":task,"prompt_format":fmt,"model":name,"decoding":method,"continuation":tokenizer.decode(output[0,ids.shape[-1]:],skip_special_tokens=True),"tokens":int(output.shape[-1]-ids.shape[-1]),"top1_prob":float(probs.max()),"finite_logits":True}),flush=True)
        print("R6_GENERATION_CONTROL_COMPLETE "+json.dumps({"family":family,"seed":seed,"cases":12}),flush=True)
        return True

def main():
    parser=argparse.ArgumentParser();parser.add_argument("--family",choices=["granite","smol"],required=True)
    args=parser.parse_args()
    token=os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:raise SystemExit("HF_TOKEN required as secret to read private snapshots")
    test_ternary_decode()
    torch.set_num_threads(4)
    reconstruct(args.family,token)
if __name__=="__main__":main()
