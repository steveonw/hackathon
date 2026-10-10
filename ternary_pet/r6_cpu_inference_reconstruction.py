# /// script
# dependencies = ["torch>=2.4", "transformers>=4.46", "accelerate>=1.0", "huggingface_hub>=0.30", "numpy>=1.26"]
# ///
"""CPU-only R6 compact ternary snapshot reconstruction and actual forward/generation.
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
        # Original Granite training rounded base FP32 weights to BF16 before QAT.
        # Unquantized base weights must match that original source roundtrip.
        if family=="granite":
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
            decoded=np.where(codes==2,-1,codes).astype(np.float32).reshape(shape)
            alpha=np.asarray(data["alpha_"+key],dtype=np.float32)
            assert np.all(np.isfinite(alpha)) and np.all(alpha>0)
            # Q3 original forward: hard_code / 1.5 multiplied by alpha.
            target=torch.from_numpy(decoded)*torch.from_numpy(alpha)/1.5
            module.weight.copy_(target)
            restored+=num
        assert restored==expected_count
        assert sum(counts.values())==restored
        ids=tokenizer(PROMPT,return_tensors="pt",add_special_tokens=True)["input_ids"]
        model.config.use_cache=True
        logits=model(input_ids=ids).logits
        assert torch.isfinite(logits).all()
        pred=int(logits[0,-1].argmax().item())
        generated=model.generate(input_ids=ids,max_new_tokens=12,do_sample=False,
                   pad_token_id=tokenizer.eos_token_id)
        decoded_text=tokenizer.decode(generated[0,ids.shape[-1]:],skip_special_tokens=True)
        assert generated.shape[-1]>ids.shape[-1]
        out={"family":family,"seed":seed,"source_model":model_id,"revision":revision,
           "snapshot_sha256":checksum,"restored_weights":restored,
           "layers":expected_layers,"ternary_code_counts":{str(k):v for k,v in counts.items()},
           "prompt":PROMPT,"new_tokens":int(generated.shape[-1]-ids.shape[-1]),
           "greedy_next_token_id":pred,"generated_continuation":decoded_text,
           "finite_logits":True}
        print("R6_INFERENCE_RECONSTRUCTION_OK "+json.dumps(out,ensure_ascii=True),flush=True)
        return out

def main():
    parser=argparse.ArgumentParser();parser.add_argument("--family",choices=["granite","smol"],required=True)
    args=parser.parse_args()
    token=os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:raise SystemExit("HF_TOKEN required as secret to read private snapshots")
    torch.set_num_threads(4)
    reconstruct(args.family,token)
if __name__=="__main__":main()
