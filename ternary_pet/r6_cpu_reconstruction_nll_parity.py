# /// script
# dependencies = ["torch>=2.4", "transformers>=4.46", "accelerate>=1.0", "huggingface_hub>=0.30", "numpy>=1.26", "datasets>=3.0"]
# ///
"""CPU-only R6 compact ternary snapshot reconstruction and actual forward/generation.
Requires private HF_TOKEN secret; downloads weights and archives, no training.
"""
import argparse, hashlib, json, os, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO="codeflash85/ternary-pet-r5-checkpoints"
CASES={
 "granite":("granite_seed170141/H_final_compact.npz","3c30ab36a6518606db74a45411b6e96639171c063f0dd2b0f6cea384581f0888",50895933,170141,168,249561088),
 "smol":("smol_seed190027/H_final_compact.npz","5d41638430f7a54fb553f2ad32d861386e5f38e42901c916fe8926172447fbfe",64195086,190027,224,314572800),
}
PROMPT="A small dog waited patiently outside the library."
REFERENCE={"granite":[{"row":264,"sha":"c70bf297646bcb3b9e33d5420923e68c9ebfc3b33b3e2bae684bfd766c282201","nll":6.4789392948150635},{"row":383,"sha":"d36575c1b1b834bb14f6a1347309760adc6ad5a6ca31ab0031aa128d3177af53","nll":6.4392149448394775},{"row":522,"sha":"5c2f352d572212e797a8db4dcd56edeea82f2cf488be8f226d3e861a25f7da35","nll":6.928325295448303},{"row":620,"sha":"b04e6f8de19daeaf67fb8f0f3f89fe634b7ea4988e61cb59d8e4d8175cfa6347","nll":6.882250428199768}],"smol":[{"row":119,"sha":"029253e074298ef33b960ecae0af4c25c087fbb637a6c8352937011d9dccbc67","nll":6.749972224235535},{"row":231,"sha":"b6898587c0be4d0b7f9e8e0e8e2939aeeaa3dab11e62bbc5a7286fe76063e4e9","nll":6.1365262269973755},{"row":264,"sha":"c70bf297646bcb3b9e33d5420923e68c9ebfc3b33b3e2bae684bfd766c282201","nll":5.878950595855713},{"row":282,"sha":"888f081936c87f274dc4e5d0b911c1064b59474e2e4a57860e3db7552bcea564","nll":5.897960782051086}]}

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
        model.config.use_cache=False
        reference=REFERENCE[family]
        target_rows={d["row"]:d for d in reference}
        observed=[]
        dataset=load_dataset("stanfordnlp/imdb",split="test",streaming=True,revision="e628166")
        for row,document in enumerate(dataset,1):
            if row>max(target_rows):break
            if row not in target_rows:continue
            text=document["text"]
            digest=hashlib.sha256((str(row)+":"+hashlib.sha256(str(text).encode("utf-8")).hexdigest()).encode("utf-8")).hexdigest()
            assert digest==target_rows[row]["sha"],("mismatched R6 document",row)
            toks=tokenizer.encode(text,add_special_tokens=False)
            assert len(toks)>=516
            total=0.0
            for k in range(4):
                window=toks[k*129:(k+1)*129]
                x=torch.tensor(window[:-1],dtype=torch.long).unsqueeze(0)
                y=torch.tensor(window[1:],dtype=torch.long).unsqueeze(0)
                logits=model(x).logits.float()
                assert torch.isfinite(logits).all()
                total+=float(F.cross_entropy(logits.reshape(-1,logits.shape[-1]),y.reshape(-1),reduction="sum").item())
            nll=total/512
            prior=target_rows[row]["nll"]
            observed.append({"row":row,"snapshot_nll":nll,"job_h_nll":prior,"delta":nll-prior})
            print("R6_PARITY_DOC "+json.dumps(observed[-1]),flush=True)
        assert len(observed)==4
        result={"family":family,"rows":[x["row"] for x in observed],
            "reconstructed_mean_nll":sum(x["snapshot_nll"] for x in observed)/4,
            "training_job_mean_nll":sum(x["job_h_nll"] for x in observed)/4,
            "mean_delta":sum(x["delta"] for x in observed)/4,
            "max_abs_doc_delta":max(abs(x["delta"]) for x in observed),
            "cpu_fp32":True,"gpu_training_evaluation_precision":"BF16" if family=="granite" else "FP16",
            "exact_numeric_parity_expected":False}
        assert all(math.isfinite(x["snapshot_nll"]) for x in observed)
        print("R6_PARITY_DIAGNOSTIC_COMPLETE "+json.dumps(result),flush=True)
        model.config.use_cache=True
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
    test_ternary_decode()
    torch.set_num_threads(4)
    reconstruct(args.family,token)
if __name__=="__main__":main()
