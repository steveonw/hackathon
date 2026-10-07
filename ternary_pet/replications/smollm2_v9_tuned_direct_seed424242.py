# /// script
# dependencies = ["torch>=2.4","transformers>=4.46","datasets>=3.0","accelerate>=1.0"]
# ///
"""
SmolLM2 direct-Q3 tuned-schedule replication.

Purpose:
Replicate the v9-selected direct-Q3 schedule on an independent v7 training
order without any new hyperparameter selection.

Schedule:
- 1200 direct Q3 updates;
- 100-step linear warmup to 1e-3;
- cosine decay to 1e-4 at step 1200.

Everything else matches the v7/v8/v9 setup. The held-out test set is used only
for final evaluation.
"""
import gc, json, math, time
from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import parametrize
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

MODEL_ID="HuggingFaceTB/SmolLM2-360M-Instruct"
DEVICE="cuda" if torch.cuda.is_available() else "cpu"
SEED=424242
SEQ=128
TRAIN_CHUNKS=1200
LRVAL_CHUNKS=24
EVAL_CHUNKS=64
LR_
CFG={"name":"warm100_cosine_1e-3","kind":"warmup_cosine",
     "peak_lr":1e-3,"warmup_steps":100,"min_lr":1e-4}
FULL_STEPS=1200

def lr_for(step0):
    warm=CFG["warmup_steps"]
    peak=CFG["peak_lr"]; floor=CFG["min_lr"]
    s=step0+1
    if s<=warm:
        return peak*s/max(1,warm)
    frac=(s-warm)/max(1,FULL_STEPS-warm)
    frac=min(max(frac,0.0),1.0)
    return floor+0.5*(peak-floor)*(1.0+math.cos(math.pi*frac))

def set_optimizer_lr(opt,lr):
    for g in opt.param_groups:
        g["lr"]=float(lr)

print(json.dumps({
    "event":"rep_start","seed":SEED,"model":MODEL_ID,"device":DEVICE,
    "gpu":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "schedule":CFG,"order_head":_order[:16]
}),flush=True)

teacher=load_model(torch.float16 if DEVICE=="cuda" else torch.float32)
teacher.requires_grad_(False)

base=load_model(torch.float32); bf16_roundtrip_(base)
bf16_test=score_on(base,teacher,eval_chunks)
del base; gc.collect()
if torch.cuda.is_available(): torch.cuda.empty_cache()

m=load_model(torch.float32); bf16_roundtrip_(m)
qs,trainable=attach_quantizers(m,3)
stage_start=snapshot_codes(qs)
prev={k:v.clone() for k,v in stage_start.items()}
opt=make_optimizer(trainable,CFG["peak_lr"])
scaler=torch.amp.GradScaler("cuda",enabled=(DEVICE=="cuda"))
trace=[]
start=time.time()

for step in range(FULL_STEPS):
    lr=lr_for(step)
    set_optimizer_lr(opt,lr)
    loss,ce,kl=one_step(m,teacher,opt,scaler,train_chunks[step])
    if not (math.isfinite(loss) and math.isfinite(ce) and math.isfinite(kl)):
        raise RuntimeError(("nonfinite",step+1,loss,ce,kl))
    if step==0 or (step+1)%100==0 or step+1==FULL_STEPS:
        prev,fs=flip_stats(qs,prev,stage_start)
        val=score_on(m,teacher,lrval_chunks) if (step+1 in [300,600,900,1200]) else None
        row={"step":step+1,"lr":lr,"loss":loss,"ce":ce,"kl":kl,
             "validation":val,**fs}
        trace.append(row)
        print(json.dumps({"event":"rep_train",**row}),flush=True)

test=score_on(m,teacher,eval_chunks)
test["gen"]=generate(m)
test["final_hist"]=code_hist(qs)
test["scale"]=scale_stats(qs)

results={
    "seed":SEED,
    "schedule":CFG,
    "training_order_head":_order[:16],
    "bf16_source_test":bf16_test,
    "validation_final":score_on(m,teacher,lrval_chunks),
    "test":test,
    "trace":trace,
    "seconds":time.time()-start
}

print("FINAL_JSON_BEGIN",flush=True)
print(json.dumps(results),flush=True)
print("FINAL_JSON_END",flush=True)
