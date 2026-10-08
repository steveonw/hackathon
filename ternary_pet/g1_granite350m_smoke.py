# /// script
# dependencies = ["torch>=2.4","transformers>=4.57.1","datasets>=3.0","accelerate>=1.0"]
# ///
"""
G1-0: Granite 4.0 350M architecture / quantization smoke test.

This is an engineering/audit gate only. It must not be interpreted as a
scientific D-vs-S result.

Checks:
- exact loaded class / parameter count;
- which nn.Linear weights are targeted by the established rule;
- excluded tensors and target coverage;
- embedding/lm_head tying;
- initial Q3/Q9 histograms;
- source held-out CE/PPL on a small fixed WikiText-2 slice;
- one CE35+teacher-KL65 Q3 training step;
- one CE35+teacher-KL65 Q9 training step;
- frozen non-quantized parameters;
- peak CUDA memory.
"""
import gc, json, math
from collections import Counter, defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import parametrize
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "ibm-granite/granite-4.0-350m"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 1729
SEQ = 128
EVAL_CHUNKS = 8
CE_W = 0.35
KL_W = 0.65
SMOKE_LR = 1e-4

torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

tok = AutoTokenizer.from_pretrained(MODEL_ID)
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token

def stream_chunks(split, n):
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split)
    buf, out = [], []
    for row in ds:
        t = row["text"]
        if not t or not t.strip():
            continue
        ids = tok.encode(t, add_special_tokens=False)
        if not ids:
            continue
        buf.extend(ids + [tok.eos_token_id])
        while len(buf) >= SEQ + 1 and len(out) < n:
            out.append(torch.tensor(buf[:SEQ+1], dtype=torch.long))
            buf = buf[SEQ+1:]
        if len(out) >= n:
            break
    if len(out) < n:
        raise RuntimeError(("not_enough_chunks", split, len(out), n))
    return out

train_chunks = stream_chunks("train", 2)
eval_chunks = stream_chunks("test", EVAL_CHUNKS)

def load_model(dtype):
    m = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(DEVICE)
    m.config.use_cache = False
    return m

@torch.no_grad()
def bf16_roundtrip_(m):
    for p in m.parameters():
        if p.is_floating_point():
            p.copy_(p.to(torch.bfloat16).to(p.dtype))

def target_linears(m):
    for name, mod in m.named_modules():
        if isinstance(mod, nn.Linear) and name != "lm_head":
            yield name, mod

def inv_softplus(x):
    return torch.where(x > 20, x, torch.log(torch.expm1(x.clamp_min(1e-7))))

class SharedScaleQuant(nn.Module):
    def __init__(self, w, levels):
        super().__init__()
        with torch.no_grad():
            meanabs = w.detach().abs().mean(dim=1, keepdim=True).float().clamp_min(1e-7)
            init_alpha = 1.5 * meanabs
        self.raw_alpha = nn.Parameter(inv_softplus(init_alpha))
        self.levels = int(levels)

    def alpha(self):
        return F.softplus(self.raw_alpha) + 1e-7

    def _n(self):
        if self.levels == 3:
            return 1.5
        if self.levels == 9:
            return 4.0
        raise ValueError(self.levels)

    def hard_code(self, w):
        a = self.alpha().to(device=w.device, dtype=w.dtype)
        z = (w / a).clamp(-0.99, 0.99)
        return torch.round(z * self._n()).to(torch.int8)

    def forward(self, w):
        a = self.alpha().to(dtype=w.dtype)
        n = self._n()
        z = (w / a).clamp(-0.99, 0.99)
        hard = torch.round(z * n) / n
        return (z + (hard - z).detach()) * a

def attach_quantizers(m, levels):
    for p in m.parameters():
        p.requires_grad_(False)
    qs, trainable = [], []
    for name, mod in target_linears(m):
        q = SharedScaleQuant(mod.weight, levels)
        parametrize.register_parametrization(mod, "weight", q)
        shadow = mod.parametrizations.weight.original
        shadow.requires_grad_(True)
        q.raw_alpha.requires_grad_(True)
        trainable.extend([shadow, q.raw_alpha])
        qs.append((name, mod, q))
    return qs, trainable

@torch.no_grad()
def code_hist(qs):
    counts = defaultdict(int)
    total = 0
    for _, mod, q in qs:
        w = mod.parametrizations.weight.original
        c = q.hard_code(w).reshape(-1).cpu().to(torch.int16)
        for val, n in zip(*torch.unique(c, return_counts=True)):
            counts[int(val.item())] += int(n.item())
        total += c.numel()
    return {
        "fractions": {str(k): counts[k] / max(1, total) for k in sorted(counts)},
        "total_codes": total,
        "zero_fraction": counts.get(0, 0) / max(1, total),
    }

@torch.no_grad()
def source_ce(m, chunks):
    m.eval()
    nll, nt = 0.0, 0
    for ids_cpu in chunks:
        x = ids_cpu[:-1].unsqueeze(0).to(DEVICE)
        y = ids_cpu[1:].unsqueeze(0).to(DEVICE)
        with torch.autocast("cuda", dtype=torch.float16, enabled=(DEVICE == "cuda")):
            logits = m(x).logits.float()
        ce = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            y.reshape(-1),
            reduction="sum",
        )
        nll += float(ce.item())
        nt += y.numel()
    loss = nll / nt
    return {"loss": loss, "ppl": math.exp(min(loss, 20)), "tokens": nt}

def one_step(m, teacher, trainable, ids_cpu):
    opt = torch.optim.AdamW(trainable, lr=SMOKE_LR, betas=(0.9, 0.95), weight_decay=0.0)
    scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE == "cuda"))
    x = ids_cpu[:-1].unsqueeze(0).to(DEVICE)
    y = ids_cpu[1:].unsqueeze(0).to(DEVICE)
    opt.zero_grad(set_to_none=True)
    with torch.no_grad():
        with torch.autocast("cuda", dtype=torch.float16, enabled=(DEVICE == "cuda")):
            t = teacher(x).logits
    with torch.autocast("cuda", dtype=torch.float16, enabled=(DEVICE == "cuda")):
        s = m(x).logits
        ce = F.cross_entropy(s.reshape(-1, s.shape[-1]), y.reshape(-1))
        kl = F.kl_div(
            F.log_softmax(s.float(), dim=-1),
            F.softmax(t.float(), dim=-1),
            reduction="none",
        ).sum(-1).mean()
        loss = CE_W * ce + KL_W * kl
    scaler.scale(loss).backward()
    scaler.unscale_(opt)
    grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable, 1.0).detach().cpu())
    scaler.step(opt)
    scaler.update()
    out = {
        "loss": float(loss.detach().cpu()),
        "ce": float(ce.detach().cpu()),
        "kl": float(kl.detach().cpu()),
        "grad_norm_preclip": grad_norm,
    }
    del opt, scaler
    return out

def gpu_peak_gib():
    if not torch.cuda.is_available():
        return None
    return torch.cuda.max_memory_allocated() / (1024 ** 3)

def reset_peak():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

def cleanup(*objs):
    for x in objs:
        del x
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# ----- architecture audit before parametrization -----
base = load_model(torch.float32)
bf16_roundtrip_(base)

total_params = sum(p.numel() for p in base.parameters())
named_params = dict(base.named_parameters())
targets = list(target_linears(base))
target_param_names = {name + ".weight" for name, _ in targets}
target_params = sum(mod.weight.numel() for _, mod in targets)

module_class_counts = Counter(type(mod).__name__ for _, mod in base.named_modules())
target_class_counts = Counter(type(mod).__name__ for _, mod in targets)

excluded = []
for name, p in named_params.items():
    if name not in target_param_names:
        excluded.append({
            "name": name,
            "shape": list(p.shape),
            "numel": p.numel(),
            "dtype": str(p.dtype),
        })

inp = base.get_input_embeddings()
out = base.get_output_embeddings()
tied = None
if inp is not None and out is not None and hasattr(inp, "weight") and hasattr(out, "weight"):
    tied = bool(inp.weight.data_ptr() == out.weight.data_ptr())

source_metrics = source_ce(base, eval_chunks)

architecture = {
    "model_id": MODEL_ID,
    "loaded_class": type(base).__name__,
    "config_model_type": getattr(base.config, "model_type", None),
    "architectures": getattr(base.config, "architectures", None),
    "tie_word_embeddings_config": getattr(base.config, "tie_word_embeddings", None),
    "input_output_weight_shared_storage": tied,
    "total_params": total_params,
    "target_linear_count": len(targets),
    "target_weight_params": target_params,
    "target_fraction_total_params": target_params / max(1, total_params),
    "target_names": [name for name, _ in targets],
    "target_class_counts": dict(target_class_counts),
    "module_class_counts": dict(module_class_counts),
    "excluded_parameter_tensors": excluded,
    "excluded_parameter_count": sum(row["numel"] for row in excluded),
    "source_eval": source_metrics,
}

print(json.dumps({"event": "g1_0_architecture", **architecture}), flush=True)
cleanup(base)

# One teacher instance shared by the two smoke arms.
teacher = load_model(torch.float16 if DEVICE == "cuda" else torch.float32)
teacher.requires_grad_(False)
teacher.eval()

arms = {}
for levels in (3, 9):
    reset_peak()
    m = load_model(torch.float32)
    bf16_roundtrip_(m)
    qs, trainable = attach_quantizers(m, levels)

    trainable_ids = {id(p) for p in trainable}
    unexpected_trainable = []
    frozen_nonquant_ok = True
    for name, p in m.named_parameters():
        if p.requires_grad and id(p) not in trainable_ids:
            unexpected_trainable.append(name)
        if id(p) not in trainable_ids and p.requires_grad:
            frozen_nonquant_ok = False

    before = code_hist(qs)
    step = one_step(m, teacher, trainable, train_chunks[0 if levels == 3 else 1])
    after = code_hist(qs)

    arms[f"q{levels}"] = {
        "initial_hist": before,
        "post_step_hist": after,
        "one_step": step,
        "trainable_parameter_objects": len(trainable),
        "trainable_parameter_count": sum(p.numel() for p in trainable),
        "unexpected_trainable_names": unexpected_trainable,
        "nonquantized_frozen_ok": frozen_nonquant_ok,
        "peak_cuda_gib": gpu_peak_gib(),
    }
    print(json.dumps({"event": "g1_0_arm", "levels": levels, **arms[f"q{levels}"]}), flush=True)
    cleanup(m, qs, trainable)

final = {
    "kind": "g1_0_granite350m_smoke",
    "seed": SEED,
    "device": DEVICE,
    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "architecture": architecture,
    "arms": arms,
    "scientific_result": False,
}

print("FINAL_JSON_BEGIN", flush=True)
print(json.dumps(final, indent=2), flush=True)
print("FINAL_JSON_END", flush=True)
