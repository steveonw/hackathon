# G1-0b — Granite 4.0 350M precision diagnostic

Job: `6ac6eec9df2184ac91ac67ca`  
Pinned code: `3b217e11853ac062ef187ee163291ac6138163c5`  
Hardware: A10G-small  
Status: **PASS — BF16 path accepted for Granite G1**

## Precision probes

| Path | Source CE | Finite logits? |
|---|---:|---|
| FP32 | 3.23537 | yes |
| BF16 autocast | 3.24056 | yes |
| FP16 autocast | NaN | **no** |
| BF16 teacher | 3.23373 | yes |
| FP16 teacher | NaN | **no** |

BF16 source CE differs from FP32 by only **0.00519 nats/token** on the smoke
slice while preserving fully finite logits. FP16 produces no finite logits on
that same probe.

## One-step quantized training checks

| Arm | CE | KL | total loss | grad norm | all grads finite? |
|---|---:|---:|---:|---:|---|
| Q3 + BF16 | 21.5361 | 18.8207 | 19.7711 | 898.63 | yes |
| Q9 + BF16 | 26.6276 | 25.1715 | 25.6811 | 2995.33 | yes |

Peak CUDA memory:
- Q3 BF16 smoke: 7.73 GiB
- Q9 BF16 smoke: 9.59 GiB

## Decision

The preregistered BF16 finite-path checks all pass.

For Granite G1, use:
- FP32 persistent master weights;
- the same BF16-rounded common source treatment as the established protocol;
- **BF16 autocast** for student/teacher forward compute;
- BF16 teacher weights;
- no FP16 GradScaler requirement; standard backward with gradient clipping;
- `use_cache=False`.

The earlier G1-0 NaNs are classified as a **Granite FP16 numerical
incompatibility**, not a quantization failure and not a scientific result.

G1-1 direct-Q3 schedule calibration is now unblocked.
