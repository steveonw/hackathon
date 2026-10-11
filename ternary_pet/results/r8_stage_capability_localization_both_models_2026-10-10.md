# R8 stage capability localization — both families completed (2026-10-10)

Two CPU evaluation jobs **COMPLETED** and emitted `R8_EVAL_FINAL valid:true`: Granite HF `6acadf3d095c5780893190ec`, Smol HF `6acadf44095c5780893190ef`. Each reconstructed **four private snapshots** with verified SHA-256, model revisions, levels, shape and weight count, and emitted all 200 greedy response records (40 prompts × 5 stages); total **400 matched stage records**. Frozen R7 diagnostic prompt SHA256 `5180b1319cca0b3b65ffa01f0940d599747c9f9169bfde7d8b14d0f1dae6008b`. These questions had been used before in R7; they're **reused diagnostic prompts**, not independently held-out prospective test data.

## Corrected permissive question scores (30 prompts: 10 arithmetic, 10 instruction, 10 extraction)

| Stage | Granite | SmolLM2 |
|---|---:|---:|
| C0 high-precision BF16-rounded original | 29/30 (10/9/10) | 24/30 (10/6/8) |
| C1 native hybrid Q9 step300 | 0/30 (0/0/0) | 0/30 (0/0/0) |
| C2 projected Q3 step300 | 0/30 (0/0/0) | 0/30 (0/0/0) |
| C3 Q3 step600 | 0/30 (0/0/0) | 0/30 (0/0/0) |
| C4 Q3 step1200 | 0/30 (0/0/0) | 0/30 (0/0/0) |

The stage-C1 **capabilities are already absent** on this battery; stage Q9→Q3 loss shock is therefore **not the first cause of measured question-answering failure**. This does not show whether degradation happens at initial Q9 quantization, during the first300 Q9 updates, or due to objective/data mismatch. All H stages are quantized; no unquantized student stage was tested. It does not establish failure on every possible task.

## Greedy continuation diversity: mean unique adjacent-token pair ratio on 10 prompts (higher = more diverse)

| Stage | Granite | SmolLM2 |
|---|---:|---:|
| C0 base | 0.684600 | 0.912698 |
| C1 Q9@300 | 0.146032 | 0.326984 |
| C2 Q3@300 | 0.052381 | 0.193651 |
| C3 Q3@600 | 0.173016 | 0.316865 |
| C4 Q3@1200 | 0.100000 | 0.393651 |

Both quantized models already show low text diversity in Q9. Q3 projection further reduces this diversity, suggesting an additional effect of the transition on repetition. Smol shows partial improvement through later Q3 training but not restored answer accuracy. This metric is descriptive and does not itself measure meaningfulness.

## Experiment and technical validation

- CPU source: `ternary_pet/r8_cpu_stage_capability_eval.py`, pinned `c6c6f91bc25fbc22540bcf9f0d4997892ac24e0f`; synthetic Q9/Q3 decoder regression `R8_STAGE_EVAL_CPU_PREFLIGHT_OK`.
- Granite reconstructs **168 linear layers / 249,561,088 weights** in each archived stage; Smol **224 / 314,572,800**.
- SHA checks passed for C1/C2/C3/C4 for both; job logs report no exceptions. Inference checkpoint reconstruction uses Q9 outer-scale H mapping and Q3 signed packed codes, with original BF16-rounded nonquantized tensors.
- Original R8 stage uploads had already been read-back verified during GPU execution; this is a *separate private CPU evaluation reconstruction*.
- Earlier R6 development diagnostics revealed matched C1 Q9→C2 Q3 NLL loss shock Granite **+0.460**, Smol **+1.353**; current stage evaluation measures discrete capabilities rather than matching teacher-forced dev NLL.

## Updated hypotheses and next cost-controlled diagnostic

The next decisive question is **whether zero-step Q9 quantization already causes failure** versus failure induced during Q9 fine-tuning. Saved stage C1 is only step300, so current evidence cannot split these cases. Prefer first a zero-training **CPU-only original-vs-initial-Q9 quantizer** diagnostic (load pinned original BF16-rounded model, attach H quantizer at original initialization, no optimizer updates, evaluate on frozen prompts); if the quantizer conversion is straightforward, it avoids GPU. If step0 H-Q9 is already impaired, address initialization/scaling/coding before revisiting Q9→Q3 training; if step0 performs well and C1 fails, investigate the first300 training updates and objective. Do not launch additional GPU training without considering this cheaper experiment.
