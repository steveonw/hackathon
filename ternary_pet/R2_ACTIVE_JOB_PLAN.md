# R2 — Q9 output-grid × code-threshold 2×2 mechanism factorial

**Live job as of 2026-10-09 EDT:** [Hugging Face `6ac985f0fee2c9007017f720`](https://huggingface.co/jobs/codeflash85/6ac985f0fee2c9007017f720). **Initial status:** SCHEDULING (not a result). **One A10G-small**, **2-hour timeout**, five sequential training arms. No duplicates or other GPU jobs authorized.

**Frozen scientific protocol:** [R2 preregistration](research_log/r2_q9_range_threshold_factorial_fineweb_seed1729_prereg_2026-10-09.md), committed **BEFORE** script/GPU execution.  
**Immutable executable commit:** `93d7aba84bff0409b8cc91603ab9605cf1a9e09d`. **Script:** [`r2_fineweb_q9_output_threshold_factorial_seed1729.py`](r2_fineweb_q9_output_threshold_factorial_seed1729.py). Source/model/dataset pins are hardcoded; GPU runs from the **raw GitHub URL containing the immutable SHA**.

**Quantizer CPU dynamic preflight:** [HF `6ac985c6095c57808930bb9c`](https://huggingface.co/jobs/codeflash85/6ac985c6095c57808930bb9c) **COMPLETED** (2026-10-10 00:25:04 UTC) with all four `R2_QA_ARM_OK` lines and `R2_PRE_GPU_PREFLIGHT_OK`. Python syntax, exactly codes -4..4 across all grids, correct amplitudes, identity of initial hard codes when threshold multiplier T is fixed and output divisor V differs, `raw_alpha` and FP32-master STE gradients, and restored original Q3 after switching were tested. On fixed test inputs, outer-code fraction for T4 = **27.09%** and T6 = **51.39%**, independent of V. These are **synthetic preflight** fractions, not actual step300 model values.

## Why

In R1, the original Q9 (range ±α, k=round(4z)) outperformed direct Q3 on 21 FineWeb-heldout documents by **+0.770405 nats/token**; uniformly range-matched Q9 (range ±2α/3, k=clip(round(6z),±4)) retained **+0.015624** and created many more outer-code assignments. R1 changed output magnitude/spacing **and** input thresholds/code saturation at the same time.

R2 separately manipulates:
- **T**: integer assignment threshold multiplier 4 (original) or 6 (tighter cells / more clipping);
- **V**: output divisor 4 (wide grid ±α) or 6 (narrow grid ±2α/3), with nine output codes either way.

At step1–300 Q9, `z=clamp(w/alpha,-0.99,0.99)`, `k=clamp(round(T*z),-4,4)`, and `q=alpha*[z+(k/V-z).detach()]`. Thus gradients use the **same z STE surrogate** for all four grids, although actual output-dependent scale gradients still differ. Step301–1200 is **identical normal Q3**, resetting original scales and AdamW/GradScaler exactly at300 for all four staged arms.

| R2 arm | V (output divisor) | T (threshold multiplier) | Max Q9 output |
|---|---:|---:|---|
| `W4T4` original | 4 | 4 | ±α |
| `W4T6` | 4 | 6 | ±α |
| `N6T4` | 6 | 4 | ±2α/3 |
| `N6T6` R1 narrow | 6 | 6 | ±2α/3 |
| `D` reference | — | — | standard Q3 all1200 |

**All five arms** use the same SmolLM2-360M-Instruct BF16 rounded source, seed1729 and identical train1200 document-derived FineWeb chunks (180 source docs), train-split dev24 chunks (3 docs), FineWeb heldout128 chunks (21 docs), WikiText2 validation128 chunks and old pinned model/dataset revisions. Same frozen FP16 teacher, 35% CE+65% KL, rowwise scale & FP32 master QAT, global LR/AdamW, unchanged nonquantized frozen modules. **One seed/same repeatedly-viewed evaluation sets**.

## Predeclared conclusions and reproducibility gate

All technical checks in the prereg must pass before interpreting science, including **R1 anchor reproduction** (D and W4T4 and N6T6 on both FineWeb and WikiText, ±0.03 NLL). Prespecified signed FineWeb contrasts:
- `N6T6−N6T4`: threshold policy effect at same narrow output.
- `W4T6−W4T4`: threshold policy effect at same wide output.
- `N6T4−W4T4`: output width/spacing effect with same T4 thresholds.
- `N6T6−W4T6`: output width/spacing effect with same T6 thresholds.
- Interaction: `(N6T6−N6T4)−(W4T6−W4T4)`.

Raw `FINAL_JSON` will report both FineWeb and WikiText validation NLL/PPL, same train-dev checkpoints, native Q9 switch diagnostics/hist/code clipping rates, AMP skips/effective updates and checks. Output width and **uniform-grid spacing cannot be separately controlled** by this factorial; threshold policy and output width/spacing are separated. Therefore avoid claiming a pure range-only causal estimate or new-dataset generalization.

## Instructions to next AI

1. Inspect **exact HF job `6ac985f0fee2c9007017f720`**. If queued/running, do not cancel, duplicate or auto-retry.
2. Upon terminal status read logs and reconstruct `FINAL_JSON_BEGIN` / `FINAL_JSON_END`. Confirm `valid_for_science`, exact historical anchor reproduction, all 5 arm outputs, five-arm 1200-step schedule, data split/order and Q9 range checks. If a check fails, mark technical comparison failure; archive even negative results.
3. Archive raw scientific JSON with HF job provenance under `ternary_pet/results/`. Write descriptive interpretation of each factorial cell/interaction and saturation rates; keep test on **same 21 FineWeb source documents**.
4. Update [CURRENT_STATE.md](CURRENT_STATE.md), [AI_HANDOFF.md](AI_HANDOFF.md), [EXPERIMENT.md](EXPERIMENT.md), README and living research reports. **No further GPUs/seeds or other designs are authorized** by the R2 request.

Prior [R1 raw](results/run_r1_q9_range_matched_fineweb_seed1729_2026-10-10.json), [R1 explanation](results/run_r1_q9_range_matched_fineweb_seed1729_summary.md), and [R1 original handoff](R1_ACTIVE_JOB_PLAN.md) remain as frozen historical references.
