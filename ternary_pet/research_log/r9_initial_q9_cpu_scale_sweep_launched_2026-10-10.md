# R9 initial-Q9 CPU scale/geometry diagnostic — launched

Following verified R8/R8-step0 discovery: both Granite and Smol drop from high-precision 29/30 and 24/30 respectively to **0/30** *immediately* on H Q9 quantization, without any training. We are investigating scaling/codebook without GPU training.

Source `ternary_pet/r9_cpu_initial_q9_scale_sweep.py` pinned Git commit `9ea045d69fef088f4109c3cb8c08114b2b8a02be`. Syntax/source preflight `R9_SCALE_SOURCE_PREFLIGHT_OK` HF job `6acaf47c095c578089319ca0`. **Source-only preflight, not numerical model evaluation.**

CPU jobs launched:
- Granite `6acaf484095c578089319cab`
- Smol `6acaf485fee2c900701906c0`

Variants initialized from the independently loaded same BF16-rounded pinned pretrained model; no fine-tuning, no GPU. All target linears (168 Granite / 224 Smol) uniformly quantized with original per-row alpha=1.5×meanabs and 9-state codes. Compare: H_original (alpha×1, hybrid inner k/6, extremes±1); H_scale_1p5; H_scale_2p0; W_original (alpha×1, regular wide k/4). Record full-model quantized-linear relative RMSE, RMS magnitude ratio, cosine similarity, zero/extreme code fractions, plus 15 frozen **reused diagnostic** prompts per variant (4 arithmetic, 4 instruction, 4 extraction, 3 continuation); 60 greedy samples/model. An exploratory screening study: cannot overclaim generalized capability, and prior 30-prompt baseline not directly matched because subset and repeated use.

No HF_TOKEN needed: all weights from public pinned models. Verify terminal `R9_SCALE_SWEEP_FINAL valid:true`, 60 records per family, 4 `R9_WEIGHT_DIAGNOSTICS` lines, and compare. No GPU runs planned.
