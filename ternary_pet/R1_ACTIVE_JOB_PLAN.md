# R1 — Range-matched nine-state Q9 controlled experiment

**Current status: R1 COMPLETED with final raw JSON and summary archived.** The original submission log and scientific design remain below. **There is one and only one R1 scientific GPU job.**

## FINAL RESULT — R1 COMPLETED 2026-10-10 UTC

[HF R1 job](https://huggingface.co/jobs/codeflash85/6ac97c30095c57808930b904) **COMPLETED**, 2026-10-10 00:05:57 UTC. **`valid_for_science=true`, 12/12 checks passed.** [Final scientific summary](results/run_r1_q9_range_matched_fineweb_seed1729_summary.md) · [complete raw JSON](results/run_r1_q9_range_matched_fineweb_seed1729_2026-10-10.json).

**FineWeb heldout NLL:** Direct Q3 **5.766660**, original wide-range Q9→Q3 **4.996255**, nine-state range-matched Q9→Q3 **5.751036**. D−wide staged advantage **+0.770405**, D−range staged advantage **+0.015624**, i.e. range-matched retains **2.03%** of the original observed benefit. WikiText validation: direct **6.558239**, wide Q9 **5.695225**, range-matched **6.566215**. Historical D/wide exact anchors reproduced on both evaluations.

**Critical caveat:** Matching nine-state output range to ternary also compresses nine-cell spacing and shifts boundaries: prep ±4 saturation **47.83%** range-matched versus **29.30%** wide, and **36.45%** of range-matched would-be integer codes were clipped. Conclude only that **this** uniformly spaced nine-state range-matched variant loses the large staged advantage; **do not ascribe causality to output range alone** or generalize beyond this seed/model/sample. No further GPU experiments launched.

**Everything after this section documents the original PRE-RUN protocol and initial queue state and is preserved historically.** For latest handoff use [CURRENT_STATE.md](CURRENT_STATE.md).

---

## Original pre-run job record

- **Hugging Face:** [codeflash85/6ac97c30095c57808930b904](https://huggingface.co/jobs/codeflash85/6ac97c30095c57808930b904). Initial stage **SCHEDULING**; A10G-small; 2-hour timeout; detached.
- **Immutable experiment source SHA:** `f28f1d4814ab0a61f2b529e2add007ce142972d5`. [R1 script](r1_fineweb_q9_range_matched_seed1729.py).
- **Original pre-run preregistration:** [R1 FineWeb range-matched Q9](research_log/r1_range_matched_q9_fineweb_seed1729_prereg_2026-10-09.md), committed **before** the GPU submission.
- **CPU quantizer-gradient/level preflight:** [HF job 6ac97b98095c57808930b8da](https://huggingface.co/jobs/codeflash85/6ac97b98095c57808930b8da), completed successfully with `R1_QUANTIZER_TEST_OK wide`, `R1_QUANTIZER_TEST_OK range_match`, `R1_CPU_PREFLIGHT_OK`. Verified all nine codes, maximum output amplitude (wide1.0α; narrow0.6666667α), integer clipping at ±4, unclipped master-weight gradient and scale-gradient STE formulas, and switch to original Q3 behavior.
- **Historical source F1:** pinned original result [F1 summary](results/run_f1_fineweb_edu_direct_vs_staged_seed1729_summary.md), [F1 raw](results/run_f1_fineweb_edu_direct_vs_staged_seed1729_2026-10-09.json). Earlier mechanism controls [G1](results/run_g1_depth_by_scale_freeze_seed1729_summary.md) and [S1](results/run_s1_fineweb_q3_sham_reset_seed1729_summary.md) valid and completed.

## Exact arms and primary comparison

Three independent-from-one-another training trajectories on the same seed **1729** and identical FineWeb-Edu `sample-10BT` QAT data/order, model teacher, objective, v10 global schedule, rowwise scales, FP32 masters and final Q3 projection:

1. `D` — all1200 steps direct original Q3; reproduces historic F1 D.
2. `S_wide` — original Q9 k=-4..4, forward levels `q=α*k/4` (max±α), Q9 steps1..300 then original-source-scale Q3 and fresh AdamW+GradScaler steps301..1200. Reproduces historic F1 staged.
3. `S_range` — **exactly nine** Q9 states with integer codes `clamp(round(6z),-4,4)`, forward `q=α*k/6` (max±2α/3, matching Q3); same Q9(300)→Q3(900) schedule/switch as S_wide.

For both Q9 variants, `z=clamp(w/α,-.99,.99)` and `q=α*(z+(hard-z).detach())`; **STE surrogate clipping rule and gradient path unchanged** through z. Range-matched Q9 clips integer code, *not z*, at ±4; this maintains nine states rather than accidentally creating thirteen. Range matching necessarily alters *state spacing and thresholds*; results assess whether Q9's wider output range is necessary, **not** an isolated range-only causal effect.

**Primary:** `NLL(S_range)-NLL(S_wide)` on the fixed FineWeb heldout 21-document slice. Also report direct-minus-range and direct-minus-wide gaps; secondary WikiText2 validation. Exact F1 direct/wide reference values and technical-reproduction tolerances in frozen prereg. All technical checks must pass before interpreting scientific outcome.

## Next AI / researcher handoff

1. Inspect HF **job `6ac97c30095c57808930b904`**. Do not duplicate, interrupt or "fix" while queued/running.
2. If terminal, fetch logs with `FINAL_JSON_BEGIN` and `FINAL_JSON_END`, parse complete JSON, and review `valid_for_science`, anchor reproductions, matched-range/wide-grid checks and full raw metrics.
3. Archive *original* `FINAL_JSON` with job provenance under `results/`, provide per-arm loss/PPL/hist/saturation, report FineWeb and WikiText gaps and exact checks. Distinguish a scientific null/negative from protocol failure.
4. Update [CURRENT_STATE.md](CURRENT_STATE.md), [AI_HANDOFF.md](AI_HANDOFF.md), [EXPERIMENT.md](EXPERIMENT.md), README and research reports. No automatic new seed, another dataset, hardware change or follow-up GPU job.
5. The small FineWeb evaluation documents may have been seen during original model pretraining; results from one seed/previously viewed evaluation are exploratory.

**R1 is the only newly paid study under this authorization.** Previous F1/F2/F3, L1, G1, S1 and other recorded results are preserved unchanged.
