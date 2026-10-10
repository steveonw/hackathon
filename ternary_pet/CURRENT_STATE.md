# Ternary Pet — CURRENT STATE

**Authoritative short handoff:** As of 2026-10-08 Eastern / 2026-10-09 UTC.  
**Repository:** `steveonw/hackathon` → `ternary_pet/`.  
**Status:** **R1 completed, 12/12 technical checks passed; see newest R1 outcome below.** G1/S1, F1/F2/F3 and L1 are completed. No additional scientific GPU job launched.

## LATEST: R1 Q9 range-match COMPLETED — full staged gain largely lost (2026-10-10 UTC)

**R1 GPU COMPLETED 2026-10-10 00:05:57 UTC (Oct 9 EDT)**: [HF 6ac97c30095c57808930b904](https://huggingface.co/jobs/codeflash85/6ac97c30095c57808930b904), A10G-small, script SHA `f28f1d4814ab0a61f2b529e2add007ce142972d5`, **all 12/12 technical checks passed, `valid_for_science=true`**. [R1 detailed results](results/run_r1_q9_range_matched_fineweb_seed1729_summary.md) · [full raw JSON](results/run_r1_q9_range_matched_fineweb_seed1729_2026-10-10.json) · [frozen preregistration](research_log/r1_range_matched_q9_fineweb_seed1729_prereg_2026-10-09.md) · [job handoff](R1_ACTIVE_JOB_PLAN.md).

**Main question:** does a nine-state Q9 prepare useful ternary Q3 assignments when its output range is reduced from max ±α (original Q9) to max ±2α/3 (same as Q3), without altering the normalized STE z clamp and with integer codes constrained to exactly -4..4? **FineWeb held-out NLL:** direct Q3 **5.766660**; original/wide Q9(300)→Q3(900) **4.996255**; **range-matched nine-state Q9→Q3 5.751036**. Original wide staged gain **+0.770405**, range-matched staged gain **+0.015624** nats/token, retaining **2.03%** of the original. Secondary WikiText validation: direct **6.558239**, wide **5.695225**, range-matched **6.566215** (range-matched marginally *worse* than direct). Both original D and wide-Q9 anchors reproduced prior F1 exactly on both evaluations.

**New prep diagnostic:** at Q9 step300 **47.83%** of range-matched Q9 codes are outer ±4, versus **29.30%** wide Q9; **36.45%** of would-be `round(6z)` codes exceed code magnitude4 before integer clipping. Range-matched native Q9 train-dev NLL **6.6035** vs wide **5.4282**. Wide→narrow matches output range while necessarily changing nine-state spacing, decision boundaries and saturation/occupancy. **Do not assert “range alone has been proven to be the cause”**; R1 refutes the narrower hypothesis that any uniformly spaced nine-state Q9 grid will preserve the stage advantage. It indicates **wider effective range and/or its coupled code geometry is critical for this tested recipe**, not that all alternative range-matched nine-state schemes fail.

**Technical/source provenance:** [R1 CPU preflight](https://huggingface.co/jobs/codeflash85/6ac97b98095c57808930b8da) passed exact nine reachable codes, max output amplitude and master/alpha surrogate derivatives. R1 GPU 3 arms all1200 scheduled steps, final Q3; F1 source, seed1729, same FineWeb 21 heldout docs and WikiText validation. **One seed, reused small sample, no production inference/generation claim**. No other GPU study was initiated in the R1 analysis. G1 and S1 remain completed with prior positive controls; all former historical experiments remain archived.

**Next unrun scientific work (proposal only):** separate nine-state range restriction from code-cell spacing and saturation in a preregistered control, or evaluate genuinely fresh heldout documents/model family. Do not automatically launch a new GPU job. Older “scheduling” status descriptions below are historical and superseded.

---

## HISTORICAL R1 SUBMISSION — Q9 range-matched to Q3 (2026-10-09 UTC)

**ONE bounded scientific GPU job submitted:** [HF 6ac97c30095c57808930b904](https://huggingface.co/jobs/codeflash85/6ac97c30095c57808930b904), A10G-small, max2h, initial stage SCHEDULING. **No R1 scientific result available at submission.** Immutable script SHA `f28f1d4814ab0a61f2b529e2add007ce142972d5`; [R1 source](r1_fineweb_q9_range_matched_seed1729.py). [Frozen pre-run R1 prereg](research_log/r1_range_matched_q9_fineweb_seed1729_prereg_2026-10-09.md). [Full active job plan/handoff](R1_ACTIVE_JOB_PLAN.md).

This controls the **representable output range** question from the external review: FineWeb seed1729 1200-step `D` direct Q3, `S_wide` original Q9(300)→Q3(900) with nine values reaching ±α, and `S_range` nine-state Q9(300)→Q3(900) with nine values reaching only ±2α/3, then identical Q3 continuation. Same Q3 source row-scale restoration and AdamW/GradScaler reset at step300 for S_wide and S_range. Same `z=clamp(w/alpha,-0.99,0.99)` and STE gradient path; range-matched clips **integer code after round(6z)** to ±4, not normalized z, so code set is exactly nine. Width matching also moves intermediate state spacing/threshold locations, so **not a perfect pure range ablation**.

**Pre-run CPU dynamic gradient/code QA [6ac97b98095c57808930b8da](https://huggingface.co/jobs/codeflash85/6ac97b98095c57808930b8da)** COMPLETED. It printed `R1_QUANTIZER_TEST_OK` for both wide/range variants and `R1_CPU_PREFLIGHT_OK`: 9 states, max output amplitude, STE master-weight/scale gradients on unclipped inputs, Q3 switch behavior, script AST. The R1 three-arm GPU job has not been completed or interpreted yet. Historical G1/S1 findings remain valid and archived; **do not launch new seeds, repeat R1, or launch other jobs without user approval**.

---

## HISTORICAL COMPLETED RESULTS — G1 and S1 (2026-10-09 UTC)

**Both mechanism-control HF jobs COMPLETED with all prespecified scientific-technical checks passing. No active scientific GPU jobs were found in this pair.** [G1 full summary](results/run_g1_depth_by_scale_freeze_seed1729_summary.md) · [G1 raw JSON](results/run_g1_depth_by_scale_freeze_seed1729_2026-10-09.json) · [S1 full summary](results/run_s1_fineweb_q3_sham_reset_seed1729_summary.md) · [S1 raw JSON](results/run_s1_fineweb_q3_sham_reset_seed1729_2026-10-09.json). Both planned jobs were authorized and executed once each; **R1 range-matched Q9 remains UNRUN**.

**G1 — Q3 within-bin depth vs trainability of row scales:** [HF job 6ac96a7efee2c9007017ea64](https://huggingface.co/jobs/codeflash85/6ac96a7efee2c9007017ea64), source SHA `059a0bb91aa55fea99ed56b1d0d980eb21fd31ae`, completed **2026-10-09 23:16:07 UTC**, **6/6 checks passed**. Four 900-step continuations seeded from identically projected Q3 forward states, with 0.03 vs 0.50 in-bin depth on the D-vs-Q9 projected-code disagreement mask, learned vs exactly frozen original Q3 row scales:

| Scale policy | Depth 0.03 final WikiText test NLL | Depth 0.50 final WikiText test NLL | Shallow-minus-deep improvement |
|---|---:|---:|---:|
| Learned row scales | 5.494818 | **4.884970** | +0.609848 |
| Frozen row scales | 5.516548 | **4.896080** | +0.620468 |

**Primary frozen−learned depth-gain interaction = +0.010620 nats/token**. Every arm had 898 successful Q3 updates and 2 AMP skips, initial code projection/forward score identical. Frozen scales remained numerically unchanged and optimizer-excluded. **Conclusion: trainable Q3 row scales are NOT necessary for the large depth effect in this recipe**; stronger support for interior placement/assignment stability, without proving exact causal mediation or removing Q9 range confound.

**S1 — FineWeb Q3 sham switch:** [HF job 6ac96a82fee2c9007017ea6b](https://huggingface.co/jobs/codeflash85/6ac96a82fee2c9007017ea6b), source SHA `4cdab8ee73ddd804e96fac43992f91ab9a54c1b9`, completed **2026-10-09 23:08:23 UTC**, **9/9 checks passed**. Same seed1729 public FineWeb 1200-step corpus, document partition and LR. Direct Q3 continuous NLL **5.766660**; direct Q3 with step300 original-scale + AdamW/GradScaler reset NLL **5.789199**, **0.022539 worse**; archived Q9(300)→Q3(900) NLL **4.996255**, **0.792944 better than sham**. WikiText validation likewise sham slightly worse (6.560840 vs6.558239) while Q9 staged 5.695225. The direct-Q3 continuous anchor reproduced original F1 exactly. **Conclusion: the reset alone does not explain the FineWeb Q9 staging gain** in this seed. The data do not prove resets have no interactions with Q9 prep.

**Next research gate:** implement a fully **range-matched nine-state Q9 versus historical wide Q9** with exact clipping/STE gradients and integer-code checks (see [technical review](research_log/2026-10-09_external_technical_review_scale_range_sham_controls.md)). Then boundary-margin-matched site controls and common-original-scale final-code readouts; fresh-document evaluation and retained checkpoints. **No R1 job launched**. [Original G1/S1 frozen protocols and job information](G1_S1_ACTIVE_JOB_PLAN.md).

---

## HISTORICAL AT SUBMISSION — G1 and S1 mechanism-control jobs (2026-10-09 EDT)

**Two bounded GPU jobs submitted**, with preregistrations committed before launch; **results not available at submission**:

- **G1 depth × scale trainability**: [HF job 6ac96a7efee2c9007017ea64](https://huggingface.co/jobs/codeflash85/6ac96a7efee2c9007017ea64), one A10G-small max **2h**, pinned Git SHA `059a0bb91aa55fea99ed56b1d0d980eb21fd31ae`, [G1 source](g1_depth_by_scale_freeze_seed1729.py). Seed1729 WikiText familiar v13A preparation, four Q3 continuation arms d003/d050 × learned/frozen Q3 row scales, common global 900-step continuation. [G1 frozen prereg](research_log/g1_frozen_vs_learned_q3_scale_depth_prereg_2026-10-09.md).
- **S1 FineWeb Q3 sham-switch**: [HF job 6ac96a82fee2c9007017ea6b](https://huggingface.co/jobs/codeflash85/6ac96a82fee2c9007017ea6b), one A10G-small max **90m**, pinned Git SHA `4cdab8ee73ddd804e96fac43992f91ab9a54c1b9`, [S1 source](s1_fineweb_q3_sham_reset_seed1729.py). Seed1729 one original direct-Q3 continuous arm and one Q3 step300 sham Adam/GradScaler/original-scale reset, with unchanged FineWeb QAT document partition/1200-step LR and historical Q9 comparison. [S1 frozen prereg](research_log/s1_fineweb_direct_q3_sham_reset_seed1729_prereg_2026-10-09.md).
- **Static preflight:** HF CPU job `6ac96a6f095c57808930b267` emitted `G1_S1_STATIC_PREFLIGHT_OK`; script Python AST and invariants checked.

**Full permanent handoff:** [G1_S1_ACTIVE_JOB_PLAN.md](G1_S1_ACTIVE_JOB_PLAN.md). **Do not relaunch these jobs, change scripts, launch additional seeds, or start R1 range-matched Q9 until new user authorization.** Inspect job statuses and exact `FINAL_JSON`, archive raw findings even nulls/technical failures, then update research reports. F1/F2/F3 and L1 historical findings stay completed and preserved.

---

## HISTORICAL VERIFIED — F1/F2/F3 and L1 COMPLETED (2026-10-09 UTC)

**No currently active scientific GPU jobs in the verified HF job list.** The four most recent studies are completed (F1 seed1729, F2 seed271828, F3 seed424242, L1 WikiText 6000 steps). Each finished successfully with all preregistered technical checks passing. No new GPU trials were launched in this archival/review action.

**FineWeb-Edu three-seed exact-recipe replication:** staged Q9(300)→Q3(900) beat direct Q3(1200) on FineWeb heldout and WikiText validation in **3/3 seeds**. Mean paired FineWeb D−S loss gap **+0.763898 nats/token**, range **+0.742353 to +0.778936**; mean WikiText validation gap **+0.801917**, range +0.696902 to +0.863013. [Aggregate analysis](results/run_f1_f3_fineweb_edu_three_seed_aggregate_summary.md); [F1 raw](results/run_f1_fineweb_edu_direct_vs_staged_seed1729_2026-10-09.json); [F2 raw](results/run_f2_fineweb_edu_direct_vs_staged_seed271828_2026-10-09.json); [F3 raw](results/run_f3_fineweb_edu_direct_vs_staged_seed424242_2026-10-09.json). **Same 21 FineWeb heldout docs for all three orders.** This is RNG/order replication, *not* independent-dataset or causal-mechanism replication. F2/F3 HF job IDs [F2](https://huggingface.co/jobs/codeflash85/6ac87415fee2c90070173baf) and [F3](https://huggingface.co/jobs/codeflash85/6ac87417095c5780893020c2).

**L1 6000-step WikiText durability COMPLETE:** [HF job](https://huggingface.co/jobs/codeflash85/6ac86e1f095c578089301e53). Paired fixed-recipe direct-Q3 versus Q9→Q3, with WikiText validation staged gap **+0.716087 nats at1200** and **+0.616040 nats at6000**; direct has **not caught up at6000**. Both historical step1200 tests reproduced exactly; all five checks passed. One seed, gap narrowing but nonmonotonic; no asymptotic proof. [L1 detailed summary](results/run_l1_6000step_wikitext_durability_seed1729_summary.md) · [raw JSON](results/run_l1_6000step_wikitext_durability_seed1729_2026-10-09.json).

**Important new independent methodological review:** [six objections and matched experimental controls](research_log/2026-10-09_external_technical_review_scale_range_sham_controls.md). Highest priority for *mechanism*, before more indiscriminate GPU scaling: within-bin depth with **fixed versus trainable Q3 row scales**, and direct-Q3 **sham step300 reset** on FineWeb, then Q9 **range-matched** vs wide, matched-margin position controls, final-code same-reference diagnostics and document-linked checkpoint evidence. These controls are **proposed only**; no new GPU jobs have been launched.

**Next AI:** use this section as current. Historical submission statuses below are preserved but superseded. Before designing new work, freeze protocol, document source SHA, and budget; avoid claiming the present 3/3-seed effect independently confirms original-pretraining dataset generalization.

---

## HISTORICAL AT SUBMISSION: FineWeb-Edu F2/F3 seed replications (2026-10-09 UTC)

**User-authorized two new-seed GPU jobs, now RUNNING:**

- **F2, seed 271828:** [F2 seed271828](https://huggingface.co/jobs/codeflash85/6ac87415fee2c90070173baf); A10G-small, 90-minute maximum; immutable source SHA `64d171a225ba2781c83148a4235272c91939f9c9`, [script](f2_fineweb_edu_qat_seed271828.py).
- **F3, seed 424242:** [F3 seed424242](https://huggingface.co/jobs/codeflash85/6ac87417095c5780893020c2); A10G-small, 90-minute maximum; immutable source SHA `a20c4cc66ae16026bb90969d1160294b3f5826e9`, [script](f3_fineweb_edu_qat_seed424242.py).

[F2/F3 frozen preregistration](research_log/f2_f3_fineweb_edu_seed271828_424242_replication_prereg_2026-10-09.md) was committed **before compute**. HF CPU static smoke `6ac87407fee2c90070173b9d` emitted `F2_F3_PREFLIGHT_OK`; normalized source-code comparison passed: only seed, seed-expected permutation assertion, and provenance identifiers changed from the successful F1 seed1729 script. Exact same source-model/dataset SHAs, same FineWeb QAT document-based train/dev/eval partition and same 1200-step matched arms. **No F2/F3 scientific outcomes available yet.**

**Original F1 seed1729 result (already completed):** [summary](results/run_f1_fineweb_edu_direct_vs_staged_seed1729_summary.md) and [raw](results/run_f1_fineweb_edu_direct_vs_staged_seed1729_2026-10-09.json): staged improves held-out FineWeb NLL by **0.770405 nats** and WikiText validation NLL by **0.863013 nats**. Three-seed aggregate must wait for F2 and F3 `FINAL_JSON` and scientific checks. The heldout 21 FineWeb documents are **shared across seeds**, not independent datasets.

**Also still RUNNING at last check:** [L1 6000-step WikiText durability](https://huggingface.co/jobs/codeflash85/6ac86e1f095c578089301e53). **Do not cancel or resubmit L1.** There are exactly two *new* scientific GPU jobs in this replication request.

**Next AI:** inspect these three HF job IDs, never duplicate them. For F2/F3 parse `FINAL_JSON_BEGIN`/`FINAL_JSON_END`; verify all five checks and source-doc partition hashes match F1. Archive per-seed raw JSON, summary and three-seed aggregate (mean/median/min-max/positive count). Clearly label whether any failure was technical versus a negative staged effect; update this page, `AI_HANDOFF.md`, `EXPERIMENT.md`, README and both research reports. No additional GPU or data-search sweep authorized.

---

## HISTORICAL AT SUBMISSION: L1 WikiText running and F1 completed

**Last checked 2026-10-09 UTC:** **L1 still RUNNING** on Hugging Face: [L1 6ac86e1f095c578089301e53](https://huggingface.co/jobs/codeflash85/6ac86e1f095c578089301e53), A10G-small, 2-hour maximum, pinned script SHA `841e0ab948991a24b7397b34d90d28c38f749879`. Matched direct Q3 versus Q9(300)→Q3 extended to **6000** WikiText training opportunities with new QAT training chunks and an independent WikiText validation curve. **L1 scientific outcome remains pending. Do not interrupt or duplicate this job.**

**F1 FineWeb-Edu COMPLETE**: [F1 6ac86eb0fee2c900701738dd](https://huggingface.co/jobs/codeflash85/6ac86eb0fee2c900701738dd), completed **2026-10-09 04:48:30 UTC**, A10G-small, pinned script SHA `64dcd7f20240b4c62e6ecac8df70a1336a57bb14`. All five frozen checks passed (`valid_for_science=true`). [F1 detailed summary](results/run_f1_fineweb_edu_direct_vs_staged_seed1729_summary.md) · [raw JSON](results/run_f1_fineweb_edu_direct_vs_staged_seed1729_2026-10-09.json) · [frozen protocol](research_log/f1_fineweb_edu_qat_seed1729_prereg_2026-10-09.md).

**F1 primary FineWeb-Edu document-held-out NLL:** D direct Q3(1200) **5.766660** (PPL319.469) versus S300 Q9(300)→Q3(900) **4.996255** (PPL147.858), staged advantage **+0.770405 nats/token**. **Secondary WikiText-2 validation:** D **6.558239** vs S **5.695225**, staged advantage **+0.863013 nats/token**. F1 trained on public `FineWeb-Edu sample-10BT`, an ingredient of SmolLM2's pretraining mix, not the full original mixture. QAT document-ID sets were disjoint (train **180 docs /1200 chunks**, dev **3 docs/24 chunks**, held-out **21 docs/128 chunks**). Only one seed/order1729; possible prior source-model pretraining exposure to sampled documents unknown. Do not claim broad independent generalization, a production model, or zero contamination.

**L1/F1 frozen protocols:** [L1](research_log/l1_6000_step_durability_wikitext_seed1729_prereg_2026-10-09.md) · [F1](research_log/f1_fineweb_edu_qat_seed1729_prereg_2026-10-09.md). **Durable current job plan:** [NEXT_JOBS_PLAN.md](NEXT_JOBS_PLAN.md). The older M1 and T1/C1 jobs are all completed. No additional GPU jobs launched for F1 analysis.

**Next AI:** inspect L1, not F1, for pending scientific results. At completion fetch its `FINAL_JSON` and checks, archive all outcomes and update this state. Do not resubmit F1, silently change hyperparameters, or launch extra seeds without a new decision.

---

## Latest completed experiment — M1 matched Q3 continuation

**Status:** COMPLETED, **2026-10-09 04:15:27 UTC**. [Hugging Face job `6ac86712fee2c900701734bb`](https://huggingface.co/jobs/codeflash85/6ac86712fee2c900701734bb); pinned script `cad64c009209a924be89b523e6a1e184a4f4137b`, `m1_smol360m_equal_q3_continuation_seed1729.py`. Seven of seven preregistered construction/reproduction checks passed (`valid_for_science=true`). [M1 detailed result](results/run_m1_equal_q3_continuation_seed1729_summary.md) · [raw JSON](results/run_m1_equal_q3_continuation_seed1729_2026-10-09.json) · [frozen preregistration](research_log/m1_equal_q3_continuation_seed1729_prereg_2026-10-08.md).

One Q9 preparation trajectory, saved at steps 250 and 300; both states were assigned **the exact same 900 Q3 training examples and LR values**, with original Q3 scales and fresh Adam. Earlier state total **1150** step opportunities; later state **1200** (not an equal-compute comparison). Both Q3 continuations had **897 effective optimizer updates and three AMP skips**.

- Q9(250)+common Q3(900): **held-out loss 4.9504320**, PPL **141.236**.
- Q9(300)+common Q3(900): **held-out loss 4.9009854**, PPL **134.422**. Exact historical v10 reproduction.
- Thus **longer Q9 preparation retains a +0.0494467-nat quality advantage under a common Q3 operator** in this seed. It remains exploratory, evaluated on previously viewed WikiText tokens.
- **6,245,131 / 314,572,800 weights (1.9853%)** have different initial original-Q3-scale codes between Q9 steps 250 and 300. After continuation, earlier arm's final Q3 code matches later Q9-prep code at **44.16%** of those positions; two final Q3 models agree at **60.00%** of those selected positions, and **93.858%** globally. This suggests partial adoption of late-Q9 choices and incomplete final convergence, **not** causal proof of optimal assignments.
- M1 differs from C1's earlier 250+950 recipe (NLL 4.930515) in Q3 **length, beginning batches and LR indices**; the difference between those runs cannot be cleanly assigned to exactly 50 extra Q3 steps.

**NO new GPU job has been launched after M1.** Next scientifically highest-value gate: an independently scoped **longer-horizon paired direct vs staged fixed-recipe experiment with fresh evaluation data**, and new training orders for confirmation. Neither has yet been approved or launched. The complete handoff and interpretation plan is [NEXT_EXPERIMENT_PLAN.md](NEXT_EXPERIMENT_PLAN.md).

---

## What is established (and scoped)

- On SmolLM2-360M / WikiText-2, matched-schedule Q9(300)→Q3(900) outperforms tuned direct-Q3(1200) by mean **0.6723 nats/token** (about 49% lower PPL) across **3/3 training orders**. This is a **finite-budget** quantization-aware-training optimization result, not a usable ternary generation model, better asymptotic basin, or deployed memory/speed win.
- Mechanistic v11/v12/v13 interventions identify **which ternary weight positions/codes are chosen and moderate within-bin master placement** as major carriers of the gain, on the tested recipe. The ~6.3% D/Q9 disagreement mask recovers ~96.4% of the full gain via master transfer (3/3). Q9-only changes and direct-only changes Q9 avoids have **not** been causally separated.
- **T1 completed, seed1729**: the projected code disagreement mask emerges gradually; at Q9 step250 it overlaps **69.40%** of the eventual step300 mask (precision **77.77%**). This is retrospective geometry, **not** the proportion of important assignments.
- **C1 completed, seed1729**: direct Q3 1200 test NLL **5.595722**, Q9 250 + Q3 950 **4.930515**, Q9 300 + Q3 900 **4.900985**. S250 retains **95.75%** of S300 improvement over direct. **Exploratory on an already-studied order and evaluation slice**, not evidence that the last 50 Q9 steps are unnecessary; earlier Q9 had 50 more Q3 steps in C1. D and S300 exactly reproduced historical baselines.
- **Generalization negative/mixed:** Granite full staging 2/3 training orders and schedule sensitivity. Periodic gridward pull helps Granite in two tested orders but worsens Smol under its tuned schedule (**S1-1 negative**). Avoid claiming a universal Q9 or gridward law. Longer v5 effect shrank but persisted on one old-schedule run; asymptotic convergence is **unresolved**.
- Held-out evaluation has been reused across exploratory decisions; free generation remains poor. Published-method baselines and genuinely fresh datasets/checkpoints remain important gaps.

## Completed experiment records

| ID | HF job | Outcome |
|---|---|---|
| T1 (Q9 projected-code discovery timing) | [6ac85534fee2c90070172a41](https://huggingface.co/jobs/codeflash85/6ac85534fee2c90070172a41) | [Summary](results/run_t1_q9_discovery_timing_seed1729_summary.md); [raw JSON](results/run_t1_q9_discovery_timing_seed1729_2026-10-08.json). Completed. |
| C1 (Q9 switch 250 vs300, equal total budget) | [6ac8597afee2c90070172c79](https://huggingface.co/jobs/codeflash85/6ac8597afee2c90070172c79) | [Summary](results/run_c1_smol360m_q9_switch250_vs300_seed1729_summary.md); [raw JSON](results/run_c1_smol360m_q9_switch250_vs300_seed1729_2026-10-08.json). Completed. |
| M1 (250 vs300 Q9 prep with *identical* Q3 continuation) | [6ac86712fee2c900701734bb](https://huggingface.co/jobs/codeflash85/6ac86712fee2c900701734bb) | [Summary](results/run_m1_equal_q3_continuation_seed1729_summary.md); [raw JSON](results/run_m1_equal_q3_continuation_seed1729_2026-10-09.json). Completed, checks passed. |

## Current next action and constraints

**No immediate compute queued.** The paired 250-vs-300 controls, both equal-total (C1) and equal-Q3-continuation (M1), have now been run on seed1729, which has repeatedly informed the experiment design. The most consequential open questions are **durability (does direct Q3 catch up at longer training horizons?), fresh evaluation, and cross-seed/family confirmation**. Decide a budget and freeze an independent next protocol before any paid compute; do not repeatedly tune using the known 8192-token WikiText test slice.

**Next AI:** Read [NEXT_EXPERIMENT_PLAN.md](NEXT_EXPERIMENT_PLAN.md) for hypotheses/controls, this current-state summary for latest status, [AI_HANDOFF.md](AI_HANDOFF.md) for chronology and [EXPERIMENT.md](EXPERIMENT.md) for historical protocols. M1 must **not** be relaunched. First check Hugging Face for any newer job before planning. Record exact source SHA, prereg, job ID, raw output, validity and negatives for each new experiment.

## Key limitations and competing explanations

The remaining scientific choice is not simply *when the step300 disagreement mask appears*. There may be early high-value decisions, Q3 recovery of missing decisions, alternative good assignment patterns, or compensation from extra Q3 steps. External AI reviewers observed that T1+C1 cannot distinguish these. They also identified a crucial durability question: does direct Q3 catch up with more training? These are **hypotheses** to test, not findings.

**Paper/publication boundary:** A strong controlled assignment-selection observation exists in the specified pilot. No claim of portable ternary QAT or production-ready compression is justified yet. Separate confirmation on new orders/data and a longer fixed-recipe comparison are priorities, **not automatically authorized jobs**.
