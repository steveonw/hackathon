# Ternary Pet — CURRENT STATE

**Authoritative short handoff:** As of 2026-10-08 Eastern / 2026-10-09 UTC.  
**Repository:** `steveonw/hackathon` → `ternary_pet/`.  
**Status:** F1 FineWeb-Edu seed1729 completed; **F2/F3 seeds271828/424242 and L1 WikiText6000 are RUNNING**. New seed replication plan is frozen; no F2/F3 results yet. See [NEXT_JOBS_PLAN.md](NEXT_JOBS_PLAN.md) for details.

## ACTIVE: FineWeb-Edu F2/F3 seed replications (2026-10-09 UTC)

**User-authorized two new-seed GPU jobs, now RUNNING:**

- **F2, seed 271828:** [F2 seed271828](https://huggingface.co/jobs/codeflash85/6ac87415fee2c90070173baf); A10G-small, 90-minute maximum; immutable source SHA `64d171a225ba2781c83148a4235272c91939f9c9`, [script](f2_fineweb_edu_qat_seed271828.py).
- **F3, seed 424242:** [F3 seed424242](https://huggingface.co/jobs/codeflash85/6ac87417095c5780893020c2); A10G-small, 90-minute maximum; immutable source SHA `a20c4cc66ae16026bb90969d1160294b3f5826e9`, [script](f3_fineweb_edu_qat_seed424242.py).

[F2/F3 frozen preregistration](research_log/f2_f3_fineweb_edu_seed271828_424242_replication_prereg_2026-10-09.md) was committed **before compute**. HF CPU static smoke `6ac87407fee2c90070173b9d` emitted `F2_F3_PREFLIGHT_OK`; normalized source-code comparison passed: only seed, seed-expected permutation assertion, and provenance identifiers changed from the successful F1 seed1729 script. Exact same source-model/dataset SHAs, same FineWeb QAT document-based train/dev/eval partition and same 1200-step matched arms. **No F2/F3 scientific outcomes available yet.**

**Original F1 seed1729 result (already completed):** [summary](results/run_f1_fineweb_edu_direct_vs_staged_seed1729_summary.md) and [raw](results/run_f1_fineweb_edu_direct_vs_staged_seed1729_2026-10-09.json): staged improves held-out FineWeb NLL by **0.770405 nats** and WikiText validation NLL by **0.863013 nats**. Three-seed aggregate must wait for F2 and F3 `FINAL_JSON` and scientific checks. The heldout 21 FineWeb documents are **shared across seeds**, not independent datasets.

**Also still RUNNING at last check:** [L1 6000-step WikiText durability](https://huggingface.co/jobs/codeflash85/6ac86e1f095c578089301e53). **Do not cancel or resubmit L1.** There are exactly two *new* scientific GPU jobs in this replication request.

**Next AI:** inspect these three HF job IDs, never duplicate them. For F2/F3 parse `FINAL_JSON_BEGIN`/`FINAL_JSON_END`; verify all five checks and source-doc partition hashes match F1. Archive per-seed raw JSON, summary and three-seed aggregate (mean/median/min-max/positive count). Clearly label whether any failure was technical versus a negative staged effect; update this page, `AI_HANDOFF.md`, `EXPERIMENT.md`, README and both research reports. No additional GPU or data-search sweep authorized.

---

## CURRENT: L1 WikiText durability running; F1 FineWeb-Edu completed

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
