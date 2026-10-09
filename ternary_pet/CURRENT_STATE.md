# Ternary Pet — CURRENT STATE

**Authoritative short handoff:** As of 2026-10-08 Eastern / 2026-10-09 UTC.  
**Repository:** `steveonw/hackathon` → `ternary_pet/`.  
**Status:** T1, C1 and M1 completed; **no active scientific GPU job**. Read [NEXT_EXPERIMENT_PLAN.md](NEXT_EXPERIMENT_PLAN.md) for archived M1 design, alternative hypotheses and subsequent research gates.

## NEW ACTIVE STUDY — L1 durability and F1 FineWeb-Edu

**2026-10-09 UTC / October 8 EDT.** **L1 GPU submitted / RUNNING at last check**: [HF 6ac86e1f095c578089301e53](https://huggingface.co/jobs/codeflash85/6ac86e1f095c578089301e53), A10G-small, <=2h, pinned script SHA `841e0ab948991a24b7397b34d90d28c38f749879`. Paired **6000-step direct Q3 vs Q9(300)→Q3** on WikiText2, preserving first1200 historic steps; subsequent 4800 are fresh QAT train chunks and LR is fixed 1e-4. New WikiText2 validation curve measured at five horizons. **Results unknown until job completion.**

**F1 FineWeb-Edu GPU SUBMITTED** as [6ac86eb0fee2c900701738dd](https://huggingface.co/jobs/codeflash85/6ac86eb0fee2c900701738dd) (A10G-small max90m; initial status SCHEDULING). Preregistered and implemented, pinned script SHA `64dcd7f20240b4c62e6ecac8df70a1336a57bb14`; initial source streaming and both static checks passed. The complete CPU dataset-partition preflight [6ac86e0d095c578089301e45](https://huggingface.co/jobs/codeflash85/6ac86e0d095c578089301e45) **PASSED** with train1200/dev24/eval128 chunks and disjoint source document IDs. One F1 GPU job was accepted; results pending. F1 changes only QAT corpus under a matched 1200-step paired trial. Train/dev/eval FineWeb documents partitioned by hash; potential overlap with *source model* pretraining explicitly unknown.

**Frozen protocols:** [L1 durability](research_log/l1_6000_step_durability_wikitext_seed1729_prereg_2026-10-09.md), [F1 FineWeb-Edu](research_log/f1_fineweb_edu_qat_seed1729_prereg_2026-10-09.md). **Copyable handoff plan:** [NEXT_JOBS_PLAN.md](NEXT_JOBS_PLAN.md).

**Handoff:** check both job stages, do not submit duplicates, verify `FINAL_JSON` and scientific validity, archive raw JSON, record negative results. **T1/C1/M1 remain completed** and are untouched.

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
