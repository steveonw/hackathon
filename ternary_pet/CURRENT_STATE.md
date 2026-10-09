# Ternary Pet — CURRENT STATE

**Authoritative short handoff:** As of 2026-10-08 Eastern / 2026-10-09 UTC.  
**Repository:** `steveonw/hackathon` → `ternary_pet/`.  
**Status:** T1 and C1 completed; **M1 next experiment is documented in [NEXT_EXPERIMENT_PLAN.md](NEXT_EXPERIMENT_PLAN.md)**. Check this page and [AI_HANDOFF.md](AI_HANDOFF.md) for latest execution status before launching compute.

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

## Current next action and constraints

**M1 proposal:** Compare saved Q9 preparation states from step250 and step300, each with **identical 900-step Q3 continuations**, shared post-transition LR and training chunks, while recording **actual final ternary codes** to assess convergence or rediscovery. This deliberately has **unequal total update opportunities (1150 vs1200)** and complements rather than replaces C1. Full plan: [NEXT_EXPERIMENT_PLAN.md](NEXT_EXPERIMENT_PLAN.md). The plan includes preregistration, GPU/time bound, checks, analysis, alternative explanations and subsequent confirmatory/long-run research gates.

**Next AI:** Read the execution status here and in the latest top of [AI_HANDOFF.md](AI_HANDOFF.md). If there is an M1 job ID, **inspect it before doing anything**, do not duplicate or silently launch new seeds. Follow the exact pinned code and job, check raw `FINAL_JSON`, archive **positive, negative or technical** outcomes verbatim and update both short status and chronological documents. The full experiment ledger is [EXPERIMENT.md](EXPERIMENT.md).

## Key limitations and competing explanations

The remaining scientific choice is not simply *when the step300 disagreement mask appears*. There may be early high-value decisions, Q3 recovery of missing decisions, alternative good assignment patterns, or compensation from extra Q3 steps. External AI reviewers observed that T1+C1 cannot distinguish these. They also identified a crucial durability question: does direct Q3 catch up with more training? These are **hypotheses** to test, not findings.

**Paper/publication boundary:** A strong controlled assignment-selection observation exists in the specified pilot. No claim of portable ternary QAT or production-ready compression is justified yet. Separate confirmation on new orders/data and a longer fixed-recipe comparison are priorities, **not automatically authorized jobs**.
