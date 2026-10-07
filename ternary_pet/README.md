# Ternary Pet Experiments

> ## 🤖 AI / researcher handoff — START HERE
>
> New AI/researcher: read **[AI_HANDOFF.md](AI_HANDOFF.md)** before changing an
> experiment or launching compute.

Experiments on whether a pretrained language model can enter ternary weight
space more gracefully through an intermediate representation.

## Current headline

**The v7 trainability effect now replicates across three independent training
orders (1729, 271828, 424242).**

At equal 300-step preparation, Q9 is a *worse* immediate Q3 checkpoint than
direct Q3 in all three orders. Yet after the identical fresh-Adam 900-step Q3
continuation, Q9 finishes substantially better in all three.

Across the three v7 orders:

- mean immediate Q9-vs-direct Q3 disadvantage at step 300:
  **+2.588 nats/token**;
- mean final Q9 advantage:
  **0.688 nats/token**;
- mean paired PPL reduction:
  **49.69%**;
- mean teacher top-1 gain:
  **+6.86 percentage points**.

The weight-selection geometry also replicates:

- direct and Q9 each change about **0.68%** of future ternary codes by step 300;
- the changed-position sets overlap only about **19.66% Jaccard** on average;
- when both change the same weight, they choose the same resulting ternary code
  **100%** of the time;
- global distance-to-Q3-threshold distributions remain effectively identical.

The FP32 control does **not** reproduce Q9's large advantage, although it is not
universally worse than direct: seed 271828 gives FP32 a tiny 0.029-nat edge over
direct. The safe claim is that generic FP32 warmup is insufficient to explain
Q9.

## Current interpretation

> Q9's intermediate discrete constraint produces a reproducible
> **trainability / weight-selection geometry** effect. It moves a mostly
> different subset of continuous master weights than direct Q3 does, and the
> resulting master state responds much better to later ternary optimization
> despite being a worse immediate ternary model.

This remains a finite-budget result on one model/data setup. v5 shows direct Q3
catches up substantially with more training, and free-running generation remains
poor.

## Repository layout

- `AI_HANDOFF.md` — start here
- `EXPERIMENT.md` — chronological protocol/outcomes
- `SHAREABLE_RESEARCH_REPORT.md` — external-review narrative
- `smollm2_v7_equal_compute_geometry.py` — v7 protocol
- `replications/v7_aggregate_summary.md` — **three-order v7 aggregate**
- `results/run_v7_summary.md` — original seed 1729 v7 result
- `results/run_v7_2026-10-07.json` — original raw v7
- `results/run_v7_seed271828_2026-10-07.json` — replication raw
- `results/run_v7_seed424242_2026-10-07.json` — replication raw
- `replications/` — replication scripts and summaries
- `results/` — all run records
