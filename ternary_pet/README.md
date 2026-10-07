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

v8 tested a more specific explanation: that Q9 uniquely moves later-flipping
weights in their eventual ternary-transition direction during preparation.
That explanation was **not supported**. Each arm was more aligned on its own
future-flip set by a similar margin, which is consistent with post-selection /
generic trajectory alignment rather than Q9-specific directional pre-loading.

So the Q9 trainability effect remains replicated, but its mechanism is deeper
than simple "point the future-flipping weights toward their next threshold."

v9 strengthened the direct-Q3 baseline with a 100-step warmup to 1e-3 followed
by cosine decay to 1e-4. v10 then applied that **same global LR schedule** to
Q9 -> Q3 on all three training orders, with the v7 transition semantics
(original Q3 scales + fresh Adam at step 300, no LR restart).

Schedule-matched Q9 remains better than tuned direct in **3/3 orders**:

- held-out loss advantage: **0.695, 0.672, 0.650 nats/token**;
- mean advantage: **0.672 nats/token**;
- mean paired PPL reduction: **48.94%**;
- mean teacher top-1 gain: **+6.82 pp**.

The central trainability pattern also survives: after equal 300-step compute,
the schedule-matched Q9 masters are still a **worse immediate Q3 checkpoint**
than tuned direct by **0.512 nats/token on average**, yet they finish much
better after the next 900 Q3 updates.

This closes the equal-global-schedule loophole. It is still not a
best-tuned-vs-best-tuned comparison because Q9 has not received an independent
equal-budget hyperparameter search.

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
- `results/run_v8_summary.md` — signed-preload diagnostic
- `results/run_v8_2026-10-07.json` — v8 raw result
- `results/run_v9_summary.md` — tuned direct-Q3 baseline result
- `results/run_v9_2026-10-07.json` — v9 raw result
- `replications/v9_tuned_direct_aggregate_summary.md` — canonical tuned-direct three-order comparison
- `results/run_v9_tuned_direct_seed271828_2026-10-07.json` — tuned-direct replication raw
- `results/run_v9_tuned_direct_seed424242_2026-10-07.json` — tuned-direct replication raw
- `replications/v10_schedule_matched_q9_aggregate_summary.md` — equal-schedule Q9 aggregate
- `results/run_v10_schedule_matched_q9_seed1729_2026-10-07.json` — v10 raw
- `results/run_v10_schedule_matched_q9_seed271828_2026-10-07.json` — v10 raw
- `results/run_v10_schedule_matched_q9_seed424242_2026-10-07.json` — v10 raw
- `replications/` — replication scripts and summaries
- `results/` — all run records


## v11 mechanism result — seed 1729

The matched-schedule hybrid-master factorial gives a clear causal split.

The direct-vs-Q9 step-300 projected-Q3 disagreement mask contains **6.458%** of
quantized weights.

After an identical fresh-Adam/original-Q3-scale continuation:

| Arm | Composition | Final loss |
|---|---|---:|
| 00 | direct masters everywhere | 5.6136 |
| 10 | Q9 masters only on code-disagreement positions | **4.9329** |
| 01 | Q9 masters only on same-code positions | 5.5020 |
| 11 | Q9 masters everywhere | **4.9010** |

Transferring Q9 masters only on the 6.458% disagreement mask recovers
**95.5%** of the full 00->11 loss gain. Same-code hidden geometry alone recovers
a much smaller **0.1116 nats** on the direct background.

The equal-forward pair assertions passed exactly before training:
Q3(00)==Q3(01) and Q3(10)==Q3(11), with zero Hamming distance and identical
diagnostic losses.

This strongly localizes the seed-1729 trainability benefit to the positions
where direct and Q9 choose different ternary assignments at step 300. It does
**not** yet prove that the discrete code labels alone are causal, because arm 10
transfers the full continuous Q9 master values on those positions.

See `results/run_v11_hybrid_factorial_summary.md`.
