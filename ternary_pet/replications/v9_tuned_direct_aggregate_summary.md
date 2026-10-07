# Tuned direct-Q3 vs Q9 — three-order aggregate

The v9-selected tuned direct-Q3 schedule was replicated on all three established
v7 training orders:

- 100-step linear warmup to LR 1e-3;
- cosine decay to LR 1e-4 by step 1200;
- otherwise the same direct-Q3 recipe.

Seed 1729 was selected in v9. Seeds 271828 and 424242 were confirmatory
replications with no additional hyperparameter search.

## Results

| Seed | Old direct loss | Tuned direct loss | Q9 -> Q3 loss | Residual Q9 gain | Old gap closed |
|---:|---:|---:|---:|---:|---:|
| 1729 | 5.8724 | 5.5957 | **5.1938** | **0.4020** | 40.77% |
| 271828 | 5.9802 | 5.6228 | **5.2417** | **0.3810** | 48.40% |
| 424242 | 5.9469 | 5.6067 | **5.3008** | **0.3059** | 52.65% |

Q9 remains better than tuned direct on held-out loss in **3/3 orders**.

Aggregate against tuned direct:

- mean Q9 held-out loss advantage: **0.3630 nats/token**;
- mean paired PPL reduction: **30.38%**;
- mean teacher top-1 gain: **+2.52 percentage points**;
- mean KL advantage: **0.3871**.

Tuning direct Q3 closes **47.27%** of the old
Q9-vs-direct loss gap on average across orders.

Per-seed tuned-direct vs Q9:

| Seed | Tuned PPL | Q9 PPL | Q9 PPL reduction | Tuned top-1 | Q9 top-1 |
|---:|---:|---:|---:|---:|---:|
| 1729 | 269.27 | **180.15** | 33.10% | 29.87% | **33.79%** |
| 271828 | 276.66 | **189.00** | 31.69% | 29.93% | **31.75%** |
| 424242 | 272.25 | **200.50** | 26.35% | 30.77% | **32.59%** |

## Interpretation

The historical direct-Q3 baseline was materially undertuned. A better schedule
explains nearly half of the original small-budget Q9 advantage.

However, the residual Q9 advantage remains directionally consistent across all
three training orders. The honest canonical small-budget comparison is therefore
no longer the old ~0.69-nat mean gap, but approximately **0.3630
nats/token** against the tuned direct schedule.

This supports proceeding to the preregistered hybrid-master causal intervention,
while keeping the finite-budget and single-model/data caveats.

Technical note: the first replication launch pair failed/cancelled due to a
generated-script truncation bug before training. The valid confirmatory jobs are:

- seed 271828: `6ac5b913404719ba37664b21`;
- seed 424242: `6ac5b915fbc85ba6823bba5b`.

Corrected replication scripts are pinned at
`3e74ccf5c448fc994d005a7baf94529b92e9996a`.
