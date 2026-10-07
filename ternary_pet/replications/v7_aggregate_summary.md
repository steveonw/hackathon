# v7 replication aggregate — three training orders

**Protocol:** exact v7 equal-compute Q3 vs Q9 vs FP32 preparation geometry.  
**Seeds/orders:** 1729, 271828, 424242.  
**Replication code commit:** `6caf3a98d485ed1fd49e22b915ddb6b175578420`

## Main replication result

The central v7 effect replicated in **3/3 training orders**:

1. after equal 300-step preparation, Q9 projects to a **worse** immediate Q3
   checkpoint than direct Q3;
2. after the identical fresh-Adam 900-step Q3 continuation, Q9 finishes
   **substantially better** than direct.

| Seed | Q3 loss @300 | Q9->Q3 loss @300 | Q9 disadvantage @300 | Final Q3 loss | Final Q9 loss | Q9 final gain |
|---:|---:|---:|---:|---:|---:|---:|
| 1729 | 6.5589 | 9.2264 | +2.6676 | 5.8724 | **5.1938** | **0.6786** |
| 271828 | 6.6449 | 8.8909 | +2.2460 | 5.9802 | **5.2417** | **0.7384** |
| 424242 | 6.5824 | 9.4334 | +2.8510 | 5.9469 | **5.3008** | **0.6461** |

Across the three orders:

- mean Q9 immediate-Q3 disadvantage at step 300: **+2.5882 nats/token**
  (range +2.2460 to +2.8510);
- mean final Q9 loss advantage: **0.6877 nats/token**
  (range 0.6461 to 0.7384);
- mean paired PPL reduction: **49.69%**
  (range 47.59% to 52.21%);
- mean teacher top-1 gain: **+6.86 percentage points**.

This makes the trainability effect substantially more credible than the original
single-order v7 result.

## FP32 control: important nuance

Q9 beats the FP32-prepared arm clearly in all three orders.

However, the stronger claim that FP32 always finishes worse than direct Q3 does
**not** replicate perfectly:

| Seed | Final direct loss | Final FP32-prep loss | FP32 minus direct |
|---:|---:|---:|---:|
| 1729 | 5.8724 | 6.0311 | +0.1588 |
| 271828 | 5.9802 | 5.9510 | -0.0292 |
| 424242 | 5.9469 | 6.0441 | +0.0972 |

Seed 271828 gives FP32 a tiny **0.0292-nat** advantage over direct, while the
other two orders favor direct.

Therefore the safe replicated conclusion is:

> generic FP32 warmup does **not reproduce the large Q9 advantage**.

Do not claim that FP32 warmup is universally worse than direct.

## Weight-selection geometry replicates

Step-300 future-Q3 code movement:

| Seed | Direct Q3 changed | Q9 changed | Changed-set Jaccard | Q3-vs-Q9 Hamming |
|---:|---:|---:|---:|---:|
| 1729 | 0.69% | 0.68% | 19.64% | 0.92% |
| 271828 | 0.68% | 0.68% | 20.06% | 0.90% |
| 424242 | 0.68% | 0.67% | 19.26% | 0.92% |

Aggregate:

- mean direct changed fraction: **0.68%**;
- mean Q9 changed fraction: **0.68%**;
- mean changed-set Jaccard: **19.66%**
  (range 19.26% to 20.06%);
- mean pairwise Q3 Hamming distance: **0.91%**;
- when both Q3 and Q9 change the same weight, the resulting ternary code is the
  same **100%** of the time in every run.

Thus the weight-selection result is highly stable: Q3 and Q9 change similar
numbers of future ternary decisions but mostly **different specific weights**.

## Threshold-margin result also replicates

The fraction of weights within normalized distance 0.02 of a Q3 decision
threshold remains effectively identical between direct and Q9 in every order:

- seed 1729: Q3 3.56% vs Q9 3.56%
- seed 271828: Q3 3.56% vs Q9 3.56%
- seed 424242: Q3 3.57% vs Q9 3.56%

So a simple global "Q9 leaves more weights near thresholds" mechanism remains
unsupported.

## Current best-supported mechanism

Across three training orders, the evidence now supports:

```
Q9 forward constraint during preparation
    -> different gradient / weight-selection geometry
    -> roughly the same amount of future Q3 code movement as direct
    -> mostly different specific weights are repositioned
    -> immediate Q3 checkpoint is worse than direct
    -> subsequent Q3 optimization is much more productive
```

This is a replicated **finite-budget trainability effect** under the current
SmolLM2/WikiText recipe.

It is still not evidence of asymptotic superiority, cross-model generality, or
usable generation quality.
