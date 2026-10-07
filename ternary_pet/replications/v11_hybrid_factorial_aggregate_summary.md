# v11 hybrid-master factorial — three-order aggregate

The v11 causal localization was run on all three established training orders
under the same matched-schedule regime.

All runs independently rebuilt direct-Q3 (D) and Q9 (S) step-300 master states,
formed the projected-Q3 disagreement mask M, and constructed the four causal
arms:

- 00 = D everywhere
- 10 = S on M, D elsewhere
- 01 = D on M, S elsewhere
- 11 = S everywhere

Before continuation, every run passed the exact forward-pair checks:

- Q3(00) == Q3(01), zero Hamming;
- Q3(10) == Q3(11), zero Hamming;
- paired pre-continuation diagnostic losses exactly matched.

## Per-order results

| Seed | Mask % | L00 | L10 | L01 | L11 | Full gain | Mask gain | Mask recovery |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1729 | 6.46% | 5.6136 | **4.9329** | 5.5020 | **4.9010** | 0.7126 | 0.6807 | **95.52%** |
| 271828 | 6.18% | 5.6524 | **4.9711** | 5.5620 | **4.9510** | 0.7014 | 0.6813 | **97.13%** |
| 424242 | 6.30% | 5.6234 | **4.9791** | 5.4979 | **4.9563** | 0.6672 | 0.6443 | **96.57%** |

The localization replicates in **3/3 orders**.

Aggregate:

- mean disagreement-mask size: **6.31%**
  (range 6.18%–6.46%);
- mean full 00 -> 11 advantage: **0.6938 nats/token**;
- mean gain from transferring Q9 masters only on M: **0.6688
  nats/token**;
- mean fraction of full gain recovered by M-only transfer:
  **96.40%**
  (range 95.52%–97.13%);
- mean same-code-complement gain on a D background:
  **0.1092 nats/token**;
- mean residual same-code contribution after M is already from Q9:
  **0.0250 nats/token**;
- mean factorial interaction: **0.0842 nats**.

## Geometry stability

Changed-set Jaccard versus the original source is:

- seed 1729: **18.85%**
- seed 271828: **19.35%**
- seed 424242: **18.97%**

Mean: **19.05%**.

Thus the ~19% changed-set overlap seen in v7 also reappears across all three
orders in the stronger tuned regime, despite much larger absolute code movement.
This is now a replicated descriptive regularity for this model/setup, though
not yet a general architectural law.

## Interpretation

The causal localization is now replicated:

> Across three training orders, about **6.31%** of quantized
> weights — the positions where direct-Q3 and Q9 preparation choose different
> projected ternary codes after 300 updates — carry enough Q9-prepared
> continuous master state to recover **96.40%** of the full
> Q9 trainability advantage on average.

The same-code majority still contributes a small secondary effect, but it is
not the dominant carrier.

This establishes a robust **where** result. It does not yet establish **what
information** on M is causal. Arm 10 transfers the full continuous Q9 master
values at those positions. The next causal refinement should separate:

1. Q9-selected ternary code identity;
2. continuous within-bin / boundary-relative position.

A clean follow-up should construct M-only interventions that preserve the
Q9-selected Q3 code while replacing the exact Q9 master value with standardized
representatives (for example the Q3 reconstruction prototype), and compare
those against the exact-S-on-M arm.

Raw confirmatory jobs:

- seed 271828: `6ac638c0c656c912b4ffae8a`
- seed 424242: `6ac638c3f0d78b8017af0d5a`

Pinned replication scripts:
`bb3ba53b4b55bfc6884d287d5a575785e212ffb4`.
