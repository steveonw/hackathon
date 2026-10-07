# v11 — matched-schedule hybrid-master factorial

**HF Job:** `6ac5c7befbc85ba6823bbef0`  
**Pinned code:** `f342fd9c706f2fe21aa00adabe6611b7f835f570`  
**Seed/order:** 1729  
**Hardware:** A10G-small

## Question

Under the stronger v10 global schedule, which part of the step-300 Q9-prepared
FP32 master state causes the later ternary trainability advantage?

Let:

- D = direct-Q3 masters after 300 matched-schedule updates;
- S = Q9 masters after 300 matched-schedule updates;
- M = exact weight positions where D and S, projected through the same original
  Q3 scales, have different Q3 codes.

The intervention constructs:

- **00** = D everywhere;
- **10** = S on M, D elsewhere;
- **01** = D on M, S elsewhere;
- **11** = S everywhere.

All four then receive original Q3 scales, fresh Adam, and the identical global
LR continuation from step 301 through step 1200.

## Pair-equality checks

The intended equal-forward pairs are exact:

- Q3(00) vs Q3(01): **0.000000 Hamming**
- Q3(10) vs Q3(11): **0.000000 Hamming**
- pre-continuation validation loss difference 00 vs 01: **0**
- pre-continuation validation loss difference 10 vs 11: **0**

Thus any later difference within those pairs comes from hidden continuous FP32
master geometry rather than the initial ternary forward model.

## Prepared-state geometry

The D-vs-S code-disagreement mask contains:

- **6.458%** of quantized weights;
- **20,315,352** of 314,572,800 weights.

Step-300 projected-Q3 movement versus the original source:

- D: **4.569%**
- S: **4.884%**
- D-vs-S pairwise Hamming: **6.458%**
- changed-set Jaccard versus initial: **18.85%**

The ~18.85% Jaccard is strikingly close to the old v7 ~19.7% value despite the
much larger absolute code movement in the tuned regime, but this is a
single-seed observation and should not yet be elevated into a general law.

## Final held-out results

| Arm | Master composition | Loss | PPL | Top-1 | KL |
|---|---|---:|---:|---:|---:|
| 00 | D everywhere | 5.6136 | 274.13 | 29.72% | 2.6791 |
| 10 | S on M only | **4.9329** | **138.78** | 36.07% | 1.9969 |
| 01 | S off M only | 5.5020 | 245.18 | 31.20% | 2.5666 |
| 11 | S everywhere | **4.9010** | **134.42** | **36.91%** | **1.9700** |

The endpoint 00-vs-11 gap is **0.7126
nats/token**, so the core trainability effect reproduces under the common
fresh-Adam/original-scale continuation.

## Causal decomposition

Using positive numbers to mean "Q9 material improved final loss":

- transfer S on M onto D background:
  **0.6807 nats**
  = **95.5%** of the
  full 00->11 gain;
- transfer S on the same-code complement onto D background:
  **0.1116 nats**
  = **15.7%**
  of the full gain;
- after M is already from S, adding the same-code complement contributes only:
  **0.0319 nats**;
- after the complement is already from S, transferring M still contributes:
  **0.6010 nats**.

The factorial interaction is **0.0797
nats**. With this sign convention it indicates modest sub-additivity /
redundancy: the two single-component gains on the D background sum to more than
the full joint gain.

## Interpretation

This is strong causal localization on seed 1729:

> Most of the Q9 trainability advantage is carried by the **~6.46% of weight
> positions where the step-300 Q9 and direct prepared states project to
> different ternary codes**.

Transferring Q9 masters only on those positions recovers **~95.5%** of the full
Q9 loss advantage and leaves only **0.0319 nats/token** between arm 10 and full
Q9 arm 11.

The other ~93.54% of weights, where D and S already share the same ternary code,
are not irrelevant: transferring their hidden Q9 continuous geometry alone
improves loss by **0.1116 nats** on the direct background. But that contribution
is much smaller, and once the code-disagreement positions already come from Q9,
the remaining same-code contribution falls to only **0.0319 nats**.

So v11 argues against the "benefit is mostly a distributed invisible shift over
the same-code majority" explanation. The dominant carrier is instead the
continuous Q9-prepared master state on the relatively small set of positions
where Q9 and direct choose **different ternary assignments at step 300**.

Important nuance: this does **not** yet prove the discrete code values alone are
causal. Arm 10 transfers the full continuous Q9 master values on M, not merely
their ternary code labels. A follow-up can separate the discrete assignment from
within-bin master position on M.

Because this is one mechanism seed, the next step should replicate the v11
factorial on orders 271828 and 424242 before making the localization claim
three-order canonical.
