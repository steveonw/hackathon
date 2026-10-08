# Granite G1 seed-271828 forensic comparison

**Type:** post-hoc analysis of existing completed runs; **no new compute**.  
**Model:** `ibm-granite/granite-4.0-350m`  
**Orders compared:** 1729, 271828, 424242  
**Question:** what distinguishes the one Granite order where full Q9→Q3 staging
loses to direct Q3?

Source files:

- `results/run_g1_3_granite350m_mechanism_seed1729_2026-10-08.json`
- `results/run_g1_4_granite350m_mechanism_seed271828_2026-10-08.json`
- `results/run_g1_5_granite350m_mechanism_seed424242_2026-10-08.json`

## Main finding

The strongest discriminator appears **during Q9 preparation itself, before the
Q3 switch**.

| Diagnostic | 1729 (+) | 271828 (−) | 424242 (+) |
|---|---:|---:|---:|
| Native Q9 minus native D loss @300 | **−0.1604** | **+0.2780** | **−0.1378** |
| Q9/D grad-norm ratio @100 | 0.82× | **3.33×** | 0.76× |
| Q9/D grad-norm ratio @300 | 0.97× | **1.73×** | 0.79× |
| D movement vs initial Q3 | 5.324% | 5.339% | 5.339% |
| Q9 movement vs initial Q3 | 5.272% | **5.083%** | 5.268% |
| D-vs-Q9 mask | 6.025% | **6.995%** | 6.030% |
| implied changed-set Jaccard proxy | ~27.5% | **~19.7%** | ~27.5% |
| layer-27 share of mask | 3.12% | **4.56%** | 3.18% |
| d50 code survival @900 | 90.70% | 89.59% | 90.57% |

Positive orders are marked (+); 271828 is the negative staging order (−).

On 1729 and 424242, by step 100 the Q9 training loss is already below direct
and remains generally favorable through preparation. On 271828, Q9 remains
worse than direct at every logged preparation checkpoint:

| prep step | Q9−D loss, 1729 | Q9−D loss, 271828 | Q9−D loss, 424242 |
|---:|---:|---:|---:|
| 100 | −0.1055 | **+2.3350** | −0.1080 |
| 200 | −0.3275 | **+0.6344** | −0.1153 |
| 300 | −0.2495 | **+0.2578** | −0.1604 |

The validation diagnostic agrees: native Q9 is better than native direct at
step 300 on both positive orders, but **0.278 nats worse** on 271828.

This suggests that order 271828 does not merely encounter a bad Q3 projection:
the **Q9 preparation trajectory itself has not settled as successfully** under
the fixed 300-step budget.

## Geometry: more disagreement, less common movement

Direct Q3 movement is nearly identical on all three orders (~5.33%). The
negative order differs in Q9:

- Q9 changes only **5.083%** of source Q3 codes, versus ~5.27% on the positive
  orders;
- yet D and Q9 disagree on **6.995%**, versus ~6.03% on the positive orders.

Using the stored D-vs-initial, Q9-vs-initial and D-vs-Q9 Hamming fractions,
the implied changed-set overlap proxy is ~**19.7%** on 271828 versus ~**27.5%**
on both positive orders.

This overlap is a proxy because the raw confirmation files do not store the
full changed-set intersection directly. D↔S sign-flip transitions are extremely
rare (<0.1% of the mask), so the approximation is close; on order 1729 the
proxy (~27.51%) agrees with the separately logged exact G1-2 Jaccard
(**27.53%**).

Interpretation: 271828's Q9 preparation selects a **more divergent set of
positions** relative to direct, even though Q9 itself moves fewer source codes.

## Layer/module composition

The broad module mix is surprisingly stable. Fraction of the disagreement mask:

| Module type | 1729 | 271828 | 424242 |
|---|---:|---:|---:|
| shared MLP input | 47.84% | 48.58% | 47.87% |
| shared MLP output | 21.49% | 21.56% | 21.53% |
| q_proj | 14.79% | 14.12% | 14.73% |
| o_proj | 9.83% | 9.66% | 9.78% |
| k_proj | 3.61% | 3.52% | 3.63% |
| v_proj | 2.44% | 2.56% | 2.46% |

So there is no gross module-type redistribution explaining the sign reversal.

There is a modest depth shift:

- late layers (19–27) contain 29.27% / **31.34%** / 29.22% of the mask;
- mask-weighted mean layer is 12.93 / **13.34** / 12.89.

The clearest individual outlier is **layer 27 shared MLP**. Layer 27 as a whole
contains **4.56%** of the negative seed's disagreement mask versus ~3.1% on the
positive seeds. Its shared-MMLP input and output modules are ~1.64× and ~1.58×
their positive-order normalized mask shares.

However, layer 27 accounts for only about 13% of the negative order's total
excess disagreement count versus the positive-order mean. It is a clue, not a
complete explanation.

## Global quantizer state does not explain the failure

Global histograms and scales at step 300 are nearly indistinguishable:

- direct Q3 zero fraction: ~31.815% on all three orders;
- Q9 zero fraction: ~12.30% on all three;
- Q9 extreme ±4 fraction: ~29.3% on all three;
- learned row-scale mean: ~0.02344 on all three.

Therefore the negative order is not obviously caused by a globally different
scale calibration or Q9 code histogram.

## Causal decomposition of the final endpoint

| Contribution | 1729 | 271828 | 424242 |
|---|---:|---:|---:|
| Full S gain vs D | +0.12710 | **−0.11431** | +0.06135 |
| Q9 off-mask contribution | +0.03030 | **−0.03651** | +0.00414 |
| Exact-M gain vs D | +0.09680 | **−0.07780** | +0.05721 |
| d=0.5 increment over exact-M | +0.05391 | +0.01951 | +0.03455 |
| M-d50 gain vs D | +0.15071 | **−0.05829** | +0.09176 |
| M-d50 advantage over matched random | +0.22380 | **+0.00766** | +0.10968 |

Positive means improvement in loss.

The negative order differs in two important ways.

### 1. Same-code / off-mask Q9 state is harmful

Replacing Q9's off-mask masters with direct masters improves 271828 from
S=5.84103 to M-exact=5.80453, recovering **0.03651 nats**.

On the positive orders, the Q9 off-mask state is neutral or helpful.

Thus 271828's failure is not confined entirely to the D-vs-S disagreement mask.

### 2. d=0.5 repairs geometry, but cannot rescue assignment quality

M-d50 improves 271828 by **0.05601 nats** relative to full S and by
**0.01951 nats** relative to M-exact. So standard interior placement still
helps.

But M-d50 remains **0.05829 nats worse than direct**, and is only **0.00766
nats better than matched random**.

By contrast, true-M d=0.5 beats matched random by 0.22380 and 0.10968 nats on
the two positive orders.

That weak true-vs-random separation is the strongest evidence that Q9's
**position/code selection itself is low quality on order 271828**, not merely
that its exact FP32 coordinates are bad.

## Continuation dynamics: not an endpoint-noise accident

The Q9 S arm remains worse than D on **every logged continuation batch** for
order 271828, from stage step 1 through 900.

Examples of S−D training loss:

- step 100: +0.641
- step 300: +0.329
- step 500: +0.264
- step 700: +0.241
- step 900: +0.109

M-d50 consistently reduces that gap, but generally remains above D too. It
briefly edges D on the logged step-600 batch (−0.009) and then loses the
advantage again.

On 1729, S overtakes D by logged continuation step 100 and stays generally
ahead. On 424242 it is roughly tied at step 100 and generally ahead thereafter.

Therefore seed 271828's negative held-out endpoint reflects a persistent
training-trajectory difference, not merely evaluation noise at step 1200.

## What does *not* distinguish the negative order

Two tempting explanations look weak:

1. **Assignment survival.** M-d50 survival at step 900 is 90.70%, 89.59%,
   90.57%. The negative order is only ~1 percentage point lower.
2. **Total continuation code churn.** S changes ~8.25–8.30% of starting Q3
   codes on all three; M-d50 changes ~6.85–6.93%. The negative order is not an
   obvious churn outlier.

So the problem seems to occur primarily **before / during Q9 assignment
discovery**, not because otherwise-good d=0.5 assignments are later erased.

## Best current hypothesis

A useful post-hoc hypothesis is:

> Q9 staging helps Granite when the 300-step Q9 preparation itself reaches a
> reasonably settled native-Q9 state and discovers a coherent set of alternative
> ternary commitments. On order 271828, Q9 remains comparatively unstable /
> under-settled, produces a larger and more disjoint disagreement mask, and its
> selected positions have little advantage over matched random positions.
> Standardizing depth repairs some continuous geometry but cannot turn weak
> assignment selection into a positive endpoint.

This is **not yet a proven mechanism**. It is inferred from only three Granite
orders and was discovered after seeing the outcomes.

## Potential cheap predictor for future preregistration

The cleanest existing-data discriminator is the **native-Q9 versus native-D
validation difference at step 300**:

- 1729: Q9 better by 0.1604 nats → final staging positive;
- 271828: Q9 worse by 0.2780 → final staging negative;
- 424242: Q9 better by 0.1378 → final staging positive.

The Q9/D gradient-norm ratio during preparation points the same way.

A future study could preregister these as *predictors* of whether a Q9 scout is
worth continuing into Q3. With n=3, they must not yet be used as validated
stopping rules.

## Bottom line

The negative seed does not look random or numerically broken. It looks like a
**bad Q9 search trajectory**:

1. Q9 preparation is visibly worse and higher-gradient before the switch;
2. its selected code set is more divergent from direct;
3. the mask is larger and somewhat shifted late, especially in layer 27;
4. both off-mask Q9 state and exact mask state are harmful;
5. d=0.5 helps, but true positions barely beat matched random;
6. survival and later code churn remain normal.

That narrows the next scientific question from “why did one endpoint flip?”
to **“what makes a Q9 preparation discover good versus bad ternary
commitments?”**
