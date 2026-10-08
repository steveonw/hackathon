# G1-3 — Granite 4.0 350M compressed mechanism test, order 1729

Job: `6ac6fe5fdf2184ac91ac6c1f`  
Pinned code: `c32fbc16f463039257d996a7b32c9eca5eada681`  
Hardware: A10G-small  
Seed/order: 1729  
Status: **complete — cross-family mechanism supported on first Granite order**

All preregistered construction assertions passed before continuation:

- S, M-exact, and M-d50 had **zero projected-Q3 Hamming**;
- those three equal-forward arms had identical pre-continuation validation diagnostics;
- Random-d50 exactly matched its deterministic planned target codes;
- Random-d50 used exactly `|M| = 15,035,048` positions;
- Random-d50 had no overlap with true M;
- its per-layer/source→target transition counts matched true M exactly.

## Final held-out endpoint

| Arm | Loss | PPL | Teacher top-1 | KL |
|---|---:|---:|---:|---:|
| D | 5.66205 | 287.74 | 27.47% | 2.48426 |
| S | **5.53495** | **253.40** | 28.04% | **2.38307** |
| M-exact | 5.56525 | 261.19 | **28.82%** | 2.38723 |
| M-d50 | **5.51134** | **247.48** | 28.72% | **2.35037** |
| Random-d50 | 5.73514 | 309.56 | 27.15% | 2.54962 |

## Causal effects

- full S gain vs D: **0.12710 nats/token**
- full S PPL reduction vs D: **11.94%**
- M-exact gain vs D: **0.09680 nats**
- M-exact recovery of full S: **76.2%**
- M-d50 gain vs D: **0.15071 nats**
- M-d50 recovery of full S: **118.6%**
- M-d50 recovery relative to M-exact gain: **155.7%**
- M-d50 PPL reduction vs D: **13.99%**
- Random-d50 gain vs D: **-0.07308 nats** (harmful)
- Random-d50 recovery of full S: **-57.5%**

True D-vs-S disagreement mask:
**6.0246%**
(15,035,048 /
249,561,088 targeted positions).

## Equal-forward interpretation

S, M-exact, and M-d50 start from the **same projected ternary forward model**
under the same original Q3 scales. They differ only in hidden FP32 master
geometry.

M-exact transfers Q9's exact FP32 masters only on the true disagreement mask
and recovers **76.2%** of full S's final loss advantage. This is clearly weaker
localization than SmolLM2's canonical ~96.4%, so do not claim the same
localization fraction across families.

However, replacing those exact Q9 masters with the same Q9-selected codes at
standardized interior depth `d=0.5` improves further: M-d50 recovers
**118.6%** of the full S advantage and beats the full-S endpoint itself
(5.51134 vs 5.53495).

This means Granite supports the broader mechanism:

> Q9 identifies useful **which-position / which-code** commitments, and robust
> interior placement of those commitments can be at least as useful as the
> exact Q9 continuous master state.

The exact fraction of full-Q9 benefit localized to M is family-dependent in
this first-order comparison.

## Position specificity

The matched Random-d50 arm uses the same number of positions, the same
per-layer allocation, and the same direct-source→Q9-target transition counts,
but at different positions outside true M.

It finishes at **5.73514 loss**, worse than direct by **0.07308 nats/token**.

Thus the benefit is not explained by merely applying the same number and types
of ternary transitions with the same depth. The **specific positions selected
by Q9 matter**.

## d=0.5 assignment survival

Q9-selected-code survival on true M:

| Continuation step | Current learned Q3 scale | Fixed original alpha0 |
|---:|---:|---:|
| 100 | 99.677% | 99.677% |
| 300 | 97.977% | 97.976% |
| 900 | **90.696%** | **90.693%** |

At step 900:
- target-zero survival: 90.394% current / 90.283% fixed-alpha0;
- target-nonzero survival: 90.999% current / 91.103% fixed-alpha0.

Learned-scale and fixed-alpha0 survival are nearly identical, again arguing
against scale drift as the explanation.

Granite's d=0.5 survival is lower than SmolLM2's seed-1729 ~97.3%, yet the arm
still surpasses full S in final loss. Survival should remain a strong dynamical
correlate/explanation, **not a proven sole mediator**.

## Conclusion

On the first Granite order, the SmolLM2 mechanism generalizes qualitatively:

1. the true D-vs-S disagreement positions carry most of the staging advantage;
2. the Q9-selected code identity plus moderate interior depth is sufficient to
   recover and exceed the full staged benefit;
3. matched random positions with the same transition structure are harmful;
4. exact Q9 within-region FP32 coordinates are not required.

The quantitative localization differs: exact-M recovery is 76.2% on Granite
versus ~96.4% canonical SmolLM2. Do not erase that difference.

## Caveats

- Granite mechanism evidence is seed/order 1729 only.
- No confirmatory Granite orders have been run.
- This remains WikiText-2; dataset generalization is untested.
- Free-running generation quality was not the target of G1-3.
- Do not claim a universal effect-size or universal mask-recovery fraction.

## Decision

G1-3 is scientifically interpretable and positive. The preregistered condition
for confirmatory Granite orders is satisfied, but no replication jobs are
launched by this result.
