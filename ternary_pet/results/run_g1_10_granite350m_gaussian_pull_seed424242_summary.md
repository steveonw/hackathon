# G1-10 — Granite-350M frozen Gaussian × gridward factorial replication, order 424242

**Completed successful run** — one-line seed-only replication of G1-9. **Gridward pull alone again substantially improves final held-out ternary loss**. Gaussian-only is modestly beneficial on this order but inconsistent with seed271828; combining Gaussian with pull is slightly **worse** than pull alone.

- HF Job: [`6ac82c93fee2c900701711db`](https://huggingface.co/jobs/codeflash85/6ac82c93fee2c900701711db) — status **COMPLETED**, 2026-10-09 00:21:46 UTC
- Submitted 2026-10-08 23:51:47 UTC, A10G-small, no extra GPU jobs
- Immutable code pin: `e08659a51fd0d72ed85f01f8e7ce739283ca6c61`
- Script: `ternary_pet/g1_10_granite350m_gaussian_pull_seed424242.py`
- Parent G1-9 pin: `6ce4c4056e8cfd3292c54f34bedc68bf7691688f`
- **Verified full script differs by precisely one line**: `SEED=271828` → `SEED=424242`
- Preregistered: `../EXPERIMENT.md`
- Raw unedited final JSON: `run_g1_10_granite350m_gaussian_pull_seed424242_2026-10-08.json`

## Design and checks

Granite-4.0-350M, FP32 masters, BF16 student and teacher, pretrained source BF16-rounded, rowwise learned ternary scales, WikiText-2 training at matched seed-shuffled 128-token chunks, 1,200 updates, CE35+KL65, constant LR 1e-4, AdamW β=(0.9,0.95), grad clip1. Same 300 direct-Q3 preparation +900 Q3 continuation in each arm; original Q3 scales restored and fresh Adam at step300 for every arm. Four arms D (clean), G (train-time Gaussian σu=.04 row-normalized until step900 decaying to zero by1200), P (gridward 10% toward current Q3 code after optimizer updates 100,200,...900), GP (combined). All validation/test forward passes are deterministic hard-Q3 without Gaussian. No Q9 intermediate, no new random control.

**Audit all pass:** initial ternary/smoke check, D/G/P/GP present, every arm 300+900 steps and 8192 held-out tokens, finite final losses, P/GP each exactly nine gridward pulls, total 249,561,088 quantized target weights. D held-out loss `5.801920056343` **exactly reproduces** historical constant-LR Granite seed424242 D loss.

## Held-out G1-9 versus G1-10

| Arm | G1-9: 271828 loss | G1-10: 424242 loss | G1-10 PPL | Gain vs G1-10 D |
|---|---:|---:|---:|---:|
| D | 5.726725 | 5.801920 | 330.93 | — |
| G | 5.743370 | 5.756362 | 316.20 | +0.045558 |
| P | 5.489709 | 5.528112 | 251.67 | +0.273808 |
| GP | 5.487853 | 5.535340 | 253.49 | +0.266580 |

**Primary predeclared finding:** pull-only P improves seed424242 direct by **0.273808 nats per token**, reducing PPL from **330.93 to 251.67 (23.95% less)**. Teacher top1 agreement: D **27.67%**, P **29.49%**; KL to teacher D **2.60690**, P **2.32513**. GP also improves D by **0.266580** but **P is better than GP** by 0.007228 nats.

Gaussian alone improves D here by **0.045558**. This contrasts with G1-9 where Gaussian alone **worsened** D by 0.016645. The noise effect is not consistent across tested orders and it does not reliably add to P.

## Independent-order comparison — same fixed settings

| Order | D loss | P loss | D−P improvement | PPL reduction | Sampled continuation flip-rate reduction | D−G improvement | P−GP advantage (+ if GP better) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 271828 | 5.726725 | 5.489709 | +0.237016 | 21.10% | 87.16% | -0.016645 | +0.001856 |
| 424242 | 5.801920 | 5.528112 | +0.273808 | 23.95% | 86.89% | +0.045558 | -0.007228 |

Mean D−P gain across the two *previously selected* Granite orders:
**0.255412 nats**.
Do **not** treat the two known historical seed orders as fully independent
new samples or claim a statistical confidence interval from n=2.

Under both orders, P outperforms D by a meaningful margin whereas
Gaussian's effect and incremental GP effect change sign. The evidence
therefore supports **gridward-interpolation robustness within the
Granite architecture across two tested orderings**, not Gaussian as
the critical mechanism, not a newly discovered general method, and
not cross-family transfer yet. Gaussian + pull requires a separate
interaction explanation before any positive claim.

## Seed424242 dynamics and validation

| Arm | Native val @300 | Post-scale-reset val @300 | Projected code movement vs initial @300 | Final movement vs initial @1200 | Sampled flips / position / continuation update | Immediate reversals per sampled flip |
|---|---:|---:|---:|---:|---:|---:|
| D | 5.68460 | 5.68631 | 5.34% | 10.33% | 0.00072842 | 2.89% |
| G | 5.70947 | 5.70135 | 5.38% | 10.36% | 0.00071977 | 2.89% |
| P | 5.56780 | 5.56723 | 4.05% | 5.37% | 0.00009549 | 2.31% |
| GP | 5.60856 | 5.60551 | 3.90% | 5.18% | 0.00008613 | 2.60% |

Deterministic sample of 32,768 weights across 16 layers, monitored
after *every* optimizer update; these are **sampled estimates**, not
whole-network exact per-step flip counts. On 424242, clean-code
flip frequency for P is **86.89% lower** than D. Source-code differences at final checkpoint are D 10.33% versus P 5.37%. Immediate reversal *fractions* are different much less dramatically; do not conflate fewer all-code transitions with specifically fewer pathological reversals. Gridward pulls also change latent master positions and future optimization; code stabilization is an associated phenomenon, not yet an isolated causal mediator.

## Explicit raw-output metadata caveat

To enforce **exactly one changed seed line**, the G1-10 script and raw JSON
retain the parent's internal `kind="g1_9_granite350m_gaussian_pull_factorial"`
and `event="g1_9_*"` logging labels. They also retain the
parent's `effects.historic_D_loss_271828=5.726725168526173` and
`effects.D_minus_historic_D_loss=+0.0751948878` comparator fields.
**Those two fields refer to the OLD SEED, not to the new seed's correct
historical direct baseline**. They should NOT be used to assess G1-10
reproducibility. Use `seed=424242`, job ID, immutable script commit,
and the independently archived constant-LR historical D at 424242,
**5.801920056343079**. The G1-10 in-job D also equals that value
exactly. We preserve the generated raw JSON unchanged to maintain
scientific provenance; this analysis is the explicit correction.

## Limitations and next-stage recommendation (not launched)

- Existing observed Granite order 424242, chosen prospectively as the
  next replication after seeing seed271828's strong P result. This is
  within-model/order replication, not blind cross-model confirmation.
- Single 8,192-token held-out WikiText-2 slice, short 1,200-step
  experiment, specific pretrained Granite 350M and BF16 inference.
- The pull method and additive latent Gaussian noise have relevant
  prior art (WinQ, ICML2026); do not claim invention of gridward interpolation.
- G1-10 is entirely direct Q3, not a test of Q9 staged preparation or
  new true-mask-vs-random position specificity. Previous random controls
  remain intact.
- Strong association between better loss and reduced code transitions,
  but this alone does not identify the causal mechanism.
- Best next scientific decision is a **predeclared frozen-setting
  cross-family confirmation**, preferably direct-Q3 versus P and perhaps
  the unchanged four arms on SmolLM2, *before* retuning Gaussian σ or
  the interpolation strength λ on already inspected Granite held-out data.
  This needs separate authorization and a fresh run; none launched here.
