# G1-9 — Granite-350M ternary QAT: Gaussian noise × gridward pull

**Status: GPU job completed successfully, all predeclared construction/audit checks passed. Large positive result for gridward interpolation only and combined arm; Gaussian noise alone slightly negative.**

- Completed HF job: [`6ac821ec095c5780892ff7f9`](https://huggingface.co/jobs/codeflash85/6ac821ec095c5780892ff7f9), 2026-10-08 23:32:17 UTC
- Code pin (before job): `6ce4c4056e8cfd3292c54f34bedc68bf7691688f`
- Script: `ternary_pet/g1_9_granite350m_gaussian_pull_seed271828.py`
- Raw result: `run_g1_9_granite350m_gaussian_pull_seed271828_2026-10-08.json`
- Before-launch preregistration: `../EXPERIMENT.md`
- Granite constant-LR historical direct seed271828 reference: **5.726725168526173** nats
- This is a **single already-known hard development order** (271828), not blind independent confirmation. No hyperparameter tuning on held-out evaluation.

## Experiment design and reproducibility

Four direct ternary/Q3 arms `D`, `G`, `P`, `GP` executed **300 initial + 900 continued optimizer updates** each, on the **same ordered WikiText-2 training chunks**, using exactly the same pretrained Granite 4.0 350M source, BF16-rounded initialization, FP32 latent/master weights, BF16 model/teacher compute, CE35 + teacher KL65 loss, AdamW beta (0.9,0.95), no weight decay, clip 1.0, and constant **`1e-4` learning rate**. Scales restored to the original rowwise Q3 values at global step300; optimizer restarted at that point for all four arms.

- **D**: clean hard Q3 with straight-through estimator; no special intervention.
- **G**: Gaussian additive noise **only during the training quantized forward**, scaled per-row `σ=0.04α` through global step900 then annealed linearly to zero at step1200; deterministic, noise-free hard-Q3 inference/validation.
- **P**: no Gaussian; after every 100th optimizer update through step900 (nine total), update latent masters **`W ← 0.9W + 0.1Q3(W)`** using current learned row scales, no extra optimizer updates or Adam resets beyond step300.
- **GP**: both schedules together, identical hyperparameters and update counts.

All arms passed the initial ternary quantizer smoke checks, finite-loss/gradient checks, 249,561,088-target-weight audit, four-arm presence, 300+900 update checks, and nine-gridward-update count assertions for P/GP. Evaluation is 8,192 held-out WikiText-2 tokens.

**D reproducibility is exact**: the new within-job D held-out loss is **5.726725168526173**, equal to the archived historical Granite direct 271828 loss to printed precision. Thus the comparison is not an unrelated baseline mismatch.

## Final held-out outcomes

Lower NLL and PPL are better. Gains are within-job `loss(D)−loss(arm)`.

| Arm | Held-out NLL | PPL | Gain vs D, nats | Teacher top-1 agreement | KL to teacher |
|---|---:|---:|---:|---:|---:|
| D | 5.72673 | 306.96237 | +0.00000 | 27.661% | 2.49071 |
| G | 5.74337 | 312.11455 | -0.01665 | 27.100% | 2.51042 |
| P | 5.48971 | 242.18679 | +0.23702 | 31.079% | 2.25599 |
| GP | 5.48785 | 241.73769 | +0.23887 | 31.042% | 2.23850 |

- **P improves direct by `0.23702` nats/token**, lowering perplexity by **21.10%** and raising top-1 teacher agreement by **3.42 percentage points**.
- **GP improves direct by `0.23887` nats/token**; the incremental benefit over P alone is only **0.00186 nats**. This tiny observed delta is not strong evidence Gaussian noise improves the gridward method.
- **G alone performs worse** than D by `0.01665` nats. Gaussian smoothing at this frozen normalized σ is not successful on its own.
- Difference-in-differences for Gaussian conditional on pull vs no-pull (improvement-sign convention): **0.01850 nats**. On only one order and one chosen noise strength, this does not establish a robust interaction.

The result is *largely attributable to the gridward intervention* in this factorial, not a demonstrated Gaussian-specific benefit.

For additional historical context only: the previously trained seed271828 constant-LR Q9→Q3 S endpoint was 5.84103 nats, and its d=0.5 mask intervention was 5.78502. New direct ternary P=5.48971 is substantially better than both under the shared 1200-step training budget, but this is a cross-run, nonfactorial comparison and does not constitute a new within-job Q9 experiment.

## Preparation, scale-reset, code movement, and transition-rate diagnostics

The 300-step native validation result and post-scale-reset diagnostic use the held-out-from-training *calibration/validation* chunks, not the 8,192-token final held-out test.

| Arm | Native val loss @300 | Post-reset val loss @300 | Projected source-code movement @300 | Source-code movement @1200 | Sampled flip probability / update, prep | Sampled flip probability / update, continuation |
|---|---:|---:|---:|---:|---:|---:|
| D | 5.64539 | 5.64945 | 5.339% | 10.335% | 0.00073 | 0.00074 |
| G | 5.67411 | 5.67719 | 5.389% | 10.377% | 0.00074 | 0.00074 |
| P | 5.50481 | 5.50404 | 4.008% | 5.281% | 0.00041 | 0.00009 |
| GP | 5.60858 | 5.60630 | 3.908% | 5.182% | 0.00037 | 0.00009 |

The 32,768 monitored weights are deterministically sampled across 16 layers
(2,048 entries/layer); their noise-free projected codes are inspected after
**each** optimizer update. These are sample estimates, not exact whole-model
per-update transition rates. Sampled continuation flip rates:

- D: 0.00074
- G: 0.00074
- P: 0.00009
- GP: 0.00009

Gridward P suppresses sampled continuation flip events by about
**87.2%** relative to D, while achieving better held-out performance.
D changes 10.335% of initial ternary codes
by the final checkpoint; P changes 5.281%
(roughly half). Fewer transitions are *associated with* better outcomes under
this intervention. The specific causal pathway (fewer flips vs repositioned
latent weights vs adaptive future STE dynamics) has not been isolated.

Immediate two-step code-reversal share among sampled continuation flips:
D **2.996%**,
G **2.767%**,
P **2.513%**,
GP **2.393%**.
The ratio changes less dramatically than the absolute flip frequency; do not
claim observed suppression is *specifically* due to correcting oscillatory
reversals. Perturbation can also reduce useful exploration, and lower flip
rate by itself is **not** a success metric.

The pull arms already improve native validation loss at step300:
D `5.64539`; P `5.50481`.
Thus this was not a post-training-only phenomenon.

## Interpretation and limitations

1. **One chosen development order.** The test was designed specifically
   on historically negative Granite seed271828. Before generalization claims,
   use the *identical frozen script settings* on previously positive Granite
   orders and then a genuinely new family/order, with appropriate statistical
   replication. Do not tune future hyperparameters against already-inspected
   test results.
2. **Prior art.** Gaussian latent perturbation and periodic interpolation
   toward quantized values appear in WinQ (ICML 2026). Our result is a
   positive adaptation of a published idea, **not original invention of
   gridward interpolation itself**.
3. **No mechanism proof from flip association.** Pull affects latent-master
   locations and future optimizer updates. Reduced flips may mediate the
   benefit, but causal isolation requires further controls, e.g. alternate
   interpolation strengths, equal flip-rate intervention and off-grid
   controls, all separately preregistered.
4. **No Q9 or random control in this factorial.** It establishes useful
   *direct ternary training* on one Granite order, not Q9 mask-position
   specificity. Preserve the earlier true-mask/random controls; consider
   margin-matched controls separately with feasible exact matching.
5. **Limited scope.** 1,200 single-chunk updates, one model and WikiText-2;
   frozen BF16 path and 8192-token held-out slice. Uncertainty on the tiny
   GP−P gap is not quantified.
6. **Subtle optimization detail:** gridward steps occur after the optimizer
   step and use the current learned row scales; pulls do **not** reset Adam
   state. All arms have the same scale and optimizer reset at global step300.
7. **Noise specific limitation:** sigma is normalized to each row's alpha,
   so numbers like 0.001 from other quantizer designs do not transfer
   directly. This single σ choice cannot rule out other noise strengths,
   though the current unambiguous evidence favors pull alone.

## Suggested next scientific action (not launched)

Prioritize a **frozen P-vs-D replication** under identical 1200 updates on
another Granite order such as 424242 and then a new family/order, before
tuning σ or claiming across-model effectiveness. The simple no-noise P arm
is already nearly indistinguishable from GP in held-out NLL and has lower
implementation complexity. A later mechanism test can measure boundary
margin distributions/row coordination and retain fair random controls.

**No additional GPU runs were launched as a consequence of G1-9 completion.**
