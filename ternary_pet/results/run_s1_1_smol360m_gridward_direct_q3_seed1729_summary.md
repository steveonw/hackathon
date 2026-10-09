# S1-1 — SmolLM2-360M gridward pull versus tuned direct ternary: negative cross-family transfer

**Status: HF job completed successfully; direct-Q3 reproducibility exact; the
predeclared fixed gridward-pull rule significantly worsens Smol held-out loss.**
This is a scientific negative, not an execution failure.

- Completed HF job: [`6ac83c3bfee2c90070171b1a`](https://huggingface.co/jobs/codeflash85/6ac83c3bfee2c90070171b1a);
  terminal `COMPLETED`, **2026-10-09 01:18:20 UTC**.
- Script pinned before launch: `a5634bce459092a503e7534e6b89b6d3a019548b`.
- `ternary_pet/s1_1_smol360m_gridward_direct_q3_seed1729.py`
- Parent source: v9 tuned Smol direct Q3 `5c088e58539b2dede93df57ac3f72dbe0480a028`.
- Raw unmodified job data: `run_s1_1_smol360m_gridward_direct_q3_seed1729_2026-10-09.json`.
- Preregistration, primary hypotheses and no-retuning rule: `../EXPERIMENT.md`.
- The original v9 Smol training/data/model/STE/optimizer utilities were verified
  byte-identical to the pinned parent after its docstring. S1-1 adds a
  two-arm matched runner, pull and deterministic sampling telemetry, not
  different training hyperparameters.

## Controlled experiment

Model `HuggingFaceTB/SmolLM2-360M-Instruct`,
seed/order 1729 (historically used, **not blind**).
Two **direct-Q3/ternary** arms over the same 1,200 training-step
opportunities, with no Q9 intermediate:

- D: Smol v9's previously best-tested direct schedule: first 100 steps
  linear warmup to `1e-3`, global cosine decay to `1e-4` by step1200.
- P: identical Smol direct schedule, with **nine 10% gridward
  master-weight pulls** after optimizer steps 100,200,...,900:
  `W <- 0.9 W + 0.1 Q3(W)`. Uses current trainable rowwise scales.
  Does **not** directly alter the scale or Adam optimizer state.
- Both retain the established v9 BF16-rounded FP32 source/master
  weights, FP16 autocast teacher/student and AMP GradScaler, AdamW,
  CE35+teacher KL65, gradient clipping, frozen nonquantized parameters,
  identical ordered WikiText-2 chunks, continuous Adam without
  a step300 reset, and same hard-ternary deterministic evaluations.
- No Gaussian noise in either arm, and no Q9 preparation.
- The finite 8,192-token WikiText-2 test is untouched for selection.

## Final held-out test

| Arm | Loss (nats/token) | PPL | Teacher top-1 | KL to teacher |
|---|---:|---:|---:|---:|
| **D tuned direct** | **5.595722** | **269.27** | **29.87%** | **2.664031** |
| P gridward pull | 5.845746 | 345.76 | 27.28% | 2.933238 |

- Primary predeclared effect `D_loss−P_loss`:
  **-0.250024 nats/token**, i.e.
  **P is 0.250024 nats WORSE**.
- PPL increases **28.41%**,
  269.27 → 345.76, instead of decreasing.
- Teacher top-1 agreement falls
  **2.59 percentage
  points**; KL-to-teacher worsens **0.269207**.
- **Within-job D exactly reproduces historical v9 tuned direct
  seed1729 test loss** `5.595722187310457`.
- Historical seed1729 schedule-matched Q9→Q3 held-out result
  `4.9009853675961494` is context only; there was no
  Q9 arm in this two-arm experiment.

## All audit checks passed

- HF reports `COMPLETED`, full `FINAL_JSON_BEGIN` and
  `FINAL_JSON_END` present and parsed.
- 2 arms, all **1,200 step opportunities** each; **six AMP-skipped
  optimizer updates in both**, hence **1,194 effective Adam updates**
  per arm. No skipped-step imbalance, and all nine P pulls occur
  after successful optimizer updates.
- Nine pulls in P and zero in D, schedule endpoints correct, exactly
  8,192 test tokens per arm, finite losses.
- Q3 smoke confirms no immediate code change or scale change during
  a convex pull and closer FP32 masters to the existing prototypes.
- Both arms have 314,572,800 identically targeted ternary weights.
- 32,768 deterministic sampled positions across 16 layer indices,
  monitored after each of the 1,200 training steps; monitor RNG
  independent of training RNG.
- **All nine saved validity checks** returned `true`.

## Dynamics: strong code suppression coexists with worse final quality

| Metric | Tuned direct D | Gridward P |
|---|---:|---:|
| Clean sampled flip probability per weight/update (all 1200 steps) | 0.000435054 | 0.000093918 |
| Sampled flips in total | 17107 | 3693 |
| Immediate two-step reversal fraction among flips | 3.00% | 2.30% |
| Final codes different from initial (exact full target) | 6.96% | 3.45% |

Pull reduces sampled per-step code flips by **78.4%** and
roughly halves how many final codes differ from initialization, yet
hard-Q3 held-out quality is worse. That **directly refutes** the
simple cross-model claim that fewer code changes necessarily
improve ternary training. The repeated code suppression may be
blocking useful exploration/late corrections, but that specific
causal pathway was not isolated.

Time course from *training-heldout validation only*, no use of
test data for checkpoint choice:

| Global step | D validation loss | P validation loss | P−D (positive=worse P) | D sampled flip rate in preceding window | P sampled flip rate |
|---|---:|---:|---:|---:|---:|
| 300 | 5.988094 | 5.976672 | -0.011422 | 0.00076721 | 0.00017914 |
| 600 | 5.740323 | 5.670643 | -0.069680 | 0.00053101 | 0.00004059 |
| 900 | 5.415100 | 5.582527 | +0.167428 | 0.00024750 | 0.00000244 |
| 1200 | 5.255937 | 5.577377 | +0.321440 | 0.00008087 | 0.00000122 |

- At step **300** P is very slightly better on validation
  (5.976672 vs
  5.988094).
- At step **600**, P remains better on validation by approximately
  0.069680 nats.
- By **900**, the sign reverses: P has
  0.167428 worse loss.
- By **1200**, P's validation loss is
  0.321440 worse.
- During the final windows (1000–1200), P's sampled
  hard-code flip counts are approximately zero while
  D retains continued low-level code movement and improves
  substantially more in validation. These observations are
  compatible with **premature commitment under Smol's
  different optimizer/LR trajectory**; they do not prove
  the cause.

## Granite comparison, keeping model and LR confounding explicit

Positive `D−P` gain means pull helps.

| Model/order | Direct D loss | P loss | D−P gain | Interpretation |
|---|---:|---:|---:|---|
| Granite-350M seed271828 (G1-9) | 5.726725 | 5.489709 | +0.237016 | P improves |
| Granite-350M seed424242 (G1-10) | 5.801920 | 5.528112 | +0.273808 | P improves |
| SmolLM2-360M seed1729 (S1-1) | 5.595722 | 5.845746 | -0.250024 | P degrades |

In Granite G1-9/G1-10, the same nine 10%-gridward pulls
were accompanied by lower loss under its **constant 1e-4**
LR, and original-scale restoration/fresh Adam at step300.
Smol S1-1 instead uses its tested **warmup to 1e-3,
cosine to 1e-4 and no step300 reset**. Thus this comparison
is a genuine cross-architecture test **of the same rule under
family-appropriate training protocols**, but it does
**not separate architecture from LR/reset-schedule interactions**.
It would be unwarranted to assert Smol as inherently
incompatible with all gridward training, or to conclude
a smaller λ/shorter window would necessarily help.

Smol's previously tested v13 direct-code prototype
firmness controls were also negative, which is qualitatively
consistent with this result; they are **not** exactly
the same gridward treatment and should not be pooled as
a quantitative replication.

## Interpretation and research constraints

1. **Failure to transfer the frozen rule.** Under the
   previously selected strong direct Smol LR schedule,
   10%-every-100-steps-through-900 gridward significantly
   degrades held-out quality, notwithstanding substantial
   code-flip suppression. This is a valid negative.
2. **Neither architecture nor LR alone established.**
   Granite and Smol differ in architecture and optimization
   schedules, especially the 1e-3 Smol peak and Granite
   step300 optimizer reset.
3. **No Gaussian in S1-1.** This does not test Gaussian's
   Smol effectiveness, or Q9 staging/rescued assignments.
4. **No test-based retuning.** The user authorized one
   two-arm job. Do not automatically launch repeats,
   tune pull rate, choose cutoff steps or look for favorable
   seeds based on the negative held-out result.
5. **Prior art.** Gridward pull has a close published
   analogue in WinQ (ICML 2026); this negative transfer
   informs architecture/schedule sensitivity but does
   not invalidate its other settings.
6. **Short-run scope.** One historically observed Smol
   training order, 1,200 step opportunities with six
   matching AMP skips, one 8,192-token held-out slice,
   full-precision latent masters and true ternary forward.

**Best next scientific activity: analyze existing code-transition
and boundary-margin logs across Smol and Granite first, without
more GPU usage, to preregister an adaptive rule that permits
valuable code changes rather than minimizing flips blindly.**
If future compute is approved, any λ/cutoff comparisons
should use training-heldout validation and fresh independent
orders, with proper original direct controls.

No further GPU jobs were launched.
