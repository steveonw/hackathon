# Experiment protocol

## Question

Does the path into ternary space affect how much behavior a pretrained model
retains?

### v1 — independent 27 -> 9 -> 3

The first experiment used independently rescaled 27-, 9-, and 3-level weight
grids with a tiny recovery budget. It did not show a staged advantage and was
numerically unstable near ternary.

Remote run: \`6ac52475404719ba37661c8b\`.

### v2 — strict nested balanced-ternary ancestry

v2 used a literal balanced-ternary hierarchy with one inherited scale:

- 27 child states,
- 9 parent states,
- 3 ternary grandparent states.

Raw direct and nested projections ended identically, as expected for
deterministic nested projection without adaptation. Recovery became stable, but
direct ternary QAT still beat the staged route.

Remote run: \`6ac527e5404719ba37661dc9\`.


## v3 — transition schedules and high-precision preparation

v3 tests whether the path into ternary space matters more than a hard sequence
of discrete codebooks. All variants start from the same BF16-rounded checkpoint
and keep trainable FP32 master weights. The ternary forward quantizer uses a
learnable per-output-row scale, and recovery uses WikiText-2 with CE + teacher
KL distillation.

The schedules are:

- direct: 600 fully ternary steps;
- soft: a 200-step quantization-strength ramp followed by 400 ternary steps;
- up/down equal-total: 200 full-precision adaptation steps followed by 400
  ternary steps;
- up/down equal-final: 200 full-precision adaptation steps followed by 600
  ternary steps.

The up/down treatment is the practical version of a proposed 16->24->3 idea.
FP32 has a 24-bit significand; merely upcasting BF16 cannot recover discarded
bits, so the experimental variable is **adaptation while the master weights are
stored and updated at FP32 precision** before the ternary switch.

Remote run: Hugging Face Job `6ac52df9404719ba37661fa1`, T4 small.
See `results/run_v3_summary.md` for the outcome.


## v4 — persistent-shadow 9 -> 3 with flip-rate diagnostics

v4 is designed to answer whether staging helps **once the discrete weights are
actually moving**.

Changes from earlier runs:

- one FP32 shadow/master weight tensor is kept for the whole run;
- no hard commit at the 9 -> 3 transition;
- Adam optimizer state is preserved across the transition;
- non-quantized parameters are frozen;
- the ternary scale is initialized from rowwise absmean and remains learnable;
- actual discrete code flips are measured every 50 steps;
- one fixed WikiText-2 test slice (8192 tokens) is used.

Before the main comparison, a short direct-ternary LR sweep tests
`2e-5`, `5e-5`, and `1e-4`; the lowest held-out diagnostic loss is selected.

Main conditions:

1. direct ternary for 1200 steps;
2. 9-state QAT for 300 steps -> ternary QAT for 900 steps, with the same FP32
   shadows and optimizer state;
3. direct ternary for 900 steps.

This lets condition 2 be compared both against equal total compute (1200 steps)
and against an equal final ternary-stage budget (900 steps).

Remote run: Hugging Face Job `6ac539b5404719ba376621a2`, T4 small,
45-minute hard timeout.


## v5 — 5x training-volume test

v5 keeps the replicated v4/v4b recipe fixed and increases only training volume.

- model: `HuggingFaceTB/SmolLM2-360M-Instruct`
- fixed evaluator: WikiText-2 test, 8192 tokens
- training set: 6000 unique 128-token chunks, shuffled with seed 1729
- LR: `1e-4`
- non-quantized parameters frozen
- persistent FP32 shadow weights
- optimizer state preserved through the 9 -> 3 transition
- absmean-initialized learnable quantizer scale
- flip-rate logging every 250 steps

Equal-total-compute pair:

1. direct: 6000 ternary steps;
2. staged: 1500 9-state steps -> 4500 ternary steps.

Each treatment therefore processes ~768k training tokens. The successful 25/75
stage split from v4/v4b is unchanged.

Pinned Git commit:
`23157f7ce03f6bbbf944806574b18c133843f049`

Hugging Face Job:
`6ac56d99fbc85ba6823ba03f`

Hardware: A10G small. Hard timeout: 75 minutes.


### v5 outcome

Job `6ac56d99fbc85ba6823ba03f` completed successfully.

Held-out results:

- BF16 source: loss 3.6854 / PPL 39.86;
- direct 6000@3: loss 4.4463 / PPL 85.31 / top-1 41.44% / KL 1.522;
- staged 1500@9 -> 4500@3: loss 4.2493 / PPL 70.05 / top-1 43.64% / KL 1.330.

The staged treatment therefore retained a **0.197 nats/token** advantage and
17.88% lower PPL at 5x training volume, but the loss gap was about **74% smaller**
than the 0.765-nat mean seen in v4/v4b.

The first post-preparation ternary training-batch loss was ~50.4% lower than
direct's first ternary batch, but those were different chunks. v6 later adds a
clean same-data Q9 -> Q3 transition measurement.

Interpretation: v5 supports a large early optimization/head-start effect with a
smaller residual advantage at 6000 updates. It does not establish a permanently
better asymptotic ternary basin.


## v6 — causal dissection of the 9 -> 3 transition

v6 returns to the v4-scale protocol to locate where the staged advantage is
stored rather than spending more compute on another scale-up.

Shared setup:

- model: \`HuggingFaceTB/SmolLM2-360M-Instruct\`
- seed/order: 1729
- 1200 shuffled 128-token training chunks
- fixed 8192-token WikiText-2 evaluator
- LR: \`1e-4\`
- frozen non-quantized parameters
- persistent FP32 shadow/master weights
- absmean-initialized learnable rowwise scales
- CE + teacher-KL objective

One common 9-state checkpoint is trained for 300 steps. That exact checkpoint
then branches into four 900-step ternary continuations:

1. \`carry_all\`: prepared masters + prepared scales + full Adam state;
2. \`reset_adam\`: prepared masters + prepared scales + fresh Adam;
3. \`masters_only\`: prepared masters + initial scales + fresh Adam;
4. \`masters_plus_weight_adam\`: prepared masters + initial scales + Adam state
   retained only for master weights; scale optimizer state is reset.

A fresh \`direct_1200\` ternary run remains the matched control.

### New transition diagnostics

Before any ternary update, the shared prepared checkpoint is evaluated on the
same fixed diagnostic chunks under Q9 and Q3. This gives a clean same-data
quantizer-switch penalty.

v6 also compares initial ternary codes against the prepared masters projected
to ternary in two ways:

- using the learned prepared scales;
- using the original initial scales.

The second comparison isolates how often 9-state training moved FP32 masters
across *future ternary decision boundaries* without allowing scale changes to
explain the crossing.

Pinned Git commit:
\`ad5376d4db03c0e467a47353293220474d6a8b42\`

Hugging Face Job:
\`6ac5816ffbc85ba6823ba8ec\`

Hardware: A10G small. Hard timeout: 50 minutes.


### v6 outcome

Job `6ac5816ffbc85ba6823ba8ec` completed successfully.

The causal branches were nearly identical:

- carry all: loss 5.2059 / PPL 182.34;
- reset Adam: loss 5.2031 / PPL 181.83;
- prepared masters + original scales + fresh Adam: loss 5.1990 / PPL 181.09;
- prepared masters + original scales + master-weight Adam only: loss 5.1799 /
  PPL 177.67;
- direct 1200@3: loss 5.8747 / PPL 355.90.

The full four-branch spread is only 0.026 nats/token. Thus learned scale state
and Adam carryover are not necessary for the staged advantage at this budget.

On the fixed diagnostic set, before any ternary update:

- initial Q3 loss: 15.4784;
- after 300 Q9 updates under Q9: 4.9320;
- same prepared weights immediately projected to Q3: 9.1771.

The clean Q9 -> Q3 switch therefore costs 4.2451 nats/token, but the prepared Q3
projection is already 6.3013 nats/token better than the initial Q3 projection.

Future ternary assignments changed by 0.6769% with learned prepared scales and
0.6755% with the original scales held fixed. This shows that the dominant
effect is movement of FP32 master weights across future ternary decision
boundaries, not scale calibration.

Interpretation: v6 materially strengthens the "reorganization before
constraint" mechanism within this setup. v5 still indicates that much of the
benefit behaves like a finite-budget head start rather than a proven asymptotic
advantage.


## v7 — equal-compute preparation geometry

v7 asks whether the Q9 advantage at 300 updates is already visible in the
immediate ternary checkpoint, or whether Q9 mainly leaves the continuous
masters in a more trainable within-bin geometry.

Three preparation arms consume the exact same first 300 shuffled chunks:

1. \`q3_300\`: direct ternary QAT for 300 updates;
2. \`q9_300\`: 9-state QAT for 300 updates;
3. \`fp32_300\`: unquantized-weight warmup for 300 updates, with FP32 master
   storage/updates and the same autocast compute used elsewhere.

At step 300, every master state is projected through the **same original Q3
rowwise scales** and evaluated on the same fixed 24-chunk diagnostic set.

Then all three receive an identical 900-step ternary continuation:

- original Q3 scales restored;
- fresh Adam;
- same remaining 900 training chunks;
- same LR \`1e-4\`.

Thus the only state carried from preparation into the continuation is the FP32
master weights.

### New diagnostics

v7 records:

- native step-300 quality in Q3, Q9, or unquantized space;
- equal-compute fixed-scale Q3 quality for all three master states;
- Q3 code changes versus the initial source;
- pairwise overlap/Jaccard of the exact weights whose Q3 codes changed;
- pairwise Q3 Hamming distance among the prepared states;
- distance of every prepared master weight to the nearest future Q3 decision
  threshold at normalized \`w/alpha = +/-1/3\`;
- final 1200-update held-out metrics and fixed generation prompts.

The decisive distinction is:

- if Q9's fixed-scale Q3 projection is already better than direct Q3@300, Q9
  produces a **better ternary entry state**;
- if direct Q3@300 is as good or better but Q9 wins after the shared 900-step
  continuation, Q9 produces **better subsequent trainability / within-bin
  master geometry**;
- the FP32 arm tests whether the intermediate discrete grid is special or
  whether a less-constrained warmup is sufficient.

Pinned Git commit:
\`3aa494a4c8418052ed9e13e0de4f97c692a29fc7\`

Hugging Face Job:
\`6ac58e76fbc85ba6823bad78\`

Hardware: A10G small. Hard timeout: 45 minutes.


### v7 outcome

Job `6ac58e76fbc85ba6823bad78` completed successfully.

At equal 300-step compute, projected through the same original Q3 scales:

- direct Q3@300 loss: **6.5589**
- Q9-prepared Q3@300 loss: **9.2264**
- FP32-prepared Q3@300 loss: **14.7346**

Therefore Q9 does **not** provide a better immediate ternary entry checkpoint.

After the identical 900-step Q3 continuation (fresh Adam, original Q3 scales,
same remaining chunks):

- Q3 -> Q3: loss **5.8724**, PPL **355.09**
- Q9 -> Q3: loss **5.1938**, PPL **180.15**
- FP32 -> Q3: loss **6.0311**, PPL **416.19**

Q9 therefore finishes **0.6786 nats/token** better than direct despite starting
the continuation from a substantially worse Q3 checkpoint. FP32 warmup does not
reproduce the effect.

At step 300, direct Q3 changes 0.6933% of future Q3 assignments and Q9 changes
0.6755%, but their changed-position sets have only **19.64% Jaccard overlap**.
When both change the same weight, they choose the same final ternary code 100%
of the time.

Global distance-to-Q3-threshold distributions are nearly identical across
initial, Q3, Q9, and FP32 states.

Interpretation: v7 favors a Q9-specific **trainability / weight-selection
geometry** mechanism. Q9 appears to reposition a different subset of continuous
masters such that later Q3 optimization is much more productive, even though
the immediate Q3 projection is worse.


## v7 replication pair — seeds 271828 and 424242

Purpose: test whether the v7 mechanism result survives independent shuffled
training orders without changing any scientific setting.

The replication scripts are exact copies of the pinned v7 protocol with only
`SEED` changed. This changes the permutation of the same 1200 training chunks.

Shared pinned commit containing both scripts:
`6caf3a98d485ed1fd49e22b915ddb6b175578420`

Replication scripts:

- `replications/smollm2_v7_seed271828.py`
- `replications/smollm2_v7_seed424242.py`

Hugging Face jobs:

- seed 271828: `6ac59807fbc85ba6823bb060`
- seed 424242: `6ac59809fbc85ba6823bb063`

Hardware: A10G-small. Hard timeout: 45 minutes each.

No result-dependent protocol changes are allowed. The confirmatory questions are:

1. At step 300, is Q9's fixed-Q3 projection still worse than direct Q3@300?
2. After the common 900-step Q3 continuation, does Q9 still finish better?
3. Does FP32 warmup still fail to match Q9?
4. Do Q3 and Q9 still change similar fractions of future Q3 codes while
   selecting substantially different positions?


### v7 replication pair outcome

Both preregistered replication jobs completed successfully:

- seed 271828: `6ac59807fbc85ba6823bb060`
- seed 424242: `6ac59809fbc85ba6823bb063`

Together with the original seed 1729, the central v7 effect now replicates in
**3/3 training orders**.

Equal-compute fixed-Q3 loss after the first 300 updates:

| Seed | Direct Q3 | Q9-prepared Q3 | Q9 disadvantage |
|---:|---:|---:|---:|
| 1729 | 6.5589 | 9.2264 | +2.6676 |
| 271828 | 6.6449 | 8.8909 | +2.2460 |
| 424242 | 6.5824 | 9.4334 | +2.8510 |

Final held-out loss after the identical fresh-Adam 900-step Q3 continuation:

| Seed | Direct | Q9 prep | Q9 gain |
|---:|---:|---:|---:|
| 1729 | 5.8724 | 5.1938 | 0.6786 |
| 271828 | 5.9802 | 5.2417 | 0.7384 |
| 424242 | 5.9469 | 5.3008 | 0.6461 |

Aggregate:

- mean final Q9 loss advantage: **0.6877 nats/token**;
- mean paired PPL reduction: **49.69%**;
- mean top-1 gain: **+6.86 percentage points**;
- mean Q3-vs-Q9 changed-position Jaccard: **19.66%**.

The changed-position result is highly stable: Q3 and Q9 move similar fractions
of future ternary decisions (~0.68%) but mostly different specific weights.

FP32 nuance: the FP32-prepared arm remains far behind Q9 in all three runs, but
it is not always worse than direct. On seed 271828 it finishes 0.0292 nats/token
better than direct; on the other two seeds it is worse. Therefore the safe
replicated conclusion is that generic FP32 warmup **does not reproduce the Q9
advantage**, not that FP32 is universally harmful.

Interpretation: the Q9-specific trainability / weight-selection geometry result
is now replicated across three training orders under the current
SmolLM2/WikiText recipe.


## v8 — signed pre-loading diagnostic

Purpose: test whether the replicated Q9 trainability advantage is explained by
**directional preparation** of FP32 masters toward the ternary transitions they
make later.

Pinned script:
`ternary_pet/smollm2_v8_signed_preload.py`

Pinned Git commit:
`356bff9769c96faa0ca139f9cba088fc1a52c2c8`

Hugging Face job:
`6ac5a0b1404719ba37664653`

Hardware: A10G-small. Hard timeout: 35 minutes.

Protocol:

- seed/order 1729;
- arm A: 300 Q3 updates;
- arm B: 300 Q9 updates;
- then both restore original Q3 scales, reset Adam, and consume the same final
  900 Q3 chunks;
- same LR, teacher, objective, evaluator, and frozen non-quantized parameters.

Primary per-weight statistic, restricted to weights whose Q3 code changes during
the 900-step continuation:

```
aligned_preload =
sign(final_Q3_code - step300_Q3_code)
* (W_step300 - W_initial)
 / alpha_initial
```

Positive values mean preparation had already moved the continuous master in the
same direction as its later ternary code transition.

Matched-set controls prevent selection bias:

1. On the exact positions/directions of Q9's later ternary transitions, compare
   Q9-prep aligned displacement against Q3-prep aligned displacement.
2. On the exact positions/directions of Q3's later transitions, compare
   Q3-prep aligned displacement against Q9-prep aligned displacement.

Interpretation preregistered before launch:

- if Q9 shows substantially more positive/aligned displacement on its own
  future-transition set than Q3 does at those same positions, that supports a
  **directional pre-loading** mechanism;
- if the two prep paths are similar on matched positions, then the benefit is
  more likely due to some other property of the selected subset or its local
  optimization geometry;
- a positive result does not establish causality by itself; it identifies a
  more specific geometric correlate to test next.


### v8 interpretation refinement added before result review

A reviewer pointed out a post-selection issue: each future-flip set is defined
after that arm's own preparation and continuation. Therefore an arm can look
aligned on its own later-flipping weights partly because those weights were
already nudged toward a boundary.

For that reason, own-set alignment alone is not sufficient evidence.

Read all four matched quantities together:

1. Q9 prep on Q9's later-flip set;
2. Q3 prep on Q9's later-flip set;
3. Q3 prep on Q3's later-flip set;
4. Q9 prep on Q3's later-flip set.

Interpretation rule:

- if each arm mainly wins on its own later-flip set by similar margins, treat
  that as compatible with post-selection;
- stronger evidence for Q9-specific directional preparation requires a clear
  asymmetry across the matched comparisons;
- if the cross-set comparisons do not favor Q9, do not claim a Q9-specific
  directional-preloading mechanism.

The current v8 script already records distributional statistics beyond the mean:
median, q10/q25/q75/q90, fraction positive/negative, mean absolute displacement,
and a per-layer breakdown.

The running v8 script does not include a random/non-flipper null baseline. Since
v8 is already running from a pinned commit, do not modify the script after
launch. If the four-way result is positive or ambiguous, add a null comparison
in a follow-up rather than changing v8 post hoc.

If v8 gives a strong correlational signal, the next causal test should be an
intervention that selectively removes the largest Q9-aligned preparation
movements before the common Q3 continuation and measures how much of the Q9
advantage is lost.


### v8 outcome

Job `6ac5a0b1404719ba37664653` completed successfully.

The seed-1729 v7 performance result reproduced:

- direct Q3 step-300 diagnostic loss: **6.5589**
- Q9-prepared Q3 step-300 diagnostic loss: **9.2264**
- final direct loss: **5.8724**
- final Q9 loss: **5.1938**

The signed-preload mechanism itself was **not supported** by the preregistered
four-way matched comparison.

On Q9's later-flip positions/directions:

- Q9 prep mean aligned displacement: **0.002328**
- Q3 prep on the same positions/directions: **0.000873**
- Q9 advantage: **+0.001455**
- Q9 is more aligned on 53.44% of matched weights.

On Q3's later-flip positions/directions:

- Q3 prep mean aligned displacement: **0.003130**
- Q9 prep on the same positions/directions: **0.001439**
- Q3 advantage: **+0.001691**
- Q3 is more aligned on 53.77% of matched weights.

Thus each arm mainly wins on its **own** future-flip set by a similar margin,
which is precisely the pattern preregistered as compatible with post-selection
/ generic trajectory alignment. The direct-Q3 own-set advantage is slightly
larger in mean magnitude.

Own-set future-changer alignment is real in both arms:

- direct Q3: mean 0.003130, median 0.002226, 59.11% positive;
- Q9: mean 0.002328, median 0.001735, 57.28% positive.

Among future-changing weights that had not yet changed Q3 code during the first
300 updates, both arms show stronger positive preparation alignment (~70%
positive). Weights that had already changed code during preparation show
strongly negative alignment with their later continuation change in both arms,
suggesting a generic crossing/reversal phenomenon rather than a Q9-specific
effect.

Interpretation: v8 **falsifies the simple Q9-specific directional-preloading
story**. The replicated Q9 trainability / weight-selection effect remains, but
its causal mechanism is not explained by a scalar "already moving toward the
eventual threshold" statistic.

Canonical files:

- `results/run_v8_2026-10-07.json`
- `results/run_v8_summary.md`


## v9 — direct-Q3 baseline tuning

Purpose: test whether the replicated Q9 finite-budget advantage survives a
stronger direct-Q3 optimizer schedule.

Pinned script:
`ternary_pet/smollm2_v9_direct_q3_tuning.py`

Pinned script commit:
`5c088e58539b2dede93df57ac3f72dbe0480a028`

Seed/order: 1729. Same model, data construction, frozen non-quantized
parameters, CE/KL objective, quantizer, and held-out evaluator as v7/v8.

### Candidate screen

Each candidate receives the same first 300 direct-Q3 training chunks.

Candidates:

- constant 1e-4 (historical reference);
- constant 3e-4;
- constant 5e-4;
- constant 1e-3;
- constant 3e-3;
- 100-step warmup then cosine, peak 3e-4, floor 3e-5;
- 100-step warmup then cosine, peak 1e-3, floor 1e-4.

Selection uses **only** the existing 24-chunk validation split. The held-out
8192-token test split is not used for hyperparameter selection.

For cosine candidates, the schedule horizon is always 1200 updates, so the
first 300 LR values during screening exactly match the first 300 values in a
full run.

### Full-run rule

After the 300-step validation screen, run exactly three 1200-step direct-Q3
conditions from the same BF16-rounded source:

1. historical constant 1e-4 reference;
2. lowest-validation-loss non-reference candidate;
3. second-lowest-validation-loss non-reference candidate.

Only after those three candidates are fixed may the held-out test evaluator be
used.

### Preregistered interpretation

Historical seed-1729 Q9 -> Q3 reference from v7:

- loss 5.193768;
- PPL 180.146;
- top-1 33.789%;
- KL 2.255982.

This historical Q9 result is **not** used for v9 candidate selection.

Interpretation:

- if a tuned direct-Q3 schedule closes most or all of the ~0.68-nat v7 gap,
  narrow the project claim to an optimization/schedule advantage under the
  original direct baseline;
- if the best tuned direct-Q3 run remains substantially behind Q9, the
  trainability effect survives a materially stronger direct baseline;
- if the winning direct schedule materially improves over constant 1e-4, it
  must be replicated on orders 271828 and 424242 before using it as the new
  canonical direct baseline;
- do not launch the hybrid-master mechanism intervention until this baseline
  check is resolved.


## v9 — direct Q3 tuning

Pinned script:
`ternary_pet/smollm2_v9_direct_q3_tuning.py`

Pinned code:
`5c088e58539b2dede93df57ac3f72dbe0480a028`

Hugging Face job: `6ac5ad77fbc85ba6823bb71c` (A10G-small, 55-minute cap).

Seed/order 1729. Same model, data construction, objective, quantizer, frozen
non-quantized parameters, and held-out evaluator as v7/v8.

The first phase compares seven direct-Q3 schedules for 300 updates on the same
training chunks:

- constant 1e-4;
- constant 3e-4;
- constant 5e-4;
- constant 1e-3;
- constant 3e-3;
- warmup 100 then cosine, peak 3e-4, floor 3e-5;
- warmup 100 then cosine, peak 1e-3, floor 1e-4.

Ranking uses only the existing 24-chunk validation split. The held-out test set
is not used to choose candidates.

For cosine schedules, the horizon is always 1200 updates, so screening uses the
same first 300 LR values as a later full run.

After screening, run exactly three 1200-step direct-Q3 conditions from the same
source:

1. constant 1e-4 reference;
2. best non-reference validation candidate;
3. second-best non-reference validation candidate.

Only then evaluate those three on the held-out 8192-token test set.

The historical seed-1729 Q9 result from v7 is kept only for comparison after
selection: loss 5.193768, PPL 180.146, top-1 33.789%, KL 2.255982.

Interpretation:

- if a tuned direct schedule closes most of the prior Q9 gap, narrow the claim
  to an advantage over the original direct recipe;
- if tuned direct remains clearly behind, the Q9 trainability result survives a
  stronger direct baseline;
- if a new direct schedule materially improves on 1e-4, repeat it on the other
  two v7 orders before making it canonical;
- do not start the hybrid-master intervention until v9 is resolved.


### v9 outcome

Job `6ac5ad77fbc85ba6823bb71c` completed successfully.

Validation-only 300-step ranking selected:

1. constant 1e-3;
2. warmup100 + cosine, peak 1e-3, floor 1e-4.

The historical constant 1e-4 reference was also run as preregistered.

Full 1200-step held-out results:

| Schedule | Loss | PPL | Top-1 | KL |
|---|---:|---:|---:|---:|
| direct constant 1e-4 | 5.8747 | 355.90 | 25.55% | 2.9536 |
| direct constant 1e-3 | 5.8356 | 342.26 | 26.68% | 2.8942 |
| direct warm100 cosine 1e-3 -> 1e-4 | **5.5957** | **269.27** | **29.87%** | **2.6640** |
| historical Q9 -> Q3 | **5.1938** | **180.15** | **33.79%** | **2.2560** |

The tuned warmup+cosine direct baseline improves by 0.2789 nats/token versus
constant 1e-4, closing about **41.0%** of the historical Q9-vs-direct loss gap.

Q9 still remains ahead of tuned direct by:

- **0.4020 nats/token** loss;
- **33.1% lower PPL**;
- **+3.92 pp** top-1;
- **0.4080 lower KL**.

Interpretation: the original direct baseline was materially undertuned, so the
small-budget Q9 advantage should be revised downward in effect size. However,
optimizer schedule tuning does not erase the Q9 result on seed/order 1729.

Per preregistration, replicate the tuned direct schedule on orders 271828 and
424242 before replacing constant 1e-4 as the canonical direct baseline or using
the residual 0.402-nat gap as a replicated effect size.

Canonical files:

- `results/run_v9_2026-10-07.json`
- `results/run_v9_summary.md`


## v9 tuned-direct replication pair — orders 271828 and 424242

Purpose: test whether the v9-selected stronger direct-Q3 schedule reduces the
Q9 advantage consistently across the two remaining v7 training orders.

Shared pinned commit containing both scripts:
`31c12d56c239d091e06731c49c809ade8739af95`

Scripts:

- `replications/smollm2_v9_tuned_direct_seed271828.py`
- `replications/smollm2_v9_tuned_direct_seed424242.py`

Fixed schedule, chosen previously on seed/order 1729:

- direct Q3 for all 1200 updates;
- linear warmup for 100 steps to LR 1e-3;
- cosine decay to LR 1e-4 at step 1200.

Hugging Face jobs:

- seed 271828: `6ac5b807404719ba37664ae8`
- seed 424242: `6ac5b809fbc85ba6823bba12`

Hardware: A10G-small. Hard timeout: 35 minutes each.

There is **no new hyperparameter search** in these replications.

Everything else follows the established v7/v8/v9 setup, including the same
data construction, CE/KL objective, frozen non-quantized parameters, quantizer,
and held-out evaluator.

Primary paired comparisons use the already-recorded v7 Q9 -> Q3 results for the
same orders. Those Q9 results are fixed historical references and are not
recomputed or used to tune the direct schedule.

Interpretation:

- if Q9 remains ahead on both orders, the residual Q9 advantage survives a
  stronger direct-Q3 schedule across all three v7 orders;
- if tuned direct closes the gap on one or both orders, report that honestly and
  revise the aggregate effect size;
- only after this pair is complete should the tuned direct schedule replace
  constant 1e-4 as the canonical small-budget direct baseline.


### Technical retry note for tuned-direct replication pair

The first launch pair used scripts generated from an incorrect substring cut:
the generator matched `CANDIDATES=[` inside the earlier identifier
`LR_CANDIDATES=[`. This truncated the shared helper prefix and left a stray
`LR_` token in both generated files.

Consequences:

- seed 271828 job `6ac5b807404719ba37664ae8` failed before data/model
  loading with `NameError: name 'LR_' is not defined`;
- seed 424242 job `6ac5b809fbc85ba6823bba12` was cancelled before it could
  reach the same failure, to avoid wasting compute.

These are **technical failures only** and are retained in the audit trail. They
contain no training result and do not count as replications.

Both scripts were rebuilt from the exact v9 prefix using the newline-delimited
marker `\nCANDIDATES=[`, and static checks confirmed the presence of the
training helpers and absence of the stray token.

Corrected shared commit:
`3e74ccf5c448fc994d005a7baf94529b92e9996a`

Scientific protocol and fixed tuned-direct schedule are unchanged.

Corrected retry jobs:

- seed 271828: `6ac5b913404719ba37664b21`
- seed 424242: `6ac5b915fbc85ba6823bba5b`


### v9 tuned-direct replication outcome

Corrected confirmatory jobs completed successfully:

- seed 271828: `6ac5b913404719ba37664b21`
- seed 424242: `6ac5b915fbc85ba6823bba5b`

The fixed v9-selected direct schedule was used with no further tuning:

- 100-step warmup to LR 1e-3;
- cosine decay to LR 1e-4 by step 1200;
- direct Q3 throughout.

Held-out comparison against the already-locked Q9 -> Q3 results:

| Seed | Old direct loss | Tuned direct loss | Q9 loss | Residual Q9 gain | Old gap closed |
|---:|---:|---:|---:|---:|---:|
| 1729 | 5.8724 | 5.5957 | **5.1938** | **0.4020** | 40.77% |
| 271828 | 5.9802 | 5.6228 | **5.2417** | **0.3810** | 48.40% |
| 424242 | 5.9469 | 5.6067 | **5.3008** | **0.3059** | 52.65% |

Q9 remains ahead of tuned direct in **3/3 orders**.

Aggregate against tuned direct:

- mean residual loss advantage: **0.3630 nats/token**;
- mean paired PPL reduction: **30.38%**;
- mean top-1 gain: **+2.52 pp**;
- mean KL advantage: **0.3871**.

The tuned schedule closes **47.27%** of the historical Q9/direct loss gap on
average. Therefore direct under-tuning explained nearly half of the original
small-budget effect, but not all of it.

This tuned schedule now replaces constant 1e-4 as the canonical small-budget
direct-Q3 baseline.

Per the preregistered decision rule, the next experiment may proceed to the
four-arm Q3-vs-Q9 hybrid-master factorial intervention.

Canonical aggregate:
`replications/v9_tuned_direct_aggregate_summary.md`


## v10 — schedule-matched Q9 -> Q3 control

Purpose: close the remaining LR/schedule fairness loophole before the
hybrid-master mechanism intervention.

Pinned scripts commit:
`4050f42cf226a300082178b2fda475ebcf31664e`

Scripts:

- `replications/smollm2_schedule_matched_q9_seed1729.py`
- `replications/smollm2_schedule_matched_q9_seed271828.py`
- `replications/smollm2_schedule_matched_q9_seed424242.py`

Hugging Face jobs (A10G-small, 35-minute cap each):

- seed 1729: `6ac5c252fbc85ba6823bbd6c`
- seed 271828: `6ac5c254fbc85ba6823bbd6e`
- seed 424242: `6ac5c256404719ba37664cf1`

No Q9-specific hyperparameter search is allowed in v10.

### Fixed global LR schedule

Use exactly the schedule selected previously for direct Q3:

- global steps 1-100: linear warmup to LR 1e-3;
- global steps 101-1200: cosine decay to LR 1e-4.

Quantization schedule:

- global steps 1-300: Q9;
- at the transition, discard learned Q9 scales, restore original Q3 scales,
  and create fresh Adam exactly as in v7;
- global steps 301-1200: Q3;
- **do not restart the LR schedule** at the transition.

Thus the first Q3 continuation update receives the global step-301 LR, while
optimizer moments restart from zero.

Everything else remains the established v7/v8/v9 setup: same model, BF16
roundtrip source, training chunks/order, CE35/KL65 objective, frozen
non-quantized parameters, grad clip, fixed validation/test evaluators, and
teacher.

### Primary comparisons

For each seed/order, compare:

1. tuned direct Q3 from v9;
2. schedule-matched Q9 -> Q3 from v10;
3. historical constant-1e-4 Q9 -> Q3 from v7.

The v10 schedule was fixed before any v10 held-out result.

### Interpretation

- if schedule-matched Q9 remains better than tuned direct in 3/3 orders, the
  residual Q9 advantage survives an equal global LR schedule;
- if the gap shrinks or reverses, revise the canonical effect size accordingly;
- compare v10 against historical Q9 to measure schedule interaction, but do not
  call v10 "optimally tuned Q9": Q9 has not received an independent equal-size
  hyperparameter search;
- after this control is resolved, proceed to the four-arm hybrid-master causal
  intervention using the stronger schedule regime.


### v10 outcome

All three preregistered schedule-matched Q9 jobs completed successfully:

- seed 1729: `6ac5c252fbc85ba6823bbd6c`
- seed 271828: `6ac5c254fbc85ba6823bbd6e`
- seed 424242: `6ac5c256404719ba37664cf1`

Held-out final losses:

| Seed | Tuned direct Q3 | Historical Q9 -> Q3 | Matched-schedule Q9 -> Q3 | Matched-Q9 gain |
|---:|---:|---:|---:|---:|
| 1729 | 5.5957 | 5.1938 | **4.9010** | **0.6947** |
| 271828 | 5.6228 | 5.2417 | **4.9510** | **0.6718** |
| 424242 | 5.6067 | 5.3008 | **4.9563** | **0.6505** |

Aggregate versus tuned direct:

- mean loss advantage: **0.6723 nats/token**;
- mean paired PPL reduction: **48.94%**;
- mean top-1 gain: **+6.82 pp**;
- mean KL advantage: **0.6817**.

The shared schedule improves Q9 itself by **0.3094 nats/token on average**
relative to historical constant-1e-4 Q9 -> Q3.

Equal-compute step-300 fixed-Q3 diagnostics:

| Seed | Tuned direct @300 | Matched Q9 @300 | Q9 immediate disadvantage |
|---:|---:|---:|---:|
| 1729 | 5.9881 | 6.5423 | +0.5542 |
| 271828 | 6.1710 | 6.7168 | +0.5458 |
| 424242 | 6.0572 | 6.4933 | +0.4361 |

Mean immediate Q9 disadvantage: **+0.5120 nats/token**.

Thus the stronger equal-schedule regime preserves the core trainability result:
Q9 preparation is a worse immediate ternary checkpoint after equal compute but
a substantially better state for later Q3 optimization.

Preparation movement is larger under the stronger schedule. At step 300,
projected Q3 code displacement from the source averages:

- tuned direct: **4.41%**;
- matched Q9: **4.88%**.

The old constant-1e-4 v7 geometry (~0.68% moved codes and ~19.7% changed-set
Jaccard) should therefore not be assumed to characterize this tuned regime.

Interpretation: v10 closes the equal-global-schedule fairness loophole. It does
not establish a final best-tuned-vs-best-tuned effect because Q9 has not
received an independent equal-budget schedule search.

Canonical aggregate:
`replications/v10_schedule_matched_q9_aggregate_summary.md`

Next: run the four-arm hybrid-master factorial under this matched schedule,
rebuilding direct and Q9 step-300 masters in the same run and using fresh Adam,
original Q3 scales, and the global step-301 LR for all continuation arms.


## v11 — matched-schedule hybrid-master factorial

Purpose: causally partition the step-300 Q9-vs-direct master-state difference
under the stronger v10 schedule.

Pinned script:
`ternary_pet/smollm2_v11_hybrid_factorial.py`

Pinned code:
`f342fd9c706f2fe21aa00adabe6611b7f835f570`

Seed/order: 1729.

Hugging Face job: `6ac5c7befbc85ba6823bbef0` (A10G-small, 50-minute cap).

### Preparation

Use the exact v10 global LR schedule for the first 300 updates:

- warmup to 1e-3 over steps 1-100;
- continue the same cosine trajectory through step 300.

Build two states from the same BF16-rounded source and same first 300 chunks:

- D: direct Q3 preparation;
- S: Q9 preparation.

Discard both learned prep-scale states for the intervention. Project D and S
through the same original Q3 scales.

Define mask M as the exact weight positions where projected Q3(D) and Q3(S)
codes differ.

### Four causal arms

- 00: D everywhere;
- 10: S on M, D on the complement;
- 01: D on M, S on the complement;
- 11: S everywhere.

All four arms start the continuation with:

- original Q3 scales;
- fresh Adam;
- identical Q3 objective/data;
- the global LR curve continuing from step 301;
- 900 Q3 updates on the same remaining chunks.

Before continuation, the script must assert:

- Q3(00) == Q3(01) with zero Hamming distance;
- Q3(10) == Q3(11) with zero Hamming distance;
- validation diagnostics for each equal-forward pair agree within numerical
  tolerance.

### Primary interpretation

Let final held-out losses be L00, L10, L01, L11.

- `L00 - L10`: benefit from transferring Q9 masters on code-disagreement
  positions M onto the direct background;
- `L00 - L01`: benefit from transferring Q9 masters on positions whose
  projected Q3 codes already agree;
- `L10 - L11`: additional same-code contribution once M already comes from Q9;
- `L01 - L11`: additional M contribution once the complement already comes
  from Q9;
- interaction:
  `L00 - L10 - L01 + L11`.

Interpretation rules fixed before results:

- if 10 approaches 11 while 01 stays near 00, the advantage is concentrated
  mainly on code-disagreement positions;
- if 01 approaches 11 while 10 stays near 00, hidden same-code continuous
  geometry carries most of the advantage;
- if both hybrids recover meaningful but incomplete portions, both components
  contribute;
- if 10 and 01 each stay near 00 but 11 is much better, the effect is strongly
  interaction-dependent and cannot be localized additively;
- if endpoint 00-vs-11 trainability gap fails to reproduce materially under the
  common fresh-Adam/original-scale continuation, do not overinterpret hybrid
  localization.

This is a seed-1729 mechanism experiment first. Replicate only after inspecting
whether the intervention yields a stable, interpretable causal split.


### v11 outcome

Job `6ac5c7befbc85ba6823bbef0` completed successfully.

The exact equal-forward assertions passed:

- Q3(00) vs Q3(01): zero Hamming distance;
- Q3(10) vs Q3(11): zero Hamming distance;
- paired pre-continuation validation losses: exactly equal.

The D-vs-S projected-Q3 disagreement mask M contains **6.4581%** of quantized
weights (20,315,352 / 314,572,800).

Final held-out losses:

| Arm | Composition | Loss |
|---|---|---:|
| 00 | D everywhere | 5.6136 |
| 10 | S on M, D elsewhere | **4.9329** |
| 01 | D on M, S elsewhere | 5.5020 |
| 11 | S everywhere | **4.9010** |

The endpoint trainability gap reproduces: 00 - 11 = **0.7126 nats/token**.

Causal contributions:

- S on M onto D background: **0.6807 nats**, or **95.5%** of the full gain;
- S on same-code complement onto D background: **0.1116 nats**;
- once M is already from S, the remaining same-code contribution is only
  **0.0319 nats**;
- once the complement is already from S, transferring M still contributes
  **0.6010 nats**;
- factorial interaction: **+0.0797 nats**, indicating modest sub-additivity /
  redundancy under the positive-benefit sign convention.

Interpretation: on seed 1729, most of the trainability advantage is causally
localized to the positions where Q9 and direct choose different projected Q3
codes at step 300. The same-code majority contributes a smaller secondary
effect.

Do **not** translate this into "the code labels alone cause the effect." Arm 10
transfers the complete Q9 continuous master values on M. A later intervention
can separate discrete assignment from within-bin continuous position on M.

Interesting secondary observation: despite much larger absolute step-300 code
movement under the tuned schedule, the D-vs-S changed-set Jaccard versus initial
is **18.85%**, close to the old v7 ~19.7% value. Treat this as a single-seed
observation until replicated.

Canonical files:

- `results/run_v11_hybrid_factorial_seed1729_2026-10-07.json`
- `results/run_v11_hybrid_factorial_summary.md`

Next: exact v11 replications on orders 271828 and 424242 before making the
localization claim canonical across orders.


## v11 replication pair — hybrid-master causal localization

Purpose: replicate the seed-1729 v11 causal localization on the two remaining
established training orders before treating the mechanism as canonical.

Pinned shared commit containing both replication scripts:
`bb3ba53b4b55bfc6884d287d5a575785e212ffb4`

Scripts:

- `replications/smollm2_v11_hybrid_factorial_seed271828.py`
- `replications/smollm2_v11_hybrid_factorial_seed424242.py`

Hugging Face jobs (A10G-small, 50-minute cap each):

- seed 271828: `6ac638c0c656c912b4ffae8a`
- seed 424242: `6ac638c3f0d78b8017af0d5a`

No design or hyperparameter changes are allowed relative to v11 seed 1729.

Each run must independently rebuild:

- D = direct-Q3 masters after 300 matched-schedule updates;
- S = Q9 masters after 300 matched-schedule updates;
- M = positions where D and S project to different Q3 codes under the same
  original Q3 scales.

Each run must construct 00, 10, 01, 11 exactly as in v11 and assert before
continuation:

- Q3(00) == Q3(01), zero Hamming;
- Q3(10) == Q3(11), zero Hamming;
- paired validation diagnostics match within the existing numerical tolerance.

All four arms then receive original Q3 scales, fresh Adam, and the same global
LR continuation from step 301 through step 1200.

Primary replication quantities:

- full endpoint gain: L00 - L11;
- mask transfer gain: L00 - L10;
- same-code transfer gain: L00 - L01;
- residual same-code contribution: L10 - L11;
- mask fraction |M| / total;
- fraction of full gain recovered by mask transfer:
  (L00 - L10) / (L00 - L11).

Preregistered interpretation:

- if arm 10 again approaches arm 11 on both orders while arm 01 stays much
  closer to 00, the mechanism claim upgrades to a replicated conclusion that
  the dominant causal carrier is the Q9-prepared continuous master state on the
  code-disagreement subset;
- if one order reverses or shows a materially different decomposition, report
  the heterogeneity and do not canonicalize the 95.5% figure;
- do not claim discrete ternary code labels alone are causal. Arm 10 transfers
  the full continuous S masters on M.


### v11 replication-pair outcome

Both preregistered confirmatory jobs completed successfully:

- seed 271828: `6ac638c0c656c912b4ffae8a`
- seed 424242: `6ac638c3f0d78b8017af0d5a`

Both passed the exact pre-continuation pair checks:

- 00/01 projected-Q3 Hamming = 0;
- 10/11 projected-Q3 Hamming = 0;
- paired validation diagnostics exactly match.

Combined with seed 1729:

| Seed | Mask % | L00 | L10 | L01 | L11 | Full gain | Mask gain | Recovery |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1729 | 6.458 | 5.6136 | 4.9329 | 5.5020 | 4.9010 | 0.7126 | 0.6807 | 95.5% |
| 271828 | 6.181 | 5.6524 | 4.9711 | 5.5620 | 4.9510 | 0.7014 | 0.6813 | 97.1% |
| 424242 | 6.301 | 5.6234 | 4.9791 | 5.4979 | 4.9563 | 0.6672 | 0.6443 | 96.6% |

Aggregate:

- mean disagreement-mask size: **6.313%**;
- mean full 00->11 gain: **0.6938 nats/token**;
- mean mask-only gain: **0.6688 nats/token**;
- mean mask-only recovery: **96.4%**;
- recovery range: **95.5%–97.1%**;
- mean same-code-complement gain on a D background: **0.1092 nats**;
- mean residual same-code contribution once M is already from Q9:
  **0.0250 nats**;
- mean factorial interaction: **0.0842 nats**.

The mechanism localization therefore replicates in **3/3 orders**.

Canonical interpretation:

> Under this SmolLM2 matched-schedule setup, the dominant causal carrier of the
> Q9 trainability advantage is the continuous Q9-prepared master state on the
> ~6.3% of quantized positions where Q9 and direct preparation project to
> different ternary codes after 300 updates.

Do not claim code labels alone cause the effect. The intervention transfers full
continuous Q9 master values on M.

A secondary descriptive regularity also replicates: changed-set Jaccard versus
the original source is 18.85%, 19.35%, and 18.97% (mean **19.05%**) across the
three orders despite larger absolute code movement than the old v7 regime.

Canonical aggregate:
`replications/v11_hybrid_factorial_aggregate_summary.md`

Next causal refinement: separate **Q9-selected Q3 code identity** from
**continuous within-bin/boundary-relative position** on M.


## v12 — code identity vs continuous position on the replicated Q9 mask

Purpose: resolve the remaining ambiguity from v11. v11 replicated **where** the
useful prepared state resides (~6.3% D-vs-S projected-Q3 disagreement mask M),
but arm 10 transferred the full continuous Q9 master values. v12 asks **what
property on M is sufficient**.

Seed/order: 1729 first.

Hugging Face job: `6ac642a0df2184ac91ac018d` (A10G-small, 55-minute cap).

Pinned script:
`ternary_pet/smollm2_v12_code_identity_position.py`

Pinned code:
`22639f5b225e56be009886bf467a409910796eef`

### Common preparation and continuation

Rebuild D and S from the same BF16-rounded source under the established matched
global schedule:

- D = 300 direct-Q3 updates;
- S = 300 Q9 updates;
- M = positions where D and S, projected through the same original Q3 scales,
  choose different Q3 codes.

All continuation arms receive:

- original Q3 scales;
- fresh Adam;
- identical chunks 301-1200;
- the same global LR curve continuing from step 301;
- 900 Q3 updates.

### Arms

1. `D`: direct masters everywhere.
2. `exact`: exact S masters on true M, D elsewhere. Positive control intended
   to reproduce v11 arm 10.
3. `proto`: on true M use the Q3 reconstruction prototype for S's projected
   code, `w=(2/3)*alpha0*c_S`; D elsewhere.
4. `minimal`: on true M move only just inside S's Q3 code region; D elsewhere.
   Fixed normalized epsilon: **0.01** in `z=w/alpha0`.
   - target +1: z=+1/3+0.01
   - target -1: z=-1/3-0.01
   - target 0 from direct +1: z=+1/3-0.01
   - target 0 from direct -1: z=-1/3+0.01
5. `random`: layer/source/target-transition-matched random reassignment outside
   M using Q3 prototypes. For each layer and direct source code c_D, sample the
   same number of outside-M positions and reproduce the exact observed
   `c_D -> c_S` transition counts from true M.

### Required pre-continuation assertions

The three true-M Q9-code arms must start from the same ternary forward model:

- projected Q3(exact) == projected Q3(proto), zero Hamming;
- projected Q3(exact) == projected Q3(minimal), zero Hamming;
- their validation diagnostics must match within the existing numerical
  tolerance.

Random-control assertions:

- random changed-position count equals |M|;
- D-vs-random projected-Q3 Hamming equals |M|/total;
- instantiated random projected codes exactly match the constructed random plan;
- random positions are strictly outside true M;
- per-layer/source/target transition counts are inherited exactly from M by
  construction.

### Diagnostics

Log:

- true-mask transition counts `c_D -> c_S`;
- target-code counts for c_S=0 versus |c_S|=1;
- per-layer transition counts;
- D and S normalized nearest-Q3-boundary distances on M, split by target zero
  versus target nonzero.

### Primary interpretation

Let `G_exact = L_D - L_exact`.

Report:

- prototype recovery = `(L_D-L_proto)/G_exact`;
- minimal-crossing recovery = `(L_D-L_minimal)/G_exact`;
- matched-random recovery = `(L_D-L_random)/G_exact`.

Preregistered reading:

- proto ~= exact: Q9-selected Q3 code identity is sufficient for most of the
  mask benefit; exact Q9 continuous values are not required.
- minimal ~= exact: merely entering the Q9-selected Q3 region is sufficient.
- proto strong but minimal weak: depth/location inside the target Q3 region
  matters, though exact Q9 values may not.
- proto and minimal both weak while exact is strong: precise continuous
  Q9-prepared geometry on M is essential.
- random near D or harmful while true-M proto/minimal help: Q9's **specific
  position selection** matters, not merely making the same number and types of
  code reassignments.
- random also helps materially: some benefit may come from disrupting the direct
  solution with matched code changes, requiring follow-up before attributing all
  benefit to Q9-specific position selection.

Do not infer target-zero versus target-nonzero causal contributions from
descriptive subgroup statistics alone. If v12 is partial/ambiguous, run a
submask intervention next.

Replication rule: run seed 1729 first. Replicate the exact v12 design on orders
271828 and 424242 only if the result is interpretable enough to justify a
canonical code-vs-position claim.


### v12 outcome

Job `6ac642a0df2184ac91ac018d` completed successfully.

All preregistered construction checks passed.

True-mask equal-forward checks:

- exact vs prototype projected-Q3 Hamming = 0;
- exact vs minimal projected-Q3 Hamming = 0;
- exact/prototype/minimal validation diagnostics exactly match before
  continuation.

Matched-random checks:

- changed-position count exactly equals true |M|;
- D-vs-random Hamming exactly equals the true mask fraction;
- random instantiated codes exactly match the constructed plan;
- random positions are outside true M;
- per-layer/source/target transition counts are matched by construction.

Held-out final results:

| Arm | Loss | PPL | Gain vs D | Recovery vs exact |
|---|---:|---:|---:|---:|
| D | 5.6136 | 274.13 | — | — |
| exact | 4.9329 | 138.78 | 0.6807 | 100% |
| prototype | **4.8884** | **132.74** | **0.7252** | **106.5%** |
| minimal | 5.4979 | 244.18 | 0.1157 | 17.0% |
| matched random | 5.6793 | 292.75 | -0.0657 | -9.7% |

Interpretation:

1. Precise Q9 continuous master values on M are not required. The standardized
   Q3 reconstruction prototype performs 0.0445 nats better than exact Q9 while
   starting from the exact same ternary forward model.
2. Q9-selected code identity alone is not sufficient. Minimal crossing has the
   same projected Q3 codes but recovers only ~17% of the exact-S gain.
3. Boundary-relative depth / position inside the selected Q3 region therefore
   matters materially.
4. Q9's specific position selection matters. Reproducing the same per-layer
   source->target transition counts at different positions is worse than direct,
   ruling against a generic "any matched 6.46% code disruption helps" story on
   this seed.

Mask composition is almost perfectly adjacent-code:

- target 0: 49.78%;
- target +/-1: 50.22%;
- one-boundary adjacent transitions: 99.95%;
- direct -1 <-> +1 flips: 0.05%.

This is seed 1729 first. It is clear enough to replicate exactly on orders
271828 and 424242.

Canonical files:

- `results/run_v12_code_identity_position_seed1729_2026-10-07.json`
- `results/run_v12_code_identity_position_summary.md`


## v12 replication pair — code identity vs continuous position

Purpose: replicate the seed-1729 v12 mechanism result on the two remaining
training orders before treating the finer code-vs-position conclusion as
canonical.

Pinned shared commit:
`06c2df406d3e2029f742d64ad1065408b9a6209b`

Scripts:

- `replications/smollm2_v12_code_identity_position_seed271828.py`
- `replications/smollm2_v12_code_identity_position_seed424242.py`

Hugging Face jobs (A10G-small, 55-minute cap each):

- seed 271828: `6ac64ccddf2184ac91ac092d`
- seed 424242: `6ac64cd2df2184ac91ac092f`

No design, epsilon, schedule, random-control, or continuation changes are
allowed relative to seed 1729.

Each run independently rebuilds D, S, the true disagreement mask M, and the
layer/source/target-transition-matched random plan.

Required construction assertions are unchanged:

- exact/prototype projected-Q3 Hamming = 0;
- exact/minimal projected-Q3 Hamming = 0;
- exact/prototype/minimal pre-continuation diagnostics match;
- matched-random changed count = |M|;
- matched-random D-vs-random Hamming = |M|/total;
- random positions are outside true M;
- instantiated random codes match the constructed plan.

Primary replication quantities per seed:

- exact gain vs D;
- prototype recovery of exact gain;
- minimal-crossing recovery of exact gain;
- matched-random recovery of exact gain.

Preregistered interpretation:

- if prototype again approaches or beats exact in both orders, exact Q9
  within-region FP32 values are not necessary;
- if minimal crossing remains much weaker in both orders, merely choosing the
  Q9 code and barely crossing its boundary is insufficient;
- if matched-random remains non-beneficial in both orders, the Q9-selected
  positions themselves matter rather than generic matched code disruption;
- if any of these patterns fails materially on one or both orders, report the
  heterogeneity and do not canonicalize the seed-1729 percentages.

After the pair completes, aggregate all three v12 orders and decide whether the
mechanism sequence can stop without a target-zero vs target-nonzero submask
follow-up.


### v12 replication-pair outcome

Both confirmatory jobs completed successfully:

- seed 271828: `6ac64ccddf2184ac91ac092d`
- seed 424242: `6ac64cd2df2184ac91ac092f`

All construction/equality assertions passed in both runs.

Three-order final losses:

| Seed | D | Exact | Prototype | Minimal | Random |
|---:|---:|---:|---:|---:|---:|
| 1729 | 5.6136 | 4.9329 | **4.8884** | 5.4979 | 5.6793 |
| 271828 | 5.6524 | 4.9711 | **4.9187** | 5.5406 | 5.7208 |
| 424242 | 5.6234 | 4.9791 | **4.9373** | 5.4911 | 5.7031 |

Aggregate recovery relative to exact-Q9 positive control:

- prototype: **106.9%**;
- minimal crossing: **18.0%**;
- matched random: **-10.7%**.

Replicated conclusions:

1. Exact Q9 within-region continuous coordinates are not required: the Q3
   reconstruction prototype beats exact Q9 in **3/3 orders**.
2. Q9-selected code identity alone is insufficient: minimal boundary crossing
   remains weak in **3/3 orders** despite the same ternary forward model.
3. Q9's specific position selection matters: the matched-random reassignment is
   worse than direct in **3/3 orders**.

The current mechanism sequence is now complete enough to stop. Any further work
should be framed as a new generalization or application phase, not as necessary
cleanup for the mechanism claim.

Canonical aggregate:
`replications/v12_code_identity_position_aggregate_summary.md`.


## v13 — optional closing depth sweep and direct-firmness controls

Status: deliberately reopened as one optional closing mechanism experiment after
the v12 stop condition. v12 remains sufficient for the canonical mechanism
claim; v13 asks a finer dynamical/practical question.

Seed/order: **1729 first**.

Hugging Face job: `6ac65927e7a0dae8a277c24c` (A10G-small, 65-minute cap).

Pinned script:
`ternary_pet/smollm2_v13_depth_sweep_firmness.py`

Pinned code:
`fc68603caa1bf0e33faadb28f97a62a1bd0e3957`

### Unchanged setup

Reuse v12 exactly for:

- matched global warmup+cosine schedule;
- BF16-rounded source / FP32 masters;
- 300-step preparation of D (direct Q3) and S (Q9);
- original Q3 reference scales alpha0;
- disagreement mask M where projected Q3(D) != projected Q3(S);
- fresh Adam at continuation;
- global LR continuation from steps 301-1200;
- 900 Q3 continuation updates;
- train/eval data and order.

Actual Q3 geometry is confirmed from the implementation:

- normalized coordinate `u=w/alpha0`;
- Q3 decision boundaries at +/-1/3;
- nonzero reconstruction prototypes at +/-2/3;
- zero prototype at 0.

No literal d=0 arm is used because exact half-integer rounding is tie-sensitive.

### Experiment A — depth sweep

Depths:

`d in {0.03, 0.25, 0.50, 0.75, 1.00}`

For target code c_S=+/-1:

`u = c_S * (1/3 + d*(1/3))`

For target code c_S=0:

`u = c_D * (1/3) * (1-d)`

where c_D is +/-1 on a disagreement position targeting zero.

Thus:

- d=0.03 exactly re-anchors the v12 epsilon=0.01 minimal crossing;
- d=1.00 exactly re-anchors the v12 Q3 prototype.

All depth arms use D outside M.

Required validity:

- every depth arm has zero projected-Q3 Hamming to d=1.00;
- every depth arm exactly equals the projected S Q3 code model;
- all depth-arm pre-continuation diagnostics match.

Primary outcome: held-out final loss as a function of depth.

### Code-survival diagnostic

For every depth arm, after continuation updates 100, 300, and 900, measure the
fraction of M still holding Q9's selected Q3 code.

Measure survival two ways:

1. under the arm's **current learned Q3 scale**;
2. under the **fixed original alpha0** that defines the depth coordinate.

Report each for:

- all M;
- target c_S=0;
- target |c_S|=1.

Interpretation:

- if deeper placement improves both code survival and final loss, flip-back /
  assignment retention is a plausible dynamical explanation;
- if survival is similar across depths while final loss differs, then retention
  alone is insufficient and hidden continuous geometry affects later
  optimization even before/without different code survival;
- subgroup survival is descriptive only. Do not infer target-zero/nonzero causal
  gain without a separate submask intervention.

### Experiment B — firmness without Q9's choices

B1:
- on M, keep direct's own projected Q3 code c_D;
- snap the master to its Q3 reconstruction prototype;
- D elsewhere.

B2:
- define D_changed as positions where D's step-300 projected Q3 code differs
  from the initial projected Q3 code;
- on D_changed, snap D's own code to its Q3 reconstruction prototype;
- D elsewhere.

Required validity:

- Q3(B1) == Q3(D), zero Hamming and matching pre-continuation diagnostics;
- Q3(B2) == Q3(D), zero Hamming and matching pre-continuation diagnostics.

Log:

- size of D_changed;
- overlap/Jaccard between D_changed and M;
- initial->D transition counts on D_changed.

Interpretation:

- B1/B2 little or no gain: firmness is useful primarily when attached to Q9's
  selected decisions;
- B1/B2 large gain: some of the v12 effect is a more general "commit direct
  decisions firmly" optimization trick.

### Run/replication rule

This first job has exactly eight continuation arms:

- D
- d003
- d025
- d050
- d075
- d100
- B1
- B2

No d=1.5 arm in the first pass.

Replicate on orders 271828 and 424242 **only if seed 1729 gives a clear,
scientifically interpretable pattern**. A messy depth curve is a stopping result,
not a reason to keep adding arms.


### v13 first launch — technical timeout, no experimental result

HF job `6ac65927e7a0dae8a277c24c` stopped because the 65-minute wall-clock limit was reached.

Before timeout, all preregistered construction assertions passed:

- every depth arm had zero projected-Q3 Hamming to d=1.00;
- every depth arm matched the projected S Q3 model;
- all depth-arm pre-continuation diagnostics matched;
- B1 and B2 had zero projected-Q3 Hamming to D and matching diagnostics.

The run did not finish the continuation arms and emitted no `FINAL_JSON`.
Treat it as a technical timeout only, with no experimental outcome.

Relaunch the exact same pinned code and design, changing only the wall-clock timeout.


### v13 timeout-safe split relaunch

The monolithic v13 design is partitioned into two jobs with no scientific
changes other than separating continuation arms.

Pinned shared commit containing both split scripts:
`9332a0a0b4063f7ed6786aa29fd049429cd50380`

Jobs:

- v13A depth sweep: `6ac6a265df2184ac91ac410d`
  - arms: D, d003, d025, d050, d075, d100
  - A10G-small
  - 120-minute wall-clock cap
- v13B firmness controls: `6ac6a272df2184ac91ac412c`
  - arms: D, B1, B2
  - A10G-small
  - 120-minute wall-clock cap

Both jobs independently rebuild the same seed-1729 D and S step-300 states
under the original v13 setup. Shared D is intentionally rerun in both jobs as
an internal consistency check.

The original timed-out job remains a technical failure only and contributes no
experimental outcome.


### v13 seed-1729 outcome — depth sweep + firmness controls

Timeout-safe split jobs completed successfully:

- depth sweep: `6ac6a265df2184ac91ac410d`
- firmness controls: `6ac6a272df2184ac91ac412c`

Pinned split commit:
`9332a0a0b4063f7ed6786aa29fd049429cd50380`

All preregistered equal-forward assertions passed.

Depth-arm final losses:

| Depth | Loss | Recovery vs d=1 prototype |
|---:|---:|---:|
| 0.03 | 5.4948 | 16.4% |
| 0.25 | 4.9262 | 94.8% |
| 0.50 | **4.8850** | 100.5% |
| 0.75 | 4.8876 | 100.1% |
| 1.00 | 4.8884 | 100.0% |

Direct baseline: **5.6136**.

The shape is threshold-like / saturating rather than monotonically "deeper is
always better": nearly all of the prototype benefit is present by d=0.25, and
performance is effectively flat from roughly d=0.5 through d=1.0.

Q9-code survival on the true disagreement mask at continuation step 900 under
the current learned Q3 scale:

- d=0.03: **51.76%**
- d=0.25: **86.48%**
- d=0.50: **97.32%**
- d=0.75: **99.13%**
- d=1.00: **99.45%**

Survival measured against fixed original alpha0 is nearly identical, so the
pattern is not explained by learned-scale drift.

Firmness-only controls:

- B1, direct codes prototyped on M: loss **5.6403**, gain vs D **-0.0267**
- B2, direct codes prototyped on D's own changed set: loss **5.6195**, gain vs D
  **-0.0059**

Thus generic "snap direct decisions firmly to prototypes" does not help on this
seed.

Current interpretation:

> Q9's selected position/code decisions require a moderate interior commitment,
> not merely threshold crossing. Moving to around d=0.25 preserves most of the
> later benefit, and by d=0.5 the effect saturates. Code survival strongly tracks
> this transition. Firmness is not generically beneficial for direct Q3
> assignments; it is useful when attached to the specific assignments selected
> by Q9.

Do not state that survival is proven to be the sole causal mediator. It is a
strongly aligned dynamical diagnostic.

D's own changed set is **4.569%** of quantized weights. Its Jaccard overlap with
the Q9 disagreement mask is **38.64%**.

Canonical files:

- `results/run_v13_seed1729_compact_2026-10-07.json`
- `results/run_v13_depth_firmness_summary.md`

The preregistered "clear pattern" condition is met, so replication on orders
271828 and 424242 is scientifically justified but remains optional because v12
already established the three-order canonical mechanism.


## G1 — cross-family small-model generalization roadmap

**Status: preregistered plan; no G1 scientific result yet.**

The SmolLM2-360M mechanism phase stopped canonically at v12, with v13 retained
as an optional one-order refinement. The next phase is therefore named **G1**
rather than v14.

Current Family-B small candidate:
`ibm-granite/granite-4.0-350m`.

The detailed gated protocol, stopping rules, architecture audit, direct-only
schedule calibration, D/S staging gate, compressed mechanism intervention,
replication rules, and later small/large x family-A/family-B matrix are frozen
in:

`G1_GENERALIZATION_PLAN.md`

Planned sequence:

1. **G1-0:** architecture / quantization smoke test;
2. **G1-1:** direct-Q3 validation-only schedule calibration;
3. **G1-2:** seed-1729 D-vs-S staging gate with equal-compute step-300
   projection diagnostics;
4. **G1-3:** only if staging is positive, compressed
   D/S/M-exact/M-d50/matched-random-d50 mechanism test;
5. **G1-4/G1-5:** only if the first-order result is interpretable, exact
   confirmatory orders 271828 and 424242 with no retuning.

Preregistered interpretation rule: a successful staging result does **not**
require Q9 to be worse at the immediate step-300 Q3 projection. If Granite wins
immediately and finally, report staging generalization with a different
mechanism signature. If the final S-vs-D effect is absent, do not rescue the
result by retuning after seeing Q9.

Do not require Granite to reproduce Smol's exact mask size or effect
percentages. The cross-family target is the within-model causal pattern.

No G1 compute is launched by this roadmap entry.


### G1-0 — Granite 4.0 350M architecture / quantization smoke

**Status before launch: preregistered engineering gate; no scientific result.**

Model:
`ibm-granite/granite-4.0-350m`

Pinned script:
`ternary_pet/g1_granite350m_smoke.py`

Pinned commit:
`1c85fdb1bdccd70c5c925ea2807bf773381739fd`

This job is allowed to answer only whether the established Smol intervention
ports cleanly enough to proceed. It records:

- loaded architecture/class and total parameter count;
- every `nn.Linear` target selected by the canonical "all linears except
  exact `lm_head`" rule;
- target parameter coverage and the excluded parameter tensors;
- embedding/output-head weight tying;
- source held-out WikiText-2 CE/PPL on a small smoke slice;
- initial Q3 and Q9 code histograms / Q3 zero fraction;
- one CE35 + teacher-KL65 Q3 update;
- one CE35 + teacher-KL65 Q9 update;
- confirmation that non-target parameters remain frozen;
- peak GPU memory for both smoke arms.

No D-vs-S comparison is made, and no scientific generalization claim may be
drawn from this job.

**Stop rule:** if the target coverage or model structure shows that "quantize
all `nn.Linear` weights except `lm_head`" is not a meaningfully comparable
intervention, stop before G1-1 and preregister an architecture-specific mapping.


### G1-0 launch record

Hugging Face job:
`6ac6eacfdf2184ac91ac658a`

Hardware:
A10G-small

Wall-clock cap:
30 minutes

Pinned code:
`1c85fdb1bdccd70c5c925ea2807bf773381739fd`

This is the architecture/quantization smoke gate only. G1-1 must not launch
until this job is inspected and the target mapping is judged comparable enough
to proceed.


### G1-0 outcome — architecture gate exposed a precision failure

Job `6ac6eacfdf2184ac91ac658a` completed.

Architecture/freezing checks were encouraging:

- 352,379,904 total parameters (derived from target + excluded);
- 249,561,088 targeted weight parameters across 168 target linears;
- 102,818,816 excluded parameters;
- no unexpected trainable tensors;
- non-quantized parameters remained frozen;
- initial Q3 zero fraction 31.99%;
- initial Q9 zero fraction 12.37%;
- peak CUDA memory 6.30 GiB for Q3 and 8.17 GiB for Q9.

However, the inherited Smol FP16 compute path is numerically invalid on Granite:

- unquantized source held-out loss/PPL were NaN;
- Q3 CE was finite (21.55) but teacher KL, total loss, and grad norm were NaN;
- Q9 CE was finite (26.62) but teacher KL, total loss, and grad norm were NaN.

This is a **technical gate failure, not a scientific negative result**. Because
the unquantized source already fails under FP16 autocast, do not attribute the
NaNs to Q3 or Q9.

Canonical summary:
`results/run_g1_0_granite350m_smoke_summary.md`

G1-1 remains blocked.

### G1-0b — preregistered Granite precision diagnostic

Purpose: determine whether the G1-0 NaNs are specifically caused by the inherited
FP16 compute path and whether BF16 is a numerically valid Granite-specific
replacement before any scientific D-vs-S result is observed.

Pinned script:
`ternary_pet/g1_granite350m_precision_smoke.py`

Pinned commit:
`3b217e11853ac062ef187ee163291ac6138163c5`

The diagnostic compares the same BF16-rounded Granite source under:

1. full FP32 compute;
2. BF16 autocast;
3. FP16 autocast.

It also compares directly loaded BF16 and FP16 teachers, then runs exactly one
Q3 and one Q9 CE35 + teacher-KL65 update using a BF16 teacher and BF16 autocast.

Required pass condition before G1-1:

- source FP32 finite;
- source BF16 finite;
- BF16 teacher finite;
- Q3 BF16 step has finite CE/KL/loss and gradients;
- Q9 BF16 step has finite CE/KL/loss and gradients.

FP16 is allowed to fail. If BF16 passes and tracks FP32 reasonably, BF16 mixed
precision becomes the preregistered Granite compute setting for G1 before any
scientific staging result is seen.

No scientific conclusion may be drawn from G1-0b.


### G1-0b launch record

Hugging Face job:
`6ac6eec9df2184ac91ac67ca`

Hardware:
A10G-small

Wall-clock cap:
30 minutes

Pinned code:
`3b217e11853ac062ef187ee163291ac6138163c5`

G1-1 remains blocked until this diagnostic satisfies the preregistered BF16
finite-path checks.


### G1-0b outcome — BF16 precision path PASSED

Job `6ac6eec9df2184ac91ac67ca` completed and satisfied every preregistered
finite-path condition.

Precision probe on the same Granite source slice:

| Path | CE | Finite logits? |
|---|---:|---|
| FP32 | 3.23537 | yes |
| BF16 autocast | 3.24056 | yes |
| FP16 autocast | NaN | no |
| BF16 teacher | 3.23373 | yes |
| FP16 teacher | NaN | no |

One-step BF16 quantized checks:

- Q3: CE 21.5361, KL 18.8207, total loss 19.7711, grad norm 898.63;
  all trainable gradients finite.
- Q9: CE 26.6276, KL 25.1715, total loss 25.6811, grad norm 2995.33;
  all trainable gradients finite.

The BF16 source CE differs from FP32 by only 0.00519 nats/token on this smoke
slice. The inherited FP16 path fails completely.

**Decision:** for all Granite G1 jobs, freeze the compute convention before any
scientific D-vs-S result:

- persistent master weights remain FP32;
- common source is BF16-rounded as in the established protocol;
- student/teacher forward compute uses BF16 autocast;
- teacher weights use BF16 on GPU;
- no FP16 GradScaler;
- standard backward with the canonical gradient clipping;
- `use_cache=False`.

The earlier G1-0 NaNs are classified as a technical FP16 incompatibility, not a
quantization failure.

Canonical files:

- `results/run_g1_0b_granite350m_precision_2026-10-08.json`
- `results/run_g1_0b_granite350m_precision_summary.md`

G1-1 is now unblocked.

### G1-1 — Granite-350M direct-Q3 schedule calibration

**Status before launch: preregistered direct-only validation screen.**

Seed/order:
1729.

Pinned script:
`ternary_pet/g1_granite350m_direct_q3_calibration.py`

Pinned commit:
`3cb0153be80d5e9fbe112460bde7bc26398d851f`

Purpose:
select one direct-Q3 global LR schedule before any Granite Q9 result exists.

All candidates:
- start from the identical BF16-rounded Granite source;
- use the same first 300 shuffled training chunks;
- use Q3 forward weights only;
- use FP32 persistent masters;
- use the G1-0b-approved BF16 compute path;
- are evaluated only on the 24 LR-validation chunks;
- never touch the held-out test evaluator.

Candidates:

1. `const_1e-4`
2. `warm100_cosine_3e-4`: 100-step warmup to 3e-4, cosine to 1e-4 over
   the fixed 1200-step horizon
3. `warm100_cosine_1e-3`: 100-step warmup to 1e-3, cosine to 1e-4 over
   the fixed 1200-step horizon

Each candidate runs exactly 300 updates. The warmup/cosine LR values are defined
against the eventual 1200-step horizon, so the first 300 values can be reused
unchanged in G1-2.

**Selection rule:** lowest validation loss after 300 updates wins. Ties are
resolved by lower validation KL, then lower peak LR.

The winner becomes frozen for G1-2. Do not retune after any Q9 result is seen.

This is a calibration job, not a cross-family scientific result.


### G1-1 launch record

Hugging Face job:
`6ac6f18ae7a0dae8a2780246`

Hardware:
A10G-small

Wall-clock cap:
90 minutes

Pinned code:
`3cb0153be80d5e9fbe112460bde7bc26398d851f`

This job contains only the preregistered direct-Q3 validation screen. No Q9 arm
and no held-out test evaluation are present.


### G1-1 outcome — direct schedule frozen

Job `6ac6f18ae7a0dae8a2780246` completed successfully.

Validation-only 300-step direct-Q3 ranking:

| Candidate | Validation loss | PPL | Q3 code movement vs source |
|---|---:|---:|---:|
| **constant 1e-4** | **5.81281** | **334.56** | 5.323% |
| warm100 -> 3e-4, cosine -> 1e-4 | 6.08270 | 438.21 | 11.616% |
| warm100 -> 1e-3, cosine -> 1e-4 | 6.74016 | 845.70 | 27.231% |

All candidates were numerically finite under BF16. Per the preregistered
selection rule, **constant LR 1e-4 wins and is frozen for G1-2**.

No Q9 arm and no held-out test metric informed this choice.

Canonical files:

- `results/run_g1_1_granite350m_calibration_2026-10-08.json`
- `results/run_g1_1_granite350m_calibration_summary.md`

### G1-2 — Granite-350M one-order D/S staging gate

**Status before launch: preregistered scientific comparison.**

Seed/order:
1729.

Pinned script:
`ternary_pet/g1_granite350m_staging_gate.py`

Pinned commit:
`f3a1f88890d1b70264f25f1be8c4d23d84594825`

Frozen compute/training semantics:

- model: `ibm-granite/granite-4.0-350m`;
- common BF16-rounded source;
- persistent FP32 master weights;
- BF16 autocast and BF16 teacher from the passed G1-0b gate;
- all `nn.Linear` weights except exact `lm_head` quantized;
- non-quantized parameters frozen;
- learnable rowwise scales;
- CE35 + teacher-KL65;
- AdamW beta=(0.9,0.95), no weight decay;
- gradient clipping 1.0;
- same 1200 shuffled chunks and order;
- **constant LR 1e-4**, selected by G1-1 direct-only validation.

Arms:

1. **D:** 1200 direct-Q3 updates with continuous Adam state.
2. **S:** 300 Q9 updates, then retain FP32 masters, restore the original Q3
   scales, create fresh Adam, and run 900 Q3 updates at the same constant LR.

Required equal-compute step-300 diagnostics:

- D native Q3 validation metrics;
- S native Q9 validation metrics;
- D masters projected through the original Q3 scales;
- S masters projected through the exact same original Q3 scales;
- projected D-vs-S Q3 Hamming/disagreement fraction;
- D and S projected-code movement versus source Q3;
- changed-set overlap/Jaccard;
- layer and source->target transition counts for the D-vs-S disagreement mask.

Required final held-out metrics:

- loss;
- PPL;
- teacher top-1 agreement;
- KL to teacher;
- fixed generation probes as descriptive output only.

### G1-2 interpretation gate

The strongest Smol-like mechanism signature would be:

1. S is no better, or is worse, than D as an immediate fixed-Q3 projection at
   step 300; and
2. S nevertheless finishes better after its 900-step Q3 continuation.

But immediate inferiority is **not required** for staging itself to generalize.
If S is better both immediately and finally, report a positive staging result
with a different mechanism signature.

**Proceed to G1-3 only if the final S endpoint has a material and technically
clean held-out advantage over D.** Do not retune the direct schedule or Q9
schedule after observing this result.

This is the first Granite G1 job allowed to support a cross-family scientific
claim.


### G1-2 launch record

Hugging Face job:
`6ac6f487df2184ac91ac693e`

Hardware:
A10G-small

Wall-clock cap:
75 minutes

Pinned code:
`f3a1f88890d1b70264f25f1be8c4d23d84594825`

This is the first Granite G1 job allowed to contribute a scientific
cross-family staging result. G1-3 remains blocked until the final D-vs-S
endpoint is inspected against the preregistered gate.


### G1-2 outcome — first Granite staging result is positive

Job `6ac6f487df2184ac91ac693e` completed successfully.

Seed/order 1729 final held-out endpoint:

| Metric | Direct Q3 | Q9→Q3 | Q9 advantage |
|---|---:|---:|---:|
| loss | 5.66581 | **5.53495** | **0.13086 nats/token** |
| PPL | 288.82 | **253.40** | **12.27% lower** |
| teacher top-1 | 27.53% | **28.04%** | **+0.51 pp** |
| KL to teacher | 2.49025 | **2.38307** | **0.10718 lower** |

Equal-compute step-300 fixed-Q3 projection:

- direct: loss **5.81745**, PPL **336.11**;
- Q9-prepared masters projected through the same original Q3 scales:
  loss **7.74247**, PPL **2304.15**.

Thus the Q9-prepared state is **1.92502 nats/token worse immediately in Q3**
yet finishes better after the common later Q3 budget. This reproduces the
Smol-like **trainability-not-entry-quality** signature on the first Granite
order.

Step-300 geometry:

- direct changed vs source Q3: 5.3242%;
- Q9-prepared changed vs source Q3: 5.2723%;
- D-vs-S projected Q3 disagreement: **6.0246%**
  (15,035,048 / 249,561,088 targeted positions);
- changed-set Jaccard: 27.53%.

This is a **one-order cross-family staging result only**. It does not yet
establish that Granite shares the same causal disagreement-mask / interior-
placement mechanism.

Canonical files:

- `results/run_g1_2_granite350m_staging_seed1729_2026-10-08.json`
- `results/run_g1_2_granite350m_staging_summary.md`

Per the preregistered G1 gate, G1-3 is now authorized.

### G1-3 — Granite compressed causal mechanism test

**Status before launch: preregistered mechanism test; seed/order 1729 only.**

Purpose:
test whether Granite's positive staging effect is carried by the same kind of
small D-vs-S projected-Q3 disagreement set seen on SmolLM2, and whether useful
interior placement at normalized depth d=0.5 is position-specific rather than a
generic code-snapping trick.

The run rebuilds D and S independently from the common BF16-rounded source
using the frozen Granite G1 semantics:

- direct-selected constant LR 1e-4;
- first 300 identical shuffled training chunks;
- FP32 persistent masters;
- BF16 autocast and BF16 teacher;
- original rowwise Q3 scales define all projected codes and constructions;
- all continuation arms receive **fresh Adam + original Q3 scales + the same
  constant 1e-4 + the same steps 301-1200**.

Define M as the positions where D and S masters, projected through the same
original Q3 scales at step 300, choose different Q3 codes.

Continuation arms:

1. **D** — direct-prepared masters everywhere.
2. **S** — Q9-prepared masters everywhere.
3. **M-exact** — Q9-prepared masters only on M; direct masters elsewhere.
4. **M-d50** — direct masters outside M; on M use S's selected Q3 code placed
   at normalized depth `d=0.5` inside that target region.
5. **Random-d50** — direct masters everywhere except an equal-size set outside
   M, matched per layer and D-source→S-target transition count, placed at the
   same `d=0.5`.

The d=0.5 coordinate is the already established v13 definition:

- target ±1: normalized u = sign(target) * (1/3 + 0.5 * 1/3) = ±1/2;
- target 0 from source ±1: normalized u = source_sign * (1/3) * (1-0.5)
  = ±1/6.

Required pre-continuation assertions:

- S, M-exact and M-d50 must have exactly the same projected Q3 codes globally;
- their pre-continuation validation diagnostics must match within numerical
  evaluation tolerance;
- Random-d50 must reproduce the planned random target codes exactly;
- Random-d50 must contain exactly |M| positions and must not overlap M;
- its per-layer/source→target transition counts must match M exactly.

Required outputs:

- final held-out loss/PPL/top-1/KL for all five arms;
- full S-vs-D gain;
- recovery of the full S gain by M-exact;
- recovery by M-d50 relative to M-exact and to full S;
- Random-d50 effect versus D;
- Q9-selected-code survival on true M for M-d50 after 100, 300 and 900 Q3
  continuation steps using both learned current Q3 scales and fixed original
  alpha0.

Interpretation remains qualitative/within-model. Do not require Granite to
match Smol's exact 96.4% localization or 106.9% prototype recovery. The
mechanism is supported if the true disagreement set carries most of the staging
benefit, d=0.5 retains substantial benefit, and the matched random intervention
is clearly weaker or harmful.

No confirmatory Granite orders are launched by this entry. G1-4/G1-5 remain
separate decisions after G1-3 is inspected.


### G1-3 pinned implementation

Script:
`ternary_pet/g1_granite350m_compressed_mechanism.py`

Pinned commit:
`c32fbc16f463039257d996a7b32c9eca5eada681`

The implementation follows the preregistered five-arm design exactly:

- D
- S
- M-exact
- M-d50
- Random-d50

Before any 900-step continuation begins, the script asserts:

- S == M-exact == M-d50 in projected Q3 codes;
- those equal-forward arms match on pre-continuation diagnostics;
- Random-d50 exactly realizes its deterministic planned target codes;
- Random-d50 contains exactly |M| positions;
- Random-d50 has zero overlap with true M by construction;
- Random-d50 matches true M's per-layer/source->target transition counts.

Any failed construction assertion terminates the job and counts as a technical
failure, not a mechanism result.


### G1-3 launch record

Hugging Face job:
`6ac6fe5fdf2184ac91ac6c1f`

Hardware:
A10G-small

Wall-clock cap:
75 minutes

Pinned code:
`c32fbc16f463039257d996a7b32c9eca5eada681`

This is the single authorized G1-3 mechanism job for seed/order 1729. No
confirmatory Granite orders and no larger-model jobs are launched by this
record.


### G1-3 outcome — Granite mechanism supported on first order

Job `6ac6fe5fdf2184ac91ac6c1f` completed successfully. All preregistered
construction assertions passed before continuation.

Final held-out losses:

| Arm | Loss | PPL |
|---|---:|---:|
| D | 5.66205 | 287.74 |
| S | 5.53495 | 253.40 |
| M-exact | 5.56525 | 261.19 |
| M-d50 | **5.51134** | **247.48** |
| Random-d50 | 5.73514 | 309.56 |

Effects relative to D:

- full S gain: **0.12710 nats/token**;
- M-exact gain: **0.09680**, recovering **76.2%** of full S;
- M-d50 gain: **0.15071**, recovering **118.6%** of full S and **155.7%**
  of the exact-M gain;
- matched Random-d50 gain: **-0.07308** (harmful), equal to **-57.5%**
  recovery of the full S effect.

The true disagreement mask is **6.0246%** =
15,035,048 / 249,561,088 targeted positions.

S, M-exact and M-d50 had exactly the same projected Q3 forward model before
continuation. Therefore their endpoint differences are caused by hidden FP32
master geometry rather than different starting ternary forward weights.

Interpretation:

> Granite reproduces the broader Smol mechanism qualitatively. Q9 identifies
> useful which-position/which-code commitments; moderate interior placement of
> those commitments is sufficient to recover and exceed the full staged gain;
> and applying the same transition pattern to matched random positions is
> harmful.

Important quantitative difference: exact-Q9-on-M recovers **76.2%** on Granite,
not SmolLM2's canonical ~96.4%. Do not claim the localization fraction is
family-invariant.

Q9-selected-code survival for M-d50:
- step 100: 99.677%;
- step 300: 97.977%;
- step 900: **90.696%** under learned scales and **90.693%** under fixed alpha0.

Scale drift is therefore not a meaningful explanation of survival. Target-zero
and target-nonzero survival are also similar at step 900.

Canonical files:

- `results/run_g1_3_granite350m_mechanism_seed1729_2026-10-08.json`
- `results/run_g1_3_granite350m_mechanism_summary.md`

G1-3 is positive and scientifically interpretable. The preregistered condition
for confirmatory Granite orders is satisfied, but **no G1-4/G1-5 jobs are
launched by this result**.


### G1-4 / G1-5 — Granite mechanism confirmation orders

**Status before launch: preregistered exact confirmations.**

Purpose:
test whether the positive G1-2/G1-3 staging + mechanism result on Granite order
1729 survives the two established independent shuffled training orders.

Confirmation orders:

- **G1-4:** seed/order 271828
- **G1-5:** seed/order 424242

Pinned scripts:

- `ternary_pet/replications/g1_granite350m_mechanism_seed271828.py`
- `ternary_pet/replications/g1_granite350m_mechanism_seed424242.py`

Shared pinned commit containing both scripts:
`6ed697a0ca5d0d19ed5ebff04b81a2ed0d259230`

The scripts are exact copies of the completed G1-3 implementation with **only
`SEED` / training-order identity changed**.

Frozen protocol:

- model: `ibm-granite/granite-4.0-350m`;
- FP32 persistent masters;
- BF16-rounded common source;
- BF16 autocast + BF16 teacher;
- constant LR 1e-4 selected by direct-only G1-1;
- first 300 chunks build D and S under Q3 and Q9 respectively;
- true M is the fixed-Q3 D-vs-S projected-code disagreement mask;
- five common-continuation arms:
  D, S, M-exact, M-d50, Random-d50;
- original Q3 scales + fresh Adam for all arms at continuation;
- identical steps 301-1200 within each order;
- same construction assertions as G1-3;
- same 8,192-token held-out evaluator.

No retuning, architecture change, precision change, arm deletion, or
result-dependent intervention is allowed.

### Confirmatory questions

Inspect **each order individually** before aggregation:

1. Does full S still beat D at the final held-out endpoint?
2. Is S still worse than D as an immediate fixed-Q3 checkpoint at step 300?
3. Does true-M exact transfer retain a material fraction of the full S gain?
4. Does true-M d=0.5 retain substantial benefit and preferably approach or beat
   full S?
5. Is transition/layer/source-matched Random-d50 clearly weaker than true-M
   d=0.5, preferably non-beneficial or harmful?

Do not require exact replication of Granite order-1729 effect size, 76.2%
exact-M recovery, 118.6% d=0.5 recovery, or 6.0246% mask size.

### Replication interpretation

A replicated Granite mechanism conclusion requires the qualitative pattern to
be supported across the three orders after inspecting each one separately.
Mixed orders must be reported plainly rather than averaged away.

No larger-model jobs are launched by this preregistration.


### G1-4 / G1-5 launch record

Shared pinned code:
`6ed697a0ca5d0d19ed5ebff04b81a2ed0d259230`

Hardware:
A10G-small for both jobs.

Wall-clock cap:
75 minutes per job.

Jobs:

- G1-4 seed/order 271828:
  `6ac70af0df2184ac91ac6ffc`
- G1-5 seed/order 424242:
  `6ac70af2df2184ac91ac7000`

These are exact frozen-protocol confirmations of G1-3 with only the seed/order
changed. No additional jobs are authorized by this launch record.


### G1-4 / G1-5 outcome — mixed staging replication

Both frozen-protocol confirmations completed successfully and all construction
assertions passed.

Jobs:
- seed/order 271828: `6ac70af0df2184ac91ac6ffc`
- seed/order 424242: `6ac70af2df2184ac91ac7000`

Final held-out losses:

| Seed | D | S | M-exact | M-d50 | Random-d50 |
|---:|---:|---:|---:|---:|---:|
| 1729 | 5.66205 | **5.53495** | 5.56525 | **5.51134** | 5.73514 |
| 271828 | **5.72673** | 5.84103 | 5.80453 | 5.78502 | 5.79268 |
| 424242 | 5.80192 | **5.74057** | 5.74471 | **5.71016** | 5.81984 |

Full-S staging gain vs D:
- 1729: **+0.12710**
- 271828: **−0.11431**
- 424242: **+0.06135**

Therefore full Q9→Q3 staging is positive on **2/3** Granite orders, not 3/3.
Mean paired gain is only **+0.02471 nats/token**. Do not claim replicated
Granite staging superiority.

Step-300 fixed-Q3 entry remains worse for Q9 on **3/3** orders:
- 1729: 7.74247 vs 5.81281
- 271828: 6.51453 vs 5.64945
- 424242: 7.57531 vs 5.68631

The D-vs-S disagreement masks remain small:
- 1729: 6.0246%
- 271828: 6.9946%
- 424242: 6.0296%
- mean: **6.3496%**

The structured d=0.5 intervention is more consistent than full S:
- M-d50 beats full S on **3/3** orders;
- M-d50 beats matched-random on **3/3** orders;
- matched-random is harmful versus D on **3/3** orders;
- but M-d50 itself beats D on only **2/3** orders.

Mean M-d50 gain vs D: **+0.06139 nats/token**.
Mean matched-random effect vs D: **−0.05232 nats/token**.

On seed 271828, the true-M d=0.5 arm is still worse than D by 0.05829 nats
and only 0.00766 nats better than matched-random. Therefore position specificity
is weakly separated on this negative order and must not be overstated.

Q9-selected-code survival at continuation step 900 is stable:
90.70%, 89.59%, 90.57% across the three orders (mean **90.28%**), with
learned-scale and fixed-alpha0 measurements nearly identical.

### Granite G1 conclusion after three orders

Safe conclusion:

> Granite shows a stable ~6% disagreement geometry and a reproducible
> worse-entry signature, but the final full-staging benefit is order-dependent
> (2/3 positive). The true-mask d=0.5 intervention is more robust than carrying
> the full Q9 state: it improves over full S on all three orders, while matched
> random placement is harmful on all three.

This is **mixed cross-family evidence**, not a clean replication of the Smol
3/3 staging result.

Canonical aggregate:
`replications/g1_granite350m_mechanism_aggregate_summary.md`.


### G1-6 — seed-conditioned trajectory-alignment selector (development order 271828)

**Status before launch: preregistered development experiment.**

This is one deliberate follow-up on the already-observed negative Granite order
271828. It is not a new confirmation seed and must not be presented as such.

#### Question

Does the usefulness of a Q9-selected ternary commitment depend on whether that
commitment **meshes with the local Q3 optimization trajectory induced by this
training order**?

The motivating observation is that order 271828 has a larger/more disjoint Q9
disagreement set, native Q9 remains under-settled during preparation, and
true-mask d=0.5 barely beats the matched-random control.

#### Frozen preparation

- model: `ibm-granite/granite-4.0-350m`
- seed/order: **271828**
- same BF16 compute path and FP32 masters as G1
- same direct-only-selected constant LR `1e-4`
- same first 300 shuffled chunks
- D = Q3 for 300 updates
- S = Q9 for 300 updates
- original Q3 scales define projected D/S codes
- M = positions where projected D and S Q3 codes disagree

No held-out test data is used for selection.

#### Trajectory-alignment score

At the completed D@300 state, restore the **original Q3 scales** and accumulate
the same CE35/KL65 Q3 gradient over the **last 24 already-consumed preparation
chunks** (training chunks 277–300 in this seed order). No optimizer step is
taken during this selector pass.

For each position `i` in M:

- `w_D[i]` is the direct-prepared FP32 master;
- `w_d50[i]` is the standard d=0.5 interior point for Q9's selected target
  ternary code;
- `g[i]` is the accumulated Q3 master-weight gradient on the selector window.

Define:

`alignment_score[i] = -g[i] * (w_d50[i] - w_D[i])`

Positive / larger score means the local first-order Q3 objective predicts that
moving D toward the Q9-selected d=0.5 commitment is helpful. Negative / smaller
score means the local trajectory opposes that commitment.

The selector window uses only training data already consumed during preparation,
so it does not look ahead into chunks 301–1200.

#### Matched alignment split

To control for layer and transition composition, M is split **within every
layer and every D-code→S-code transition class**.

For each such class with `n` positions:

- top `floor(n/2)` by alignment score → **AlignTop50**
- bottom `floor(n/2)` → **AlignBottom50**
- if `n` is odd, the one middle-ranked position is dropped from both halves

Thus AlignTop50 and AlignBottom50 must have exactly matched per-layer
source→target transition counts and equal total size. They must be disjoint and
subsets of M.

#### Continuation arms

All arms use D masters outside their selected mask, original Q3 scales, fresh
Adam, constant LR `1e-4`, and the exact same chunks 301–1200 for 900 Q3
updates.

1. **D** — D masters everywhere.
2. **All-d50** — all true-M Q9-selected codes placed at d=0.5.
3. **AlignTop50-d50** — only the matched top-half trajectory-aligned proposals
   placed at d=0.5.
4. **AlignBottom50-d50** — only the matched bottom-half proposals placed at
   d=0.5.

The historical full-S arm is not needed for this development question; D and
All-d50 are rebuilt in the same job as reproducibility anchors.

#### Hard construction assertions

Before continuation:

- AlignTop50 and AlignBottom50 have equal total size;
- they have exactly equal per-layer D→S transition counts;
- they are disjoint;
- both are strict subsets of M;
- All-d50 exactly realizes S's projected Q3 codes on all of M;
- each half-arm realizes S's projected Q3 code exactly on its selected half and
  D's projected Q3 code elsewhere.

Any assertion failure is a **technical failure**, not a scientific result.

#### Primary interpretation

This is a development experiment, so the outcomes are descriptive rather than
confirmatory.

Evidence for a seed/Q9 trajectory-mesh signal requires:

1. `AlignTop50-d50` finishes better than the exactly matched
   `AlignBottom50-d50` arm; and
2. the top-half selector improves meaningfully over the unfiltered
   `All-d50` arm.

A **practical rescue** additionally requires AlignTop50-d50 to beat D on the
held-out endpoint.

If top and bottom are similar, the gradient-alignment concept is not supported.
If All-d50 remains best, filtering by this score is counterproductive.

No 1729/424242 replication and no larger-model job is authorized by this
preregistration.


#### G1-6 pinned implementation

Script:
`ternary_pet/g1_granite350m_trajectory_alignment_seed271828.py`

Pinned implementation commit:
`10110ffba0ce9b1069f015cd32b235958e37f6c9`

The implementation follows the preregistration above. The alignment ranking is
performed separately within each layer and D→S transition class, so the top and
bottom halves are exactly matched for those known structural factors.


#### G1-6 launch record

Pinned code:
`10110ffba0ce9b1069f015cd32b235958e37f6c9`

Hugging Face job:
`6ac71a23df2184ac91ac73ca`

Hardware:
A10G-small

Wall-clock cap:
75 minutes

This job is the single authorized trajectory-alignment development run on seed
271828. No replication jobs are authorized by this launch record.


### G1-7 — Granite port of the Smol v10-v13 matched schedule, hard order 271828

**Status before launch: preregistered schedule-transfer experiment.**

#### Motivation

Granite G1 used the v7-like constant learning rate `1e-4`, selected by the
Granite direct-only 300-step calibration. The strongest later Smol mechanism
experiments (v10-v13), however, used a different matched global schedule:

- steps 1-100: linear warmup to `1e-3`;
- steps 101-1200: cosine decay to `1e-4`;
- Q9 occupies global steps 1-300;
- Q3 occupies global steps 301-1200;
- the LR curve continues across the Q9-to-Q3 boundary and does not restart.

G1-7 tests whether that protocol difference explains some of Granite order
271828's failure.

The Granite G1-1 calibration found the warm-`1e-3` schedule worse than
constant `1e-4` at the 300-step direct-Q3 validation checkpoint. G1-7 does
not override that calibration or claim the schedule is Granite-optimal. It is a
deliberate transfer of the **full Smol v10-v13 schedule** to test protocol
sensitivity.

#### Frozen model / data / precision

- model: `ibm-granite/granite-4.0-350m`
- seed/order: **271828**
- same 1,200 shuffled WikiText-2 chunks, sequence length 128
- same fixed 24-chunk validation diagnostic and 64-chunk held-out evaluator
- same FP32 persistent masters
- same BF16-rounded common source
- same BF16 autocast and BF16 teacher required by Granite
- same CE35 + KL65 objective
- AdamW beta=(0.9,0.95), zero weight decay, clip norm 1.0
- target all `nn.Linear` except exact `lm_head`
- nonquantized parameters frozen
- original Q3 scales restored and fresh Adam at the step-300 mechanism
  continuation, exactly as in the Granite G1-3 mechanism design

The only intended scientific change relative to the prior Granite mechanism
run on 271828 is the **global learning-rate schedule**.

#### Preparation and mask

Rebuild, from the common source:

- D: direct Q3 for 300 updates under the v10-v13 global schedule;
- S: Q9 for 300 updates under the same global schedule.

Project both through the original Q3 scales and define M at positions where
their projected Q3 codes disagree.

#### Continuation arms

All arms use original Q3 scales, fresh Adam, and the same remaining chunks
301-1200. Their optimizer LR begins at the global step-301 value of the same
v10-v13 schedule and continues to `1e-4` at step 1200.

1. D
2. S
3. M-exact
4. M-d50
5. Random-d50, matched outside M by layer and source-to-target transition counts

The same hard construction assertions as G1-3/G1-4/G1-5 must pass.

#### Primary questions

For order 271828 under the transferred Smol schedule:

1. Does native Q9 preparation settle better relative to direct by step 300?
2. Does full S beat D after the common Q3 continuation?
3. Does M-d50 beat D and/or full S?
4. Does true-M d50 clearly beat the matched-random control?
5. How do mask size, changed-set overlap, and code survival compare with the
   historical constant-`1e-4` 271828 result?

This is a **development/protocol-sensitivity** test on a known negative order,
not an independent confirmation. No other seed or larger-model job is
authorized by this preregistration.


#### G1-7 pinned implementation

Script:
`ternary_pet/g1_granite350m_v10_schedule_mechanism_seed271828.py`

Pinned implementation commit:
`ab02c7a64a357bf792eee8d361032d2e2310bb1b`

This is the Granite BF16-safe port of the Smol v10-v13 global schedule. The
five mechanism arms and construction assertions are otherwise inherited from
the established Granite G1-3 design.


#### G1-7 launch record

Pinned code:
`ab02c7a64a357bf792eee8d361032d2e2310bb1b`

Hugging Face job:
`6ac722dfdf2184ac91ac75e9`

Hardware:
A10G-small

Wall-clock cap:
75 minutes

This is the single authorized Granite hard-order matched-schedule transfer
experiment. Do not launch 1729/424242 replicas unless this result is inspected
first.


#### G1-7 first launch technical failure and preregistered amendment

First launch:
`6ac722dfdf2184ac91ac75e9`

The job completed both 300-step preparation arms successfully, then failed
before any continuation or held-out endpoint was evaluated while constructing
the preregistered full-size matched-random control.

Failure:
`('insufficient_random_candidates',
'model.layers.0.shared_mlp.input_linear', source_code=0,
available_outside_M=342644, required=595275)`

Under the transferred v10-v13 schedule, at least one per-layer/source-code
stratum contains more true-mask transitions than there are distinct outside-M
positions with the same direct source code. Therefore an exact full-size random
control matched simultaneously by layer and source→target transition is
mathematically infeasible without sampling with replacement or relaxing the
matching constraints.

This is a **technical design failure**, not a scientific outcome.

Before retrying, the G1-7 schedule-transfer protocol is amended as follows:

- keep D, S, M-exact, and M-d50 unchanged;
- omit Random-d50 from this retry;
- preserve the exact same model, seed/order 271828, BF16-safe compute path,
  300/900 split, v10-v13 global LR curve, scale reset, fresh Adam, data, and
  evaluators;
- retain all equal-forward assertions among S / M-exact / M-d50;
- report D/S geometry and M-d50 survival as before;
- do **not** make any new position-specificity claim from G1-7.

This amendment is made after observing only preparation-stage training logs and
the random-control construction failure, but **before any continuation or
held-out G1-7 endpoint exists**.

If a schedule-transfer effect appears, the matched-random specificity control
must be addressed separately with a newly preregistered feasible subset design.


#### G1-7b retry launch

Amended pinned code:
`808fd4975d16111d9c1c841c44d9038a01995ba3`

Hugging Face retry job:
`6ac7248adf2184ac91ac768d`

Hardware:
A10G-small

Wall-clock cap:
75 minutes

Arms:
D, S, M-exact, M-d50.

This retry follows the preregistered amendment above. The omitted random control
must not be inferred or reconstructed from this run.


#### G1-7b completed outcome — negative schedule transfer (2026-10-08)

Retry job `6ac7248adf2184ac91ac768d` completed successfully at
2026-10-08T05:26:40.955Z. Frozen amended implementation:
`808fd4975d16111d9c1c841c44d9038a01995ba3`.

All four retained arms (D, S, M-exact, M-d50) ran their complete 300-step
preparations / 900-step Q3 continuations. All equal-forward assertions passed;
S, M-exact and M-d50 had identical projected Q3 codes and initial diagnostics.
There was **no matched-random arm** in this run.

| Arm | Constant-LR 271828 loss | v10-v13 LR transferred loss | New PPL |
|---|---:|---:|---:|
| D | 5.72673 | **6.00895** | **407.05** |
| S | 5.84103 | 6.36920 | 583.59 |
| M-exact | 5.80453 | 6.35442 | 575.03 |
| M-d50 | 5.78502 | **6.30402** | **546.77** |

The S-vs-D deficit is now **0.36025 nats** (versus 0.11431 nats under constant
1e-4). M-d50 remains **0.29508 nats worse than D** despite improving S by
0.06517 nats.

At step 300:
- D and Q9 native validation losses: **6.45034** and **6.83970**.
- Fixed-Q3 projected D and S/M-exact/M-d50 validation losses:
  **6.81094** and **7.23919**.
- D changed **27.107%** of original Q3 codes; S changed **23.432%**.
- D-vs-S projected disagreement mask grew to **32.7145%**
  (81,642,667 / 249,561,088 targeted weights), compared with **6.9946%**
  under historical constant-LR Granite 271828.
- At prep step 300, Q9 preclip gradient norm was **9.7784**, versus
  **1.8998** for D (5.15x). These were clipped to norm 1.0 for optimizer steps.

M-d50 Q9-selected code survival at continuation 900: **60.75%**, versus
**89.59%** under the earlier constant-LR run; fixed-alpha0 code survival was
60.70%, closely matching learned-scale survival.

**Conclusion:** direct transfer of the Smol v10–v13 warmup-to-1e-3/cosine
schedule is **harmful for Granite order 271828**, hurting both D and S and
hurting S more. The large disagreement mask and lower survival mark a very
different assignment regime; they do not independently prove what causes the
final loss. This is a known-hard-order development/protocol-sensitivity result,
not a new confirmatory seed. No inference about matched-random specificity
under this schedule is possible.

Raw: `results/run_g1_7b_granite350m_v10schedule_seed271828_2026-10-08.json`.
Summary: `results/run_g1_7b_granite350m_v10schedule_summary.md`.
No additional jobs launched.


### G1-8 — Granite Smol v10–v13 schedule transfer on the weaker positive order 424242

**Status: preregistered before launch. One additional GPU experiment authorized
by user.**

#### Scientific motivation and selection rule

The v10–v13 Smol LR schedule was transferred to Granite on order 271828 in
G1-7b, with a negative result: D=6.00895, S=6.36920, M-d50=6.30402, mask
32.7145%. The earlier Granite constant-1e-4 confirmation gave mixed staging
effects across orders:

| Training order | D loss | S loss | D-S gain |
|---|---:|---:|---:|
| 1729 | 5.66205 | 5.53495 | +0.12710 |
| 271828 | 5.72673 | 5.84103 | -0.11431 |
| **424242** | **5.80192** | **5.74057** | **+0.06135** |

**424242 is selected prospectively as the harder of the remaining two positive
orders**: its historical staging gain (+0.06135) is smaller than 1729's
(+0.12710). This is a choice using *known outcomes*, not an unseen random
selection. It is therefore a targeted robustness/protocol-sensitivity test,
not fully independent confirmation.

Question: does the destructive response to the Smol v10–v13 learning-rate
schedule occur beyond the historically negative order 271828, or does
order-specific heterogeneity dominate even under the transferred schedule?

#### Frozen protocol — exact G1-7b harness, seed only changes

Use the exact G1-7b successful four-arm implementation, change **only**
`SEED=271828` to `SEED=424242`, and commit/pin the resulting separate script
before any GPU launch. No algorithmic tuning, gradient-selector logic, or
data/evaluator changes.

- model: `ibm-granite/granite-4.0-350m`
- source BF16 rounded; persistent FP32 master weights
- BF16 autocast/student and BF16 teacher (Granite-safe; no FP16)
- same WikiText-2 tokenizer, training chunks and evaluation slices
- same CE 0.35 + teacher KL 0.65 objective, rowwise trainable scales
- same quantized target modules, nonquantized parameters frozen
- AdamW betas (0.9,0.95), weight decay 0, clip norm 1
- 1200 chunks of length 128, shuffled by order 424242
- 100-step linear LR warmup to `1e-3`, then cosine decline to `1e-4` at
  global step 1200; no LR restart at switch
- prep: 300 Q3 updates (D) and independently 300 Q9 updates (S)
- continuation: 900 Q3 updates over the same remaining ordered chunks,
  fresh Adam and original Q3 scales
- same 8192-token held-out test, same validation setup

#### Four arms, as G1-7b

1. **D** — direct-prepared Q3 master state.
2. **S** — full 300-step Q9-prepared master state.
3. **M-exact** — D off the D/S disagreement mask; exact S master inside mask.
4. **M-d50** — D off mask; S-selected projected Q3 target codes set at
   normalized interior depth 0.5 on the true mask.

**No Random-d50.** The full-size exactly matched random control became
mathematically infeasible under this schedule on 271828. Keeping the exact
four-arm G1-7b design means G1-8 tests schedule transfer without claiming
new position-specificity evidence. Do not secretly substitute a weaker random
matching scheme.

#### Hard checks and diagnostics

Assert identical ternary projected codes and identical pre-continuation
validation diagnostics among S, M-exact, and M-d50. Abort as a technical failure
if any construction or numerical assertion fails.

Log native validation and preclip gradient norms at step 300, projected D
and S movement relative to original ternary codes, full D-vs-S disagreement
mask, M-d50 code survival after 100/300/900 Q3 continuation steps, final
8192-token held-out CE/PPL/top1/KL for all four arms, and the complete JSON.

#### Interpretation frozen before seeing results

- Compare within-order D vs S and D vs M-d50; compare all four absolute
  endpoints, mask fraction, and survival against the historical **constant-LR
  424242** run.
- If D and S both degrade and the mask grows substantially on 424242, that
  supports **general Granite schedule sensitivity** beyond the hard 271828
  order; it does **not** prove all Granite seeds fail at `1e-3`.
- If 424242 instead retains a staged advantage or similar mask/survival to
  the constant schedule, that favors a seed-by-schedule interaction over a
  simple family-wide explanation.
- If full S loses but M-d50 wins, report that distinction honestly.
- These are targeted, historically informed trials, not an unbiased
  population estimate; do not re-label old 2/3 constant-LR results.
- This cannot by itself invalidate the independently replicated Smol v10
  result, which holds under its own family-specific schedule and setup.

No 1729 test, intermediate-LR tuning, adaptive LR, additional seeds, new
model families or automatic reruns are authorized as part of G1-8.


#### G1-8 code pin (before GPU launch)

- Script: `ternary_pet/g1_granite350m_v10_schedule_mechanism_seed424242.py`
- Frozen implementation commit: `39399baff38b15f961e9571f142e193a783c925b`
- Parent implementation: completed G1-7b (`808fd4975d16111d9c1c841c44d9038a01995ba3`)
- Code delta: **exactly one line**, `SEED=271828` → `SEED=424242`
- Arms: D, S, M-exact, M-d50 (no Random-d50)
- Runtime: A10G-small, one detached job; no auto-replication.


#### G1-8 actual launch record

- HF Job: `6ac79551e7a0dae8a2788f0b`
- URL: `https://huggingface.co/jobs/codeflash85/6ac79551e7a0dae8a2788f0b`
- Status at submission: SCHEDULING (2026-10-08 13:06:25 UTC)
- A10G-small, 75-minute wall-clock cap, detached
- Pinned code: `39399baff38b15f961e9571f142e193a783c925b`
- The run was authorized as **one order 424242 test**. No 1729 launch, tuning,
  or other trial has been requested.


#### G1-8 completed result — Granite 424242 schedule transfer (2026-10-08)

G1-8 job `6ac79551e7a0dae8a2788f0b` finished successfully at
2026-10-08T13:30:40.325Z. Code:
`39399baff38b15f961e9571f142e193a783c925b`.
Four arms D, S, M-exact, M-d50 completed 900 Q3 continuation steps each
after 300-step preparation. All code identity and pre-continuation
equal-forward assertions passed.

| Arm | Constant-1e-4 424242 held-out loss | Transferred Smol LR held-out loss |
|---|---:|---:|
| D | 5.80192 | **6.05708** |
| S | 5.74057 | 6.06241 |
| M-exact | 5.74471 | 6.04609 |
| M-d50 | **5.71016** | **6.02857** |

The transfer worsens **every arm**. Full S vs D moves from a +0.06135-nat
advantage to a -0.00534-nat disadvantage (a near-tie on this one run).
M-d50 **still beats D by 0.02850 nats**, but its original constant-LR
advantage was 0.09176 nats.

At step 300, projected D and S changed **27.280%** and **27.124%**,
respectively, of original Q3 codes (compared with ~5.34% and ~5.27% at
constant 1e-4). D/S disagreement grows from **6.0296% to 29.4904%**:
15,047,433 to **73,596,548** targeted weight positions.

The M-d50 Q9-selected-code survival at Q3 step 900 drops from **90.57%**
(constant-LR) to **57.14%** (Smol schedule).

**Important distinction from the prior negative seed**: Q9 native prep at
step 300 remains *better* than direct on 424242, by **0.09232 nats**,
and Q9 preclip gradient norm is **1.257** vs D **1.941** at step 300.
Under the transferred schedule on 271828, native Q9 was 0.38936 nats
worse than direct and Q9/D gradient ratio 5.15x. The high-peak
schedule is harmful to both seed orders, but their preparation dynamics
are not identical.

The transferred-Smol results therefore support **schedule sensitivity in
Granite beyond seed 271828**, not the strong claim that Q9 always loses to
direct under this schedule. On 424242, the M-d50 intervention retains a
modest within-run benefit even though all absolute endpoint losses are worse.

Random-d50 was NOT run in this four-arm protocol. Seed 424242 was selected
using the historical smaller positive staging gain of the two remaining
positive Granite orders, not independently sampled. No causal claim that
mask expansion alone produces the final loss is established.

Raw: `results/run_g1_8_granite350m_v10schedule_seed424242_2026-10-08.json`
Summary: `results/run_g1_8_granite350m_v10schedule_summary.md`

G1-7b and G1-8 now show two-order negative schedule sensitivity:
the copy of Smol's peak 1e-3 LR hurts D and S absolute endpoints and
produces roughly 30%-plus D/S disagreement on both orders. The direct
model's response and the Q9-specific response must still be analyzed
separately. Do not automatically launch seed 1729 or a new LR sweep.


### G1-9 — Granite ternary QAT Gaussian noise × gridward interpolation factorial (pre-registered)

**Status before code / GPU launch: preregistered. User approved one experiment.**

#### Motivation

G1-1 chose constant `1e-4` from direct-Q3 validation after 300 steps.
G1-7b and G1-8 demonstrated that copying Smol's high-peak `1e-3`
schedule makes Granite absolute endpoints worse on two chosen orders while
increasing D/S projected code disagreement to ~30%. The near-boundary
geometry and WinQ (Li et al., ICML 2026) motivate investigating low-bit
training *within Granite's better-tested constant LR*.

WinQ applies Gaussian noise to continuous latent weights **before** their
quantization for the forward/gradient and periodically interpolates latent
weights toward their own quantized values. The proposed G1-9 is
**WinQ-inspired**, adapted to per-row scales and a short 1,200-update budget;
it is not a faithful benchmark of the authors' full long-run recipe.

#### Chosen order and decision status

Use **seed/order 271828** (known hard Granite order) for a **development
rescue test**, not independent confirmation. No new inference about family
generalization from this single order. No hyperparameter tuning against the
8,192-token held-out test. Model: `ibm-granite/granite-4.0-350m`.

#### Frozen common conditions

- BF16-rounded common model source; persistent FP32 masters; BF16
  student/teacher compute; exactly same target linears excluding lm_head;
  all other parameters frozen.
- 1,200 identical seed-shuffled WikiText-2 train chunks of 128 input tokens;
  identical separate 24-chunk *training-split* validation diagnostic and
  64-chunk / 8,192-token held-out test; same tokenizer, same fixed teacher.
- objective: CE35 + teacher KL65; AdamW beta=(0.9,0.95), weight decay zero;
  gradient clipping at norm 1.0; **constant LR=1e-4 on every update**.
- every arm starts from the same BF16-rounded weights and initialized row
  scales and sees the exact same ordered input chunks.
- each arm runs **300 direct-Q3 preparation + 900 direct-Q3 continuation
  updates**. At global step300, restore the original per-row Q3 scales,
  preserve the FP32 masters, and reset Adam (as in historical Granite D).
  The 1,200-step budget and optimizer reset apply identically to all arms.
- No Q9 intermediate phase in G1-9; **all arms are three-state ternary**.
  Inherited quantizer Q3 levels {0, ±(2/3)alpha}, boundaries ±alpha/3.

#### Frozen four-arm 2×2 design

| Arm | Gaussian perturbation before ternary quantization | Periodic gridward master interpolation |
|---|---|---|
| D | no | no |
| G | yes | no |
| P | no | yes |
| GP | yes | yes |

- **Noise:** for each *training* forward of targeted FP32 master W,
  with current rowwise alpha, sample independently
  `epsilon ~ N(0,1)` and compute the hard Q3 forward from
  `Q3(W + sigma_u(step)*alpha*epsilon)`; use the **same STE identity
  gradient wrt W** as the parent Granite code. During validation,
  code snapshots, pull, and held-out inference, use deterministic
  **Q3(W)**, with **no injected noise**. Noise is not a relaxation to FP32
  inference. Seed RNG at each arm creation and switch; no extra optimizer
  updates.
- **Noise schedule:** `sigma_u=0.04` normalized per-row for
  global steps 1–900, then linearly decrease to `0` by global step
  1200; deterministic zero-noise evaluation throughout. This normalized
  schedule is an *a priori adaptation* to Granite row scales, not a
  reported WinQ-optimal setting.
- **Gridward interpolation (P and GP):** *after* optimizer updates at
  global steps 100, 200, 300, 400, 500, 600, 700, 800 and 900,
  apply `W <- 0.90 W + 0.10 Q3(W)`, using each row's **current**
  learned scale, with no change to Adam states. No gridward pulls during
  steps 901–1200, to provide a clean recovery window. Original Q3
  scales are nevertheless restored at step 300 in all arms.
- No noise or gridward-pull hyperparameter choices may be changed after
  viewing the held-out result. Both G and GP use exactly the same Gaussian
  schedule, both P and GP use the same gridward schedule.

#### Hard invariants and diagnostics

- Assert all initial Q3 codes / rowwise scales match the common reference,
  quantization remains ternary, FP32 masters persist, and that all arms
  complete the same 1200 updates, with finite loss/gradients.
- Verify that the D arm reproduces the historical G1 271828 constant-LR
  direct endpoint to a reasonable numerical tolerance. If it does not,
  **report the discrepancy**, and treat within-job D as the primary
  baseline rather than silently correcting the run.
- Log native deterministic validation at step300 (before reference
  scale reset), and deterministic fixed-Q3 post-reset validation at
  step300; log the noised forward training CE/KL and pre-clip grad norm
  at steps 1/100/200/300/.../1200; log discrete Q3 code movement
  versus initialization and final held-out CE/PPL, teacher top1 and KL
  for each arm.
- Log **per-update quantized-code transition and immediate reversal**
  estimates on a predetermined, reproducible sample of weights,
  using noise-free codes even for noise arms. This distinguishes
  persistent decisions from simple back-and-forth oscillation. All
  sample positions must be determined from seed without validation/test.
- For any numerical or shape-construction failure, halt and report a
  **technical failure**, not a scientific negative.
- Raw JSON, implementation SHA, job ID, positive/negative outcomes,
  audit records, and all arms to be written to project reports.

#### Frozen interpretation, not tuned on test

Primary: does the combination GP improve held-out hard ternary CE loss over
**within-run D**, and does either isolated component explain the effect?
Secondary: margins/flip/reversal stability and code movement. Report all
four results even if all modifications are worse. Compare with historical
G1-4 D=5.72672517, but interpret paired within-job D first.

This factorial isolates the direct-Q3 **training mechanism**; it does **not**
by itself establish Q9→Q3 rescue, position specificity, new Gaussian
boundary predictor, or generalization across independent orders. The
historical random and matched-random controls remain intact and are
*not replaced*; a new margin-matched random/predictor study requires
a separate preregistration.

**One A10G-small job with four arms is authorized; no automatic additional
seeds, tuning, or larger models.**


#### G1-9 implementation pin before launch

- Immutable script: `ternary_pet/g1_9_granite350m_gaussian_pull_seed271828.py`
- Pinned commit: `6ce4c4056e8cfd3292c54f34bedc68bf7691688f`
- Origin: same Granite G1-3 BF16 source/quantizer/optimizer/data/score
  utilities; only direct-Q3 factorial experimental harness added.
- Startup smoke asserts zero-noise Q3 equivalence, deterministic
  noise-free eval, correct sigma annealing, and a fixed 249,561,088
  quantized-weight target count before scientific endpoints.
- 16 stratified layers × 2048 sampled weights each (32,768 positions)
  monitor clean per-step ternary-code flips and immediate reversals;
  sampled indices depend only on seed.
- **No pilot/parameter sweep**, one fixed factorial run; first arm is
  within-job D, followed by G, P, GP.


#### G1-9 first GPU submission attempt: rate-limited (no job created)

A single detached `uv run` submission was attempted for the pinned script
`6ce4c4056e8cfd3292c54f34bedc68bf7691688f`, A10G-small,
90-minute cap. Hugging Face connector replied:

`429 Too Many Requests` at `https://huggingface.co/mcp?login`.

**No HF job ID was issued, so no G1-9 run was launched or completed
on that attempt.** This is an infrastructure submission failure and is
not a scientific result. It authorizes no change to arms, seed,
hyperparameters or held-out interpretation.


#### G1-9 second GPU submission attempt: same rate limit

A single subsequent retry with **identical** pinned script and identical
A10G-small `uv run` parameters also returned
`429 Too Many Requests` from Hugging Face MCP login. Again no HF job ID.
**G1-9 is prepared but NOT RUNNING.** No automatic further submission
attempts should be made in this turn; do not interpret lack of results as
experimental failure.


#### G1-9 successful GPU resubmission

On 2026-10-08T23:06:20.440Z, after the previous two 429 failures, Hugging Face accepted exactly one detached G1-9 run on A10G-small (90-minute cap).

- Job ID: `6ac821ec095c5780892ff7f9`
- Job URL: `https://huggingface.co/jobs/codeflash85/6ac821ec095c5780892ff7f9`
- Status at submission: **SCHEDULING**, not scientifically complete
- Script pin unchanged: `6ce4c4056e8cfd3292c54f34bedc68bf7691688f`
- Arms/settings/seed unchanged; no duplicate jobs found in HF job list before launch.
- Previous two 429 responses remain documented as infrastructure failures, not jobs.


#### G1-9 FINAL result — gridward pull rescues Granite direct Q3 on 271828

HF job `6ac821ec095c5780892ff7f9` completed **successfully** at
2026-10-08T23:32:17.205Z (same code pin `6ce4c4056e8cfd3292c54f34bedc68bf7691688f`).
The two historical 429 submissions were infrastructure failures, but this
third successfully accepted job produced a full valid experiment.

All four arms D/G/P/GP completed their preregistered 300+900 optimizer
updates. The direct control exactly reproduced the independent historical
Granite 271828 endpoint. All 5 recorded structural/finiteness/step-count
checks passed, including 9 gridward pulls in each P/GP arm.

| Arm | Final heldout CE loss | PPL | D-minus-arm improvement |
|---|---:|---:|---:|
| D (clean Q3) | 5.726725 | 306.96 | — |
| G (Gaussian only) | 5.743370 | 312.11 | −0.016645 |
| P (gridward only) | **5.489709** | **242.19** | **+0.237016** |
| GP (both) | **5.487853** | **241.74** | **+0.238872** |

The **single positive main effect comes from gridward interpolation**:
`W <- 0.9W + 0.1Q3(W)` after optimizer steps 100,200,...,900.
Gaussian noise alone was a small negative; adding Gaussian to gridward
improved P by only **0.001856 nats**, far too small to call a robust benefit.

The 32,768-weight deterministic sample measured per-update, *noise-free*
code-transition frequency over the 900-step continuation:
D **0.000735745**; G **0.000741340**; P **0.000094469**;
GP **0.000089281**. P has **87.2% fewer sampled code-flip events**.
Final Q3 code movement from source: D **10.335%**, P **5.281%**.
Sampled immediate reversal *fraction of flips* does not fall by nearly
the same proportion, so transition reduction is not proof oscillations
specifically caused the better endpoint.

This is a **targeted, one-order (previously hard seed 271828) development
result**, not cross-seed confirmation. The GP−P delta is tiny and without
uncertainty quantification. Gridward interpolation and latent-noise training
have relevant published WinQ prior art; do not claim invention of the general
method. No Q9 arm, no random specificity control, and no new model family
were tested in G1-9. Future frozen replication needed.

Raw: `results/run_g1_9_granite350m_gaussian_pull_seed271828_2026-10-08.json`.
Summary: `results/run_g1_9_granite350m_gaussian_pull_summary.md`.
No follow-up GPU jobs launched.


### G1-10 — frozen G1-9 factorial replication on Granite seed 424242

**Status: preregistered BEFORE producing G1-10 code or launching GPU.**
User explicitly authorized one next-seed replication of G1-9. Order 424242
was named prospectively as the next Granite replication in the G1-9
outcome discussion (2026-10-08). This is another *previously observed*
Granite order (constant-LR Q9→Q3 advantage +0.06135), not a new blind order.

#### Scientific question

Does the strong periodic gridward latent-weight interpolation result from
G1-9 on Granite hard order 271828 (within-run D=5.726725, P=5.489709,
GP=5.487853, Gaussian-only G=5.743370) replicate on historically positive
Granite order **424242**, or is it development-order-specific?

The primary comparison is **P versus D on seed424242**. Secondary
comparisons are G versus D, GP versus P and D, deterministic code
movement, sampled per-step flips and immediate reversals, and top1/KL.
Every arm and metric is reported, whether positive or negative.

#### Strict frozen protocol (seed-only code change)

Create `ternary_pet/g1_10_granite350m_gaussian_pull_seed424242.py`
by copying **verbatim** the completed G1-9 script at pinned commit
`6ce4c4056e8cfd3292c54f34bedc68bf7691688f`, changing **only**
`SEED=271828` to `SEED=424242`. No parameter, optimizer, dataset,
architecture, quantizer, measurement, or algorithm changes.

- Model `ibm-granite/granite-4.0-350m`; BF16-rounded original source;
  FP32 master weights, BF16 student and teacher, same quantized
  linear-weight target (249,561,088 weights, nonquantized params frozen).
- Same 1,200 WikiText-2 train chunks, shuffled using the new seed,
  24 validation chunks and 64 final held-out test chunks (8,192 tokens);
  teacher objective `0.35*CE+0.65*KL`.
- Same direct **ternary / Q3** training in every arm; no Q9 phase;
  300 updates of direct-Q3 preparation + 900 Q3 continuation
  (original reference Q3 scales and fresh Adam at step300 for all arms).
- All arms same LR **constant `1e-4`**, AdamW beta(0.9,0.95)
  no weight decay, clip norm 1, exact same training order and
  evaluated slices. Same BF16 autocast protocol.
- D: standard hard ternary STE.
- G: Gaussian perturbation before **training forward quantization**
  `sigma_u=0.04` in each row's normalized scale, global steps1–900;
  linear anneal to sigma0 at step1200; noise off for all validations,
  code snapshots, grid pulls and held-out evaluation.
- P: FP32-master gridward step `W <- 0.9W+0.1Q3(W)`
  after Adam update at global steps100,200,...,900 (nine pulls),
  current learned row scales, no extra optimizer steps and no
  Adam reset besides common step300 reset.
- GP: exactly the combination of G and P.
- Identical 16 layer × 2048 weight deterministic code monitor,
  seed-based positions, clean code flips and immediate reversals.
- No random-mask arm is inserted into this direct-training factorial.
  Previous matched/random controls are preserved as separate experiments,
  not discarded or retroactively changed.

#### Validity and stopping rules

Check code equals G1-9 **except one seed assignment** before launch;
commit and pin immutable SHA. Check current HF job list to avoid duplicate
job submissions. Startup Q3/no-noise-eval smoke checks and all step,
arm, finiteness, target-weight, pull-count assertions unchanged.
For an exception or failed check, report a technical failure rather
than scientific failure and do not launch an automatic replacement.

Historical constant-LR Granite 424242 D held-out loss **5.80192006**
is a contextual reproducibility check; paired *within-G1-10* D
is the primary comparator. Report an unexpected D mismatch without
silently rerunning/adjusting the held-out evaluator.

A **single four-arm A10G-small Hugging Face Jobs run** is authorized,
same 90-minute cap as G1-9. No further order 1729, Smol, LR/noise/λ
calibration, or automatic expansion is authorized by this preregistration.
Read the final JSON before interpreting any scientific claim.

#### Frozen outcomes and interpretation

- If P and/or GP improves D substantially again, this strengthens
  **within-Granite cross-order evidence for WinQ-inspired gridward pull**,
  still short of new-model-family generalization.
- If G alone is negative/neutral, keep that as a separate conclusion;
  do not claim Gaussian caused a pull effect.
- If P fails, disclose order-specific instability; do not retune
  using the held-out sample.
- No assertion that code-flip suppression *causes* the improvement
  without a separate causal control.


#### G1-10 immutable implementation pin (before GPU launch)

- Frozen script: `ternary_pet/g1_10_granite350m_gaussian_pull_seed424242.py`
- Code commit: `e08659a51fd0d72ed85f01f8e7ce739283ca6c61`
- Parent G1-9: `6ce4c4056e8cfd3292c54f34bedc68bf7691688f`
- Verified difference: exactly `SEED=271828` → `SEED=424242`;
  *no other changes*.
- One job to run on `a10g-small`, 90-minute cap, detached;
  arm order D/G/P/GP, 300+900, same tests.


#### G1-10 launch record

Hugging Face accepted **one** detached A10G-small job on
2026-10-08T23:51:47.315Z (90-minute cap):
- HF job: `6ac82c93fee2c900701711db`
- URL: `https://huggingface.co/jobs/codeflash85/6ac82c93fee2c900701711db`
- Status at submission: **SCHEDULING**, not yet a scientific outcome.
- Immutable code SHA: `e08659a51fd0d72ed85f01f8e7ce739283ca6c61`
- This is the exact G1-9 factorial script except the training-order seed
  `271828` → `424242`. No duplicates were listed before launch.
- Archive complete JSON, validate all four arm checkpoints, and
  compare paired D/P/GP/G after job reaches terminal state.


#### G1-10 COMPLETED — seed-only replication confirms gridward pull on 424242

HF job `6ac82c93fee2c900701711db` **COMPLETED successfully**
at 2026-10-09T00:21:46.422Z. Code pin `e08659a51fd0d72ed85f01f8e7ce739283ca6c61`
(the exact one-line G1-9 seed substitution). All D/G/P/GP arms
completed 300+900 updates and the five final recorded checks passed,
including nine gridward steps in P/GP, finite losses and expected
249,561,088 quantized target weights. Held-out evaluation is the
same 8192-token deterministic hard Q3 slice.

| Arm | Seed424242 held-out loss | PPL | D−arm gain (positive better) |
|---|---:|---:|---:|
| D | 5.801920056 | 330.9344 | — |
| G | 5.756361738 | 316.1958 | +0.045558 |
| **P** | **5.528112054** | **251.6683** | **+0.273808** |
| GP | 5.535340339 | 253.4940 | +0.266580 |

D reproduces the earlier fixed-LR Granite 424242 historical D
**exactly**: 5.801920056. The primary **P** result is positive
again at the unchanged schedule. P cuts held-out PPL by **23.95%**.
GP remains strongly better than D, but is **0.007228 nats worse
than P** on 424242; Gaussian-only G mildly improves D on this order,
contrary to seed271828 (where G was 0.016645 worse).

The *two-seed mean* D−P gain, for selected 271828 and 424242
orders, is **+0.255412 nats**; both seeds are positive. This
is **within-family cross-order** confirmation, **not** new-family
generalization or an unbiased replication sampling plan.

Monitored continuation-stage Q3 code-flip rate (32,768 fixed sampled
positions) was D=0.000728421 and P=0.000095486 per sampled
weight/update — P has **86.89% fewer sampled flips**.
D changed 10.327% of source Q3 codes by the final checkpoint,
P 5.369%. Flip reductions do not prove the gain is mediated solely
by oscillation correction.

**Raw JSON metadata caveat:** Because G1-10 differs in just
`SEED=424242`, inherited `kind=g1_9_...`,
`event=g1_9_...`, and
`effects.historic_D_loss_271828=5.7267251685` /
`D_minus_historic_D_loss=+0.07519489` labels are
**stale parent-run comparator fields**, not the proper historical
424242 baseline. The independent historical 424242 direct
loss is 5.801920056, and it exactly matches the G1-10 D.
Keep raw JSON unedited; use seed, HF job ID, and script pin for
identification.

Raw JSON: `results/run_g1_10_granite350m_gaussian_pull_seed424242_2026-10-08.json`.
Analysis: `results/run_g1_10_granite350m_gaussian_pull_seed424242_summary.md`.
Previous G1-9: `results/run_g1_9_granite350m_gaussian_pull_summary.md`.

The original Q9 mixed-seed findings and prior matched/random controls
remain separately valid; neither G1-9 nor G1-10 tested Q9 or random
assignment masks. Gridward interpolation has related published WinQ
prior art; cross-family validation would need fresh authorization.
No further GPU jobs were launched.


### S1-1 — SmolLM2 cross-family gridward replication against tuned direct Q3 (seed 1729)

**Preregistered before coding and GPU launch.** The user authorized trying
the Granite-positive gridward-pull training rule on SmolLM2-360M. We use
**seed/order 1729**, historically used for direct-Q3 tuning; therefore
the result is a **first cross-family development transfer**, not an
unseen-order or blinded confirmation. Do not select a better-seeming
Smol seed after viewing its outcome.

#### Primary question and fixed design

Does the frozen Granite G1-9/G1-10 gridward rule improve
SmolLM2-360M-Instruct **direct three-state ternary Q3** under Smol's
own established **best-tested direct-Q3 schedule**?

Two arms only, equal compute and data:

- **D**: canonical tuned direct Q3, no pulls.
- **P**: exactly the same direct Q3, except after training step
  `100,200,300,...,900` (nine pulls), update FP32 master weights
  `W ← 0.9 W + 0.1 Q3(W)`, where Q3 uses the **current learned
  output-row scales** and same true hard ternary grid.
  Pull occurs under `torch.no_grad()`, with no additional optimizer
  update, no optimizer-state reset, and **no adjustment of scales**.

The entire Smol direct recipe, sourced from the v9 tuned baseline
script `ternary_pet/smollm2_v9_direct_q3_tuning.py`, is held constant:

- model `HuggingFaceTB/SmolLM2-360M-Instruct`; original BF16-rounded
  FP32 master weights and trainable rowwise ternary scales;
  frozen other parameters; LM head excluded from quantized linears;
  original FP16 teacher/student autocast and AMP GradScaler (the
  **Smol**-validated compute path, not Granite BF16 autocast).
- 1,200 optimizer-step **opportunities** using the same 1,200
  seed-shuffled WikiText-2 chunks of 128 tokens, same 24 train-heldout
  diagnostic chunks and 64 held-out test chunks / 8192 tokens.
  Log AMP skipped steps if any; pulls require a valid optimizer
  update at their scheduled boundaries.
- same `CE_W=0.35` plus `KL_W=0.65` teacher distillation,
  AdamW β=(0.9,0.95), weight decay0, clip norm 1.
- global LR curve **steps 1-100 linear warmup to 1e-3, then cosine
  to 1e-4 at step1200**. Do **not** restart LR at step300.
- unlike Granite G1-9, Smol's **strong direct-v9 baseline trains
  continuously through step300**: **no original-scale restoration or
  fresh Adam at step300** in either arm. The transfer under test is the
  10%-gridward *rule*, not a full import of the Granite optimizer
  reset schedule.
- same original model, training/evaluation chunks, prompt generation,
  evaluation at 300/600/900/1200, and v9 telemetry in both arms.
  Add deterministic, dedicated-RNG 16-layer×2048-weight per-step clean-code
  monitor in both arms for flips and immediate two-step reversals;
  an independent CPU generator must not perturb global training RNG.
  Preserve baseline training/calibration call ordering.

**No Gaussian noise**, no Q9 arm, no margin/predictor experiment.
All original Q9 mask/random controls remain unchanged in prior scripts.

#### Required validity checks and interpretation

- Explicit verify source lineage from original v9 direct training
  functions and Smol global LR values; all 1,200 step opportunities,
  same order/config and 9 scheduled pulls only in P.
- Verify exact CPU smoke behavior of pull on representative Q3
  values: hard ternary codes remain the same during a convex pull,
  the FP32 latent weight moves closer to its current quantized
  prototype, and learned scales are not directly altered.
- At final report **test NLL, PPL, teacher-top1, teacher KL** for D and P,
  validation over time, noise-free hard-Q3 code movement from source,
  sampled flip rates and immediate reversals, AMP skipped-step counts.
- Historical tuned Smol v9 seed1729 D test-loss comparator:
  **5.595722187310457**. If within-job D does not reproduce it to
  sensible numerical tolerance, **report the mismatch explicitly**,
  do not hide it or reinterpret the historical baseline as control.
  New *paired within-run* D remains the primary baseline.
- Historical matched-schedule staged-Q9 result for 1729: ~4.9010
  loss, strictly **context only**. This test does not claim to
  replace staged Q9 without showing corresponding results.
- If P beats D, this is positive **cross-architecture evidence**
  for a published (WinQ-related) technique under two family-specific
  schedules; it remains short-budget/one-Smollm order and requires
  additional seeds before broad generalization.
- If P does not beat D, record a **cross-family negative** without
  retuning the strength/timing using the 8192-token test.
  No automatic repeat, LR change, bigger model, or seed sweep.

**Only one two-arm A10G-small Hugging Face Jobs launch
(90-minute cap) is authorized.** Code must be pinned immutably
before launch; avoid duplicates. Archive complete JSON regardless of
sign and update the reports.


#### S1-1 implementation pin before GPU launch

- Script: `ternary_pet/s1_1_smol360m_gridward_direct_q3_seed1729.py`
- Immutable GitHub commit: `a5634bce459092a503e7534e6b89b6d3a019548b`
- Historical Smol tuned-direct v9 source parent pin:
  `5c088e58539b2dede93df57ac3f72dbe0480a028`
- Verified that the v9 data, quantizer, optimizer, loss, FP16 AMP,
  model loading, schedule logic and original direct training helpers
  are **byte-for-byte the original pinned source after its docstring**;
  S1-1 adds a new two-arm harness without modifying those helpers.
- S1-1 arms `D,P`; chosen existing v9
  `warm100_cosine_1e-3` schedule; no step300 reset. Same nine
  10%-pull events as Granite. Independent CPU-RNG-sampled
  weight-transition logging should not perturb global training RNG.
- Baseline reproducibility target: original v9 seed1729 tuned
  direct Q3 test loss `5.595722187310457`.
- One accepted HF job ID to be appended below; no duplicate launch.


#### S1-1 successful cross-family Smol GPU submission

On 2026-10-09T00:58:35.070Z, Hugging Face accepted **one**
detached `a10g-small` job with 90-minute cap; job
`6ac83c3bfee2c90070171b1a`, URL https://huggingface.co/jobs/codeflash85/6ac83c3bfee2c90070171b1a.
Initial stage **SCHEDULING**, not scientifically complete.
Exact frozen code pin `a5634bce459092a503e7534e6b89b6d3a019548b`. No S1-1 duplicate found
in HF jobs list immediately before launch. Settings and two-arm
protocol unchanged from preceding S1-1 preregistration. Await
final explicit `FINAL_JSON` and baseline reproducibility check.


#### S1-1 FINAL result — significant negative Smol transfer, no job error

The preregistered S1-1 two-arm SmolLM2 gridward direct-Q3 experiment,
HF job `6ac83c3bfee2c90070171b1a`, completed **successfully**
2026-10-09 01:18:20.104 UTC on unchanged pinned code
`a5634bce459092a503e7534e6b89b6d3a019548b`.

Both arms D and P had all 1,200 global training-step opportunities,
**six identical AMP-skipped steps each**, **1,194 effective AdamW
updates each**, 8192 held-out test tokens, the same
Smol validated warmup-to-1e-3 and cosine-to-1e-4 schedule,
and identical objective/quantizer/evaluator. The P arm had
exactly nine `W ← 0.9W+0.1Q3(W)` pulls at
steps100,...,900, D zero. All **9** recorded
scientific construction/finite/schedule/smoke checks passed.

| Arm | Final heldout loss | PPL | Teacher top1 | Teacher KL |
|---|---:|---:|---:|---:|
| D tuned direct Q3 | **5.595722** | **269.27** | 29.87% | 2.66403 |
| P gridward only | 5.845746 | 345.76 | 27.28% | 2.93324 |

**Scientific negative:** P has **+0.250024 worse NLL** and
**28.41% higher PPL** than tuned D. The within-run
D reproduces Smol v9 historical baseline `5.595722187310457`
**exactly**, ruling out direct-baseline mismatch.
This is NOT a Q9 experiment and is not a GPU/script failure.

Despite worse heldout quality, P lowers per-weight per-step
sampled clean-Q3 transitions by **78.41%** relative to D
(D 0.000435054, P 0.000093918); full final code movement
from source D **6.960%** vs P **3.445%**. The
32,768-weight sample spans 16 layers and 1,200 steps.
Fewer Q3 code changes do not guarantee better heldout quality.

Time-course validation: P is slightly BETTER at global
step300 (D 5.98809, P 5.97667) and 600
(D 5.74032, P 5.67064), then WORSE at step900
(D 5.41510, P 5.58253) and 1200
(D 5.25594, P 5.57738). The large late quality
gap and near-zero monitored P flips at the end are
compatible with premature assignment commitment under
Smol's more aggressive LR schedule, but the causal
mechanism has NOT been isolated.

**Two Granite P-positive orders do not make the
pull strength portable without adaptation.**
Granite employed constant 1e-4 and optimizer/scale
reset at step300; Smol used warmup peak1e-3,
cosine and continuous Adam. Architecture and LR
interactions cannot be separated by this result.
Previous Smol Q9→Q3 3/3 improvement
remains intact as a distinct mechanism. The method
has published WinQ-type prior art.

Full raw, unmodified JSON:
`results/run_s1_1_smol360m_gridward_direct_q3_seed1729_2026-10-09.json`.
Summary:
`results/run_s1_1_smol360m_gridward_direct_q3_seed1729_summary.md`.

The previous submission/running records are historical;
**S1-1 is COMPLETED with a negative scientific result**.
No additional GPU jobs or retuning launched.

### T1 — Q9 assignment discovery timing, seed1729 (COMPLETED, 2026-10-08 EDT)

**Frozen prereg before compute:** `research_log/phase_a_q9_timing_seed1729_prereg_2026-10-08.md`.  
**HF job:** `codeflash85/6ac85534fee2c90070172a41`, [link](https://huggingface.co/jobs/codeflash85/6ac85534fee2c90070172a41), completed 2026-10-09 02:51:49 UTC.  
**Pinned code:** `2a982be6910d99eaf53d74f6e3ec3dae5807de17`.  
**Raw parsed FINAL_JSON:** `results/run_t1_q9_discovery_timing_seed1729_2026-10-08.json`.  
**Interpretation:** `results/run_t1_q9_discovery_timing_seed1729_summary.md`.

One bounded GPU job: matched 300-step D(Q3) and S(Q9) preparation under v10 global schedule, fixed-original-row-scale Q3 code snapshots at steps 0/50/100/150/200/250/300. No held-out evaluation, no Q3 continuation, and no Gaussian/gridward. All code positions represented by two-bit packing for retrospective comparisons within job. Construction checks passed: `valid_for_science=true`, exact native D/S validation losses, exact S original-Q3 projected validation loss, exact step300 disagreement mask **6.45808%**. D fixed-original-scale validation loss **6.004324** was +0.016230 nats above native D **5.988094**, within prereg 0.035 tolerance; record rather than suppress this diagnostic difference.

| Preparation step | D/S mask among all weights | Precision vs step300 mask | Recall of step300 mask |
|---:|---:|---:|---:|
| 50 | 0.8724% | 49.04% | 6.63% |
| 100 | 2.2303% | 53.88% | 18.61% |
| 150 | 3.7682% | 60.71% | 35.42% |
| 200 | 4.8758% | 68.02% | 51.35% |
| 250 | 5.7629% | 77.77% | 69.40% |
| 300 | 6.4581% | 100.00% | 100.00% |

The Q9-only and direct-only components both emerge gradually. By step250, **67.60% of final Q9-only** and **68.97% of final direct-only** disagreement positions are present. Early comparisons use *hindsight* step300 labels; this is not a predictor available online. A descriptive result with one seed, **not** proof that Q9 can transition at 200/250 with unchanged final quality. Roughly **30.60%** of final disagreement positions remain absent at step250. Which subgroup causally matters more, final optimal transition time, and cross-seed timing are unknown. This work did **not** test shorter Q9→Q3 training or improve the model.

**Next gate:** preregister either a causal split of the two disagreement categories (Phase B) or a matched-total-update Q9 switch-time experiment (Phase C) on a bounded GPU budget; seek explicit approval before more compute. Historical negative experiments preserved.

### C1 — Smol Q9 switch 250 versus 300, seed1729 (GPU SUBMITTED)

**Date:** 2026-10-08 EDT; **scientific status:** pending completion, not a result.

User requested a test of whether 250 Q9 prep steps can preserve the downstream benefit of the established 300-step Q9→Q3 switch, while fixing total training budget. [Frozen preregistration](research_log/phase_c_q9_switch250_vs300_seed1729_prereg_2026-10-08.md) committed **before GPU submission**. Step250 was selected after inspecting the T1 single-order timing masks; therefore the test is explicitly exploratory.

**HF job:** [codeflash85/6ac8597afee2c90070172c79](https://huggingface.co/jobs/codeflash85/6ac8597afee2c90070172c79), submitted with initial status SCHEDULING. Hardware A10G-small, timeout 90m. **Pinned script commit:** `5577a771ed57409283690097a58bae0c8966fdb1`; **script:** `ternary_pet/c1_smol360m_q9_switch250_vs300_seed1729.py`. Static Hugging Face syntax smoke passed, job `6ac85970095c578089301448`.

Three equal-budget arms on identical seed-shuffled WikiText-2 chunks and original BF16-rounded model:
- D, direct Q3 for 1,200 step opportunities (continuous optimizer/scales; tuned v9 baseline).
- S250, Q9 global steps 1–250, restore original ternary scales/reset Adam and GradScaler, then Q3 steps 251–1200 (950 continuation opportunities).
- S300, Q9 global steps 1–300, same reset, then Q3 steps 301–1200 (900 continuation opportunities; replicate v10 positive arm).

Same warmup+cosine global LR, FP32 masters/FP16 autocast, 35% CE+65% KL, fixed teacher, original Q3 scales after switching. Log same-train-chunk validation at switch before/after resetting scales, fixed validation checkpoints, optimizer skipped-step counts, final heldout 8192-token CE/PPL/top1/KL and code movement. **Predefined exploratory thresholds:** `L250-L300 <= 0.07` and 90% or more of `(LD-L300)` gain retained. D and S300 reproducibility checks each at 0.06 nats absolute against historical heldout.

**No results available as of submission.** On completion archive raw JSON and report both positive and negative outcomes. No additional seeds or follow-on studies authorized by this experiment.

### C1 result — switch250 preserved 95.75% of switch300 benefit (COMPLETED; 2026-10-08 EDT)

The **C1 run completed successfully** 2026-10-09 03:24:11 UTC; [HF job `6ac8597afee2c90070172c79`](https://huggingface.co/jobs/codeflash85/6ac8597afee2c90070172c79), pinned code `5577a771ed57409283690097a58bae0c8966fdb1`. Scientific construction checks passed, `valid_for_science=true`, no duplicate or follow-up job launched.

Raw, unchanged scientific FINAL_JSON values with provenance: [C1 full JSON](results/run_c1_smol360m_q9_switch250_vs300_seed1729_2026-10-08.json). Detailed findings: [C1 result summary](results/run_c1_smol360m_q9_switch250_vs300_seed1729_summary.md). Before-job prereg: [Phase C protocol](research_log/phase_c_q9_switch250_vs300_seed1729_prereg_2026-10-08.md). **This is exploratory seed/order1729**, with step250 chosen in response to already-viewed T1 and prior v10 data.

| Arm | Q9/Q3 schedule | Actual optimizer updates / 1200 opportunities | Held-out loss | Held-out PPL | Train-split dev loss @1200 |
|---|---|---:|---:|---:|---:|
| D | Q3 for 1200 | 1194 (6 skipped) | 5.595722 | 269.272 | 5.255937 |
| S250 | Q9 250 then Q3 950 | 1190 (10 skipped) | 4.930515 | 138.451 | 4.575021 |
| S300 | Q9 300 then Q3 900 | 1190 (10 skipped) | **4.900985** | **134.422** | **4.521716** |

Both historical endpoints **reproduced exactly**: D test loss `5.595722187310457`, S300 `4.9009853675961494`. S250 is **+0.0295298882 nats** worse than S300, but retains **95.7495%** of S300's gain over D. Both prospectively set **exploratory** thresholds PASS: `L250-L300 <= 0.07`, and `(LD-L250)/(LD-L300) >= 0.90`. S250's absolute gain over D = **0.6652069315 nats/token**.

At switch S250 Q9 native validation loss was **5.390830** and immediate original-Q3-scale projection loss **6.542107**. At switch S300 native Q9 loss **5.182719** and immediate projected Q3 loss **6.542292**; the two Q3 immediate projection losses are almost equal although Q9 native improves in the extra 50 preparation steps. This does NOT prove equal downstream trainability; their final held-out NLL differs by +0.02953 nats.

T1 diagnostic observed just **69.40%** step300 disagreement-mask recall at step250. Thus C1 demonstrates that **most benefit survives a 50-step shift from Q9 into Q3**, not that all late Q9 assignments were redundant. Earlier switch receives 50 extra Q3 updates, and learned optimizer states are reset at different times as defined in recipe. Do not claim pure assignment-geometry causal inference from this intervention.

**Next gate:** Freeze this 250-vs-300 comparison on fresh orders without retuning for confirmation, OR isolate causal contributions of Q9-only code changes and the direct-only changes Q9 avoids. No added GPU jobs until separate authorization. Preserve all previous negative reports.

### M1 — matched 900-step Q3 continuation from Q9(250) vs Q9(300) (SUBMITTED, no results yet)

**Frozen preregistration:** [M1 protocol](research_log/m1_equal_q3_continuation_seed1729_prereg_2026-10-08.md) recorded before compute; **copyable research plan:** [NEXT_EXPERIMENT_PLAN.md](NEXT_EXPERIMENT_PLAN.md). **HF job:** [6ac86712fee2c900701734bb](https://huggingface.co/jobs/codeflash85/6ac86712fee2c900701734bb); one A10G-small with maximum90m, initial state SCHEDULING. **Script:** `ternary_pet/m1_smol360m_equal_q3_continuation_seed1729.py`, pinned SHA `cad64c009209a924be89b523e6a1e184a4f4137b`. CPU syntax smoke HF `6ac86709fee2c900701734b2` passed.

**Question:** does the Q9 step250 prepared state produce similar final Q3 quality to Q9 step300 state if we remove C1's **extra fifty Q3 steps**? How often does continued Q3 training from step250 select code identities also present after Q9 step300?

**Design:** One Q9 seed1729 trajectory with FP32 master snapshots at prep steps250 and300. Both start Q3 with original source Q3 scales and new AdamW/GradScaler, train on the same **900** WikiText chunks historically global steps301..1200, using the same matched global LR values301..1200. Thus P250 arm gets 250+900 = **1150** total scheduled steps, P300 300+900 = **1200**. This is intentionally unequal-total compute and not directly a head-to-head equal-budget recipe test; existing C1 is the equal-budget experiment. The same Q3 continuation is the controlled manipulation. Evaluate existing heldout 8192 tokens only after continuation, no retuning; record Q9 native reference geometry/validation at both snapshots and exact final position-level code concordance/rediscovery counters across 314,572,800 targeted weights. Require S300 exact-family reproduction within 0.06 nats of 4.900985, prep native validation and code movement within frozen tolerances. Use only single familiar order; all outcomes exploratory, no generalization claim.

**At submission:** No M1 endpoint data available. On completion archive final JSON and positive, null, negative or technical findings; update [CURRENT_STATE.md](CURRENT_STATE.md), living reports, handoff; no automatic extra seeds/jobs.

### M1 result — equal Q3 continuation after 250 vs 300 Q9 prep (COMPLETED; 2026-10-09 UTC)

**HF job:** [6ac86712fee2c900701734bb](https://huggingface.co/jobs/codeflash85/6ac86712fee2c900701734bb); completed 2026-10-09 04:15:27 UTC. **Pinned script SHA:** `cad64c009209a924be89b523e6a1e184a4f4137b`; [frozen M1 preregistration](research_log/m1_equal_q3_continuation_seed1729_prereg_2026-10-08.md). [Raw M1 JSON](results/run_m1_equal_q3_continuation_seed1729_2026-10-09.json); [detailed summary](results/run_m1_equal_q3_continuation_seed1729_summary.md). All seven frozen checks passed, `valid_for_science=true`. This run is **exploratory** on seed1729 and the same previously viewed WikiText held-out slice.

M1 preserved the exact Q9 preparation setup from prior experiments. A single Q9 trajectory saved **original-scale projected Q3 snapshots and FP32 masters at step250/300**. Each state received **900 identical Q3 continuation chunks and LR values (global reference301..1200)** with fresh Adam/GradScaler and original Q3 scales. The 250-arm used **1150 total opportunities**, the 300-arm **1200**. This is not C1's equal-total-budget comparison.

| Arm | Prep Q9 steps | Q3 continuation | Effective Q3 optimizer updates | Heldout NLL | PPL | Dev NLL after Q3 |
|---|---:|---:|---:|---:|---:|---:|
| P250_900 | 250 | 900 | 897 (3 AMP skips) | **4.9504320212** | 141.23597 | 4.586918 |
| P300_900 | 300 | 900 | 897 (3 AMP skips) | **4.9009853676** | 134.42217 | 4.521716 |

**Scientific result:** P250 is **+0.0494466536 nats** worse under the same Q3 continuation. P300 heldout NLL, Q9 preparation native val and projected Q3 source-code changes all reproduce historical references; numerical and design checks passed. The shorter Q9 preparation remains far better than the historical direct-Q3 1200-step result, but that is not a matched-compute comparison.

**New full-model code result:** The t250 and t300 Q9 prep original-scale Q3 code snapshots disagree at exactly **6,245,131 / 314,572,800 positions (1.9852737%)**. During the identical Q3 continuation, the early-prepared final model selects the late-step300 Q9 code at **44.1636%** of these positions; the late-prepared final model retains its own t300 code at **59.2998%**. Their final codes agree at **60.0011%** of these initial-disagreement positions and **93.8580%** of all positions. Last 50 Q9 steps both add new source deviations (3,914,292 weights) and revert others (2,329,571 weights), with 1,268 rare recodes. This is code **movement/correspondence**, not causal importance, an oracle of optimal labels, or proof of distinct loss basins.

**Relation to C1:** C1 Q9(250)+Q3(950) NLL **4.930515** vs M1 Q9(250)+Q3(900) NLL **4.950432**. Cannot attribute the difference solely to 50 extra Q3 updates: C1's Q3 learning-rate values and training-batch indices begin at global step251, M1's at301. Preserve both controlled questions (equal total budget vs identical continuation) separately.

**Follow-up gate:** NO new GPU job launched. The most valuable next experiment is **longer paired direct/staged training with fresh evaluation data** and a frozen recipe, then independent training-order confirmation and stronger external QAT baselines. Any new jobs require a separately frozen plan and budget. [Authoritative status](CURRENT_STATE.md), [durable plan](NEXT_EXPERIMENT_PLAN.md).

### New L1 and F1 jobs — long-run durability vs pretraining-style corpus (2026-10-09 UTC)

The user requested the next jobs and suggested using some of the data SmolLM2 was originally trained on. These are **separate studies** to prevent confounding horizon and distribution.

**L1** freeze: [preregistration](research_log/l1_6000_step_durability_wikitext_seed1729_prereg_2026-10-09.md), [script](l1_6000step_wikitext_durability_seed1729.py) pinned `841e0ab948991a24b7397b34d90d28c38f749879`; GPU [6ac86e1f095c578089301e53](https://huggingface.co/jobs/codeflash85/6ac86e1f095c578089301e53) RUNNING, A10G-small max2h. Paired D direct Q3 and S300 staged, each 6000 scheduled steps (first1200 source/sequence/schedule of historic v10, then4800 new WikiText2 train chunks at LR1e-4). New WikiText validation 16384-token evaluation at 1200/2400/3600/4800/6000; historic WikiText test at1200 for reproduction only.

**F1** freeze: [preregistration](research_log/f1_fineweb_edu_qat_seed1729_prereg_2026-10-09.md), [script](f1_fineweb_edu_qat_seed1729.py) pinned `64dcd7f20240b4c62e6ecac8df70a1336a57bb14`; dataset access/static QA and full document partition CPU preflight `6ac86e0d095c578089301e45` all **PASSED**. F1 GPU [6ac86eb0fee2c900701738dd](https://huggingface.co/jobs/codeflash85/6ac86eb0fee2c900701738dd) submitted with initial status SCHEDULING, A10G-small max90m, results pending. Paired1200 Q3 vs Q9→Q3 on public FineWeb-Edu sample of SmolLM2's pretraining-type corpus; exact model/dataset revisions and SHA256 document partition fixed. Separate FineWeb source documents train/dev/test and WikiText validation; pretrained source-model exposure of FineWeb documents unknown.

**Handoff:** [NEXT_JOBS_PLAN.md](NEXT_JOBS_PLAN.md). **Status:** no new results at submission; archive raw outcome including failures. No other seeds/retries authorized without new decision.

### F1 RESULT — FineWeb-Edu heldout Q3 vs Q9→Q3, seed1729 (COMPLETED, 2026-10-09 UTC)

**Full scientific results:** [F1 raw FINAL_JSON](results/run_f1_fineweb_edu_direct_vs_staged_seed1729_2026-10-09.json). **Interpretation:** [F1 summary](results/run_f1_fineweb_edu_direct_vs_staged_seed1729_summary.md). **Frozen prereg:** [F1 protocol](research_log/f1_fineweb_edu_qat_seed1729_prereg_2026-10-09.md), source SHA `64dcd7f20240b4c62e6ecac8df70a1336a57bb14`, [HF job `6ac86eb0fee2c900701738dd`](https://huggingface.co/jobs/codeflash85/6ac86eb0fee2c900701738dd), completed 2026-10-09 04:48:30 UTC, A10G-small.

**Question:** Does the original v10 paired 1200-step direct-Q3 vs Q9(300)→Q3(900) advantage survive when QAT uses **public FineWeb-Edu sample-10BT**, a source distribution from SmolLM2's pretraining mixture, rather than WikiText2? The entire original model pretraining mixture and instruction-tuning data were not reconstructed.

**Controlled setup:** Same SmolLM2-360M-Instruct BF16-rounded source, frozen non-quantized modules, FP32 master + rowwise learned scales, 35% CE /65% teacher KL, AdamW, FP16 autocast, v10 matched LR, common deterministic seed1729 shuffled 1200 chunks. D direct Q3 continuous, S Q9 first300 with original-scale Q3 projection + fresh Adam for remaining900. Source/dataset pinned SHAs; both arms equal **1200 scheduled** opportunities, AMP effective steps D **1194** (6 skips), S **1192** (8 skips).

| Metric | Direct Q3 | Q9(300)→Q3(900) | D−S staged advantage |
|---|---:|---:|---:|
| FineWeb-Edu document-heldout NLL ↓ | 5.766660 | **4.996255** | **+0.770405** |
| FineWeb-Edu document-heldout PPL ↓ | 319.469 | **147.858** | 53.72% relative reduction |
| WikiText2 validation NLL ↓ | 6.558239 | **5.695225** | **+0.863013** |
| WikiText2 validation PPL ↓ | 705.029 | **297.444** | 57.81% relative reduction |
| FineWeb heldout top1 teacher agreement ↑ | 28.11% | **36.02%** | +7.91 percentage points |

**All 5/5 frozen checks passed** (`valid_for_science=true`): both1200 opportunities, QAT train/dev/eval FineWeb document ID sets disjoint, exact chunk counts, finite final losses, identical order. The FineWeb sample involved **180 train docs / 1200 chunks**, **3 dev docs /24 chunks**, and **21 document-heldout docs /128 chunks** (16384 tokens), from first 340 scanned documents. The heldout WikiText validation also comprised 128 chunks (16384 tokens). Within-document chunks are correlated; don't treat token count as independent document sample size. The QAT document split is disjoint but the pretrained language model might have seen some of these FineWeb source documents in pretraining; do **not** claim a novel unseen test distribution.

**Scientific outcome:** One positive *paired* adaptation result on pretraining-style QAT data, including secondary WikiText validation improvement. It is **not** cross-family confirmation or evidence for full 4T-token mixture behavior. Strong effect size, but **one familiar seed/order**, very small dev/eval documents, no formal uncertainty. S Q9 native train-dev loss at step300 5.428179 vs immediate original-Q3-scale projection 6.942240 (shock +1.514062), then Q3 recovery, consistent with earlier trainability observations but not mechanistic proof.

**Independent still-running job:** [L1 6000-step WikiText2 durability](https://huggingface.co/jobs/codeflash85/6ac86e1f095c578089301e53) RUNNING at last check. Do not conflate F1 result with L1 longer-horizon evidence, do not stop or duplicate the separate L1 job. No new GPU jobs launched in F1 archival. The next action is to inspect L1 and archive its own raw results at completion.

### F2/F3 FineWeb-Edu exact replication at two further seeds (SUBMITTED 2026-10-09 UTC)

User approved the two additional seeds **271828** and **424242** after F1's positive seed1729 result. [F2/F3 frozen preregistration](research_log/f2_f3_fineweb_edu_seed271828_424242_replication_prereg_2026-10-09.md); base F1 source code and data SHA unchanged, frozen matched arms D direct Q3 all1200 versus S Q9(300)→Q3(900), 1200 shared scheduled opportunities, same hashed FineWeb train/dev/eval source-doc partition and same heldout FineWeb/WikiText validation chunks. ONLY seed/permutation-check/provenance change; no retuning or fresh evaluation docs.

- **F2 seed271828:** [F2 seed271828](https://huggingface.co/jobs/codeflash85/6ac87415fee2c90070173baf), source SHA `64d171a225ba2781c83148a4235272c91939f9c9`, A10G-small, max90m. Last status RUNNING.
- **F3 seed424242:** [F3 seed424242](https://huggingface.co/jobs/codeflash85/6ac87417095c5780893020c2), source SHA `a20c4cc66ae16026bb90969d1160294b3f5826e9`, A10G-small, max90m. Last status RUNNING.
- **QA:** HF CPU job `6ac87407fee2c90070173b9d` passed, `F2_F3_PREFLIGHT_OK`; expected seed-specific `torch.randperm(1200)` first16 entries frozen in code; normalized script diff verifies only allowed changes.
- **Outcomes:** NOT YET KNOWN. Future: archive both raw `FINAL_JSON` and full three-seed paired gap on the **shared 21 FineWeb heldout docs**, plus wiki validation. Report all negatives/technical mismatches. Do not claim new documents or independent dataset replication.
- **Independent job:** L1 WikiText6k [6ac86e1f095c578089301e53](https://huggingface.co/jobs/codeflash85/6ac86e1f095c578089301e53) remains RUNNING; do not interrupt/duplicate.

### Post-study follow-up: F2/F3 FineWeb 3-order confirmation and L1 WikiText6000 durability (COMPLETED, 2026-10-09 UTC)

The original source-pinned F2 and F3 jobs finished on Hugging Face, both valid for science and with all five per-job checks passed. With F1 seed1729, **three of three** paired FineWeb training orders favor Q9→Q3 at the same 1200-step budget. FineWeb heldout loss advantages `D-S`: **seed1729 +0.770405**, **seed271828 +0.778936**, **seed424242 +0.742353**, mean **+0.763898 nats/token**, sample SD **0.019140**. Secondary WikiText validation gaps: **+0.863013, +0.845837, +0.696902**, mean **+0.801917**. Three-seed [aggregate](results/run_f1_f3_fineweb_edu_three_seed_aggregate_summary.md), [F2 raw](results/run_f2_fineweb_edu_direct_vs_staged_seed271828_2026-10-09.json) (HF job `6ac87415fee2c90070173baf`, script SHA `64d171a225ba2781c83148a4235272c91939f9c9`), [F3 raw](results/run_f3_fineweb_edu_direct_vs_staged_seed424242_2026-10-09.json) (HF job `6ac87417095c5780893020c2`, script SHA `a20c4cc66ae16026bb90969d1160294b3f5826e9`). All orders tested against the **same FineWeb 21 document-heldout sample**, not new datasets, and old source-model overlap is unknown.

L1 WikiText6000 HF job `6ac86e1f095c578089301e53` (pinned SHA `841e0ab948991a24b7397b34d90d28c38f749879`) completed with five of five protocol checks passed, reproducing both previously observed step1200 old test losses exactly. Staged minus direct advantage in WikiText validation remained: direct − staged **+0.716087 @1200, +0.714249 @2400, +0.684715 @3600, +0.696733 @4800, +0.616040 @6000**. One seed and a specified constant 1e-4 LR after1200; neither consistent convergence nor a production model established. [L1 summary](results/run_l1_6000step_wikitext_durability_seed1729_summary.md) · [raw](results/run_l1_6000step_wikitext_durability_seed1729_2026-10-09.json).

A third-party code review raised **six genuine unresolved controls**: learnable-scale-gradient confound in depth sweeps, nine-state range mismatch, direct-Q3 sham optimizer/scale switch, unmatched boundary margins in random position controls, different learned/fixed scales in M1 code-adoption measurements, and absent document-linked diagnostics/checkpoints. Independently verified and recorded with falsifiable G1/S1/R1/P1/M2 protocols in [technical review](research_log/2026-10-09_external_technical_review_scale_range_sham_controls.md). All **proposed but unrun**; next recommended decision is frozen-scale depth then sham-switch or range control, subject to explicit GPU budget. No additional experiment launched during archival.

### G1 and S1 new preregistered mechanism controls — SUBMITTED 2026-10-09 EDT

## NOW ACTIVE — G1 and S1 mechanism-control jobs (2026-10-09 EDT)

**Two bounded GPU jobs submitted**, with preregistrations committed before launch; **results not available at submission**:

- **G1 depth × scale trainability**: [HF job 6ac96a7efee2c9007017ea64](https://huggingface.co/jobs/codeflash85/6ac96a7efee2c9007017ea64), one A10G-small max **2h**, pinned Git SHA `059a0bb91aa55fea99ed56b1d0d980eb21fd31ae`, [G1 source](g1_depth_by_scale_freeze_seed1729.py). Seed1729 WikiText familiar v13A preparation, four Q3 continuation arms d003/d050 × learned/frozen Q3 row scales, common global 900-step continuation. [G1 frozen prereg](research_log/g1_frozen_vs_learned_q3_scale_depth_prereg_2026-10-09.md).
- **S1 FineWeb Q3 sham-switch**: [HF job 6ac96a82fee2c9007017ea6b](https://huggingface.co/jobs/codeflash85/6ac96a82fee2c9007017ea6b), one A10G-small max **90m**, pinned Git SHA `4cdab8ee73ddd804e96fac43992f91ab9a54c1b9`, [S1 source](s1_fineweb_q3_sham_reset_seed1729.py). Seed1729 one original direct-Q3 continuous arm and one Q3 step300 sham Adam/GradScaler/original-scale reset, with unchanged FineWeb QAT document partition/1200-step LR and historical Q9 comparison. [S1 frozen prereg](research_log/s1_fineweb_direct_q3_sham_reset_seed1729_prereg_2026-10-09.md).
- **Static preflight:** HF CPU job `6ac96a6f095c57808930b267` emitted `G1_S1_STATIC_PREFLIGHT_OK`; script Python AST and invariants checked.

**Full permanent handoff:** [G1_S1_ACTIVE_JOB_PLAN.md](G1_S1_ACTIVE_JOB_PLAN.md). **Do not relaunch these jobs, change scripts, launch additional seeds, or start R1 range-matched Q9 until new user authorization.** Inspect job statuses and exact `FINAL_JSON`, archive raw findings even nulls/technical failures, then update research reports. F1/F2/F3 and L1 historical findings stay completed and preserved.

---

## G1/S1 COMPLETE — controls on scale gradients and sham resets (2026-10-09 UTC)

G1 HF [6ac96a7efee2c9007017ea64](https://huggingface.co/jobs/codeflash85/6ac96a7efee2c9007017ea64) COMPLETED, 6/6 checks, pinned source SHA `059a0bb91aa55fea99ed56b1d0d980eb21fd31ae`. Final WikiText heldout NLL: d0.03 learned **5.494818**, d0.50 learned **4.884970**, shallow-minus-deep gain **+0.609848**; d0.03 frozen **5.516548**, d0.50 frozen **4.896080**, gain **+0.620468**; frozen-minus-learned depth interaction **+0.010620**. Identical initial Q3 codes/scores; raw scales frozen exactly and excluded from optimizer, all 900 opportunities (898 effective/2 AMP skips). **Depth effect persists without row-scale learning**, supporting interior assignment placement and survival rather than requiring scale-gradient learning. Does not isolate Q9's wider range or fully prove mediation.

S1 HF [6ac96a82fee2c9007017ea6b](https://huggingface.co/jobs/codeflash85/6ac96a82fee2c9007017ea6b) COMPLETED, 9/9 checks, pinned SHA `4cdab8ee73ddd804e96fac43992f91ab9a54c1b9`. On matched seed1729 FineWeb, D Q3 continuous loss **5.766660**, D Q3 with original-scale+Adam/GradScaler sham reset@300 **5.789199** (sham worse **0.022539**), historical same-seed Q9→Q3 **4.996255** (**0.792944** better than sham). WikiText validation likewise sham 6.560840, continuous6.558239, staged5.695225. Anchor D reproduces original F1 exactly. **Reset alone is insufficient** to explain staged FineWeb benefit. One seed, same 21-doc heldout sample.

**Raw evidence:** [G1 JSON](results/run_g1_depth_by_scale_freeze_seed1729_2026-10-09.json) · [G1 summary](results/run_g1_depth_by_scale_freeze_seed1729_summary.md) · [S1 JSON](results/run_s1_fineweb_q3_sham_reset_seed1729_2026-10-09.json) · [S1 summary](results/run_s1_fineweb_q3_sham_reset_seed1729_summary.md). Original frozen protocol/job handoff [G1_S1_ACTIVE_JOB_PLAN.md](G1_S1_ACTIVE_JOB_PLAN.md).

**Next:** strongest unrun mechanistic confound is Q9's broader output range (range-matched nine levels versus original Q9, correct STE/clipping); boundary-margin matched controls and truly new evaluation docs follow. No R1 or other paid follow-up job automatically launched. [CURRENT_STATE.md](CURRENT_STATE.md) is authoritative; old “active/submitted” headings elsewhere are historical.

---

### R1 Q9 range-match control — submitted 2026-10-09 UTC, no results yet

## R1 newly submitted — Q9 range-matched to Q3 (2026-10-09 UTC)

**ONE bounded scientific GPU job submitted:** [HF 6ac97c30095c57808930b904](https://huggingface.co/jobs/codeflash85/6ac97c30095c57808930b904), A10G-small, max2h, initial stage SCHEDULING. **No R1 scientific result available at submission.** Immutable script SHA `f28f1d4814ab0a61f2b529e2add007ce142972d5`; [R1 source](r1_fineweb_q9_range_matched_seed1729.py). [Frozen pre-run R1 prereg](research_log/r1_range_matched_q9_fineweb_seed1729_prereg_2026-10-09.md). [Full active job plan/handoff](R1_ACTIVE_JOB_PLAN.md).

This controls the **representable output range** question from the external review: FineWeb seed1729 1200-step `D` direct Q3, `S_wide` original Q9(300)→Q3(900) with nine values reaching ±α, and `S_range` nine-state Q9(300)→Q3(900) with nine values reaching only ±2α/3, then identical Q3 continuation. Same Q3 source row-scale restoration and AdamW/GradScaler reset at step300 for S_wide and S_range. Same `z=clamp(w/alpha,-0.99,0.99)` and STE gradient path; range-matched clips **integer code after round(6z)** to ±4, not normalized z, so code set is exactly nine. Width matching also moves intermediate state spacing/threshold locations, so **not a perfect pure range ablation**.

**Pre-run CPU dynamic gradient/code QA [6ac97b98095c57808930b8da](https://huggingface.co/jobs/codeflash85/6ac97b98095c57808930b8da)** COMPLETED. It printed `R1_QUANTIZER_TEST_OK` for both wide/range variants and `R1_CPU_PREFLIGHT_OK`: 9 states, max output amplitude, STE master-weight/scale gradients on unclipped inputs, Q3 switch behavior, script AST. The R1 three-arm GPU job has not been completed or interpreted yet. Historical G1/S1 findings remain valid and archived; **do not launch new seeds, repeat R1, or launch other jobs without user approval**.

---

## R1 COMPLETED — range-matched nine-state Q9 nearly erases staging gain (2026-10-10 UTC)

**Valid scientific R1**: [HF job](https://huggingface.co/jobs/codeflash85/6ac97c30095c57808930b904), finished **2026-10-10 00:05:57 UTC**, source SHA `f28f1d4814ab0a61f2b529e2add007ce142972d5`. **12/12 technical/reproduction checks passed**, including *exact* archived FineWeb F1 direct-Q3 and original wide Q9→Q3 heldout NLL reproductions. [R1 full raw JSON](results/run_r1_q9_range_matched_fineweb_seed1729_2026-10-10.json) · [R1 detailed summary](results/run_r1_q9_range_matched_fineweb_seed1729_summary.md) · [original prereg](research_log/r1_range_matched_q9_fineweb_seed1729_prereg_2026-10-09.md) · [R1 handoff](R1_ACTIVE_JOB_PLAN.md).

**R1 question:** is having nine intermediate levels enough, or is the originally wider Q9 representation (max ±α) important compared with Q3's max ±2α/3? A new nine-state range-matched Q9 uses `q=α*clamp(round(6*clip(w/α,-.99,.99)),-4,4)/6`, preserving nine codes, normalized `z` outer clip and surrogate gradient formula, but deliberately compressing output range to ±2α/3. All arms use the same seed1729 FineWeb `sample-10BT`, original model source, 300 Q9 +900 Q3 continuation/switch, and fixed heldout evaluation.

| Endpoint | Direct Q3 | Original wide Q9→Q3 | Range-matched Q9→Q3 |
|---|---:|---:|---:|
| FineWeb doc-heldout NLL ↓ | 5.766660 | **4.996255** | 5.751036 |
| WikiText validation NLL ↓ | 6.558239 | **5.695225** | 6.566215 |

FineWeb wide-Q9 gain `D−wide` **+0.770405 nats/token**; range-matched gain `D−range` **+0.015624** (only **2.03%** of original). Range-matched was 0.0080 nats **worse than direct** on WikiText validation; original wide gain there **+0.863013**.

**Crucial confound still present:** A uniformly spaced nine-state grid constrained to Q3's range also changes quantization cell width, internal thresholds and code saturation. At Q9 preparation300 the proportion of extreme code ±4 was **47.83%** range-matched vs **29.30%** wide; **36.45%** of range-matched would-be raw round(6z) codes exceed ±4 and are clipped. We can state **the wider output range and/or its coupled code geometry is necessary for *this particular range-matched implementation* to retain the effect**, not that range alone caused it or any narrow-range nine-level method must fail. The new S_range already has worse Q9-native train-dev loss, not merely larger switch shock.

**Scope:** single familiar seed1729 and 21 small FineWeb heldout source documents, pretrained-model document exposure unknown, no restored model checkpoint, no production-generation claim. Previous G1 frozen-scale and S1 sham-reset controls passed and support a meaningful optimization-path/assignment-stability story but do not render R1 a pure isolation of range. **No additional GPU job was launched after R1.** [CURRENT_STATE.md](CURRENT_STATE.md) is authoritative; older “R1 submitted” sections are historical.

---

