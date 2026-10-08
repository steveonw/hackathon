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
