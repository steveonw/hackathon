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
