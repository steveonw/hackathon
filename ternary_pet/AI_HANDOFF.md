# AI HANDOFF — Ternary Pet Quantization Research

> **START HERE if you are a new AI taking over this project.**
>
> Repository: `steveonw/hackathon`  
> Project directory: `ternary_pet/`  
> Current state: **v8 completed; v7 trainability effect replicated, simple signed-preload mechanism not supported**  
> Latest mechanism script: `smollm2_v8_signed_preload.py` at `356bff9769c96faa0ca139f9cba088fc1a52c2c8`  
> Latest completed HF job: `6ac5a0b1404719ba37664653`  
> Canonical current summaries: `replications/v7_aggregate_summary.md` and `results/run_v8_summary.md`

## 1. What the user is trying to discover

The research question is whether a pretrained language model can enter a final
ternary weight space more effectively if it first adapts in an intermediate,
less restrictive representation.

The original intuition was "lose precision gradually instead of jumping
straight to ternary." Experiments have narrowed that into a more precise
optimization hypothesis:

> Intermediate-state training may let continuous FP32 master/shadow weights
> reposition themselves before the harsh ternary partition is imposed.

This is **not** currently a claim that staged rounding mechanically preserves
bits, nor a claim of a production-ready ternary model.

## 2. Model and standard setup

Model:

- `HuggingFaceTB/SmolLM2-360M-Instruct`
- ~362M parameters
- Llama-family architecture

Core training setup in the successful experiments:

- source checkpoint BF16-rounded before treatment;
- target linears quantized except `lm_head`;
- non-quantized parameters frozen;
- persistent FP32 master/shadow weights;
- learnable rowwise quantizer scales;
- AdamW, usually LR `1e-4`;
- loss = 35% CE + 65% teacher KL;
- WikiText-2 training/evaluation;
- fixed 8192-token held-out evaluator;
- seed/order 1729 is a key replicated ordering;
- generated text remains poor/degenerate despite metric improvements.

## 3. What has already been learned

### v1 — naive 27 -> 9 -> 3

Failed. Direct and staged models were badly damaged. This did not support the
hypothesis.

Job: `6ac52475404719ba37661c8b`

### v2 — strict nested hierarchy

Important negative result:

- deterministic direct ternary projection and deterministic nested 27 -> 9 -> 3
  projection ended identically;
- therefore staging without learning does **not** preserve extra information.

Hard commits and optimizer resets made this a weak adaptive test.

Job: `6ac527e5404719ba37661dc9`

### v3 — persistent FP32 masters / transition schedules

Persistent continuous masters helped make the experiment meaningful, but
full-precision warmup, soft ramp, and direct ternary were roughly tied once
final ternary training budget was matched.

Confounds later identified:

- ~79% zeros in the ternary representation;
- non-quantized components could train;
- no direct code-movement metric.

Job: `6ac52df9404719ba37661fa1`

### v4 / v4b — first reproducible positive 9 -> 3 result

Protocol:

- direct: 1200 @ Q3
- staged: 300 @ Q9 -> 900 @ Q3
- same total updates
- persistent FP32 masters
- no hard commit
- frozen non-quantized parameters
- code movement logged
- LR `1e-4`

Across three paired training orders, staged Q9 -> Q3 beat direct Q3 on all main
held-out metrics.

Aggregate v4/v4b:

- mean held-out loss improvement: **0.765 nats/token**
- mean PPL reduction: **53.35%**
- mean teacher top-1 gain: **+7.54 percentage points**
- mean KL reduction: **24.47%**

Important replication correction:

Changing only the random seed originally produced identical traces because the
data order was fixed. Those false starts were cancelled and **do not count**.
The valid v4b replications explicitly shuffle the same training chunks.

Key jobs:

- v4: `6ac539b5404719ba376621a2`
- v4b seed/order 1729: `6ac546b5fbc85ba6823b8941`
- v4b seed/order 271828: `6ac546bafbc85ba6823b8946`

See:

- `replications/v4b_aggregate_summary.md`
- `results/run_v4_summary.md`

### v5 — 5x training-volume test

Protocol:

- direct: 6000 @ Q3
- staged: 1500 @ Q9 -> 4500 @ Q3
- ~768k training-token presentations per treatment

Results:

| Variant | Loss | PPL | Teacher top-1 | KL |
|---|---:|---:|---:|---:|
| BF16 source | 3.6854 | 39.86 | 99.68% | ~0 |
| Direct 6000@3 | 4.4463 | 85.31 | 41.44% | 1.522 |
| Staged 1500@9 -> 4500@3 | **4.2493** | **70.05** | **43.64%** | **1.330** |

The staged advantage survived but shrank:

- loss advantage: **0.197 nats/token**
- PPL reduction: **17.88%**

This is evidence that a substantial part of the v4/v4b effect is an
**early-optimization/head-start effect**. It does **not** prove a permanently
better asymptotic ternary basin.

Job: `6ac56d99fbc85ba6823ba03f`

See:

- `results/run_v5_summary.md`
- `results/run_v5_2026-10-06.json`

### v6 — causal transition ablation

Purpose: determine whether the v4-scale staged advantage is stored in:

- prepared FP32 master weights;
- learned quantizer scales;
- Adam optimizer state.

One shared 300-step Q9 checkpoint was branched into 900-step Q3 continuations.

Results:

| Condition | Loss | PPL |
|---|---:|---:|
| Direct 1200@3 | 5.8747 | 355.90 |
| Carry masters + scales + full Adam | 5.2059 | 182.34 |
| Reset Adam | 5.2031 | 181.83 |
| Prepared masters only; original scales + fresh Adam | 5.1990 | 181.09 |
| Prepared masters + weight Adam; original scales | 5.1799 | 177.67 |

Critical result:

> Resetting Adam and restoring the original scales does **not** remove the
> staged advantage.

The four prepared-master branches span only **0.026 nats/token**, versus roughly
0.67-0.69 nats/token over direct.

Therefore, under this setup, the early staged advantage is carried primarily by
the **adapted FP32 master weights**.

#### v6 clean transition diagnostic

On the same fixed diagnostic data, without an optimizer update between
evaluations:

- initial Q3 loss: **15.4784**
- after 300 Q9 updates, evaluated as Q9: **4.9320**
- same prepared master weights immediately projected to Q3: **9.1771**

So the clean Q9 -> Q3 switch costs:

- **+4.2451 nats/token**

but the Q9-prepared masters projected to Q3 are 6.3013 nats/token better than
the **untrained** initial Q3 projection.

#### v6 future Q3 boundary crossings

After Q9 preparation:

- future Q3 assignments changed with learned scales: **0.6769%**
- future Q3 assignments changed with original scales fixed: **0.6755%**

This is strong evidence that the FP32 master weights themselves move across
future ternary decision boundaries.

Important refinement after external review:

> v6 did **not** compare Q9-prepared Q3@300 against direct Q3@300 on the same
> diagnostic data.

Therefore v6 does **not yet distinguish**:

1. **better entry state** — Q9@300 is already a better ternary checkpoint than
   direct Q3@300; from
2. **better trainability** — direct Q3@300 may be as good or better, while Q9
   leaves continuous masters in a geometry that trains better over the next 900
   ternary updates.

Also: direct Q3 had moved about the same aggregate fraction of codes by step
300 (~0.69%), often in similar broad layers. Do not claim that raw code count or
broad layer identity explains the effect.

Job: `6ac5816ffbc85ba6823ba8ec`

See:

- `results/run_v6_summary.md`
- `results/run_v6_2026-10-06.json`

## 4. CURRENT EXPERIMENT — v7

### Purpose

v7 is specifically designed to separate:

- **better ternary entry state**;
- **better subsequent trainability / within-bin geometry**;
- a generic benefit from any less-constrained warmup.

### Pinned code

`ternary_pet/smollm2_v7_equal_compute_geometry.py`

Pinned commit:

`3aa494a4c8418052ed9e13e0de4f97c692a29fc7`

### Hugging Face job

`6ac58e76fbc85ba6823bad78`

The job completed successfully on **A10G-small**.

### v7 arms

All arms consume the same first 300 shuffled chunks:

1. `q3_300`: direct Q3 for 300 updates;
2. `q9_300`: Q9 for 300 updates;
3. `fp32_300`: unquantized-weight warmup for 300 updates.

At step 300, all three master states are projected through the **same original
Q3 scales** and scored on the **same fixed 24-chunk diagnostic set**.

Then all three get the same 900-step ternary continuation:

- same original Q3 scales;
- fresh Adam;
- same LR `1e-4`;
- same remaining 900 chunks.

Therefore the only state carried from preparation into continuation is the
prepared FP32 master weights.

### v7 diagnostics

v7 records:

- native step-300 quality in each preparation space;
- fixed-scale Q3 quality for all three states at equal compute;
- Q3 code changes versus initial;
- exact pairwise overlap/Jaccard of changed Q3 codes;
- pairwise Q3 Hamming distance;
- distance of every master to nearest future Q3 threshold
  `w/alpha = +/-1/3`;
- final held-out metrics;
- fixed free-running generation prompts.

### How to interpret v7

If:

`Q9-prepared fixed-Q3@300 < direct-Q3@300 loss`

then Q9 creates a **better ternary entry state**.

If direct Q3@300 is equal/better, but Q9 finishes the shared 900-step
continuation better, then the likely mechanism is **better trainability or
within-bin master geometry**.

The FP32 arm answers whether Q9 itself is special:

- if FP32 warmup matches Q9, a less-constrained warmup may be sufficient;
- if Q9 beats FP32, the intermediate discrete grid may provide useful structure.

Threshold-margin diagnostics may reveal an effect that raw code Hamming misses:
two master states can have the same Q3 code but sit very differently relative to
their next decision boundary.

## 5. What the next AI should do when v7 finishes

1. Inspect HF job `6ac58e76fbc85ba6823bad78`.
2. Fetch logs and locate text between:
   - `FINAL_JSON_BEGIN`
   - `FINAL_JSON_END`
3. Extract at minimum:
   - `initial_q3_reference`
   - each arm's `prep.native_diag`
   - each arm's `fixed_q3_projection.diag`
   - each arm's `fixed_q3_projection.margin`
   - each arm's `fixed_q3_projection.vs_initial_codes`
   - `step300_code_comparisons`
   - each arm's final loss / PPL / teacher top-1 / KL
   - generations
4. Make the step-300 equal-compute comparison **before** looking at final
   1200-step outcomes.
5. Do not change or rerun the protocol merely because one arm loses.
6. Save:
   - raw JSON under `ternary_pet/results/`
   - a human-readable v7 summary
   - an update to `EXPERIMENT.md`
   - an update to this handoff sheet
   - a concise README headline if the result materially changes the mechanism.
7. If the job failed for an infrastructure/code reason, fix it, pin a new Git
   commit, rerun the **same scientific protocol**, and retain the failure for
   auditability.

## 6. Claims that are currently safe

Safe:

> Under this SmolLM2/WikiText QAT setup, a 9-state preparation phase produces a
> large finite-budget ternary recovery advantage at 1200 updates.

Safe:

> The early advantage is carried mainly by the adapted FP32 master weights,
> rather than by inherited Adam moments or learned-scale state.

Safe:

> Q9 preparation moves some continuous masters across future Q3 decision
> boundaries even when Q3 scales are held fixed.

Safe:

> At 5x training volume, the staged advantage remains but becomes much smaller,
> consistent with a large head-start component.

## 7. Claims that are NOT established

Do **not** claim:

- Q9 is globally optimal;
- 9 states are uniquely special;
- staged QAT reaches a permanently better asymptotic basin;
- gradual rounding mechanically preserves information;
- the effect generalizes beyond SmolLM2/WikiText;
- the current ternary model is usable;
- broad layer-level flip concentration proves causal importance;
- `flip_since_last` counts every discrete crossing.

Important measurement caveat:

`flip_since_last` is a snapshot Hamming difference. A code may flip and return
between snapshots without being counted.

## 8. Known technical limitations / reproducibility caveats

- Dependency declarations use minimum versions such as
  `torch>=2.4`, not exact package pins.
- Model and dataset revisions are not exact-pinned.
- Exact experiment code **is** pinned by Git commit.
- A10G and T4 results need not be bit-for-bit identical.
- WikiText teacher-forced metrics have improved much more than free-running
  generation quality.
- The direct ternary baseline has only a limited LR diagnostic sweep; it has not
  yet received a full optimizer/scheduler search.
- The Q9 grid itself may be suboptimally calibrated/clipped.
- Learned scales changed very little in v6, so "scales not necessary here"
  should not be generalized to "scales never matter."

## 9. Research trajectory after v7

Do not blindly launch another large run.

Depending on v7:

- if Q9 already wins at fixed-Q3@300, study **which exact ternary decisions**
  differ from direct and whether a better-calibrated intermediate grid improves
  the entry state;
- if Q9 only wins after continuation, focus on **within-bin geometry /
  threshold-margin / gradient conditioning**;
- if FP32 matches Q9, test whether the phenomenon is generic continuation/warmup
  rather than Q9-specific;
- if Q9 beats both Q3 and FP32, compare other intermediate codebooks
  (5/7/9/15 states) under equal compute;
- regardless of outcome, a stronger tuned direct-Q3 baseline is eventually
  required before a broad research claim;
- only scale to larger models after the mechanism survives at least one other
  architecture/data setting.

## 10. Repository map

Start with:

- `AI_HANDOFF.md` — this file
- `README.md` — current headline
- `EXPERIMENT.md` — chronological protocol and outcomes
- `SHAREABLE_RESEARCH_REPORT.md` — external-review narrative

Scripts:

- `smollm2_staircase.py` — v1
- `smollm2_nested_v2.py` — v2
- `smollm2_transition_v3.py` — v3
- `smollm2_v4_fliprate_9to3.py` — v4
- `smollm2_v5_5x.py` — v5
- `smollm2_v6_transition_ablation.py` — v6
- `smollm2_v7_equal_compute_geometry.py` — v7
- `replications/` — v4b replication scripts and aggregate

Results:

- `results/run_v4_summary.md`
- `replications/v4b_aggregate_summary.md`
- `results/run_v5_summary.md`
- `results/run_v6_summary.md`
- machine-readable JSON files in `results/`

## 11. Working style / user intent

The user wants real empirical work, not speculative storytelling.

Preferred workflow:

`idea -> GitHub pinned code -> HF GPU job -> raw metrics/logs -> analysis -> GitHub writeback`

Treat GitHub as the lab notebook/source of truth.

Negative results and failed jobs should remain visible and auditable.

The user has authorized bounded paid Hugging Face compute, but compute should be
used prudently. Do not spend on a bigger run when a cheaper causal ablation can
answer the question.

External AI reviews are advisory. Adopt a suggestion only when it improves the
scientific test; do not let outside reviewers silently redefine the experiment.

## 12. One-sentence state of the project

> Across three training orders, Q9 preparation creates a worse immediate Q3
> checkpoint but a substantially more trainable FP32-master state for the same
> later Q3 continuation; v8 did not support the tested simple signed-threshold
> preload explanation, so the remaining mechanism is a deeper
> trainability/weight-selection geometry effect.


## v7 OUTCOME — read this before planning the next experiment

v7 resolves the main ambiguity left by v6.

Equal-compute fixed-Q3 diagnostic at step 300:

- direct Q3: **6.5589**
- Q9-prepared masters: **9.2264**
- FP32-prepared masters: **14.7346**

Final after identical fresh-Adam 900-step Q3 continuation:

- direct Q3 -> Q3: **5.8724 loss / 355.09 PPL**
- Q9 -> Q3: **5.1938 / 180.15**
- FP32 -> Q3: **6.0311 / 416.19**

Therefore:

1. Q9 does **not** win by producing a better immediate Q3 checkpoint.
2. Q9 does produce a master state that is dramatically more trainable during
   later Q3 optimization.
3. Generic FP32 warmup is not enough; the intermediate discrete constraint
   matters in this setup.
4. Direct and Q9 change similar counts of future ternary codes (~0.69% vs
   ~0.68%) but only **19.64% Jaccard** of changed positions overlap.
5. Simple global threshold-margin distributions are effectively identical.

The current mechanism hypothesis is now **Q9-specific gradient/weight-selection
geometry**, not entry quality, Adam carryover, scale carryover, or generic
high-precision warmup.

See `results/run_v7_summary.md` and `results/run_v7_2026-10-07.json`.

### Next decision

Do not automatically scale up. The highest-value next work is to determine why
the Q9-selected subset is more useful, or to compare intermediate state counts
(5/7/9/15) using the same equal-compute v7 framework. A stronger tuned direct-Q3
baseline remains necessary before broad claims.


## v7 REPLICATIONS — COMPLETED

The two confirmatory jobs completed successfully:

- seed 271828 — `6ac59807fbc85ba6823bb060`
- seed 424242 — `6ac59809fbc85ba6823bb063`

Together with seed 1729, the v7 trainability effect now holds in **3/3 training
orders**.

Aggregate:

- Q9 is a worse fixed-Q3 checkpoint at step 300 in all 3;
- Q9 finishes better after the identical 900-step continuation in all 3;
- mean final loss gain: **0.6877 nats/token**;
- mean PPL reduction: **49.69%**;
- mean top-1 gain: **+6.86 pp**;
- mean Q3-vs-Q9 changed-set Jaccard: **19.66%**.

FP32 warmup does not reproduce Q9. Do **not** say FP32 is always worse than
direct: seed 271828 gives FP32 a small 0.0292-nat improvement over direct.

Canonical aggregate:
`replications/v7_aggregate_summary.md`

Raw replications:

- `results/run_v7_seed271828_2026-10-07.json`
- `results/run_v7_seed424242_2026-10-07.json`

### Current next-question priority

The replication question is now substantially answered. The strongest next
mechanistic test is a signed **pre-loading diagnostic**: among weights that cross
a Q3 boundary during the 900-step continuation, measure whether Q9 preparation
had already moved their FP32 masters toward that eventual threshold more often
or more strongly than direct Q3 preparation.

A level-count sweep (5/7/9/15 -> 3) is also high value after or alongside that
diagnostic. A stronger tuned direct-Q3 optimizer/scheduler baseline remains
required before broad claims.

## v8 — COMPLETED signed pre-loading diagnostic

Job: `6ac5a0b1404719ba37664653`  
Pinned code: `356bff9769c96faa0ca139f9cba088fc1a52c2c8`

v8 tested whether Q9 uniquely moves later-flipping weights in their eventual
ternary-transition direction during preparation.

**Result: negative for that specific mechanism.**

Four-way matched comparison:

- on Q9's later-flip set: Q9 = **0.002328**, Q3 = **0.000873**,
  Q9-Q3 = **+0.001455**;
- on Q3's later-flip set: Q3 = **0.003130**, Q9 = **0.001439**,
  Q3-Q9 = **+0.001691**.

Each arm wins on its own future-flip set by a similar margin. That matches the
preregistered post-selection pattern rather than Q9-specific directional
pre-loading.

Do **not** claim that Q9's advantage is explained by pointing future-flipping
weights toward their next ternary threshold. v8 falsified that simple story.

The broader replicated result from v7 remains:

- Q9 is a worse immediate Q3 checkpoint after equal 300-step compute;
- Q9 becomes much better after the same 900-step Q3 continuation;
- the effect holds across 3/3 training orders;
- Q3 and Q9 move similar numbers of future ternary codes but mostly different
  specific weights.

Current best description:

> Q9 changes the optimization / weight-selection geometry in a reproducible
> way, but neither global threshold proximity nor simple signed future-threshold
> preloading explains why the selected Q9 master state is more trainable.

Canonical v8 files:

- `results/run_v8_summary.md`
- `results/run_v8_2026-10-07.json`

### Current next-question priority

A causal intervention should now focus on **which Q9-selected weights carry the
benefit**, without assuming signed-preload is the reason they matter. Another
high-value axis is the intermediate level-count sweep (5/7/9/15 -> 3) under the
same equal-compute v7 protocol. A stronger tuned direct-Q3 optimizer/scheduler
baseline remains necessary before broad claims.

## v9 — COMPLETED direct-Q3 tuning

Job: `6ac5ad77fbc85ba6823bb71c`  
Pinned code: `5c088e58539b2dede93df57ac3f72dbe0480a028`

v9 confirmed the direct baseline was undertuned.

Best full direct schedule on seed 1729:

- 100-step warmup;
- peak LR 1e-3;
- cosine decay to 1e-4 by step 1200;
- held-out loss **5.5957**;
- PPL **269.27**;
- top-1 **29.87%**;
- KL **2.6640**.

Historical constant-1e-4 direct was 5.8747 loss / 355.90 PPL.

Historical Q9 -> Q3 remains better at 5.1938 / 180.15.

Thus tuned direct closes about **41%** of the old loss gap but leaves a
**0.402-nat/token** Q9 advantage on seed 1729.

Do **not** keep quoting the old ~0.68-nat seed-1729 effect as if the direct
baseline were well tuned. The honest current number for this seed is ~0.40 nats
against the better direct schedule.

### Next decision

Per the preregistered rule, run the tuned direct schedule on training orders
271828 and 424242. Only after that should it become the canonical direct
baseline and should the residual Q9 effect be aggregated.

If the residual Q9 advantage survives both orders, proceed to the four-arm
Q3-vs-Q9 hybrid-master factorial intervention.

Canonical v9 files:

- `results/run_v9_summary.md`
- `results/run_v9_2026-10-07.json`

## v9 tuned-direct replication pair — COMPLETED

Valid confirmatory jobs:

- seed 271828: `6ac5b913404719ba37664b21`
- seed 424242: `6ac5b915fbc85ba6823bba5b`

Corrected scripts:
`3e74ccf5c448fc994d005a7baf94529b92e9996a`

Q9 remains better than the tuned direct-Q3 schedule in **3/3 orders**.

| Seed | Tuned direct loss | Q9 loss | Residual Q9 gain |
|---:|---:|---:|---:|
| 1729 | 5.5957 | 5.1938 | 0.4020 |
| 271828 | 5.6228 | 5.2417 | 0.3810 |
| 424242 | 5.6067 | 5.3008 | 0.3059 |

Canonical aggregate:

- mean residual Q9 loss advantage: **0.3630 nats/token**;
- mean PPL reduction: **30.38%**;
- mean top-1 gain: **+2.52 pp**;
- mean KL advantage: **0.3871**;
- tuned direct closes **47.27%** of the old gap on average.

Do not quote the older ~0.688-nat mean v7 gap as the current baseline
comparison. The canonical small-budget effect is now ~**0.363 nats/token**
against tuned direct.

The initial failed/cancelled replication jobs remain technical audit records and
do not count as experimental runs.

Canonical file:
`replications/v9_tuned_direct_aggregate_summary.md`

### Current next-question priority

Before the hybrid-master intervention, close the remaining schedule-fairness
loophole with a **schedule-matched Q9 control**.

The current 0.363-nat three-order comparison is asymmetric in tuning effort:
direct Q3 uses the v9-selected warmup+cosine schedule, while Q9 -> Q3 is still
the historical constant-1e-4 run. Therefore **0.363 nats is not a proven lower
bound or final intrinsic method advantage**.

First apply the exact direct-selected global LR curve to Q9 -> Q3:

- steps 1-100: linear warmup to 1e-3;
- steps 101-1200: cosine decay to 1e-4;
- Q9 for global steps 1-300;
- at the Q9 -> Q3 transition restore original Q3 scales and use fresh Adam as
  in v7, but continue the LR curve from global step 301;
- no additional schedule tuning and no test-set selection.

Run that schedule-matched Q9 protocol on the same three training orders and
compare it with tuned direct and historical Q9. This is an equal-schedule
control, **not** a full equal-search-budget Q9 optimization.

If Q9 still wins clearly, then proceed to the four-arm Q3-vs-Q9 hybrid-master
causal intervention using the schedule-matched preparation/continuation recipe.
Use the equal-compute step-300 direct-Q3 and Q9-prepared master states to
partition the actual prepared-state difference into:

- positions where their projected Q3 codes differ;
- positions where projected Q3 codes agree but continuous FP32 masters differ.

Construct the 2x2 hybrids so paired arms have exactly identical ternary forward
weights at continuation start, assert zero Hamming distance within those pairs,
then run the same Q3 continuation. This tests whether the residual Q9 advantage
is carried mainly by code-disagreement positions, hidden same-code master
geometry, or their interaction.

## v10 — COMPLETED schedule-matched Q9 control

Pinned code:
`4050f42cf226a300082178b2fda475ebcf31664e`

Jobs:

- 1729: `6ac5c252fbc85ba6823bbd6c`
- 271828: `6ac5c254fbc85ba6823bbd6e`
- 424242: `6ac5c256404719ba37664cf1`

The v9-selected global LR schedule was applied unchanged to Q9 -> Q3:

- 100-step warmup to 1e-3;
- cosine decay to 1e-4 through step 1200;
- Q9 steps 1-300;
- original Q3 scales + fresh Adam at transition;
- Q3 steps 301-1200;
- LR curve continues rather than restarting.

Result: matched-schedule Q9 beats tuned direct in **3/3 orders**.

| Seed | Tuned direct | Matched Q9 | Q9 gain |
|---:|---:|---:|---:|
| 1729 | 5.5957 | 4.9010 | 0.6947 |
| 271828 | 5.6228 | 4.9510 | 0.6718 |
| 424242 | 5.6067 | 4.9563 | 0.6505 |

Aggregate:

- mean loss advantage: **0.6723 nats/token**;
- mean PPL reduction: **48.94%**;
- mean top-1 gain: **+6.82 pp**;
- mean KL advantage: **0.6817**.

The step-300 diagnostic still points the opposite way: matched Q9 is **0.5120
nats/token worse** than tuned direct as an immediate fixed-Q3 checkpoint after
equal 300-step compute.

Therefore the trainability effect survives a much stronger and
schedule-symmetric regime.

Do not call 0.6723 a best-tuned-vs-best-tuned intrinsic advantage. Q9 received
the direct-selected schedule, not an independent equal-budget search.

Important geometry update: under this stronger schedule, step-300 projected-Q3
code displacement is ~4.41% for direct and ~4.88% for Q9, much larger than the
old ~0.68% v7 regime. Do not reuse the old v7 Jaccard/code-movement
interpretation as if it automatically applies here.

Canonical aggregate:
`replications/v10_schedule_matched_q9_aggregate_summary.md`

### Current next-question priority

Proceed to the four-arm hybrid-master factorial in the **matched-schedule
regime**.

Rebuild both step-300 master states inside the same run:

- D = direct-Q3 masters after 300 global-schedule updates;
- S = Q9 masters after 300 global-schedule updates.

Project both through the same original Q3 scales and define mask M where their
Q3 codes differ.

Construct:

- 00: D everywhere;
- 10: S on M, D elsewhere;
- 01: D on M, S elsewhere;
- 11: S everywhere.

Assert before continuation:

- Q3(00) == Q3(01) exactly;
- Q3(10) == Q3(11) exactly;
- paired diagnostic losses match to numerical tolerance.

Then give **all four arms** original Q3 scales, fresh Adam, and the same global
LR curve starting at step 301 for 900 Q3 updates.

First verify that endpoints 00 and 11 reproduce a material direct-vs-Q9
trainability gap under this common fresh-Adam continuation. Then interpret 10
and 01 to localize the causal contribution to code-disagreement positions,
same-code hidden master geometry, or their interaction.

## v11 — COMPLETED matched-schedule hybrid-master factorial

Job: `6ac5c7befbc85ba6823bbef0`  
Pinned code: `f342fd9c706f2fe21aa00adabe6611b7f835f570`  
Seed/order: 1729.

All pair assertions passed exactly before continuation:

- Q3(00) == Q3(01), zero Hamming;
- Q3(10) == Q3(11), zero Hamming;
- paired validation diagnostics exactly match.

D-vs-S step-300 projected-Q3 disagreement mask:

- **6.4581%** of quantized weights;
- 20,315,352 / 314,572,800 positions.

Final losses:

| Arm | Meaning | Loss |
|---|---|---:|
| 00 | D everywhere | 5.6136 |
| 10 | S on disagreement mask only | **4.9329** |
| 01 | S on same-code complement only | 5.5020 |
| 11 | S everywhere | **4.9010** |

Full endpoint gain: **0.7126 nats**.

The disagreement-mask transfer alone contributes **0.6807 nats**, recovering
**95.5%** of the full gain. Same-code hidden geometry alone contributes only
**0.1116 nats** on the direct background; once the disagreement mask is already
from Q9, the remaining same-code contribution is only **0.0319 nats**.

Current mechanism interpretation:

> On seed 1729, the Q9 trainability advantage is carried predominantly by the
> continuous Q9-prepared masters at the ~6.46% of positions where Q9 and direct
> produce different projected Q3 codes at step 300.

Do not claim the discrete code values alone are causal. Arm 10 transfers the
full continuous Q9 master values on those positions.

Canonical files:

- `results/run_v11_hybrid_factorial_summary.md`
- `results/run_v11_hybrid_factorial_seed1729_2026-10-07.json`

### Current next-question priority

Replicate the exact v11 factorial on orders 271828 and 424242 before making the
localization result canonical across orders.

If the same pattern replicates, the next causal refinement should separate
**code identity** from **continuous within-bin position on the disagreement
mask** itself, e.g. by constructing masters that preserve the Q9-selected Q3
code while recentering each selected master within its ternary bin.

## v11 hybrid-factorial localization — REPLICATED 3/3

Confirmatory jobs:

- seed 271828: `6ac638c0c656c912b4ffae8a`
- seed 424242: `6ac638c3f0d78b8017af0d5a`

All three orders pass the exact pair-equality assertions.

| Seed | Mask | Full 00->11 gain | M-only gain | Recovery |
|---:|---:|---:|---:|---:|
| 1729 | 6.458% | 0.7126 | 0.6807 | 95.5% |
| 271828 | 6.181% | 0.7014 | 0.6813 | 97.1% |
| 424242 | 6.301% | 0.6672 | 0.6443 | 96.6% |

Aggregate:

- mean mask size: **6.313%**;
- mean full gain: **0.6938 nats**;
- mean M-only gain: **0.6688 nats**;
- mean recovery: **96.4%**;
- mean same-code-only gain on D background: **0.1092 nats**;
- mean residual same-code gain after M is already from Q9: **0.0250 nats**.

The mechanism claim is now replicated:

> The dominant causal carrier is the continuous Q9-prepared master state on the
> ~6.3% of positions where Q9 and direct preparation disagree on the projected
> Q3 code after 300 updates.

This is a robust **where** result. It is not yet a **code-label sufficiency**
result.

Do not say "the discrete ternary assignments alone explain 96%." Arm 10
transfers exact continuous Q9 masters on M.

Canonical aggregate:
`replications/v11_hybrid_factorial_aggregate_summary.md`

### Current next-question priority

Run a code-identity-vs-continuous-position intervention on M.

Recommended first intervention:

- rebuild D and S at step 300 under the matched schedule;
- define M as before;
- keep D outside M;
- on M compare at least:
  1. exact S masters (positive control; reproduces arm 10);
  2. **S-code prototype** masters using the original Q3 scale:
     `w = (2/3) * alpha * c_S`, where `c_S in {-1,0,+1}`;
  3. optionally a minimal-crossing construction starting from D but moving each
     M weight only far enough to enter S's Q3 code region.

All arms should have the same intended S projected Q3 code on M at continuation
start. If the prototype/minimal-crossing arms retain most of the exact-S gain,
discrete code identity is sufficient for much of the effect. If they collapse
toward D despite identical projected codes, the continuous boundary-relative
geometry inside M is essential.

### v12 design refinement — code identity vs boundary-relative geometry on M

External review agrees with the next causal question and adds useful controls.
Use seed/order 1729 first.

Important wording: v11 proves exact equality of the **quantized Q3 forward
weights** within the paired arms (same projected codes under the same original
Q3 scales, with identical diagnostics). Do not describe this as literal
byte-for-byte equality of serialized model files.

Recommended v12 arms, all rebuilt in one run from matched-schedule step-300
states D and S:

1. **D baseline**
   - D masters everywhere.

2. **Exact-S-on-M positive control**
   - S masters on the true disagreement mask M;
   - D elsewhere;
   - should reproduce v11 arm 10.

3. **S-code prototype on M**
   - preserve S's projected Q3 code on every position in M;
   - replace the exact S master with the Q3 reconstruction prototype
     `w = (2/3) * alpha0 * c_S`;
   - D elsewhere.

4. **Minimal-crossing S-code on M**
   - start conceptually from D;
   - move each M weight only far enough to enter S's Q3 code region under the
     original Q3 scale, with a small preregistered normalized epsilon inside the
     target region;
   - D elsewhere.

The exact-S, prototype, and minimal-crossing arms must all assert the **same
projected Q3 codes on every quantized weight** before continuation. Thus they
start from the same ternary forward model and differ only in hidden continuous
master position.

Interpretation:

- prototype ~= exact-S: S code identity is sufficient for most of the benefit;
- minimal-crossing ~= exact-S: merely crossing into the Q9-selected code region
  is sufficient;
- prototype helps but minimal-crossing does not: depth/location inside the Q3
  region matters, but exact Q9 values may not;
- both prototype and minimal-crossing collapse toward D: precise continuous
  Q9-prepared geometry on M is essential.

Because the prototype can be a large displacement, especially for target
`c_S=0`, log the disagreement mask by transition class:

- `c_D -> c_S` transition counts per layer;
- target-code groups `c_S=0` versus `|c_S|=1`;
- normalized distances `w/alpha0` from the relevant Q3 boundary for D and S.

If v12 is partial/ambiguous, follow with target-zero vs target-nonzero submask
interventions rather than trying to infer those contributions from aggregate
statistics alone.

#### Random reassignment control

Do **not** use an unconstrained random 6.3% code flip.

A scientifically matched random control should preserve the real mask's
distribution as closely as practical:

- within each layer and source direct-Q3 code `c_D`, sample the same number of
  positions from outside M as occur in M;
- assign target codes using the observed per-layer `c_D -> c_S` transition
  counts from M (randomly permuted among the matched sampled positions);
- place those random reassigned weights at the same standardized
  representative used by the true-M comparison, preferably the Q3 prototype.

This controls for mask size, layer allocation, source-code mix, and transition
type. It asks whether the **specific positions selected by Q9** matter, versus
merely making the same number and kinds of ternary code changes elsewhere.

Run the true-M identity/position arms first. The matched-random arm is valuable
and can be included in the same seed-1729 job if memory/runtime remains
comfortable.


## v12 — COMPLETED seed-1729 code identity vs continuous position

Job: `6ac642a0df2184ac91ac018d`  
Pinned code: `22639f5b225e56be009886bf467a409910796eef`

All exact/prototype/minimal equal-forward assertions passed with zero projected
Q3 Hamming and identical pre-continuation diagnostics.

| Arm | Final loss | Interpretation |
|---|---:|---|
| D | 5.6136 | direct baseline |
| exact | 4.9329 | exact Q9 masters on true M |
| prototype | **4.8884** | standardized Q3 reconstruction level for Q9 code |
| minimal | 5.4979 | epsilon=0.01 inside Q9-selected region |
| random | 5.6793 | transition-matched random positions |

Key effects:

- exact gain vs D: **0.6807 nats**;
- prototype gain: **0.7252 nats** = **106.5%** of exact positive-control gain;
- minimal gain: **0.1157 nats** = **17.0%** recovery;
- matched-random gain: **-0.0657 nats** = worse than D.

Current mechanism interpretation:

> Q9's exact within-region FP32 coordinates are unnecessary, but simply choosing
> the Q9 code and barely crossing its boundary is also insufficient. The useful
> information is the **specific position/code selection**, combined with a
> master state placed safely inside the selected Q3 region. The canonical Q3
> reconstruction prototype is sufficient on seed 1729 and slightly better than
> the exact Q9 master values.

The matched-random control indicates that applying the same number and types of
code changes to different positions does not help.

Do not yet claim this finer result is replicated; v12 has only seed 1729.

Canonical files:

- `results/run_v12_code_identity_position_summary.md`
- `results/run_v12_code_identity_position_seed1729_2026-10-07.json`

### Current next-question priority

Replicate the exact v12 design on orders 271828 and 424242.

If prototype again approaches or beats exact, minimal crossing remains weak,
and matched-random remains non-beneficial on both orders, the mechanism can be
upgraded to a three-order conclusion:

1. Q9 identifies useful **which-position/which-code** assignments;
2. masters need to be placed well inside the selected ternary regions;
3. exact Q9 continuous values are not required.

Only after those two confirmations should we decide whether any target-zero vs
target-nonzero submask experiment is necessary.