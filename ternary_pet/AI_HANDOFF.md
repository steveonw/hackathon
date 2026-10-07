# AI HANDOFF — Ternary Pet Quantization Research

> **START HERE if you are a new AI taking over this project.**
>
> Repository: `steveonw/hackathon`  
> Project directory: `ternary_pet/`  
> Current experiment: **v7 completed — trainability/weight-selection geometry result**  
> Current v7 pinned code: `3aa494a4c8418052ed9e13e0de4f97c692a29fc7`  
> Hugging Face job: `6ac58e76fbc85ba6823bad78`

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

> We have reproducible evidence that Q9 preparation gives a large early ternary
> recovery advantage carried in adapted FP32 masters; v7 is now testing whether
> those masters are already a better equal-compute ternary checkpoint, are
> merely more trainable inside Q3 geometry, and whether Q9 is special compared
> with FP32 warmup.


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


## ACTIVE v7 REPLICATIONS

Two confirmatory replications of the completed v7 protocol are running. They
change **only the training-order seed**.

Pinned commit:
`6caf3a98d485ed1fd49e22b915ddb6b175578420`

Jobs:

- seed 271828 — `6ac59807fbc85ba6823bb060`
- seed 424242 — `6ac59809fbc85ba6823bb063`

When they finish, compare each against the original seed 1729 on these exact
questions:

- step-300 fixed-Q3 ordering: direct Q3 vs Q9 vs FP32;
- final 1200-step ordering after the identical Q3 continuation;
- Q9-vs-direct final loss/PPL/top-1/KL effect size;
- Q3-vs-Q9 changed-code Jaccard and pairwise Hamming;
- threshold-margin distributions;
- qualitative generation.

Do not alter the protocol between seeds. If one seed contradicts the original,
record it as a real replication failure rather than tuning it away.
