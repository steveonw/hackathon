# External Review Report: Progressive 9-State -> Ternary Quantization

## 1. Purpose

This project tests whether a pretrained language model can survive conversion to
ternary weights more effectively if it is first allowed to adapt in an
intermediate 9-state weight representation.

The core question is:

> Given the same pretrained checkpoint and the same total number of optimizer
> updates, does 9-state QAT followed by ternary QAT produce a better final
> ternary model than direct ternary QAT?

This is not currently a claim of a new quantization algorithm or production-ready
model. The goal is to test a narrow optimization-path hypothesis.

---

## 2. Current model and evaluation setup

Model:

- `HuggingFaceTB/SmolLM2-360M-Instruct`
- approximately 362M parameters

Current validated protocol:

- source checkpoint is rounded to BF16 before each treatment
- quantized targets: linear layers except `lm_head`
- non-quantized parameters are frozen
- one continuous FP32 shadow/master weight tensor is retained throughout QAT
- optimizer: AdamW
- optimizer state is retained across the 9 -> 3 stage transition
- learning rate: `1e-4`
- quantizer scale: learnable per output row
- scale initialization: absmean-based
- training objective: next-token cross entropy plus KL distillation to an
  untouched teacher
- fixed evaluator: WikiText-2 test slice, 8192 tokens
- discrete code-flip rates are logged during training

Ternary states are three weight values corresponding to approximately
1.585 bits of information per weight. The intermediate 9-state representation
contains approximately 3.17 bits of state information.

---

## 3. Original hypothesis and how it changed

The initial intuition was that reducing precision in stages might preserve more
information than jumping directly from high precision to three states.

Early experiments showed that this literal interpretation is not sufficient.

With deterministic projection and no adaptation:

```
FP -> 3
```

and

```
FP -> 27 -> 9 -> 3
```

ended in the same final ternary assignments under the nested mapping.

Therefore the current hypothesis is more specific:

> The advantage, if real, is not that intermediate rounding mechanically
> preserves bits. The advantage is that the model may be able to reorganize its
> continuous master weights while the forward representation still has more
> discrete states available.

In other words, the proposed mechanism is optimization-space preparation, not
information recovery from an upcast.

---

## 4. Experiment history

### v1: naive 27 -> 9 -> 3

The first real-model experiment used independently rescaled 27-, 9-, and
3-level grids with a very small QAT budget.

Result:

- direct ternary and staged ternary were both severely damaged
- staged did not outperform direct
- the 9-state phase became unstable

Interpretation:

The experiment was too small and the quantizer/training recipe was crude. It
did not provide evidence for the staircase hypothesis.

### v2: strict nested 27 -> 9 -> 3

v2 introduced an exact nested hierarchy and teacher distillation.

Important result:

- raw direct ternary and raw nested 27 -> 9 -> 3 were exactly identical

This confirmed that without learning between stages, the staircase itself does
not preserve extra information.

However, v2 hard-committed weights at each stage and recreated AdamW. The final
ternary codes also appeared effectively trapped: 30 and 90 final ternary steps
produced identical evaluated models.

Later review identified this as a major experimental weakness.

### v3: persistent FP32 masters and transition schedules

v3 retained FP32 shadow weights and introduced learnable scales.

Tested:

- direct ternary
- smooth quantization ramp
- full-precision adaptation followed by ternary
- full-precision adaptation followed by the same final ternary budget as direct

Main result:

Once final ternary training budget was matched, the schedules were close. The
full-precision preparation phase slightly reduced the immediate ternary shock,
but did not clearly outperform direct ternary.

Important confounds discovered afterward:

- the ternary distribution was approximately 79% zero
- non-quantized model components were allowed to train
- code-flip rate was not directly measured

These motivated v4.

---

## 5. v4: first clean positive experiment

v4 made the following changes:

- persistent FP32 shadow weights
- no hard commit at 9 -> 3
- Adam state preserved across the transition
- non-quantized parameters frozen
- absmean-based ternary initialization
- learnable rowwise scale
- exact code-flip diagnostics
- fixed 8192-token evaluator
- short LR diagnostic sweep

The selected learning rate was `1e-4`.

Primary equal-total-compute comparison:

```
Direct:
1200 steps @ 3 states

Staged:
300 steps @ 9 states
900 steps @ 3 states
```

Both conditions consumed the same 1200 training chunks in the same global
order.

### v4 result

| Variant | PPL lower is better | Teacher top-1 higher is better | KL lower is better |
|---|---:|---:|---:|
| BF16 source | 39.86 | 99.65% | ~0 |
| Direct ternary 1200 | 666.66 | 22.84% | 3.546 |
| **9 -> 3 staged** | **286.97** | **30.49%** | **2.716** |

Relative to direct ternary:

- perplexity reduced by 56.95%
- teacher top-1 agreement improved by 7.65 percentage points
- KL divergence reduced by 23.39%

The first direct ternary training loss was 17.31.

After 300 steps of 9-state preparation, the first ternary training loss was
6.59, a 61.9% smaller transition shock.

The new quantizer also produced a much healthier final ternary state
distribution:

- about 34.1% negative
- about 31.7% zero
- about 34.1% positive

Code-flip logging confirmed that the quantized backbone was actually changing.

---

## 6. v4b: independent training-order confirmation

A methodological issue was caught during replication.

Changing only `torch.manual_seed` produced bit-for-bit identical traces because
the training order was fixed and the relevant training path was effectively
deterministic. Those jobs were canceled and are not counted as replications.

The corrected v4b protocol used the seed to permute the same 1200 training
chunks while keeping:

- model
- evaluator
- optimizer
- learning rate
- quantizer
- objective
- treatment schedules

fixed.

Within each replication, the direct and staged treatments consumed the exact
same examples in the same order.

### Replicated results

| Training order | Direct PPL | 9 -> 3 PPL | PPL reduction | Direct top-1 | 9 -> 3 top-1 | Direct KL | 9 -> 3 KL |
|---|---:|---:|---:|---:|---:|---:|---:|
| Reference v4 | 666.66 | **286.97** | **56.95%** | 22.84% | **30.49%** | 3.546 | **2.716** |
| Seed/order 1729 | 352.77 | **178.85** | **49.30%** | 25.45% | **33.76%** | 2.950 | **2.238** |
| Seed/order 271828 | 383.56 | **177.20** | **53.80%** | 25.44% | **32.08%** | 3.046 | **2.258** |

All three paired comparisons favored 9 -> 3 on all three held-out metrics.

Across the three paired orderings:

- mean paired perplexity reduction: **53.35%**
- range of PPL reduction: **49.30% to 56.95%**
- mean teacher top-1 gain: **+7.54 percentage points**
- mean paired KL reduction: **24.47%**
- mean held-out loss improvement: **0.765 nats/token**

---

## 7. Transition-shock result

The first fully ternary loss was lower after 9-state preparation in every
paired run:

| Training order | Direct first ternary loss | First ternary loss after 9-state prep | Reduction |
|---|---:|---:|---:|
| Reference v4 | 17.31 | 6.59 | 61.91% |
| Seed/order 1729 | 13.12 | 7.50 | 42.81% |
| Seed/order 271828 | 12.51 | 6.47 | 48.31% |

Mean transition-shock reduction: **51.01%**.

This is one of the strongest repeated effects in the current experiment series.

---

## 8. Code-movement diagnostics

The project originally lacked a direct test of whether ternary weights were
actually changing. v4/v4b added code-flip logging.

In the two shuffled-order replications:

- after 300 9-state steps, about 2.35% of codes differed from their 9-state
  stage-start assignments
- after the following 900 ternary steps, about 1.14% of ternary codes differed
  from their ternary-stage-start assignments
- direct ternary after 1200 steps ended with about 1.48% of codes displaced
  from its initial ternary assignments

This does not prove that fewer ternary flips are better.

A plausible interpretation is that the 9-state phase permits larger preparatory
rearrangements before the network enters the more restrictive ternary state
space.

Some of the largest code movement repeatedly occurred in early attention
projections.

---

## 9. Current interpretation

The data currently support the following narrow statement:

> On SmolLM2-360M-Instruct under this QAT recipe and evaluator, allowing the
> model to adapt for the first 25% of its updates in a 9-state weight
> representation before switching to ternary consistently produces a less
> damaged final ternary model than spending the same total number of updates
> directly in ternary space.

The current evidence does NOT establish:

- that 9 -> 3 is globally optimal
- that 27 -> 9 -> 3 is useful
- that the effect generalizes to other architectures
- that the effect generalizes to other datasets
- that the method produces a practically useful ternary model
- that the method is novel in the research literature
- that the observed mechanism has been causally proven

---

## 10. Important negative evidence

Several results argue against an overly broad version of the hypothesis:

1. Deterministic staged projection without learning gives no benefit.
2. Hard commits between quantization stages performed poorly.
3. Resetting optimizer state between stages is likely harmful.
4. A smooth quantization-strength ramp did not outperform direct ternary in v3.
5. Full-precision adaptation before ternary did not clearly outperform direct
   ternary once final ternary training budget was matched.
6. Even the successful v4/v4b ternary models still generated degraded,
   repetitive text.

Therefore the positive result appears to depend on a specific combination:

```
continuous FP32 master weights
+ meaningful code movement
+ preserved optimizer state
+ 9-state intermediate representation
+ later ternary QAT
```

rather than staging alone.

---

## 11. Current limitation: absolute model quality remains poor

Although staged models are substantially better than direct ternary controls,
they are not yet healthy language models.

The BF16 source has held-out perplexity around 39.9.

The best v4/v4b staged models are still roughly 177-287 perplexity, and sample
generations remain repetitive or degenerate.

So the current result should be described as:

> reproducibly less damaging ternarization

not:

> successful ternary conversion.

---

## 12. v5: current scale-up experiment

The next experiment keeps the successful protocol fixed and increases training
volume by 5x.

Pinned code commit:

`23157f7ce03f6bbbf944806574b18c133843f049`

Hugging Face job:

`6ac56d99fbc85ba6823ba03f`

Design:

```
Direct:
6000 steps @ 3 states

Staged:
1500 steps @ 9 states
4500 steps @ 3 states
```

Other key settings remain fixed:

- same model
- same 25% / 75% stage split
- LR 1e-4
- persistent FP32 shadows
- optimizer state preserved
- non-quantized parameters frozen
- same fixed evaluator
- 6000 unique 128-token training chunks
- about 768k training tokens per treatment

The purpose of v5 is not to test another schedule. It tests whether the
replicated relative advantage persists with substantially more opportunity for
both treatments to recover language behavior.

At the time this report was prepared, v5 had been launched and its final result
was not yet included here.

---

## 13. Possible alternative explanations that still need testing

A critical reviewer should consider at least these possibilities:

1. **9-state optimization geometry rather than staging per se.**
   The first 300 updates may simply have better-conditioned gradients.

2. **Learning-rate interaction.**
   A different ternary learning rate or scheduler might close the gap without
   staging.

3. **Quantizer-scale interaction.**
   The learned rowwise scale could behave differently during the 9-state phase
   and carry an advantage into ternary.

4. **Optimizer-state carryover.**
   Adam moments learned during the 9-state phase may be responsible for part of
   the advantage.

5. **Teacher-distillation interaction.**
   The effect may depend on the CE/KL objective rather than on ordinary
   language-model training.

6. **Data-order sensitivity.**
   Two shuffled-order replications reduce this concern but do not eliminate it.

7. **Single-model effect.**
   SmolLM2 may have architecture- or checkpoint-specific sensitivity.

8. **Evaluation narrowness.**
   WikiText perplexity and teacher top-1 agreement may not predict useful
   instruction-following behavior.

9. **Stage-length choice.**
   The current 25% at 9 states was not exhaustively optimized.

10. **Nine states may not be uniquely special.**
    Five-, seven-, or other intermediate codebooks could perform as well or
    better.

---

## 14. Strong falsification tests

The following experiments would be especially informative:

1. Repeat the paired direct-vs-9->3 comparison on another architecture.
2. Repeat on another text corpus and evaluator.
3. Compare 5 -> 3, 7 -> 3, 9 -> 3, and 15 -> 3 under equal total compute.
4. Compare multiple stage lengths such as 10%, 25%, 50%, and 75%.
5. Reset Adam state exactly at 9 -> 3 in one ablation.
6. Freeze learned quantizer scales at the transition in one ablation.
7. Train direct ternary with a tuned LR/scheduler budget as strong as the staged
   treatment.
8. Remove teacher KL and test pure next-token training.
9. Increase total training enough that both treatments approach useful language
   quality and determine whether the staged advantage persists or disappears.
10. Measure representation similarity or layerwise activation divergence across
    the transition, not only weight-code movement.

If the staged advantage disappears under stronger direct-ternary optimization,
or fails on other architectures/data, the hypothesis should be narrowed or
rejected accordingly.

---

## 15. Questions for an external AI/reviewer

Please critique this project as if reviewing a preliminary research result.

In particular:

1. Is the paired v4/v4b design sufficient to support the narrow claim that
   9-state preparation improves this ternary QAT setup?
2. What hidden confound could still explain a roughly 49-57% paired perplexity
   improvement across three training orders?
3. Does the code-movement pattern support the proposed "reorganization before
   constraint" interpretation, or is there a better explanation?
4. Is direct ternary at LR 1e-4 a sufficiently strong control?
5. What single ablation would most strongly distinguish a true staging effect
   from optimizer-state, learned-scale, or conditioning effects?
6. What experiment would most efficiently falsify the hypothesis?
7. At what point would these results be strong enough to justify testing on a
   larger model?
8. Are there known papers or methods that make this result unsurprising or
   already explain it?
9. What metrics should be added before making any research claim?
10. What would you change in v5 before interpreting its outcome?

---

## 16. Bottom line

The first three experiment generations failed to show a useful staircase
advantage and exposed several methodological problems.

After fixing those problems, the 9 -> 3 treatment beat direct ternary in three
paired training orders under matched total updates and matched example order,
with an average 53.35% perplexity reduction and a repeated ~51% reduction in
the immediate ternary transition shock.

That is a reproducible experimental effect on this setup.

It is not yet a general result, a novelty claim, or a useful ternary LLM.


---

## 17. v5 result: the advantage persists but contracts

v5 completed after the original external-review report was prepared.

The training budget was increased from 1200 to 6000 updates per treatment,
using 6000 unique 128-token chunks (~768k training tokens per treatment).

| Variant | Held-out loss | PPL | Teacher top-1 | KL |
|---|---:|---:|---:|---:|
| BF16 source | 3.6854 | 39.86 | 99.68% | ~0 |
| Direct 6000@3 | 4.4463 | 85.31 | 41.44% | 1.522 |
| **1500@9 -> 4500@3** | **4.2493** | **70.05** | **43.64%** | **1.330** |

The v5 paired staged effect is:

- **0.197 nats/token** lower held-out loss;
- **17.88%** lower perplexity;
- **+2.20 percentage points** teacher top-1;
- **12.61%** lower KL.

This remains a staged win, but it is much smaller than the v4/v4b mean of
**0.765 nats/token**. The measured loss advantage contracted by roughly **74%**
when training volume increased 5x.

The first ternary-step loss was still reduced from 13.105 (direct) to 6.501
(after 9-state prep), a **50.39% transition-shock reduction**, closely matching
the earlier ~51% replicated mean.

### Revised interpretation after v5

v5 strengthens the evidence that 9-state preparation is a robust way to reduce
the initial optimization shock of entering ternary space.

At the same time, it weakens the stronger claim that staging necessarily leads
to a permanently superior ternary basin.

The current evidence is most consistent with:

> a large early optimization/head-start effect, with a smaller residual
> advantage still measurable after 5x more training.

A longer run could show either continued convergence or a persistent floor.
Before paying for that, the higher-value experiment is a causal transition
ablation separating adapted master weights, learned quantizer scales, and Adam
state, alongside a stronger tuned direct-ternary baseline.

Qualitative generations remain poor for both conditions; the v5 held-out metric
win has not yet become a clear generation-quality win.


---

## 18. v6 result: the advantage is carried by the master weights

v6 performed a controlled transition ablation at the v4-scale budget.

One shared 300-step Q9 checkpoint was branched into four 900-step ternary
continuations that selectively preserved or reset learned scales and Adam state.

| Condition | Held-out loss | PPL | Teacher top-1 | KL |
|---|---:|---:|---:|---:|
| Direct 1200@3 | 5.8747 | 355.90 | 25.55% | 2.954 |
| Carry masters + scales + full Adam | 5.2059 | 182.34 | 32.80% | 2.281 |
| Reset Adam | 5.2031 | 181.83 | 33.31% | 2.275 |
| Masters only, original scales, fresh Adam | 5.1990 | 181.09 | 32.74% | 2.262 |
| Masters + weight Adam, original scales | 5.1799 | 177.67 | 32.96% | 2.252 |

The four staged branches differ by only **0.026 nats/token**. Resetting Adam and
restoring the original quantizer scales does not remove the benefit.

This strongly indicates that the small-budget staged advantage is carried
primarily in the **adapted FP32 master weights**.

### Clean same-data transition test

On a fixed diagnostic set, with no optimizer update between Q9 and Q3:

- initial Q3 loss before preparation: **15.4784**;
- Q9 loss after 300 preparation steps: **4.9320**;
- immediate Q3 loss of those same prepared weights: **9.1771**.

Thus Q9 -> Q3 creates a genuine **+4.2451 nat** quantization penalty, but the
prepared Q3 projection is already **6.3013 nats/token better** than the original
Q3 projection before any ternary training.

### Future ternary boundary crossings

Q9 preparation changes **0.6769%** of eventual ternary assignments using the
learned scales.

With the **original scales held fixed**, it still changes **0.6755%**.

Therefore the boundary movement is overwhelmingly explained by motion of the
continuous master weights rather than scale calibration.

This is the strongest evidence so far for the project's mechanistic hypothesis:
intermediate-state training reorganizes continuous weights across future ternary
decision boundaries before the final constraint is imposed.

The result remains narrow: it does not prove that nine states are optimal, that
the advantage persists asymptotically, or that a stronger direct ternary
optimizer cannot close the gap.


---

## 19. v7 result: Q9 improves later trainability, not immediate Q3 entry quality

v7 directly compared three equal-compute 300-step preparation paths and then
gave each the same 900-step Q3 continuation with original Q3 scales and fresh
Adam.

At step 300, all master states were projected through the same original Q3
quantizer:

| Preparation | Fixed-Q3 loss |
|---|---:|
| Direct Q3 | **6.5589** |
| Q9 | 9.2264 |
| FP32/unquantized | 14.7346 |

Thus Q9's immediate Q3 projection is **worse**, not better, than direct Q3 after
equal compute.

Yet after the shared continuation:

| Path | Final loss | PPL |
|---|---:|---:|
| Q3 -> Q3 | 5.8724 | 355.09 |
| **Q9 -> Q3** | **5.1938** | **180.15** |
| FP32 -> Q3 | 6.0311 | 416.19 |

Q9 finishes 0.6786 nats/token better than direct. FP32 warmup fails to reproduce
the effect.

Direct Q3 and Q9 move almost the same fraction of future Q3 assignments by step
300 (0.6933% vs 0.6755%), but the changed-position sets overlap weakly
(**19.64% Jaccard**). Their global distance-to-Q3-threshold distributions are
nearly identical.

The best-supported explanation is therefore no longer "better ternary starting
checkpoint." It is a **Q9-specific trainability / weight-selection geometry**
effect: the intermediate grid changes which continuous masters are repositioned,
and that prepared state responds much better to later ternary training.


---

## 20. v7 replication: trainability effect holds across three orders

Two exact training-order replications of v7 were run at seeds 271828 and 424242.
Only the data-order seed changed.

The central result replicated in both.

| Seed | Direct Q3 @300 | Q9->Q3 @300 | Final direct | Final Q9 | Final Q9 gain |
|---:|---:|---:|---:|---:|---:|
| 1729 | 6.5589 | 9.2264 | 5.8724 | **5.1938** | **0.6786** |
| 271828 | 6.6449 | 8.8909 | 5.9802 | **5.2417** | **0.7384** |
| 424242 | 6.5824 | 9.4334 | 5.9469 | **5.3008** | **0.6461** |

Thus in all three orders:

1. Q9 preparation produces a **worse immediate fixed-Q3 checkpoint** than direct
   Q3 after equal 300-step compute;
2. after the same 900-step Q3 continuation, Q9 finishes substantially better.

Across all three:

- mean final loss advantage: **0.6877 nats/token**;
- mean paired PPL reduction: **49.69%**;
- mean teacher top-1 gain: **+6.86 pp**.

The weight-selection signal also replicates. Direct Q3 and Q9 change similar
fractions of future ternary codes, while the mean changed-position Jaccard is
only **19.66%**.

The FP32 control requires one refinement: FP32 remains far worse than Q9 in all
three orders, but on seed 271828 it slightly beats direct Q3. Therefore the
replicated conclusion is that **generic FP32 warmup does not reproduce Q9's
large trainability advantage**, not that FP32 warmup is always harmful.

This substantially strengthens the narrow claim that, under this recipe, the
intermediate Q9 forward constraint creates a reproducible finite-budget
trainability / weight-selection geometry effect.


---

## 21. v8 result: simple directional pre-loading does not explain Q9

v8 tested a more specific mechanism for the replicated v7 trainability effect.

For each weight whose Q3 code changes during the common 900-step continuation,
the diagnostic measured whether its first-300-step FP32 master motion was
already aligned with the direction of that later ternary transition.

Because later-flip sets are selected after each arm's own trajectory, the
interpretation was preregistered around a four-way matched-position comparison.

On Q9's later-flip set:

- Q9 prep aligned displacement: **0.002328**
- Q3 prep on the same positions/directions: **0.000873**
- Q9 matched advantage: **+0.001455**

On Q3's later-flip set:

- Q3 prep aligned displacement: **0.003130**
- Q9 prep on the same positions/directions: **0.001439**
- Q3 matched advantage: **+0.001691**

Thus each preparation mainly wins on its own later-flip set by a similar
amount. The direct-Q3 own-set advantage is slightly larger.

That is the pattern expected from post-selection / trajectory-specific
alignment, not evidence that Q9 uniquely "pre-points" useful weights toward
their future ternary thresholds.

The v7 trainability effect itself remains intact and reproduced inside v8.
Therefore the mechanism should remain described at the broader
**trainability / weight-selection geometry** level. Global threshold proximity
(v7) and simple signed future-threshold preloading (v8) have both failed to
explain the Q9 advantage.


---

## 22. v9 result: stronger direct optimization narrows but does not erase Q9

v9 addressed a major baseline concern: the historical direct-Q3 LR search had
ended at 1e-4, which was also the best tested value.

Seven direct schedules were screened for 300 updates using only the validation
split. The two best non-reference candidates were then run for the full 1200
updates alongside the historical 1e-4 reference.

The best full direct schedule used a 100-step warmup to 1e-3 followed by cosine
decay to 1e-4.

| Condition | Held-out loss | PPL | Top-1 | KL |
|---|---:|---:|---:|---:|
| direct 1e-4 | 5.8747 | 355.90 | 25.55% | 2.954 |
| direct 1e-3 | 5.8356 | 342.26 | 26.68% | 2.894 |
| **direct warmup+cosine** | **5.5957** | **269.27** | **29.87%** | **2.664** |
| historical Q9 -> Q3 | **5.1938** | **180.15** | **33.79%** | **2.256** |

The tuned schedule closes about **41%** of the historical seed-1729 Q9/direct
loss gap.

Therefore the earlier direct baseline was materially weak, and the Q9 effect
size must be revised downward when comparing against a stronger optimizer
schedule. However, Q9 still retains a **0.402-nat/token** loss advantage and
about **33% lower perplexity** on this seed.

This tuned direct schedule now requires replication on the other two v7
training orders before being treated as the canonical baseline.


---

## 23. Tuned direct replication: residual Q9 advantage survives 3/3 orders

The v9-selected direct-Q3 schedule was replicated without further tuning on the
two remaining v7 training orders.

| Seed | Tuned direct loss | Q9 -> Q3 loss | Q9 advantage |
|---:|---:|---:|---:|
| 1729 | 5.5957 | **5.1938** | **0.4020** |
| 271828 | 5.6228 | **5.2417** | **0.3810** |
| 424242 | 5.6067 | **5.3008** | **0.3059** |

Thus Q9 remains better than tuned direct in **3/3 orders**.

Across the three orders, the mean residual Q9 advantage is **0.3630
nats/token**, with **30.38% lower perplexity** and **+2.52 percentage points**
higher teacher top-1 agreement.

The stronger direct schedule closes about **47%** of the historical Q9-vs-direct
loss gap on average. This materially reduces the original effect size and
confirms that the historical direct recipe was weak, but does not erase the Q9
advantage.

The tuned warmup+cosine schedule should therefore be treated as the canonical
small-budget direct baseline for subsequent mechanism experiments.


---

## 24. v10 result: Q9 advantage survives equal global LR schedule

v10 applied the exact direct-selected warmup+cosine global LR schedule to the
Q9 -> Q3 path on all three established training orders.

| Seed | Tuned direct | Historical Q9 | **Schedule-matched Q9** | Matched-Q9 gain |
|---:|---:|---:|---:|---:|
| 1729 | 5.5957 | 5.1938 | **4.9010** | **0.6947** |
| 271828 | 5.6228 | 5.2417 | **4.9510** | **0.6718** |
| 424242 | 5.6067 | 5.3008 | **4.9563** | **0.6505** |

Schedule-matched Q9 therefore beats tuned direct in **3/3 orders**, with a mean
held-out loss advantage of **0.6723 nats/token**, **48.94% lower perplexity**,
and **+6.82 pp** teacher top-1 agreement.

The same schedule improves Q9 by **0.3094 nats/token on average** relative to
the older constant-1e-4 staged runs.

Crucially, the equal-compute step-300 diagnostic still goes the other way.
Projected through the same original Q3 quantizer, Q9 is **0.5120 nats/token
worse than tuned direct on average** after the first 300 updates. It only
becomes much better after the common later Q3 optimization budget.

This strengthens the trainability interpretation: the effect is not explained
by the old direct LR schedule, and it survives an equal global schedule.

The stronger LR schedule changes the preparation regime substantially:
step-300 projected-Q3 code displacement rises to roughly **4.41% for direct**
and **4.88% for Q9**, compared with ~0.68% in the old v7 constant-LR regime.
Accordingly, the old v7 changed-set/Jaccard geometry should not be assumed to
transfer unchanged to the tuned regime.

This remains a finite-budget, single-model/data result, and v10 is an
equal-schedule control rather than a full equal-search-budget tuning study.
