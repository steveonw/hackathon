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


---

## 25. v11 result: the Q9 advantage is causally concentrated on code-disagreement positions

v11 performed a four-arm hybrid-master intervention under the stronger
matched-schedule regime.

At step 300, direct-Q3 masters D and Q9-prepared masters S were projected through
the same original Q3 scales. The mask M where their Q3 codes differ contains
**6.458%** of quantized weights.

Four master states were built:

- 00 = D everywhere;
- 10 = S on M only;
- 01 = S off M only;
- 11 = S everywhere.

Before continuation, Q3(00) and Q3(01) were exactly identical, as were Q3(10)
and Q3(11), with zero Hamming distance and identical validation diagnostics.
Therefore differences within each pair arise from hidden continuous master
geometry rather than a different ternary forward model at continuation start.

Final held-out losses:

| Arm | Loss |
|---|---:|
| 00 | 5.6136 |
| 10 | **4.9329** |
| 01 | 5.5020 |
| 11 | **4.9010** |

The full 00->11 advantage is **0.7126 nats/token**. Transferring Q9 masters only
on M recovers **0.6807 nats**, or **95.5%** of that gain. By contrast,
transferring Q9 masters only on the same-code majority improves loss by just
**0.1116 nats** on the direct background.

Thus, on seed 1729, the dominant causal carrier of the Q9 trainability advantage
is the relatively small set of positions where Q9 and direct choose different
projected ternary assignments after preparation.

This does not yet prove that the discrete ternary code identities themselves
are sufficient. The intervention transfers the full continuous Q9 master values
on M. Replication across the other two training orders and a code-vs-within-bin
intervention on M are the appropriate next steps.


---

## 26. v11 mechanism replication: localization survives 3/3 orders

The hybrid-master intervention was replicated on orders 271828 and 424242 with
no design changes.

| Seed | D-vs-S mask | Full Q9 gain | Mask-only gain | Recovery |
|---:|---:|---:|---:|---:|
| 1729 | 6.458% | 0.7126 | 0.6807 | 95.5% |
| 271828 | 6.181% | 0.7014 | 0.6813 | 97.1% |
| 424242 | 6.301% | 0.6672 | 0.6443 | 96.6% |

The mask-only intervention therefore recovers **96.4% of the full Q9
trainability advantage on average**, from a disagreement set containing only
**6.31% of quantized weights on average**.

The same-code majority still produces a smaller secondary effect, but after the
disagreement-mask positions already use Q9 masters, the remaining same-code
contribution averages only **0.025 nats/token**.

This upgrades the seed-1729 localization to a replicated conclusion for this
model/data/training setup.

The result identifies **where** the useful prepared state resides, not yet
**which property** of that state is sufficient. The mask-only intervention
transfers exact continuous Q9 master values. A code-identity-vs-within-bin
intervention is therefore the next causal test.


---

## 27. v12 seed-1729 result: Q9's exact master value is unnecessary, but boundary depth matters

v12 separated code identity from continuous position on the v11 disagreement
mask while holding the ternary forward model fixed.

The exact-Q9, reconstruction-prototype, and minimal-crossing arms all began
with exactly identical projected Q3 weights and diagnostics.

Final losses:

| Arm | Loss |
|---|---:|
| Direct | 5.6136 |
| Exact Q9 masters on M | 4.9329 |
| Q3 prototype for Q9 code on M | **4.8884** |
| Minimal crossing into Q9 code region | 5.4979 |
| Transition-matched random positions | 5.6793 |

The Q3 prototype slightly **outperforms the exact Q9 master values**, showing
that precise Q9 within-region continuous coordinates are not required.

However, merely crossing the boundary into the same Q9-selected code region
recovers only ~17% of the exact positive-control gain. Thus code identity by
itself is insufficient; boundary-relative depth / continuous placement inside
the selected region matters strongly.

The matched-random reassignment is worse than direct despite preserving the
real mask's layer/source/target transition counts, supporting the conclusion
that Q9's specific position selection is important rather than generic code
disruption.

This is a seed-1729 mechanism result and should be replicated on the other two
training orders before becoming canonical.


---

## 28. v12 mechanism replication: code choice plus interior placement explains the effect

The seed-1729 v12 intervention was replicated exactly on orders 271828 and
424242.

In all three orders, exact Q9 masters, Q3 reconstruction prototypes, and
minimal-crossing masters began from the same projected Q3 forward model.

| Seed | Exact Q9 | Prototype | Minimal crossing | Matched random |
|---:|---:|---:|---:|---:|
| 1729 | 4.9329 | **4.8884** | 5.4979 | 5.6793 |
| 271828 | 4.9711 | **4.9187** | 5.5406 | 5.7208 |
| 424242 | 4.9791 | **4.9373** | 5.4911 | 5.7031 |

The prototype recovers **106.9%** of the exact-Q9 positive-control gain on
average and beats exact Q9 in all three orders. Minimal crossing recovers only
**18.0%** on average. The matched-random reassignment is worse than direct in
all three orders.

The mechanism is therefore sharper than "Q9's exact continuous masters are
special." Instead:

- Q9 identifies the useful positions and target ternary codes;
- those masters must be placed meaningfully inside the selected Q3 regions;
- the standard Q3 reconstruction prototype is sufficient and slightly better
  than the exact Q9 values;
- making the same kinds of code transitions at different matched positions does
  not help.

Within the present SmolLM2/WikiText-2 setup, this closes the mechanism sequence.
Further work should test generality or exploit the finding practically rather
than continue subdividing the same causal story.


---

## 29. Optional v13 closing experiment: moderate interior depth is enough

A seed-1729 depth sweep refined the v12 prototype result. All depth arms began
from the same projected Q3 forward model, differing only in hidden master
placement inside the Q9-selected regions.

Final losses were 5.4948, 4.9262, 4.8850, 4.8876, and 4.8884 for depths 0.03,
0.25, 0.50, 0.75, and 1.00 respectively, versus a direct baseline of 5.6136.

The useful effect is therefore not monotonic with increasing depth. It rises
sharply between the boundary and moderate interior placement, then saturates by
about d=0.5.

Q9-code survival after 900 continuation updates rises in parallel from 51.76%
at d=0.03 to 86.48% at d=0.25 and 97.32% at d=0.50, reaching ~99% for deeper
placements. Fixed-alpha0 and learned-scale survival are nearly identical.

Two firmness-only controls that kept direct's own code choices but snapped their
masters to prototypes were non-beneficial. This supports the interpretation
that Q9's specific assignment choices are essential, while moderate interior
placement helps preserve those assignments during later Q3 optimization.

This v13 result is seed 1729 only and should be presented as an optional
refinement. The v12 three-order result remains the replicated mechanism claim.


---

## 30. G1 begins: Granite-350M engineering gate and precision resolution

The cross-family phase began with
`ibm-granite/granite-4.0-350m`, chosen as the small Family-B cell before any
larger-model scale-up.

G1-0 first audited the canonical intervention on Granite. The target/freezing
mapping looked clean: 249,561,088 weight parameters across 168 linear modules
were selected, non-quantized parameters remained frozen, and no unexpected
trainable tensors appeared. But the inherited Smol FP16 compute path failed
numerically even on the unquantized Granite source. The source evaluator became
non-finite, and Q3/Q9 teacher-KL training steps had non-finite loss and
gradients.

This was retained as a **technical failure only**, not interpreted as evidence
against ternary or Q9 staging.

A preregistered G1-0b precision diagnostic then compared FP32, BF16 and FP16 on
the same source before any D-vs-S experiment:

| Path | CE | Finite? |
|---|---:|---|
| FP32 source | 3.23537 | yes |
| BF16 source | 3.24056 | yes |
| FP16 source | NaN | no |
| BF16 teacher | 3.23373 | yes |
| FP16 teacher | NaN | no |

One-step BF16 Q3 and Q9 training also produced finite CE, KL, total loss and
gradients. BF16 source CE differed from FP32 by only 0.00519 nats/token on the
smoke slice.

Therefore Granite G1 now locks:
FP32 persistent masters + BF16-rounded common source + BF16 autocast/BF16
teacher. FP16 is not used.

This precision choice was frozen **before any Granite Q9-vs-direct scientific
result existed**, preserving the fairness of the generalization phase.

Jobs:
- G1-0: `6ac6eacfdf2184ac91ac658a`
- G1-0b: `6ac6eec9df2184ac91ac67ca`

G1-1 direct-Q3 schedule calibration was then launched as
`6ac6f18ae7a0dae8a2780246`, screening only the preregistered direct schedules
on the LR-validation split. It contains no Q9 arm and does not touch the held-out
test set.


---

## 31. G1-1 — Granite direct-Q3 schedule calibration

After the BF16 precision gate passed, Granite's direct ternary schedule was
selected **before any Q9 scientific result existed**.

Job: `6ac6f18ae7a0dae8a2780246`  
Pinned code: `3cb0153be80d5e9fbe112460bde7bc26398d851f`

All candidates used seed/order 1729, the same first 300 shuffled chunks,
direct-Q3 only, FP32 masters, and the locked BF16 compute path. Selection used
only the 24-chunk LR-validation set; the held-out test evaluator was not used.

| Schedule | validation loss | PPL | Q3 code movement @300 |
|---|---:|---:|---:|
| **constant 1e-4** | **5.8128** | **334.56** | 5.323% |
| warm100 -> 3e-4, cosine -> 1e-4 | 6.0827 | 438.21 | 11.616% |
| warm100 -> 1e-3, cosine -> 1e-4 | 6.7402 | 845.70 | 27.231% |

The conservative constant schedule won clearly. Larger LR schedules moved many
more ternary assignments but generalized worse on the validation slice.

Per preregistration, **constant 1e-4 is now frozen** for both direct and staged
Granite treatments. No retuning is allowed after observing Q9.

G1-2, the first scientific Granite D-vs-S gate, was launched as
`6ac6f487df2184ac91ac693e`.


---

## 32. G1-2/G1-3 — first Granite staging and mechanism result

Granite-4.0-350M produced the first scientific cross-family positive result on
training order 1729.

G1-2 first compared direct Q3 with Q9→Q3 under the direct-selected constant
1e-4 schedule.

| Metric | Direct | Q9→Q3 |
|---|---:|---:|
| held-out loss | 5.66581 | **5.53495** |
| PPL | 288.82 | **253.40** |
| teacher top-1 | 27.53% | **28.04%** |
| KL | 2.49025 | **2.38307** |

The staged gain is 0.13086 nats/token with 12.27% lower PPL.

Crucially, at equal 300-step compute the Q9-prepared state is a *much worse*
immediate ternary model: fixed-Q3 loss 7.74247 versus 5.81745 for direct.
Thus Granite reproduces the Smol trainability-not-entry-quality signature.

The step-300 projected D-vs-S disagreement mask contains **6.0246%** of
targeted weights.

G1-3 then tested that mask causally with five common-continuation arms. All
equal-forward and matched-random construction assertions passed.

| Arm | Loss | Interpretation |
|---|---:|---|
| D | 5.66205 | direct-prepared baseline |
| S | 5.53495 | full Q9 preparation |
| M-exact | 5.56525 | exact Q9 masters only on true disagreement mask |
| M-d50 | **5.51134** | Q9-selected codes on true mask at d=0.5 |
| Random-d50 | 5.73514 | same transition structure at matched other positions |

M-exact recovers **76.2%** of the full S gain. This is less complete than the
~96.4% localization on SmolLM2.

However, true-mask d=0.5 placement recovers **118.6%** of the full S gain and
beats the full staged endpoint. The matched-random intervention is harmful,
worse than D by 0.07308 nats.

Therefore the deeper causal story generalizes qualitatively: Q9 finds useful
specific position/code commitments, and moderate interior placement of those
commitments is sufficient; exact Q9 continuous values are not required, while
making equivalent transitions elsewhere does not help.

Q9-code survival for M-d50 is 99.68% at continuation step 100, 97.98% at step
300, and 90.70% at step 900. Fixed-original-scale and learned-scale survival
are almost identical.

This remains **one Granite training order**. No confirmatory Granite order was
launched automatically.

Jobs:
- G1-2: `6ac6f487df2184ac91ac693e`
- G1-3: `6ac6fe5fdf2184ac91ac6c1f`


---

## 33. Granite confirmations: one negative order changes the headline

The two preregistered Granite confirmation orders completed with no protocol
changes.

| Seed | D | S | M-exact | M-d50 | Random-d50 |
|---:|---:|---:|---:|---:|---:|
| 1729 | 5.66205 | **5.53495** | 5.56525 | **5.51134** | 5.73514 |
| 271828 | **5.72673** | 5.84103 | 5.80453 | 5.78502 | 5.79268 |
| 424242 | 5.80192 | **5.74057** | 5.74471 | **5.71016** | 5.81984 |

Seed 271828 is a genuine negative for the raw staging hypothesis: full Q9→Q3
finishes 0.11431 nats worse than direct. Therefore Granite does **not** give a
3/3 replication of the Smol staging advantage.

At the same time, several pieces are more stable than the headline endpoint:

- Q9 is worse as an immediate fixed-Q3 checkpoint on 3/3 orders;
- the D-vs-S disagreement set remains about 6% (mean 6.35%);
- true-mask d=0.5 beats full S on 3/3;
- true-mask d=0.5 beats the matched-random control on 3/3;
- matched-random is harmful versus D on 3/3;
- but true-mask d=0.5 itself beats D only 2/3.

Thus the most defensible cross-family interpretation is no longer "Q9 staging
always wins." Instead, Q9 consistently exposes a structured set of alternative
ternary commitments, and standardized interior placement of those commitments
is more robust than carrying the full Q9-prepared state. Whether those
commitments are globally beneficial still depends on the training order.

This is a stronger scientific outcome than averaging the three seeds into a
small positive mean, because it preserves the observed heterogeneity.

Canonical aggregate:
`replications/g1_granite350m_mechanism_aggregate_summary.md`.


---

## 34. Forensics: the negative Granite order is already different during Q9 prep

No additional GPU job was required. The three raw G1 mechanism files were
compared directly.

The clearest discriminator is visible before the Q3 switch:

| diagnostic | 1729 | 271828 | 424242 |
|---|---:|---:|---:|
| native Q9 minus direct loss @300 | −0.160 | **+0.278** | −0.138 |
| Q9/direct grad ratio @100 | 0.82× | **3.33×** | 0.76× |
| disagreement mask | 6.02% | **6.99%** | 6.03% |

On the positive orders, Q9 itself is already training well in its own 9-state
space by step 300. On the negative order, Q9 remains worse than direct and its
gradient norms remain elevated.

The negative order's Q9 mask is also more disjoint from direct's source-code
movement: an overlap proxy is ~19.7% versus ~27.5% on the two positive orders.

Causal decomposition adds a second clue. On 271828, even Q9 state outside the
D-vs-S mask is harmful. Standard d=0.5 placement improves the true-mask state,
but true positions finish only 0.0077 nats better than transition-matched random
positions. On the positive orders that separation is 0.224 and 0.110 nats.

Meanwhile global code histograms, scales, d=0.5 survival and continuation churn
are almost unchanged.

The resulting hypothesis is narrower than "staging sometimes fails": Q9 may
need to **successfully settle and discover good assignments during its own prep
phase**. Order 271828 appears to be a failed assignment-discovery trajectory.
This is post-hoc evidence from three orders and must be preregistered before
being used predictively.

Detailed note:
`replications/g1_granite350m_seed271828_forensics.md`.


---

## 35. Granite 271828: seed-conditioned gradient alignment did not rescue the hard order (G1-6)

A deliberate development experiment on the known-negative order 271828
tested whether Q9-proposed code changes could be selected by their alignment
with the Q3 gradient over already-seen training chunks 277–300.

Within each layer and direct→Q9 source/target transition, the disagreement set
was split into equal, exactly transition-matched top and bottom halves by the
first-order score `−gradient × (w_d50 − w_direct)`. Both halves had 8,727,709
positions; all construction assertions passed.

| Arm | Held-out loss |
|---|---:|
| Direct | **5.72673** |
| All-mask d=0.5 | 5.78502 |
| Top50 aligned d=0.5 | **5.76870** |
| Bottom50 aligned d=0.5 | 5.77363 |

The selector ordered the arms weakly in the predicted direction: Top50 improved
over Bottom50 by **0.00494 nats** and over unfiltered All-d50 by **0.01632**.
But Top50 still lost to D by **0.04197**. Thus the alignment selector is not
a successful practical rescue.

A striking unplanned observation was that the step-300 fixed-Q3 validation loss
was much worse for *either individual half* (Top50 **13.697**, Bottom50
**18.004**) than for the all-mask arm (**6.515**). This suggests network-level
interactions between Q9-selected commitments, but did **not** prove a
specific synergy mechanism.

Job: `6ac71a23df2184ac91ac73ca`.
Pinned code: `10110ffba0ce9b1069f015cd32b235958e37f6c9`.

---

## 36. Granite 271828: the successful Smol LR schedule does not transfer (G1-7/G1-7b)

Following the user's observation that the canonical Smol mechanism used
a v10–v13 matched LR schedule while Granite used the older constant-`1e-4`
schedule, we directly transferred Smol's curve:

- 100-step linear warmup to `1e-3`;
- cosine decay to `1e-4` over steps 101–1200;
- D and Q9 prep both use the first 300 steps;
- 900 Q3 continuation steps follow the global curve without restarting it.

Granite kept its BF16-safe pathway. The original five-arm attempt
(`6ac722dfdf2184ac91ac75e9`) failed *technically* after preparation,
before continuation, when a full-size layer/source→target matched random
control became impossible: one stratum required **595,275** outside-mask
positions but had only **342,644**. A preregistered amendment removed that
control; the four-arm retry completed with all equal-forward assertions.

| Arm | Earlier constant-1e-4 Granite 271828 | Smol v10–v13 schedule transferred |
|---|---:|---:|
| Direct D | **5.72673** | **6.00895** |
| Full Q9→Q3 S | 5.84103 | 6.36920 |
| M-exact | 5.80453 | 6.35442 |
| M-d50 | 5.78502 | **6.30402** |

Full staging's deficit grew from 0.11431 to **0.36025 nats/token**. D itself
also deteriorated. Most notably, after 300 steps, D changed **27.11%** of
initial projected codes, Q9 changed **23.43%**, and their disagreement mask
expanded from about **6.99% to 32.71%** (81.6 million positions).
At 900 continuation steps the d=0.5 Q9-selected-code survival was only
**60.75%**, compared with 89.59% under constant LR.

Q9's native validation loss at step 300 remained worse than direct's by
0.3894 nats and its gradient norm was 5.15× larger. The immediate ternary
projection gap was actually smaller under the higher-LR schedule, even though
the final staging disadvantage grew.

The direct transfer of a Smol-tuned `1e-3` peak therefore failed to rescue
the known-hard Granite order; the same nominal 300/900 steps corresponded to
vastly different code-selection dynamics. The large mask and poor survival
are correlates of this failure, not separately proven causes. Since the
matched-random arm could not be run in this schedule, no new position-specificity
claim is made.

Completed job: `6ac7248adf2184ac91ac768d`.
Pinned code: `808fd4975d16111d9c1c841c44d9038a01995ba3`.

Canonical result:
`results/run_g1_7b_granite350m_v10schedule_summary.md`.
Raw:
`results/run_g1_7b_granite350m_v10schedule_seed271828_2026-10-08.json`.

No additional replication or scale job was launched.
