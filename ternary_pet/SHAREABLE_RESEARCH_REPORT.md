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


---

## 37. Granite seed 424242: Smol's stronger learning-rate schedule also harms a formerly positive order (G1-8)

The prior schedule-transfer test on Granite's known-negative order 271828
showed that copying Smol's 100-step warmup to 1e-3 plus cosine decay to 1e-4
made both Q3 direct and full Q9→Q3 worse, inflating code disagreements to
32.7%. To test whether that effect was isolated to the bad seed, we ran
**424242**, chosen as the *weaker positive* of the two remaining historical
Granite orders. Unlike an independent blind seed, this was a deliberate
post-confirmation choice using known outcomes.

The G1-8 script changed **only the seed** relative to G1-7b; it kept
Granite's BF16 compute path, FP32 masters, the 300+900 split, original scales,
fresh Adam, and all four arms D, S, M-exact, and M-d50. No random arm was run
under the aggressive schedule because the previously specified full-size
outside-M matched-random control became infeasible.

| Arm | Previous Granite 424242 constant-1e-4 loss | Smol-schedule Granite 424242 loss |
|---|---:|---:|
| Direct D | **5.80192** | 6.05708 |
| Full Q9→Q3 S | 5.74057 | 6.06241 |
| M-exact | 5.74471 | 6.04609 |
| M-d50 | **5.71016** | **6.02857** |

All four arms worsen under the transferred high-peak schedule. Full staging's
old **+0.06135-nat advantage** becomes **−0.00534 nats**, essentially a
near-tie. But **M-d50 still beats D by 0.02850 nats** under the new LR and
beats full S by 0.03384 nats. Therefore high LR degrades Granite globally
while retaining at least some useful code-placement signal on this seed.

The D-vs-S disagreement mask grows from **6.03%** to **29.49%**
(15.05 million to 73.60 million weights), Q3 and Q9 code movement both
reach ~27% from initialization, and true-mask code survival at the
900th Q3 step drops from **90.57%** to **57.14%**. This reproduces the
large-disagreement and low-survival phenomenon seen on 271828, but not
the exact Q9 preparation failure: on 424242 Q9 native step300 validation
was *better* than D (6.21319 vs 6.30551) and Q9 gradient norm was
*smaller* than D (1.257 vs 1.941), whereas on 271828 Q9 prep remained
worse with much larger gradients.

This distinction matters. The results suggest a **cross-seed schedule
mismatch** with Granite, not an explanation that all failures follow
the same Q9-specific preparation path. Nor do they prove that margin growth
itself causes worse validation, or that the ternary code-change mechanism
is false. The original three-order constant-1e-4 Granite evidence
remains 2/3 positive for full staging and 3/3 positive for d50 vs full S.
The replicated Smol v10 results also remain unchanged.

G1-8 job: `6ac79551e7a0dae8a2788f0b`, completed 2026-10-08 13:30 UTC.
Frozen code: `39399baff38b15f961e9571f142e193a783c925b`.
Raw: `results/run_g1_8_granite350m_v10schedule_seed424242_2026-10-08.json`.
Analysis: `results/run_g1_8_granite350m_v10schedule_summary.md`.

No further GPU jobs were launched.

---


## 38. G1-9: Direct ternary QAT improves dramatically using nine gridward master-weight pulls

We returned to the known-hard Granite-4.0-350M order **271828** for a
pre-registered training-method test inspired by published WinQ techniques.
Unlike the earlier Q9→Q3 mechanism tests, **every arm trained direct Q3**
for 1,200 optimizer steps using the same BF16-safe Granite path and
constant `1e-4` learning rate. A 2×2 design separated Gaussian
latent-weight perturbation from periodic interpolation toward the
current ternary code's quantized value.

| Direct ternary training arm | Holdout CE loss | PPL | D−arm gain |
|---|---:|---:|---:|
| D — direct baseline | 5.726725 | 306.96 | — |
| G — Gaussian only | 5.743370 | 312.11 | −0.016645 |
| **P — gridward pull only** | **5.489709** | **242.19** | **+0.237016** |
| GP — Gaussian plus pull | **5.487853** | **241.74** | **+0.238872** |

The nine **post-optimizer** gridward interventions at global steps
100,200,...,900 used
`W ← 0.90W + 0.10 Q3(W)`; only the full-precision master
weights were adjusted, with the current trainable rowwise Q3 scale and no
optimizer-step advantage. The Gaussian arms used training-only
`σ=0.04×α_row` through step900, linearly annealing to zero by step1200.
Every held-out and validation forward was *hard ternary, noise free*.

**The historical hard-order D loss was reproduced exactly**, not just
approximately, and all four arms and hard checks completed. Gridward-only
P cut perplexity **21.10%** and improved teacher-top1 from
27.66% to 31.08% versus direct. GP edged P by a negligible 0.001856
nats; Gaussian alone was actually worse than D. This supports a
strong **pull effect**, not a confirmed Gaussian smoothing effect.

Sampled clean-code transitions (32,768 deterministic positions, each
observed after every optimizer step) during 900 Q3 continuation updates:

- D: 0.000735745 flips/weight/update;
- G: 0.000741340;
- P: 0.000094469 (**87.2% lower than D**);
- GP: 0.000089281.

D changed 10.335% of initial source Q3 codes by the final checkpoint;
P changed 5.281%. The immediate reversal fraction conditional on a
flip only decreased modestly, so the evidence is better characterized
as **reduced code-transition activity** than as a proven solution
specifically to two-step oscillation. The intervention also alters the
location of FP32 masters inside their quantization cells, making
causal attribution to flips alone premature.

**Interpretation boundary:** one deliberately selected hard development
order, one fixed interpolation strength/noise schedule and a single
8192-token held-out slice. No Q9 stage or new matched-random
position-control was present. This does not invalidate prior
Smol/Granite staged conclusions. Related methods of Gaussian latent
noise and periodic gridward interpolation were already published in
WinQ (ICML 2026), so this outcome validates an adaptation rather than
establishing invention of the intervention. A replication with frozen
P/GP settings on a separate Granite order and another architecture
would be scientifically more valuable than immediately tuning more
hyperparameters on 271828.

The original two failed submissions were Hugging Face HTTP 429
infrastructure responses **without any job IDs**. The successful
third accepted job was `6ac821ec095c5780892ff7f9`,
completed **2026-10-08 23:32:17 UTC**.

- Pinned code: `6ce4c4056e8cfd3292c54f34bedc68bf7691688f`
- [Raw JSON](results/run_g1_9_granite350m_gaussian_pull_seed271828_2026-10-08.json)
- [Mechanism and result summary](results/run_g1_9_granite350m_gaussian_pull_summary.md)
- Protocol: `EXPERIMENT.md`

No further GPU jobs launched in response to this result.

---


## 39. G1-10: a positive Granite training order replicates the gridward direct-Q3 effect without retuning

We preregistered a **seed-only reproduction** of G1-9 on previously
Q9-staging-positive Granite order **424242**. The new source changes only
`SEED=271828` to `SEED=424242`, leaving all WinQ-inspired
Gaussian/pull settings, hard-Q3 forward, FP32 masters, BF16 compute,
same 300/900 optimizer/scale reset, constant `1e-4` LR,
WikiText-2 objective/evaluator, and step budget unchanged.
Every arm and all structural checks passed. No new mask/random
intervention was introduced.

| Direct Q3 training arm | Previous G1-9 271828 loss | New G1-10 424242 loss | New 424242 PPL |
|---|---:|---:|---:|
| D — baseline | 5.726725 | 5.801920 | 330.93 |
| G — Gaussian only | 5.743370 | 5.756362 | 316.20 |
| **P — gridward only** | **5.489709** | **5.528112** | **251.67** |
| GP — combined | 5.487853 | 5.535340 | 253.49 |

**P wins again:** New D−P held-out loss gain **+0.273808 nats**,
PPL **−23.95%** on 424242. Prior 271828 D−P gain +0.237016
and PPL −21.10%. Together, the two known orderings yield
mean D−P gain **+0.255412 nats**, without changing any
hyperparameters. Both within-run D arms reproduced their historical
constant-LR Granite baselines exactly.

The Gaussian portion proved **less consistent**: G alone was
0.016645 nats *worse* than D on seed271828 yet 0.045558 nats
*better* on seed424242; combining G with P was 0.001856 nats
better than P on 271828 but 0.007228 nats *worse* on 424242.
Thus we have **within-Granite order replication for P**, not
a confirmed Gaussian-only mechanism or an advantage of GP.

A deterministic sample of 32,768 weights across 16 layers tracked
clean ternary codes after every optimizer step. On 424242 in the
900-update continuation, sampled transition frequency was
**0.000728421 per sampled weight/update** in D and
**0.000095486** in P — **~86.9% lower**. The previous
271828 result had similarly ~87.2% lower sampled flip frequency
under P. Final initial-reference code-movement fractions were
424242 D **10.327%**, P **5.369%**. Improved performance
and reduced transitions co-occur, but the evidence is not yet
a causal demonstration that eliminating two-step oscillations
produced the observed gains; gridward pulls change where latent
FP32 masters lie within their ternary cells.

**Metadata transparency:** Because the frozen G1-10 implementation
is exactly a one-line seed change, its raw `FINAL_JSON` still
uses inherited `kind=g1_9...` / `event=g1_9...`
labels and `effects.historic_D_loss_271828=5.726725` /
`effects.D_minus_historic_D_loss=0.075195`
metadata. These are **parent-run historical comparator labels**,
not valid same-seed 424242 comparisons. The correct historical
424242 D is **5.801920056343079**, exactly equal to
this G1-10 D. To maintain auditability, raw output was
archived without editing; interpret it with job ID, code pin,
actual seed, and the corrected analysis.

**Scope:** The positive pull effect now spans two *previously
observed training orders* but just one Granite model family,
one short 1,200-update regime, one held-out test slice, and the
same published-interpolation technique. It has **not yet**
transferred to Smol, and is not a novel invention of WinQ-style
gridward interpolation. Neither the Smol Q9 staging successes
nor the mixed original Granite staging results are overwritten.
A frozen-setting cross-family experiment would be the natural
next confirmation; it has **not** been authorized or launched.

Job `6ac82c93fee2c900701711db`, completed
**2026-10-09 00:21:46 UTC**.
Pinned script commit: `e08659a51fd0d72ed85f01f8e7ce739283ca6c61`.

- [Raw unmodified result](results/run_g1_10_granite350m_gaussian_pull_seed424242_2026-10-08.json)
- [Detailed G1-10 and two-order analysis](results/run_g1_10_granite350m_gaussian_pull_seed424242_summary.md)
- Preregistration and immutable code record: `EXPERIMENT.md`

No additional jobs launched.

---


## 40. S1-1: Gridward interpolation fails its Smol cross-family transfer despite much lower code-flip rate

Following two preregistered positive direct-Q3 gridward-pull results on
Granite orders 271828 and 424242, we used an independently established,
validation-selected **SmolLM2-360M-Instruct** training protocol to test
the same **nine 10% toward-current-ternary-grid master-weight pulls**.

The frozen S1-1 setup, approved before implementing or launching,
had **two direct-ternary arms**: tuned **D** versus **P** with the
additional `W ← 0.9W+0.1Q3(W)` after steps100,200,...,900.
Smol seed **1729**, the historical v9 direct-tuning seed, received
its proven **100-step warmup to 1e-3 and cosine decay to 1e-4** over
1,200 steps. Unlike Granite, Smol's direct baseline uses *continuous*
Adam without a step300 reference-scale restoration or optimizer reset.
Identical pretrained BF16-rounded FP32 master weights, Smol FP16
autocast/AMP GradScaler, teacher, CE/KL objective, same train chunks,
and hard ternary validation/test were used in both arms. No Gaussian
or intermediate Q9 phase was tested.

The Hugging Face A10G job **COMPLETED** successfully on
2026-10-09 01:18:20 UTC. All structural checks passed; each arm
received 1,200 training-step opportunities, **six identical
AMP-skipped optimizer updates** (hence 1,194 actual updates),
and P performed nine valid pulls. The new D arm exactly
reproduced the v9 historical tuned direct-Q3 test endpoint,
`5.595722187310457`.

| Smol training arm | 8192-token heldout loss | PPL | Teacher top-1 | KL |
|---|---:|---:|---:|---:|
| **D — tuned direct** | **5.595722** | **269.27** | **29.87%** | **2.66403** |
| P — identical + gridward | 5.845746 | 345.76 | 27.28% | 2.93324 |

**P is 0.250024 nats worse**; perplexity increases
**28.41%**. This is a genuine *negative cross-family
result for the exact previously Granite-positive rule*,
not a problem with the baseline, run completion, or seed labels.

The code-transition telemetry makes the negative result more
informative. A dedicated deterministic monitor examined
32,768 weights in 16 layers after each update. Sampled
per-weight-per-step clean-Q3 flip rate decreased from
**0.000435054 (D)** to **0.000093918 (P)**:
**78.41% fewer code flips** with gridward.
Final exact source-code movement fell from
**6.960% (D)** to **3.445% (P)**.

Why the striking difference? The 24-chunk *training-validation*
time course supplies one important clue, without proving causation:

| Checkpoint | D validation NLL | P validation NLL | What happened? |
|---|---:|---:|---|
| 300 | 5.98809 | **5.97667** | P narrowly ahead |
| 600 | 5.74032 | **5.67064** | P ahead |
| 900 | **5.41510** | 5.58253 | D overtakes P |
| 1200 | **5.25594** | 5.57738 | D finishes strongly ahead |

At the final training stage, P's sampled code-change counts
were almost zero, whereas D still changed some ternary
codes and improved much more on validation. This pattern
is **compatible with premature overcommitment** under
Smol's high-peak schedule. But the experiment does not
isolate whether *the model architecture*, *the LR/optimizer
reset*, *the pull strength/timing* or their interaction
causes the cross-family sign reversal.

This result is a counterexample to a universal theory
that "stabilizing more weight codes must improve
ternary training." It also reconciles with previous
negative Smol v13 direct-code prototype firming:
more confident commitments can be harmful when
current codes are not the right ones. Those older
interventions were not the same as S1-1 and must
not be treated as exact replications.

Importantly, S1-1 was **direct ternary training**:
the earlier Smol Q9→Q3 **three-of-three** matched-schedule
staging gains are unaffected, and the Granite G1-9/G1-10
two-order gridward benefits remain valid only within
their original conditions. Published WinQ-related
gridward interpolation has prior art; simply transferring
it is not original invention. The increasingly interesting
hypothesis is **when and for which boundary margins a
commitment should be made**, not whether all ternary
code changes should be suppressed.

- Completed HF job: `6ac83c3bfee2c90070171b1a`.
- Script pin: `a5634bce459092a503e7534e6b89b6d3a019548b`.
- [Full raw JSON](results/run_s1_1_smol360m_gridward_direct_q3_seed1729_2026-10-09.json).
- [S1-1 scientific summary](results/run_s1_1_smol360m_gridward_direct_q3_seed1729_summary.md).
- Preregistration and run audit in `EXPERIMENT.md`.

**No follow-up GPU jobs were launched; no retuning against the
viewed heldout Smol result was undertaken.**

---

## T1 and C1 follow-up: timing and earlier transition (October 8, 2026 EDT)

**T1 diagnostic** ([summary](results/run_t1_q9_discovery_timing_seed1729_summary.md), [raw JSON](results/run_t1_q9_discovery_timing_seed1729_2026-10-08.json)) compared Q9 vs direct Q3 **projected ternary assignment masks** at equal preparation steps 0,50,100,150,200,250,300, on seed/order1729 under matched v10 training. The eventual step300 disagreement mask (6.4581% of weights) was reproduced exactly; only **51.35%** of its locations were already in the step200 disagreement mask and **69.40%** at step250 (precision **77.77%** at 250). This is **retrospective descriptive geometry**; knowing the future mask is not a viable online selection policy. Both types of disagreement emerged progressively: Q9-specific moves versus direct-Q3 moves that Q9 avoids. The experiment did not determine their individual downstream causal value.

**C1 switch experiment** ([frozen exploratory prereg](research_log/phase_c_q9_switch250_vs300_seed1729_prereg_2026-10-08.md), [summary](results/run_c1_smol360m_q9_switch250_vs300_seed1729_summary.md), [raw JSON](results/run_c1_smol360m_q9_switch250_vs300_seed1729_2026-10-08.json)) actually trained all three equal-1200-scheduled-step arms on seed1729:

| Arm | Global training path | Held-out NLL | PPL |
|---|---|---:|---:|
| D | Direct Q3 for all 1200 | 5.595722 | 269.272 |
| S250 | Q9 steps 1–250, Q3 steps 251–1200 | 4.930515 | 138.451 |
| S300 | Q9 steps 1–300, Q3 steps 301–1200 | **4.900985** | **134.422** |

The **250-step** staged preparation recovered **95.75% of S300's improvement over D**, with a modest **+0.02953 nats** extra loss relative to S300. It passed both criteria specified before launching the C1 job: loss penalty <=0.07 and benefit retention >=90%. Historical D and S300 held-out results reproduced *exactly*, and the job's construction checks passed. Successful HF C1 job `6ac8597afee2c90070172c79`, pinned script SHA `5577a771ed57409283690097a58bae0c8966fdb1`, completed 2026-10-09 03:24:11 UTC.

**Narrow interpretation:** changing 50 late Q9 steps into 50 extra Q3 steps kept nearly all the original finite-budget advantage on this studied seed. This does not imply all useful assignment decisions were discovered by step250, that fewer than 250 Q9 steps are equally good, or that switch250 is globally optimal. C1 was **informed by observed T1 masks on seed1729**, the same familiar evaluation slice was used as before, and other seeds/datasets remain untested. The additional Q3 training and different optimizer reset time are part of the recipe change, not isolated causally. Actual optimizer updates were D1194 and S250/S3001190 after recorded AMP skips.

**Scientific next question, not yet tested:** test the exact switch250 method on fresh orders without parameter changes, or isolate the causal contribution of **Q9-only changes** versus **direct-only changes Q9 prevents**. No automatic follow-up GPU jobs or retuning authorized.

---

## M1 — identical 900-step Q3 continuation after 250 vs 300 Q9 steps (October 9, 2026 UTC)

The C1 equal-total-budget result raised a clean confound: Q9(250) had **950** subsequent Q3 training steps, while Q9(300) had **900**. M1 instead takes snapshots from a **single shared Q9 seed1729 preparation trajectory** at steps250 and300 and gives both states **identical** Q3 training batches, learning-rate sequence, optimizer reset and original-source Q3 scales for **900 steps**. This isolates the downstream Q3 operator, **not total training compute** (1150 vs1200 opportunities).

| Arm | Q9 prep | Common Q3 continuation | Held-out NLL | PPL |
|---|---:|---:|---:|---:|
| P250_900 | 250 | 900 | 4.950432 | 141.236 |
| P300_900 | 300 | 900 | **4.900985** | **134.422** |

**Observed gap:** the shorter Q9 preparation is **0.0494467 nats/token worse** when Q3 continuation is held fixed. Both arms had **897 successful Q3 updates and 3 AMP skips**, all preregistered construction/reproducibility checks passed, and P300 reproduces the historical endpoint exactly. The Q9 difference is therefore associated with better downstream quality beyond any advantage from giving the early-switch model extra Q3 steps. The comparison remains exploratory (same familiar order/test).

M1 further compares individual ternary weights: after Q9 prep, t250 and t300 have **6.245M differing projected Q3 codes (1.9853% of the 314.57M targeted weights)**. After the identical continuation, early-prepared Q3 **adopts the t300 Q9-preparation choice at 44.16%** of these locations; both final models **agree at 60.0%** of these original disagreement sites and **93.86%** of all sites. P300 itself retains the t300-prep code only **59.30%** at those contested positions. These are **per-position descriptive comparisons**; code matches do not prove causal importance, and the t300-preparation code is not guaranteed to be optimal.

Combined with C1, the defensible conclusion is: **some additional Q9 preparation improves quality at matched subsequent Q3 effort, but switching earlier with more Q3 effort retains much of the finite-budget benefit**. No conclusion about best asymptotic solution, truly unseen evaluation, cross-family portability or healthy generation follows. New evaluation data, independent training orders and a substantial paired longer-horizon test remain priorities. Do not treat the C1-vs-M1 difference as the isolated marginal value of 50 Q3 steps because their Q3 continuation begins with different data/LR indices.

**M1 evidence:** [full raw JSON](results/run_m1_equal_q3_continuation_seed1729_2026-10-09.json) · [summary](results/run_m1_equal_q3_continuation_seed1729_summary.md) · [pre-registered protocol](research_log/m1_equal_q3_continuation_seed1729_prereg_2026-10-08.md) · [HF job 6ac86712fee2c900701734bb](https://huggingface.co/jobs/codeflash85/6ac86712fee2c900701734bb). Job completed with `valid_for_science=true`. No new jobs launched.

---

## F1 — FineWeb-Edu pretraining-style QAT data: paired Q9→Q3 gain survives (October 9, 2026 UTC)

The user proposed evaluating our staged quantization method with data closer to SmolLM2's original pretraining distribution. We tested **FineWeb-Edu sample-10BT**, a public subset of **one ingredient** in SmolLM2 pretraining, without claiming to reconstruct the original multi-dataset pretraining or Instruct tuning. **F1 is an exploratory one-seed test of QAT-data distribution transfer, not cross-architecture transfer or full pretraining-data replication.**

One paired job compared direct Q3 all1200 training opportunities with Q9 for300 then Q3 for900. Both used the same BF16-rounded SmolLM2-360M-Instruct source, trainable target linear master weights/rowwise scales, frozen other modules, fixed teacher, 35% CE+65% KL, AdamW and same v10 global LR schedule. Training chunks were fixed and paired between treatments.

| Final evaluation | Direct Q3 NLL ↓ | Staged Q9→Q3 NLL ↓ | Advantage D−S ↑ | Direct PPL | Staged PPL |
|---|---:|---:|---:|---:|---:|
| FineWeb-Edu doc-heldout (primary) | 5.766660 | **4.996255** | **+0.770405** | 319.469 | **147.858** |
| WikiText2 validation (secondary) | 6.558239 | **5.695225** | **+0.863013** | 705.029 | **297.444** |

**HF job [6ac86eb0fee2c900701738dd](https://huggingface.co/jobs/codeflash85/6ac86eb0fee2c900701738dd) COMPLETED on 2026-10-09 04:48:30 UTC**, scientific `valid_for_science=true` with **5/5 design checks passed**. Pinned source commit `64dcd7f20240b4c62e6ecac8df70a1336a57bb14`. D actual training updates1194 (6 skipped), S1192 (8 skipped), from 1200 scheduled each.

**Evaluation split caution:** FineWeb sample used first 340 streamed docs, with hash-bucket document-disjoint sets: **180 QAT train docs** supplying1200 chunks, **3 train-dev docs** supplying24 chunks, **21 heldout docs** supplying128 chunks. The QAT train and heldout data did not share source document IDs, but the pretrained language model **may have already encountered some sampled FineWeb documents during pretraining**. Token chunks within a document are correlated. This is a positive signal on a small source-distribution sample, not comprehensive evidence on independently unseen pretraining content. WikiText2 validation data were not used to choose F1 hyperparameters and were distinct from prior WikiText test samples.

The stage transition incurs a large one-step Q3 projection shock on the tiny FineWeb train-split dev set (native Q9 NLL5.428179; initial-scale Q3 NLL6.942240); staged Q3 then recovers strongly. This is consistent with the earlier optimization-path hypothesis, but does **not** prove code assignment causality, model quality after generation or permanence of the gain at long horizons.

**L1 longer-horizon WikiText job is separate** ([job 6ac86e1f095c578089301e53](https://huggingface.co/jobs/codeflash85/6ac86e1f095c578089301e53)) and was still running during F1 archival. Results here neither imply nor predict whether direct Q3 catches up by6000 steps.

[F1 full result JSON](results/run_f1_fineweb_edu_direct_vs_staged_seed1729_2026-10-09.json) · [F1 detailed scientific summary](results/run_f1_fineweb_edu_direct_vs_staged_seed1729_summary.md) · [frozen preregistration](research_log/f1_fineweb_edu_qat_seed1729_prereg_2026-10-09.md) · [live state](CURRENT_STATE.md). No additional GPU jobs were launched to analyze F1.

## Latest completed studies and external mechanism-review follow-up (2026-10-09 UTC)

**All F1/F2/F3 FineWeb seeds and L1 WikiText6000 job have COMPLETED**, all final scientific validity flags passed. No research jobs currently active in the last verified HF listing. **FineWeb aggregate:** Q9→Q3 beats direct Q3 in **3/3 orders**, mean held-out advantage **+0.763898 nats/token** (range +0.742353..+0.778936), WikiText validation mean +0.801917. Identical 21 FineWeb evaluation documents across orders, *not independent validation samples*. [3-seed aggregate](results/run_f1_f3_fineweb_edu_three_seed_aggregate_summary.md) · [F2 raw](results/run_f2_fineweb_edu_direct_vs_staged_seed271828_2026-10-09.json) · [F3 raw](results/run_f3_fineweb_edu_direct_vs_staged_seed424242_2026-10-09.json).

**L1 six-thousand-step:** [result summary](results/run_l1_6000step_wikitext_durability_seed1729_summary.md) · [raw](results/run_l1_6000step_wikitext_durability_seed1729_2026-10-09.json). On fixed WikiText validation, staged advantage declines **+0.716087 @1200 → +0.616040 nats/token @6000** (nonmonotonic), so direct Q3 hasn't caught up by6000. Reproduced both step1200 historical tests exactly. This is one long training order, not a convergence proof.

**Separate independent review of unresolved mechanism controls:** [frozen-scale depth / Q9 range match / FineWeb sham switch / margin-matched positions / common-scale code comparisons / document-linked evidence](research_log/2026-10-09_external_technical_review_scale_range_sham_controls.md). Reviewing `smollm2_v12_code_identity_position.py` confirms scale-gradient confounding is real; v13's nearly-equal fixed/learned alpha *measurement* of code survival does not eliminate a scale-*training-gradient* mechanism. Next recommended controls **G1 frozen-vs-learned-scale depth** and **S1 FineWeb direct-Q3 sham step300 reset**, followed by range-matched Q9. No control GPU jobs launched as part of this review. [Authoritative current state](CURRENT_STATE.md) takes precedence over historical “running” sections.

---

