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
