# Run v5 summary — 5x training-volume test

**Hugging Face Job:** `6ac56d99fbc85ba6823ba03f`  
**Pinned code:** `23157f7ce03f6bbbf944806574b18c133843f049`  
**Model:** `HuggingFaceTB/SmolLM2-360M-Instruct`  
**Hardware:** A10G small  
**Training volume:** 768k tokens per treatment

## Main result

The 9 -> 3 advantage **survived** at 5x the v4 training volume, but it became
substantially smaller.

| Variant | Held-out loss ↓ | PPL ↓ | Teacher top-1 ↑ | KL ↓ |
|---|---:|---:|---:|---:|
| BF16 source | 3.6854 | 39.86 | 99.68% | ~0 |
| Direct 6000@3 | 4.4463 | 85.31 | 41.44% | 1.522 |
| **1500@9 -> 4500@3** | **4.2493** | **70.05** | **43.64%** | **1.330** |

Paired staged advantage:

- **0.197 nats/token** lower held-out loss;
- **17.88%** lower perplexity;
- **+2.20 percentage points** teacher top-1 agreement;
- **12.61%** lower KL;
- closes **25.89%** of the remaining direct-vs-BF16 held-out loss gap.

## Compared with v4/v4b

The earlier three-ordering mean staged effect was **0.765 nats/token**.
At 5x training volume it is **0.197 nats/token**.

So the measured loss advantage is about **74% smaller** than the earlier mean.

This strongly suggests that a substantial part of the v4/v4b benefit was an
**early-optimization/head-start effect**, although staging still retains a
measurable advantage at 6000 total updates.

This result is compatible with two possibilities that require longer or better
controlled experiments to distinguish:

1. direct ternary continues catching up and the gap eventually disappears;
2. staging keeps a smaller but persistent asymptotic advantage.

v5 alone cannot distinguish them.

## Transition shock still replicates

Direct ternary first-step mixed loss: **13.105**.

After 1500 9-state updates, the first ternary step was **6.501**.

That is a **50.39% reduction** in immediate ternary-entry shock, almost exactly
in line with the ~51% mean shock reduction from v4/v4b.

So the transition-smoothing effect remains highly stable even though the final
held-out advantage shrank.

## Code movement

- direct ternary after 6000 steps: **3.17%** net displacement from initial
  ternary codes;
- 9-state phase after 1500 steps: **5.03%** net 9-state code displacement;
- staged ternary phase after 4500 steps: **2.51%** net displacement from the
  ternary-stage start.

These figures are not directly comparable across codebooks. In particular,
v5 still does not measure how much the 9-state phase changes the *eventual
ternary projection* `Q3(W_after_9) vs Q3(W_initial)`, which external reviewers
correctly identified as an important missing diagnostic.

## Qualitative generations

Neither treatment is a healthy assistant yet.

Direct produced one somewhat coherent story fragment but also heavy repetition.
The staged model still produced strongly repetitive/degenerate outputs on the
fixed prompt set.

Therefore the held-out metric advantage has **not yet translated into a clear
generation-quality advantage**.

## Interpretation

v5 weakens the strongest version of the staircase hypothesis.

The data now support:

> 9-state preparation provides a large early optimization advantage and still
> leaves a smaller measurable benefit after 5x more training.

They do **not** yet support:

> 9-state preparation reaches a permanently better ternary basin.

The next highest-value test is not another large scale-up. It is a small causal
dissection of the transition: separate the contribution of staged FP32 master
weights, learned scales, and Adam state, while also giving direct ternary a
stronger LR/warmup baseline.
