# Ternary Pet Experiments

Experiments on whether a pretrained language model can enter ternary weight
space more gracefully through an intermediate representation.

## Current model

**HuggingFaceTB/SmolLM2-360M-Instruct** (~362M parameters).

## Current result

The 9 -> 3 effect replicated strongly at 1200 total updates, then **survived but
shrunk substantially** when training volume increased 5x.

### v4/v4b: 1200-update regime

Across three paired training orders, 9 -> 3 improved held-out loss by an average
of **0.765 nats/token**, equivalent to about **53.35% lower perplexity** under
those damaged-model conditions.

### v5: 6000-update / 768k-token regime

| Variant | Loss ↓ | PPL ↓ | Teacher top-1 ↑ | KL ↓ |
|---|---:|---:|---:|---:|
| BF16 source | 3.685 | 39.86 | 99.68% | ~0 |
| Direct 6000@3 | 4.446 | 85.31 | 41.44% | 1.522 |
| **1500@9 -> 4500@3** | **4.249** | **70.05** | **43.64%** | **1.330** |

The staged v5 advantage is **0.197 nats/token** / **17.88% lower PPL**. That is
still a clear paired win, but the loss advantage is roughly **74% smaller** than
the v4/v4b mean.

The first ternary training-batch loss after 9-state preparation was **50.39%**
lower than direct's first ternary batch, but those measurements used different
training chunks. Treat this as a transition-entry signal, not yet a clean
same-batch shock measurement.

## Working interpretation

The evidence now favors a more cautious statement:

> 9-state preparation gives a large early optimization/head-start benefit when
> entering ternary space, and a smaller benefit is still present after 5x more
> training.

We do not yet know whether the residual gap is asymptotic or whether direct
ternary eventually catches up.

External review also identified a key unresolved mechanism question: the staged
model carries adapted FP32 masters, learned scales, **and** Adam moments into
ternary. A causal transition ablation is needed to determine where the benefit
actually lives.

## Limits

Both ternary variants remain qualitatively degraded. v5 staged generations are
still repetitive despite better held-out metrics. This remains one 360M model,
one training/evaluation corpus, and an under-tuned direct baseline.

## Repository layout

- `smollm2_staircase.py` — v1
- `smollm2_nested_v2.py` — v2
- `smollm2_transition_v3.py` — v3
- `smollm2_v4_fliprate_9to3.py` — v4
- `smollm2_v5_5x.py` — v5
- `replications/` — v4b confirmation
- `results/` — immutable run records and summaries
- `SHAREABLE_RESEARCH_REPORT.md` — external-review report
- `EXPERIMENT.md` — protocol history
