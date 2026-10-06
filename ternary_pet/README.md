# Ternary Pet Experiments

Experiments on whether a pretrained language model can enter ternary weight
space more gracefully through an intermediate representation.

## Current model

**HuggingFaceTB/SmolLM2-360M-Instruct** (~362M parameters).

## Current headline

v4/v4b established a reproducible 9 -> 3 advantage at small training budgets.
v5 showed that the advantage shrinks substantially with 5x more training.

**v6 now locates the small-budget advantage primarily in the adapted FP32 master
weights.**

### v6 causal transition ablation

| Condition | Loss ↓ | PPL ↓ |
|---|---:|---:|
| Direct 1200@3 | 5.875 | 355.90 |
| Carry all staged state | 5.206 | 182.34 |
| Reset Adam | 5.203 | 181.83 |
| **Prepared masters only; reset scales + Adam** | **5.199** | **181.09** |
| Prepared masters + weight Adam | 5.180 | 177.67 |

Resetting Adam and restoring the original quantizer scales does **not** remove
the benefit. The four staged branches span only **0.026 nats/token**, compared
with a ~0.67-0.69 nat advantage over direct.

## Mechanistic diagnostic

After 300 Q9 updates, using the same fixed diagnostic data:

- Q9 loss: **4.932**
- immediate Q3 projection of the same weights: **9.177**
- original Q3 projection before preparation: **15.478**

So Q9 -> Q3 still causes a large discrete shock, but Q9 training has already
made the eventual ternary projection **6.30 nats/token better** before a single
ternary optimizer update.

With the original quantizer scales held fixed, **0.6755%** of future ternary
assignments changed during Q9 training. The learned-scale value is almost
identical (**0.6769%**).

That is direct evidence that the FP32 masters themselves cross future ternary
decision boundaries during the intermediate stage.

## Current interpretation

The narrow mechanism supported by the experiments is now:

> Intermediate-state QAT can move continuous master weights into a configuration
> whose later ternary projection is substantially less damaging. The early
> benefit is carried mainly by those adapted master weights, not by Adam moments
> or scale calibration.

v5 still shows that direct ternary catches up substantially at larger training
budgets, so this does not establish a permanently better asymptotic basin.

## Repository layout

- `smollm2_staircase.py` — v1
- `smollm2_nested_v2.py` — v2
- `smollm2_transition_v3.py` — v3
- `smollm2_v4_fliprate_9to3.py` — v4
- `smollm2_v5_5x.py` — v5
- `smollm2_v6_transition_ablation.py` — v6
- `replications/` — v4b confirmation
- `results/` — immutable run records and summaries
- `SHAREABLE_RESEARCH_REPORT.md` — external-review report
- `EXPERIMENT.md` — protocol history
