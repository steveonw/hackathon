# Ternary Pet Experiments

Experiments on whether a pretrained language model can enter ternary weight
space more gracefully through an intermediate representation.

## Current model

**HuggingFaceTB/SmolLM2-360M-Instruct** (~362M parameters).

## What we learned

- **v1:** naive 27 -> 9 -> 3 was unstable.
- **v2:** hard nested commits trapped the discrete states.
- **v3:** persistent FP32 shadows and learnable scales made ternary recovery
  much healthier, but transition schedules mostly tied.
- **v4:** with the major confounds removed, 300 steps at 9 states followed by
  900 ternary steps beat a 1200-step direct ternary control.
- **v4b:** that v4 advantage reproduced under two independent shuffled training
  orders.

## Replicated paired result

| Training order | Direct PPL ↓ | 9->3 PPL ↓ | Direct top-1 | 9->3 top-1 |
|---|---:|---:|---:|---:|
| reference v4 | 666.66 | **286.97** | 22.84% | **30.49%** |
| seed 1729 | 352.77 | **178.85** | 25.45% | **33.76%** |
| seed 271828 | 383.56 | **177.20** | 25.44% | **32.08%** |

Across the three paired orderings, staging reduced perplexity by an average of
**53.35%** and improved teacher top-1 agreement by **7.54 percentage points**.

The first ternary-step shock was also smaller after 9-state preparation in all
three runs.

## Working interpretation

The evidence now supports a narrower version of the staircase hypothesis:

> A persistent continuous master weight can use a 9-state (~3.17-bit) forward
> representation to reorganize before the final 3-state / 1.58-bit constraint,
> producing a less damaged ternary model than spending the same total updates
> directly in ternary space.

The mechanism is not "rounding through more steps preserves information."
Without adaptation, staged and direct projection are identical. The apparent
benefit comes from **learning while the intermediate states still exist**.

## Limits

The ternary models are still qualitatively degraded and repetitive. This is one
360M checkpoint, one calibration/evaluation corpus, and only three paired
training orders. It is a reproducible experimental effect, not yet a general
result or a production-ready model.

## Repository layout

- `smollm2_staircase.py` — v1
- `smollm2_nested_v2.py` — v2
- `smollm2_transition_v3.py` — v3
- `smollm2_v4_fliprate_9to3.py` — v4
- `replications/` — v4b scripts, manifests, and confirmation results
- `results/` — immutable run records and summaries
- `EXPERIMENT.md` — protocol history
