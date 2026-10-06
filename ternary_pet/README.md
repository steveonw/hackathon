# Ternary Pet Experiments

Small, reproducible experiments around converting pretrained language models
to ternary weights.

## Current pet

**HuggingFaceTB/SmolLM2-360M-Instruct** (~362M parameters).

## Evolution of the hypothesis

- **v1:** naive 27 -> 9 -> 3 staging; unstable and worse than direct QAT.
- **v2:** strict nested ancestry; stable, but hard commits effectively trapped
  the ternary codes.
- **v3:** FP32 shadow weights + transition schedules; direct and up/down nearly
  tied when ternary training budget was matched.
- **v4:** persistent FP32 shadows, preserved optimizer state, absmean ternary
  initialization, frozen non-quantized components, and exact code-flip logging.

## First positive staged result: v4

On the fixed 8192-token WikiText-2 evaluator:

| Variant | PPL ↓ | Top-1 agreement ↑ | KL ↓ |
|---|---:|---:|---:|
| Direct ternary, 1200 steps | 666.66 | 22.84% | 3.546 |
| **9-state 300 -> ternary 900** | **286.97** | **30.49%** | **2.716** |
| Direct ternary, 900 steps | 670.74 | 22.96% | 3.508 |

The equal-total-compute comparison is the key one: both direct-1200 and staged
300+900 process the same 1200 training chunks in the same global order. The
staged model gets only one special treatment: its first 300 updates use a
9-state forward representation instead of ternary.

The 9-state preparation reduced the first ternary-step loss from 17.31 to 6.59,
a ~62% smaller transition shock.

Exact flip-rate logging also confirmed that the quantized codes themselves were
moving, so this result is not the frozen-weight artifact seen in earlier runs.

## Important limitation

This is one seed. The staged model is **less damaged**, not yet healthy: its
sample generations remain repetitive and degenerate. The next scientifically
useful move is replication, not another new recipe.

## Repository layout

- `smollm2_staircase.py` — v1
- `smollm2_nested_v2.py` — v2
- `smollm2_transition_v3.py` — v3
- `smollm2_v4_fliprate_9to3.py` — v4
- `EXPERIMENT.md` — protocols
- `results/` — immutable metrics and summaries

## Interpretation rule

A staged schedule counts as interesting only if it beats a fair direct control
under matched data/compute and its quantized codes are demonstrably learning.
