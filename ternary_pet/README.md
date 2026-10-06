# Ternary Pet Experiments

Small, reproducible experiments around converting pretrained language models
to ternary weights.

## Current pet

**HuggingFaceTB/SmolLM2-360M-Instruct** (~362M parameters).

## Evolution of the hypothesis

- **v1:** naive 27 -> 9 -> 3 staging. Unstable and worse than direct QAT.
- **v2:** strict nested balanced-ternary ancestry. Stable, but direct QAT still won.
- **v3:** learnable ternary scales + FP32 shadow weights, comparing direct,
  gradual, and "step up then down" transition schedules.

The strongest v3 result is subtle: 200 FP32-master adaptation steps reduced the
immediate ternary-switch shock by ~4%. When that path was then given the same
600 fully ternary steps as direct QAT, the two finished nearly tied: direct had
slightly better perplexity/KL, while up/down had slightly higher teacher
top-token agreement.

No ternary variant is yet qualitatively healthy; the project has not produced a
usable converted checkpoint.

## Repository layout

- `smollm2_staircase.py` — v1
- `smollm2_nested_v2.py` — v2 strict ancestry
- `smollm2_transition_v3.py` — v3 transition schedules
- `EXPERIMENT.md` — experiment definitions
- `results/` — immutable metrics and written summaries

## Next hypothesis

The most justified next route combines the two ideas that reduced transition
shock without hard-committing intermediate weights:

```text
BF16 source
 -> FP32 master adaptation
 -> soft ternary phase-in
 -> full ternary QAT
```

The full-precision shadow/master weights should remain continuous throughout.

## Interpretation rule

A new schedule is interesting only if it beats a fair direct-ternary control
under matched data and clearly stated compute budgets. Anecdotal generations
alone do not count as evidence.
