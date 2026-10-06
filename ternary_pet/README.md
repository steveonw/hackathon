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


## Active v4 diagnostic

Hugging Face Job: `6ac539b5404719ba376621a2`

v4 directly addresses the main confounds found after reviewing v2/v3:

- persistent FP32 shadow weights;
- no hard commit at 9 -> 3;
- optimizer state survives the stage transition;
- non-quantized parameters are frozen;
- BitNet-like absmean ternary initialization;
- exact code-flip logging;
- fixed 8192-token WikiText-2 test slice.

The LR diagnostic tested 2e-5, 5e-5, and 1e-4. All three caused real ternary
code changes. The script selected **1e-4** from validation loss; after 100
diagnostic steps, about 0.38% of all targeted codes differed from their initial
state, with some early attention projections above 1%.

The main comparison is direct ternary (1200), persistent-shadow 9->3
(300+900), and direct ternary (900).
