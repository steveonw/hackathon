# Ternary Pet Experiments

Small, reproducible experiments around staged ternary quantization of language
models.

## Current hypothesis

Instead of jumping directly from high precision to ternary weights, test a
base-3-aligned staircase:

```text
full precision -> 27 states -> 9 states -> 3 states
                  3^3          3^2        3^1
```

The main comparison is always against a direct `full precision -> 3` control.

## Current pet

**HuggingFaceTB/SmolLM2-360M-Instruct** (~362M parameters).

It is small enough for inexpensive repeated GPU runs, but large enough to make
before/after language behavior measurable.

## Repository layout

- `smollm2_staircase.py` — exact v1 experiment script
- `EXPERIMENT.md` — hypothesis, controls, quantizer definition, and v2 plan
- `results/` — immutable run outputs and summaries

## Completed v1 run

Hugging Face Job: `6ac52475404719ba37661c8b`

Result: **no staged advantage demonstrated in v1**. Both raw ternary paths collapsed; direct QAT recovered more validation likelihood, while staged QAT retained slightly more top-1 baseline agreement but remained qualitatively broken. See `results/run_v1_summary.md`.

Configuration:

- hardware: `t4-small`
- timeout: 30 minutes
- model: `HuggingFaceTB/SmolLM2-360M-Instruct`
- direct QAT: 30 optimizer steps at 3 levels
- staged QAT: 10 steps each at 27, 9, and 3 levels

## Interpretation rule

A staged result is interesting only if it beats a fair control. We will not
treat a prettier anecdotal generation as evidence if perplexity/token behavior
does not support it.

Also, v1 uses independently rescaled 27/9/3 quantizers. The stricter
**nested ternary ancestry** version is reserved for v2 and is documented in
`EXPERIMENT.md`.
