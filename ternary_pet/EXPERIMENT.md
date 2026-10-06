# Experiment protocol

## Question

Does a ternary-aligned reduction in available weight states,

```text
full precision -> 27 levels -> 9 levels -> 3 levels
```

preserve more model behavior than a direct jump,

```text
full precision -> 3 levels
```

when both routes start from the same pretrained model?

## Model

- `HuggingFaceTB/SmolLM2-360M-Instruct`
- ~362M parameters
- Llama-family causal transformer
- Apache-2.0 license

## v1 treatments

| Treatment | Path | Recovery budget |
|---|---|---:|
| Baseline | full precision | 0 |
| Direct PTQ | FP -> 3 | 0 |
| Staged PTQ | FP -> 27 -> 9 -> 3 | 0 |
| Direct QAT | FP -> 3 | 30 optimizer steps |
| Staged QAT | FP -> 27 -> 9 -> 3 | 10 + 10 + 10 optimizer steps |

The QAT comparison intentionally keeps **total optimizer steps equal**.

## Metrics

- Cross-entropy loss
- Perplexity
- Top-1 next-token agreement with the original model
- Fixed greedy chat generations
- Training-loss traces and elapsed recovery time

## Quantizer used in v1

The v1 quantizer is symmetric and per-output-channel for linear-layer weights.
For an odd number of levels `L`:

```text
k = (L - 1) / 2
scale = max(abs(weight_row)) / k
q = clamp(round(weight / scale), -k, +k)
weight_quantized = q * scale
```

The LM head is left unquantized in v1.

## Important control / limitation

**v1 is not yet the strongest form of the ternary-ancestry hypothesis.**

At each stage, the scale is recomputed from the current tensor. Therefore the
27-, 9-, and 3-level alphabets are related by state count, but they are not a
strict nested codebook where every 27-level state has a predetermined 9-level
parent and every 9-level state has a predetermined ternary parent.

That stricter nested experiment should be v2. It is the cleaner test of whether
preserving a base-3 "ancestry" itself helps optimization.

## Planned v2 controls

1. Strict nested 27 -> 9 -> 3 codebook.
2. Direct ternary with the **same number of final ternary recovery steps** as
   staged ternary, in addition to equal-total-compute comparison.
3. More evaluation text and a standard public perplexity slice.
4. Hidden-state/logit KL measurements.
5. Layer-family ablations: attention, MLP, embeddings/LM head.
6. Multiple random seeds before drawing conclusions.

## Reproducibility

The first remote run was launched as Hugging Face Job:

`6ac52475404719ba37661c8b`

with a 30-minute timeout on `t4-small`.
