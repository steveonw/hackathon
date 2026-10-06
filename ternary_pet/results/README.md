# Results

Raw machine-readable and summarized results from each remote run belong here.

## Run registry

| Run | Model | Hardware | Runtime | Status | Purpose |
|---|---|---|---:|---|---|
| v1 / HF Job `6ac52475404719ba37661c8b` | SmolLM2-360M-Instruct | T4 small | ~1m40s | completed | naive direct/staged PTQ + tiny QAT |
| v2 / HF Job `6ac527e5404719ba37661dc9` | SmolLM2-360M-Instruct | T4 small | ~8m05s | completed | strict nested 27->9->3 ancestry + distillation |
| v3 / HF Job `6ac52df9404719ba37661fa1` | SmolLM2-360M-Instruct | T4 small | ~22m35s | completed | direct vs soft vs FP32-prep transitions |
| v4 / HF Job `6ac539b5404719ba376621a2` | SmolLM2-360M-Instruct | T4 small | ~31m19s | **completed** | persistent-shadow 9->3 with code-flip diagnostics |

## v4 headline

The staged **9 -> 3** route is the first staged treatment to beat its clean
direct control:

- staged 300@9 + 900@3: **PPL 286.97**, **30.49% agreement**, **KL 2.716**
- direct 1200@3: PPL 666.66, 22.84% agreement, KL 3.546

The staged route reduced perplexity by ~57% and reduced the ternary-entry
training-loss shock by ~62%.

Exact code-flip instrumentation confirmed that the quantized weights were
actually changing, and absmean initialization brought the ternary zero fraction
down to ~31.7%.

This is positive evidence, not yet a conclusion: v4 is one seed and all ternary
variants still generate degenerate text.

Files:

- `run_v1_2026-10-06.json`
- `run_v1_summary.md`
- `run_v2_2026-10-06.json`
- `run_v2_summary.md`
- `run_v3_2026-10-06.json`
- `run_v3_summary.md`
- `run_v4_2026-10-06.json`
- `run_v4_summary.md`

Negative and failed runs are retained so the experiment history stays auditable.
