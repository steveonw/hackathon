# Results

Raw machine-readable and summarized results from each remote run belong here.

## Run registry

| Run | Model | Hardware | Timeout | Status | Purpose |
|---|---|---|---|---|---|
| v1 / HF Job `6ac52475404719ba37661c8b` | SmolLM2-360M-Instruct | T4 small | 30 min | **completed** | baseline vs direct/staged PTQ and equal-compute QAT |

### v1 headline

- Raw direct ternary: collapsed.
- Raw staged 27 -> 9 -> 3: collapsed and slightly worse by perplexity.
- Direct QAT recovered more validation likelihood than staged QAT.
- Staged QAT retained somewhat more top-1 baseline agreement, but generated
  degenerate repeated text.
- Result: **no staged advantage demonstrated in v1**.

Files:

- `run_v1_2026-10-06.json` — compact machine-readable metrics.
- `run_v1_summary.md` — interpretation and v2 recommendations.

Do not replace failed runs. Keep them in the registry so experiment history
remains auditable.
