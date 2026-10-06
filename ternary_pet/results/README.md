# Results

Raw machine-readable and summarized results from each remote run belong here.

## Run registry

| Run | Model | Hardware | Runtime | Status | Purpose |
|---|---|---|---:|---|---|
| v1 / HF Job `6ac52475404719ba37661c8b` | SmolLM2-360M-Instruct | T4 small | ~1m40s | completed | naive direct/staged PTQ + tiny QAT |
| v2 / HF Job `6ac527e5404719ba37661dc9` | SmolLM2-360M-Instruct | T4 small | ~8m05s | **completed** | strict nested balanced-ternary ancestry + distillation |

## v2 headline

- Raw direct ternary and raw nested 27 -> 9 -> 3 were **exactly identical**.
- The strict nested 9-level stage was stable, unlike v1.
- Direct ternary QAT still beat staged QAT by a large margin on held-out loss,
  perplexity, top-1 agreement, and KL to the original model.
- Giving staged QAT the same final ternary-step budget did not change its final
  metrics or generated behavior.
- Result: **strict nested ancestry is not enough under hard commits and frozen
  scales.**

Files:

- `run_v1_2026-10-06.json`
- `run_v1_summary.md`
- `run_v2_2026-10-06.json`
- `run_v2_summary.md`

Do not replace failed or negative runs. Keep them so the experiment history
remains auditable.
