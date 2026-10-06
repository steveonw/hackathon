# Results

Raw machine-readable and summarized results from each remote run belong here.

## Run registry

| Run | Model | Hardware | Runtime | Status | Purpose |
|---|---|---|---:|---|---|
| v1 / HF Job `6ac52475404719ba37661c8b` | SmolLM2-360M-Instruct | T4 small | ~1m40s | completed | naive direct/staged PTQ + tiny QAT |
| v2 / HF Job `6ac527e5404719ba37661dc9` | SmolLM2-360M-Instruct | T4 small | ~8m05s | completed | strict nested 27->9->3 ancestry + distillation |
| v3 / HF Job `6ac52df9404719ba37661fa1` | SmolLM2-360M-Instruct | T4 small | ~22m35s | **completed** | direct vs soft vs FP32-prep-then-ternary transitions |

## v3 headline

- Learnable per-row ternary scales made recovery dramatically healthier.
- A soft 0->1 transition almost eliminated the first-step quantization shock,
  but did not improve final held-out quality versus direct ternary.
- 200 FP32-master adaptation steps before the ternary switch reduced immediate
  switch loss by about 4%.
- With equal total compute, the up/down path was worse.
- With the **same 600 fully ternary steps** as direct, up/down nearly tied direct:
  direct had slightly better PPL/KL, while up/down had slightly higher top-1
  teacher agreement.
- All ternary variants are still qualitatively broken; no usable converted
  checkpoint has been demonstrated yet.

Files:

- `run_v1_2026-10-06.json`
- `run_v1_summary.md`
- `run_v2_2026-10-06.json`
- `run_v2_summary.md`
- `run_v3_2026-10-06.json`
- `run_v3_summary.md`

Negative and failed runs are retained so the experiment history stays auditable.
