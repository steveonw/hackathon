# R8 CPU stage capability evaluator ready — 2026-10-10

The two authorized GPU R8 jobs completed and independently read-back-verified all 8 checkpoints. CPU evaluation runner is published as [r8_cpu_stage_capability_eval.py](../r8_cpu_stage_capability_eval.py) pinned source **`c6c6f91bc25fbc22540bcf9f0d4997892ac24e0f`**. CPU preflight HF `6acadead095c578089319093` emitted `R8_STAGE_EVAL_CPU_PREFLIGHT_OK`; checks script syntax plus 4-bit Q9 and 2-bit Q3 synthetic decoding. **Full private model reconstruction and inference not yet executed.**

Each family job loads the pinned BF16-rounded original C0 and independently SHA256-verifies four private stage archives C1 Q9, C2 Q3, C3 Q3, C4 Q3. It reconstructs every quantized linear into CPU float32 weights (Q9 hybrid mapping: interior k/6, extreme ±1 times alpha; Q3 k/1.5 times alpha) with unchanged nonquantized base weights. It runs the 40 frozen R7 prompts using prior family-specific chat formatting and deterministic greedy max64 decode at each of the five stages (200 records/family), calculates corrected strict and permissive arithmetic/instruction/extraction scores, and continuation unique bigram ratios. Emits `R8_EVAL_RECONSTRUCTED`, `R8_EVAL_RECORD` and terminal `R8_EVAL_FINAL`. The already-used R7 prompts are **diagnostic reused**, not fresh independent holdout; avoid treating them as untouched evaluation.

Use local authenticated Windows CLI because dataset is private:

```powershell
hf jobs uv run --flavor cpu-basic --timeout 120m --secrets HF_TOKEN "https://raw.githubusercontent.com/steveonw/hackathon/c6c6f91bc25fbc22540bcf9f0d4997892ac24e0f/ternary_pet/r8_cpu_stage_capability_eval.py" -- --family granite
hf jobs uv run --flavor cpu-basic --timeout 120m --secrets HF_TOKEN "https://raw.githubusercontent.com/steveonw/hackathon/c6c6f91bc25fbc22540bcf9f0d4997892ac24e0f/ternary_pet/r8_cpu_stage_capability_eval.py" -- --family smol
```

CPU run time unknown; 120m is a limit, not a guaranteed duration. On completion fetch full HF logs with `tail:500`, require exactly 4 reconstruction markers/family, 200 records/family, `valid:true`, correct stage names and SHA hashes. Archive stage-by-stage paired results and limits. No more GPU training authorized or required.
