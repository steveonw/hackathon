# R8 step-zero hybrid Q9 capability diagnostic — CPU-only jobs launched

2026-10-10 US Eastern (2026-10-11 UTC). **Research question:** Do original pretrained models lose capability immediately upon initial H-Q9 quantization, before *any* training? This follows successful five-stage R8 tests where both model families already score 0/30 on the frozen capability battery at Q9@300.

Source: [r8_cpu_step0_q9_diagnostic.py](../r8_cpu_step0_q9_diagnostic.py), pinned commit `119114d5277853f5cbd35f1c4e39a9d8ebc84a9c`. Derives the initial row-wise per-linear `alpha=1.5*mean(abs(w))` through the original inverse-softplus and softplus implementation; original BF16-roundtrip, Q9 rounded code `k=round(clamp(w/alpha,-0.99,0.99)*4)` and hybrid `alpha*(k/6 + sign(k)/3 for abs(k)==4)`. Reconstructs weight-tensor inference at initial step0 without training or STE gradient, preserving unchanged base parameters. Tests 40 original fixed diagnostic R7 prompts with family-specific format on C0 pinned high-precision source and new Q9_step0, 80 greedy records/model, corrected permissive scores, unique-bigram ratio.

CPU AST, token-code and model-agnostic quantizer preflight completed: `R8_STEP0_CPU_PREFLIGHT_OK` on HF `6acaececfee2c9007019016e`. This is not an end-to-end model inference verification.

Both public-source-only CPU jobs launched directly via HF connector (no private HF token needed):
- Granite: `6acaed16fee2c90070190189`, https://huggingface.co/jobs/codeflash85/6acaed16fee2c90070190189
- Smol: `6acaed17fee2c9007019018b`, https://huggingface.co/jobs/codeflash85/6acaed17fee2c9007019018b

**State at writing: scheduling; not complete.** No GPU training, no checkpoint reupload, and no private-secret intervention. Verify `R8_EVAL_FINAL valid:true`, record count80, compared C0 against Q9_step0, both families; fetch full job logs with tail500 and inspect exact arithmetic/instruction/extraction and continuation repetition. If initial Q9 fails before any updates, prioritize scale/codebook/initialization. If initial Q9 retains ability, investigate first300 optimization updates. Reused R7 prompts are diagnostics not unseen scientific holdout.
