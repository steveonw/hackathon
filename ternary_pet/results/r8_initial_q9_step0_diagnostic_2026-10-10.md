# R8 initial, untrained hybrid Q9 step-zero diagnostic

2026-10-10 EDT / Oct 11 UTC. This test runs CPU-only with **no new training** and no private snapshot access. [Script](../r8_cpu_step0_q9_diagnostic.py) pinned SHA `119114d5277853f5cbd35f1c4e39a9d8ebc84a9c`, CPU quantizer logic preflight HF `6acaececfee2c9007019016e` passed. Reuses R7's 40 frozen *diagnostic, not new independent* prompts, with corrected permissive answer grading and unique-bigram repetition metric. Initial Q9 weights are computed from BF16-rounded pinned original base weights using the H nine-state quantizer initialization: per-row alpha derived from 1.5×meanabs, k=round(clamp(w/alpha,-.99,.99)×4), inner k/6 and outer ±1, multiplied by alpha.

## Confirmed Granite

CPU job [6acaed16fee2c90070190189](https://huggingface.co/jobs/codeflash85/6acaed16fee2c90070190189) **COMPLETED** and printed `R8_EVAL_FINAL valid:true`, all 80 greedy responses, 168 original target linears quantized. Base permissive scores arithmetic10/10, instruction9/10, extraction10/10, total **29/30**. **Untrained Q9_step0: 0/30** across three categories. Mean unique-bigram ratio on 10 continuation cases base **0.6845997973657548**; untrained Q9 **0.023809523809523808**. This shows immediate massive functional degradation from the initial Q9 quantizer **before any optimizer update** on this frozen battery. It strongly prioritizes quantizer initialization/scale/codebook as a cause over damage solely induced during steps1–300. It does not prove that Q9 quantization fundamentally cannot retain capabilities with a different initialization.

## Smol status during first capture

CPU job [6acaed17fee2c9007019018b](https://huggingface.co/jobs/codeflash85/6acaed17fee2c9007019018b) **RUNNING** at time of first archive; base responses and initial Q9 prompts showed repetitive poor outputs, but **no terminal `R8_EVAL_FINAL` yet**. Do not claim scored Smol outcome until terminal check.

## Next research

After Smol completes, compare both complete stage0 curves (base→Q9@0→Q9@300→Q3@300→Q3@600→Q3@1200). Before further GPUs, investigate alpha scaling, layerwise perturbation and quantization shock; constrain experiments to CPU inference and new frozen holdout for prospective claims. No new GPU training launched.
