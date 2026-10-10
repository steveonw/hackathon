# R6 reconstruction H loss parity — confirmed from original remote CPU logs

**2026-10-10 UTC.** Hugging Face job logs fetched directly from the connected account (not user transcription). Both secure, CPU-only inference reconstruction plus fixed four-document IMDb parity jobs **COMPLETED**. No GPU training.

- **Granite seed170141:** [HF job 6ac9e829095c57808930fd96](https://huggingface.co/jobs/codeflash85/6ac9e829095c57808930fd96), source pinned commit `ddcb84cc4566b989429d5cd131199f264c5b6d81`. Selected R6 test rows 264, 383, 522, 620 (512 tokens each). CPU FP32 reconstructed H mean NLL **6.681938976049423** versus original R6 GPU BF16-autocast H mean NLL **6.682182490825653**; mean delta **−0.0002435147762298584** nats/token; largest per-document absolute delta **0.0007123947143554688**.
- **SmolLM2 seed190027:** [HF job 6ac9e841fee2c9007018468b](https://huggingface.co/jobs/codeflash85/6ac9e841fee2c9007018468b), same pinned source. Test rows 119, 231, 264, 282. CPU FP32 reconstructed H mean NLL **6.165739893913269** versus R6 GPU FP16-autocast H mean NLL **6.165852457284927**; mean delta **−0.0001125633716583252** nats/token; largest per-document absolute delta **0.0001367330551147461**.

## Per-document detail

| Model | Row | Original job H NLL | Reconstructed H NLL | Reconstructed minus original |
|---|---:|---:|---:|---:|
| Granite | 264 | 6.4789392948150635 | 6.478226900100708 | −0.0007123947143554688 |
| Granite | 383 | 6.4392149448394775 | 6.439190626144409 | −0.000024318695068359375 |
| Granite | 522 | 6.928325295448303 | 6.928314447402954 | −0.000010848045349121094 |
| Granite | 620 | 6.882250428199768 | 6.882023930549622 | −0.00022649765014648438 |
| Smol | 119 | 6.749972224235535 | 6.749850511550903 | −0.00012171268463134766 |
| Smol | 231 | 6.1365262269973755 | 6.136430263519287 | −0.0000959634780883789 |
| Smol | 264 | 5.878950595855713 | 5.878813862800598 | −0.0001367330551147461 |
| Smol | 282 | 5.897960782051086 | 5.897864937782288 | −0.00009584426879882812 |

## Interpretation and limits

Reconstruction preserves per-document loss to within about 0.0007 nats/token, strongly arguing against a **major** compact-code reconstruction fault for these document windows. The remaining small deltas are consistent with distinct CPU FP32 vs original GPU autocast precision but not proven to be caused entirely by precision. This checks **four of the preregistered 32 evaluation documents per model**, not full-corpus numeric identity, and compares NLL, not full logits/activations. Both jobs also emitted `R6_INFERENCE_RECONSTRUCTION_OK` again.

Earlier one-prompt greedy generation remained degenerate (periods/newlines); NLL parity strongly suggests the degenerate text is not solely an accidental weight unpacking failure. Useful conversational generation is **not** demonstrated. Future work could test proper chat templates (SmolLM2-Instruct), prompts from the fine-tuning domain, multiple decoding settings, and generation-side vs base model comparisons before diagnosing quality. No new GPU experiment was launched.
