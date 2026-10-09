# F1 — FineWeb-Edu QAT: staged Q9→Q3 versus direct Q3

**Result:** COMPLETED, exploratory one seed, `valid_for_science=true`; **all 5/5 protocol checks passed**.  
**Completed:** 2026-10-09 04:48:30 UTC (Oct 9 at 00:48 EDT).  
**HF job:** [codeflash85/6ac86eb0fee2c900701738dd](https://huggingface.co/jobs/codeflash85/6ac86eb0fee2c900701738dd) · **A10G-small**.  
**Immutable script commit:** `64dcd7f20240b4c62e6ecac8df70a1336a57bb14` · [source](../f1_fineweb_edu_qat_seed1729.py).  
**Frozen preregistration:** [FineWeb-Edu document-partition study](../research_log/f1_fineweb_edu_qat_seed1729_prereg_2026-10-09.md).  
**Raw terminal `FINAL_JSON` values with job provenance:** [archived F1 JSON](run_f1_fineweb_edu_direct_vs_staged_seed1729_2026-10-09.json).

## The question

Does the already-established Q9(300)→Q3(900) finite-budget staged advantage survive when **QAT training data** switches from WikiText-2 to a public sample of **FineWeb-Edu**, which was an ingredient of SmolLM2's pretraining?

FineWeb-Edu is **not** a reconstruction of the entire pretrained data mixture, nor the Instruct fine-tuning data. Pretrained-model overlap with the sampled documents is **unknown**. The experiment compares QAT adaptation, not generalization to wholly never-seen text in the underlying language model.

## Controlled protocol

- Source `HuggingFaceTB/SmolLM2-360M-Instruct`, pinned revision `a10cc1512eabd3dde888204e902eca88bddb4951`.
- QAT dataset `HuggingFaceFW/fineweb-edu`, `sample-10BT`, revision `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9`. WikiText validation held-out secondary evaluation uses revision `b08601e04326c79dfdd32d625aee71d232d685c3`.
- Fixed source BF16 rounding; all targeted transformer `nn.Linear` layers except lm_head use trainable FP32 masters and learned row scales, all other modules frozen.
- Teacher fixed FP16 source; objective `35% CE + 65% teacher KL`; AdamW (betas0.9/0.95, wd0), grad clip1, FP16 autocast/GradScaler, shared global v10 100-step warmup to1e-3 then cosine to1e-4 at step1200.
- Seed/order **1729**, same 1200 shuffled FineWeb training chunks and same LR values per arm. **D:** Q3 all1200, continuous optimizer/scales. **S300:** Q9 first300, then original Q3 scales + fresh Adam/GradScaler, Q3 next900.
- Deterministic `sha256(document id)` bucket split: 90% nominal train IDs; 5% dev; 5% eval. Loaded first train **1200** 129-token chunks from **180 documents**, dev **24** chunks from **3 documents**, eval **128** chunks from **21 documents**, scanning first **340** documents. **Document-ID sets disjoint by construction and check.** This is a *small convenience subset*, not a random independent broad web-corpus sample; non-independent chunks within documents must not be treated as 16,384 independent observations.
- Secondary WikiText-2 validation: first 128 129-token chunks = 16,384 target tokens, not the familiar previously inspected WikiText test used for prior model selection.

## Primary and secondary outcomes

| Evaluation | Direct Q3 NLL ↓ | Q9(300)→Q3(900) NLL ↓ | Direct PPL ↓ | Staged PPL ↓ | Paired NLL advantage (D − S) ↑ |
|---|---:|---:|---:|---:|---:|
| **FineWeb-Edu held-out documents (primary)** | **5.766660** | **4.996255** | 319.469 | 147.858 | **+0.770405** |
| **WikiText-2 validation (secondary)** | 6.558239 | 5.695225 | 705.029 | 297.444 | **+0.863013** |

FineWeb-Edu: staged reduces PPL by **53.72%** relative to D within this study; WikiText-2 validation: **57.81%** reduction. These reflect **one model source, one order, one dataset sample and one adapted heldout set**, not cross-seed confidence intervals.

**Further final diagnostics:**

- D FineWeb primary: 5.766659807 NLL; 319.469 PPL; 28.11% teacher top1; 2.880934 KL; 16384 tokens.
- S300 FineWeb primary: 4.996255074 NLL; 147.858 PPL; 36.02% teacher top1; 2.086872 KL; 16384 tokens.
- D WikiText secondary: 6.558238909 NLL; 705.029 PPL; 24.95% teacher top1; 3.513345 KL; 16384 tokens.
- S300 WikiText secondary: 5.695225466 NLL; 297.444 PPL; 29.79% teacher top1; 2.691722 KL; 16384 tokens.
- Effective updates: D **1194** with **6** AMP skips; S300 **1192** with **8** skips. Equal **1200 scheduled** opportunities, not exactly matched effective updates.
- At S300 switch, native-Q9 *train-split dev* NLL 5.428179, immediate initial-scale Q3 projection 6.942240, shock **+1.514062** nats. This dev slice contains just 3 source documents, so a single-point transition diagnostic is noisy.

## Protocol checks

- both_1200: `true`
- disjoint_fineweb_documents: `true`
- correct_split_count: `true`
- both_final_finite: `true`
- same_order: `true`

**All five were true;** `valid_for_science=true`. CPU source-data preflight `6ac86d5e095c578089301e06` and full document-partition preflight `6ac86e0d095c578089301e45` passed before GPU submission. Final HF stage was COMPLETED (not just last log line).

## Interpretation and limitations

**Positive observation:** the staged Q9→Q3 advantage persists under a *different QAT training distribution* based on a real component of SmolLM2 pretraining. The paired advantage also carries to WikiText validation when both arms are FineWeb-QAT-trained, although that does **not** prove new-model-family portability.

**Not established:** whether staged wins on unseen pretraining documents (pretrained-model contamination unknown); whether performance extrapolates to full FineWeb/other pretraining ingredients; whether the advantage survives much longer training; whether generators produce useful text; whether 1,200 or 300 Q9 steps are optimal. Dataset partitions are doc-disjoint **within F1**, but token chunks are correlated within each small number of documents. Results may vary with seeds/order/other independent document samples. No formal CI is estimated.

**Next research gate:** wait for separately launched [L1 6000-step WikiText durability job](https://huggingface.co/jobs/codeflash85/6ac86e1f095c578089301e53). Then decide whether to pre-register fresh FineWeb documents/seed replication and stronger quantization baselines. **No additional GPU runs were launched by this reporting step.**

For up-to-date status see [CURRENT_STATE.md](../CURRENT_STATE.md); full background [NEXT_JOBS_PLAN.md](../NEXT_JOBS_PLAN.md).
