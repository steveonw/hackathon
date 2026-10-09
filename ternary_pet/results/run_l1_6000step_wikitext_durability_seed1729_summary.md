# L1: 6000-step WikiText durability of staged vs direct ternary QAT

**Status:** Completed 2026-10-09 05:33:40 UTC; `valid_for_science=true`; all 5 frozen checks passed.  
**HF job:** [6ac86e1f095c578089301e53](https://huggingface.co/jobs/codeflash85/6ac86e1f095c578089301e53) · A10G-small.  
**Pinned script:** `841e0ab948991a24b7397b34d90d28c38f749879`, [source](../l1_6000step_wikitext_durability_seed1729.py).  
**Frozen preregistration:** [L1 protocol](../research_log/l1_6000_step_durability_wikitext_seed1729_prereg_2026-10-09.md).  
**Full original FINAL_JSON with provenance:** [raw L1](run_l1_6000step_wikitext_durability_seed1729_2026-10-09.json).

## Question

Does direct Q3 catch up with Q9(300)→Q3 when the same historical quantizer, frozen-parameter set, student objective, optimizer, source model and first1200 global schedule are extended to **6000 training opportunities**?

Both arms used 1200 previously matched seed1729 WikiText2 training chunks, followed by 4800 additional nonrepeated WikiText2 train chunks from the same source corpus. LRs through1200 matched v10 exactly (100-step warmup then cosine to1e-4), and beyond1200 are constant1e-4. D runs Q3 continuously; S runs Q9 for300, resets original Q3 scales + new Adam/GradScaler, and continues Q3. Same chunks and LRs for both arms; fixed reference teacher and 35% CE/65% KL.

**New WikiText2 validation** (first128 evaluation chunks, 16384 next tokens) measured at fixed points without changing training. The original historically inspected WikiText test set was evaluated only at step1200 for an exact-family reproduction check, not to choose settings.

## Full preregistered validation-loss trajectory

| Global scheduled step | Direct Q3 validation NLL ↓ | Q9→Q3 validation NLL ↓ | D−S gap, positive favors staged |
|---:|---:|---:|---:|
| 1200 | 5.047604 | **4.331518** | **+0.716087** |
| 2400 | 4.796893 | **4.082644** | **+0.714249** |
| 3600 | 4.684793 | **4.000079** | **+0.684715** |
| 4800 | 4.673563 | **3.976829** | **+0.696733** |
| 6000 | 4.500474 | **3.884434** | **+0.616040** |

**Primary step6000 endpoint:** D loss **4.500474** (PPL 90.06), staged loss **3.884434** (PPL 48.64). Positive staged gap **+0.616040 nats/token**. The gap narrowed from **0.716087** at step1200 but did not vanish. Gap is **not monotone** (0.6847 at3600, 0.6967 at4800), so avoid fitting or claiming long-run convergence from one curve. This is one seed/order under a single manually specified 1e-4 continuation schedule—not an asymptotic result or optimized direct baseline.

## Reproduction, checks and AMP steps

- Historical seed1729 test at step1200 **D 5.595722187310**, **S 4.900985367596**, matching the archived `5.595722187310457` and `4.9009853675961494` exactly. This supports comparability through the original v10 training horizon.
- 6000 scheduled opportunities each, D **5994** actual updates with **6** AMP skips, S **5990** actual updates with **10** skips. Equal scheduled horizon does not mean equal number of successful optimizer updates.
- Technical flags: `all_steps=true`; `all_q3_final=true`; `direct_historic_reproduced=true`; `staged_historic_reproduced=true`; `same_first1200_order=true`.
- The source model may have encountered some WikiText content in original pretraining. The new **validation** slice was not used to tune these QAT experiments, but it is drawn from the same WikiText2 corpus and is only one fixed sample.
- Original-source Q3 code movement at final and code histograms are in the raw JSON. No full quantized checkpoint was saved; no generation or long-context/latency measurement was performed.

## Interpretation and next gates

**Observed:** the staged recipe's finite-training-horizon advantage remains substantial even after 6000 matched scheduled opportunities, at five times the original 1200-step budget. The gap generally narrows but nonmonotonically; no proof of persistent positive difference at convergence and no guarantee on another optimizer, dataset, family or scale.

**Not resolved:** the external methodological critique: Q9 scale-gradient/representable-range differences, FineWeb direct-Q3 sham step300 optimizer/scale reset, position-margin matching, and M1 initial-vs-learned-scale code attribution. See [method review + proposed controls](../research_log/2026-10-09_external_technical_review_scale_range_sham_controls.md).

**Recommended next research decision:** hold off on unbounded duration scaling; first test **frozen-vs-learned-scale depth** and **FineWeb direct-Q3 sham switch**, then range-matched Q9. Use new documents/order for future dataset confirmation. None is authorized or launched by this report. [CURRENT_STATE.md](../CURRENT_STATE.md) is the authoritative handoff.
