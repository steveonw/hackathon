# FineWeb-Edu: F1/F2/F3 three-training-order aggregate

**Run status:** all three separate Hugging Face GPU jobs **COMPLETED**; the scientific `valid_for_science` flag and all five frozen invariants passed for each seed.  
**Protocol frozen before new seeds:** [F2/F3 two-seed replication](../research_log/f2_f3_fineweb_edu_seed271828_424242_replication_prereg_2026-10-09.md).  
**Critical limitation:** **the same fixed 21 FineWeb held-out documents and the same WikiText-2 validation chunks are reused across all three seeds**. This is training-order/initial-RNG robustness, not independent-document generalization or a 3-dataset confirmation.

## Results (all heldout final NLL; lower is better)

| Seed | FineWeb direct Q3 | FineWeb Q9→Q3 | FineWeb D−S | WikiText validation direct | WikiText validation Q9→Q3 | WikiText D−S |
|---|---:|---:|---:|---:|---:|---:|
| 1729 | 5.766660 | **4.996255** | +0.770405 | 6.558239 | **5.695225** | +0.863013 |
| 271828 | 5.758656 | **4.979720** | +0.778936 | 6.589705 | **5.743868** | +0.845837 |
| 424242 | 5.760642 | **5.018289** | +0.742353 | 6.476450 | **5.779549** | +0.696902 |

**FineWeb-Edu advantage:** mean **+0.763898 nats/token**, median +0.770405, range **[+0.742353, +0.778936]**, descriptive sample SD **0.019140**; **3/3** orders favor Q9→Q3.  
**WikiText-2 validation advantage:** mean **+0.801917 nats/token**, median +0.845837, range **[+0.696902, +0.863013]**, descriptive sample SD **0.091351**; **3/3** orders favor Q9→Q3.

The mean is across three previously chosen training orders, **not** a confidence interval and not a sampling distribution across new documents. Relative PPL improvements by order are 1729: 53.72%, 271828: 54.11%, 424242: 52.40% on FineWeb.

## Reproducibility and provenance

| ID | Seed | HF job | Pinned script commit | Raw result |
|---|---:|---|---|---|
| F1 | 1729 | [6ac86eb0fee2c900701738dd](https://huggingface.co/jobs/codeflash85/6ac86eb0fee2c900701738dd) | `64dcd7f20240b4c62e6ecac8df70a1336a57bb14` | [JSON](run_f1_fineweb_edu_direct_vs_staged_seed1729_2026-10-09.json) |
| F2 | 271828 | [6ac87415fee2c90070173baf](https://huggingface.co/jobs/codeflash85/6ac87415fee2c90070173baf) | `64d171a225ba2781c83148a4235272c91939f9c9` | [JSON](run_f2_fineweb_edu_direct_vs_staged_seed271828_2026-10-09.json) |
| F3 | 424242 | [6ac87417095c5780893020c2](https://huggingface.co/jobs/codeflash85/6ac87417095c5780893020c2) | `a20c4cc66ae16026bb90969d1160294b3f5826e9` | [JSON](run_f3_fineweb_edu_direct_vs_staged_seed424242_2026-10-09.json) |

Each arm uses 1200 scheduled updates, D continuous Q3 and S Q9(300)→Q3(900) with switch reset. F2/F3 scripts differ from F1 only by numeric seed, frozen expected permutation prefix, and provenance/result identifier changes; static CPU preflight passed `F2_F3_PREFLIGHT_OK`. Source model/Dataset revisions and data partition identical across seeds. **Identical evaluated document hashes across seeds? true.** Doc count: FineWeb QAT training 180, dev 3, heldout 21. Do not treat 16384 chunk-level evaluation tokens as 16384 independent sampling units.

## Mechanistic limitations remain

This improves evidence that the **entire Q9-staging-plus-scale/optimizer-reset recipe** works reliably across the tested training orders in this model/corpus/sample. It does not isolate (a) scale-gradient effects versus within-bin depth, (b) Q9's larger representable range, (c) whether Q3 step300 **sham reset** alone helps on FineWeb, (d) margin/sensitivity-matched code selection, or (e) M1 learned-vs-original scale choices. The critique and preregisterable controls are recorded in [external technical review](../research_log/2026-10-09_external_technical_review_scale_range_sham_controls.md). The next highest-value mechanism controls are **frozen-scale depth** and **FineWeb direct-Q3 sham switch**, then range-matched Q9; no extra GPU jobs were launched to write this summary.

**Status note:** the independent L1 WikiText 6000-step durability job may still be running; inspect it separately. No cross-model, fresh-document or long-horizon claim can follow from these three FineWeb orders.
