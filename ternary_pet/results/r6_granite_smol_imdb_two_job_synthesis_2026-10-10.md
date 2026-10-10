# R6 terminal scientific results — Granite and SmolLM2, IMDb cross-corpus evaluation

**2026-10-10 UTC.** Both frozen, authorized R6 A10G-small training jobs **COMPLETED**; both emitted fully parseable `FINAL_JSON` with `valid_for_science=true`, all checks true, 4 complete D/W/N/H arm evaluations, 32 matched documents/arm and 16,384 target tokens/arm. **Both private H inference snapshots retained** according to job-produced reports. No extra seeds or GPU retries.

## Original job and preregistration provenance

- [Frozen R6 protocol](../research_log/r6_imdb_two_seed_prereg_2026-10-10.md), source commit `0d6f9440d197c67f837dd1a62f23c48c6399843c`; [CPU preflight](https://huggingface.co/jobs/codeflash85/6ac9d3defee2c90070183201) succeeded before the scientific GPU jobs.
- **Granite-4.0-350M seed170141:** [HF `6ac9d4cffee2c90070183341`](https://huggingface.co/jobs/codeflash85/6ac9d4cffee2c90070183341), completed 06:26:04 UTC; **15/15 validity checks**; [complete original final JSON](run_r6_granite_seed170141_imdb_2026-10-10.json).
- **SmolLM2-360M-Instruct seed190027:** [HF `6ac9d5d3095c57808930ed92`](https://huggingface.co/jobs/codeflash85/6ac9d5d3095c57808930ed92), completed 06:35:57 UTC; **17/17 validity checks**; [complete original final JSON](run_r6_smol_seed190027_imdb_2026-10-10.json).
- Both jobs show an injected secret name, without the credential value in logs; private snapshot reports `retained:true`. This is job-side upload success evidence, not yet a separate external checksum download audit.

## Primary: IMDb test reviews, next-token NLL (lower better)

**32 distinct documents per tokenizer/model** using frozen hashed document selector; **512 evaluated next-token targets each**, so 16,384/model/arm. Different tokenizers can select different documents. Original inherited output JSON names `fresh_fineweb` and `fineweb_revision` refer to this R6 **IMDb**, verified by `fineweb_new_doc_audit.dataset == "stanfordnlp/imdb"`; do **not** relabel this as FineWeb.

| Family/seed | Direct D | Wide W | Narrow N | **Hybrid H** |
|---|---:|---:|---:|---:|
| Granite 170141 | 6.756248 | 6.664138 | 6.766883 | **6.643789** |
| SmolLM2 190027 | 6.738819 | 6.048038 | 6.675319 | **5.958103** |

**Granite:** D−H=+0.112459 (H lower on **32/32** documents); W−H=+0.020349 (**25/32**); N−H=+0.123094 (**31/32**). D−W=+0.092110 (W lower on 32/32). H beats all three on the aggregate.

**Smol:** D−H=+0.780716 (**32/32**); W−H=+0.089934 (**30/32**); N−H=+0.717216 (**32/32**). D−W=+0.690782 (W lower on 32/32). H beats all three on the aggregate. All differences are nats/token and within-family, not pooled cross-model.

## Hybrid inference snapshots — private dataset

Dataset: `codeflash85/ternary-pet-r5-checkpoints` (the established private repository, reused for R6).

| Run | Private remote path | Job-reported bytes | SHA256 | Job report |
|---|---|---:|---|---|
| Granite | `granite_seed170141/H_final_compact.npz` | 50,895,933 | `3c30ab36a6518606db74a45411b6e96639171c063f0dd2b0f6cea384581f0888` | `retained=true` |
| Smol | `smol_seed190027/H_final_compact.npz` | 64,195,086 | `5d41638430f7a54fb553f2ad32d861386e5f38e42901c916fe8926172447fbfe` | `retained=true` |

These packed Q3 inference-state snapshots **are not fully resumable optimizer/training checkpoints**. Original R5 seed104729/130363 snapshots remain lost due earlier auth failures; the new R6 files are distinct.

## Conclusions and limits

R6 reinforces the observed H-first pattern across the prior R4 and R5 studies with two newly frozen seed orders on a **different genre** from FineWeb: IMDb movie-review texts. Unlike Smol's R5 technical failure, both R6 runs have valid final checks and archived per-document outputs; unlike R5's missing H models, both R6 inference snapshots were reported retained. The result supports a robust *within-the-tested-settings* trend, not a universal claim: one new seed/model family, within-document token dependence, possible pretraining overlap, codebook effects mediated by learned scales and optimizer trajectories, and differing Granite/Smol training recipes. Raw NLLs should not be pooled across families. Don't represent observational within-seed differences as a population effect size.

**Next scientific step, not launched:** separate external read-back SHA audit and inference reconstruction/generation check from the retained snapshots, or pre-registered larger multi-seed independent validation if separately authorized. No new GPU job has been initiated during this archival step.
