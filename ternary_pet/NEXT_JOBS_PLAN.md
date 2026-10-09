# Next jobs — Q9→Q3 durability and original-training-distribution test

**Created:** 2026-10-09 UTC (Oct 8 EDT); proposed by the user after M1 completion.  
**Authoritative current status:** [CURRENT_STATE.md](CURRENT_STATE.md). **Historical rationale:** [NEXT_EXPERIMENT_PLAN.md](NEXT_EXPERIMENT_PLAN.md).  
**Scope authorized in conversation:** "set up the next jobs", plus user request to try the dataset SmolLM2 was originally trained on. Two carefully separated jobs; **no other seeds, large sweeps, hardware upgrades or automatic extensions**.

## Job L1 — WikiText-2 longer-horizon durability (one GPU)

**Pre-job frozen protocol:** [L1 preregistration](research_log/l1_6000_step_durability_wikitext_seed1729_prereg_2026-10-09.md).  
**Pinned script:** [l1_6000step_wikitext_durability_seed1729.py](l1_6000step_wikitext_durability_seed1729.py); code SHA `841e0ab948991a24b7397b34d90d28c38f749879`.  
**HF job submitted:** [6ac86e1f095c578089301e53](https://huggingface.co/jobs/codeflash85/6ac86e1f095c578089301e53), A10G-small, 2-hour timeout. Submission accepted with initial status SCHEDULING; **not yet a result**.

Direct Q3 vs Q9(300)→Q3, both **6000** scheduled updates on WikiText2 seed1729, with **fresh extra 4800 training chunks** after the original matched 1200. First1200 identical to previously archived v10 recipe, including shared global cosine LR to 1e-4; then LR held at **1e-4** after1200, no posthoc search. WikiText2 validation (128 nonoverlapping chunks) evaluated at 1200/2400/3600/4800/6000 to see whether the staged gap decays, persists or grows. Previously-viewed WikiText2 test slice used **at step1200 only as an exact-family reproduction check**, not as novel validation.

**What L1 answers:** whether direct Q3 catches up *under this continued fixed training recipe*. One six-thousand-step run still cannot establish asymptotic superiority; do not equate validation curves with true convergence.

## Job F1 — FineWeb-Edu public sample (one GPU, separate mechanism question)

**Pre-job frozen protocol:** [F1 preregistration](research_log/f1_fineweb_edu_qat_seed1729_prereg_2026-10-09.md).  
**Pinned script:** [f1_fineweb_edu_qat_seed1729.py](f1_fineweb_edu_qat_seed1729.py); code SHA `64dcd7f20240b4c62e6ecac8df70a1336a57bb14`.  
**HF GPU job submitted:** [6ac86eb0fee2c900701738dd](https://huggingface.co/jobs/codeflash85/6ac86eb0fee2c900701738dd), A10G-small, max 90m, initial status SCHEDULING. Scientific outcome not yet known. CPU dataset-partition integrity test `6ac86e0d095c578089301e45` **COMPLETED**, producing train1200/dev24/heldout128 chunks from 340 scanned documents with disjoint document IDs.

Same *original 1200-step* Q9/Q3 recipe, but WikiText2 QAT training replaced by public `HuggingFaceFW/fineweb-edu` `sample-10BT` sample. FineWeb-Edu was part of the pretraining mixture of `SmolLM2-360M` according to its official model card. This experiment does **not** recreate SmolLM2's entire 4-trillion-token mixed pretraining or Instruct fine-tuning recipe. Keep model/objective/quantizer/training budget fixed **within F1**.

Train/dev/heldout FineWeb-Edu sets are drawn from deterministic **SHA256(document ID) modulo20** buckets to prevent source-document overlap across QAT sets; a second endpoint uses the not-previously-used WikiText2 validation split. Raw FineWeb content may have been seen by the **base model during pretraining**; do not claim independently never-seen text relative to the source model. Freeze dataset/model SHAs as stated in F1 prereg.

**What F1 answers:** whether Q9→Q3 vs direct Q3 behaves similarly when QAT uses a corpus closer to the model's pretraining distribution. A different validation outcome cannot be attributed exclusively to corpus if model/budget also change; within F1 arms all those factors are fixed.

## Preflight and reproducibility provenance

- Official SmolLM2-360M-Instruct model revision `a10cc1512eabd3dde888204e902eca88bddb4951`.
- WikiText2 revision `b08601e04326c79dfdd32d625aee71d232d685c3`.
- FineWeb-Edu revision `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9`.
- CPU source-access preflight `6ac86d5e095c578089301e06` confirmed metadata, tokenizer and source documents (`DATA_SOURCE_PREFLIGHT_OK`).
- Static checks `6ac86e01fee2c900701738a9` printed `BOTH_STATIC_QA_OK` from pinned scripts.
- Full FineWeb document-partition preflight (CPU-basic) `6ac86e0d095c578089301e45` **COMPLETED** with `F1_PARTITION_PREFLIGHT_OK` and document-disjoint chunk counts train1200/dev24/eval128. F1 GPU was submitted once; do not duplicate.

## Required after completion

1. Confirm HF terminal stage **COMPLETED** / failure and retrieve full terminal `FINAL_JSON_BEGIN`…`FINAL_JSON_END` from each job.
2. Validate the frozen checks, including L1 step1200 historical control reproduction and F1 document disjointness; distinguish technical failure from scientific null.
3. Archive exact result JSON + provenance in `ternary_pet/results/`, add concise descriptive summary, update `CURRENT_STATE.md`, `AI_HANDOFF.md`, `EXPERIMENT.md`, `RESEARCH_REPORT.md`, `SHAREABLE_RESEARCH_REPORT.md` and `README.md`.
4. Preserve all null or negative findings; do not quietly relaunch failed jobs, adjust LR, choose new datasets, add seeds or treat results as a paper-ready replication.
