# Ternary Pet — CURRENT STATE

## R6 GENERATION CONTROL SCRIPT BUG FIXED — 2026-10-10

Two CPU-only controlled-generation jobs ([Granite](https://huggingface.co/jobs/codeflash85/6ac9ea5ffee2c90070184895), [Smol](https://huggingface.co/jobs/codeflash85/6ac9ea68095c57808930ff7c)) **ERROR** because `model.generate(..., generator=torch.Generator(...))` is rejected by the currently installed Transformers generation API (`ValueError: unused model_kwargs ['generator']`), not due to snapshot integrity. [Corrected script](r6_cpu_generation_base_vs_hybrid.py) pinned SHA `7f641c27470748fd1ff139f5a84590a6c63d18ba` now calls `torch.manual_seed(20261010)` before sampled generation, and passes only supported sampling options. [Standalone CPU sampling test](https://huggingface.co/jobs/codeflash85/6ac9eda9095c5780893100ce) emitted `R6_GENERATION_SAMPLING_PATCH_TEST_OK` using Transformers `generate()` with sampling; full original base-vs-H comparison still **not** rerun and must not be claimed. R6 training results and independently verified snapshot NLL parity remain unchanged. Private models require local CLI `--secrets HF_TOKEN`.

---

## R6 RECONSTRUCTION LOSS PARITY VERIFIED — 2026-10-10 UTC

[Original remote CPU NLL parity logs and per-document results](results/r6_reconstruction_loss_parity_verified_2026-10-10.md) are now independently accessible: [Granite CPU job](https://huggingface.co/jobs/codeflash85/6ac9e829095c57808930fd96) and [Smol CPU job](https://huggingface.co/jobs/codeflash85/6ac9e841fee2c9007018468b), both **COMPLETED**, fixed script SHA `ddcb84cc4566b989429d5cd131199f264c5b6d81`. First 4 pinned R6 IMDb documents per model (2,048 tokens/model) CPU FP32 reconstructed H vs original GPU AMP H NLL: **Granite 6.681939 vs 6.682182, delta -0.000244, max doc delta 0.000712**; **Smol 6.165740 vs 6.165852, delta -0.000113, max doc delta 0.000137**. Supports high-fidelity inference reconstruction, **not** numeric parity of every token/logit or useful generation; both single-prompt generations were degenerate. Original HF logs fetched directly, no further user copy needed. No new GPU training.

---

## R6 CPU INFERENCE RECONSTRUCTION COMPLETED — quality caveat (2026-10-10 UTC)

[Recorded end-to-end CPU reconstruction results](results/r6_cpu_inference_reconstruction_outcomes_2026-10-10.md) from user-provided terminal markers and HF jobs confirmed **COMPLETED**: [Granite job](https://huggingface.co/jobs/codeflash85/6ac9e3fd095c57808930fb10) reconstructed 249,561,088 ternary weights/168 layers and [Smol job](https://huggingface.co/jobs/codeflash85/6ac9e405fee2c90070184365) reconstructed 314,572,800 weights/224 layers. Both passed snapshot SHA check, ran finite logits and generated 12 tokens. **Outputs are degenerate**: Granite generated 12 periods; Smol predominantly newlines/periods. This proves technical decoding, not useful language generation or parity with job-side H evaluation; investigate before any deployment claim. Full remote log retrieval was blocked in this session, so observations come from user-pasted original success lines, corroborated by job statuses. No additional GPU launched.

---

## R6 CPU INFERENCE RECONSTRUCTION — initial attempts failed, decode fix ready (2026-10-10 UTC)

Both authorized CPU-only proof attempts, [Granite `6ac9e36cfee2c90070184271`](https://huggingface.co/jobs/codeflash85/6ac9e36cfee2c90070184271) and [Smol `6ac9e389095c57808930fa68`](https://huggingface.co/jobs/codeflash85/6ac9e389095c57808930fa68), **ERROR**, at the identical NumPy unsigned-to-negative `np.where(codes==2,-1,codes)` conversion (`OverflowError: Python integer -1 out of bounds for uint8`). Both original base weights and private snapshots downloaded successfully; no real inference was run. This does not invalidate R6 science or the independent snapshot SHA audit. The [reconstruction script](r6_cpu_inference_reconstruction.py) was fixed at commit `056545171e015e821a8e6a5a7bfc3eebadbcbe2c` by converting codes to signed int8 first; a separate [CPU regression job](https://huggingface.co/jobs/codeflash85/6ac9e3ddfee2c9007018432f) printed `R6_TERNARY_DECODE_REGRESSION_OK 2.5.3`. **End-to-end inference reconstruction still unverified**; rerun the fixed script via authenticated Windows CLI `--secrets HF_TOKEN`, not via unauthenticated connector. No GPU jobs needed.

---

## R6 INDEPENDENT PRIVATE SNAPSHOT AUDIT PASSED — 2026-10-10 UTC

**Both saved R6 H inference-only snapshots independently downloaded, SHA-256-verified, and structurally validated.** CPU HF job [6ac9e014095c57808930f5dd](https://huggingface.co/jobs/codeflash85/6ac9e014095c57808930f5dd) **COMPLETED**, emitting `R6_SNAPSHOT_AUDIT_OK` for both files and `R6_BOTH_PRIVATE_SNAPSHOTS_VERIFIED`. Original audit source pinned commit `7e4758ad81a98269932f918b6014adfcd8a5e3c3` ([script](r6_private_snapshot_cpu_audit.py)). Granite: 50,895,933 bytes, SHA-256 `3c30ab36a6518606db74a45411b6e96639171c063f0dd2b0f6cea384581f0888`, 168 layers/249,561,088 ternary weights. Smol: 64,195,086 bytes, SHA-256 `5d41638430f7a54fb553f2ad32d861386e5f38e42901c916fe8926172447fbfe`, 224 layers/314,572,800 weights. Audit checks private dataset, size, hash, manifest, packed code ranges, and finite positive scales; it does **not** yet test model loading or generated outputs. No GPU jobs initiated.

---

## R6 COMPLETE — two verified new-seed cross-corpus jobs (2026-10-10 UTC)

**Both authorized A10G-small jobs COMPLETED with complete final JSON and `valid_for_science=true`.** [R6 two-job scientific synthesis](results/r6_granite_smol_imdb_two_job_synthesis_2026-10-10.md) · [Frozen prereg](research_log/r6_imdb_two_seed_prereg_2026-10-10.md). Source commit `0d6f9440d197c67f837dd1a62f23c48c6399843c`. Primary evaluation is **32 pinned IMDb test movie reviews/model, 16,384 targets/arm**, NOT FineWeb despite inherited JSON variable keys.

| Model/new seed | Direct D | Wide W | Narrow N | **Hybrid H** | Job validity |
|---|---:|---:|---:|---:|---|
| Granite-4.0-350M **170141** | 6.756248 | 6.664138 | 6.766883 | **6.643789** | **15/15 passed** |
| SmolLM2-360M **190027** | 6.738819 | 6.048038 | 6.675319 | **5.958103** | **17/17 passed** |

Granite H beats D/W/N on 32/25/31 of 32 paired documents respectively; Smol H beats D/W/N on 32/30/32 of 32. [Granite HF job](https://huggingface.co/jobs/codeflash85/6ac9d4cffee2c90070183341) · [full JSON](results/run_r6_granite_seed170141_imdb_2026-10-10.json); [Smol HF job](https://huggingface.co/jobs/codeflash85/6ac9d5d3095c57808930ed92) · [full JSON](results/run_r6_smol_seed190027_imdb_2026-10-10.json). **Both H inference snapshots report `retained=true`** in the established private HF dataset, Granite `granite_seed170141/H_final_compact.npz` and Smol `smol_seed190027/H_final_compact.npz`; independent SHA download audit still pending. These are not full training restart checkpoints. **No more R6 GPU jobs should be launched.** Limits: only one new seed/model and selected genre, possible base-model pretraining overlap, family-specific optimizer/precision confounds. Historic R5 snapshots remain unavailable.

---

## R6 REVIEW DRAFT — not authorized (2026-10-10)

[Prospective R6 protocol and compute-gating checklist](research_log/r6_candidate_protocol_review_draft_2026-10-10.md) created using archived R5 evidence. This is **not a frozen preregistration** and does **not** authorize paid GPU runs. Before launching, determine job budget, fresh seeds, independent evaluation corpus, source pins and a CPU-only preflight; preserve original R5 evidence and verified `--secrets HF_TOKEN` upload path.

---

## R5 ARCHIVED POSTRUN SYNTHESIS — 2026-10-10

[**R5 cross-model final synthesis**](results/r5_cross_model_final_synthesis_2026-10-10.md) now records verified Granite per-document D/W/N/H differences (32 paired docs), SmolLM2 log-only aggregate measurements and incomplete validation, WikiText secondary outcomes, original provenance/limitations, and the **successfully completed remote CPU secret-backed private upload check**. The original R5 GPU snapshots remain unavailable. No further GPU run is authorized or scheduled; next R6 design is a proposal only.

---


## R5 TERMINAL — Granite valid; SmolLM2 reporting failure (2026-10-10 UTC)

**This is the newest status; the older R5 ACTIVE sections below are historical submission records.** Both preregistered jobs are terminal, with no further GPUs or retries launched.

- **Granite-4.0-350M seed104729**: [HF job](https://huggingface.co/jobs/codeflash85/6ac9b8c8095c57808930d94f) **COMPLETED** 04:27:14 UTC; its original log emitted a complete `FINAL_JSON_BEGIN`/`FINAL_JSON_END` object, with `valid_for_science=true` and **15/15 checks passed**. New 32-document FineWeb-Edu NLL (lower better): D **6.490101**, W **6.388855**, N **6.506379**, H **6.358198**. H improved over D on **32/32** documents, over N on **31/32**, and over W on **27/32**. The [complete original final JSON](results/run_r5_granite_seed104729_freshdocs_2026-10-10.json) (all 15 checks, 4 arms and 32 paired documents) is archived in `ternary_pet/results/`, with HF job provenance and source commit.
- **SmolLM2-360M-Instruct seed130363**: [HF job](https://huggingface.co/jobs/codeflash85/6ac9b8cefee2c90070181ae3) **ERROR** exit 1 at 04:32:04 UTC **after all four arms logged final evaluations**. The original immutable script raised `KeyError: 'total'` during final aggregation, because `code_hist()` returns normalized code fractions without a `total` key. New FineWeb NLL from arm logs: D **6.617192**, W **5.775722**, N **6.591928**, H **5.658846**. **These are log-recovered descriptive measurements, NOT a validated final scientific result:** no `FINAL_JSON`, original paired per-document arrays, or all postrun checks were emitted. [Recovery and caveats](results/r5_smol_seed130363_log_recovery_2026-10-10.md).
- **Neither H inference snapshot was retained.** Both private Hugging Face uploads returned HTTP **401 Unauthorized**; temporary local file sizes do not establish persistence. Authentication must be corrected separately; no checkpoint rescue is claimed.

**Research interpretation:** On the new Granite seed, H wins against D/W/N with full job-side validation. Smol's aggregate logs also rank H first, but the failure means it must be clearly separated from validated replications. Distinct family-specific optimizers, possible pretraining-document overlap, one new seed per family, and absence of retained checkpoints still limit generalization. **No retuning or GPU retry authorized.** The reporting-only fix is on a separate review branch and does not change the original SHA-pinned R5 run.

---


**Authoritative short handoff:** As of 2026-10-08 Eastern / 2026-10-09 UTC.  
**Repository:** `steveonw/hackathon` → `ternary_pet/`.  
**Status:** **R3 scientific GPU job COMPLETED 2026-10-10 02:04 UTC with 16/16 checks passed and raw data archived**; previous R2/R1/G1/S1/F1–F3/L1 complete. See latest R3 evidence and [R3_ACTIVE_JOB_PLAN.md](R3_ACTIVE_JOB_PLAN.md).

## R5 ACTIVE — NEW preregistered seeds + previously unexamined 32-doc FineWeb eval (2026-10-10 UTC)

**One Granite + one SmolLM2 GPU job SUBMITTED, science still PENDING.**
- **Granite-4.0-350m seed104729**, [HF `6ac9b8c8095c57808930d94f`](https://huggingface.co/jobs/codeflash85/6ac9b8c8095c57808930d94f), immutable script SHA `c1f0d61aa69a85899203647b88277382275c2a3a`, [source](r5_granite_new_seed104729_fineweb_hybrid.py). **RUNNING at last HF check**, direct Q3 prep step100 logged.
- **SmolLM2-360M-Instruct seed130363**, [HF `6ac9b8cefee2c90070181ae3`](https://huggingface.co/jobs/codeflash85/6ac9b8cefee2c90070181ae3), immutable SHA `4560d829f483386eb39411c2a9fbf204d4157c4d`, [source](r5_smol_new_seed130363_fineweb_hybrid.py). **RUNNING at last HF check**.

Both single-A10G-small jobs have **90-minute limits** and each repeats four matched D/W/N/H QAT arms over1200 scheduled opportunities, with historical model-specific optimizer/precision settings. Newly chosen seeds are arbitrary precommitted (104729/130363), **not outcome-selected**, but there is only one new seed per family. The **primary new evaluation** is 32 *previously unused in QAT research* FineWeb-Edu source documents per model (rows≥1223, bucket7 by SHA256(doc-id) %20, first32 qualified ≥516 tokenizer tokens) and 16,384 paired target tokens per model/arm, full per-doc losses saved; source document IDs can differ across model tokenizers and may have been in base pretraining. Primary N−H/W−H/D−H. WikiText2 validation/test only secondary. **Private compact H Q3 inference snapshots are best-effort**, not restartable full-FP32/Adam checkpoints; explicitly check successful uploaded artifact in raw result and avoid claiming retention on an upload failure.

**CPU code+data preflight COMPLETED** [HF `6ac9b87dfee2c90070181aa5`](https://huggingface.co/jobs/codeflash85/6ac9b87dfee2c90070181aa5), with `R5_ALL_CPU_PREFLIGHT_OK`, both codebooks/STE and both tokenizer-specific 32-doc selectors verified (Granite rows1487..3052, Smol1300..2984). **Frozen-before-compute prereg** [R5 prereg](research_log/r5_independent_seed_new_fineweb_validation_prereg_2026-10-09.md), commit `55786ecbd3baa63f5ded08287934d1b09d9e59a9`. **Authoritative exact-job handoff:** [R5_FRESH_SEED_NEW_DOCS_ACTIVE_JOB_PLAN.md](R5_FRESH_SEED_NEW_DOCS_ACTIVE_JOB_PLAN.md). No extra GPUs, variant searches or retries; previous R4 four valid jobs all archived; R5 results not known yet.

---

## PREVIOUS VERIFIED — R4 Granite + Smol four-job hard/easy hybrid study COMPLETE (2026-10-10 UTC)

**Four GPU jobs COMPLETED, all scientific validity gates passed**: each `valid_for_science=true`, **12/12 checks per job, 48/48 in all**, historical direct-Q3 and original-Q9 WikiText test anchors reproduced. All exact scientific `FINAL_JSON` objects saved with pinned source SHA, model/data revisions, per-chunk losses, checks, and HF provenance. [R4 detailed 4-job aggregate](results/run_r4_granite_smol_easy_hard_four_job_aggregate_2026-10-10.md) · [R4 complete 2+2 handoff](R4_GRANITE_SMOL_EASY_HARD_JOBS.md). **No additional GPU job, duplicate or retry.**

| Family, previously selected seed | D direct | W wide Q9→Q3 | N narrow Q9→Q3 | **H hybrid** |
|---|---:|---:|---:|---:|
| **Granite-4.0-350M 271828, historically hard/negative** | 5.438636 | 5.577830 | 5.416681 | **5.370064** |
| Granite-4.0-350M 424242, historically easy/positive | 5.420453 | 5.349609 | 5.396033 | **5.295450** |
| SmolLM2-360M 271828 | 5.293719 | 4.567472 | 5.259229 | **4.480920** |
| SmolLM2-360M 424242 | 5.265099 | 4.553739 | 5.273762 | **4.520216** |

**Primary:** WikiText2 validation first64 aligned 129-token windows = 8192 target tokens each arm, NLL lower better. **Hybrid H beats all other arms on all four jobs.** Key **Granite hard reversal**: original W remains **+0.139194 NLL worse** than direct D, whereas H finishes **−0.068572 better** than D (H vs W **−0.207766**). Granite easy H vs W **−0.054158**; Smol seed271828 H vs W **−0.086552**; Smol seed424242 H vs W **−0.033523**. The hybrid preserves narrow Q9's seven central outputs and changes **only the ±4 extreme codebook reconstruction levels** to original wide ±α. These are two states shared across many weights, not two specific parameters. [Full paired chunk signs/median contrasts and original WikiText test reproduction](results/run_r4_granite_smol_easy_hard_four_job_aggregate_2026-10-10.md).

**All four HF jobs:** Granite hard [6ac9a43d095c57808930ccfe](https://huggingface.co/jobs/codeflash85/6ac9a43d095c57808930ccfe) completed 03:00:26 UTC, [raw JSON](results/run_r4_granite_hard_seed271828_2026-10-10.json); Granite easy [6ac9a443095c57808930cd03](https://huggingface.co/jobs/codeflash85/6ac9a443095c57808930cd03) completed 03:00:36 UTC, [raw JSON](results/run_r4_granite_easy_seed424242_2026-10-10.json); Smol seed271828 [6ac9a5c2fee2c90070180ce3](https://huggingface.co/jobs/codeflash85/6ac9a5c2fee2c90070180ce3) completed 03:10:10 UTC, [raw JSON](results/run_r4_smol_seed271828_2026-10-10.json); Smol seed424242 [6ac9a5c9095c57808930cdde](https://huggingface.co/jobs/codeflash85/6ac9a5c9095c57808930cdde) completed 03:09:31 UTC, [raw JSON](results/run_r4_smol_seed424242_2026-10-10.json). Each was pinned and CPU codebook QA had passed before paid GPU.

**Interpretation limits:** GRANITE was historically mixed before this experiment, hard/easy seeds **selected after seeing past behavior** and 2 per family do not estimate a population effect. Primary first64 validation windows are contiguous, **not independent documents or a fresh unrelated corpus**. Cross-family LR/precision/direct-baseline reset differ: Granite constant1e-4 BF16 and step300 direct sham reset, Smol warmup/cosine FP16 GradScaler and uninterrupted direct optimizer; **do not pool raw NLL/gain across model families**. H's intervention changes extreme reconstruction outputs, but gradients through scale and training states may change too, so magnitude alone is not proven sole mediator. No final model checkpoints, not a generation-quality test. Scientific next step is independent randomized seeds/new document corpus with retained checkpoint or scale/assignment diagnostics — *only as a proposal, not a launched job*. Older “R4 running/submitted” text below is historical.

---

## HISTORICAL AT SUBMISSION — R4 2+2 MATRIX: all FOUR scientific GPUs submitted (2026-10-10 UTC)

**Batch A, Granite 4.0 350M** previously submitted: historically hard seed271828 [HF `6ac9a43d095c57808930ccfe`](https://huggingface.co/jobs/codeflash85/6ac9a43d095c57808930ccfe) and easy seed424242 [HF `6ac9a443095c57808930cd03`](https://huggingface.co/jobs/codeflash85/6ac9a443095c57808930cd03). Both remained RUNNING at start of this second request; final science unknown. Immutable Granite script SHA `33dfae38bd016946ad5d37bd633fdf02818fbacd`, Granite BF16 + LR constant1e-4.

**Batch B, SmolLM2-360M-Instruct** **NOW SUBMITTED at user request**: seed271828 [HF `6ac9a5c2fee2c90070180ce3`](https://huggingface.co/jobs/codeflash85/6ac9a5c2fee2c90070180ce3), seed424242 [HF `6ac9a5c9095c57808930cdde`](https://huggingface.co/jobs/codeflash85/6ac9a5c9095c57808930cdde). One A10G-small, 90m hard limit per job. Exact seed-only scripts [271828](r4_smol_q9_hybrid_seed271828.py) · [424242](r4_smol_q9_hybrid_seed424242.py), immutable common SHA `65af3057b248f38a6d46ad8b7cd1bc1c52d2e654`. **Prepaid-GPU CPU dynamic QA** [HF `6ac9a578095c57808930cd9c`](https://huggingface.co/jobs/codeflash85/6ac9a578095c57808930cd9c) **COMPLETED**, verifying both script AST/pins, nine Q9 codes, H/N inner levels match, outer ±4 restored in H only, STE gradients, post-switch original Q3 and seed-only script variation.

**Each of four GPU jobs runs the same D/W/N/H family experiment** (matched within seed): D direct Q3 1200, W original wide Q9(300)→Q3(900), N narrow T4 Q9(300)→Q3(900), H nonuniform Q9(300)→Q3(900) with narrow 7 inner levels and original wide ±α two extreme levels. Smol uses **original tuned warmup100→1e-3 cosine→1e-4, FP16 teacher/autocast + GradScaler, FP32 masters**, BF16-rounded source; D direct original continuous Adam, W/N/H original Q3 row scale + Adam/GradScaler reset at300. Granite uses previously calibrated **constantLR1e-4, BF16 teacher/autocast no GradScaler**, all arms reset source Q3 scales/fresh Adam at300. Consequently architectural and optimizer policies remain somewhat confounded across families; compare within-family paired NLL differences. Original WikiText2 test first64 chunks anchors; **primary first64 WikiText validation chunks**, matched 8192 tokens and per-chunk losses; historical seeds are **not new random confirmation**, samples may have appeared in pretraining. Smol historical anchors seed271828 D≈5.6228/W≈4.9510, seed424242 D≈5.6067/W≈4.9563, fixed tolerance ±0.08 nats. No source/output retuning after observing jobs.

**Frozen before jobs:** [R4 2+2 preregistration](research_log/r4_2026-10-09_granite_smol_easy_hard_two_batch_prereg.md). **Authoritative four job IDs, precise protocol/recovery rules:** [R4_GRANITE_SMOL_EASY_HARD_JOBS.md](R4_GRANITE_SMOL_EASY_HARD_JOBS.md). **No fifth GPU/retry planned.** Wait for all four terminal results, extract and archive original FINAL_JSONs, all validity checks, negative outcomes, comparisons and limitations; update reports then.

---

## HISTORICAL — R4 Granite hard/easy pair submitted, Smol deferred (2026-10-09 EDT)

**User requested exactly 2 jobs in this prompt and 2 only in the following prompt, waiting for Granite to finish.** Batch A **TWO Granite-4.0-350m jobs submitted**, one per previously identified training-order seed, both using a four-arm D/W/N/H controlled hybrid Q9 comparison:

- **Hard seed271828** (historically full wide Q9 lost): [HF `6ac9a43d095c57808930ccfe`](https://huggingface.co/jobs/codeflash85/6ac9a43d095c57808930ccfe), A10G-small, 90m max.
- **Easy seed424242** (historically wide Q9 won): [HF `6ac9a443095c57808930cd03`](https://huggingface.co/jobs/codeflash85/6ac9a443095c57808930cd03), A10G-small, 90m max.

**Immutable single source SHA** `33dfae38bd016946ad5d37bd633fdf02818fbacd`, [hard script](r4_granite_q9_hybrid_hard_seed271828.py), [easy script](r4_granite_q9_hybrid_easy_seed424242.py); source scripts differ by exactly one seed line. Frozen [R4 2+2 prereg](research_log/r4_2026-10-09_granite_smol_easy_hard_two_batch_prereg.md), [complete job handoff](R4_GRANITE_SMOL_EASY_HARD_JOBS.md). BF16 Granite teacher/autocast, FP32 masters, historical constant LR1e-4, original WikiText2 QAT training split, W nine-state wide, N narrow, H seven narrow center levels with two ±4 outputs restored wide. All three Q9 arms stage Q9(300)→Q3(900), original Q3 source row scales and fresh AdamW at300. D matched direct Q3 with same reset at300. **Primary new-to-R4 validation:** WikiText2 validation64 chunks×128 targets, per-chunk loss; historical WikiText test first64 remains secondary reproduction anchor. CPU pin job `6ac9a397fee2c90070180b8d` and CPU nine-code/gradient QA `6ac9a40b095c57808930ccec` both **COMPLETED** before GPUs; model and dataset commits pinned.

**Batch B planned but NOT LAUNCHED:** SmolLM2-360M-Instruct seed271828 and seed424242, 2 scientific GPU jobs **only upon next explicit user prompt after Granite jobs finish**. Smol should use its own calibrated LR/precision recipe and same D/W/N/H codebooks on WikiText2; no premature claims of cross-family effect. R4 study uses already viewed Granite hard/easy orders, not blind novel seeds. **R4 results pending** until exact `FINAL_JSON` checks; preserve null/negative arms and no duplicate or retry GPU runs.

All R3/FineWeb studies and earlier G1/S1/R1/R2 remain archived and complete. **Never misread historical submission sections below as live state.**

---

## MOST RECENT COMPLETED — R3 hybrid outer-level Q9 nearly matches original wide Q9 on NEW 32-document holdout (2026-10-10 02:04 UTC)

**R3 GPU COMPLETED and VALID**: [HF `6ac99666fee2c9007018011f`](https://huggingface.co/jobs/codeflash85/6ac99666fee2c9007018011f), A10G-small, finished **2026-10-10 02:04:07.293 UTC / 2026-10-09 22:04 EDT**. Pinned source commit `a4abf256c0be9db771e3ec271c1f0a5e6d605816`; frozen protocol commit `bf89baef162f2a9e948ab702a1708c74b856d3a4`, before job. **`valid_for_science=true`, all 16/16 checks passed.** [R3 complete 32-doc/per-arm raw JSON](results/run_r3_nonuniform_q9_outer_freshdocs_seed1729_2026-10-10.json) · [detailed scientific result](results/run_r3_nonuniform_q9_outer_freshdocs_seed1729_summary.md) · [R3 handoff](R3_ACTIVE_JOB_PLAN.md).

**Main controlled intervention:** same seed1729 SmolLM2/FineWeb QAT data/order, same T4 ternary-prep integer code thresholds and identical master/z STE surrogate across all Q9 variants. Three Q9(300)→Q3(900) arms: W original wide reconstruction values `{0,±α/4,±α/2,±3α/4,±α}`; N narrow `{0,±α/6,±α/3,±α/2,±2α/3}`; **H hybrid `{0,±α/6,±α/3,±α/2,±α}`**, differing from narrow N *only* in the output of codes ±4, where H restores ±α, leaving all seven central outputs and integer code assignment thresholds identical. Plus direct Q3 D. Same historical step300 original Q3 scale and Adam/GradScaler reset for W/N/H, FP32 masters and all original 1200-step schedule.

**PRIMARY evaluation is 32 genuinely new-to-this-QAT-study FineWeb-Edu doc IDs**, preselected by frozen SHA doc hash bucket0 from **source rows>340** (actual first343,last1222), not any original train/dev/21 old heldout docs; 4×128 prediction tokens/document, total **16,384**, with per-document SHA256 IDs and NLL sums/means archived. **Not guaranteed absent from SmolLM2 underlying pretraining data.**

| Arm | NEW FineWeb 32-doc heldout NLL ↓ | D−arm staged advantage | Historical FineWeb 21-doc NLL ↓ | WikiText validation NLL ↓ |
|---|---:|---:|---:|---:|
| D direct Q3 | 5.789299 | — | 5.766660 | 6.558239 |
| **W original wide Q9** | **4.885759** | **+0.903540** | **4.996255** | **5.695225** |
| N original-threshold narrow Q9 | 5.813866 | −0.024567 | 5.780473 | 6.593468 |
| **H hybrid outer-value restored Q9** | **4.916794** | **+0.872505** | **5.039273** | **5.730818** |

**KEY RESULT:** Hybrid H retains **96.57%** of the W staged advantage on new docs (`(D−H)/(D−W)`) and recovers **96.66%** of N→W NLL improvement. **All 32/32 fresh docs** favor H over N and H over direct D. W beats H on 23/32 docs, but their aggregate gap is only **H−W +0.031035 nats/token**, compared with **N−H +0.897071** and **D−H +0.872505**. All old D/W/N FineWeb+WikiText anchors reproduce their archived losses exactly, 16/16 technical checks pass; Q9-native prep dev H **5.365745** vs W **5.428179** and N **6.478196**. Extreme Q9 occupancy H29.34%, W29.30%, N29.22%; the effect is not solely different code occupancy.

**Scientific conclusion:** within this Q9→Q3 recipe and one seed, **restoring the high-magnitude outputs at just the two extreme Q9 codebook states** is sufficient to recover nearly the entire wide-grid staging advantage even when all interior state outputs remain narrow. This is NOT a claim about only two individual weights; extreme codebook levels are shared across many weights. This does not prove output-range magnitude rather than changed scale-gradient/optimization trajectories or ternary code survival is sole mediator. Fresh docs are new to *our QAT evaluation*, **not necessarily original pretrained model**; still one model/seed/source dataset and no saved trained checkpoint.

**Preflight provenance:** original CPU job `6ac992d4095c57808930c1b0` passed code/data assertions but crashed during interpreter teardown, status ERROR; separate finalization-safe CPU job [`6ac99425095c57808930c23c`](https://huggingface.co/jobs/codeflash85/6ac99425095c57808930c23c) **COMPLETED** after printing exact nonuniform codebook/gradient and 32-doc selection checks before R3 GPU launch. Only ONE R3 scientific GPU job submitted; now terminal. All earlier R2/R1/G1/S1/F1–F3/L1 verified and preserved. **No follow-up GPU job automatically launched.**

**Next research proposal, not launched:** cross-model/fresh-corpus external validation or targeted measurement of scale gradients/master weight placement and Q3 code survival under H vs W vs N, with restorable checkpoints and per-doc uncertainty, rather than adaptively tuning more variants on this same holdout. Read [R3 result summary](results/run_r3_nonuniform_q9_outer_freshdocs_seed1729_summary.md) for design, checks and limitations.

---

## HISTORICAL AT SUBMISSION — R3 LAUNCHED (2026-10-10 UTC)

**R3 single scientific GPU job ACCEPTED:** [HF `6ac99666fee2c9007018011f`](https://huggingface.co/jobs/codeflash85/6ac99666fee2c9007018011f), `a10g-small`, max **2 hours**, initial SCHEDULING, subsequently **RUNNING** with first D training step logged. One scientific R3 GPU job only, no seed sweep. [Exact R3 job and archival handoff](R3_ACTIVE_JOB_PLAN.md). [Immutable script](r3_nonuniform_q9_outer_freshdocs_seed1729.py) code commit `a4abf256c0be9db771e3ec271c1f0a5e6d605816`; [frozen prereg](research_log/r3_nonuniform_q9_outer_recovery_freshdoc_prereg_2026-10-09.md) commit `bf89baef162f2a9e948ab702a1708c74b856d3a4`.

**CPU reproducibility and data-selection gate PASSED:** [HF finalization-safe CPU preflight `6ac99425095c57808930c23c`](https://huggingface.co/jobs/codeflash85/6ac99425095c57808930c23c) **COMPLETED** 2026-10-10 01:35:09 UTC, emitted `R3_SAFE_QUANTIZER_PRECHECK_OK`, `R3_SAFE_FRESH_DOC_SELECTION_OK` and `R3_SAFE_ALL_PREFLIGHT_OK`. Checked 9 codes, W/N/H correct level outputs, same integer decisions and STE, post-Q3 switch and 32 real FineWeb docs beyond original QAT scan rows340 (source rows first343, last1222; first 32 qualifying docs ≥516 tokenizer tokens). Initial CPU job `6ac992d4095c57808930c1b0` had passed scientific tests but failed during external dataset-interpreter teardown (`PyGILState_Release`, exit134); preserved this as an infrastructure failure; a clean independent CPU pass preceded GPU submission.

**Frozen R3 arms:** direct Q3 D; original wide Q9 W (levels 0,±1/4,±1/2,±3/4,±1); narrow Q9 N (0,±1/6,±2/6,±3/6,±4/6); new nonuniform hybrid H (0,±1/6,±2/6,±3/6,±1), keeping original T4 integer assignment thresholds for all Q9 variants and original Q3 through steps301–1200. All use same seed1729, SmolLM2 source, FineWeb training order/data and optimizer resets.

**Primary:** *new QAT-heldout* 32 FineWeb documents, 512 tokens each, **per-document loss evidence**, aggregate contrasts N−H, H−W and D−H with paired descriptive per-doc statistics. Historical FineWeb21-doc and WikiText validation used as reproduction/secondary results only. Documents fresh to these QAT experiments, **NOT guaranteed absent from source model pretraining**. All previous R2 (16/16) and G1/S1/R1/F1–F3/L1 studies remain intact. **No R3 outcome available until GPU job completes and `FINAL_JSON` validity checks pass. No additional GPU job/retry authorized.**

---

## HISTORICAL: R3 PRE-GPU PREFLIGHT AND PREPARATION (2026-10-09 EDT)

**Status at this revision: preregistered and script pinned, CPU preflight submitted; GPU study NOT YET SUBMITTED pending successful preflight and Hugging Face service availability.** **No duplicate or automatic second job authorized.** [Full R3 handoff](R3_ACTIVE_JOB_PLAN.md). [Frozen R3 prereg](research_log/r3_nonuniform_q9_outer_recovery_freshdoc_prereg_2026-10-09.md), committed SHA `bf89baef162f2a9e948ab702a1708c74b856d3a4` **before script/GPU run**. [Pinned R3 script](r3_nonuniform_q9_outer_freshdocs_seed1729.py), code commit `a4abf256c0be9db771e3ec271c1f0a5e6d605816`.

**CPU preflight job:** [HF `6ac992d4095c57808930c1b0`](https://huggingface.co/jobs/codeflash85/6ac992d4095c57808930c1b0), checks whole-script AST, W/N/H exactly nine states, identical T4 code assignments, nonuniform H matching narrow levels at k0..3 but restoring wide outputs at±4, FP32 master/scale STE derivatives and return to exact Q3. Independently checks deterministic public FineWeb **32 new heldout docs** with source rows **after the initial 340 used by F1**, bucket0 hash criterion, first 32 distinct docs with ≥516 tokenizer tokens each; 512 next tokens per document, full per-doc NLL saved. This is fresh to **QAT evaluation**, not guaranteed unseen by original model pretraining.

**Scientific four-arm design:** Direct Q3 D1200; original wide W Q9(300)→Q3(900) with outputs k/4; old narrow N Q9→Q3 with outputs k/6; **hybrid H** Q9→Q3 with N's exact inner outputs k=0,±1,±2,±3 but ±4 output restored to±α, using identical T4 rounding thresholds and z/STE surrogate. All old model/data/order/teacher/LR/reset controls frozen and historic anchors D/W/N reproduced as technical checks. **Primary** is paired N−H, H−W, D−H NLL on 32 new docs; old FineWeb21 docs and WikiText validation are secondary/reproduction only. One A10G-small max2h planned **only after CPU preflight passes**.

**Current blocker:** transient Hugging Face MCP unavailability / rate-limit while inspecting CPU job. Do not claim successful preflight or launch, do not submit to paid GPU without confirmed `R3_ALL_PREFLIGHT_OK`. Inspect the preflight job first, then submit a **single** SHA-pinned GPU job if valid. Historical R2 (16/16 checks) and prior G1/S1/R1/F1–F3/L1 remain complete and untouched; R3 result not known.

---

## PRIOR VERIFIED R2 RESULT — 2×2 Q9 grid factorial COMPLETED (2026-10-10 01:00 UTC)

**R2 scientific GPU job COMPLETED SUCCESSFULLY** [HF 6ac985f0fee2c9007017f720](https://huggingface.co/jobs/codeflash85/6ac985f0fee2c9007017f720), ended **2026-10-10 01:00:05 UTC** (Oct9 9:00 PM EDT), pinned source commit `93d7aba84bff0409b8cc91603ab9605cf1a9e09d`. `valid_for_science=true`; **16/16 technical/reproduction checks passed**. [R2 detailed scientific summary](results/run_r2_fineweb_q9_output_threshold_factorial_seed1729_summary.md), [complete raw FINAL_JSON](results/run_r2_fineweb_q9_output_threshold_factorial_seed1729_2026-10-10.json), [frozen prereg](research_log/r2_q9_range_threshold_factorial_fineweb_seed1729_prereg_2026-10-09.md), [R2 job/handoff plan](R2_ACTIVE_JOB_PLAN.md). CPU quantizer preflight passed, no R2 duplicates or extra GPU jobs launched.

**R2 tests:** with nine Q9 integer codes -4..4, manipulate **V = output amplitude/spacing divisor (wide V4 ±α versus narrow V6 ±2α/3)** independently of **T = rounding/threshold multiplier (T4 original, T6 tighter and more outer-code occupancy)**, keeping same clipped-z STE, original-source row scales and fresh optimizer/GradScaler at Q9→Q3 switch after300, same seed1729 FineWeb 1200-step order/corpus and WikiText validation. Direct Q3 reference D.

| Q9 mode | FineWeb heldout NLL (lower better) | D−arm advantage | WikiText validation NLL | Q9 prep extreme ±4 fraction |
|---|---:|---:|---:|---:|
| D (direct Q3) | 5.766660 | — | 6.558239 | — |
| **W4T4** original wide, original thresholds | **4.996255** | **+0.770405** | **5.695225** | 29.30% |
| **W4T6** wide output, tighter thresholds | **5.263767** | **+0.502892** | **6.137133** | 47.87% |
| N6T4 narrow output, original thresholds | 5.780473 | −0.013813 | 6.593468 | 29.22% |
| N6T6 R1 narrow, tight thresholds | 5.751036 | +0.015624 | 6.566215 | 47.83% |

**Main new inference:** restoring original (T4) decision thresholds while keeping **narrow output grid** does **not** rescue performance (N6T4 even slightly worse than D); tightening thresholds while keeping **wide output grid** reduces but does **not erase** staging benefit (W4T6 still +0.5029 over D). The **output amplitude/spacing package** is central within this custom quantizer family; elevated **extreme-code occupancy alone** cannot explain R1 failure, since both wide and narrow arms have comparable extreme-code fractions within each T level while vastly different losses. Historical D/W4T4/N6T6 anchors reproduced original FineWeb+WikiText exactly; **all 16 checks passed**. Predefined FineWeb output-grid contrast at T4 **+0.784218 nats (narrow-wider)**, at T6 **+0.487269**; threshold contrast T6-T4 wide **+0.267512**, narrow **−0.029437**. Factorial threshold interaction = **−0.296950**.

**Limitations:** `V` simultaneously changes output range **and** uniform codebook spacing, so R2 **does not prove range alone is causally decisive**; T controls threshold/code assignment independently, not all possible geometry. One seed1729/model/schedule and the same 21 previously viewed FineWeb heldout docs, possible pretrained-model overlap; all other modules frozen and no production-level inference. Other previously completed F1–F3, L1, G1, S1, R1 remain valid archived records. Next **proposals only**: nonuniform nine-level grid separating central spacing and extreme output range, and fresh independent-document evaluation plus checkpoint retention; do not launch more GPU automatically.

---

## HISTORICAL SUBMISSION: R2 — threshold/saturation versus output range (2026-10-09 EDT)

**R2 is SUBMITTED, scientific outcome PENDING:** [HF `6ac985f0fee2c9007017f720`](https://huggingface.co/jobs/codeflash85/6ac985f0fee2c9007017f720), A10G-small, **2-hour limit**, source SHA `93d7aba84bff0409b8cc91603ab9605cf1a9e09d` ([source](r2_fineweb_q9_output_threshold_factorial_seed1729.py)). [**Full R2 job handoff**](R2_ACTIVE_JOB_PLAN.md) · [**frozen before GPU prereg**](research_log/r2_q9_range_threshold_factorial_fineweb_seed1729_prereg_2026-10-09.md).

**Scientific question:** R1's range-matched nine-state Q9 lost almost the entire staged improvement, but changing range also changed spacing, thresholds and code saturation. R2 does a controlled **2×2 factorial of Q9 output divisor V∈{4,6} and rounding threshold multiplier T∈{4,6}**, with exactly nine integer codes in every preparatory grid `k=clip(round(T*clip(w/α,-.99,.99)),-4,4)`, `q=α*[z+(k/V-z).detach()]`. Arms: `W4T4` historic wide, `W4T6` wide with tighter thresholds, `N6T4` narrow with original thresholds, `N6T6` historic R1 narrow, plus `D` original direct Q3 control. Each: same seed1729 FineWeb data/doc partition and SmolLM2 source, 1200 global steps, four Q9(300)→Q3(900) trajectories, normal original-scale/optimizer reset at300, dev+heldout FineWeb and WikiText validation. Anchors D/W4T4/N6T6 and signed interaction prespecified; actual outcomes pending.

**CPU dynamic preflight completed** [HF `6ac985c6095c57808930bb9c`](https://huggingface.co/jobs/codeflash85/6ac985c6095c57808930bb9c): `R2_PRE_GPU_PREFLIGHT_OK` plus 4× `R2_QA_ARM_OK`, validating exactly nine codes, amplitudes, code identity for same T, STE FP32/master/scale gradients, original Q3 switch and expected differing T4 vsT6 outer-code occupancy. Original R1/G1/S1 and FineWeb/F1–F3/L1 results remain archived, unchanged.

**Interpretation limitations:** The factorial separates **input threshold/saturation** from **output amplitude/spacing taken together**; cannot separately isolate amplitude from uniform-grid spacing. Single familiar seed and repeatedly evaluated 21 FineWeb document-heldout slice, pretraining overlap unknown. **No other GPU job/retry/new seed authorized.** Next AI: inspect exactly the one R2 job, parse `FINAL_JSON` and per-arm checks/contrasts, preserve raw and negative results, update reports. Older “R1 completed / no active GPU” text below is *historical*.

---

## PREVIOUS COMPLETED: R1 Q9 range-match — full staged gain largely lost (2026-10-10 UTC)

**R1 GPU COMPLETED 2026-10-10 00:05:57 UTC (Oct 9 EDT)**: [HF 6ac97c30095c57808930b904](https://huggingface.co/jobs/codeflash85/6ac97c30095c57808930b904), A10G-small, script SHA `f28f1d4814ab0a61f2b529e2add007ce142972d5`, **all 12/12 technical checks passed, `valid_for_science=true`**. [R1 detailed results](results/run_r1_q9_range_matched_fineweb_seed1729_summary.md) · [full raw JSON](results/run_r1_q9_range_matched_fineweb_seed1729_2026-10-10.json) · [frozen preregistration](research_log/r1_range_matched_q9_fineweb_seed1729_prereg_2026-10-09.md) · [job handoff](R1_ACTIVE_JOB_PLAN.md).

**Main question:** does a nine-state Q9 prepare useful ternary Q3 assignments when its output range is reduced from max ±α (original Q9) to max ±2α/3 (same as Q3), without altering the normalized STE z clamp and with integer codes constrained to exactly -4..4? **FineWeb held-out NLL:** direct Q3 **5.766660**; original/wide Q9(300)→Q3(900) **4.996255**; **range-matched nine-state Q9→Q3 5.751036**. Original wide staged gain **+0.770405**, range-matched staged gain **+0.015624** nats/token, retaining **2.03%** of the original. Secondary WikiText validation: direct **6.558239**, wide **5.695225**, range-matched **6.566215** (range-matched marginally *worse* than direct). Both original D and wide-Q9 anchors reproduced prior F1 exactly on both evaluations.

**New prep diagnostic:** at Q9 step300 **47.83%** of range-matched Q9 codes are outer ±4, versus **29.30%** wide Q9; **36.45%** of would-be `round(6z)` codes exceed code magnitude4 before integer clipping. Range-matched native Q9 train-dev NLL **6.6035** vs wide **5.4282**. Wide→narrow matches output range while necessarily changing nine-state spacing, decision boundaries and saturation/occupancy. **Do not assert “range alone has been proven to be the cause”**; R1 refutes the narrower hypothesis that any uniformly spaced nine-state Q9 grid will preserve the stage advantage. It indicates **wider effective range and/or its coupled code geometry is critical for this tested recipe**, not that all alternative range-matched nine-state schemes fail.

**Technical/source provenance:** [R1 CPU preflight](https://huggingface.co/jobs/codeflash85/6ac97b98095c57808930b8da) passed exact nine reachable codes, max output amplitude and master/alpha surrogate derivatives. R1 GPU 3 arms all1200 scheduled steps, final Q3; F1 source, seed1729, same FineWeb 21 heldout docs and WikiText validation. **One seed, reused small sample, no production inference/generation claim**. No other GPU study was initiated in the R1 analysis. G1 and S1 remain completed with prior positive controls; all former historical experiments remain archived.

**Next unrun scientific work (proposal only):** separate nine-state range restriction from code-cell spacing and saturation in a preregistered control, or evaluate genuinely fresh heldout documents/model family. Do not automatically launch a new GPU job. Older “scheduling” status descriptions below are historical and superseded.

---

## HISTORICAL R1 SUBMISSION — Q9 range-matched to Q3 (2026-10-09 UTC)

**ONE bounded scientific GPU job submitted:** [HF 6ac97c30095c57808930b904](https://huggingface.co/jobs/codeflash85/6ac97c30095c57808930b904), A10G-small, max2h, initial stage SCHEDULING. **No R1 scientific result available at submission.** Immutable script SHA `f28f1d4814ab0a61f2b529e2add007ce142972d5`; [R1 source](r1_fineweb_q9_range_matched_seed1729.py). [Frozen pre-run R1 prereg](research_log/r1_range_matched_q9_fineweb_seed1729_prereg_2026-10-09.md). [Full active job plan/handoff](R1_ACTIVE_JOB_PLAN.md).

This controls the **representable output range** question from the external review: FineWeb seed1729 1200-step `D` direct Q3, `S_wide` original Q9(300)→Q3(900) with nine values reaching ±α, and `S_range` nine-state Q9(300)→Q3(900) with nine values reaching only ±2α/3, then identical Q3 continuation. Same Q3 source row-scale restoration and AdamW/GradScaler reset at step300 for S_wide and S_range. Same `z=clamp(w/alpha,-0.99,0.99)` and STE gradient path; range-matched clips **integer code after round(6z)** to ±4, not normalized z, so code set is exactly nine. Width matching also moves intermediate state spacing/threshold locations, so **not a perfect pure range ablation**.

**Pre-run CPU dynamic gradient/code QA [6ac97b98095c57808930b8da](https://huggingface.co/jobs/codeflash85/6ac97b98095c57808930b8da)** COMPLETED. It printed `R1_QUANTIZER_TEST_OK` for both wide/range variants and `R1_CPU_PREFLIGHT_OK`: 9 states, max output amplitude, STE master-weight/scale gradients on unclipped inputs, Q3 switch behavior, script AST. The R1 three-arm GPU job has not been completed or interpreted yet. Historical G1/S1 findings remain valid and archived; **do not launch new seeds, repeat R1, or launch other jobs without user approval**.

---

## HISTORICAL COMPLETED RESULTS — G1 and S1 (2026-10-09 UTC)

**Both mechanism-control HF jobs COMPLETED with all prespecified scientific-technical checks passing. No active scientific GPU jobs were found in this pair.** [G1 full summary](results/run_g1_depth_by_scale_freeze_seed1729_summary.md) · [G1 raw JSON](results/run_g1_depth_by_scale_freeze_seed1729_2026-10-09.json) · [S1 full summary](results/run_s1_fineweb_q3_sham_reset_seed1729_summary.md) · [S1 raw JSON](results/run_s1_fineweb_q3_sham_reset_seed1729_2026-10-09.json). Both planned jobs were authorized and executed once each; **R1 range-matched Q9 remains UNRUN**.

**G1 — Q3 within-bin depth vs trainability of row scales:** [HF job 6ac96a7efee2c9007017ea64](https://huggingface.co/jobs/codeflash85/6ac96a7efee2c9007017ea64), source SHA `059a0bb91aa55fea99ed56b1d0d980eb21fd31ae`, completed **2026-10-09 23:16:07 UTC**, **6/6 checks passed**. Four 900-step continuations seeded from identically projected Q3 forward states, with 0.03 vs 0.50 in-bin depth on the D-vs-Q9 projected-code disagreement mask, learned vs exactly frozen original Q3 row scales:

| Scale policy | Depth 0.03 final WikiText test NLL | Depth 0.50 final WikiText test NLL | Shallow-minus-deep improvement |
|---|---:|---:|---:|
| Learned row scales | 5.494818 | **4.884970** | +0.609848 |
| Frozen row scales | 5.516548 | **4.896080** | +0.620468 |

**Primary frozen−learned depth-gain interaction = +0.010620 nats/token**. Every arm had 898 successful Q3 updates and 2 AMP skips, initial code projection/forward score identical. Frozen scales remained numerically unchanged and optimizer-excluded. **Conclusion: trainable Q3 row scales are NOT necessary for the large depth effect in this recipe**; stronger support for interior placement/assignment stability, without proving exact causal mediation or removing Q9 range confound.

**S1 — FineWeb Q3 sham switch:** [HF job 6ac96a82fee2c9007017ea6b](https://huggingface.co/jobs/codeflash85/6ac96a82fee2c9007017ea6b), source SHA `4cdab8ee73ddd804e96fac43992f91ab9a54c1b9`, completed **2026-10-09 23:08:23 UTC**, **9/9 checks passed**. Same seed1729 public FineWeb 1200-step corpus, document partition and LR. Direct Q3 continuous NLL **5.766660**; direct Q3 with step300 original-scale + AdamW/GradScaler reset NLL **5.789199**, **0.022539 worse**; archived Q9(300)→Q3(900) NLL **4.996255**, **0.792944 better than sham**. WikiText validation likewise sham slightly worse (6.560840 vs6.558239) while Q9 staged 5.695225. The direct-Q3 continuous anchor reproduced original F1 exactly. **Conclusion: the reset alone does not explain the FineWeb Q9 staging gain** in this seed. The data do not prove resets have no interactions with Q9 prep.

**Next research gate:** implement a fully **range-matched nine-state Q9 versus historical wide Q9** with exact clipping/STE gradients and integer-code checks (see [technical review](research_log/2026-10-09_external_technical_review_scale_range_sham_controls.md)). Then boundary-margin-matched site controls and common-original-scale final-code readouts; fresh-document evaluation and retained checkpoints. **No R1 job launched**. [Original G1/S1 frozen protocols and job information](G1_S1_ACTIVE_JOB_PLAN.md).

---

## HISTORICAL AT SUBMISSION — G1 and S1 mechanism-control jobs (2026-10-09 EDT)

**Two bounded GPU jobs submitted**, with preregistrations committed before launch; **results not available at submission**:

- **G1 depth × scale trainability**: [HF job 6ac96a7efee2c9007017ea64](https://huggingface.co/jobs/codeflash85/6ac96a7efee2c9007017ea64), one A10G-small max **2h**, pinned Git SHA `059a0bb91aa55fea99ed56b1d0d980eb21fd31ae`, [G1 source](g1_depth_by_scale_freeze_seed1729.py). Seed1729 WikiText familiar v13A preparation, four Q3 continuation arms d003/d050 × learned/frozen Q3 row scales, common global 900-step continuation. [G1 frozen prereg](research_log/g1_frozen_vs_learned_q3_scale_depth_prereg_2026-10-09.md).
- **S1 FineWeb Q3 sham-switch**: [HF job 6ac96a82fee2c9007017ea6b](https://huggingface.co/jobs/codeflash85/6ac96a82fee2c9007017ea6b), one A10G-small max **90m**, pinned Git SHA `4cdab8ee73ddd804e96fac43992f91ab9a54c1b9`, [S1 source](s1_fineweb_q3_sham_reset_seed1729.py). Seed1729 one original direct-Q3 continuous arm and one Q3 step300 sham Adam/GradScaler/original-scale reset, with unchanged FineWeb QAT document partition/1200-step LR and historical Q9 comparison. [S1 frozen prereg](research_log/s1_fineweb_direct_q3_sham_reset_seed1729_prereg_2026-10-09.md).
- **Static preflight:** HF CPU job `6ac96a6f095c57808930b267` emitted `G1_S1_STATIC_PREFLIGHT_OK`; script Python AST and invariants checked.

**Full permanent handoff:** [G1_S1_ACTIVE_JOB_PLAN.md](G1_S1_ACTIVE_JOB_PLAN.md). **Do not relaunch these jobs, change scripts, launch additional seeds, or start R1 range-matched Q9 until new user authorization.** Inspect job statuses and exact `FINAL_JSON`, archive raw findings even nulls/technical failures, then update research reports. F1/F2/F3 and L1 historical findings stay completed and preserved.

---

## HISTORICAL VERIFIED — F1/F2/F3 and L1 COMPLETED (2026-10-09 UTC)

**No currently active scientific GPU jobs in the verified HF job list.** The four most recent studies are completed (F1 seed1729, F2 seed271828, F3 seed424242, L1 WikiText 6000 steps). Each finished successfully with all preregistered technical checks passing. No new GPU trials were launched in this archival/review action.

**FineWeb-Edu three-seed exact-recipe replication:** staged Q9(300)→Q3(900) beat direct Q3(1200) on FineWeb heldout and WikiText validation in **3/3 seeds**. Mean paired FineWeb D−S loss gap **+0.763898 nats/token**, range **+0.742353 to +0.778936**; mean WikiText validation gap **+0.801917**, range +0.696902 to +0.863013. [Aggregate analysis](results/run_f1_f3_fineweb_edu_three_seed_aggregate_summary.md); [F1 raw](results/run_f1_fineweb_edu_direct_vs_staged_seed1729_2026-10-09.json); [F2 raw](results/run_f2_fineweb_edu_direct_vs_staged_seed271828_2026-10-09.json); [F3 raw](results/run_f3_fineweb_edu_direct_vs_staged_seed424242_2026-10-09.json). **Same 21 FineWeb heldout docs for all three orders.** This is RNG/order replication, *not* independent-dataset or causal-mechanism replication. F2/F3 HF job IDs [F2](https://huggingface.co/jobs/codeflash85/6ac87415fee2c90070173baf) and [F3](https://huggingface.co/jobs/codeflash85/6ac87417095c5780893020c2).

**L1 6000-step WikiText durability COMPLETE:** [HF job](https://huggingface.co/jobs/codeflash85/6ac86e1f095c578089301e53). Paired fixed-recipe direct-Q3 versus Q9→Q3, with WikiText validation staged gap **+0.716087 nats at1200** and **+0.616040 nats at6000**; direct has **not caught up at6000**. Both historical step1200 tests reproduced exactly; all five checks passed. One seed, gap narrowing but nonmonotonic; no asymptotic proof. [L1 detailed summary](results/run_l1_6000step_wikitext_durability_seed1729_summary.md) · [raw JSON](results/run_l1_6000step_wikitext_durability_seed1729_2026-10-09.json).

**Important new independent methodological review:** [six objections and matched experimental controls](research_log/2026-10-09_external_technical_review_scale_range_sham_controls.md). Highest priority for *mechanism*, before more indiscriminate GPU scaling: within-bin depth with **fixed versus trainable Q3 row scales**, and direct-Q3 **sham step300 reset** on FineWeb, then Q9 **range-matched** vs wide, matched-margin position controls, final-code same-reference diagnostics and document-linked checkpoint evidence. These controls are **proposed only**; no new GPU jobs have been launched.

**Next AI:** use this section as current. Historical submission statuses below are preserved but superseded. Before designing new work, freeze protocol, document source SHA, and budget; avoid claiming the present 3/3-seed effect independently confirms original-pretraining dataset generalization.

---

## HISTORICAL AT SUBMISSION: FineWeb-Edu F2/F3 seed replications (2026-10-09 UTC)

**User-authorized two new-seed GPU jobs, now RUNNING:**

- **F2, seed 271828:** [F2 seed271828](https://huggingface.co/jobs/codeflash85/6ac87415fee2c90070173baf); A10G-small, 90-minute maximum; immutable source SHA `64d171a225ba2781c83148a4235272c91939f9c9`, [script](f2_fineweb_edu_qat_seed271828.py).
- **F3, seed 424242:** [F3 seed424242](https://huggingface.co/jobs/codeflash85/6ac87417095c5780893020c2); A10G-small, 90-minute maximum; immutable source SHA `a20c4cc66ae16026bb90969d1160294b3f5826e9`, [script](f3_fineweb_edu_qat_seed424242.py).

[F2/F3 frozen preregistration](research_log/f2_f3_fineweb_edu_seed271828_424242_replication_prereg_2026-10-09.md) was committed **before compute**. HF CPU static smoke `6ac87407fee2c90070173b9d` emitted `F2_F3_PREFLIGHT_OK`; normalized source-code comparison passed: only seed, seed-expected permutation assertion, and provenance identifiers changed from the successful F1 seed1729 script. Exact same source-model/dataset SHAs, same FineWeb QAT document-based train/dev/eval partition and same 1200-step matched arms. **No F2/F3 scientific outcomes available yet.**

**Original F1 seed1729 result (already completed):** [summary](results/run_f1_fineweb_edu_direct_vs_staged_seed1729_summary.md) and [raw](results/run_f1_fineweb_edu_direct_vs_staged_seed1729_2026-10-09.json): staged improves held-out FineWeb NLL by **0.770405 nats** and WikiText validation NLL by **0.863013 nats**. Three-seed aggregate must wait for F2 and F3 `FINAL_JSON` and scientific checks. The heldout 21 FineWeb documents are **shared across seeds**, not independent datasets.

**Also still RUNNING at last check:** [L1 6000-step WikiText durability](https://huggingface.co/jobs/codeflash85/6ac86e1f095c578089301e53). **Do not cancel or resubmit L1.** There are exactly two *new* scientific GPU jobs in this replication request.

**Next AI:** inspect these three HF job IDs, never duplicate them. For F2/F3 parse `FINAL_JSON_BEGIN`/`FINAL_JSON_END`; verify all five checks and source-doc partition hashes match F1. Archive per-seed raw JSON, summary and three-seed aggregate (mean/median/min-max/positive count). Clearly label whether any failure was technical versus a negative staged effect; update this page, `AI_HANDOFF.md`, `EXPERIMENT.md`, README and both research reports. No additional GPU or data-search sweep authorized.

---

## HISTORICAL AT SUBMISSION: L1 WikiText running and F1 completed

**Last checked 2026-10-09 UTC:** **L1 still RUNNING** on Hugging Face: [L1 6ac86e1f095c578089301e53](https://huggingface.co/jobs/codeflash85/6ac86e1f095c578089301e53), A10G-small, 2-hour maximum, pinned script SHA `841e0ab948991a24b7397b34d90d28c38f749879`. Matched direct Q3 versus Q9(300)→Q3 extended to **6000** WikiText training opportunities with new QAT training chunks and an independent WikiText validation curve. **L1 scientific outcome remains pending. Do not interrupt or duplicate this job.**

**F1 FineWeb-Edu COMPLETE**: [F1 6ac86eb0fee2c900701738dd](https://huggingface.co/jobs/codeflash85/6ac86eb0fee2c900701738dd), completed **2026-10-09 04:48:30 UTC**, A10G-small, pinned script SHA `64dcd7f20240b4c62e6ecac8df70a1336a57bb14`. All five frozen checks passed (`valid_for_science=true`). [F1 detailed summary](results/run_f1_fineweb_edu_direct_vs_staged_seed1729_summary.md) · [raw JSON](results/run_f1_fineweb_edu_direct_vs_staged_seed1729_2026-10-09.json) · [frozen protocol](research_log/f1_fineweb_edu_qat_seed1729_prereg_2026-10-09.md).

**F1 primary FineWeb-Edu document-held-out NLL:** D direct Q3(1200) **5.766660** (PPL319.469) versus S300 Q9(300)→Q3(900) **4.996255** (PPL147.858), staged advantage **+0.770405 nats/token**. **Secondary WikiText-2 validation:** D **6.558239** vs S **5.695225**, staged advantage **+0.863013 nats/token**. F1 trained on public `FineWeb-Edu sample-10BT`, an ingredient of SmolLM2's pretraining mix, not the full original mixture. QAT document-ID sets were disjoint (train **180 docs /1200 chunks**, dev **3 docs/24 chunks**, held-out **21 docs/128 chunks**). Only one seed/order1729; possible prior source-model pretraining exposure to sampled documents unknown. Do not claim broad independent generalization, a production model, or zero contamination.

**L1/F1 frozen protocols:** [L1](research_log/l1_6000_step_durability_wikitext_seed1729_prereg_2026-10-09.md) · [F1](research_log/f1_fineweb_edu_qat_seed1729_prereg_2026-10-09.md). **Durable current job plan:** [NEXT_JOBS_PLAN.md](NEXT_JOBS_PLAN.md). The older M1 and T1/C1 jobs are all completed. No additional GPU jobs launched for F1 analysis.

**Next AI:** inspect L1, not F1, for pending scientific results. At completion fetch its `FINAL_JSON` and checks, archive all outcomes and update this state. Do not resubmit F1, silently change hyperparameters, or launch extra seeds without a new decision.

---

## Latest completed experiment — M1 matched Q3 continuation

**Status:** COMPLETED, **2026-10-09 04:15:27 UTC**. [Hugging Face job `6ac86712fee2c900701734bb`](https://huggingface.co/jobs/codeflash85/6ac86712fee2c900701734bb); pinned script `cad64c009209a924be89b523e6a1e184a4f4137b`, `m1_smol360m_equal_q3_continuation_seed1729.py`. Seven of seven preregistered construction/reproduction checks passed (`valid_for_science=true`). [M1 detailed result](results/run_m1_equal_q3_continuation_seed1729_summary.md) · [raw JSON](results/run_m1_equal_q3_continuation_seed1729_2026-10-09.json) · [frozen preregistration](research_log/m1_equal_q3_continuation_seed1729_prereg_2026-10-08.md).

One Q9 preparation trajectory, saved at steps 250 and 300; both states were assigned **the exact same 900 Q3 training examples and LR values**, with original Q3 scales and fresh Adam. Earlier state total **1150** step opportunities; later state **1200** (not an equal-compute comparison). Both Q3 continuations had **897 effective optimizer updates and three AMP skips**.

- Q9(250)+common Q3(900): **held-out loss 4.9504320**, PPL **141.236**.
- Q9(300)+common Q3(900): **held-out loss 4.9009854**, PPL **134.422**. Exact historical v10 reproduction.
- Thus **longer Q9 preparation retains a +0.0494467-nat quality advantage under a common Q3 operator** in this seed. It remains exploratory, evaluated on previously viewed WikiText tokens.
- **6,245,131 / 314,572,800 weights (1.9853%)** have different initial original-Q3-scale codes between Q9 steps 250 and 300. After continuation, earlier arm's final Q3 code matches later Q9-prep code at **44.16%** of those positions; two final Q3 models agree at **60.00%** of those selected positions, and **93.858%** globally. This suggests partial adoption of late-Q9 choices and incomplete final convergence, **not** causal proof of optimal assignments.
- M1 differs from C1's earlier 250+950 recipe (NLL 4.930515) in Q3 **length, beginning batches and LR indices**; the difference between those runs cannot be cleanly assigned to exactly 50 extra Q3 steps.

**NO new GPU job has been launched after M1.** Next scientifically highest-value gate: an independently scoped **longer-horizon paired direct vs staged fixed-recipe experiment with fresh evaluation data**, and new training orders for confirmation. Neither has yet been approved or launched. The complete handoff and interpretation plan is [NEXT_EXPERIMENT_PLAN.md](NEXT_EXPERIMENT_PLAN.md).

---

## What is established (and scoped)

- On SmolLM2-360M / WikiText-2, matched-schedule Q9(300)→Q3(900) outperforms tuned direct-Q3(1200) by mean **0.6723 nats/token** (about 49% lower PPL) across **3/3 training orders**. This is a **finite-budget** quantization-aware-training optimization result, not a usable ternary generation model, better asymptotic basin, or deployed memory/speed win.
- Mechanistic v11/v12/v13 interventions identify **which ternary weight positions/codes are chosen and moderate within-bin master placement** as major carriers of the gain, on the tested recipe. The ~6.3% D/Q9 disagreement mask recovers ~96.4% of the full gain via master transfer (3/3). Q9-only changes and direct-only changes Q9 avoids have **not** been causally separated.
- **T1 completed, seed1729**: the projected code disagreement mask emerges gradually; at Q9 step250 it overlaps **69.40%** of the eventual step300 mask (precision **77.77%**). This is retrospective geometry, **not** the proportion of important assignments.
- **C1 completed, seed1729**: direct Q3 1200 test NLL **5.595722**, Q9 250 + Q3 950 **4.930515**, Q9 300 + Q3 900 **4.900985**. S250 retains **95.75%** of S300 improvement over direct. **Exploratory on an already-studied order and evaluation slice**, not evidence that the last 50 Q9 steps are unnecessary; earlier Q9 had 50 more Q3 steps in C1. D and S300 exactly reproduced historical baselines.
- **Generalization negative/mixed:** Granite full staging 2/3 training orders and schedule sensitivity. Periodic gridward pull helps Granite in two tested orders but worsens Smol under its tuned schedule (**S1-1 negative**). Avoid claiming a universal Q9 or gridward law. Longer v5 effect shrank but persisted on one old-schedule run; asymptotic convergence is **unresolved**.
- Held-out evaluation has been reused across exploratory decisions; free generation remains poor. Published-method baselines and genuinely fresh datasets/checkpoints remain important gaps.

## Completed experiment records

| ID | HF job | Outcome |
|---|---|---|
| T1 (Q9 projected-code discovery timing) | [6ac85534fee2c90070172a41](https://huggingface.co/jobs/codeflash85/6ac85534fee2c90070172a41) | [Summary](results/run_t1_q9_discovery_timing_seed1729_summary.md); [raw JSON](results/run_t1_q9_discovery_timing_seed1729_2026-10-08.json). Completed. |
| C1 (Q9 switch 250 vs300, equal total budget) | [6ac8597afee2c90070172c79](https://huggingface.co/jobs/codeflash85/6ac8597afee2c90070172c79) | [Summary](results/run_c1_smol360m_q9_switch250_vs300_seed1729_summary.md); [raw JSON](results/run_c1_smol360m_q9_switch250_vs300_seed1729_2026-10-08.json). Completed. |
| M1 (250 vs300 Q9 prep with *identical* Q3 continuation) | [6ac86712fee2c900701734bb](https://huggingface.co/jobs/codeflash85/6ac86712fee2c900701734bb) | [Summary](results/run_m1_equal_q3_continuation_seed1729_summary.md); [raw JSON](results/run_m1_equal_q3_continuation_seed1729_2026-10-09.json). Completed, checks passed. |

## Current next action and constraints

**No immediate compute queued.** The paired 250-vs-300 controls, both equal-total (C1) and equal-Q3-continuation (M1), have now been run on seed1729, which has repeatedly informed the experiment design. The most consequential open questions are **durability (does direct Q3 catch up at longer training horizons?), fresh evaluation, and cross-seed/family confirmation**. Decide a budget and freeze an independent next protocol before any paid compute; do not repeatedly tune using the known 8192-token WikiText test slice.

**Next AI:** Read [NEXT_EXPERIMENT_PLAN.md](NEXT_EXPERIMENT_PLAN.md) for hypotheses/controls, this current-state summary for latest status, [AI_HANDOFF.md](AI_HANDOFF.md) for chronology and [EXPERIMENT.md](EXPERIMENT.md) for historical protocols. M1 must **not** be relaunched. First check Hugging Face for any newer job before planning. Record exact source SHA, prereg, job ID, raw output, validity and negatives for each new experiment.

## Key limitations and competing explanations

The remaining scientific choice is not simply *when the step300 disagreement mask appears*. There may be early high-value decisions, Q3 recovery of missing decisions, alternative good assignment patterns, or compensation from extra Q3 steps. External AI reviewers observed that T1+C1 cannot distinguish these. They also identified a crucial durability question: does direct Q3 catch up with more training? These are **hypotheses** to test, not findings.

**Paper/publication boundary:** A strong controlled assignment-selection observation exists in the specified pilot. No claim of portable ternary QAT or production-ready compression is justified yet. Separate confirmation on new orders/data and a longer fixed-recipe comparison are priorities, **not automatically authorized jobs**.
