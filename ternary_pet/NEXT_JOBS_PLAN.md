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
**HF GPU job COMPLETED:** [6ac86eb0fee2c900701738dd](https://huggingface.co/jobs/codeflash85/6ac86eb0fee2c900701738dd), A10G-small, completed 2026-10-09 04:48:30 UTC, five of five preregistered design checks passed, \`valid_for_science=true\`. [F1 detailed results](results/run_f1_fineweb_edu_direct_vs_staged_seed1729_summary.md) · [F1 raw JSON](results/run_f1_fineweb_edu_direct_vs_staged_seed1729_2026-10-09.json). **F1 primary FineWeb NLL** D5.766660 vs S4.996255 (+0.770405 staged); **WikiText validation** D6.558239 vs S5.695225 (+0.863013 staged). One seed, 21 heldout FineWeb docs, pretraining overlap unknown. CPU dataset-partition integrity test `6ac86e0d095c578089301e45` **COMPLETED**, producing train1200/dev24/heldout128 chunks from 340 scanned documents with disjoint document IDs.

Same *original 1200-step* Q9/Q3 recipe, but WikiText2 QAT training replaced by public `HuggingFaceFW/fineweb-edu` `sample-10BT` sample. FineWeb-Edu was part of the pretraining mixture of `SmolLM2-360M` according to its official model card. This experiment does **not** recreate SmolLM2's entire 4-trillion-token mixed pretraining or Instruct fine-tuning recipe. Keep model/objective/quantizer/training budget fixed **within F1**.

Train/dev/heldout FineWeb-Edu sets are drawn from deterministic **SHA256(document ID) modulo20** buckets to prevent source-document overlap across QAT sets; a second endpoint uses the not-previously-used WikiText2 validation split. Raw FineWeb content may have been seen by the **base model during pretraining**; do not claim independently never-seen text relative to the source model. Freeze dataset/model SHAs as stated in F1 prereg.

**What F1 answers:** whether Q9→Q3 vs direct Q3 behaves similarly when QAT uses a corpus closer to the model's pretraining distribution. A different validation outcome cannot be attributed exclusively to corpus if model/budget also change; within F1 arms all those factors are fixed.

## Preflight and reproducibility provenance

- Official SmolLM2-360M-Instruct model revision `a10cc1512eabd3dde888204e902eca88bddb4951`.
- WikiText2 revision `b08601e04326c79dfdd32d625aee71d232d685c3`.
- FineWeb-Edu revision `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9`.
- CPU source-access preflight `6ac86d5e095c578089301e06` confirmed metadata, tokenizer and source documents (`DATA_SOURCE_PREFLIGHT_OK`).
- Static checks `6ac86e01fee2c900701738a9` printed `BOTH_STATIC_QA_OK` from pinned scripts.
- Full FineWeb document-partition preflight (CPU-basic) `6ac86e0d095c578089301e45` **COMPLETED** with `F1_PARTITION_PREFLIGHT_OK` and document-disjoint chunk counts train1200/dev24/eval128. F1 GPU COMPLETED; do not duplicate. L1 still RUNNING at last check; analyze it separately when finished.

## Required after completion

1. Confirm HF terminal stage **COMPLETED** / failure and retrieve full terminal `FINAL_JSON_BEGIN`…`FINAL_JSON_END` from each job.
2. Validate the frozen checks, including L1 step1200 historical control reproduction and F1 document disjointness; distinguish technical failure from scientific null.
3. Archive exact result JSON + provenance in `ternary_pet/results/`, add concise descriptive summary, update `CURRENT_STATE.md`, `AI_HANDOFF.md`, `EXPERIMENT.md`, `RESEARCH_REPORT.md`, `SHAREABLE_RESEARCH_REPORT.md` and `README.md`.
4. Preserve all null or negative findings; do not quietly relaunch failed jobs, adjust LR, choose new datasets, add seeds or treat results as a paper-ready replication.

## Active follow-up F2/F3 — replicate FineWeb-Edu on two other seeds

**Frozen prereg:** [F2/F3 frozen preregistration](research_log/f2_f3_fineweb_edu_seed271828_424242_replication_prereg_2026-10-09.md).

- F2: seed **271828**, [F2 seed271828](https://huggingface.co/jobs/codeflash85/6ac87415fee2c90070173baf), pinned source `64d171a225ba2781c83148a4235272c91939f9c9`.
- F3: seed **424242**, [F3 seed424242](https://huggingface.co/jobs/codeflash85/6ac87417095c5780893020c2), pinned source `a20c4cc66ae16026bb90969d1160294b3f5826e9`.
- Both one A10G-small each, 90m timeout, last inspected **RUNNING**. CPU preflight `6ac87407fee2c90070173b9d` passed.
- Seed1729 historical F1 remains [here](results/run_f1_fineweb_edu_direct_vs_staged_seed1729_summary.md). The same FineWeb document split and evaluation data are used for **all three seeds**; test only training-order and RNG robustness.
- At completion validate all F1 protocol invariants; save raw files and summarize paired FineWeb and WikiText-validation NLL and PPL by seed, mean, median, range, positive counts. Avoid independent dataset claims.
- L1 WikiText6k still running; do not submit duplicates or auto-launch further jobs.

## Latest completed studies and external mechanism-review follow-up (2026-10-09 UTC)

**All F1/F2/F3 FineWeb seeds and L1 WikiText6000 job have COMPLETED**, all final scientific validity flags passed. No research jobs currently active in the last verified HF listing. **FineWeb aggregate:** Q9→Q3 beats direct Q3 in **3/3 orders**, mean held-out advantage **+0.763898 nats/token** (range +0.742353..+0.778936), WikiText validation mean +0.801917. Identical 21 FineWeb evaluation documents across orders, *not independent validation samples*. [3-seed aggregate](results/run_f1_f3_fineweb_edu_three_seed_aggregate_summary.md) · [F2 raw](results/run_f2_fineweb_edu_direct_vs_staged_seed271828_2026-10-09.json) · [F3 raw](results/run_f3_fineweb_edu_direct_vs_staged_seed424242_2026-10-09.json).

**L1 six-thousand-step:** [result summary](results/run_l1_6000step_wikitext_durability_seed1729_summary.md) · [raw](results/run_l1_6000step_wikitext_durability_seed1729_2026-10-09.json). On fixed WikiText validation, staged advantage declines **+0.716087 @1200 → +0.616040 nats/token @6000** (nonmonotonic), so direct Q3 hasn't caught up by6000. Reproduced both step1200 historical tests exactly. This is one long training order, not a convergence proof.

**Separate independent review of unresolved mechanism controls:** [frozen-scale depth / Q9 range match / FineWeb sham switch / margin-matched positions / common-scale code comparisons / document-linked evidence](research_log/2026-10-09_external_technical_review_scale_range_sham_controls.md). Reviewing `smollm2_v12_code_identity_position.py` confirms scale-gradient confounding is real; v13's nearly-equal fixed/learned alpha *measurement* of code survival does not eliminate a scale-*training-gradient* mechanism. Next recommended controls **G1 frozen-vs-learned-scale depth** and **S1 FineWeb direct-Q3 sham step300 reset**, followed by range-matched Q9. No control GPU jobs launched as part of this review. [Authoritative current state](CURRENT_STATE.md) takes precedence over historical “running” sections.

---

## NOW ACTIVE — G1 and S1 mechanism-control jobs (2026-10-09 EDT)

**Two bounded GPU jobs submitted**, with preregistrations committed before launch; **results not available at submission**:

- **G1 depth × scale trainability**: [HF job 6ac96a7efee2c9007017ea64](https://huggingface.co/jobs/codeflash85/6ac96a7efee2c9007017ea64), one A10G-small max **2h**, pinned Git SHA `059a0bb91aa55fea99ed56b1d0d980eb21fd31ae`, [G1 source](g1_depth_by_scale_freeze_seed1729.py). Seed1729 WikiText familiar v13A preparation, four Q3 continuation arms d003/d050 × learned/frozen Q3 row scales, common global 900-step continuation. [G1 frozen prereg](research_log/g1_frozen_vs_learned_q3_scale_depth_prereg_2026-10-09.md).
- **S1 FineWeb Q3 sham-switch**: [HF job 6ac96a82fee2c9007017ea6b](https://huggingface.co/jobs/codeflash85/6ac96a82fee2c9007017ea6b), one A10G-small max **90m**, pinned Git SHA `4cdab8ee73ddd804e96fac43992f91ab9a54c1b9`, [S1 source](s1_fineweb_q3_sham_reset_seed1729.py). Seed1729 one original direct-Q3 continuous arm and one Q3 step300 sham Adam/GradScaler/original-scale reset, with unchanged FineWeb QAT document partition/1200-step LR and historical Q9 comparison. [S1 frozen prereg](research_log/s1_fineweb_direct_q3_sham_reset_seed1729_prereg_2026-10-09.md).
- **Static preflight:** HF CPU job `6ac96a6f095c57808930b267` emitted `G1_S1_STATIC_PREFLIGHT_OK`; script Python AST and invariants checked.

**Full permanent handoff:** [G1_S1_ACTIVE_JOB_PLAN.md](G1_S1_ACTIVE_JOB_PLAN.md). **Do not relaunch these jobs, change scripts, launch additional seeds, or start R1 range-matched Q9 until new user authorization.** Inspect job statuses and exact `FINAL_JSON`, archive raw findings even nulls/technical failures, then update research reports. F1/F2/F3 and L1 historical findings stay completed and preserved.

---

## R1 newly submitted — Q9 range-matched to Q3 (2026-10-09 UTC)

**ONE bounded scientific GPU job submitted:** [HF 6ac97c30095c57808930b904](https://huggingface.co/jobs/codeflash85/6ac97c30095c57808930b904), A10G-small, max2h, initial stage SCHEDULING. **No R1 scientific result available at submission.** Immutable script SHA `f28f1d4814ab0a61f2b529e2add007ce142972d5`; [R1 source](r1_fineweb_q9_range_matched_seed1729.py). [Frozen pre-run R1 prereg](research_log/r1_range_matched_q9_fineweb_seed1729_prereg_2026-10-09.md). [Full active job plan/handoff](R1_ACTIVE_JOB_PLAN.md).

This controls the **representable output range** question from the external review: FineWeb seed1729 1200-step `D` direct Q3, `S_wide` original Q9(300)→Q3(900) with nine values reaching ±α, and `S_range` nine-state Q9(300)→Q3(900) with nine values reaching only ±2α/3, then identical Q3 continuation. Same Q3 source row-scale restoration and AdamW/GradScaler reset at step300 for S_wide and S_range. Same `z=clamp(w/alpha,-0.99,0.99)` and STE gradient path; range-matched clips **integer code after round(6z)** to ±4, not normalized z, so code set is exactly nine. Width matching also moves intermediate state spacing/threshold locations, so **not a perfect pure range ablation**.

**Pre-run CPU dynamic gradient/code QA [6ac97b98095c57808930b8da](https://huggingface.co/jobs/codeflash85/6ac97b98095c57808930b8da)** COMPLETED. It printed `R1_QUANTIZER_TEST_OK` for both wide/range variants and `R1_CPU_PREFLIGHT_OK`: 9 states, max output amplitude, STE master-weight/scale gradients on unclipped inputs, Q3 switch behavior, script AST. The R1 three-arm GPU job has not been completed or interpreted yet. Historical G1/S1 findings remain valid and archived; **do not launch new seeds, repeat R1, or launch other jobs without user approval**.

---

## NEW ACTIVE R2 — threshold/saturation versus output range (2026-10-09 EDT)

**R2 is SUBMITTED, scientific outcome PENDING:** [HF `6ac985f0fee2c9007017f720`](https://huggingface.co/jobs/codeflash85/6ac985f0fee2c9007017f720), A10G-small, **2-hour limit**, source SHA `93d7aba84bff0409b8cc91603ab9605cf1a9e09d` ([source](r2_fineweb_q9_output_threshold_factorial_seed1729.py)). [**Full R2 job handoff**](R2_ACTIVE_JOB_PLAN.md) · [**frozen before GPU prereg**](research_log/r2_q9_range_threshold_factorial_fineweb_seed1729_prereg_2026-10-09.md).

**Scientific question:** R1's range-matched nine-state Q9 lost almost the entire staged improvement, but changing range also changed spacing, thresholds and code saturation. R2 does a controlled **2×2 factorial of Q9 output divisor V∈{4,6} and rounding threshold multiplier T∈{4,6}**, with exactly nine integer codes in every preparatory grid `k=clip(round(T*clip(w/α,-.99,.99)),-4,4)`, `q=α*[z+(k/V-z).detach()]`. Arms: `W4T4` historic wide, `W4T6` wide with tighter thresholds, `N6T4` narrow with original thresholds, `N6T6` historic R1 narrow, plus `D` original direct Q3 control. Each: same seed1729 FineWeb data/doc partition and SmolLM2 source, 1200 global steps, four Q9(300)→Q3(900) trajectories, normal original-scale/optimizer reset at300, dev+heldout FineWeb and WikiText validation. Anchors D/W4T4/N6T6 and signed interaction prespecified; actual outcomes pending.

**CPU dynamic preflight completed** [HF `6ac985c6095c57808930bb9c`](https://huggingface.co/jobs/codeflash85/6ac985c6095c57808930bb9c): `R2_PRE_GPU_PREFLIGHT_OK` plus 4× `R2_QA_ARM_OK`, validating exactly nine codes, amplitudes, code identity for same T, STE FP32/master/scale gradients, original Q3 switch and expected differing T4 vsT6 outer-code occupancy. Original R1/G1/S1 and FineWeb/F1–F3/L1 results remain archived, unchanged.

**Interpretation limitations:** The factorial separates **input threshold/saturation** from **output amplitude/spacing taken together**; cannot separately isolate amplitude from uniform-grid spacing. Single familiar seed and repeatedly evaluated 21 FineWeb document-heldout slice, pretraining overlap unknown. **No other GPU job/retry/new seed authorized.** Next AI: inspect exactly the one R2 job, parse `FINAL_JSON` and per-arm checks/contrasts, preserve raw and negative results, update reports. Older “R1 completed / no active GPU” text below is *historical*.

---

## R2 COMPLETED — output grid size versus threshold/saturation confound (2026-10-10 UTC)

**R2 COMPLETED VALID**, [HF GPU job 6ac985f0fee2c9007017f720](https://huggingface.co/jobs/codeflash85/6ac985f0fee2c9007017f720), finished **2026-10-10 01:00:05 UTC**, pinned script SHA `93d7aba84bff0409b8cc91603ab9605cf1a9e09d`. All **16/16 technical and reproduction controls passed**. [Full original FINAL_JSON with job metadata](results/run_r2_fineweb_q9_output_threshold_factorial_seed1729_2026-10-10.json) · [Detailed analysis](results/run_r2_fineweb_q9_output_threshold_factorial_seed1729_summary.md) · [preregistered plan](research_log/r2_q9_range_threshold_factorial_fineweb_seed1729_prereg_2026-10-09.md) · [durable R2 handoff](R2_ACTIVE_JOB_PLAN.md).

**R2 intervention:** seed1729 matched FineWeb-Edu 1200 opportunities, direct-Q3 reference plus four Q9(300)→Q3(900) variants arranged by independent normalized rounding threshold multiplier `T=4` vs `6` and Q9 output divisor `V=4` (wide max ±α) vs `6` (narrow max ±2α/3), all exactly nine codes with the original clipped-z STE gradient surrogate. All four Q9 stages reset source Q3 row scales and Adam/GradScaler identically and use the same fixed teacher/data/LR.

| R2 approach | FineWeb 21-doc heldout NLL ↓ | D−Q9 FineWeb advantage | WikiText validation NLL ↓ | Fraction outer ±4 at prep300 |
|---|---:|---:|---:|---:|
| D direct Q3 | 5.766660 | — | 6.558239 | — |
| W4T4 wide/original | **4.996255** | **+0.770405** | 5.695225 | 29.30% |
| W4T6 wide/tighter thresholds | 5.263767 | +0.502892 | 6.137133 | 47.87% |
| N6T4 narrow/original thresholds | 5.780473 | −0.013813 | 6.593468 | 29.22% |
| N6T6 narrow/tighter thresholds | 5.751036 | +0.015624 | 6.566215 | 47.83% |

**Core finding:** R1's narrow-Q9 failure does **not** arise *solely* from increasing saturation/outer code assignment. W4T6 retains a large benefit despite ~48% outer code prevalence; N6T4 lacks a gain despite ~29% outer prevalence. Holding the threshold multiplier fixed, switching wide→narrow worsens FineWeb NLL **+0.784218** at T4 and **+0.487269** at T6. Tightening thresholds T4→T6 worsens wide-Q9 **+0.267512**, whereas narrow-Q9 changes **−0.029437**; interaction **−0.296950**. All 3 historical D/W4T4/N6T6 anchors reproduced exactly. **Crucial caveat:** the V factor changes **both output range and equally spaced codebook spacing**; the result does **not** isolate range amplitude alone and doesn't guarantee effect on other models/datasets.

**Interpretation bounds:** Same small **21 FineWeb heldout documents**, same familiar seed1729, original pretraining overlap unknown, WikiText validation also previously reused. No confidence intervals over new documents. Archived raw scores and technical negatives preserved; no extra GPU job, retuning or new seed launched during result analysis. Next possible discriminant: nonuniform Q9 codebook independently varying extreme output values versus central spacing; or fresh-doc/model-family confirmation. Both are **proposed only**. [CURRENT_STATE.md](CURRENT_STATE.md) takes precedence over historical “running/submitted” text.

---

## R3 AUTHORIZED — nonuniform Q9 extreme recovery plus fresh document-level heldout (2026-10-09 EDT)

**Status at this revision: preregistered and script pinned, CPU preflight submitted; GPU study NOT YET SUBMITTED pending successful preflight and Hugging Face service availability.** **No duplicate or automatic second job authorized.** [Full R3 handoff](R3_ACTIVE_JOB_PLAN.md). [Frozen R3 prereg](research_log/r3_nonuniform_q9_outer_recovery_freshdoc_prereg_2026-10-09.md), committed SHA `bf89baef162f2a9e948ab702a1708c74b856d3a4` **before script/GPU run**. [Pinned R3 script](r3_nonuniform_q9_outer_freshdocs_seed1729.py), code commit `a4abf256c0be9db771e3ec271c1f0a5e6d605816`.

**CPU preflight job:** [HF `6ac992d4095c57808930c1b0`](https://huggingface.co/jobs/codeflash85/6ac992d4095c57808930c1b0), checks whole-script AST, W/N/H exactly nine states, identical T4 code assignments, nonuniform H matching narrow levels at k0..3 but restoring wide outputs at±4, FP32 master/scale STE derivatives and return to exact Q3. Independently checks deterministic public FineWeb **32 new heldout docs** with source rows **after the initial 340 used by F1**, bucket0 hash criterion, first 32 distinct docs with ≥516 tokenizer tokens each; 512 next tokens per document, full per-doc NLL saved. This is fresh to **QAT evaluation**, not guaranteed unseen by original model pretraining.

**Scientific four-arm design:** Direct Q3 D1200; original wide W Q9(300)→Q3(900) with outputs k/4; old narrow N Q9→Q3 with outputs k/6; **hybrid H** Q9→Q3 with N's exact inner outputs k=0,±1,±2,±3 but ±4 output restored to±α, using identical T4 rounding thresholds and z/STE surrogate. All old model/data/order/teacher/LR/reset controls frozen and historic anchors D/W/N reproduced as technical checks. **Primary** is paired N−H, H−W, D−H NLL on 32 new docs; old FineWeb21 docs and WikiText validation are secondary/reproduction only. One A10G-small max2h planned **only after CPU preflight passes**.

**Current blocker:** transient Hugging Face MCP unavailability / rate-limit while inspecting CPU job. Do not claim successful preflight or launch, do not submit to paid GPU without confirmed `R3_ALL_PREFLIGHT_OK`. Inspect the preflight job first, then submit a **single** SHA-pinned GPU job if valid. Historical R2 (16/16 checks) and prior G1/S1/R1/F1–F3/L1 remain complete and untouched; R3 result not known.

---

## R3 LAUNCHED — nonuniform Q9 outer-level rescue and fresh 32-document evaluation (2026-10-10 UTC)

**R3 single scientific GPU job ACCEPTED:** [HF `6ac99666fee2c9007018011f`](https://huggingface.co/jobs/codeflash85/6ac99666fee2c9007018011f), `a10g-small`, max **2 hours**, initial **SCHEDULING** status. One scientific R3 GPU job only, no seed sweep. [Exact R3 job and archival handoff](R3_ACTIVE_JOB_PLAN.md). [Immutable script](r3_nonuniform_q9_outer_freshdocs_seed1729.py) code commit `a4abf256c0be9db771e3ec271c1f0a5e6d605816`; [frozen prereg](research_log/r3_nonuniform_q9_outer_recovery_freshdoc_prereg_2026-10-09.md) commit `bf89baef162f2a9e948ab702a1708c74b856d3a4`.

**CPU reproducibility and data-selection gate PASSED:** [HF finalization-safe CPU preflight `6ac99425095c57808930c23c`](https://huggingface.co/jobs/codeflash85/6ac99425095c57808930c23c) **COMPLETED** 2026-10-10 01:35:09 UTC, emitted `R3_SAFE_QUANTIZER_PRECHECK_OK`, `R3_SAFE_FRESH_DOC_SELECTION_OK` and `R3_SAFE_ALL_PREFLIGHT_OK`. Checked 9 codes, W/N/H correct level outputs, same integer decisions and STE, post-Q3 switch and 32 real FineWeb docs beyond original QAT scan rows340 (source rows first343, last1222; first 32 qualifying docs ≥516 tokenizer tokens). Initial CPU job `6ac992d4095c57808930c1b0` had passed scientific tests but failed during external dataset-interpreter teardown (`PyGILState_Release`, exit134); preserved this as an infrastructure failure; a clean independent CPU pass preceded GPU submission.

**Frozen R3 arms:** direct Q3 D; original wide Q9 W (levels 0,±1/4,±1/2,±3/4,±1); narrow Q9 N (0,±1/6,±2/6,±3/6,±4/6); new nonuniform hybrid H (0,±1/6,±2/6,±3/6,±1), keeping original T4 integer assignment thresholds for all Q9 variants and original Q3 through steps301–1200. All use same seed1729, SmolLM2 source, FineWeb training order/data and optimizer resets.

**Primary:** *new QAT-heldout* 32 FineWeb documents, 512 tokens each, **per-document loss evidence**, aggregate contrasts N−H, H−W and D−H with paired descriptive per-doc statistics. Historical FineWeb21-doc and WikiText validation used as reproduction/secondary results only. Documents fresh to these QAT experiments, **NOT guaranteed absent from source model pretraining**. All previous R2 (16/16) and G1/S1/R1/F1–F3/L1 studies remain intact. **No R3 outcome available until GPU job completes and `FINAL_JSON` validity checks pass. No additional GPU job/retry authorized.**

---

