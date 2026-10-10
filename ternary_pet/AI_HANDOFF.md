# AI HANDOFF — Ternary Pet Quantization Research

## R3 AUTHORIZED — nonuniform Q9 extreme recovery plus fresh document-level heldout (2026-10-09 EDT)

**Status at this revision: preregistered and script pinned, CPU preflight submitted; GPU study NOT YET SUBMITTED pending successful preflight and Hugging Face service availability.** **No duplicate or automatic second job authorized.** [Full R3 handoff](R3_ACTIVE_JOB_PLAN.md). [Frozen R3 prereg](research_log/r3_nonuniform_q9_outer_recovery_freshdoc_prereg_2026-10-09.md), committed SHA `bf89baef162f2a9e948ab702a1708c74b856d3a4` **before script/GPU run**. [Pinned R3 script](r3_nonuniform_q9_outer_freshdocs_seed1729.py), code commit `a4abf256c0be9db771e3ec271c1f0a5e6d605816`.

**CPU preflight job:** [HF `6ac992d4095c57808930c1b0`](https://huggingface.co/jobs/codeflash85/6ac992d4095c57808930c1b0), checks whole-script AST, W/N/H exactly nine states, identical T4 code assignments, nonuniform H matching narrow levels at k0..3 but restoring wide outputs at±4, FP32 master/scale STE derivatives and return to exact Q3. Independently checks deterministic public FineWeb **32 new heldout docs** with source rows **after the initial 340 used by F1**, bucket0 hash criterion, first 32 distinct docs with ≥516 tokenizer tokens each; 512 next tokens per document, full per-doc NLL saved. This is fresh to **QAT evaluation**, not guaranteed unseen by original model pretraining.

**Scientific four-arm design:** Direct Q3 D1200; original wide W Q9(300)→Q3(900) with outputs k/4; old narrow N Q9→Q3 with outputs k/6; **hybrid H** Q9→Q3 with N's exact inner outputs k=0,±1,±2,±3 but ±4 output restored to±α, using identical T4 rounding thresholds and z/STE surrogate. All old model/data/order/teacher/LR/reset controls frozen and historic anchors D/W/N reproduced as technical checks. **Primary** is paired N−H, H−W, D−H NLL on 32 new docs; old FineWeb21 docs and WikiText validation are secondary/reproduction only. One A10G-small max2h planned **only after CPU preflight passes**.

**Current blocker:** transient Hugging Face MCP unavailability / rate-limit while inspecting CPU job. Do not claim successful preflight or launch, do not submit to paid GPU without confirmed `R3_ALL_PREFLIGHT_OK`. Inspect the preflight job first, then submit a **single** SHA-pinned GPU job if valid. Historical R2 (16/16 checks) and prior G1/S1/R1/F1–F3/L1 remain complete and untouched; R3 result not known.

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


## NEW ACTIVE R2 — threshold/saturation versus output range (2026-10-09 EDT)

**R2 is SUBMITTED, scientific outcome PENDING:** [HF `6ac985f0fee2c9007017f720`](https://huggingface.co/jobs/codeflash85/6ac985f0fee2c9007017f720), A10G-small, **2-hour limit**, source SHA `93d7aba84bff0409b8cc91603ab9605cf1a9e09d` ([source](r2_fineweb_q9_output_threshold_factorial_seed1729.py)). [**Full R2 job handoff**](R2_ACTIVE_JOB_PLAN.md) · [**frozen before GPU prereg**](research_log/r2_q9_range_threshold_factorial_fineweb_seed1729_prereg_2026-10-09.md).

**Scientific question:** R1's range-matched nine-state Q9 lost almost the entire staged improvement, but changing range also changed spacing, thresholds and code saturation. R2 does a controlled **2×2 factorial of Q9 output divisor V∈{4,6} and rounding threshold multiplier T∈{4,6}**, with exactly nine integer codes in every preparatory grid `k=clip(round(T*clip(w/α,-.99,.99)),-4,4)`, `q=α*[z+(k/V-z).detach()]`. Arms: `W4T4` historic wide, `W4T6` wide with tighter thresholds, `N6T4` narrow with original thresholds, `N6T6` historic R1 narrow, plus `D` original direct Q3 control. Each: same seed1729 FineWeb data/doc partition and SmolLM2 source, 1200 global steps, four Q9(300)→Q3(900) trajectories, normal original-scale/optimizer reset at300, dev+heldout FineWeb and WikiText validation. Anchors D/W4T4/N6T6 and signed interaction prespecified; actual outcomes pending.

**CPU dynamic preflight completed** [HF `6ac985c6095c57808930bb9c`](https://huggingface.co/jobs/codeflash85/6ac985c6095c57808930bb9c): `R2_PRE_GPU_PREFLIGHT_OK` plus 4× `R2_QA_ARM_OK`, validating exactly nine codes, amplitudes, code identity for same T, STE FP32/master/scale gradients, original Q3 switch and expected differing T4 vsT6 outer-code occupancy. Original R1/G1/S1 and FineWeb/F1–F3/L1 results remain archived, unchanged.

**Interpretation limitations:** The factorial separates **input threshold/saturation** from **output amplitude/spacing taken together**; cannot separately isolate amplitude from uniform-grid spacing. Single familiar seed and repeatedly evaluated 21 FineWeb document-heldout slice, pretraining overlap unknown. **No other GPU job/retry/new seed authorized.** Next AI: inspect exactly the one R2 job, parse `FINAL_JSON` and per-arm checks/contrasts, preserve raw and negative results, update reports. Older “R1 completed / no active GPU” text below is *historical*.

---


## R1 COMPLETED — range-matched nine-state Q9 nearly erases staging gain (2026-10-10 UTC)

**Valid scientific R1**: [HF job](https://huggingface.co/jobs/codeflash85/6ac97c30095c57808930b904), finished **2026-10-10 00:05:57 UTC**, source SHA `f28f1d4814ab0a61f2b529e2add007ce142972d5`. **12/12 technical/reproduction checks passed**, including *exact* archived FineWeb F1 direct-Q3 and original wide Q9→Q3 heldout NLL reproductions. [R1 full raw JSON](results/run_r1_q9_range_matched_fineweb_seed1729_2026-10-10.json) · [R1 detailed summary](results/run_r1_q9_range_matched_fineweb_seed1729_summary.md) · [original prereg](research_log/r1_range_matched_q9_fineweb_seed1729_prereg_2026-10-09.md) · [R1 handoff](R1_ACTIVE_JOB_PLAN.md).

**R1 question:** is having nine intermediate levels enough, or is the originally wider Q9 representation (max ±α) important compared with Q3's max ±2α/3? A new nine-state range-matched Q9 uses `q=α*clamp(round(6*clip(w/α,-.99,.99)),-4,4)/6`, preserving nine codes, normalized `z` outer clip and surrogate gradient formula, but deliberately compressing output range to ±2α/3. All arms use the same seed1729 FineWeb `sample-10BT`, original model source, 300 Q9 +900 Q3 continuation/switch, and fixed heldout evaluation.

| Endpoint | Direct Q3 | Original wide Q9→Q3 | Range-matched Q9→Q3 |
|---|---:|---:|---:|
| FineWeb doc-heldout NLL ↓ | 5.766660 | **4.996255** | 5.751036 |
| WikiText validation NLL ↓ | 6.558239 | **5.695225** | 6.566215 |

FineWeb wide-Q9 gain `D−wide` **+0.770405 nats/token**; range-matched gain `D−range` **+0.015624** (only **2.03%** of original). Range-matched was 0.0080 nats **worse than direct** on WikiText validation; original wide gain there **+0.863013**.

**Crucial confound still present:** A uniformly spaced nine-state grid constrained to Q3's range also changes quantization cell width, internal thresholds and code saturation. At Q9 preparation300 the proportion of extreme code ±4 was **47.83%** range-matched vs **29.30%** wide; **36.45%** of range-matched would-be raw round(6z) codes exceed ±4 and are clipped. We can state **the wider output range and/or its coupled code geometry is necessary for *this particular range-matched implementation* to retain the effect**, not that range alone caused it or any narrow-range nine-level method must fail. The new S_range already has worse Q9-native train-dev loss, not merely larger switch shock.

**Scope:** single familiar seed1729 and 21 small FineWeb heldout source documents, pretrained-model document exposure unknown, no restored model checkpoint, no production-generation claim. Previous G1 frozen-scale and S1 sham-reset controls passed and support a meaningful optimization-path/assignment-stability story but do not render R1 a pure isolation of range. **No additional GPU job was launched after R1.** [CURRENT_STATE.md](CURRENT_STATE.md) is authoritative; older “R1 submitted” sections are historical.

---


## R1 newly submitted — Q9 range-matched to Q3 (2026-10-09 UTC)

**ONE bounded scientific GPU job submitted:** [HF 6ac97c30095c57808930b904](https://huggingface.co/jobs/codeflash85/6ac97c30095c57808930b904), A10G-small, max2h, initial stage SCHEDULING. **No R1 scientific result available at submission.** Immutable script SHA `f28f1d4814ab0a61f2b529e2add007ce142972d5`; [R1 source](r1_fineweb_q9_range_matched_seed1729.py). [Frozen pre-run R1 prereg](research_log/r1_range_matched_q9_fineweb_seed1729_prereg_2026-10-09.md). [Full active job plan/handoff](R1_ACTIVE_JOB_PLAN.md).

This controls the **representable output range** question from the external review: FineWeb seed1729 1200-step `D` direct Q3, `S_wide` original Q9(300)→Q3(900) with nine values reaching ±α, and `S_range` nine-state Q9(300)→Q3(900) with nine values reaching only ±2α/3, then identical Q3 continuation. Same Q3 source row-scale restoration and AdamW/GradScaler reset at step300 for S_wide and S_range. Same `z=clamp(w/alpha,-0.99,0.99)` and STE gradient path; range-matched clips **integer code after round(6z)** to ±4, not normalized z, so code set is exactly nine. Width matching also moves intermediate state spacing/threshold locations, so **not a perfect pure range ablation**.

**Pre-run CPU dynamic gradient/code QA [6ac97b98095c57808930b8da](https://huggingface.co/jobs/codeflash85/6ac97b98095c57808930b8da)** COMPLETED. It printed `R1_QUANTIZER_TEST_OK` for both wide/range variants and `R1_CPU_PREFLIGHT_OK`: 9 states, max output amplitude, STE master-weight/scale gradients on unclipped inputs, Q3 switch behavior, script AST. The R1 three-arm GPU job has not been completed or interpreted yet. Historical G1/S1 findings remain valid and archived; **do not launch new seeds, repeat R1, or launch other jobs without user approval**.

---


## G1/S1 COMPLETE — controls on scale gradients and sham resets (2026-10-09 UTC)

G1 HF [6ac96a7efee2c9007017ea64](https://huggingface.co/jobs/codeflash85/6ac96a7efee2c9007017ea64) COMPLETED, 6/6 checks, pinned source SHA `059a0bb91aa55fea99ed56b1d0d980eb21fd31ae`. Final WikiText heldout NLL: d0.03 learned **5.494818**, d0.50 learned **4.884970**, shallow-minus-deep gain **+0.609848**; d0.03 frozen **5.516548**, d0.50 frozen **4.896080**, gain **+0.620468**; frozen-minus-learned depth interaction **+0.010620**. Identical initial Q3 codes/scores; raw scales frozen exactly and excluded from optimizer, all 900 opportunities (898 effective/2 AMP skips). **Depth effect persists without row-scale learning**, supporting interior assignment placement and survival rather than requiring scale-gradient learning. Does not isolate Q9's wider range or fully prove mediation.

S1 HF [6ac96a82fee2c9007017ea6b](https://huggingface.co/jobs/codeflash85/6ac96a82fee2c9007017ea6b) COMPLETED, 9/9 checks, pinned SHA `4cdab8ee73ddd804e96fac43992f91ab9a54c1b9`. On matched seed1729 FineWeb, D Q3 continuous loss **5.766660**, D Q3 with original-scale+Adam/GradScaler sham reset@300 **5.789199** (sham worse **0.022539**), historical same-seed Q9→Q3 **4.996255** (**0.792944** better than sham). WikiText validation likewise sham 6.560840, continuous6.558239, staged5.695225. Anchor D reproduces original F1 exactly. **Reset alone is insufficient** to explain staged FineWeb benefit. One seed, same 21-doc heldout sample.

**Raw evidence:** [G1 JSON](results/run_g1_depth_by_scale_freeze_seed1729_2026-10-09.json) · [G1 summary](results/run_g1_depth_by_scale_freeze_seed1729_summary.md) · [S1 JSON](results/run_s1_fineweb_q3_sham_reset_seed1729_2026-10-09.json) · [S1 summary](results/run_s1_fineweb_q3_sham_reset_seed1729_summary.md). Original frozen protocol/job handoff [G1_S1_ACTIVE_JOB_PLAN.md](G1_S1_ACTIVE_JOB_PLAN.md).

**Next:** strongest unrun mechanistic confound is Q9's broader output range (range-matched nine levels versus original Q9, correct STE/clipping); boundary-margin matched controls and truly new evaluation docs follow. No R1 or other paid follow-up job automatically launched. [CURRENT_STATE.md](CURRENT_STATE.md) is authoritative; old “active/submitted” headings elsewhere are historical.

---


## NOW ACTIVE — G1 and S1 mechanism-control jobs (2026-10-09 EDT)

**Two bounded GPU jobs submitted**, with preregistrations committed before launch; **results not available at submission**:

- **G1 depth × scale trainability**: [HF job 6ac96a7efee2c9007017ea64](https://huggingface.co/jobs/codeflash85/6ac96a7efee2c9007017ea64), one A10G-small max **2h**, pinned Git SHA `059a0bb91aa55fea99ed56b1d0d980eb21fd31ae`, [G1 source](g1_depth_by_scale_freeze_seed1729.py). Seed1729 WikiText familiar v13A preparation, four Q3 continuation arms d003/d050 × learned/frozen Q3 row scales, common global 900-step continuation. [G1 frozen prereg](research_log/g1_frozen_vs_learned_q3_scale_depth_prereg_2026-10-09.md).
- **S1 FineWeb Q3 sham-switch**: [HF job 6ac96a82fee2c9007017ea6b](https://huggingface.co/jobs/codeflash85/6ac96a82fee2c9007017ea6b), one A10G-small max **90m**, pinned Git SHA `4cdab8ee73ddd804e96fac43992f91ab9a54c1b9`, [S1 source](s1_fineweb_q3_sham_reset_seed1729.py). Seed1729 one original direct-Q3 continuous arm and one Q3 step300 sham Adam/GradScaler/original-scale reset, with unchanged FineWeb QAT document partition/1200-step LR and historical Q9 comparison. [S1 frozen prereg](research_log/s1_fineweb_direct_q3_sham_reset_seed1729_prereg_2026-10-09.md).
- **Static preflight:** HF CPU job `6ac96a6f095c57808930b267` emitted `G1_S1_STATIC_PREFLIGHT_OK`; script Python AST and invariants checked.

**Full permanent handoff:** [G1_S1_ACTIVE_JOB_PLAN.md](G1_S1_ACTIVE_JOB_PLAN.md). **Do not relaunch these jobs, change scripts, launch additional seeds, or start R1 range-matched Q9 until new user authorization.** Inspect job statuses and exact `FINAL_JSON`, archive raw findings even nulls/technical failures, then update research reports. F1/F2/F3 and L1 historical findings stay completed and preserved.

---


## Latest completed studies and external mechanism-review follow-up (2026-10-09 UTC)

**All F1/F2/F3 FineWeb seeds and L1 WikiText6000 job have COMPLETED**, all final scientific validity flags passed. No research jobs currently active in the last verified HF listing. **FineWeb aggregate:** Q9→Q3 beats direct Q3 in **3/3 orders**, mean held-out advantage **+0.763898 nats/token** (range +0.742353..+0.778936), WikiText validation mean +0.801917. Identical 21 FineWeb evaluation documents across orders, *not independent validation samples*. [3-seed aggregate](results/run_f1_f3_fineweb_edu_three_seed_aggregate_summary.md) · [F2 raw](results/run_f2_fineweb_edu_direct_vs_staged_seed271828_2026-10-09.json) · [F3 raw](results/run_f3_fineweb_edu_direct_vs_staged_seed424242_2026-10-09.json).

**L1 six-thousand-step:** [result summary](results/run_l1_6000step_wikitext_durability_seed1729_summary.md) · [raw](results/run_l1_6000step_wikitext_durability_seed1729_2026-10-09.json). On fixed WikiText validation, staged advantage declines **+0.716087 @1200 → +0.616040 nats/token @6000** (nonmonotonic), so direct Q3 hasn't caught up by6000. Reproduced both step1200 historical tests exactly. This is one long training order, not a convergence proof.

**Separate independent review of unresolved mechanism controls:** [frozen-scale depth / Q9 range match / FineWeb sham switch / margin-matched positions / common-scale code comparisons / document-linked evidence](research_log/2026-10-09_external_technical_review_scale_range_sham_controls.md). Reviewing `smollm2_v12_code_identity_position.py` confirms scale-gradient confounding is real; v13's nearly-equal fixed/learned alpha *measurement* of code survival does not eliminate a scale-*training-gradient* mechanism. Next recommended controls **G1 frozen-vs-learned-scale depth** and **S1 FineWeb direct-Q3 sham step300 reset**, followed by range-matched Q9. No control GPU jobs launched as part of this review. [Authoritative current state](CURRENT_STATE.md) takes precedence over historical “running” sections.

---


## ACTIVE: FineWeb-Edu F2/F3 seed replications (2026-10-09 UTC)

**User-authorized two new-seed GPU jobs, now RUNNING:**

- **F2, seed 271828:** [F2 seed271828](https://huggingface.co/jobs/codeflash85/6ac87415fee2c90070173baf); A10G-small, 90-minute maximum; immutable source SHA `64d171a225ba2781c83148a4235272c91939f9c9`, [script](f2_fineweb_edu_qat_seed271828.py).
- **F3, seed 424242:** [F3 seed424242](https://huggingface.co/jobs/codeflash85/6ac87417095c5780893020c2); A10G-small, 90-minute maximum; immutable source SHA `a20c4cc66ae16026bb90969d1160294b3f5826e9`, [script](f3_fineweb_edu_qat_seed424242.py).

[F2/F3 frozen preregistration](research_log/f2_f3_fineweb_edu_seed271828_424242_replication_prereg_2026-10-09.md) was committed **before compute**. HF CPU static smoke `6ac87407fee2c90070173b9d` emitted `F2_F3_PREFLIGHT_OK`; normalized source-code comparison passed: only seed, seed-expected permutation assertion, and provenance identifiers changed from the successful F1 seed1729 script. Exact same source-model/dataset SHAs, same FineWeb QAT document-based train/dev/eval partition and same 1200-step matched arms. **No F2/F3 scientific outcomes available yet.**

**Original F1 seed1729 result (already completed):** [summary](results/run_f1_fineweb_edu_direct_vs_staged_seed1729_summary.md) and [raw](results/run_f1_fineweb_edu_direct_vs_staged_seed1729_2026-10-09.json): staged improves held-out FineWeb NLL by **0.770405 nats** and WikiText validation NLL by **0.863013 nats**. Three-seed aggregate must wait for F2 and F3 `FINAL_JSON` and scientific checks. The heldout 21 FineWeb documents are **shared across seeds**, not independent datasets.

**Also still RUNNING at last check:** [L1 6000-step WikiText durability](https://huggingface.co/jobs/codeflash85/6ac86e1f095c578089301e53). **Do not cancel or resubmit L1.** There are exactly two *new* scientific GPU jobs in this replication request.

**Next AI:** inspect these three HF job IDs, never duplicate them. For F2/F3 parse `FINAL_JSON_BEGIN`/`FINAL_JSON_END`; verify all five checks and source-doc partition hashes match F1. Archive per-seed raw JSON, summary and three-seed aggregate (mean/median/min-max/positive count). Clearly label whether any failure was technical versus a negative staged effect; update this page, `AI_HANDOFF.md`, `EXPERIMENT.md`, README and both research reports. No additional GPU or data-search sweep authorized.

---


## NEW STATUS: F1 FineWeb-Edu COMPLETED; L1 WikiText 6000 RUNNING (2026-10-09 UTC)

**F1 complete** — Hugging Face [6ac86eb0fee2c900701738dd](https://huggingface.co/jobs/codeflash85/6ac86eb0fee2c900701738dd), 2026-10-09 04:48:30 UTC; script SHA `64dcd7f20240b4c62e6ecac8df70a1336a57bb14`. **5/5 checks PASSED**, `valid_for_science=true`. [F1 full raw JSON](results/run_f1_fineweb_edu_direct_vs_staged_seed1729_2026-10-09.json) · [Detailed summary](results/run_f1_fineweb_edu_direct_vs_staged_seed1729_summary.md) · [pre-job protocol](research_log/f1_fineweb_edu_qat_seed1729_prereg_2026-10-09.md). FineWeb-Edu public sample seed1729, fixed QAT 1200 scheduled opportunities each. **FineWeb doc-disjoint heldout NLL:** D direct Q3 **5.766660**, S Q9(300)→Q3(900) **4.996255** (D-S **+0.770405**). **Secondary WikiText2 validation:** D **6.558239**, S **5.695225** (D-S **+0.863013**). Both positive. FineWeb doc sets disjoint: train180 documents, dev3, eval21; only seed1729 and small sample, pretrained-model document overlap unknown. Do NOT generalize to full SmolLM2 pretraining mixture or unseen-by-source corpus. Equal scheduled but AMP actual optimizer updates D1194 vs S1192. Preserve all caveats, no automatic repetition.

**L1 still RUNNING at last check** — [6ac86e1f095c578089301e53](https://huggingface.co/jobs/codeflash85/6ac86e1f095c578089301e53), pinned script SHA `841e0ab948991a24b7397b34d90d28c38f749879`, A10G-small 2h timeout. This **separate** paired direct/staged 6000-step WikiText2 durability test preserves the first1200 historical steps and uses fresh QAT chunks after. Evaluate new WikiText validation curve at1200/2400/3600/4800/6000. **Do not interrupt, duplicate, or infer its scientific outcome from F1**; inspect its own status/logs and archive `FINAL_JSON` when completed.

**Working handoff plan:** [NEXT_JOBS_PLAN.md](NEXT_JOBS_PLAN.md). [Authoritative current state](CURRENT_STATE.md). F1 finished; only L1 is pending. Older T1, C1, M1 studies are completed. Do not launch more GPU jobs automatically.

---

## M1 COMPLETE — controlled Q3 continuation and late-code adoption (2026-10-09 UTC)

**HF job:** [`6ac86712fee2c900701734bb`](https://huggingface.co/jobs/codeflash85/6ac86712fee2c900701734bb) — completed **2026-10-09 04:15:27 UTC**, A10G-small, seven of seven checks passed, `valid_for_science=true`. **Pinned script SHA:** `cad64c009209a924be89b523e6a1e184a4f4137b`. [M1 preregistration](research_log/m1_equal_q3_continuation_seed1729_prereg_2026-10-08.md). [M1 full raw JSON](results/run_m1_equal_q3_continuation_seed1729_2026-10-09.json). [Detailed summary](results/run_m1_equal_q3_continuation_seed1729_summary.md).

One continuous Q9 trajectory saved FP32-master states at steps250 and300. Both states were projected with original source Q3 row scales and given **exact same 900 Q3 batches and LR values** (and new AdamW/GradScaler). P250 had **1150 total opportunities**, P300 **1200**, so this is **not equal-total compute** (C1 was equal-total).

**Heldout NLL:** P250+900 **4.9504320212** (PPL141.236); P300+900 **4.9009853676** (PPL134.422), **+0.0494466536 nats** early-prep penalty. Both Q3 arms had **897 actual Q3 optimizer steps** / **3 skips**; late P300 endpoint and native Q9 preparation matched history exactly.

**Position-level code result:** Among **6,245,131 / 314,572,800** weights where step250 and step300 projected Q3 codes differ (**1.9853%**), early P250 continuation ends at step300-prep code at **44.16%**. Final P250/P300 agree at **60.00%** of these prep-disagreement positions, and **93.858%** overall. P300 itself retains the step300-prep code in these positions only **59.30%** of the time, so the step300-prep mask is **not a proven final target**. Code convergence is incomplete, and code matches are not causal attribution or evidence of universal optimal codes.

**Scientific boundary:** M1 shows last50 Q9-preparation steps help when Q3 continuation is fixed in this repeatedly seen seed/test set; C1 showed early switching with *extra* Q3 can retain most finite-budget gain. The between-experiment difference is confounded by different Q3 batch/LR starting indices. These data do NOT resolve asymptotics or test generality. See [CURRENT_STATE.md](CURRENT_STATE.md) (authoritative current status), [NEXT_EXPERIMENT_PLAN.md](NEXT_EXPERIMENT_PLAN.md) (copyable history and future gates). The earlier submitted status is historical; **M1 is DONE. No follow-up job automatically launched**.

---


**Authoritative live state:** [CURRENT_STATE.md](CURRENT_STATE.md) · **Copyable next-experiment/handoff plan:** [NEXT_EXPERIMENT_PLAN.md](NEXT_EXPERIMENT_PLAN.md). The long AI handoff and experiment ledger preserve historical statuses, including superseded “current” headings.

## C1 COMPLETED — 250-step Q9 switch retains 95.75% of staged benefit (2026-10-08 EDT)

**COMPLETED:** 2026-10-09 03:24:11 UTC. HF job [`6ac8597afee2c90070172c79`](https://huggingface.co/jobs/codeflash85/6ac8597afee2c90070172c79), A10G-small. **Pinned code:** `5577a771ed57409283690097a58bae0c8966fdb1`, `ternary_pet/c1_smol360m_q9_switch250_vs300_seed1729.py`.  
**Prereg:** [C1 frozen plan](research_log/phase_c_q9_switch250_vs300_seed1729_prereg_2026-10-08.md). **Raw parsed FINAL_JSON:** [C1 JSON](results/run_c1_smol360m_q9_switch250_vs300_seed1729_2026-10-08.json). **Detailed results:** [C1 summary](results/run_c1_smol360m_q9_switch250_vs300_seed1729_summary.md).

**SEED1729 HELD-OUT TEST LOSS:** D direct Q3(1200) **5.5957221873** (PPL269.272), S250 Q9(250)+Q3(950) **4.9305152558** (PPL138.451), S300 Q9(300)+Q3(900) **4.9009853676** (PPL134.422). Both D and S300 reproduced historical results **exactly**, all scheduled steps matched (1200 each); `valid_for_science=true`. AMP skipped opportunities D **6** vs S250/S300 **10** (1194 vs1190 actual optimizer steps). All transitions restored initial original Q3 scales and reset optimizer, with the full global LR schedule continuous.

**Prespecified exploratory tests:** S250 - S300 = **+0.0295298882 nats** (S250 slightly worse; <0.07 threshold) and S250 retained **95.7495%** of S300's absolute loss gain over D (>90% threshold). **Both PASS.** Relative to direct Q3, S250 still improves held-out loss by **0.6652069315 nats**. Transition fixed-Q3 losses are nearly equal at switch: S250 step250 **6.5421072443**, S300 step300 **6.5422919790**, though native Q9 differs. Train-split final dev D5.255937, S2504.575021, S3004.521716.

**Interpretation:** 250 Q9 steps plus 950 Q3 steps preserve most of the established staged benefit **under this one order and previously viewed WikiText heldout**. It does **not** show full mask formation by 250: T1 saw only 69.4% M300 recall. Extra 50 Q3 steps compensate for some effect but mechanism undetermined. Step250 was **chosen after T1**; positive criteria are pre-job but scientific conclusion remains **exploratory, not independent confirmation**. S250 is modestly worse than S300. Do not call 250 universally optimal or claim healthy generation.

**NEXT AI:** C1 job is done; do NOT launch duplicates. Preserve result archives, S1-1 negative, old v10 baseline. For a new study either preregister **independent seeds/orders for fixed 250 vs 300** (confirmatory) or test the **Q9-only vs direct-only causal mask split** (new mechanism). Neither is authorized automatically; user approval needed for additional paid GPU. Earlier submission-status blocks below are historical.

---


## C1 Phase C Q9→Q3 early switch — HF GPU job submitted (2026-10-08 EDT)

**Status at launch:** SCHEDULING; results **not yet known**.  
**Job:** [codeflash85/6ac8597afee2c90070172c79](https://huggingface.co/jobs/codeflash85/6ac8597afee2c90070172c79)  
**Pinned script commit:** `5577a771ed57409283690097a58bae0c8966fdb1`  
**Script:** `ternary_pet/c1_smol360m_q9_switch250_vs300_seed1729.py`  
**Preregistration:** [Phase C 250/300 switch test](research_log/phase_c_q9_switch250_vs300_seed1729_prereg_2026-10-08.md).  
**Prior diagnostic:** [T1 Phase A timing finding](results/run_t1_q9_discovery_timing_seed1729_summary.md).

User explicitly authorized the next experimental test ("sure do it"). ONE bounded GPU job: `a10g-small`, 90-minute max, seed1729, 1200 globally matched scheduled opportunities per arm. Arms D: 1200 direct-Q3, continuous Adam/scales; S250: Q9(250)+Q3(950), original Q3 scales and fresh Adam at step250; S300: Q9(300)+Q3(900), same reset at step300. Global warmup-to-1e-3 cosine-to-1e-4 schedule never restarts. Same teacher, data order, FP32 master/FP16 compute, objective, optimizer, train-split diagnostics and 8192-token held-out evaluation. Code static smoke HF job `6ac85970095c578089301448` reported `C1_STATIC_CHECK_OK` before GPU submission. Do not submit duplicate.

**Primary exploratory outcome:** test loss `L250-L300` and preservation of relative-to-D improvement. Prespecified "preserve benefit" thresholds: `L250-L300 <= +0.07` nats AND `(LD-L250)/(LD-L300) >= 0.90`, only when S300 improves over D. Check historical D ~5.595722 and S300 ~4.900985 (within prereg 0.06 nats), all arms 1200 scheduled opportunities, skips explicitly measured. This is **exploratory**, since t250 was selected after observing T1 step300 mask and historical v10 heldout.

**After job completion:** inspect HF status and full `FINAL_JSON_BEGIN...FINAL_JSON_END`, verify validity before scientific interpretation, archive original raw JSON including negatives, update `EXPERIMENT.md`, `RESEARCH_REPORT.md`, README and AI handoff. Technical failure vs scientific null must be distinguished. No other seed/sweep authorized automatically, no additional jobs launched.

---


## LATEST COMPLETED HF JOB: T1 Q9 assignment timing (2026-10-08 evening EDT)

**Status:** COMPLETED; completed **2026-10-09 02:51:49 UTC**.  
**HF job:** [codeflash85/6ac85534fee2c90070172a41](https://huggingface.co/jobs/codeflash85/6ac85534fee2c90070172a41)  
**Pinned code:** `2a982be6910d99eaf53d74f6e3ec3dae5807de17`  
**Script:** `ternary_pet/t1_q9_discovery_timing_seed1729.py`  
**Preregistration:** [Phase A diagnostic](research_log/phase_a_q9_timing_seed1729_prereg_2026-10-08.md).  
**Full original parsed FINAL_JSON archive:** [T1 raw result](results/run_t1_q9_discovery_timing_seed1729_2026-10-08.json).  
**Full interpretation:** [T1 summary](results/run_t1_q9_discovery_timing_seed1729_summary.md).

Seed1729, two paired 300-step-only preparation arms D(Q3) and S(Q9), shared v10 global LR and original Q3 scales for code snapshots at 0,50,100,150,200,250,300. One A10G-small job, no held-out model test, Q3 continuation, Gaussian, or gridward. Two-bit packed snapshots analyzed in job memory and not uploaded. Successful `valid_for_science=true` and **exact** reproduction of D native loss **5.9880944689**, S native loss **5.1827188333**, S fixed-Q3 loss **6.5422919790**, and step300 D/S projected mask **6.458076477%**. The D fixed-original-scale validation was **6.0043244561** versus historical native D **5.9880944689** (small +0.01623 nats, within preregistered 0.035 tolerance; *do not claim bitwise equality of these two distinct diagnostics*).

**Timing: step100 M300 recall 18.61% (precision 53.88%), step150 recall 35.42% (precision 60.71%), step200 recall 51.35% (precision 68.02%), step250 recall 69.40% (precision 77.77%); step300 by definition 100%.** M300 is 6.458% of all targeted weights: Q9-only 3.3849%, D-only 3.0701%, tiny both-changed/different 0.0031%. At step250 eventual Q9-only recall 67.60%, D-only recall 68.97%. No sharp early plateau: ~30.60% of M300 is still missing at step250. All overlaps are retrospective oracle comparisons with t=300, NOT an online feature or causal demonstration of when final downstream Q3 quality becomes available.

**Decision:** Phase A one-seed pilot positive as a *descriptive timing measurement*, not a demonstrated shorter training recipe. Phase B causal split of Q9-only vs D-only groups and Phase C 200/250/300 Q9→Q3 switch-time test are unrun. Next AI should start from archived raw and summary, freeze controls before any further GPU, obtain user authorization for compute, and log negative results. **No other GPU jobs launched with this analysis.**

---


## Latest research-history note — Q9 assignment discovery and timing (2026-10-08 EDT)

**Read:** [Q9 assignment-discovery findings and gated test plan](research_log/2026-10-08_q9_assignment_discovery_timing.md). This is an evidence-labeled **discussion record / draft plan**, not a completed experiment or frozen preregistration.

The user's guiding intuition is **"a rough 9, but a better 3 afterwards"**: Q9 as an assignment-selection/search process, not the final objective. v11 raw overlap statistics, decomposed retrospectively at step 300, show approximately **3.385–3.398% Q9-only Q3-code changes** versus source and **2.785–3.070% direct-only changes that Q9 avoids** over the three established Smol seeds; together they explain nearly all of the ~6.18–6.46% disagreement mask. The **combined mask** is causally useful (v11/v12), but the two categories' **individual causal contributions remain untested**.

The existing Q9 code-movement traces at steps 50–300 are **native Q9-state** changes, not projected Q3 assignment trajectories. Therefore **the timing when useful ternary selections emerge remains unknown**. Draft plan: matched Q9/direct Q3 checkpoint snapshots projected through fixed initial Q3 scales at 0/50/100/150/200/250/300; retrospective stability analysis; then selectively test Q9-only versus direct-only masks with paired Q3 continuation, before proposing shorter Q9 phases. Avoid future-information leakage for online predictors. The prior v12 three-order and v13 one-order depth findings justify **moderate interior placement** of selected assignments, not indiscriminate gridward commitment.

**State:** Documentation only; no new GPU job or experimental results from this discussion. Any launch needs a separately frozen protocol and explicit compute authorization. Preserve S1-1 negative unchanged. Earlier "ACTIVE" status blocks below are historical; the S1-1 job completed.

---


> **START HERE if you are a new AI taking over this project.**
>
> Repository: `steveonw/hackathon`  
> Project directory: `ternary_pet/`  
> Current state: **SmolLM2-360M mechanism complete; Granite-350M 3-order G1 mixed (2/3 positive), G1-6/G1-7b development negatives**  
> Canonical Smol mechanism: `replications/v12_code_identity_position_aggregate_summary.md`  
> Latest completed HF job before G1-9: **G1-8 `6ac79551e7a0dae8a2788f0b` — negative Smol LR schedule transfer on Granite order 424242, with surviving positive d50 intervention**  
> Latest completed HF jobs: G1-4/G1-5 Granite confirmations `6ac70af0df2184ac91ac6ffc` / `6ac70af2df2184ac91ac7000`  
> Granite compute rule: FP32 masters + BF16-rounded source + BF16 autocast/BF16 teacher; never FP16  
> Canonical current docs: `RESEARCH_REPORT.md`, `G1_GENERALIZATION_PLAN.md`, `results/run_g1_2_granite350m_staging_summary.md`

## 1. What the user is trying to discover

The research question is whether a pretrained language model can enter a final
ternary weight space more effectively if it first adapts in an intermediate,
less restrictive representation.

The original intuition was "lose precision gradually instead of jumping
straight to ternary." Experiments have narrowed that into a more precise
optimization hypothesis:

> Intermediate-state training may let continuous FP32 master/shadow weights
> reposition themselves before the harsh ternary partition is imposed.

This is **not** currently a claim that staged rounding mechanically preserves
bits, nor a claim of a production-ready ternary model.

## 2. Model and standard setup

Model:

- `HuggingFaceTB/SmolLM2-360M-Instruct`
- ~362M parameters
- Llama-family architecture

Core training setup in the successful experiments:

- source checkpoint BF16-rounded before treatment;
- target linears quantized except `lm_head`;
- non-quantized parameters frozen;
- persistent FP32 master/shadow weights;
- learnable rowwise quantizer scales;
- AdamW, usually LR `1e-4`;
- loss = 35% CE + 65% teacher KL;
- WikiText-2 training/evaluation;
- fixed 8192-token held-out evaluator;
- seed/order 1729 is a key replicated ordering;
- generated text remains poor/degenerate despite metric improvements.

## 3. What has already been learned

### v1 — naive 27 -> 9 -> 3

Failed. Direct and staged models were badly damaged. This did not support the
hypothesis.

Job: `6ac52475404719ba37661c8b`

### v2 — strict nested hierarchy

Important negative result:

- deterministic direct ternary projection and deterministic nested 27 -> 9 -> 3
  projection ended identically;
- therefore staging without learning does **not** preserve extra information.

Hard commits and optimizer resets made this a weak adaptive test.

Job: `6ac527e5404719ba37661dc9`

### v3 — persistent FP32 masters / transition schedules

Persistent continuous masters helped make the experiment meaningful, but
full-precision warmup, soft ramp, and direct ternary were roughly tied once
final ternary training budget was matched.

Confounds later identified:

- ~79% zeros in the ternary representation;
- non-quantized components could train;
- no direct code-movement metric.

Job: `6ac52df9404719ba37661fa1`

### v4 / v4b — first reproducible positive 9 -> 3 result

Protocol:

- direct: 1200 @ Q3
- staged: 300 @ Q9 -> 900 @ Q3
- same total updates
- persistent FP32 masters
- no hard commit
- frozen non-quantized parameters
- code movement logged
- LR `1e-4`

Across three paired training orders, staged Q9 -> Q3 beat direct Q3 on all main
held-out metrics.

Aggregate v4/v4b:

- mean held-out loss improvement: **0.765 nats/token**
- mean PPL reduction: **53.35%**
- mean teacher top-1 gain: **+7.54 percentage points**
- mean KL reduction: **24.47%**

Important replication correction:

Changing only the random seed originally produced identical traces because the
data order was fixed. Those false starts were cancelled and **do not count**.
The valid v4b replications explicitly shuffle the same training chunks.

Key jobs:

- v4: `6ac539b5404719ba376621a2`
- v4b seed/order 1729: `6ac546b5fbc85ba6823b8941`
- v4b seed/order 271828: `6ac546bafbc85ba6823b8946`

See:

- `replications/v4b_aggregate_summary.md`
- `results/run_v4_summary.md`

### v5 — 5x training-volume test

Protocol:

- direct: 6000 @ Q3
- staged: 1500 @ Q9 -> 4500 @ Q3
- ~768k training-token presentations per treatment

Results:

| Variant | Loss | PPL | Teacher top-1 | KL |
|---|---:|---:|---:|---:|
| BF16 source | 3.6854 | 39.86 | 99.68% | ~0 |
| Direct 6000@3 | 4.4463 | 85.31 | 41.44% | 1.522 |
| Staged 1500@9 -> 4500@3 | **4.2493** | **70.05** | **43.64%** | **1.330** |

The staged advantage survived but shrank:

- loss advantage: **0.197 nats/token**
- PPL reduction: **17.88%**

This is evidence that a substantial part of the v4/v4b effect is an
**early-optimization/head-start effect**. It does **not** prove a permanently
better asymptotic ternary basin.

Job: `6ac56d99fbc85ba6823ba03f`

See:

- `results/run_v5_summary.md`
- `results/run_v5_2026-10-06.json`

### v6 — causal transition ablation

Purpose: determine whether the v4-scale staged advantage is stored in:

- prepared FP32 master weights;
- learned quantizer scales;
- Adam optimizer state.

One shared 300-step Q9 checkpoint was branched into 900-step Q3 continuations.

Results:

| Condition | Loss | PPL |
|---|---:|---:|
| Direct 1200@3 | 5.8747 | 355.90 |
| Carry masters + scales + full Adam | 5.2059 | 182.34 |
| Reset Adam | 5.2031 | 181.83 |
| Prepared masters only; original scales + fresh Adam | 5.1990 | 181.09 |
| Prepared masters + weight Adam; original scales | 5.1799 | 177.67 |

Critical result:

> Resetting Adam and restoring the original scales does **not** remove the
> staged advantage.

The four prepared-master branches span only **0.026 nats/token**, versus roughly
0.67-0.69 nats/token over direct.

Therefore, under this setup, the early staged advantage is carried primarily by
the **adapted FP32 master weights**.

#### v6 clean transition diagnostic

On the same fixed diagnostic data, without an optimizer update between
evaluations:

- initial Q3 loss: **15.4784**
- after 300 Q9 updates, evaluated as Q9: **4.9320**
- same prepared master weights immediately projected to Q3: **9.1771**

So the clean Q9 -> Q3 switch costs:

- **+4.2451 nats/token**

but the Q9-prepared masters projected to Q3 are 6.3013 nats/token better than
the **untrained** initial Q3 projection.

#### v6 future Q3 boundary crossings

After Q9 preparation:

- future Q3 assignments changed with learned scales: **0.6769%**
- future Q3 assignments changed with original scales fixed: **0.6755%**

This is strong evidence that the FP32 master weights themselves move across
future ternary decision boundaries.

Important refinement after external review:

> v6 did **not** compare Q9-prepared Q3@300 against direct Q3@300 on the same
> diagnostic data.

Therefore v6 does **not yet distinguish**:

1. **better entry state** — Q9@300 is already a better ternary checkpoint than
   direct Q3@300; from
2. **better trainability** — direct Q3@300 may be as good or better, while Q9
   leaves continuous masters in a geometry that trains better over the next 900
   ternary updates.

Also: direct Q3 had moved about the same aggregate fraction of codes by step
300 (~0.69%), often in similar broad layers. Do not claim that raw code count or
broad layer identity explains the effect.

Job: `6ac5816ffbc85ba6823ba8ec`

See:

- `results/run_v6_summary.md`
- `results/run_v6_2026-10-06.json`

## 4. CURRENT EXPERIMENT — v7

### Purpose

v7 is specifically designed to separate:

- **better ternary entry state**;
- **better subsequent trainability / within-bin geometry**;
- a generic benefit from any less-constrained warmup.

### Pinned code

`ternary_pet/smollm2_v7_equal_compute_geometry.py`

Pinned commit:

`3aa494a4c8418052ed9e13e0de4f97c692a29fc7`

### Hugging Face job

`6ac58e76fbc85ba6823bad78`

The job completed successfully on **A10G-small**.

### v7 arms

All arms consume the same first 300 shuffled chunks:

1. `q3_300`: direct Q3 for 300 updates;
2. `q9_300`: Q9 for 300 updates;
3. `fp32_300`: unquantized-weight warmup for 300 updates.

At step 300, all three master states are projected through the **same original
Q3 scales** and scored on the **same fixed 24-chunk diagnostic set**.

Then all three get the same 900-step ternary continuation:

- same original Q3 scales;
- fresh Adam;
- same LR `1e-4`;
- same remaining 900 chunks.

Therefore the only state carried from preparation into continuation is the
prepared FP32 master weights.

### v7 diagnostics

v7 records:

- native step-300 quality in each preparation space;
- fixed-scale Q3 quality for all three states at equal compute;
- Q3 code changes versus initial;
- exact pairwise overlap/Jaccard of changed Q3 codes;
- pairwise Q3 Hamming distance;
- distance of every master to nearest future Q3 threshold
  `w/alpha = +/-1/3`;
- final held-out metrics;
- fixed free-running generation prompts.

### How to interpret v7

If:

`Q9-prepared fixed-Q3@300 < direct-Q3@300 loss`

then Q9 creates a **better ternary entry state**.

If direct Q3@300 is equal/better, but Q9 finishes the shared 900-step
continuation better, then the likely mechanism is **better trainability or
within-bin master geometry**.

The FP32 arm answers whether Q9 itself is special:

- if FP32 warmup matches Q9, a less-constrained warmup may be sufficient;
- if Q9 beats FP32, the intermediate discrete grid may provide useful structure.

Threshold-margin diagnostics may reveal an effect that raw code Hamming misses:
two master states can have the same Q3 code but sit very differently relative to
their next decision boundary.

## 5. What the next AI should do when v7 finishes

1. Inspect HF job `6ac58e76fbc85ba6823bad78`.
2. Fetch logs and locate text between:
   - `FINAL_JSON_BEGIN`
   - `FINAL_JSON_END`
3. Extract at minimum:
   - `initial_q3_reference`
   - each arm's `prep.native_diag`
   - each arm's `fixed_q3_projection.diag`
   - each arm's `fixed_q3_projection.margin`
   - each arm's `fixed_q3_projection.vs_initial_codes`
   - `step300_code_comparisons`
   - each arm's final loss / PPL / teacher top-1 / KL
   - generations
4. Make the step-300 equal-compute comparison **before** looking at final
   1200-step outcomes.
5. Do not change or rerun the protocol merely because one arm loses.
6. Save:
   - raw JSON under `ternary_pet/results/`
   - a human-readable v7 summary
   - an update to `EXPERIMENT.md`
   - an update to this handoff sheet
   - a concise README headline if the result materially changes the mechanism.
7. If the job failed for an infrastructure/code reason, fix it, pin a new Git
   commit, rerun the **same scientific protocol**, and retain the failure for
   auditability.

## 6. Claims that are currently safe

Safe:

> Under this SmolLM2/WikiText QAT setup, a 9-state preparation phase produces a
> large finite-budget ternary recovery advantage at 1200 updates.

Safe:

> The early advantage is carried mainly by the adapted FP32 master weights,
> rather than by inherited Adam moments or learned-scale state.

Safe:

> Q9 preparation moves some continuous masters across future Q3 decision
> boundaries even when Q3 scales are held fixed.

Safe:

> At 5x training volume, the staged advantage remains but becomes much smaller,
> consistent with a large head-start component.

## 7. Claims that are NOT established

Do **not** claim:

- Q9 is globally optimal;
- 9 states are uniquely special;
- staged QAT reaches a permanently better asymptotic basin;
- gradual rounding mechanically preserves information;
- the effect generalizes beyond SmolLM2/WikiText;
- the current ternary model is usable;
- broad layer-level flip concentration proves causal importance;
- `flip_since_last` counts every discrete crossing.

Important measurement caveat:

`flip_since_last` is a snapshot Hamming difference. A code may flip and return
between snapshots without being counted.

## 8. Known technical limitations / reproducibility caveats

- Dependency declarations use minimum versions such as
  `torch>=2.4`, not exact package pins.
- Model and dataset revisions are not exact-pinned.
- Exact experiment code **is** pinned by Git commit.
- A10G and T4 results need not be bit-for-bit identical.
- WikiText teacher-forced metrics have improved much more than free-running
  generation quality.
- The direct ternary baseline has only a limited LR diagnostic sweep; it has not
  yet received a full optimizer/scheduler search.
- The Q9 grid itself may be suboptimally calibrated/clipped.
- Learned scales changed very little in v6, so "scales not necessary here"
  should not be generalized to "scales never matter."

## 9. Research trajectory after v7

Do not blindly launch another large run.

Depending on v7:

- if Q9 already wins at fixed-Q3@300, study **which exact ternary decisions**
  differ from direct and whether a better-calibrated intermediate grid improves
  the entry state;
- if Q9 only wins after continuation, focus on **within-bin geometry /
  threshold-margin / gradient conditioning**;
- if FP32 matches Q9, test whether the phenomenon is generic continuation/warmup
  rather than Q9-specific;
- if Q9 beats both Q3 and FP32, compare other intermediate codebooks
  (5/7/9/15 states) under equal compute;
- regardless of outcome, a stronger tuned direct-Q3 baseline is eventually
  required before a broad research claim;
- only scale to larger models after the mechanism survives at least one other
  architecture/data setting.

## 10. Repository map

Start with:

- `AI_HANDOFF.md` — this file
- `README.md` — current headline
- `EXPERIMENT.md` — chronological protocol and outcomes
- `SHAREABLE_RESEARCH_REPORT.md` — external-review narrative

Scripts:

- `smollm2_staircase.py` — v1
- `smollm2_nested_v2.py` — v2
- `smollm2_transition_v3.py` — v3
- `smollm2_v4_fliprate_9to3.py` — v4
- `smollm2_v5_5x.py` — v5
- `smollm2_v6_transition_ablation.py` — v6
- `smollm2_v7_equal_compute_geometry.py` — v7
- `replications/` — v4b replication scripts and aggregate

Results:

- `results/run_v4_summary.md`
- `replications/v4b_aggregate_summary.md`
- `results/run_v5_summary.md`
- `results/run_v6_summary.md`
- machine-readable JSON files in `results/`

## 11. Working style / user intent

The user wants real empirical work, not speculative storytelling.

Preferred workflow:

`idea -> GitHub pinned code -> HF GPU job -> raw metrics/logs -> analysis -> GitHub writeback`

Treat GitHub as the lab notebook/source of truth.

Negative results and failed jobs should remain visible and auditable.

The user has authorized bounded paid Hugging Face compute, but compute should be
used prudently. Do not spend on a bigger run when a cheaper causal ablation can
answer the question.

External AI reviews are advisory. Adopt a suggestion only when it improves the
scientific test; do not let outside reviewers silently redefine the experiment.

### Documentation and reporting protocol going forward

Keep the existing audit trail exactly: preregister each experiment in
`EXPERIMENT.md`, record job IDs, save raw JSON plus a human-readable run
summary, update the results registry, update `AI_HANDOFF.md`, and maintain
replication aggregates when a claim is tested across orders.

Treat `RESEARCH_REPORT.md` as the **living current-state document**. After each
experiment, edit it in place: refresh the Summary, the "Results at a glance"
table (claim, effect size, number of orders, run), and the relevant Findings
section. Move superseded interpretations or obsolete headline numbers to the
appendix rather than leaving conflicting current claims in the main text.

Keep `SHAREABLE_RESEARCH_REPORT.md` as the **append-only chronological log**.
Do not rewrite old entries there merely because the interpretation changed.

Keep the README headline current with a concise 3–4 line summary of the main
finding and a link to `RESEARCH_REPORT.md`.

When reporting results in chat:
1. start with one table of the key held-out numbers;
2. follow with 2–3 plain-language sentences explaining what they mean;
3. list caveats, including seed/order count and what remains untested;
4. give exactly one recommended next step;
5. avoid boxed equations unless a formula is genuinely necessary.

Before writing any replicated conclusion, inspect **every seed/order
individually**, not only the average. State plainly when a claim is mixed across
orders. Example: the FP32 warm-up control does not reproduce Q9 overall, but it
is not worse than direct in every order; seed 271828 gives FP32 a small
0.0292-nat improvement over direct.

## 12. One-sentence state of the project

> Across three training orders, Q9 preparation creates a worse immediate Q3
> checkpoint but a substantially more trainable FP32-master state for the same
> later Q3 continuation; v8 did not support the tested simple signed-threshold
> preload explanation, so the remaining mechanism is a deeper
> trainability/weight-selection geometry effect.


## v7 OUTCOME — read this before planning the next experiment

v7 resolves the main ambiguity left by v6.

Equal-compute fixed-Q3 diagnostic at step 300:

- direct Q3: **6.5589**
- Q9-prepared masters: **9.2264**
- FP32-prepared masters: **14.7346**

Final after identical fresh-Adam 900-step Q3 continuation:

- direct Q3 -> Q3: **5.8724 loss / 355.09 PPL**
- Q9 -> Q3: **5.1938 / 180.15**
- FP32 -> Q3: **6.0311 / 416.19**

Therefore:

1. Q9 does **not** win by producing a better immediate Q3 checkpoint.
2. Q9 does produce a master state that is dramatically more trainable during
   later Q3 optimization.
3. Generic FP32 warmup is not enough; the intermediate discrete constraint
   matters in this setup.
4. Direct and Q9 change similar counts of future ternary codes (~0.69% vs
   ~0.68%) but only **19.64% Jaccard** of changed positions overlap.
5. Simple global threshold-margin distributions are effectively identical.

The current mechanism hypothesis is now **Q9-specific gradient/weight-selection
geometry**, not entry quality, Adam carryover, scale carryover, or generic
high-precision warmup.

See `results/run_v7_summary.md` and `results/run_v7_2026-10-07.json`.

### Next decision

Do not automatically scale up. The highest-value next work is to determine why
the Q9-selected subset is more useful, or to compare intermediate state counts
(5/7/9/15) using the same equal-compute v7 framework. A stronger tuned direct-Q3
baseline remains necessary before broad claims.


## v7 REPLICATIONS — COMPLETED

The two confirmatory jobs completed successfully:

- seed 271828 — `6ac59807fbc85ba6823bb060`
- seed 424242 — `6ac59809fbc85ba6823bb063`

Together with seed 1729, the v7 trainability effect now holds in **3/3 training
orders**.

Aggregate:

- Q9 is a worse fixed-Q3 checkpoint at step 300 in all 3;
- Q9 finishes better after the identical 900-step continuation in all 3;
- mean final loss gain: **0.6877 nats/token**;
- mean PPL reduction: **49.69%**;
- mean top-1 gain: **+6.86 pp**;
- mean Q3-vs-Q9 changed-set Jaccard: **19.66%**.

FP32 warmup does not reproduce Q9. Do **not** say FP32 is always worse than
direct: seed 271828 gives FP32 a small 0.0292-nat improvement over direct.

Canonical aggregate:
`replications/v7_aggregate_summary.md`

Raw replications:

- `results/run_v7_seed271828_2026-10-07.json`
- `results/run_v7_seed424242_2026-10-07.json`

### Current next-question priority

The replication question is now substantially answered. The strongest next
mechanistic test is a signed **pre-loading diagnostic**: among weights that cross
a Q3 boundary during the 900-step continuation, measure whether Q9 preparation
had already moved their FP32 masters toward that eventual threshold more often
or more strongly than direct Q3 preparation.

A level-count sweep (5/7/9/15 -> 3) is also high value after or alongside that
diagnostic. A stronger tuned direct-Q3 optimizer/scheduler baseline remains
required before broad claims.

## v8 — COMPLETED signed pre-loading diagnostic

Job: `6ac5a0b1404719ba37664653`  
Pinned code: `356bff9769c96faa0ca139f9cba088fc1a52c2c8`

v8 tested whether Q9 uniquely moves later-flipping weights in their eventual
ternary-transition direction during preparation.

**Result: negative for that specific mechanism.**

Four-way matched comparison:

- on Q9's later-flip set: Q9 = **0.002328**, Q3 = **0.000873**,
  Q9-Q3 = **+0.001455**;
- on Q3's later-flip set: Q3 = **0.003130**, Q9 = **0.001439**,
  Q3-Q9 = **+0.001691**.

Each arm wins on its own future-flip set by a similar margin. That matches the
preregistered post-selection pattern rather than Q9-specific directional
pre-loading.

Do **not** claim that Q9's advantage is explained by pointing future-flipping
weights toward their next ternary threshold. v8 falsified that simple story.

The broader replicated result from v7 remains:

- Q9 is a worse immediate Q3 checkpoint after equal 300-step compute;
- Q9 becomes much better after the same 900-step Q3 continuation;
- the effect holds across 3/3 training orders;
- Q3 and Q9 move similar numbers of future ternary codes but mostly different
  specific weights.

Current best description:

> Q9 changes the optimization / weight-selection geometry in a reproducible
> way, but neither global threshold proximity nor simple signed future-threshold
> preloading explains why the selected Q9 master state is more trainable.

Canonical v8 files:

- `results/run_v8_summary.md`
- `results/run_v8_2026-10-07.json`

### Current next-question priority

A causal intervention should now focus on **which Q9-selected weights carry the
benefit**, without assuming signed-preload is the reason they matter. Another
high-value axis is the intermediate level-count sweep (5/7/9/15 -> 3) under the
same equal-compute v7 protocol. A stronger tuned direct-Q3 optimizer/scheduler
baseline remains necessary before broad claims.

## v9 — COMPLETED direct-Q3 tuning

Job: `6ac5ad77fbc85ba6823bb71c`  
Pinned code: `5c088e58539b2dede93df57ac3f72dbe0480a028`

v9 confirmed the direct baseline was undertuned.

Best full direct schedule on seed 1729:

- 100-step warmup;
- peak LR 1e-3;
- cosine decay to 1e-4 by step 1200;
- held-out loss **5.5957**;
- PPL **269.27**;
- top-1 **29.87%**;
- KL **2.6640**.

Historical constant-1e-4 direct was 5.8747 loss / 355.90 PPL.

Historical Q9 -> Q3 remains better at 5.1938 / 180.15.

Thus tuned direct closes about **41%** of the old loss gap but leaves a
**0.402-nat/token** Q9 advantage on seed 1729.

Do **not** keep quoting the old ~0.68-nat seed-1729 effect as if the direct
baseline were well tuned. The honest current number for this seed is ~0.40 nats
against the better direct schedule.

### Next decision

Per the preregistered rule, run the tuned direct schedule on training orders
271828 and 424242. Only after that should it become the canonical direct
baseline and should the residual Q9 effect be aggregated.

If the residual Q9 advantage survives both orders, proceed to the four-arm
Q3-vs-Q9 hybrid-master factorial intervention.

Canonical v9 files:

- `results/run_v9_summary.md`
- `results/run_v9_2026-10-07.json`

## v9 tuned-direct replication pair — COMPLETED

Valid confirmatory jobs:

- seed 271828: `6ac5b913404719ba37664b21`
- seed 424242: `6ac5b915fbc85ba6823bba5b`

Corrected scripts:
`3e74ccf5c448fc994d005a7baf94529b92e9996a`

Q9 remains better than the tuned direct-Q3 schedule in **3/3 orders**.

| Seed | Tuned direct loss | Q9 loss | Residual Q9 gain |
|---:|---:|---:|---:|
| 1729 | 5.5957 | 5.1938 | 0.4020 |
| 271828 | 5.6228 | 5.2417 | 0.3810 |
| 424242 | 5.6067 | 5.3008 | 0.3059 |

Canonical aggregate:

- mean residual Q9 loss advantage: **0.3630 nats/token**;
- mean PPL reduction: **30.38%**;
- mean top-1 gain: **+2.52 pp**;
- mean KL advantage: **0.3871**;
- tuned direct closes **47.27%** of the old gap on average.

Do not quote the older ~0.688-nat mean v7 gap as the current baseline
comparison. The canonical small-budget effect is now ~**0.363 nats/token**
against tuned direct.

The initial failed/cancelled replication jobs remain technical audit records and
do not count as experimental runs.

Canonical file:
`replications/v9_tuned_direct_aggregate_summary.md`

### Current next-question priority

Before the hybrid-master intervention, close the remaining schedule-fairness
loophole with a **schedule-matched Q9 control**.

The current 0.363-nat three-order comparison is asymmetric in tuning effort:
direct Q3 uses the v9-selected warmup+cosine schedule, while Q9 -> Q3 is still
the historical constant-1e-4 run. Therefore **0.363 nats is not a proven lower
bound or final intrinsic method advantage**.

First apply the exact direct-selected global LR curve to Q9 -> Q3:

- steps 1-100: linear warmup to 1e-3;
- steps 101-1200: cosine decay to 1e-4;
- Q9 for global steps 1-300;
- at the Q9 -> Q3 transition restore original Q3 scales and use fresh Adam as
  in v7, but continue the LR curve from global step 301;
- no additional schedule tuning and no test-set selection.

Run that schedule-matched Q9 protocol on the same three training orders and
compare it with tuned direct and historical Q9. This is an equal-schedule
control, **not** a full equal-search-budget Q9 optimization.

If Q9 still wins clearly, then proceed to the four-arm Q3-vs-Q9 hybrid-master
causal intervention using the schedule-matched preparation/continuation recipe.
Use the equal-compute step-300 direct-Q3 and Q9-prepared master states to
partition the actual prepared-state difference into:

- positions where their projected Q3 codes differ;
- positions where projected Q3 codes agree but continuous FP32 masters differ.

Construct the 2x2 hybrids so paired arms have exactly identical ternary forward
weights at continuation start, assert zero Hamming distance within those pairs,
then run the same Q3 continuation. This tests whether the residual Q9 advantage
is carried mainly by code-disagreement positions, hidden same-code master
geometry, or their interaction.

## v10 — COMPLETED schedule-matched Q9 control

Pinned code:
`4050f42cf226a300082178b2fda475ebcf31664e`

Jobs:

- 1729: `6ac5c252fbc85ba6823bbd6c`
- 271828: `6ac5c254fbc85ba6823bbd6e`
- 424242: `6ac5c256404719ba37664cf1`

The v9-selected global LR schedule was applied unchanged to Q9 -> Q3:

- 100-step warmup to 1e-3;
- cosine decay to 1e-4 through step 1200;
- Q9 steps 1-300;
- original Q3 scales + fresh Adam at transition;
- Q3 steps 301-1200;
- LR curve continues rather than restarting.

Result: matched-schedule Q9 beats tuned direct in **3/3 orders**.

| Seed | Tuned direct | Matched Q9 | Q9 gain |
|---:|---:|---:|---:|
| 1729 | 5.5957 | 4.9010 | 0.6947 |
| 271828 | 5.6228 | 4.9510 | 0.6718 |
| 424242 | 5.6067 | 4.9563 | 0.6505 |

Aggregate:

- mean loss advantage: **0.6723 nats/token**;
- mean PPL reduction: **48.94%**;
- mean top-1 gain: **+6.82 pp**;
- mean KL advantage: **0.6817**.

The step-300 diagnostic still points the opposite way: matched Q9 is **0.5120
nats/token worse** than tuned direct as an immediate fixed-Q3 checkpoint after
equal 300-step compute.

Therefore the trainability effect survives a much stronger and
schedule-symmetric regime.

Do not call 0.6723 a best-tuned-vs-best-tuned intrinsic advantage. Q9 received
the direct-selected schedule, not an independent equal-budget search.

Important geometry update: under this stronger schedule, step-300 projected-Q3
code displacement is ~4.41% for direct and ~4.88% for Q9, much larger than the
old ~0.68% v7 regime. Do not reuse the old v7 Jaccard/code-movement
interpretation as if it automatically applies here.

Canonical aggregate:
`replications/v10_schedule_matched_q9_aggregate_summary.md`

### Current next-question priority

Proceed to the four-arm hybrid-master factorial in the **matched-schedule
regime**.

Rebuild both step-300 master states inside the same run:

- D = direct-Q3 masters after 300 global-schedule updates;
- S = Q9 masters after 300 global-schedule updates.

Project both through the same original Q3 scales and define mask M where their
Q3 codes differ.

Construct:

- 00: D everywhere;
- 10: S on M, D elsewhere;
- 01: D on M, S elsewhere;
- 11: S everywhere.

Assert before continuation:

- Q3(00) == Q3(01) exactly;
- Q3(10) == Q3(11) exactly;
- paired diagnostic losses match to numerical tolerance.

Then give **all four arms** original Q3 scales, fresh Adam, and the same global
LR curve starting at step 301 for 900 Q3 updates.

First verify that endpoints 00 and 11 reproduce a material direct-vs-Q9
trainability gap under this common fresh-Adam continuation. Then interpret 10
and 01 to localize the causal contribution to code-disagreement positions,
same-code hidden master geometry, or their interaction.

## v11 — COMPLETED matched-schedule hybrid-master factorial

Job: `6ac5c7befbc85ba6823bbef0`  
Pinned code: `f342fd9c706f2fe21aa00adabe6611b7f835f570`  
Seed/order: 1729.

All pair assertions passed exactly before continuation:

- Q3(00) == Q3(01), zero Hamming;
- Q3(10) == Q3(11), zero Hamming;
- paired validation diagnostics exactly match.

D-vs-S step-300 projected-Q3 disagreement mask:

- **6.4581%** of quantized weights;
- 20,315,352 / 314,572,800 positions.

Final losses:

| Arm | Meaning | Loss |
|---|---|---:|
| 00 | D everywhere | 5.6136 |
| 10 | S on disagreement mask only | **4.9329** |
| 01 | S on same-code complement only | 5.5020 |
| 11 | S everywhere | **4.9010** |

Full endpoint gain: **0.7126 nats**.

The disagreement-mask transfer alone contributes **0.6807 nats**, recovering
**95.5%** of the full gain. Same-code hidden geometry alone contributes only
**0.1116 nats** on the direct background; once the disagreement mask is already
from Q9, the remaining same-code contribution is only **0.0319 nats**.

Current mechanism interpretation:

> On seed 1729, the Q9 trainability advantage is carried predominantly by the
> continuous Q9-prepared masters at the ~6.46% of positions where Q9 and direct
> produce different projected Q3 codes at step 300.

Do not claim the discrete code values alone are causal. Arm 10 transfers the
full continuous Q9 master values on those positions.

Canonical files:

- `results/run_v11_hybrid_factorial_summary.md`
- `results/run_v11_hybrid_factorial_seed1729_2026-10-07.json`

### Current next-question priority

Replicate the exact v11 factorial on orders 271828 and 424242 before making the
localization result canonical across orders.

If the same pattern replicates, the next causal refinement should separate
**code identity** from **continuous within-bin position on the disagreement
mask** itself, e.g. by constructing masters that preserve the Q9-selected Q3
code while recentering each selected master within its ternary bin.

## v11 hybrid-factorial localization — REPLICATED 3/3

Confirmatory jobs:

- seed 271828: `6ac638c0c656c912b4ffae8a`
- seed 424242: `6ac638c3f0d78b8017af0d5a`

All three orders pass the exact pair-equality assertions.

| Seed | Mask | Full 00->11 gain | M-only gain | Recovery |
|---:|---:|---:|---:|---:|
| 1729 | 6.458% | 0.7126 | 0.6807 | 95.5% |
| 271828 | 6.181% | 0.7014 | 0.6813 | 97.1% |
| 424242 | 6.301% | 0.6672 | 0.6443 | 96.6% |

Aggregate:

- mean mask size: **6.313%**;
- mean full gain: **0.6938 nats**;
- mean M-only gain: **0.6688 nats**;
- mean recovery: **96.4%**;
- mean same-code-only gain on D background: **0.1092 nats**;
- mean residual same-code gain after M is already from Q9: **0.0250 nats**.

The mechanism claim is now replicated:

> The dominant causal carrier is the continuous Q9-prepared master state on the
> ~6.3% of positions where Q9 and direct preparation disagree on the projected
> Q3 code after 300 updates.

This is a robust **where** result. It is not yet a **code-label sufficiency**
result.

Do not say "the discrete ternary assignments alone explain 96%." Arm 10
transfers exact continuous Q9 masters on M.

Canonical aggregate:
`replications/v11_hybrid_factorial_aggregate_summary.md`

### Current next-question priority

Run a code-identity-vs-continuous-position intervention on M.

Recommended first intervention:

- rebuild D and S at step 300 under the matched schedule;
- define M as before;
- keep D outside M;
- on M compare at least:
  1. exact S masters (positive control; reproduces arm 10);
  2. **S-code prototype** masters using the original Q3 scale:
     `w = (2/3) * alpha * c_S`, where `c_S in {-1,0,+1}`;
  3. optionally a minimal-crossing construction starting from D but moving each
     M weight only far enough to enter S's Q3 code region.

All arms should have the same intended S projected Q3 code on M at continuation
start. If the prototype/minimal-crossing arms retain most of the exact-S gain,
discrete code identity is sufficient for much of the effect. If they collapse
toward D despite identical projected codes, the continuous boundary-relative
geometry inside M is essential.

### v12 design refinement — code identity vs boundary-relative geometry on M

External review agrees with the next causal question and adds useful controls.
Use seed/order 1729 first.

Important wording: v11 proves exact equality of the **quantized Q3 forward
weights** within the paired arms (same projected codes under the same original
Q3 scales, with identical diagnostics). Do not describe this as literal
byte-for-byte equality of serialized model files.

Recommended v12 arms, all rebuilt in one run from matched-schedule step-300
states D and S:

1. **D baseline**
   - D masters everywhere.

2. **Exact-S-on-M positive control**
   - S masters on the true disagreement mask M;
   - D elsewhere;
   - should reproduce v11 arm 10.

3. **S-code prototype on M**
   - preserve S's projected Q3 code on every position in M;
   - replace the exact S master with the Q3 reconstruction prototype
     `w = (2/3) * alpha0 * c_S`;
   - D elsewhere.

4. **Minimal-crossing S-code on M**
   - start conceptually from D;
   - move each M weight only far enough to enter S's Q3 code region under the
     original Q3 scale, with a small preregistered normalized epsilon inside the
     target region;
   - D elsewhere.

The exact-S, prototype, and minimal-crossing arms must all assert the **same
projected Q3 codes on every quantized weight** before continuation. Thus they
start from the same ternary forward model and differ only in hidden continuous
master position.

Interpretation:

- prototype ~= exact-S: S code identity is sufficient for most of the benefit;
- minimal-crossing ~= exact-S: merely crossing into the Q9-selected code region
  is sufficient;
- prototype helps but minimal-crossing does not: depth/location inside the Q3
  region matters, but exact Q9 values may not;
- both prototype and minimal-crossing collapse toward D: precise continuous
  Q9-prepared geometry on M is essential.

Because the prototype can be a large displacement, especially for target
`c_S=0`, log the disagreement mask by transition class:

- `c_D -> c_S` transition counts per layer;
- target-code groups `c_S=0` versus `|c_S|=1`;
- normalized distances `w/alpha0` from the relevant Q3 boundary for D and S.

If v12 is partial/ambiguous, follow with target-zero vs target-nonzero submask
interventions rather than trying to infer those contributions from aggregate
statistics alone.

#### Random reassignment control

Do **not** use an unconstrained random 6.3% code flip.

A scientifically matched random control should preserve the real mask's
distribution as closely as practical:

- within each layer and source direct-Q3 code `c_D`, sample the same number of
  positions from outside M as occur in M;
- assign target codes using the observed per-layer `c_D -> c_S` transition
  counts from M (randomly permuted among the matched sampled positions);
- place those random reassigned weights at the same standardized
  representative used by the true-M comparison, preferably the Q3 prototype.

This controls for mask size, layer allocation, source-code mix, and transition
type. It asks whether the **specific positions selected by Q9** matter, versus
merely making the same number and kinds of ternary code changes elsewhere.

Run the true-M identity/position arms first. The matched-random arm is valuable
and can be included in the same seed-1729 job if memory/runtime remains
comfortable.


## v12 — COMPLETED seed-1729 code identity vs continuous position

Job: `6ac642a0df2184ac91ac018d`  
Pinned code: `22639f5b225e56be009886bf467a409910796eef`

All exact/prototype/minimal equal-forward assertions passed with zero projected
Q3 Hamming and identical pre-continuation diagnostics.

| Arm | Final loss | Interpretation |
|---|---:|---|
| D | 5.6136 | direct baseline |
| exact | 4.9329 | exact Q9 masters on true M |
| prototype | **4.8884** | standardized Q3 reconstruction level for Q9 code |
| minimal | 5.4979 | epsilon=0.01 inside Q9-selected region |
| random | 5.6793 | transition-matched random positions |

Key effects:

- exact gain vs D: **0.6807 nats**;
- prototype gain: **0.7252 nats** = **106.5%** of exact positive-control gain;
- minimal gain: **0.1157 nats** = **17.0%** recovery;
- matched-random gain: **-0.0657 nats** = worse than D.

Current mechanism interpretation:

> Q9's exact within-region FP32 coordinates are unnecessary, but simply choosing
> the Q9 code and barely crossing its boundary is also insufficient. The useful
> information is the **specific position/code selection**, combined with a
> master state placed safely inside the selected Q3 region. The canonical Q3
> reconstruction prototype is sufficient on seed 1729 and slightly better than
> the exact Q9 master values.

The matched-random control indicates that applying the same number and types of
code changes to different positions does not help.

Do not yet claim this finer result is replicated; v12 has only seed 1729.

Canonical files:

- `results/run_v12_code_identity_position_summary.md`
- `results/run_v12_code_identity_position_seed1729_2026-10-07.json`

### Current next-question priority

Replicate the exact v12 design on orders 271828 and 424242.

If prototype again approaches or beats exact, minimal crossing remains weak,
and matched-random remains non-beneficial on both orders, the mechanism can be
upgraded to a three-order conclusion:

1. Q9 identifies useful **which-position/which-code** assignments;
2. masters need to be placed well inside the selected ternary regions;
3. exact Q9 continuous values are not required.

Only after those two confirmations should we decide whether any target-zero vs
target-nonzero submask experiment is necessary.

## v12 code/position mechanism — REPLICATED 3/3

Confirmatory jobs:

- seed 271828: `6ac64ccddf2184ac91ac092d`
- seed 424242: `6ac64cd2df2184ac91ac092f`

All construction assertions passed.

| Seed | Exact gain | Prototype recovery | Minimal recovery | Random recovery |
|---:|---:|---:|---:|---:|
| 1729 | 0.6807 | 106.5% | 17.0% | -9.7% |
| 271828 | 0.6813 | 107.7% | 16.4% | -10.0% |
| 424242 | 0.6443 | 106.5% | 20.5% | -12.4% |

Aggregate:

- mean true-mask size: **6.313%**;
- mean exact-Q9-on-mask gain: **0.6688 nats**;
- mean prototype gain: **0.7150 nats**;
- mean prototype recovery: **106.9%**;
- mean minimal recovery: **18.0%**;
- mean matched-random recovery: **-10.7%**.

Canonical mechanism conclusion:

> Q9's useful prepared state is concentrated on the ~6.3% disagreement
> positions. What matters there is the specific **which-position / which-code**
> choice plus useful placement inside the chosen ternary region. Exact Q9
> within-region FP32 coordinates are unnecessary; merely crossing the boundary
> is insufficient; and applying the same transition pattern to different
> positions is harmful.

This is now replicated across all three established training orders.

### Mechanism phase status

**STOP CONDITION MET.**

No additional mechanism jobs are required for the current SmolLM2/WikiText-2
story unless a new question is deliberately opened.

Possible next phases, each scientifically separate:

1. cross-model / cross-dataset generalization;
2. cheap practical recipe for predicting/learning the useful Q9-selected
   positions/codes without a full Q9 phase;
3. quantizer redesign using the position/code + prototype insight.

Canonical aggregate:
`replications/v12_code_identity_position_aggregate_summary.md`.

Compact confirmatory JSON summaries:

- `results/run_v12_code_identity_position_seed271828_compact_2026-10-07.json`
- `results/run_v12_code_identity_position_seed424242_compact_2026-10-07.json`

Full confirmatory traces remain auditable in the Hugging Face job logs above.

## ACTIVE OPTIONAL v13 — depth sweep + firmness controls

This is an optional closing experiment; the v12 three-order mechanism result
remains canonical and already met the stop condition.

Seed/order: 1729 first.

Hugging Face job: `6ac65927e7a0dae8a277c24c` (A10G-small, 65-minute cap).

Pinned code:
`fc68603caa1bf0e33faadb28f97a62a1bd0e3957`

Eight continuation arms:

- D;
- depth d=0.03, 0.25, 0.50, 0.75, 1.00 for Q9's selected codes on M;
- B1 = direct's own codes prototyped on M;
- B2 = direct's own codes prototyped on D's own changed-vs-initial set.

Depth d=.03 exactly re-anchors v12 minimal crossing; d=1 exactly re-anchors v12
prototype.

For depth arms log Q9-code survival after 100/300/900 continuation steps under:

- current learned Q3 scale;
- fixed original alpha0;

and split target zero vs target nonzero.

All depth arms must have identical projected Q3 forward models before
continuation. B1/B2 must be identical to D before continuation.

Replication only if the seed-1729 depth/firmness result is clean.


### v13 split relaunch active

The original monolithic v13 job timed out after passing all construction checks.
It remains a technical failure with no experimental result.

Timeout-safe split, same seed-1729 design:

- v13A depth sweep — HF job `6ac6a265df2184ac91ac410d`
  - D + d003/d025/d050/d075/d100
- v13B firmness controls — HF job `6ac6a272df2184ac91ac412c`
  - D + B1 + B2

Both use A10G-small with 120-minute caps.

Pinned shared commit:
`9332a0a0b4063f7ed6786aa29fd049429cd50380`

Do not combine either split result with the timed-out monolithic run except for
the already-passed construction-check audit trail.


## v13 optional closing experiment — seed 1729 COMPLETE

Jobs:

- v13A depth sweep: `6ac6a265df2184ac91ac410d`
- v13B firmness controls: `6ac6a272df2184ac91ac412c`

Pinned split commit:
`9332a0a0b4063f7ed6786aa29fd049429cd50380`

All equal-forward assertions passed.

Depth sweep:

| d | loss | recovery vs d=1 | Q9-code survival at step 900 |
|---:|---:|---:|---:|
| 0.03 | 5.4948 | 16.4% | 51.76% |
| 0.25 | 4.9262 | 94.8% | 86.48% |
| 0.50 | **4.8850** | 100.5% | 97.32% |
| 0.75 | 4.8876 | 100.1% | 99.13% |
| 1.00 | 4.8884 | 100.0% | 99.45% |

Direct baseline: 5.6136.

Firmness controls:

- B1 = 5.6403 (worse than D by 0.0267 nats)
- B2 = 5.6195 (worse than D by 0.0059 nats)

Interpretation:

> The depth effect is sharply saturating. Barely crossing the boundary is weak,
> d≈0.25 already recovers nearly all benefit, and d≈0.5 reaches the plateau.
> Q9-code survival rises in parallel from ~52% to ~97-99%. Direct's own code
> decisions do not benefit from prototype snapping, so firmness is not a generic
> direct-Q3 optimization trick.

Current learned-scale and fixed-alpha0 survival are nearly identical, arguing
against scale drift as the main source of the survival pattern.

This is seed 1729 only. The result is clear enough to satisfy the preregistered
replication criterion, but replication is optional because v12 already met the
mechanism stop condition.

Do not launch 271828/424242 v13 replications without explicit user approval.

Canonical summary:
`results/run_v13_depth_firmness_summary.md`.


## ACTIVE NEXT PHASE — G1 cross-family generalization

The next scientific phase is **G1**, not v14.

Selected Family-B small model:
`ibm-granite/granite-4.0-350m`.

G1-0 architecture/freezing coverage was acceptable. G1-0b established a
Granite-specific precision rule **before any D-vs-S result**: FP32 and BF16
source paths are finite and closely matched, while FP16 is non-finite. Granite
G1 therefore uses FP32 masters + BF16-rounded source + BF16 autocast/BF16
teacher. Do not revert Granite jobs to FP16.

Canonical roadmap:
`G1_GENERALIZATION_PLAN.md`.

Do not jump directly to large-model scale-up or reopen more
SmolLM2-360M mechanism work. The gated sequence is:

1. G1-0 architecture / quantization smoke — complete, exposed FP16 blocker;
2. G1-0b BF16 precision diagnostic — complete, passed;
3. G1-1 direct-only schedule calibration — complete; constant 1e-4 selected by validation and frozen;
4. G1-2 one-order D/S staging gate — complete and positive on order 1729;
5. G1-3 compressed mechanism test — complete and positive on order 1729;
6. two fixed-protocol confirmatory orders only if the first-order result is
   scientifically interpretable.

If G1 is positive, the intended next quadrant is SmolLM2-1.7B, followed by the
larger Granite sibling.

A Family-B-small negative result does **not** by itself cancel the
SmolLM2-1.7B within-family scale test.

Before any conclusion, inspect every completed order individually. Preserve the
existing documentation workflow and chat-reporting format already recorded in
this handoff.


### G1-1 calibration result

Direct-Q3-only validation screen, seed/order 1729:

| Schedule | val loss | PPL | code movement @300 |
|---|---:|---:|---:|
| constant 1e-4 | **5.8128** | **334.56** | 5.323% |
| warm100 -> 3e-4, cosine -> 1e-4 | 6.0827 | 438.21 | 11.616% |
| warm100 -> 1e-3, cosine -> 1e-4 | 6.7402 | 845.70 | 27.231% |

The held-out test set was not used and no Q9 arm was present. Constant 1e-4 is
therefore frozen for all G1-2 treatments. Do not retune after seeing Q9.

Canonical summary:
`results/run_g1_1_granite350m_calibration_summary.md`.

G1-2 pinned script:
`g1_granite350m_staging_gate.py` at
`f3a1f88890d1b70264f25f1be8c4d23d84594825`.

G1-2 job:
`6ac6f487df2184ac91ac693e`.


### G1-2 positive staging result

Granite-350M seed/order 1729:

| Metric | Direct Q3 | Q9→Q3 | Q9 advantage |
|---|---:|---:|---:|
| held-out loss | 5.66581 | **5.53495** | **0.13086 nats** |
| PPL | 288.82 | **253.40** | **12.27% lower** |
| teacher top-1 | 27.53% | **28.04%** | **+0.51 pp** |
| KL | 2.49025 | **2.38307** | **0.10718 lower** |

At equal 300-step compute, fixed-Q3 direct is 5.81745 loss while the
Q9-prepared masters projected through the same original Q3 scales are 7.74247
loss: Q9 is **1.92502 nats worse immediately**, yet later finishes better.

Step-300 projected geometry:
- D changed vs source: 5.3242%;
- S changed vs source: 5.2723%;
- D-vs-S disagreement mask: **6.0246%**;
- changed-set Jaccard: 27.53%.

Interpretation: first cross-family order reproduces the key
trainability-not-entry-quality signature. This is **one order only**, not yet a
replicated Granite conclusion.

Canonical summary:
`results/run_g1_2_granite350m_staging_summary.md`.

### ACTIVE G1-3 mechanism job

HF job:
`6ac6fe5fdf2184ac91ac6c1f`

Pinned script:
`g1_granite350m_compressed_mechanism.py`

Pinned commit:
`c32fbc16f463039257d996a7b32c9eca5eada681`

Arms:
D, S, M-exact, M-d50, Random-d50.

All arms use original Q3 scales + fresh Adam + constant 1e-4 for steps
301-1200. Construction assertions must pass before continuation. The job asks
whether Granite's ~6.02% disagreement set carries the staging gain and whether
true-position d=0.5 placement beats a matched random reassignment.

Do not launch confirmatory orders or larger-model jobs until G1-3 is inspected.


### G1-3 Granite mechanism result

Job:
`6ac6fe5fdf2184ac91ac6c1f`

Pinned code:
`c32fbc16f463039257d996a7b32c9eca5eada681`

All construction assertions passed.

Final losses:
- D 5.66205
- S 5.53495
- M-exact 5.56525
- M-d50 **5.51134**
- Random-d50 5.73514

Key effects:
- full S gain vs D: 0.12710 nats;
- exact true-M gain: 0.09680 = **76.2%** of full S;
- true-M d=0.5 gain: 0.15071 = **118.6%** of full S;
- matched-random d=0.5: -0.07308 = harmful.

The true mask is 6.0246% of targeted positions.

This is qualitatively consistent with the Smol mechanism:
specific Q9-selected positions/codes matter, moderate interior placement is
sufficient, exact Q9 continuous coordinates are not required, and applying the
same transitions elsewhere is harmful.

Quantitative difference: exact-M localization is only 76.2% on Granite versus
~96.4% canonical Smol. Preserve that heterogeneity.

M-d50 Q9-code survival:
- step100 99.677%
- step300 97.977%
- step900 90.696%
with fixed-alpha0 almost identical.

This is **one Granite order only**. No confirmatory order has been run yet.
Canonical summary:
`results/run_g1_3_granite350m_mechanism_summary.md`.

Next gated action is G1-4/G1-5 confirmation on orders 271828 and 424242, with
the exact frozen G1 protocol. Do not launch without a separate user decision.


### ACTIVE Granite confirmation pair

Exact G1-3 confirmation scripts are pinned together at:
`6ed697a0ca5d0d19ed5ebff04b81a2ed0d259230`.

Jobs:
- seed/order 271828: `6ac70af0df2184ac91ac6ffc`
- seed/order 424242: `6ac70af2df2184ac91ac7000`

Both are A10G-small with 75-minute caps and differ from seed 1729 only in
`SEED` / shuffled training order.

Do not aggregate until both complete and each order is inspected individually.
Do not launch any larger-model jobs from this state.


### Cross-model comparison guardrails

External/secondary reviews may compare the Granite G1 result against different
historical Smol regimes. Preserve these distinctions:

1. **Granite's ~5.3% step-300 Q3 code movement is ~8x the old Smol v7
   constant-1e-4 regime (~0.68%), but not ~8x the canonical tuned Smol v10
   regime.** Under v10's direct-selected warmup/cosine schedule, Smol direct
   moves about **4.41%** and Q9 about **4.88%** of projected Q3 codes at step
   300. Therefore "Granite direct explores ~8x more" is a valid same-LR
   historical comparison, but not a schedule-independent cross-family fact.
   Treat "more direct exploration leaves less for Q9 to add" as a hypothesis,
   not a finding.

2. **G1-2 direct 5.66581 vs G1-3 D 5.66205 is not a clean BF16
   run-to-run-noise estimate.** G1-2 direct uses continuous direct-Q3 training
   through 1200 steps, whereas G1-3's mechanism baseline deliberately rebuilds
   the 300-step D state and gives the common continuation **original Q3 scales
   + fresh Adam**. The endpoints are close, which is reassuring, but their
   ~0.0038-nat difference should not be attributed purely to BF16 noise.

3. **"Firm placement beats full staging" is directionally true, but quote the
   correct comparator.** On Smol seed 1729, the v12 prototype is 4.8884 versus
   full staged S 4.9010, only ~**0.0126 nats** better; its ~0.0445-nat edge is
   versus the **exact-S-on-M** arm (4.9329), not versus full S. Across the three
   Smol orders, prototype beats full S by only about **0.02 nats on average**.
   On Granite seed 1729, M-d50 beats full S by ~**0.0236 nats**.

These are interpretation guardrails, not new experiment results.


### Granite three-order result — mixed

All three Granite mechanism orders are now complete.

| Seed | D | S | S gain | M-d50 | Random |
|---:|---:|---:|---:|---:|---:|
| 1729 | 5.66205 | 5.53495 | +0.12710 | **5.51134** | 5.73514 |
| 271828 | **5.72673** | 5.84103 | **−0.11431** | 5.78502 | 5.79268 |
| 424242 | 5.80192 | 5.74057 | +0.06135 | **5.71016** | 5.81984 |

Interpretation boundary:
- Full S > D only 2/3. **Do not claim 3/3 Granite staging replication.**
- Q9 immediate fixed-Q3 entry is worse than D 3/3.
- Mean D-vs-S mask is 6.35%.
- M-d50 beats full S 3/3.
- M-d50 beats matched-random 3/3.
- matched-random is harmful vs D 3/3.
- M-d50 beats D only 2/3.
- seed 271828 is a real negative and must remain visible.

The best current cross-family wording is:

> Granite reproduces the ~6% disagreement geometry and worse-entry signature,
> but the final staging advantage is order-sensitive. Standardized d=0.5
> placement of Q9-selected commitments is more robust than the full staged
> master state, while matched random placement is consistently harmful.

Canonical aggregate:
`replications/g1_granite350m_mechanism_aggregate_summary.md`.

Deferred idea only, **not an active experiment**: compare the Q9 disagreement
mask with AWQ-style activation-salient channels/weights to test whether Q9 is
discovering static quantization saliency or a distinct optimization-path
saliency. Do not launch this before the next phase is deliberately chosen.


### Granite negative-order forensics

No-new-compute comparison of 1729 / 271828 / 424242 is now in:
`replications/g1_granite350m_seed271828_forensics.md`.

Strongest clue: the negative seed is already abnormal during Q9 preparation.

- native Q9−D val loss @300: −0.160 / **+0.278** / −0.138;
- Q9/D grad ratio @100: 0.82 / **3.33** / 0.76;
- D-vs-S mask: 6.02% / **6.99%** / 6.03%;
- changed-set Jaccard proxy: ~27.5% / **~19.7%** / ~27.5%;
- M-d50 minus matched-random advantage:
  0.224 / **0.0077** / 0.110 nats.

On 271828, Q9's off-mask state is harmful too. Replacing off-mask S masters
with D masters recovers 0.0365 nats. d=0.5 then repairs another 0.0195, but the
true-M arm still loses to D and barely beats matched random.

Global code histograms/scales, late code churn and d50 survival are not strong
outliers.

Current hypothesis (post-hoc, not proven): Q9 benefit depends on successful
native-Q9 preparation / assignment discovery. Seed 271828 appears under-settled
and selects a larger, more disjoint, lower-quality set of ternary commitments.

Potential future predictor: native Q9 vs native D validation at step 300 plus
prep gradient settling. **Do not use as a validated gate yet; n=3 and post-hoc.**


### COMPLETED G1-6 — trajectory-aligned commitments on hard seed 271828

Job:
`6ac71a23df2184ac91ac73ca`

Pinned code:
`10110ffba0ce9b1069f015cd32b235958e37f6c9`

Development question:
Does Q9 help when its proposed d=0.5 commitments are aligned with the local
seed-conditioned Q3 gradient field?

Selector:
- build D300 and S300 as before;
- at D300/original-Q3 scales, accumulate Q3 master gradients over the last 24
  already-consumed prep chunks (277–300);
- score each true-M position as
  `-grad * (w_d50 - w_D)`;
- within every layer and D→S transition class, split positions into exactly
  matched top and bottom halves by score.

Arms:
- D
- All-d50
- AlignTop50-d50
- AlignBottom50-d50

All arms get fresh Adam, original Q3 scales, constant 1e-4 and the same
900-step continuation.

Primary signal:
Top50 better than matched Bottom50 and better than All-d50.
Practical rescue additionally requires Top50 < D held-out loss.

This is deliberately one development run on the known hard seed. Do not launch
1729/424242 or larger-model replications until this result is inspected.


### G1-7 initial attempt — technical failure only

Job:
`6ac722dfdf2184ac91ac75e9`

Pinned code:
`ab02c7a64a357bf792eee8d361032d2e2310bb1b`

Seed/order:
271828 only.

Purpose:
test whether Granite's earlier use of the v7-like constant 1e-4 schedule, rather
than Smol's later v10-v13 matched schedule, explains the hard-order failure.

Schedule:
- global steps 1-100 warm linearly to 1e-3;
- global steps 101-1200 cosine decay to 1e-4;
- Q9 prep is steps 1-300;
- continuation uses steps 301-1200 without LR restart.

Granite-specific safety adaptation:
BF16 autocast/BF16 teacher, no FP16 GradScaler; FP32 masters and clip=1.0 remain.

Arms:
D, S, M-exact, M-d50, Random-d50, with original Q3 scales and fresh Adam at
the mechanism continuation exactly as in prior Granite G1-3/G1-4/G1-5.

Important context:
G1-1 found warm1e-3 worse than constant1e-4 at the *300-step direct-Q3
calibration checkpoint*. G1-7 intentionally tests the full v10-v13 trajectory
rather than claiming the transferred schedule is already Granite-optimal.

Do not launch other seeds or larger models until G1-7 is inspected.


### COMPLETED G1-7b — Granite v10-v13 schedule retry

Original G1-7 job:
`6ac722dfdf2184ac91ac75e9` — **technical failure only**.

Failure happened after D300 and S300 preparation, before continuation/endpoints:
the exact full-size per-layer/source→target matched-random control was
mathematically infeasible in at least one stratum
(342,644 available outside-M positions vs 595,275 required).

Preregistered amendment:
omit Random-d50 and keep the schedule-transfer scientific core unchanged.

Retry:
`6ac7248adf2184ac91ac768d`

Pinned amended code:
`808fd4975d16111d9c1c841c44d9038a01995ba3`

Arms:
D, S, M-exact, M-d50.

Do not make a new random-position specificity claim from G1-7b. No other seed
or larger-model job is authorized automatically.


### G1-7b FINAL — aggressive Smol schedule does not transfer to Granite 271828

Job:
`6ac7248adf2184ac91ac768d` — completed successfully.

Pinned amended code:
`808fd4975d16111d9c1c841c44d9038a01995ba3`.

Raw:
`results/run_g1_7b_granite350m_v10schedule_seed271828_2026-10-08.json`

Summary:
`results/run_g1_7b_granite350m_v10schedule_summary.md`

Four continuation arms and all equal-forward assertions passed. Random-d50
was **not** run due to the preregistered control-feasibility amendment.

| Arm | Constant-LR 271828 loss | Transferred Smol v10–v13 loss |
|---|---:|---:|
| D | 5.72673 | **6.00895** |
| S | 5.84103 | 6.36920 |
| M-exact | 5.80453 | 6.35442 |
| M-d50 | 5.78502 | **6.30402** |

Staged deficit vs D grows from 0.11431 to **0.36025 nats**; M-d50 still
loses to D by 0.29508 nats.

Preparation with the 100-step warmup-to-1e-3 / global cosine schedule is
qualitatively different:
- D code movement vs initial: **27.107%**
- S code movement vs initial: **23.432%**
- D/S projected-Q3 disagreement: **32.7145%** = 81,642,667 positions, versus
  6.9946% under constant 1e-4.
- Native Q9−D validation loss @300 is **+0.38936**.
- Preclip Q9/D grad-norm ratio @300 is **5.15×**.
- M-d50 code survival @900 is **60.75%** learned / 60.70% fixed alpha0,
  versus **89.59%** under constant 1e-4.

Meaning: copying the Smol v10–v13 peak LR to Granite does not cure the known
hard order; it destabilizes the discrete commitment geometry and degrades both
D and S. This *does not prove* the mask growth alone caused the loss.

The original G1-7 job `6ac722dfdf2184ac91ac75e9` failed *technically*
before any continuation because exact matching of full random mask became
infeasible: 595,275 required vs 342,644 available outside M for one stratum.
Do not treat it as a scientific negative or claim a new matched-random result.

This is one known-hard-order development test. No new seeds or larger-model
jobs have been launched. Do not override the earlier preregistered Granite
G1-1 direct calibration: constant 1e-4 remains the better tested schedule
for this order.

Potential future direction (no authorization to run): if investigating a
family-aware LR, preregister intermediate peak-LR candidates and diagnostics
*before* new held-out comparisons. Or return to assignment-level mechanism
tests under the established constant-LR regime.


### COMPLETED G1-8 — Granite 424242, Smol v10-v13 LR schedule

HF job: `6ac79551e7a0dae8a2788f0b` (submitted 2026-10-08 13:06 UTC; A10G-small;
75-minute cap; detached).

Pinned code: `39399baff38b15f961e9571f142e193a783c925b`.
Script: `ternary_pet/g1_granite350m_v10_schedule_mechanism_seed424242.py`.

This is the one authorized targeted schedule-transfer test on the *weaker
positive* Granite order 424242. The previously tested order 271828 was a
known negative under constant LR and worse under the Smol v10-v13 schedule.
424242 was the harder of the remaining two **positive orders** by historical
staging gain: +0.06135 nats vs 1729's +0.12710. It was chosen using known
outcomes; it is not a fresh random confirmation order.

G1-8 differs from G1-7b successful four-arm script by exactly one seed
assignment `271828` to `424242`. Events and `kind` retain some G1-7
prefixes, so record results using **job ID + order 424242**.

Arms D, S, M-exact, M-d50, with v10-v13 100-step warmup to 1e-3 and
cosine to 1e-4 through 1200 steps, same 300 Q3/Q9 prep + 900 Q3
continuation, Granite-safe BF16/FP32-master implementation, same matched
training-order chunks and held-out evaluator. Random-d50 is intentionally
omitted as in G1-7b.

Frozen historical constant-LR 424242 held-out results:
D=5.80192, S=5.74057, M-exact=5.74471, M-d50=5.71016,
mask=6.02956%.
Historical G1-7b transferred-LR 271828:
D=6.00895, S=6.36920, M-exact=6.35442, M-d50=6.30402,
mask=32.7145%.

When job completes: verify status and equal-forward assertions, archive raw
final JSON and per-run summary, compare every final arm to the historical
constant-1e-4 424242 results, log movement, native prep, mask, survival and
effect on staging gap, then update living report, chronology and handoff.
Do not claim a family-universal effect or new matched-random control.

No further new seed or model jobs authorized.


### G1-8 COMPLETED — Granite 424242 confirms schedule sensitivity, but d50 still helps

HF job:
`6ac79551e7a0dae8a2788f0b` — **COMPLETED** 2026-10-08 13:30 UTC.

Pinned code:
`39399baff38b15f961e9571f142e193a783c925b`.
The one-line-only change from successful G1-7b is `SEED=271828` →
`SEED=424242`. The result JSON retains inherited `kind=g1_7...`
and log event `g1_7`; identify it by actual `seed=424242` and job ID.

Raw:
`results/run_g1_8_granite350m_v10schedule_seed424242_2026-10-08.json`
Summary:
`results/run_g1_8_granite350m_v10schedule_summary.md`

| Arm | Historical constant-LR 424242 | Transferred Smol LR 424242 |
|---|---:|---:|
| D | 5.80192 | 6.05708 |
| S | 5.74057 | 6.06241 |
| M-exact | 5.74471 | 6.04609 |
| M-d50 | **5.71016** | **6.02857** |

All four endpoints worsen, but **M-d50 remains better than D by
0.02850 nats**, whereas full S loses to D by 0.00534 nats (near tie).
All construction assertions passed; no Random-d50 arm (as preregistered).
Old full S gain +0.06135; new full S gain -0.00534.

Native Q9 was still **better than D** at step300 by 0.09232 nats under
transferred Smol LR (S 6.21319; D 6.30551). On 271828 Q9 native
was worse by 0.38936 under the same schedule. Therefore two Granite
orders share *aggressive LR schedule sensitivity* but not identical
Q9 prep failure dynamics.

D/S projected Q3 disagreement increases from **6.0296%**
(15,047,433 positions) to **29.4904%** (73,596,548 positions).
Direct and Q9 code movements from initial reach 27.280% and 27.124%
respectively. M-d50 selected-code survival after 900 Q3 steps declines
from 90.57% to **57.14%** (learned scales; fixed-alpha0 57.12%).

Together with completed G1-7b (271828: D=6.00895, S=6.36920,
M-d50=6.30402, mask 32.7145%), this supports the statement
**Smol v10-v13 high peak 1e-3 is not a good direct transfer to Granite
under the current 1200-step recipe on either tested order**.
This does **not** prove the same outcome for untested 1729, a
model-family universal statement, or a causal role for mask inflation.
Seed 424242 was selected using the historical weaker positive
staging benefit; treat as a targeted development follow-up.

Do not launch seed 1729 or other jobs without a separate authorization.

Future research direction: boundary-margin/row-column-coupled Q3 code
predictor with *retained* original random baseline, an audited
margin-matched feasible subset, and proper no-heldout-leak evaluation.
Do not alter G1-8 retrospectively.


### G1-9 protocol and launch status

User approved one 4-arm Gaussian noise × gridward interpolation factorial
on Granite known-hard order **271828**, constant LR `1e-4`. All
settings frozen in `EXPERIMENT.md` before code. Code is on main at
pinned commit `6ce4c4056e8cfd3292c54f34bedc68bf7691688f`,
file `ternary_pet/g1_9_granite350m_gaussian_pull_seed271828.py`.

Arms D, G, P, GP. Each has 300+900 direct-Q3 updates with historical
Granite BF16 compute and original row-scale/fresh Adam switch at300.
Gaussian σ normalized to row α: 0.04 until step900, decays to 0 at1200;
gridward λ=0.10 at steps100..900 every100. A deterministic
32,768-weight stratified subsample measures per-update Q3 clean-code
flips and immediate two-step reversals. Final heldout is hard noise-free
Q3, no threshold tuning or test-driven selection.

Two identical attempts to launch on A10G-small returned **HTTP 429
at HF MCP login**. Neither returned a job ID. G1-9 is **NOT RUNNING**,
and no GPU run has been verified. No more automatic attempts in that turn.
Check actual job submission outcome before claiming progress, and
do not launch more seeds or tweak hyperparameters automatically.


#### G1-9 accepted by HF, GPU job pending scientific completion

Two earlier Hugging Face MCP 429 attempts returned no job IDs. On 2026-10-08 23:06:20 UTC, the subsequent authorized retry was accepted:
- Job `6ac821ec095c5780892ff7f9` — https://huggingface.co/jobs/codeflash85/6ac821ec095c5780892ff7f9
- initial stage SCHEDULING; A10G-small, 90-minute cap
- pinned script `6ce4c4056e8cfd3292c54f34bedc68bf7691688f`; no changes since preregistration
- check status/logs and archive the full FINAL_JSON if completed. Check D/G/P/GP 1200-step, pull counts, sampled flip/reversal diagnostics, and baseline D reproducibility. Do not attribute infrastructure errors to experiments.


### COMPLETED G1-9 — strong gridward pull effect (2026-10-08)

Accepted after two logged HF 429 attempts. Job
`6ac821ec095c5780892ff7f9`, **COMPLETED**
2026-10-08 23:32:17 UTC. Code pin
`6ce4c4056e8cfd3292c54f34bedc68bf7691688f`.

**All four direct Q3 arms and checks valid; full final JSON archived**:
- `results/run_g1_9_granite350m_gaussian_pull_seed271828_2026-10-08.json`
- `results/run_g1_9_granite350m_gaussian_pull_summary.md`

| Arm | Final heldout NLL |
|---|---:|
| D (clean) | 5.7267251685 |
| G (Gaussian only) | 5.7433702722 |
| P (gridward only) | **5.4897092953** |
| GP (combined) | **5.4878531992** |

D is **identical to historical Granite direct seed271828** at 5.726725.
P improves by **0.237016 nats**, PPL 306.96→242.19
(**21.10% reduction**), teacher-top1 agreement 27.66→31.08%.
GP improves by 0.238872 nats but the incremental improvement
over P alone is **0.001856**. G alone worse by 0.016645.

All arms had 1,200 AdamW updates, constant LR 1e-4, original rowwise
Q3 scale reset and fresh Adam at step300. P/GP applied
`W ← 0.90W + 0.10Q3(W)` at steps100–900 every100.
G/GP applied training-only normalized Gaussian σ_u=0.04,
annealed to zero steps900–1200, deterministic hard-Q3 validation.

Sampled continuation clean-code flips per step: D .000735745,
G .000741340, P .000094469, GP .000089281.
P is **87.2% less** than D. Final source-code movement D 10.335%,
P 5.281%. Observed flip rate is strongly associated with improved
loss, but reversal ratio changes much less: do not overclaim
oscillation as proven causal pathway.

All checks passed, single historically hard order and one σ/λ choice.
No Q9, no random control in G1-9. WinQ-like gridward interp has prior
art. Important next action would be frozen-parameter replication
on a positive Granite order plus new-model confirmation; **not
automatically authorized**. Preserve negative G1-7b/G1-8 and
mixed original Granite 3-seed results, not retroactively rewrite them.

Earlier handoff's pending/429 text is historical: **G1-9 is now
complete**. No further jobs launched.


### G1-10 ACTIVE — one-line seed-only frozen replication of G1-9

**Job `6ac82c93fee2c900701711db`**, scheduling since
2026-10-08T23:51:47.315Z; URL:
https://huggingface.co/jobs/codeflash85/6ac82c93fee2c900701711db

Immutable code `e08659a51fd0d72ed85f01f8e7ce739283ca6c61`,
`ternary_pet/g1_10_granite350m_gaussian_pull_seed424242.py`,
identical to G1-9 `6ce4c4056e8cfd3292c54f34bedc68bf7691688f`
except `SEED=424242` instead of 271828.
Preregistered in `EXPERIMENT.md` before implementation and launch.

Single A10G-small job; no duplicates found. Four direct-Q3 arms D/G/P/GP,
1,200 steps each, constant 1e-4 LR, BF16 student/teacher, FP32 masters,
rowwise Q3 original-scale and fresh Adam reset at step300.
Gaussian noise in G/GP σu=.04 until900 then annealed to zero;
P/GP gridward convex 10% pull every100 steps through900.
Same WikiText-2 8192-token held-out evaluation, hard Q3 inference.
When complete: verify stage, checks, all arms, D vs historical 424242
loss5.80192, paired P vs D gain, GP vs P, code movement, sampled flip
rates; archive full FINAL_JSON and summary, update all reports.
**No further jobs authorized** beyond this single one.


### LATEST COMPLETED: G1-10 Granite 424242 replicates gridward pull

HF job `6ac82c93fee2c900701711db` completed
**2026-10-09 00:21:46 UTC**, pinned
`e08659a51fd0d72ed85f01f8e7ce739283ca6c61`.
New script `ternary_pet/g1_10_granite350m_gaussian_pull_seed424242.py`
differs **only in the SEED line** from completed G1-9.

Preregistered experiment D/G/P/GP; 1200 equal-budget
direct Q3 steps at constant LR1e-4; sigma_u0.04 training-noise
for G/GP; 10% gridward pulls nine times for P/GP; same BF16/FP32
Granite path, scale and Adam reset at step300. All arms/checks passed.
G1-10 complete, NOT active.

Seed424242 held-out loss:
- D **5.8019200563** PPL330.9344 (exact reproducibility of historic 424242 D)
- G **5.7563617378** PPL316.1958 (gain +0.045558)
- P **5.5281120539** PPL251.6683 (gain **+0.273808**, PPL −23.95%)
- GP **5.5353403389** PPL253.4940 (gain +0.266580; worse than P).

Seed271828 prior:
D5.726725, G5.743370, P5.489709, GP5.487853.
**P-only wins in BOTH selected Granite training orders**,
+0.237016 and +0.273808 nats; average +0.255412.
GP increment vs P changed sign, G alone also changed sign.
Only P method can currently be called cross-order robust
within Granite. It remains a WinQ-inspired published-method
adaptation, not original invention and not independently
confirmed in Smol.

Sampled 32768-weight continuation Q3 flip rates D .000728421,
P .000095486 on 424242 (86.89% fewer sampled flips).
This does not establish oscillation suppression as sole cause.
No Q9 arm or random-mask controls tested in these direct Q3
factorials. Preserve mixed Q9 cross-seed results and all
negative findings.

**Critical inherited script metadata:** Raw seed424242 JSON
still reports `kind=g1_9...`, `event=g1_9...`,
`historic_D_loss_271828=5.726725`,
`D_minus_historic_D_loss=0.075195`.
Those latter two values compare **different seeds** and
must NOT be used to assess within-G1-10 reproducibility.
Correct seed424242 historical D=5.8019200563, exactly
same as paired D in G1-10. Full raw data deliberately
unchanged, interpret with seed/job/code pin.

Results:
`results/run_g1_10_granite350m_gaussian_pull_seed424242_2026-10-08.json`
`results/run_g1_10_granite350m_gaussian_pull_seed424242_summary.md`.

Prior G1-9 result:
`results/run_g1_9_granite350m_gaussian_pull_summary.md`.

No further GPU jobs launched; next scientific request
could authorize frozen-setting cross-model replication
on Smol, or independent new Granite seeds. No
automatic expansion permitted.


### S1-1 ACTIVE — Gridward-pull cross-model transfer on SmolLM2

HF job `6ac83c3bfee2c90070171b1a` submitted 2026-10-09
00:58:35 UTC, initial stage SCHEDULING.
Monitor https://huggingface.co/jobs/codeflash85/6ac83c3bfee2c90070171b1a.
Code pin `a5634bce459092a503e7534e6b89b6d3a019548b`;
file `ternary_pet/s1_1_smol360m_gridward_direct_q3_seed1729.py`.
Preregistered first. No duplicate existing before launch.

Scientific question: do nine periodic 10% master-to-current-
ternary-grid pulls improve tuned direct Q3 on Smol, after
Granite G1-9/G1-10 both improved P on seeds271828,424242?
This is true cross-family testing of the pull rule, not
importing Granite's LR or reset: Smol v9 tuned direct
1200-step continuous AdamW, warmup100 to1e-3 then
cosine→1e-4, FP32 masters BF16-rounded source and FP16 AMP
student/teacher. Two equal-budget arms D/P on same seed1729
order and identical 8192-token test slice. Q9 and Gaussian
absent. D historical seed1729 test-loss target 5.59572218731.

When complete inspect FINAL_JSON; check baseline reproducibility,
arm step counts, exactly nine pulls in P and none in D,
AMP skipped-step opportunities, deterministic code movement
and flip/reversal diagnostics. Raw JSON and reports to be
archived, *including negatives*. Do not launch other seeds
or hyperparameter tuning without separate request.


### LATEST COMPLETED: S1-1 — GRIDWARD TRANSFER FAILS ON SMOL

**Job COMPLETED**, `6ac83c3bfee2c90070171b1a`,
2026-10-09 01:18:20 UTC.
Code pin `a5634bce459092a503e7534e6b89b6d3a019548b`,
`ternary_pet/s1_1_smol360m_gridward_direct_q3_seed1729.py`.
Frozen prereg in `EXPERIMENT.md`. No job error.

Direct ternary Smol v9 schedule (warmup100 to1e-3,
cosine to1e-4 at step1200, continuous Adam without
scale/optimizer reset at300). Two arms, D and P.
P nine 10%-gridward pulls after global steps100,...,900.
Both arms exactly 1200 step opportunities, SIX AMP-skipped
steps each = 1194 effective optimizer updates, all
construction checks pass. Within-job D exactly reproduces
historical tuned direct seed1729 loss **5.595722187310457**.

S1-1 heldout:
- D loss **5.5957221873**, PPL269.2720, top1 29.87%, KL2.66403
- P loss **5.8457463756**, PPL345.7605, top1 27.28%, KL2.93324
- P WORSE **0.2500241883 nats**, PPL +28.4056%.

Despite loss deterioration, P suppresses sampled
noise-free Q3 transitions **78.41%**:
D per sampled weight/update 0.0004350535,
P 0.0000939178, sampled 32768 weights every step.
Final source-code movement D6.960%, P3.445%.
Validation @300 D5.98809/P5.97667; @600 D5.74032/
P5.67064; @900 D5.41510/P5.58253; @1200
D5.25594/P5.57738. Thus P initially mildly
improved validation but fell decisively behind later,
consistent with (not proof of) premature commitment.
Do **not** infer simply decreasing flips improves accuracy.

Granite G1-9/G1-10 P was positive at matched constant
LR1e-4 (+0.237 and +0.274 nats), but had Granite-specific
scale restoration/fresh Adam at step300. This S1-1
negative falsifies the **unqualified cross-architecture
rule** of porting the same nine 10% pulls under
the architecture's preferred LR schedule.
Architecture vs learning-rate/reset effects remain
unresolved; do not claim Smol can never benefit at
other strengths/schedules.

This does NOT negate previously positive Smol
Q9→Q3 three-seed matched-schedule benefits.
The S1-1 experiment was direct-only (no Q9/Gaussian).
No retuning against viewed heldout values.

Raw archive:
`results/run_s1_1_smol360m_gridward_direct_q3_seed1729_2026-10-09.json`.
Detailed interpretation:
`results/run_s1_1_smol360m_gridward_direct_q3_seed1729_summary.md`.
HF link: https://huggingface.co/jobs/codeflash85/6ac83c3bfee2c90070171b1a

S1-1 prior "ACTIVE" sections are historical.
**No further GPU jobs launched.** Focus next on
analysis of assignment margins, boundary transition
retention, or an independently preregistered adaptive
schedule (requires separate user approval for GPU).
