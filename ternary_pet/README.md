# Ternary Pet Experiments

## LATEST — R4 2+2 MATRIX: all FOUR scientific GPUs submitted in TWO user prompts (2026-10-10 UTC)

**Batch A, Granite 4.0 350M** previously submitted: historically hard seed271828 [HF `6ac9a43d095c57808930ccfe`](https://huggingface.co/jobs/codeflash85/6ac9a43d095c57808930ccfe) and easy seed424242 [HF `6ac9a443095c57808930cd03`](https://huggingface.co/jobs/codeflash85/6ac9a443095c57808930cd03). Both remained RUNNING at start of this second request; final science unknown. Immutable Granite script SHA `33dfae38bd016946ad5d37bd633fdf02818fbacd`, Granite BF16 + LR constant1e-4.

**Batch B, SmolLM2-360M-Instruct** **NOW SUBMITTED at user request**: seed271828 [HF `6ac9a5c2fee2c90070180ce3`](https://huggingface.co/jobs/codeflash85/6ac9a5c2fee2c90070180ce3), seed424242 [HF `6ac9a5c9095c57808930cdde`](https://huggingface.co/jobs/codeflash85/6ac9a5c9095c57808930cdde). One A10G-small, 90m hard limit per job. Exact seed-only scripts [271828](r4_smol_q9_hybrid_seed271828.py) · [424242](r4_smol_q9_hybrid_seed424242.py), immutable common SHA `65af3057b248f38a6d46ad8b7cd1bc1c52d2e654`. **Prepaid-GPU CPU dynamic QA** [HF `6ac9a578095c57808930cd9c`](https://huggingface.co/jobs/codeflash85/6ac9a578095c57808930cd9c) **COMPLETED**, verifying both script AST/pins, nine Q9 codes, H/N inner levels match, outer ±4 restored in H only, STE gradients, post-switch original Q3 and seed-only script variation.

**Each of four GPU jobs runs the same D/W/N/H family experiment** (matched within seed): D direct Q3 1200, W original wide Q9(300)→Q3(900), N narrow T4 Q9(300)→Q3(900), H nonuniform Q9(300)→Q3(900) with narrow 7 inner levels and original wide ±α two extreme levels. Smol uses **original tuned warmup100→1e-3 cosine→1e-4, FP16 teacher/autocast + GradScaler, FP32 masters**, BF16-rounded source; D direct original continuous Adam, W/N/H original Q3 row scale + Adam/GradScaler reset at300. Granite uses previously calibrated **constantLR1e-4, BF16 teacher/autocast no GradScaler**, all arms reset source Q3 scales/fresh Adam at300. Consequently architectural and optimizer policies remain somewhat confounded across families; compare within-family paired NLL differences. Original WikiText2 test first64 chunks anchors; **primary first64 WikiText validation chunks**, matched 8192 tokens and per-chunk losses; historical seeds are **not new random confirmation**, samples may have appeared in pretraining. Smol historical anchors seed271828 D≈5.6228/W≈4.9510, seed424242 D≈5.6067/W≈4.9563, fixed tolerance ±0.08 nats. No source/output retuning after observing jobs.

**Frozen before jobs:** [R4 2+2 preregistration](research_log/r4_2026-10-09_granite_smol_easy_hard_two_batch_prereg.md). **Authoritative four job IDs, precise protocol/recovery rules:** [R4_GRANITE_SMOL_EASY_HARD_JOBS.md](R4_GRANITE_SMOL_EASY_HARD_JOBS.md). **No fifth GPU/retry planned.** Wait for all four terminal results, extract and archive original FINAL_JSONs, all validity checks, negative outcomes, comparisons and limitations; update reports then.

---


## LATEST — R4 Granite hard/easy pair SUBMITTED, Smol pair reserved for NEXT prompt (2026-10-09 EDT)

**User requested exactly 2 jobs in this prompt and 2 only in the following prompt, waiting for Granite to finish.** Batch A **TWO Granite-4.0-350m jobs submitted**, one per previously identified training-order seed, both using a four-arm D/W/N/H controlled hybrid Q9 comparison:

- **Hard seed271828** (historically full wide Q9 lost): [HF `6ac9a43d095c57808930ccfe`](https://huggingface.co/jobs/codeflash85/6ac9a43d095c57808930ccfe), A10G-small, 90m max.
- **Easy seed424242** (historically wide Q9 won): [HF `6ac9a443095c57808930cd03`](https://huggingface.co/jobs/codeflash85/6ac9a443095c57808930cd03), A10G-small, 90m max.

**Immutable single source SHA** `33dfae38bd016946ad5d37bd633fdf02818fbacd`, [hard script](r4_granite_q9_hybrid_hard_seed271828.py), [easy script](r4_granite_q9_hybrid_easy_seed424242.py); source scripts differ by exactly one seed line. Frozen [R4 2+2 prereg](research_log/r4_2026-10-09_granite_smol_easy_hard_two_batch_prereg.md), [complete job handoff](R4_GRANITE_SMOL_EASY_HARD_JOBS.md). BF16 Granite teacher/autocast, FP32 masters, historical constant LR1e-4, original WikiText2 QAT training split, W nine-state wide, N narrow, H seven narrow center levels with two ±4 outputs restored wide. All three Q9 arms stage Q9(300)→Q3(900), original Q3 source row scales and fresh AdamW at300. D matched direct Q3 with same reset at300. **Primary new-to-R4 validation:** WikiText2 validation64 chunks×128 targets, per-chunk loss; historical WikiText test first64 remains secondary reproduction anchor. CPU pin job `6ac9a397fee2c90070180b8d` and CPU nine-code/gradient QA `6ac9a40b095c57808930ccec` both **COMPLETED** before GPUs; model and dataset commits pinned.

**Batch B planned but NOT LAUNCHED:** SmolLM2-360M-Instruct seed271828 and seed424242, 2 scientific GPU jobs **only upon next explicit user prompt after Granite jobs finish**. Smol should use its own calibrated LR/precision recipe and same D/W/N/H codebooks on WikiText2; no premature claims of cross-family effect. R4 study uses already viewed Granite hard/easy orders, not blind novel seeds. **R4 results pending** until exact `FINAL_JSON` checks; preserve null/negative arms and no duplicate or retry GPU runs.

All R3/FineWeb studies and earlier G1/S1/R1/R2 remain archived and complete. **Never misread historical submission sections below as live state.**

---


## R3 COMPLETED — just Q9 outermost reconstruction values recover 96.6% of wide-Q9 benefit (2026-10-10 UTC)

**Verified and archived R3** [HF `6ac99666fee2c9007018011f`](https://huggingface.co/jobs/codeflash85/6ac99666fee2c9007018011f), completed **2026-10-10 02:04:07 UTC**, frozen experiment script SHA `a4abf256c0be9db771e3ec271c1f0a5e6d605816`, prereg SHA `bf89baef162f2a9e948ab702a1708c74b856d3a4`. **16/16 technical/reproduction checks passed and `valid_for_science=true`.** [Full R3 original FINAL_JSON + all 32-doc per-arm losses](results/run_r3_nonuniform_q9_outer_freshdocs_seed1729_2026-10-10.json) · [detailed R3 scientific result](results/run_r3_nonuniform_q9_outer_freshdocs_seed1729_summary.md) · [R3 handoff](R3_ACTIVE_JOB_PLAN.md) · [prereg](research_log/r3_nonuniform_q9_outer_recovery_freshdoc_prereg_2026-10-09.md).

**Design:** 4 matched seed1729 QAT trajectories, all same FineWeb-Edu source QAT train docs/order/teacher/global 1200-step schedule: D direct Q3; W original nine-state Q9(300)→Q3(900) with outputs `(0,±α/4,±α/2,±3α/4,±α)`; N nine-state narrow Q9→Q3 with outputs `(0,±α/6,±α/3,±α/2,±2α/3)`; H **nonuniform hybrid** with `(0,±α/6,±α/3,±α/2,±α)`. All Q9 arms use original **T4 code decision boundaries** and identical clipped-z STE surrogate; N and H differ only in the **two extreme codebook states** ±4, not only two individual weights. At step300 Q9 arms reset source Q3 scales + Adam/GradScaler, then identical original Q3 continuation900. Different output values can still influence subsequent learned master/scale gradients and states.

**Primary data genuinely new to our QAT evaluation:** 32 pinned FineWeb-Edu eval-bucket0 documents strictly after the first340 source rows previously read; actual source row343..1222, unique ID SHA256 saved, 512 target tokens/doc = **16,384** paired tokens, each document's NLL sum/mean available. These docs may have appeared in pretrained SmolLM2's original training; *not* guaranteed independent pretraining test distribution.

| R3 arm | NEW 32-doc FineWeb NLL ↓ | D−arm advantage | OLD 21-doc FineWeb NLL ↓ | WikiText validation NLL ↓ |
|---|---:|---:|---:|---:|
| D direct Q3 | 5.789299 | — | 5.766660 | 6.558239 |
| **W original wide Q9** | **4.885759** | **+0.903540** | **4.996255** | **5.695225** |
| N narrow Q9 original thresholds | 5.813866 | −0.024567 | 5.780473 | 6.593468 |
| **H hybrid restore outer levels** | **4.916794** | **+0.872505** | **5.039273** | **5.730818** |

**Primary contrast:** `N−H=+0.897071` nats; `D−H=+0.872505`, `H−W=+0.031035`. H retains **96.565% of W's D-relative benefit** and recovers **96.656% of the N-to-W advantage**; **32/32 fresh docs** favor H over N and over direct D. W beats H on 23/32 docs but differences are small relative to N−H. All historic old FineWeb and WikiText D/W/N anchors reproduced **exactly**. GPU prep H native dev NLL **5.365745** vs W **5.428179**, N **6.478196**, 29.3% outer-code occupancy for all three T4 modes. AMP effective updates vary D1194/W1192/N1190/H1191; no hidden matching of effective counts.

**Scientific inference:** outermost nine-state **reconstruction magnitudes (±α)**, or their enabled training dynamics, account for most of the wide Q9→Q3 staging benefit **within this fixed custom recipe and one seed**. This is more specific than R1/R2's range-plus-spacing evidence, but not proof that scale gradients, master placements or code-survival mechanisms are solely explained by forward amplitude; full trainable trajectories diverge after output intervention. No broad model-family, generation quality, or original-pretraining-new-doc claim. No saved model checkpoint; full machine-readable per-doc results + pinned source and package versions.

**Infrastructure QA:** first CPU preflight printed correct quantizer/data checks but failed on Python interpreter shutdown (exit134); second finalization-safe CPU check `6ac99425095c57808930c23c` **COMPLETED** before R3 GPU submission. R3 scientific GPU also COMPLETED. All earlier F1–F3, L1, G1, S1, R1, R2 results remain archived. **No new seed or GPU launched in R3 result analysis.** Next suggested (NOT launched): a different model/corpus, or structural measurement of H/W/N scale-gradients and Q3 assignment survival with restorable checkpoints. See [CURRENT_STATE.md](CURRENT_STATE.md) for authoritative status. Older "R3 running" references are historical.

---


## R3 LAUNCHED — nonuniform Q9 outer-level rescue and fresh 32-document evaluation (2026-10-10 UTC)

**R3 single scientific GPU job ACCEPTED:** [HF `6ac99666fee2c9007018011f`](https://huggingface.co/jobs/codeflash85/6ac99666fee2c9007018011f), `a10g-small`, max **2 hours**, initial **SCHEDULING** status. One scientific R3 GPU job only, no seed sweep. [Exact R3 job and archival handoff](R3_ACTIVE_JOB_PLAN.md). [Immutable script](r3_nonuniform_q9_outer_freshdocs_seed1729.py) code commit `a4abf256c0be9db771e3ec271c1f0a5e6d605816`; [frozen prereg](research_log/r3_nonuniform_q9_outer_recovery_freshdoc_prereg_2026-10-09.md) commit `bf89baef162f2a9e948ab702a1708c74b856d3a4`.

**CPU reproducibility and data-selection gate PASSED:** [HF finalization-safe CPU preflight `6ac99425095c57808930c23c`](https://huggingface.co/jobs/codeflash85/6ac99425095c57808930c23c) **COMPLETED** 2026-10-10 01:35:09 UTC, emitted `R3_SAFE_QUANTIZER_PRECHECK_OK`, `R3_SAFE_FRESH_DOC_SELECTION_OK` and `R3_SAFE_ALL_PREFLIGHT_OK`. Checked 9 codes, W/N/H correct level outputs, same integer decisions and STE, post-Q3 switch and 32 real FineWeb docs beyond original QAT scan rows340 (source rows first343, last1222; first 32 qualifying docs ≥516 tokenizer tokens). Initial CPU job `6ac992d4095c57808930c1b0` had passed scientific tests but failed during external dataset-interpreter teardown (`PyGILState_Release`, exit134); preserved this as an infrastructure failure; a clean independent CPU pass preceded GPU submission.

**Frozen R3 arms:** direct Q3 D; original wide Q9 W (levels 0,±1/4,±1/2,±3/4,±1); narrow Q9 N (0,±1/6,±2/6,±3/6,±4/6); new nonuniform hybrid H (0,±1/6,±2/6,±3/6,±1), keeping original T4 integer assignment thresholds for all Q9 variants and original Q3 through steps301–1200. All use same seed1729, SmolLM2 source, FineWeb training order/data and optimizer resets.

**Primary:** *new QAT-heldout* 32 FineWeb documents, 512 tokens each, **per-document loss evidence**, aggregate contrasts N−H, H−W and D−H with paired descriptive per-doc statistics. Historical FineWeb21-doc and WikiText validation used as reproduction/secondary results only. Documents fresh to these QAT experiments, **NOT guaranteed absent from source model pretraining**. All previous R2 (16/16) and G1/S1/R1/F1–F3/L1 studies remain intact. **No R3 outcome available until GPU job completes and `FINAL_JSON` validity checks pass. No additional GPU job/retry authorized.**

---


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


> ## 🤖 AI / researcher handoff — START HERE
>
> New AI/researcher: read **[AI_HANDOFF.md](AI_HANDOFF.md)** before changing an
> experiment or launching compute.

**Authoritative live state:** [CURRENT_STATE.md](CURRENT_STATE.md) · **Copyable next-experiment/handoff plan:** [NEXT_EXPERIMENT_PLAN.md](NEXT_EXPERIMENT_PLAN.md). The long AI handoff and experiment ledger preserve historical statuses, including superseded “current” headings.

**F2/F3 FineWeb-Edu replications RUNNING:** [F2 seed271828](https://huggingface.co/jobs/codeflash85/6ac87415fee2c90070173baf) (pinned `64d171a225ba2781c83148a4235272c91939f9c9`) and [F3 seed424242](https://huggingface.co/jobs/codeflash85/6ac87417095c5780893020c2) (pinned `a20c4cc66ae16026bb90969d1160294b3f5826e9`). Both match F1 training/evaluation recipe, differing only in seed/permutation expected check/provenance. [F2/F3 frozen preregistration](research_log/f2_f3_fineweb_edu_seed271828_424242_replication_prereg_2026-10-09.md). Outcomes pending; L1 WikiText6k also running. The prior **FineWeb-Edu F1 experiment COMPLETED:** D direct Q3 held-out NLL **5.766660** vs S Q9(300)→Q3(900) **4.996255** (gain **+0.770405 nats**). On independent-from-QAT-training WikiText-2 validation, **6.558239 vs 5.695225** (+0.863013 staged). [F1 results](results/run_f1_fineweb_edu_direct_vs_staged_seed1729_summary.md) · [full raw JSON](results/run_f1_fineweb_edu_direct_vs_staged_seed1729_2026-10-09.json). This is one seed and a small public sample of a SmolLM2 pretraining ingredient; pretrained-source overlap unknown. **L1 6000-step WikiText durability remains RUNNING** ([HF job](https://huggingface.co/jobs/codeflash85/6ac86e1f095c578089301e53)); no result yet. [Full active plan](NEXT_JOBS_PLAN.md).

**LATEST (2026-10-09):** [FineWeb F1/F2/F3 aggregate](results/run_f1_f3_fineweb_edu_three_seed_aggregate_summary.md): **3/3 seeds** favor Q9→Q3, mean heldout advantage **+0.763898 nats** (same 21 documents across all seeds). [L1 WikiText6000 result](results/run_l1_6000step_wikitext_durability_seed1729_summary.md): staged advantage **+0.616040 nats at6000**, not eliminated, but narrowing overall. [Six unresolved mechanism controls / outside review](research_log/2026-10-09_external_technical_review_scale_range_sham_controls.md): freeze scale-depth comparison, FineWeb sham Q3 switch and Q9 range match next; proposed only, not GPU launched. [Current state](CURRENT_STATE.md) is authoritative; older “running” prose below describes historical job submission status.

**LIVE STUDY:** [G1/S1 mechanism-control plan](G1_S1_ACTIVE_JOB_PLAN.md): frozen-vs-learned-scale Q3 depth (G1, HF `6ac96a7efee2c9007017ea64`, SHA `059a0bb91aa55fea99ed56b1d0d980eb21fd31ae`) and FineWeb direct-Q3 sham reset (S1, HF `6ac96a82fee2c9007017ea6b`, SHA `4cdab8ee73ddd804e96fac43992f91ab9a54c1b9`). **Submitted; results pending.** Both frozen preregs and pinned scripts linked in active plan. R1 not launched. Older job statuses are historical, use [CURRENT_STATE.md](CURRENT_STATE.md).

**G1/S1 completed:** [G1 frozen-scale depth](results/run_g1_depth_by_scale_freeze_seed1729_summary.md) found deep-vs-shallow gain **+0.620468 nats** with frozen scales vs **+0.609848** with trainable scales; row-scale training is not needed for depth benefit. [S1 FineWeb sham switch](results/run_s1_fineweb_q3_sham_reset_seed1729_summary.md) found direct Q3 reset at300 **5.789199** versus continuous **5.766660**, both far worse than historical Q9 staged **4.996255**. All G1 (6/6) and S1 (9/9) checks passed. [Current status](CURRENT_STATE.md). R1 range-matched Q9 not yet launched.

**R1 ACTIVE:** [Current job plan](R1_ACTIVE_JOB_PLAN.md): Q9 nine-state output range match on FineWeb, 3-arm D vs wide Q9→Q3 vs range-matched Q9→Q3. [HF job](https://huggingface.co/jobs/codeflash85/6ac97c30095c57808930b904) accepted, scientific outcome pending; immutable script `f28f1d4814ab0a61f2b529e2add007ce142972d5`. Pre-run 9-code/gradient/STE CPU checks passed. G1 and S1 completed separately; their findings unchanged. Use [CURRENT_STATE.md](CURRENT_STATE.md) for live status.

**R1 range-matched Q9 COMPLETED (2026-10-10 UTC):** [full result](results/run_r1_q9_range_matched_fineweb_seed1729_summary.md) · [raw](results/run_r1_q9_range_matched_fineweb_seed1729_2026-10-10.json). Q9-wide staged FineWeb NLL **4.996255** vs direct **5.766660**; nine-state Q9 range-matched to Q3's ±2α/3 gave **5.751036**, retaining **2.03%** of wide-Q9 advantage. **12/12 prereg technical checks passed.** The narrow grid also changes spacing, thresholds and saturation (47.83% extreme codes vs29.30% wide): range alone not causally isolated. [Current state](CURRENT_STATE.md) supersedes historical R1 submitted prose. No new GPU jobs launched.

Experiments on whether a pretrained language model can enter ternary weight
space more effectively after adapting on an intermediate discrete grid.

## Current headline

Matched-schedule **Q9→Q3 beats tuned direct Q3 in 3/3 training orders**: mean held-out loss improves by **0.672 nats/token** and PPL by **48.94%**.
The benefit localizes to the **~6.3%** of weights where Q9 and direct choose different Q3 codes; that subset recovers **96.4%** of the full gain across 3/3 orders.
A standardized interior placement for those Q9-selected codes recovers **106.9%** across 3/3 orders; v13 further shows the depth effect saturates around **d≈0.5** on seed 1729.
**New G1-9/G1-10 within-Granite replication (2026-10-08):**
a preregistered equal-budget Gaussian × gridward-pull direct-Q3
factorial showed that **periodically moving FP32 masters 10% toward
their current ternary grid values** improves hard-Q3 held-out loss
on **both** tested Granite training orders at the same fixed
`1e-4` LR and 1,200 optimizer steps:
- Seed **271828**: D **5.72673** → P **5.48971**
  (+0.23702 nats, 21.10% lower perplexity).
- Seed **424242**: D **5.80192** → P **5.52811**
  (+0.27381 nats, 23.95% lower perplexity).
Both direct baselines reproduce historical results exactly.
Gaussian noise alone and its marginal benefit when combined with
pull change sign between the two orders. Sampled per-update
ternary-code transitions fall ~87% under pull, but that is
**not** a proof they cause the improvement. The method has
published prior art (WinQ); this is **same-model, two-order
replication, not cross-model validation**.
[G1-9](results/run_g1_9_granite350m_gaussian_pull_summary.md) ·
[G1-10](results/run_g1_10_granite350m_gaussian_pull_seed424242_summary.md).

**S1-1 cross-model negative (2026-10-09):** applying the same
nine 10%-gridward pulls to **SmolLM2-360M's tuned direct-Q3
training schedule** made the result **worse**, not better:
seed1729 D held-out loss **5.59572** → P **5.84575**
(**+0.25002 nats**, PPL **28.41% higher**).
Both arms had exactly matched 1,200 scheduled steps,
six identical AMP-skipped optimizer updates and all
checks passing; D exactly reproduced historical v9.
Gridward still cut sampled ternary code-flip frequency
by **78.41%**, demonstrating that **fewer code flips
are not a universal guarantee of improvement**.
P was slightly better on validation at steps 300/600,
but fell behind by 900/1200, suggesting possible
premature commitment, *not proving the cause*.
Granite and Smol have different architectures AND
previously selected optimizer/LR schedules.
The three-seed Smol Q9→Q3 **staging advantage**
remains independently intact.
[S1-1 summary](results/run_s1_1_smol360m_gridward_direct_q3_seed1729_summary.md).

**Living current-state report:** [RESEARCH_REPORT.md](RESEARCH_REPORT.md) · **Chronological log:** [SHAREABLE_RESEARCH_REPORT.md](SHAREABLE_RESEARCH_REPORT.md)


**New research-history note (2026-10-08 EDT):** [Q9 assignment discovery and checkpoint-timing plan](research_log/2026-10-08_q9_assignment_discovery_timing.md) records the user's "rough 9, better 3" selection hypothesis, a retrospective decomposition of the v11 disagreement mask into Q9-only versus direct-only code changes, and a gated proposal to measure *when* useful Q3 decisions emerge. This is **documentation and a draft plan, not a completed test or permission to launch GPU jobs**.

**T1 assignment-timing pilot completed (2026-10-08 EDT):** On Smol seed1729, matched 300-step Q9/direct-Q3 projected-code checkpoints reveal gradual emergence of the final disagreement mask: at step100 only **18.61%** of the step300 mask is present; at step200 **51.35%**; at step250 **69.40%** (precision **77.77%**). The complete step300 **6.458%** mask reproduces historical v11 exactly. This is a **retrospective one-seed timing diagnostic**, not evidence that an earlier Q9→Q3 switch preserves final quality. [Summary](results/run_t1_q9_discovery_timing_seed1729_summary.md) · [Raw JSON](results/run_t1_q9_discovery_timing_seed1729_2026-10-08.json) · [Preregistration](research_log/phase_a_q9_timing_seed1729_prereg_2026-10-08.md). Further causal/switch-time GPU jobs have not been launched.

**C1 early-switch experiment completed (2026-10-08 EDT):** A controlled seed1729 trial compared **direct Q3(1200)** test loss **5.595722**, **Q9(250)→Q3(950)** **4.930515**, and **Q9(300)→Q3(900)** **4.900985**. The earlier switch retained **95.75%** of the staged benefit over direct, passed both predeclared exploratory thresholds, and the two historical reference arms reproduced their held-out losses exactly. S250 still lagged S300 by **0.02953 nats**, and this one already-studied order is **not independent confirmation of a universally optimal switch time**. [Full results](results/run_c1_smol360m_q9_switch250_vs300_seed1729_summary.md) · [Raw JSON](results/run_c1_smol360m_q9_switch250_vs300_seed1729_2026-10-08.json) · [Preregistration](research_log/phase_c_q9_switch250_vs300_seed1729_prereg_2026-10-08.md). **No follow-up GPU jobs launched.**

**M1 matched-continuation experiment completed (2026-10-09 UTC):** With exactly the **same 900 Q3 training batches/LR values** after preparation, Q9(250)+Q3(900) held-out NLL **4.950432** versus Q9(300)+Q3(900) **4.900985** (+0.049447 for earlier prep). Both Q3 arms had 897 effective updates and 3 AMP skips. They started with different projected ternary codes on **6.245M (1.985%)** positions; after Q3 training, the shorter-prepared final arm adopted the later-prepared code at **44.16%** of these locations. Their final codes agreed at **60.0%** on the selected locations and **93.86%** overall. Reproduction checks passed. **This is an unequal-total-budget, one-seed exploratory preparation-state test**; C1 remains the separate equal-total-budget comparison. [M1 detailed result](results/run_m1_equal_q3_continuation_seed1729_summary.md) · [raw JSON](results/run_m1_equal_q3_continuation_seed1729_2026-10-09.json) · [frozen protocol](research_log/m1_equal_q3_continuation_seed1729_prereg_2026-10-08.md). No new GPU jobs launched.

## Current interpretation

Q9 does not simply preserve more information or create a better ternary model at
the switch. It discovers a small, specific set of useful position/code
assignments; those assignments need enough interior margin to survive later Q3
optimization. Exact Q9 continuous values are unnecessary, and generic
prototype snapping does not help direct Q3.

The mechanism phase on SmolLM2-360M / WikiText-2 has reached its stop condition.
**Granite-4.0-350M gives a mixed three-order cross-family result.** Full Q9→Q3
beats direct on **2/3** orders, not 3/3. However, the ~6% disagreement geometry
persists, true-mask **d=0.5 beats full S on 3/3**, and matched-random d=0.5 is
harmful on **3/3**. The reusable position/code signal appears more stable than
the full staged trajectory itself.
A direct **Smol v10–v13 LR schedule transfer** on two selected Granite
orders made **both direct and full staged QAT worse in absolute held-out loss**,
expanding the disagreement masks from ~6–7% to **29–33%**.
On negative order 271828, S fell 0.360 nats behind D; on historically positive
order 424242, S was essentially tied with D, while **M-d50 still improved D by
0.0285 nats**. This is *two-order schedule sensitivity*, not a new
independent confirmation or proof that useful Q9 assignments have vanished:
[G1-7b](results/run_g1_7b_granite350m_v10schedule_summary.md) /
[G1-8](results/run_g1_8_granite350m_v10schedule_summary.md).

## Evidence hierarchy

- **Replicated across 3/3 orders:** matched-schedule staging advantage (v10),
  disagreement-mask localization (v11), code-choice/interior-placement result
  and matched-random failure (v12).
- **One-order supporting mechanism:** master-weight carryover (v6), v13 depth
  saturation and firmness controls.
- **Important negative result:** generic FP32 warm-up does not reproduce Q9,
  but it is **not worse than direct in every order**; order 271828 gives FP32 a
  small 0.0292-nat improvement over direct.
- **Cross-family Granite:** full staging is mixed (2/3 positive); the worse-entry
  signature is 3/3, M-d50 beats full S 3/3, and matched-random is harmful 3/3.
  Seed 271828 is a real negative and remains part of the conclusion.
- **Scope:** WikiText-2 so far; free-running generation remains poor, and
  longer-run asymptotics are unresolved.

## Repository map

- [RESEARCH_REPORT.md](RESEARCH_REPORT.md) — **living current state**; edit in
  place after each experiment
- [SHAREABLE_RESEARCH_REPORT.md](SHAREABLE_RESEARCH_REPORT.md) — append-only
  chronological narrative
- [AI_HANDOFF.md](AI_HANDOFF.md) — operational handoff, current protocol and
  claim boundaries
- [EXPERIMENT.md](EXPERIMENT.md) — preregistrations and chronological
  experiment record
- `results/` — raw JSON and per-run summaries
- `replications/` — replication scripts and aggregate summaries
- `smollm2_v*.py` — experiment scripts

## Research workflow

`preregister → pin code → launch job → record job ID → save raw JSON → write run summary → update aggregates → update RESEARCH_REPORT.md → update AI_HANDOFF.md / README as needed`

Before making a replicated conclusion, inspect **every seed/order individually**,
not only the mean, and say plainly when a result is mixed.
