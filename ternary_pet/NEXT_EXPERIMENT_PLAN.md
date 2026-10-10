# Ternary Pet — next experiment plan and handoff contract

**Prepared:** 2026-10-08, US Eastern (jobs may be dated 2026-10-09 UTC).  
**Status:** M1 finished and archived. The M1 method below is an **unchanged historical plan**, followed by still-unrun research priorities. The authoritative execution state is [CURRENT_STATE.md](CURRENT_STATE.md).  
**Repo:** `steveonw/hackathon`, project `ternary_pet/`.  
**Context:** [research discussion](research_log/2026-10-08_q9_assignment_discovery_timing.md), [T1 timing](results/run_t1_q9_discovery_timing_seed1729_summary.md), [C1 shorter-switch](results/run_c1_smol360m_q9_switch250_vs300_seed1729_summary.md).

**CURRENT NEW JOBS (2026-10-09 UTC):** The post-M1 research priorities were activated by user request. **L1** long-horizon 6000-step WikiText paired job [6ac86e1f095c578089301e53](https://huggingface.co/jobs/codeflash85/6ac86e1f095c578089301e53) and **F1** FineWeb-Edu pretraining-style corpus paired job [6ac86eb0fee2c900701738dd](https://huggingface.co/jobs/codeflash85/6ac86eb0fee2c900701738dd) are both submitted/running. Full frozen protocols, model/data revision pins and handoff: [NEXT_JOBS_PLAN.md](NEXT_JOBS_PLAN.md) and [CURRENT_STATE.md](CURRENT_STATE.md). The original M1 plan below remains a historical record; do not infer these two new jobs have already yielded results.

**F1 OUTCOME UPDATE, 2026-10-09 UTC:** [F1 FineWeb-Edu heldout results](results/run_f1_fineweb_edu_direct_vs_staged_seed1729_summary.md) now **COMPLETED**, NLL D 5.766660 vs Q9→Q3 4.996255 (+0.770405), 5/5 validity checks passed; secondary WikiText2 validation D 6.558239 vs S 5.695225 (+0.863013). **L1 6000-step WikiText remains RUNNING** at last check. Use [NEXT_JOBS_PLAN.md](NEXT_JOBS_PLAN.md) and [CURRENT_STATE.md](CURRENT_STATE.md) for authoritative statuses; older plan text below remains historical.

## M1 execution record — COMPLETED (plan preserved below)

**Finished:** 2026-10-09 04:15:27 UTC on HF [codeflash85/6ac86712fee2c900701734bb](https://huggingface.co/jobs/codeflash85/6ac86712fee2c900701734bb); A10G-small, original pinned script SHA `cad64c009209a924be89b523e6a1e184a4f4137b`. Frozen prereg [M1 protocol](research_log/m1_equal_q3_continuation_seed1729_prereg_2026-10-08.md); original CPU static check `6ac86709fee2c900701734b2` passed.

**Completed data:** [Full raw JSON](results/run_m1_equal_q3_continuation_seed1729_2026-10-09.json) · [M1 detailed findings](results/run_m1_equal_q3_continuation_seed1729_summary.md). Seven of seven numerical/design checks passed. Same 900 Q3 batches/LR values produced **P250+900 loss 4.950432**, **P300+900 loss 4.900985** (**+0.049447** early-prep disadvantage). The late prep starts differed from early prep on **6,245,131** ternary codes. The earlier arm's final Q3 code matched the late-prep code on **44.16%** of those positions; the final two arms agreed on **60.00%** there. This is partial code adoption and incomplete convergence; no causal assignment-importance metric.

**Important:** The subsequent research phases in section 3 below remain **proposals only**, not authorized or launched experiments. M1 is finished; do not submit duplicate. The next recommended decision is an independently preregistered longer-horizon and fresh-evaluation study, with equal-cost recipe controls, or confirmatory new orders. See [CURRENT_STATE.md](CURRENT_STATE.md) for authoritative status. The pre-M1 reasoning below is retained as its historical frozen plan and should not be interpreted as pending work.

---

## 0. Read this first: the precise scientific claim

Staged Q9→Q3 training offers a **finite-budget, recipe-specific improvement in ternary QAT** on SmolLM2-360M/WikiText-2 relative to tuned direct Q3. Existing interventions show that selected Q3 **positions, code identity, and interior placement** matter. These experiments do **not** yet establish a usable low-bit language model, asymptotically superior basin, cross-model transfer, or a broadly deployable algorithm.

In C1, step250 Q9 followed by **950 Q3** steps retained **95.75%** of the step300 Q9 + **900 Q3** advantage over direct Q3, with held-out NLLs D **5.595722**, S250 **4.930515**, S300 **4.900985**. C1 is exploratory, on one familiar order (1729) and a repeatedly examined held-out slice. T1's **69.40%** step250 mask recall is descriptive and **is not** a fraction of important assignments, nor was that mask directly transferred in C1.

**Critical correction from independent AI critiques:** The C1 result cannot isolate shorter Q9 preparation from **50 additional Q3 steps**, nor can its aggregate outputs tell whether Q3 rediscovers later Q9 assignments. Do not infer keystone/Pareto assignments or code rediscovery yet. It would be false to claim T1+C1 causally proved the final step300 code mask unnecessary.

## 1. Priority experiment M1: identical Q3 continuation after 250 vs 300 Q9 preparation

### Question

If Q9 preparation is ended at 250 versus 300, **and both resulting FP32 master states are given exactly the same 900 Q3 update opportunities, same ordered training chunks and same LR-by-continuation-step curve**, is the step250 preparation nearly as effective? What happens to the individual ternary assignments by the end?

This answers a **preparation-state quality** question, different from C1's equal-total-budget **recipe** comparison.

### Frozen setup / arms

- Model: `HuggingFaceTB/SmolLM2-360M-Instruct`, seed/order `1729`, BF16-rounded original source, WikiText-2 128-token blocks, quantizer targeting the same linears except `lm_head`, frozen other parameters.
- Objective: 35% CE + 65% fixed-teacher KL, FP32 master weights, rowwise learnable scales, FP16 CUDA autocast+GradScaler, AdamW betas (0.9,0.95), 0 weight decay, clip norm 1.0.
- Q9 preparation uses the established *global* v10 schedule for each of steps 1..250 (and 1..300 respectively); no search or retuning.
- Save **exact FP32 master tensor states** and projected ternary codes at Q9 checkpoints t250 and t300 from a **single continuous 300-step Q9 preparation trajectory**. Clone, never mutate shared dictionaries on transfer.
- Project both saved states into **original initial-source Q3 row scales**, preserving their FP32 master values and resetting AdamW+GradScaler identically as in v10. Run **900 Q3 continuation opportunities from each** with the **same train chunks originally indexed 301..1200** and **the same matched_lr(global_step0=300..1199)**. Thus both Q3 phases see byte-identical example order and LR values, even though the 250 arm never saw training batches 251..300 during Q9 preparation.
- **P250+900:** Q9(250), skip Q9(251..300), then common 900 Q3 batches. **Total=1150 scheduled opportunities.**
- **P300+900:** Q9(300), then same 900 Q3 batches. **Total=1200 scheduled opportunities.**
- This deliberately **does not match total training compute or source-example exposure** between the arms; it isolates the *Q3 continuation* given the differently prepared states. It does not fully isolate optimizer reset timing or the extra Q9 compute from state quality, and cannot show equal-compute superiority. The already archived C1 supplies the **separate equal-total-budget** comparison.
- Keep the current held-out 8192-token WikiText slice for descriptive comparability, and **do not use it to select any new settings**. Fixed 24 train-split diagnostic chunks for continuation time course; each final model scored once. The data are previously viewed, so all seed1729 conclusions remain exploratory.

### Must-save diagnostics

- Identical t250 and t300 master/checkpoint trajectories to historical v10/v11/T1 seed1729 where comparable. Verify Q9 native Q3-projected t300 changed fraction **0.0488410886**; t250 **0.0438033962**, each tolerance at most 0.002 absolute. Verify original Q3 code and quantizer scale restoration at both starts.
- Log both start-of-Q3 projected code sets and **both final Q3 code sets** (in-job tensors, not necessarily uploaded huge binary snapshots). Compute in-job exact position-level counts on all 314,572,800 target weights.
- Causal-relevant descriptive questions: For positions where t250 projected Q3 code disagrees with t300 Q9-chosen projected Q3 code, what fraction end in t300 code after P250's Q3 continuation (code-rediscovery)? What fraction of P300 ending codes match P250? Report this also stratified by the D-vs-Q9 step300 disagreement reference **if an exact matched D300 mask is available**; otherwise **do not invent D masks** or claim masks are equivalent.
- Capture the final held-out loss/PPL/top1/KL for both arms, their loss difference, startup Q3 validation, validation checkpoints at continuation steps 0/300/600/900, gradient AMP skips, native Q9 movement and final code changes relative to source, with code checks.
- Don't retain both full 314M FP32 state copies unnecessarily; capture t250/t300 masters on CPU, instantiate Q3 continuation arms sequentially and clean up GPU memory. Full snapshots can remain in memory until diagnostics and may be reduced to sufficient per-layer counters in raw final JSON. Do not post massive tensors or pretrained weights to GitHub.

### Decision logic, before observing results

- Primary: **`L(P250+900) − L(P300+900)`** on the fixed 8192-token held-out evaluation; report raw difference without declaring success based on a post-hoc cutoff.
- Secondary: relative loss gain vs **historical direct Q3 D1200=5.595722** (historical comparison only, not paired equal-budget arm), PPL, final code agreement and rediscovery fraction.
- If `L(P250+900)` worsens substantially versus P300+900, C1's 50 additional Q3 steps probably compensate for **some** shorter-preparation disadvantage; the experiment still does not attribute the entire gap to one source.
- If they remain close, extra 50 Q3 steps are **not necessary** for comparable endpoints under this matched-Q3 setup, though total Q9 budgets are unequal.
- If P250 final codes converge toward Q9 t300 codes, Q3 rediscovery becomes plausible. If performance is similar without such convergence, multiple good code configurations or unmeasured geometry may be plausible. **Neither observational relationship independently proves causal importance.**
- T300 reproduction and historical S300 held-out loss ~**4.90098537** should be verified with a prespecified abs tolerance of **0.06 nats**. A larger mismatch is a **technical discrepancy** to investigate and archive, not a discovery.
- One A10G-small Hugging Face job, maximum 90 minutes, seed1729 pilot only. Do not start follow-up seeds automatically. Pin immutable Git commit and log job ID.

### Execution and archival checklist

1. Commit this plan and a separate final preregistration **before** GPU submission. Do not tune based on observed final outcomes.
2. Implement a dedicated M1 script reusing v10/v11 optimizer, quantizer, loader and evaluation code. Run CPU static syntax checks, confirm no test leakage in selection logic, no accidental resume from historical CPU/GPU state, no change in teacher/objective.
3. Launch **one** code-pinned `a10g-small` job in Hugging Face namespace `codeflash85`; log link/ID/timeout to `CURRENT_STATE.md` and `AI_HANDOFF.md`.
4. At completion verify stages, skipped opportunities, code-movement and reproducibility assertions. Save **unmodified terminal FINAL_JSON scientific fields** plus provenance, and a summary, under `ternary_pet/results/`.
5. Update `CURRENT_STATE.md`, `AI_HANDOFF.md`, `EXPERIMENT.md`, `RESEARCH_REPORT.md`, README. Preserve all negative/technical outcomes. Do not launch extra trials from the report.

## 2. The competing hypotheses to preserve for handoff

- **H1: Early high-value assignments.** Useful Q3 changes appear before other changes, so more position-overlap at t300 need not correspond to proportionate final quality. **Untested as causal attribution.**
- **H2: Q3 rediscovery/repair.** The Q3 continuation can discover some Q3 code choices not yet present at Q9 step250. Requires final code-position comparison, not aggregate counts.
- **H3: Multiple sufficiently good code configurations.** Similar final losses may arise from nonidentical final ternary configurations. Requires final position-level comparison and eventually controlled interventions.
- **H4: Extra Q3 compensation.** C1's 50 extra Q3 opportunities mask a less-effective shorter prep. M1 matches Q3 continuation length/stream to probe this, with unequal total training steps openly reported.
- **H5: Long-run head start only.** Direct Q3 may catch up at higher budgets. Prior v5 showed a narrower but nonzero effect on one schedule/order; asymptotics unknown.
- **H6: Nonportable/custom-baseline artifact.** Q9 might work under one quantizer, frozen module set, dataset or training scale but not standard recipes/other families. Granite's full staging is mixed and Smol gridward transfers negatively.

## 3. Longer-term research order (not automatically authorized)

**Next priority A — confirmation on fresh orders and evaluation:** Freeze the relevant direct/Q9 recipe. Confirm on independent train shuffles (`271828`, `424242` can replicate existing dataset conditions but are not independent datasets). Choose a genuinely new, previously unexamined evaluation dataset and keep it out of tuning. Do not claim fixed prior WikiText scores provide new independent test evidence.

**Next priority B — durability/head start:** Hold quantizer/model/data/objective fixed and run paired direct/staged arms over several **common training horizon** checkpoints, beyond the existing 1200-step budget. Do not change model, baseline quantizer, trainable modules, objective and time horizon simultaneously. Both arms must get fresh additional training data or explicitly disclose data reuse. A single longer endpoint cannot establish asymptotic convergence; show loss curves and extrapolation uncertainty.

**Next priority C — assignment mechanism/cheap Q9 teacher:** If quality/replication holds, test Q9-only vs direct-only causal masks at identical within-bin depth; compare t250/t300 code prototypes under identical Q3 continuation, matched-random controls; only then attempt low-cost predicted Q3 assignments from features measured without look-ahead. Treat feature fitting on established seeds/test as exploratory.

**Next priority D — stronger external QAT baseline:** At a frozen longer budget, introduce one standard ternary baseline modification at a time (e.g. scaling rule, norms trainable) and compare fairly, retaining both internal and published-method context.

## 4. Operational guardrails

- Existing completed jobs **T1 `6ac85534fee2c90070172a41`** and **C1 `6ac8597afee2c90070172c79`** must not be repeated without scientific reason.
- Always record frozen protocol → pinned script SHA → exact HF job/seed/environment → raw JSON → diagnostic checks → negative findings → summarized interpretation. Keep public and internal state labels aligned.
- Old sections in the append-only `AI_HANDOFF.md` labeled “current” are historical; use [CURRENT_STATE.md](CURRENT_STATE.md) for the **sole authoritative front page**.
- Never silently launch extra expensive arms or seeds; explicit scope per job. If an HF job remains running after an assistant response, capture its ID and status and explain that results are pending.

## Latest completed studies and external mechanism-review follow-up (2026-10-09 UTC)

**All F1/F2/F3 FineWeb seeds and L1 WikiText6000 job have COMPLETED**, all final scientific validity flags passed. No research jobs currently active in the last verified HF listing. **FineWeb aggregate:** Q9→Q3 beats direct Q3 in **3/3 orders**, mean held-out advantage **+0.763898 nats/token** (range +0.742353..+0.778936), WikiText validation mean +0.801917. Identical 21 FineWeb evaluation documents across orders, *not independent validation samples*. [3-seed aggregate](results/run_f1_f3_fineweb_edu_three_seed_aggregate_summary.md) · [F2 raw](results/run_f2_fineweb_edu_direct_vs_staged_seed271828_2026-10-09.json) · [F3 raw](results/run_f3_fineweb_edu_direct_vs_staged_seed424242_2026-10-09.json).

**L1 six-thousand-step:** [result summary](results/run_l1_6000step_wikitext_durability_seed1729_summary.md) · [raw](results/run_l1_6000step_wikitext_durability_seed1729_2026-10-09.json). On fixed WikiText validation, staged advantage declines **+0.716087 @1200 → +0.616040 nats/token @6000** (nonmonotonic), so direct Q3 hasn't caught up by6000. Reproduced both step1200 historical tests exactly. This is one long training order, not a convergence proof.

**Separate independent review of unresolved mechanism controls:** [frozen-scale depth / Q9 range match / FineWeb sham switch / margin-matched positions / common-scale code comparisons / document-linked evidence](research_log/2026-10-09_external_technical_review_scale_range_sham_controls.md). Reviewing `smollm2_v12_code_identity_position.py` confirms scale-gradient confounding is real; v13's nearly-equal fixed/learned alpha *measurement* of code survival does not eliminate a scale-*training-gradient* mechanism. Next recommended controls **G1 frozen-vs-learned-scale depth** and **S1 FineWeb direct-Q3 sham step300 reset**, followed by range-matched Q9. No control GPU jobs launched as part of this review. [Authoritative current state](CURRENT_STATE.md) takes precedence over historical “running” sections.

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

## R4 COMPLETE — Granite + Smol 2×2 hard/easy hybrid Q9 replication (2026-10-10 UTC)

**ALL FOUR GPU jobs completed and archived; `valid_for_science=true` for each; 48/48 reported checks passed.** [Authoritative four-job result and paired controls](results/run_r4_granite_smol_easy_hard_four_job_aggregate_2026-10-10.md) · [Permanent 2+2 HF IDs/protocol](R4_GRANITE_SMOL_EASY_HARD_JOBS.md) · [Frozen R4 prereg](research_log/r4_2026-10-09_granite_smol_easy_hard_two_batch_prereg.md).

**Primary aligned WikiText2 validation loss, lower better, first64 129-token windows, 8192 target tokens:**

| Family/seed | D direct Q3 | W original wide Q9→Q3 | N narrow Q9→Q3 | **H hybrid outer-restored Q9→Q3** |
|---|---:|---:|---:|---:|
| Granite-4.0-350M **271828 known hard** | 5.438636 | 5.577830 | 5.416681 | **5.370064** |
| Granite-4.0-350M **424242 known easy** | 5.420453 | 5.349609 | 5.396033 | **5.295450** |
| SmolLM2-360M 271828 | 5.293719 | 4.567472 | 5.259229 | **4.480920** |
| SmolLM2-360M 424242 | 5.265099 | 4.553739 | 5.273762 | **4.520216** |

**Major result:** In all four model/seed combinations H beats D/W/N. On **historically negative Granite seed271828**, W is **0.139194 NLL worse than D**, but H is **0.068572 better than D**, a reversal. H vs W improves by **0.207766 Granite hard**, **0.054158 Granite easy**, **0.086552 Smol271828**, **0.033523 Smol424242**. W original wide remains negative on the Granite hard seed, and this is preserved explicitly. On Smol both H and W are much better than D; narrow N is close to D. Historical WikiText *test* D and W anchors reproduced in every run, all four jobs pass 12/12 checks. Per-chunk paired output saved as complete raw JSON, not just averages.

**HF job provenance and raw scientific evidence:** Granite hard [HF 6ac9a43d095c57808930ccfe](https://huggingface.co/jobs/codeflash85/6ac9a43d095c57808930ccfe), [raw](results/run_r4_granite_hard_seed271828_2026-10-10.json); Granite easy [HF 6ac9a443095c57808930cd03](https://huggingface.co/jobs/codeflash85/6ac9a443095c57808930cd03), [raw](results/run_r4_granite_easy_seed424242_2026-10-10.json); Smol271828 [HF 6ac9a5c2fee2c90070180ce3](https://huggingface.co/jobs/codeflash85/6ac9a5c2fee2c90070180ce3), [raw](results/run_r4_smol_seed271828_2026-10-10.json); Smol424242 [HF 6ac9a5c9095c57808930cdde](https://huggingface.co/jobs/codeflash85/6ac9a5c9095c57808930cdde), [raw](results/run_r4_smol_seed424242_2026-10-10.json). Executables pinned at Git SHA `33dfae38bd016946ad5d37bd633fdf02818fbacd` Granite and `65af3057b248f38a6d46ad8b7cd1bc1c52d2e654` Smol. All data/model revisions pinned and CPU STE/nine-code QA passed.

**Scientific limits:** Two **historically selected** seeds per family, no blind random sample; primary WikiText2 validation is contiguous chunks **not independent documents**, dataset/source pretraining overlap unknown; family protocols differ materially (Granite BF16 constant1e-4 and direct step300 reset vs Smol FP16 warmup/cosine and continuous direct Adam), so **do not pool raw losses or infer pure architectural comparison**. H differs from N only in the two extreme *codebook levels* (±4 output restored to ±α), but many weights use them; downstream learned scale gradients, masters and ternary assignment evolution can also change. No final model checkpoints and no production-generation inference. Earlier Smol FineWeb R3 32-doc sample and G1/S1/R1/R2 remain archived, but R4 does not use a second, independent corpus. **All GPUs terminal; none newly launched during archival.** Next research could validate on previously untouched external-document evaluation and independently selected seeds with checkpoint retention, **not authorized/launched yet**. Old 'R4 pending' prose elsewhere is historical. [CURRENT_STATE.md](CURRENT_STATE.md) is the current reference.

---

