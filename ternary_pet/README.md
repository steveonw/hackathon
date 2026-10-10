# Ternary Pet Experiments

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
