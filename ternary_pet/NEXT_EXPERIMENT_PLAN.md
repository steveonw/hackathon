# Ternary Pet — next experiment plan and handoff contract

**Prepared:** 2026-10-08, US Eastern (jobs may be dated 2026-10-09 UTC).  
**Status:** Active working plan, with **one bounded next experiment** specified. This is a persistent plan for handoff; an experiment's execution status is tracked in [CURRENT_STATE.md](CURRENT_STATE.md) and [AI_HANDOFF.md](AI_HANDOFF.md).  
**Repo:** `steveonw/hackathon`, project `ternary_pet/`.  
**Context:** [research discussion](research_log/2026-10-08_q9_assignment_discovery_timing.md), [T1 timing](results/run_t1_q9_discovery_timing_seed1729_summary.md), [C1 shorter-switch](results/run_c1_smol360m_q9_switch250_vs300_seed1729_summary.md).

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
