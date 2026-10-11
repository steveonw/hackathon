# R7 stage-localization diagnostic — archival findings and controlled next experiment

**2026-10-10. Status: archival stage metrics analyzed; stage-specific capability loss NOT proven. No new training, GPUs, or CPU checkpoint jobs launched.**

## Existing matched stage metrics from R6

From immutable archived R6 jobs:
- Granite seed170141 ([run](../results/run_r6_granite_seed170141_imdb_2026-10-10.json)): H native 9-level at step300 dev loss **5.46626337369283** and teacher top1 agreement **0.2819010416666667**, immediately projected Q3 dev loss **5.925915420055389**, agreement **0.20345052083333334**. Projection shock **+0.459652046362559** loss, **−0.07845052083333336** top1 agreement.
- Smol seed190027 ([run](../results/run_r6_smol_seed190027_imdb_2026-10-10.json)): native 9-level dev loss **5.043259421984355**, top1 agreement **0.3421223958333333**, immediately projected Q3 loss **6.396277328332265**, top1 agreement **0.22005208333333334**. Shock **+1.35301790634791** loss, **−0.1220703125** agreement.
- Comparisons are **within family on the same 3,072 training-development target tokens** at the exact Q9→Q3 transition, not cross-family quality comparisons. Metrics are teacher-forced loss/teacher-top1 and cannot establish instruction ability.
- R7 final H scored **0/30 permissive** against base **29/30 Granite**, **24/30 Smol** on distinct frozen capability tasks ([scoring audit](../results/r7_posthoc_scoring_repetition_audit_2026-10-10.md)). R7 tested **final Q3 only**.
- R6 saved **final inference-only Q3 H checkpoints**; no Q9@300 or projected-Q3@300 retained checkpoints were identified in the archived R6 output. Historical R5 checkpoint failures are irrelevant to stage identification.
 
## Localization hypothesis and minimal controlled test

**H1 (possible):** a material fraction of capability loss is introduced at the Q9→Q3 projection. **Alternative:** Q9 itself already loses ability; or later Q3 optimization further damages ability. The measured projection loss shock motivates H1 but does NOT prove capability localization.

Before spending training compute, perform a static source audit of Q9 and Q3 map/weight scales and assess whether **CPU forward-only** Q9 reconstruction at step300 is possible (it is *not* possible from current final Q3 inference snapshots alone). Do not manufacture Q9 checkpoint data. The first scientifically decisive comparison would be *same trained H master weights and code/scales at step300*, evaluated in both native Q9 and immediately projected Q3 **before any optimizer step**, plus the original base and step1200 Q3; identical prompts, chat templates, greedy decoding and evaluation set. Also save **pre-Q9@step1** or near initialization state if affordable, to measure early Q9 loss.

Evaluation stages:
1. **base** high precision, pinned
2. **H Q9 at step300**, state frozen
3. **H projected to Q3 at step300 without any optimization**, same master weights/scales
4. **H Q3 at step1200** final
5. Optional near-initial Q9 stage and additional timepoints only if predeclared and affordable.

Per stage: strict and corrected permissive arithmetic, instruction following, self-contained extraction; repetition via unique-bigram ratio; matched teacher-forced NLL and teacher top1. Freeze a **new held-out task set** before running, not reusing R7 scored prompts for development. Report matched changes Base→Q9 (initial Q9 degradation), Q9→Projected Q3 (quantization shock), Projected Q3→Final (Q3 training effect). Preserve matched seeds, data order and optimizer recipe within each family; stage snapshots are diagnostic only and don't require optimizer restoration. Log checksums, stage state, numerical stability, expected code alphabet, and state provenance.

## Resource gate

No stored step300 inference snapshots were located in original R6 JSON, and final ternary checkpoints alone cannot reconstruct earlier master weights. **A fresh matched training run is necessary for exact stage-localization unless historical job artifacts provide independent intermediate checkpoints.** It would consume additional GPU compute; do not claim it is CPU-only or zero-cost. Propose **one pilot model/seed** (Smol priority given larger measured Q3 projection shock), preserve the established seed/training recipe and add only diagnostic stage captures and safeguards, then decide whether second-family confirmation merits budget. Do not launch until GPU scope/budget is agreed. A cheaper partial pilot through step300 can establish Base→Q9 and Q9→Projected Q3, but cannot measure projected→Final without completing steps301–1200.

Scientific caveat: changing the training program to capture states may affect memory/time; pin original R6 source and verify measurement and codebook parity. Numerical scale and optimizer effects after projection may be causal contributors, not just the 3-code representation.
