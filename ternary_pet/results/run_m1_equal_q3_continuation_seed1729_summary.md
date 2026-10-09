# M1 — Same Q3 continuation after Q9 preparation at 250 versus 300

**Status:** COMPLETED; Hugging Face success, `valid_for_science=true` (**7/7 protocol checks passed**). **Scientific status:** exploratory, single previously observed seed/order and WikiText-2 heldout.  
**GPU job:** [6ac86712fee2c900701734bb](https://huggingface.co/jobs/codeflash85/6ac86712fee2c900701734bb) — A10G-small, completed **2026-10-09 04:15:27 UTC**.  
**Pinned code:** [`cad64c009209a924be89b523e6a1e184a4f4137b`](https://github.com/steveonw/hackathon/blob/cad64c009209a924be89b523e6a1e184a4f4137b/ternary_pet/m1_smol360m_equal_q3_continuation_seed1729.py).  
**Preregistration:** [M1 matched Q3-continuation protocol](../research_log/m1_equal_q3_continuation_seed1729_prereg_2026-10-08.md).  
**Full machine-readable results:** [M1 FINAL_JSON with provenance](run_m1_equal_q3_continuation_seed1729_2026-10-09.json).  
**Reasoning/handoff:** [NEXT_EXPERIMENT_PLAN](../NEXT_EXPERIMENT_PLAN.md) and [CURRENT_STATE](../CURRENT_STATE.md).

## 1. What was actually controlled

One Q9 preparation run for **300 global training opportunities** produced independent FP32-master snapshots at steps **250 and 300**. Both saved states were projected into **original frozen-source Q3 row scales**, with fresh AdamW and GradScaler, then given the **exact same 900 Q3 continuation batches (original train chunks 301–1200)** and **exact same 900 LR values from the v10 global 1200-step schedule**. Both started fresh optimizer state at the moment Q3 began. This eliminates the differing Q3 continuation length/batch/LR confound within **M1**.

**Not controlled:** total compute and total training exposure are unequal by design, with **1150** scheduled steps in the early arm and **1200** in the late arm. This is not a fair equal-budget recipe comparison; **C1** is that experiment. The test remains exploratory and uses the same viewed evaluation set.

## 2. Final held-out test

| Arm | Q9 preparation | Identical Q3 continuation | Total scheduled steps | Q3 actual Adam updates | Held-out NLL ↓ | PPL ↓ | Teacher top1 | Teacher KL ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| P250_900 | 250 | 900 | 1150 | 897 /900 (3 AMP skips) | **4.950432** | **141.236** | 37.07% | 2.000353 |
| P300_900 | 300 | 900 | 1200 | 897 /900 (3 AMP skips) | **4.900985** | **134.422** | 36.91% | 1.969979 |

**Main finding:** `L(P250_900) - L(P300_900) = +0.049446654 nats/token`. **Longer Q9 preparation produces a better final ternary model when Q3 continuation is identical in this run.** The historical 300-step recipe reproduces its previous held-out loss `4.9009853675961494` exactly. The early arm remains substantially better than historical D1200 direct-Q3 loss `5.595722187310457` (different total opportunities; contextual comparison only); measured improvement `0.645290` nats, or **92.88%** of the historical Q9(300)+Q3(900) gain vs D. Do not call this an equal-compute relative gain.

**Contrast to C1 without pooling them:** C1 **250 Q9+950 Q3** held-out loss `4.930515`; M1 **250 Q9+900 Q3 with different first Q3 LR/batch index** ended `4.950432`, a difference of `0.019917` nats. This difference is compatible with beneficial additional Q3 training, **but** the Q3 phase in C1 used a different starting LR and first 50 training chunks, so it cannot be attributed exclusively to 50 extra Q3 steps. No same-schedule length-only effect has been isolated across C1/M1.

## 3. The previously missing code-destination analysis

Exact position-level counts span **314,572,800** target weights. These are **unweighted per-position** counts, not gradient-weighted causal importance.

| Metric | Full-model count | Interpretation |
|---|---:|---|
| Different Q3-projected code at Q9 steps 250 vs300 | 6,245,131 (1.985%) | Last 50 Q9 steps changed the specific ternary label |
| Changed from source by Q9 at step250 | 13,779,357 (4.380%) | Step250 snapshot's original-Q3-scale code displacement |
| Changed from source by Q9 at step300 | 15,364,078 (4.884%) | Step300 snapshot's code displacement |
| New/added source code changes in steps251–300 | 3,914,292 | Step250 source code unchanged, step300 changed |
| Reverted changes toward source in steps251–300 | 2,329,571 | Step250 changed, step300 restored source |
| Both changed but different code | 1,268 | Small residual code category |
| Final P250 and P300 same Q3 code overall | 93.858% | Complement **6.142%** final code disagreement |

**Conditional specifically on the 6,245,131 prep-disagreement positions:**

- After Q3 continuation, **P250 final matched the Q9 step300 code at 44.16%** of these positions; it retained its own step250 code at **55.80%**.
- P300 final retained its own step300 code at **59.30%**.
- Both final endpoints agreed with each other at **60.00%** of these selected positions (not 100%).

This is direct **descriptive evidence that Q3-from-step250 can adopt some code identities present at later Q9 step300**. But it also demonstrates **incomplete convergence** and divergent final choices. Simple `F250 == S300` is a code coincidence metric; it does not by itself prove that the assignment was learned from the same underlying signal or that the code is performance-critical. Furthermore, P300 itself kept its step300 choice at only 59.30% of these disputed locations; **the later-Q9 projected snapshot is not a frozen target or an oracle of final correctness**.

The overall final Q3 code disagreement is **6.142%** despite only **1.985%** differing immediately at the two prep snapshots, showing the different starting continuous states can yield other positional differences after continuation. This does not independently establish distinct loss basins.

## 4. Validation trajectories and reproducibility

| Common continuation step | P250 dev NLL | P300 dev NLL |
|---:|---:|---:|
| 0 | 6.542107 | 6.542292 |
| 300 | 5.259507 | 5.204416 |
| 600 | 4.787009 | 4.735228 |
| 900 | 4.586918 | 4.521716 |

- Q9 native step250 dev **5.390829504**, step300 **5.182718833**, both reproduced historical values and within frozen 0.035 tolerance.
- Projected Q3 source-code change **4.38034%** at step250, **4.88411%** at step300, matched established snapshots.
- Fixed original Q3 scales at start yielded almost equal immediate train-split Q3 dev NLL **6.542107** vs **6.542292**, while the final validation and held-out endpoints favor the later prep.
- Preparation actual updates t250 243 (7 AMP skips), t300 293 (7 skips), **50 additional effective Q9 updates**. Both Q3 arms had **897 successful Q3 updates** with **3 skips**.
- All seven frozen checks passed and `valid_for_science=true` after exact t300 held-out baseline reproduction.

## 5. Conclusions, limitations, next decisions

**Direct conclusion:** The last 50 Q9-preparation steps matter to the final Q3 endpoint under a strictly common continuation operator in this seed. A shorter stage still achieves a strong finite-budget advantage. The assignments are not all fixed by Q9 step300, and Q3 can change several million positions in ways that partly converge but also diverge.

**Does not show:** that the additional 50 Q9 steps are always worthwhile per unit total compute (C1 is the equal-budget complement), how much of the benefit is causally attributable to any specific of the 6.25M changed positions, whether final codes represent separate good basins, whether direct Q3 catches up with more training, or whether any of this generalizes to independent datasets/hardware/model families. Dataset and test slice were previously viewed; seed1729 is exploratory.

**Scientific next step (not auto-launched):** freeze a paired longer-horizon experiment with identical underlying quantizer/objective/model and several common checkpoints, with *fresh evaluation data not used for recipe selection*. Also replicate M1 on genuinely separate training orders if investing in the assignment-dynamics mechanism. A causal split of Q9-only vs direct-only positions remains interesting but subordinate to reproducibility and long-run testing. Update [CURRENT_STATE.md](../CURRENT_STATE.md) and [NEXT_EXPERIMENT_PLAN.md](../NEXT_EXPERIMENT_PLAN.md) before any new compute.

**Historical tests/negatives retained:** T1 timing, C1 equal-total switch, v10-v13 mechanism, v5 longer-budget attenuation, Granite mixed transfer, Smol S1-1 gridward negative. No follow-up GPU job or retuning triggered by this summary.
