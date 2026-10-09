# G1 + S1 — active 2026-10-09 falsification-control jobs

**This is the completed cross-session handoff for G1 and S1.** Check [CURRENT_STATE.md](CURRENT_STATE.md) for newer execution status. **R1 range-matched Q9 is not approved/launched.**

## Final execution record — both jobs COMPLETED

- **G1 [HF 6ac96a7efee2c9007017ea64](https://huggingface.co/jobs/codeflash85/6ac96a7efee2c9007017ea64)** completed **2026-10-09 23:16:07 UTC** with `valid_for_science=true` and 6/6 checks passed. [Raw JSON](results/run_g1_depth_by_scale_freeze_seed1729_2026-10-09.json) · [Summary](results/run_g1_depth_by_scale_freeze_seed1729_summary.md). Frozen scales retain **+0.620468** depth benefit, learned scales **+0.609848** nats/token; interaction +0.010620. Scale-gradient training NOT required for depth gain in this setting.
- **S1 [HF 6ac96a82fee2c9007017ea6b](https://huggingface.co/jobs/codeflash85/6ac96a82fee2c9007017ea6b)** completed **2026-10-09 23:08:23 UTC** with `valid_for_science=true` and 9/9 checks passed. [Raw JSON](results/run_s1_fineweb_q3_sham_reset_seed1729_2026-10-09.json) · [Summary](results/run_s1_fineweb_q3_sham_reset_seed1729_summary.md). Direct Q3 NLL 5.766660 continuous vs5.789199 sham reset (slightly WORSE); historic Q9 staged4.996255. Reset alone cannot explain the Q9 advantage in this seed.
- No new follow-up GPU jobs were launched to analyze these results. **R1 range-matched Q9 still unrun** and requires a separate engineering/protocol/compute decision.

**Notice:** the rest of this document is the ORIGINAL job plan with statuses *as of submission*. The authoritative updated execution status is [CURRENT_STATE.md](CURRENT_STATE.md); do not confuse archived `SCHEDULING` text with live state.

---

## Live jobs and exact source

| Study | Purpose | HF job | Pinned Git script SHA | Maximum GPU |
|---|---|---|---|---|
| G1 | Q3 within-bin depth × frozen/learned original row scale | [6ac96a7efee2c9007017ea64](https://huggingface.co/jobs/codeflash85/6ac96a7efee2c9007017ea64) | `059a0bb91aa55fea99ed56b1d0d980eb21fd31ae` · [G1 script](g1_depth_by_scale_freeze_seed1729.py) | A10G-small, 2h |
| S1 | FineWeb direct-Q3 with/without step-300 optimizer and original-scale reset | [6ac96a82fee2c9007017ea6b](https://huggingface.co/jobs/codeflash85/6ac96a82fee2c9007017ea6b) | `4cdab8ee73ddd804e96fac43992f91ab9a54c1b9` · [S1 script](s1_fineweb_q3_sham_reset_seed1729.py) | A10G-small, 90m |

**Both submitted 2026-10-09 EDT. Initial HF stage:** SCHEDULING; do not mistake this for a terminal result. **CPU static QA job:** `6ac96a6f095c57808930b267`, printed `G1_S1_STATIC_PREFLIGHT_OK`. **Frozen pre-job preregs:** [G1](research_log/g1_frozen_vs_learned_q3_scale_depth_prereg_2026-10-09.md), [S1](research_log/s1_fineweb_direct_q3_sham_reset_seed1729_prereg_2026-10-09.md).

## G1 details

Reproduce v13A 300-step D-Q3 and S-Q9 preparation under seed1729 and define the exact same projected D/S Q3 disagreement mask using common original Q3 row scales. Construct four Q3 starts differing only by depth d=0.03 vs0.50 on the Q9 target codes and scale policy. Learned-scale arms train FP32 masters + raw_alpha. Frozen-scale arms train FP32 masters only; raw_alpha is untrainable and excluded from optimizer. All four receive the **same 900 Q3 continuation batches and global LRs**. Primary predeclared interaction: (shallow−deep loss improvement)_frozen − (shallow−deep loss improvement)_learned. Record all code-survival checks at 100/300/900 under learned and common-original scale references; check d003/d050 learned-scale reproduction against historic v13A (±0.06 heldout). Familiar seed1729 and historically viewed WikiText test; exploratory.

## S1 details

On the same document-disjoint public FineWeb-Edu sample as F1, do seed1729 `D_continuous` (1200 original steps) and `D_sham` (same direct Q3 training, but at 300 reset original ternary row scales, AdamW and GradScaler without resetting FP32 masters or global LR). Both arms use the exact F1 document split and shuffled training order. Historical staged comparison is F1 S300 NLL **4.996255073696375** on FineWeb heldout and **5.695225466042757** on WikiText validation; baseline D must reproduce F1 D **5.7666598074138165** and **6.558238908648491** within ±0.03. FineWeb heldout includes only 21 documents; test order 1729 only, no new documents.

## Execution/archival instructions for the next AI

1. **Inspect the exact HF job IDs above** before doing anything. Do not start duplicates, R1, extra seeds, higher GPU, or an automatic retry.
2. Once a job reaches a terminal state, read logs with sufficient `tail` to reconstruct `FINAL_JSON_BEGIN`.. `FINAL_JSON_END` (lines may be split). Parse original JSON exactly. Require `valid_for_science` and all check flags; if failed, archive as a technical mismatch and preserve logs rather than claiming the scientific hypothesis failed.
3. Archive `FINAL_JSON` scientific fields plus HF job ID, terminal stage/completion time, immutable code SHA under `ternary_pet/results/`. Add exact numeric summary: all G1 arms, depth gains under both scale policies and interaction; or S1 D-continuous vs D-sham and historical F1 Q9 staged. Include AMP skip counts and limitations.
4. Update [CURRENT_STATE.md](CURRENT_STATE.md), [AI_HANDOFF.md](AI_HANDOFF.md), [EXPERIMENT.md](EXPERIMENT.md), README and living research reports. Keep completed historical F1/F2/F3 FineWeb and L1 6000-step findings intact.
5. If a job is still running, state it remains pending; never infer its result from earlier runs. G1 and S1 each have **only one preregistered seed (1729)**.

**Primary interpretation:** positive results support specific mechanisms *within* the already-valid finite-budget Q9 recipe; nulls/negative results may alter mechanistic attribution but do not erase documented recipe comparisons. In particular, G1 equal survival under fixed and learned scales is not sufficient to establish equal scale-gradient training effects without a frozen-scale intervention.
