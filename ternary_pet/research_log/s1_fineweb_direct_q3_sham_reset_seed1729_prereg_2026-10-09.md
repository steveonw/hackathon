# S1 preregistration — FineWeb direct-Q3 sham switch at step300

**Preregistered:** 2026-10-09 EDT before new paid GPU job and outcomes; user approved G1 and S1 together; **R1 not launched**.  
**Prior F1 seed1729:** [raw](../results/run_f1_fineweb_edu_direct_vs_staged_seed1729_2026-10-09.json), [summary](../results/run_f1_fineweb_edu_direct_vs_staged_seed1729_summary.md). FineWeb F1/F2/F3 demonstrated paired staged benefits for 3/3 orders but did **not** have a Q3 sham optimizer/scale reset.

## Question / frozen methods

Would simply **resetting original source Q3 row scales, AdamW state, and GradScaler at global step300** cause direct Q3 training to match much of the staged benefit on the FineWeb sample?

- **Seed1729 only**; reuse exact F1 pinned model `HuggingFaceTB/SmolLM2-360M-Instruct` revision `a10cc1512eabd3dde888204e902eca88bddb4951`, fixed public FineWeb-Edu `sample-10BT` revision `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9`, WikiText2 validation revision `b08601e04326c79dfdd32d625aee71d232d685c3`. Same deterministic hashed doc split, train1200 chunks (180 docs), dev24 chunks (3 docs), FineWeb doc-heldout128 chunks (21 docs); same independent-from-QAT-training WikiText2 validation128 chunks.
- Same F1 base/student BF16 rounding, teacher FP16 fixed, same targeted trainable linears + per-row `raw_alpha`, frozen other modules, 35% CE/65% KL, 1e-3 peak/warmup100/cosine through1200, AdamW betas(0.9,0.95) wd0, grad norm1, GradScaler.
- **Two paired arms sequentially in one job:** `D_continuous` = original F1 direct Q3 all1200 without intervention (checks reproduction); `D_sham` = direct Q3 steps1..300, then while preserving FP32 master weights restore **original initialized Q3 row scales**, reset AdamW and GradScaler **at the same global step300** as F1 S300, continue direct Q3 steps301..1200 with unchanged global data chunks and LR (no restart). Native forward remains ternary in both phases; no Q9 training.
- **Do not** simultaneously freeze row scales or change range; S1 is a single sham-switch control, not an optimizer-reset-vs-scale-reset factorial.

## Endpoints, acceptance and scientific interpretation

- **Primary:** `NLL(D_continuous) - NLL(D_sham)` on heldout FineWeb. Secondary: NLL and PPL difference against **archived F1 stage-S300 (4.996255073696375)** as historical same-code/data/seed reference (not run in this job); same against WikiText validation (archived S300 5.695225466042757); dev trajectories, transition shock of Q3 original-scale restoration, AMP skipped/effective updates, final codes/hist/scales.
- Reproduction acceptance: D_continuous FineWeb heldout must match archived F1 D **5.7666598074138165** within **0.03 nats**, and WikiText2 validation archived D **6.558238908648491** within 0.03; first16 order exactly F1 order; 1200 scheduled opportunities/arm; Q3 all along; correct data split and disjoint docs, final finite NLL, preswitch direct Q3 dev identical between both arms within 1e-5, at step300 original Q3 scale restoration/optimizer reset exact and no LR restart.
- Honest interpretation: D_sham near historical staged advantage weakens the unique-Q9 explanation; sham much worse than staged weakens reset-alone explanation. Extra AMP skips are reported, not silently equalized; shared 21 heldout FineWeb docs/one order limit claims. Don't quietly retune when a check fails, preserve technical discrepancy.
- **Compute:** exactly **one HF `a10g-small` 90m job**; static QA then immutable SHA pin then paid GPU. Full `FINAL_JSON` archived after terminal status, plus summaries/handoff updates. No new seeds/jobs or R1 as side effects.
