# G1 — frozen versus learned per-row scales during Q3 depth continuation

**Status:** COMPLETED, **2026-10-09 23:16:07 UTC**, `valid_for_science=true` and **6/6 technical/reproduction checks passed**.  
**Job:** [HF `6ac96a7efee2c9007017ea64`](https://huggingface.co/jobs/codeflash85/6ac96a7efee2c9007017ea64), A10G-small.  
**Pinned code:** `059a0bb91aa55fea99ed56b1d0d980eb21fd31ae`, [G1 script](../g1_depth_by_scale_freeze_seed1729.py).  
**Preregistration:** [G1 fixed protocol](../research_log/g1_frozen_vs_learned_q3_scale_depth_prereg_2026-10-09.md).  
**Full scientific raw output:** [G1 JSON](run_g1_depth_by_scale_freeze_seed1729_2026-10-09.json).

## Question and isolated manipulation

The v13A depth intervention placed the Q9-selected projected ternary codes in shallow or deeper positions within their final Q3 bins. The third-party reviewer correctly noticed this also changes the STE gradient with respect to learned row scale. G1 asks whether **trainable scales are necessary for the depth benefit**.

One familiar SmolLM2-360M seed1729 / WikiText2 recipe: independently prep Q3 and Q9 states through300 scheduled steps, determine Q3 code disagreement mask M (20,315,352 of 314,572,800 target positions; 6.4581%). On M, start from Q9-selected Q3 code with normalized bin depth either `0.03` or `0.50`; use direct Q3-prepared FP32 masters elsewhere. Projected initial Q3 code matrices and starting evaluations are identical across all four depth × scale-policy arms. Fresh AdamW+GradScaler; identical 900 remaining Q3 batches and global LR values. **Frozen** variants omit `raw_alpha` parameters from the optimizer and disable their gradients; original Q3 scales stay exactly unchanged.

## Final outcomes

| Arm | Depth | Row scale | Heldout NLL ↓ | PPL ↓ | Successful Q3 updates | AMP skipped | Target-code survival learned scale @900 | Fixed initial scale @900 | Max raw-scale drift |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| d003_learned | shallow 0.03 | learned | 5.494817656 | 243.427 | 898 | 2 | 51.761% | 51.753% | 0.224320 |
| d050_learned | deeper 0.50 | learned | 4.884970061 | 132.287 | 898 | 2 | 97.317% | 97.332% | 0.306376 |
| d003_frozen | shallow 0.03 | frozen | 5.516547829 | 248.775 | 898 | 2 | 51.674% | 51.674% | 0.000000 |
| d050_frozen | deeper 0.50 | frozen | 4.896080192 | 133.764 | 898 | 2 | 97.344% | 97.344% | 0.000000 |

**Primary contrast:** shallow minus deep gain with trainable scales **+0.609847594 nats/token**, versus **+0.620467637** with scales frozen. The prespecified interaction `frozen_depth_gain − learned_depth_gain` is **+0.010620043**.

The depth effect persists **essentially fully** when row scales cannot learn. In this one seed, the frozen-policy depth contrast is numerically 0.0106 nats larger. This refutes **scale learning being necessary** for the observed depth effect under this preparation and continuation; it **does not** prove that scales never affect Q9 training or explain every aspect of Q9's full stage-vs-direct advantage.

**Code-survival diagnostic:** Under learned scales, Q9 target code survival at continuation900 rises from **51.76%** (shallow) to **97.32%** (deep). With *frozen* scales, survival rises from **51.67%** to **97.34%**. The fixed-vs-current-scale survival readouts remain close for learned-scale arms. This is consistent with the assignment-retention explanation but is **not** a separately controlled intervention on survival alone.

## Technical checks

- `all_start_code_and_dev_equal`: **true**
- `all_four_900_q3_steps`: **true**
- `both_frozen_scale_unchanged`: **true**
- `all_finite`: **true**
- `learned_d003_reproduced`: **true**
- `learned_d050_reproduced`: **true**

- Every arm had precisely 900 Q3 continuation opportunities, **898 successful and 2 AMP-skipped**, with final Q3 quantization.
- Learned shallow NLL **5.494817656**, learned deep **4.884970061**, reproducing historical v13A depths 0.03 and 0.50 within the preregistered ±0.06.
- Both frozen variants demonstrated **zero max raw_alpha drift**, optimizer did not contain scales, and scale gradients were absent (checked inside job).
- At the start, all four depth arms matched the same ternary codes and train-split diagnostic; all matched the Q9 projected target codes.

## Interpretation boundaries

This does **not** isolate Q9's wider representable range, the role of trainable scales **during Q9 preparation**, row/position selection beyond the Q9 mask, or why any individual assignment is important. It is one seed with historical familiar WikiText test data and the custom quantizer/frozen-module protocol. It supports **interior positioning and subsequent assignment stability** as meaningful state variables even in the absence of trainable row scales. Loss/retention correlations alone do not prove exact causal mediation by assignment survival.

**Next scientific gate:** S1 FineWeb direct-Q3 sham-reset result is a complementary check; R1 range-matched nine-state preparation remains unrun. See [CURRENT_STATE.md](../CURRENT_STATE.md), [S1 companion result](run_s1_fineweb_q3_sham_reset_seed1729_2026-10-09.json) and [G1/S1 handoff](../G1_S1_ACTIVE_JOB_PLAN.md).
