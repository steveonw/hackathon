# G1-7b — Granite-350M transfer of Smol v10–v13 learning-rate schedule

**Status:** completed successfully; negative schedule-transfer result on development order **271828**.

- Successful retry job: `6ac7248adf2184ac91ac768d` (A10G-small, completed 2026-10-08 05:26:40 UTC)
- Pinned amended implementation: `808fd4975d16111d9c1c841c44d9038a01995ba3`
- Prior technical failure: `6ac722dfdf2184ac91ac75e9` (full-size matched-random control infeasible; no final endpoint)
- Raw JSON: `run_g1_7b_granite350m_v10schedule_seed271828_2026-10-08.json`
- Historical fixed-LR comparator: `run_g1_4_granite350m_mechanism_seed271828_2026-10-08.json`

## Frozen protocol

Granite-4.0-350M, seed/order 271828, WikiText-2, BF16-rounded common source, FP32 masters, BF16 student and teacher forward, same CE35 + teacher KL65, original Q3 scales, fresh Adam for all mechanism continuations. Global LR follows **Smol v10–v13**: 100 updates warmup to `1e-3`, then cosine decay toward `1e-4` through step 1200 without restart at the 300-step boundary. D trains Q3 for 300; S trains Q9 for 300; all continuations train Q3 for steps 301–1200. The Random-d50 arm was explicitly omitted by a preregistered amendment after the first technical failure, before any G1-7 held-out endpoint was seen.

## Held-out comparison with prior constant-LR Granite 271828

Lower loss and PPL are better.

| Arm | Constant 1e-4 loss | Transferred v10–v13 loss | Transferred PPL | Loss increase |
|---|---:|---:|---:|---:|
| D | 5.72673 | **6.00895** | **407.05** | +0.28222 |
| S | 5.84103 | 6.36920 | 583.59 | +0.52816 |
| M-exact | 5.80453 | 6.35442 | 575.03 | +0.54989 |
| M-d50 | 5.78502 | **6.30402** | **546.77** | +0.51900 |

**Staging deficit on the hard order** grows from `−0.11431` to `−0.36025` nats (gain D−S). M-d50 remains `0.29508` nats worse than D, although it improves over full S by `0.06517` nats. M-d50 improves over M-exact by `0.05040` nats.

All four arms completed and construction assertions passed: S/M-exact/M-d50 have exactly identical projected ternary codes and identical pre-continuation validation diagnostics (loss **7.23919**, PPL **1392.97**). Direct pre-continuation fixed-Q3 loss was **6.81094**.

## Preparation and geometry

| Diagnostic | Constant 1e-4 | Smol v10–v13 schedule |
|---|---:|---:|
| Native D validation loss at 300 | 5.64539 | **6.45034** |
| Native Q9 validation loss at 300 | 5.92340 | **6.83970** |
| Native Q9 minus D | +0.27801 | **+0.38936** |
| Fixed-Q3 Q9 minus D immediate loss | +0.86508 | **+0.42825** |
| D projected codes changed vs initial | ~5.34% | **27.107%** |
| Q9 projected codes changed vs initial | ~5.08% | **23.432%** |
| D-vs-S disagreement mask | **6.995%** | **32.715%** |
| Mask positions | ~17.46M | **81,642,667** |

The transferred LR schedule drives roughly five times the D/S disagreement fraction and dramatically more source-code movement. Even though the *immediate fixed-Q3 gap* shrinks, the *final S deficit* grows. Thus immediate projection quality alone does not predict the final outcome.

At preparation step 300, logged pre-clip gradient norms were D **1.8998** and Q9 **9.7784**, ratio **5.15×**, despite both being clipped to norm 1 for the update. Q9 remained behind D during preparation under the aggressive schedule.

### d=0.5 commitment survival

True-mask S-selected-code survival during Q3 continuation under the transferred schedule:

| Continuation step | Learned scales | Fixed original scales |
|---:|---:|---:|
| 100 | 74.24% | 74.23% |
| 300 | 65.72% | 65.68% |
| 900 | **60.75%** | **60.70%** |

Historical constant-LR 271828 M-d50 survival at continuation 900 was **89.59%**. The new schedule both selects vastly more positions and retains markedly fewer selected codes after continuation. Similar learned/fixed results again argue against row-scale drift as the main cause of that difference.

## Interpretation boundaries

- **Negative schedule-transfer test:** copying the successful Smol v10–v13 LR curve worsens both direct and staged Granite endpoints on order 271828; it hurts staged more.
- **No practical rescue:** full S and M-d50 remain worse than D.
- **Interior placement still helps relative to S**, but this effect alone cannot overcome the much larger set of changing commitments.
- **Do not equate** the larger mask with all harmful positions; source movement, effective step size, and seed-specific dynamics are jointly altered by the schedule. Causation of the endpoint gap is not isolated.
- **No new random specificity test:** the originally planned exact full-size random match became mathematically impossible in one layer/source-code stratum (595,275 required versus 342,644 available outside M); the amended retry omits the arm. Earlier valid matched-random results apply only to their original schedules.
- **One known negative order only:** no independent replication or claim about the two positive Granite orders or another model family.
- This is a protocol-sensitivity result, not a clean v13 depth sweep. It keeps the inherited established `d=0.5` arm under the v10–v13 schedule.

## Scientific consequence

The same **number of updates** and apparently similar Q9→Q3 training recipe can traverse vastly different assignment regimes depending on the LR schedule. On Granite 271828, a Smol-tuned peak LR of `1e-3` produces a ~33% disagreement mask and dramatically reduced final selected-code survival. The cross-family algorithm cannot be defined by an LR number copied from Smol: *model-aware schedule calibration and assignment dynamics are essential*.

No further GPU jobs were launched as part of this outcome.
