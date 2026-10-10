# R1 — Does Q9 need its wider representable range?

**Status:** COMPLETED 2026-10-10 00:05:57 UTC (2026-10-09 20:05:57 EDT). **VALID for the preregistered comparison:** `valid_for_science=true` and **12/12 technical checks passed**.  
**HF:** [job 6ac97c30095c57808930b904](https://huggingface.co/jobs/codeflash85/6ac97c30095c57808930b904), single A10G-small.  
**Pinned code SHA:** `f28f1d4814ab0a61f2b529e2add007ce142972d5`, [script](../r1_fineweb_q9_range_matched_seed1729.py).  
**Frozen BEFORE running:** [R1 preregistration](../research_log/r1_range_matched_q9_fineweb_seed1729_prereg_2026-10-09.md).  
**Full machine-readable terminal FINAL_JSON with provenance:** [R1 archived raw](run_r1_q9_range_matched_fineweb_seed1729_2026-10-10.json).

## Key answer

**Under the frozen R1 variant, reducing the Q9 preparation's representable output range from ±α to ±2α/3 nearly eliminates the staged-training advantage** on the fixed FineWeb heldout 21-document dataset. The narrower Q9 retains *exactly nine states*, the original outer normalized input clamp and same surrogate-gradient formula, and the same final Q3 continuation/schedule. A broad Q9 output range **or some coupled consequence of range-matching (spacing, thresholds, saturation)** appears important under this custom quantizer.

**Do not say range alone is proven causal.** A nine-state evenly spaced grid cannot have range narrowed without modifying spacing, decision boundaries and saturation; R1 is an intervention on that package, not a fully isolated range-only experiment.

## Fixed design and outcomes

Three individually initialized seed1729 student trajectories all using same SmolLM2-360M-Instruct BF16 source, FineWeb-Edu `sample-10BT` QAT document split and shared 1200 next-token training chunks/order, fixed teacher, 35% CE +65% teacher KL, same global v10 LR/AdamW/autocast/STE/row scales.

- **D:** direct Q3 all1200 with no stage switch, exact historical F1 direct anchor.
- **S_wide:** original nine-state Q9 outputs `α*k/4`, `k∈[-4,4]` → max ±α. Q9 300 steps, source-original Q3 scales + fresh AdamW/GradScaler, Q3 900 steps. Exact historical F1 staged anchor.
- **S_range:** range-matched nine-state Q9 outputs `α*k/6`, `k=clamp(round(6*z),-4,4)`, `z=clamp(w/α,-.99,.99)` → max ±2α/3. Same stage switch and continuation. Integer code clipped; unlike simply clipping z, **the original surrogate gradient through the unclipped middle of z is preserved**.

| Approach | FineWeb heldout NLL ↓ | FineWeb PPL ↓ | WikiText validation NLL ↓ | WikiText PPL ↓ | Effective updates / AMP skips |
|---|---:|---:|---:|---:|---:|
| Direct Q3 | 5.766660 | 319.469 | 6.558239 | 705.029 | 1194 / 6 |
| Original Q9→Q3, ±α | 4.996255 | 147.858 | 5.695225 | 297.444 | 1192 / 8 |
| Range-matched Q9→Q3, ±2α/3 | 5.751036 | 314.516 | 6.566215 | 710.675 | 1192 / 8 |

**Primary preregistered FineWeb difference** `S_range NLL − S_wide NLL` = **+0.754781067 nats/token** (range-matched is worse).  
**FineWeb staged advantage over direct:**
- **Wide Q9:** `D−S_wide` **+0.770404734**.
- **Range-matched Q9:** `D−S_range` **+0.015623666**.
- Range-matched retains **2.03%** of original wide-Q9 FineWeb advantage under this sample and training order.

**Secondary WikiText2 validation:**
- Wide Q9: `D−S_wide` **+0.863013443** nats/token.
- Range-matched Q9: `D−S_range` **-0.007976480** nats/token (negative; range-matched *slightly worse* than direct).
- `S_range−S_wide` **+0.870989922**.

## Why the matched-range quantizer's behavior differs

At the end of the 300 Q9 preparation opportunities, the frozen diagnostics were:

| Diagnostic | S_wide | S_range |
|---|---:|---:|
| Native-Q9 train-split dev NLL (24 chunks) | 5.428179 | 6.603545 |
| Projected-to-Q3 train-split dev NLL immediately after switch | 6.942240 | 6.739163 |
| Native→Q3 dev shock | +1.514062 | +0.135618 |
| Source master positions exceeding Q3 magnitude range `|w/α|>2/3` | 41.88% | 41.89% |
| Outer Q9 codes at ±4 | 29.30% | 47.83% |
| Hypothetical raw nine-level rounding codes requiring ±4 clipping | 0.00% | 36.45% |
| FP32 master positions beyond outer normalized z-clamp 0.99 | 25.37% | 25.12% |
| Max absolute Q9 output divided by α | 1.000000 | 0.666667 |

Range-matched Q9 sits at outermost ±4 codes much more often (**47.83% versus 29.30%**). Under `round(6z)` without integer clipping, **36.45%** of target masters would take code magnitude above4 and are instead clipped to4. Therefore the failure may result from constraining large-magnitude outputs, **changed cell occupancy and effective saturation, spacing/threshold shifts or their interactions**; this one job does not distinguish them.

The narrow variant has much poorer native-Q9 dev NLL after preparation (6.6035 versus5.4282), but a smaller immediate Q3 projection shock (0.1356 versus1.5141). This underscores why instantaneous projection shock alone is not a proxy for the final Q3 trajectory.

## Acceptance checks and reproducibility

- `all_1200_same_opportunities`: `true`
- `all_final_q3`: `true`
- `fixed_fineweb_document_split`: `true`
- `seed1729_training_order`: `true`
- `D_fineweb_reproduction`: `true`
- `S_wide_fineweb_reproduction`: `true`
- `D_wikitext_reproduction`: `true`
- `S_wide_wikitext_reproduction`: `true`
- `all_finite`: `true`
- `Q9_native_range_match`: `true`
- `Q9_wide_native_range`: `true`
- `same_prep_target_weights`: `true`

**12/12 all true**, including identical F1 original direct and original wide-stage NLL on both evaluation sets to recorded precision, known dataset split/order and matched maximal output magnitudes; all three arms completed exactly1200 scheduled opportunities with expected actual/skipped counts, ending in Q3.

The separate [CPU preflight HF job 6ac97b98095c57808930b8da](https://huggingface.co/jobs/codeflash85/6ac97b98095c57808930b8da) completed and checked code reachability `-4..4`, correct ±α/±2α/3 amplitudes, an unclipped STE gradient with respect to masters and scale, and original Q3 behavior after switching.

**Dependency versions** are in the raw R1 JSON; provenance includes the completed HF job and exact code Git commit. No trained weight checkpoint or document-linked per-document losses was saved.

## Scientific limits and next question

The measured difference is strong **within this recipe**, but it is single-seed1729, one SmolLM2 architecture, one fixed small FineWeb evaluation slice (21 docs) that may overlap with source-model pretraining, and old frequently referenced WikiText validation. Not a generalization or efficiency test, not an absolute calibration that the range matched scheme cannot ever work with a different re-initialization, clipping geometry, regularization or LR. All current experiments use frozen other nonquantized modules and no production generation-quality claim.

**Suggested next science:** disentangle saturation/spacing from magnitude range using a carefully preregistered controlled quantizer family and gradient/activation statistics, or test fresh heldout documents/model families before generalizing. No extra GPU jobs were launched to inspect R1. For authoritative status and broader context read [CURRENT_STATE.md](../CURRENT_STATE.md) and [R1 original job plan](../R1_ACTIVE_JOB_PLAN.md).
