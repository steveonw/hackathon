# R2 — Factorial Q9 range/spacing versus threshold policy, FineWeb seed1729

**Status:** Completed successfully **2026-10-10 01:00:05 UTC** (Oct 9, 2026, 9:00 PM EDT). Hugging Face A10G-small [job `6ac985f0fee2c9007017f720`](https://huggingface.co/jobs/codeflash85/6ac985f0fee2c9007017f720). `valid_for_science=true`; **all 16/16 preregistered technical and reproduction checks passed**.

**Frozen study before GPU execution:** [R2 preregistration](../research_log/r2_q9_range_threshold_factorial_fineweb_seed1729_prereg_2026-10-09.md), [immutable source](../r2_fineweb_q9_output_threshold_factorial_seed1729.py) at commit `93d7aba84bff0409b8cc91603ab9605cf1a9e09d`, [exact full FINAL_JSON and job provenance](run_r2_fineweb_q9_output_threshold_factorial_seed1729_2026-10-10.json). CPU [preflight job `6ac985c6095c57808930bb9c`](https://huggingface.co/jobs/codeflash85/6ac985c6095c57808930bb9c) passed all 4× quantizer grid/STE checks before paid training.

## Primary finding

**Tightening nine-state code thresholds alone does not explain R1's loss of the staging advantage.** Keeping the Q9 output grid **wide** (`q=alpha*k/4`, max ±α) retains meaningful improvement relative to direct ternary, even when the rounding thresholds are made tighter (T6) and almost **48%** of prep codes occupy extreme ±4 states. Conversely, using the **narrow** output grid (`q=alpha*k/6`, max ±2α/3) erases the staged gain even with the historical (T4) thresholds and the lower ~29% extreme-code occupation.

This **disfavors elevated extreme-code occupancy as the sole explanation** for R1's failure. A property of the **output amplitude/spacing package** is critical *under this exact 2×2 quantizer family*. Because evenly spaced nine-state grids couple range and spacing, this **does not isolate wider representable range alone** as the causal variable; no claim on other architecture/quantizer/corpus variants.

## Design

Five matched initial BF16-rounded SmolLM2-360M-Instruct seed1729 trajectories, using same frozen model rev and FineWeb-Edu sample-10BT data rev, first1200 training chunks/180 QAT training docs, 24 dev chunks/3 docs, 128 heldout chunks/21 docs, separate WikiText2 validation 128 chunks. Same source teacher, 0.35 CE+0.65 teacher-KL, FP32 master/row scales with other parameters frozen, same AdamW, AMP, global warmup/cosine LR and order. **D** direct Q3 all1200. Four Q9 arms each train Q9 for300 steps, reset original-source Q3 scales plus AdamW/GradScaler, and continue Q3 for900 steps.

At Q9-only stage, `z=clamp(w/alpha,-0.99,0.99)`, `k=clamp(round(T*z),-4,4)`, `q=alpha*(z+(k/V-z).detach())` (same clipped-z STE surrogate in all arms). `W` V4 max ±α and 1/4-α level spacing; `N` V6 max ±2α/3 and 1/6-α spacing. `T4` historical thresholds, `T6` tighter thresholds and higher saturation. Exactly nine integer codes in every Q9 prep. Original Q3 quantizer for all arms at endpoint.

## Exact held-out outcomes (lower is better)

| Arm | FineWeb heldout NLL | FineWeb PPL | WikiText validation NLL | WikiText PPL | Effective updates / AMP skips |
|---|---:|---:|---:|---:|---:|
| `D` | 5.766660 | 319.47 | 6.558239 | 705.03 | 1194 / 6 |
| `W4T4` | 4.996255 | 147.86 | 5.695225 | 297.44 | 1192 / 8 |
| `W4T6` | 5.263767 | 193.21 | 6.137133 | 462.72 | 1194 / 6 |
| `N6T4` | 5.780473 | 323.91 | 6.593468 | 730.31 | 1190 / 10 |
| `N6T6` | 5.751036 | 314.52 | 6.566215 | 710.68 | 1192 / 8 |

**Direct-Q3 FineWeb baseline:** 5.766660. Improvement `D−Q9`: W4T4 **+0.770405**, W4T6 **+0.502892**, N6T4 **-0.013813**, N6T6 **+0.015624**.

**WikiText validation baseline:** 6.558239. Improvements: W4T4 **+0.863013**, W4T6 **+0.421106**, N6T4 **-0.035229**, N6T6 **-0.007976**.

## Prespecified 2×2 contrasts

Here positive `NLL(A)−NLL(B)` means B achieved lower loss:

| Prespecified contrast | FineWeb difference (nats/token) | WikiText validation difference |
|---|---:|---:|
| `N6T6_minus_N6T4_threshold_effect_narrow` | -0.029437 | -0.027253 |
| `W4T6_minus_W4T4_threshold_effect_wide` | 0.267512 | 0.441907 |
| `N6T4_minus_W4T4_output_effect_T4` | 0.784218 | 0.898243 |
| `N6T6_minus_W4T6_output_effect_T6` | 0.487269 | 0.429083 |
| `interaction_narrow_threshold_effect_minus_wide` | -0.296950 | -0.469160 |

**Interpretation:**
- Holding original T4 threshold policy fixed, narrowing V4→V6 *worsened FineWeb* by **0.784218 nats/token**. Thus simply restoring T4 did **not** rescue narrow Q9.
- Holding tighter T6 thresholds fixed, narrowing V4→V6 worsened FineWeb by **0.487269 nats/token**.
- With wide V4 outputs fixed, tightening thresholds T4→T6 worsened FineWeb by **0.267512**, yet the resulting W4T6 still beats direct Q3 by **0.502892**.
- With narrow V6 outputs fixed, tighter thresholds T4→T6 changed loss by **-0.029437**, small relative to the V contrast; negative means T6 marginally improved the narrow variant.
- Threshold × output-grid interaction on FineWeb = **-0.296950** (difference of differences). Treat descriptively, no inferential p-value or CIs from one seed/heldout sample.

## Preparatory-state diagnostic at scheduled step300

| Arm | Native Q9 dev NLL | Immediately projected original-scale Q3 dev NLL | Native→Q3 loss shock | Fraction at integer codes ±4 | Would-be round(Tz) beyond ±4 | Max output / alpha |
|---|---:|---:|---:|---:|---:|---:|
| `W4T4` | 5.428179 | 6.942240 | 1.514062 | 29.30% | 0.00% | 1.000000 |
| `W4T6` | 6.225459 | 6.704398 | 0.478939 | 47.87% | 36.49% | 1.000000 |
| `N6T4` | 6.478196 | 6.782903 | 0.304707 | 29.22% | 0.00% | 0.666667 |
| `N6T6` | 6.603545 | 6.739163 | 0.135618 | 47.83% | 36.45% | 0.666667 |

The key dissociation: **W4T6 and N6T6 both have ~47.9% outer codes**, but wide W4T6 outperforms narrow N6T6 by **0.487269 nats**. **W4T4 and N6T4 both have ~29.2%**, yet wide W4T4 outperforms narrow N6T4 by **0.784218**. Saturation fraction alone cannot capture the behavior in this factorial.

Native Q9 dev loss and immediate Q3 projection shock vary substantially; the eventual 900-step Q3 recovery must be measured, not inferred solely from the immediate switch.

## Protocol checks and scientific limitations

- `five_arms_equal1200`: **true**
- `all_four_q9_switch_at300`: **true**
- `D_without_switch`: **true**
- `all_final_finite`: **true**
- `fixed_disjoint_docs`: **true**
- `fixed_order1729`: **true**
- `all_Q9_prep_9_codes`: **true**
- `all_Q9_prep_314m_weights`: **true**
- `factorial_output_ranges`: **true**
- `factorial_grid_metadata`: **true**
- `reproduce_D_FineWeb`: **true**
- `reproduce_D_WikiText`: **true**
- `reproduce_W4T4_FineWeb`: **true**
- `reproduce_W4T4_WikiText`: **true**
- `reproduce_N6T6_FineWeb`: **true**
- `reproduce_N6T6_WikiText`: **true**

All three historical anchors **D**, **W4T4** and **N6T6** reproduced R1/F1 FineWeb and WikiText losses exactly. All five arms completed1200 *scheduled* opportunities; AMP skips vary (see table) and should not be called equal successful optimizer steps. The CPU preflight confirmed nine code reachability, maximum magnitudes, identical codes at initial reference state when threshold multiplier matches and STE gradients; the GPU final checks confirm observed ranges and reproducibility.

**Caveats:** seed1729 only, SmolLM2-360M-Instruct only, one frozen v10 QAT schedule, limited previously viewed **21 FineWeb heldout documents**, previously reused WikiText validation, possible pretraining overlap and no quantized checkpoint. Uniform-grid range and spacing remain coupled; any time-varying master/scale distribution/STE scale-gradient changes are downstream consequences, not independently isolated. This work strengthens a causal claim *within* this operational grid family, **not** a general theorem about large-range quantization or a guarantee for production model quality.

**Next scientific gate (not launched):** design an explicit **nonuniform nine-level codebook** keeping range/spacing summaries while selectively modifying outer magnitudes, or test new heldout data/model before stronger generalization. Preserve negative controls and avoid tuning on repeatedly used heldouts. [CURRENT_STATE.md](../CURRENT_STATE.md) is the authoritative state.
