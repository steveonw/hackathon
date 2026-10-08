# G1-8 — Granite 350M, seed 424242: Smol v10–v13 schedule transfer

**Result:** successful four-arm run, all equal-forward construction checks passed.
This is the *weaker historically positive Granite order* chosen after inspecting
historical constant-LR results, and is a targeted protocol-sensitivity test.

- Job: `6ac79551e7a0dae8a2788f0b`, completed 2026-10-08 13:30:40 UTC on A10G-small
- Code pin: `39399baff38b15f961e9571f142e193a783c925b`
- Script: `ternary_pet/g1_granite350m_v10_schedule_mechanism_seed424242.py`
- Raw: `run_g1_8_granite350m_v10schedule_seed424242_2026-10-08.json`
- Constant-LR reference: `run_g1_5_granite350m_mechanism_seed424242_2026-10-08.json`

## Protocol and successful checks

The pinned script differs from the completed G1-7b development harness by
**exactly one line**: seed/order 271828 → 424242. Four arms D, S, M-exact, M-d50;
no Random-d50 (the exact full-size random control was infeasible on G1-7b).
Original Q3 scales and fresh Adam at continuation, same 300-step Q3/Q9
preparation and 900-step Q3 continuation. Smol v10–v13 global LR: steps 1–100
linear warmup to 1e-3, then cosine decay to 1e-4 at global step 1200; LR at
step 300 0.000928564, at step 301
0.000927868. Granite BF16 student/teacher
autocast and FP32 masters retained.

All four arms reached the final 900th Q3 continuation step; all endpoint losses
finite. For all 249,561,088 quantized target weights, the S/M-exact and
S/M-d50 projected Q3 code mismatch fractions at the intervention were both
**0**, and validation diagnostics were identical to numerical tolerance.
The Random-d50 arm was explicitly omitted, so this run cannot establish
position-specificity against a new matched random baseline.

## Final held-out results

| Arm | Historical constant-1e-4 loss | Smol schedule loss | New PPL | Increase in loss |
|---|---:|---:|---:|---:|
| D | 5.80192 | 6.05708 | 427.12478 | +0.25516 |
| S | 5.74057 | 6.06241 | 429.41116 | +0.32184 |
| M-exact | 5.74471 | 6.04609 | 422.45680 | +0.30138 |
| M-d50 | 5.71016 | 6.02857 | 415.12254 | +0.31841 |

All four endpoints worsen under the transferred schedule.

- Full S versus D: earlier **+0.06135** nats advantage, now
  **−0.00534** nats (slight disadvantage). Change **−0.06669**
  nats in D−S advantage.
- M-d50 versus D: earlier **+0.09176**, now
  **+0.02850** nats: weakened, but still a
  **positive** d50 intervention effect.
- M-d50 versus S under transferred schedule: **+0.03384**
  nats.
- M-exact versus D under transferred schedule: **+0.01099**
  nats. M-d50 remains superior to M-exact.

## Preparation quality, geometry, and survival

| Diagnostic | Historical constant-1e-4 | Transferred Smol schedule |
|---|---:|---:|
| Native D validation loss @300 | 5.68460 | 6.30551 |
| Native Q9 validation loss @300 | 5.54681 | 6.21319 |
| Native Q9 minus D @300 | −0.13780 | −0.09232 |
| D codes moved vs initial | 5.339% | 27.280% |
| Q9 projected codes moved vs initial | 5.268% | 27.124% |
| D−S projected Q3 disagreement mask | **6.030%** | **29.490%** |
| Count of disagreeing weights | 15,047,433 | 73,596,548 |
| d50 S-code survival @Q3 continuation 900 | 90.569% | 57.143% |

Native Q9 prep **still beats D** at step 300 in seed 424242 despite the
high-LR schedule, by 0.09232 nats. The 300th-step
preclip gradient norms are D **1.941**,
Q9 **1.257**
(Q9/D 0.65×).
This contrasts sharply with seed 271828, where Q9 prep was worse than D by
0.38936 nats and
Q9 gradient norms remained >5× D. So the native-prep *failure mode* differs,
even though the aggressive LR expands the projected-code disagreement masks
dramatically on **both** tested orders.

At the Q3 switch under the transferred schedule, fixed-reference-scale direct
validation loss is **6.49547**; S/M-exact/M-d50 all match at
**7.01969**. Thus Q9 still projects to a worse immediate
Q3 model despite its better **native Q9** prep. Learned-scale versus fixed
alpha0 selected-code survival at Q3 continuation step 900 is
57.143% vs
57.116%;
row-scale drift does not explain the survival loss alone.

## Two-order schedule comparison

| Arm | 271828 const | 271828 Smol LR | 424242 const | 424242 Smol LR |
|---|---:|---:|---:|---:|
| D | 5.72673 | 6.00895 | 5.80192 | 6.05708 |
| S | 5.84103 | 6.36920 | 5.74057 | 6.06241 |
| M-exact | 5.80453 | 6.35442 | 5.74471 | 6.04609 |
| M-d50 | 5.78502 | 6.30402 | 5.71016 | 6.02857 |

The D/S projected mask under Smol LR was **32.715%**
for 271828 and **29.490%** for 424242, versus
6.995% and 6.030%, respectively, under
the original constant LR.

The stronger schedule **degraded every final arm on both tested orders**,
weakened or reversed the full S advantage, expanded the code disagreement to
roughly 30%, and lowered selected-code survival substantially. This supports
a **Granite-specific schedule sensitivity** beyond one pathological order.
It does not demonstrate that all Granite seeds fail, nor that mask expansion
alone causally explains the losses.

The `d=0.5` intervention is still beneficial *relative to the within-run D*
on seed 424242; on seed 271828 it was worse than D. Therefore the distinction
between useful code commitments and a globally robust full-staging trajectory
remains important.

This was a **targeted follow-up selected using known historical seed outcomes**.
It is not an unbiased fourth seed, and cannot overturn the Smol 3/3 matched
schedule confirmations. No new random-control specificity claim is permitted.
G1-8 reuses the old event/kind string `g1_7` from the otherwise identical
parent script; refer to it by **job ID + seed 424242** to avoid ambiguity.

## Next scientific decision (not yet authorized to run)

The directly copied Smol peak LR is too disruptive on both tested Granite
orders at this 300/900 budget. Keep original constant-1e-4 G1 outcomes as the
valid Granite family comparison, and test a *Granite-calibrated* schedule if
pursuing further optimization. Separately, the new boundary-margin/row
coordination proposal may address whether the Q9 mask contains predictable
beneficial ternary decisions. Any predictor/matched-random control needs its
own preregistration and a fair, feasible matched subset.

No additional GPU job is authorized by this result.
