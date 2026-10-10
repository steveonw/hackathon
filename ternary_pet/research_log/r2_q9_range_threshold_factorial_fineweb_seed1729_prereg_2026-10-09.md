# R2 preregistration — output-range/spacing × code-threshold/saturation geometry

**Frozen before submission, 2026-10-09 (Eastern; job may complete Oct 10 UTC).** Explicit authorization: user's "next job" after reviewing completed R1. Exactly **one bounded GPU experiment**, not an open-ended sweep or background monitor.

## Prior observation and question

R1's original nine-state Q9 preparation (k=round(4z), output α·k/4) improved the seed1729 FineWeb heldout NLL by **+0.770405** over direct Q3; R1's narrow nine-state variant (k=clip(round(6z),±4), output α·k/6) preserved just **+0.015624**, with 47.83% outermost ±4 codes vs 29.30% in wide. This simultaneously changed **output amplitude/spacing** and **rounding thresholds/saturation**, so claiming "range alone" would be unjustified.

**R2 question:** With nine integer codes `k∈{-4,...,4}` in every preparatory arm, does moving the *rounding thresholds* independently of the *output grid's range/spacing* recover (or degrade) the Q9→Q3 advantage?

## Preregistered 2×2 quantizer grid (plus exact direct Q3 anchor)

At 300-step **nine-state preparation only**, let `z=clamp(w/α,-0.99,0.99)`. Compute discrete nine-state code `k=clamp(round(T·z),-4,4)` and forward value `q=α·k/V`; STE uses `q=α·[z + (k/V-z).detach()]` in every arm, identical clipped-z backward surrogate. Independent factors:

| Q9 arm | Rounding multiplier T | Output divisor V | Output max magnitude | Expected qualitative code saturation |
|---|---:|---:|---|---|
| W4T4 historical original wide | 4 | 4 | ±α | Historical moderate |
| W4T6 wide output / narrow thresholds | 6 | 4 | ±α | High outer ±4 code use |
| N6T4 narrow output / wide thresholds | 4 | 6 | ±2α/3 | Moderate outer-code use |
| N6T6 R1 narrow | 6 | 6 | ±2α/3 | Historical high |

The names are **output divisor first**, then threshold multiplier: `W4T4=(V4,T4)`, `W4T6=(V4,T6)`, `N6T4=(V6,T4)`, `N6T6=(V6,T6)`.

**Important:** Range and quantization *spacing* remain inseparable within each evenly spaced 9-state output codebook, while **threshold policy is manipulated independently**. Thus R2 isolates the consequence of changed threshold/code-saturation geometry *conditional on each of two output grids*, not range alone, nor effects from scale learning during preparation.

**Fifth control arm D:** original continuous direct Q3(1200), with original Q3 STE `q=α·round(1.5z)/1.5`, no resets. All four Q9 arms prep Q9 for steps1–300, switch to exactly original Q3 301–1200, restore original-source Q3 raw row scales and fresh AdamW/GradScaler at the same global step300; no change to master weights, teacher, optimizer betas, LR schedule or examples. At switch, no Q9 threshold/divisor parameter remains active for Q3.

## Fixed model, data, optimizer and interpretive scope

Exact R1/F1 seed1729 SmolLM2-360M-Instruct BF16-rounded source revision `a10cc1512eabd3dde888204e902eca88bddb4951`; pinned public `HuggingFaceFW/fineweb-edu` `sample-10BT` revision `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9`; pinned WikiText2 validation `b08601e04326c79dfdd32d625aee71d232d685c3`. Training first 1200 fixed 129-token chunks sourced from 180 SHA256-document-disjoint FineWeb train docs, train-split dev24 chunks/3 docs, primary FineWeb evaluation128 chunks/21 doc-disjoint docs, WikiText validation128 chunks. Same order head `[221,850,89,747,685,1055,781,170,233,62,421,806,1031,1187,683,619]`. Teacher FP16; 0.35 CE+0.65 fixed-teacher KL; frozen nonquantized modules, train FP32 masters/row scales; AdamW betas 0.9/0.95, wd0, grad clip1, FP16 autocast/GradScaler, v10 global LR 100 warmup to1e-3 then cosine to1e-4 at 1200. Equal **1200 scheduled opportunities per arm**; count AMP skips separately. All students initialized from identical BF16-rounded source and RNG before their own run.

This sample has only 21 held-out FineWeb docs, may overlap with pretrained source model's historical training, and has been adaptively seen in research; R2 is a **mechanism probe**, not fresh-document generalization or independent cross-family validation.

## Fixed endpoints and predictions (no data-dependent selection)

Primary endpoints on the exact **FineWeb heldout**:
- For same narrow output V6, effect of changing threshold: `L(N6T6)−L(N6T4)`, positive means wider thresholds (T4) help narrow output.
- For same wide output V4, effect of changing threshold: `L(W4T6)−L(W4T4)`, positive means historical thresholds T4 are better than T6 under the same wide output.
- For same T4 thresholds, output-range/spacing effect: `L(N6T4)−L(W4T4)`, positive favors wide outputs with same instantaneous starting code assignments.
- For same T6 thresholds, output-range/spacing effect: `L(N6T6)−L(W4T6)`.
- Interaction `[L(N6T6)−L(N6T4)] − [L(W4T6)−L(W4T4)]`. Report all four contrasts in signed nats and raw arm NLL/PPL. No after-the-fact threshold optimisation.

Secondary outcomes: WikiText validation analogues, train-split dev at300/600/900/1200, switch native-Q9→Q3 NLL shock, per-arm prep code ±4 fraction/histogram, would-be unclamped round(Tz) code beyond4, outside STE z clip fraction, master|w/α|>2/3 fraction, final source-Q3 code changes, scale hist/summary, AMP skips and effective updates. At fixed pretrain input weights, the 2×2 variants sharing T must have identical integer code matrices before Q9 training; report CPU smoke checks explicitly.

**Anchors:** D FineWeb `5.7666598074138165`, WikiText `6.558238908648491`; W4T4 FineWeb `4.996255073696375`, WikiText `5.695225466042757`; N6T6 FineWeb `5.751036141067743`, WikiText `6.566215388476849`. Technical reproduction tolerance each **±0.03 nats**. Do not change anchors after results.

## Pass/fail and resource rules

1. Prior to paid GPU submission, parse whole pinned Python with `ast.parse`, verify exactly nine attainable states -4..4 for **all four** Q9 configurations across representative normalized inputs, ±α/±2α/3 outputs, explicit integer clipping after rounding and same surrogate derivative for unclipped weights; no Q3 branch affected. Test codes identity for pairs with same T at identical masters/scales and verify outer-code prevalence for T6 vsT4 on a fixed representative normalized-value grid. Check all 5 named arms.
2. Full experiment verifies split, order, 1200 opportunities/arm, all final Q3, exact 300 step switch for four Q9 arms, all Q9 grids' output bounds and code count, finite evaluations, reproduction anchors for D/W4T4/N6T6. Any failed prerequisite means `valid_for_science=false` and is a **technical comparability issue**, not hypothesis falsification.
3. **One A10G-small Hugging Face job with a 2-hour timeout**, sequential five arms, detached after submission. Full script pinned at immutable Git SHA. Do not launch retries, extra seeds or a sixth optimized quantizer as part of this authorization. Preserve numerical negatives and all logs without tuning.
4. On terminal status, archive exact `FINAL_JSON`, checks, job provenance, per-arm effects, R1 prior references, program versions and honest scope limitations. Update [CURRENT_STATE.md](../CURRENT_STATE.md), AI_HANDOFF, EXPERIMENT, README and living reports. No new job launched during archival.
