# Run v2 summary — strict nested balanced ternary

**Hugging Face Job:** `6ac527e5404719ba37661dc9`  
**Model:** `HuggingFaceTB/SmolLM2-360M-Instruct`  
**Hardware:** Tesla T4 (`t4-small`)  
**Wall-clock runtime:** about 8 minutes 5 seconds  
**Status:** completed

## What changed from v1

v2 used a literal nested balanced-ternary code family.

For each output row, the final ternary magnitude `alpha` was estimated once
from the original full-precision weights and frozen. With `s = alpha / 9`:

- 27 states: `-13s ... +13s`
- 9 parent states: `-12s, -9s, ..., +12s`
- 3 grandparent states: `-9s, 0, +9s`

Every three adjacent child states map to exactly one parent. Recovery used
WikiText-2 plus teacher-logit distillation instead of the tiny hand-written
calibration set used in v1.

## Held-out results

| Variant | PPL | Top-1 agreement | KL to teacher |
|---|---:|---:|---:|
| Original baseline | 33.96 | 100.00% | ~0 |
| Direct 3-level PTQ | 1,159,381 | 0.163% | 10.919 |
| Nested 27 -> 9 -> 3 PTQ | 1,159,381 | 0.163% | 10.919 |
| Direct ternary QAT, 90 steps | **50,175** | **2.181%** | **7.790** |
| Nested QAT, 30+30+30 | 920,837 | 0.260% | 10.710 |
| Nested QAT, 30+30+90 | 920,837 | 0.260% | 10.710 |

Absolute perplexity is specific to this short WikiText-2 slice. The relative
comparison is the useful part.

## Finding 1: raw staging is mathematically neutral here

Direct ternary PTQ and strict nested PTQ were **exactly identical** on loss,
perplexity, top-1 agreement, KL, ternary-state histogram, and generated text.

That is an important confirmation of the earlier theoretical warning:
if every stage is a deterministic nested projection and there is no adaptation
between projections, taking the scenic route cannot improve the final ternary
model. The 27 and 9 stops only become meaningful if optimization changes the
model while it is there.

## Finding 2: strict nesting fixed the v1 instability

The 9-level recovery stage no longer exploded.

For the equal-total nested run:

- 27 levels: mixed loss 12.40 -> 7.32
- 9 levels: mixed loss 11.54 -> 9.29
- 3 levels: mixed loss 12.73 -> 10.00

So the nested codebook + broader calibration + distillation made the staircase
much more numerically controlled than v1.

## Finding 3: stability did not translate into a better final ternary model

Direct ternary QAT still won decisively on every main held-out metric.

Even giving the staged route the **same 90 final ternary steps** as the direct
route did not improve its final evaluation. The equal-total and equal-final
nested variants produced exactly the same final metrics, generations, and state
histogram.

The most likely explanation is that the extra ternary STE updates did not push
enough weights across fixed discrete-state boundaries after the hard parent
transition. That should be tested explicitly in a future run by counting
per-weight code changes.

## Qualitative behavior

The original model remained coherent.

All ternary variants were still badly damaged. Direct QAT was the least damaged
numerically, but its sample generations were still degenerate. The nested
variants mostly generated repeated token patterns such as `monton`, `ston`,
and asterisks.

## v2 conclusion

**Strict 27 -> 9 -> 3 ancestry by itself did not beat direct ternary QAT.**

However, v2 taught us something cleaner than v1:

1. deterministic nested staging alone is exactly neutral;
2. nested recovery can be stable;
3. the destructive event is the final ternary projection;
4. with frozen scales and hard stage commits, recovery appears unable to move
   enough weights into better ternary states.

That points the next experiment away from "more steps" and toward changing the
optimization geometry: preserve continuous shadow weights across stage
transitions, learn the ternary scale/threshold, or soften the parent transition
rather than hard-committing it.
