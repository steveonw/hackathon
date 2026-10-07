# Run v8 summary — signed preload diagnostic

**HF Job:** `6ac5a0b1404719ba37664653`  
**Pinned code:** `356bff9769c96faa0ca139f9cba088fc1a52c2c8`  
**Seed/order:** 1729  
**Hardware:** A10G-small

## Result

v8 does **not** support a Q9-specific directional pre-loading mechanism.

The experiment measured, for weights whose Q3 code later changes during the
common 900-step ternary continuation:

```
aligned_preload =
sign(final_code - step300_code)
* (W300 - W0) / alpha0
```

Positive values mean preparation had already moved the FP32 master in the same
direction as its later ternary transition.

Because later-flip sets are selected after each arm's own trajectory, the
preregistered interpretation required a four-way matched comparison rather than
looking only at each arm on its own future-flip set.

## Four-way matched result

### On Q9's future-flip positions and directions

- Q9 preparation mean aligned displacement: **0.002328**
- Q3 preparation on those exact same positions/directions: **0.000873**
- Q9 minus Q3: **+0.001455**
- fraction of matched weights where Q9 displacement is more aligned than Q3:
  **53.44%**

So Q9 wins on its own future-flip set.

### On Q3's future-flip positions and directions

- Q3 preparation mean aligned displacement: **0.003130**
- Q9 preparation on those exact same positions/directions: **0.001439**
- Q3 minus Q9: **+0.001691**
- fraction of matched weights where Q3 displacement is more aligned than Q9:
  **53.77%**

So Q3 also wins on its own future-flip set.

The reciprocal own-set advantages are similar, and the direct-Q3 own-set
advantage is actually slightly larger in mean magnitude:

- Q9 own-set advantage: **0.001455**
- Q3 own-set advantage: **0.001691**

This is the pattern preregistered as compatible with **post-selection/general
trajectory alignment**, not Q9-specific directional foresight.

## Own-set alignment is real but generic

Q3 on its own future changers:

- continuation-changing fraction: **1.195%**
- mean aligned preload: **0.003130**
- median: **0.002226**
- fraction positive: **59.11%**

Q9 on its own future changers:

- continuation-changing fraction: **1.150%**
- mean aligned preload: **0.002328**
- median: **0.001735**
- fraction positive: **57.28%**

Both trajectories therefore show a genuine tendency for later-changing weights
to have already moved in the eventual direction during preparation, but Q9 is
not special on this statistic.

## A useful secondary pattern

Among weights that **did not change Q3 code during the first 300 updates** but
later changed during continuation:

- Q3: mean aligned preload **0.006054**,
  **70.96%** positive;
- Q9: mean **0.005254**,
  **69.68%** positive.

By contrast, weights whose Q3 code had **already changed during preparation**
show strongly negative alignment with their later continuation code change in
both arms:

- Q3 mean: **-0.011454**
- Q9 mean: **-0.011192**

That looks like a generic crossing/reversal or oscillation phenomenon rather
than a Q9-specific mechanism. It should not be overinterpreted without a more
explicit transition-history measurement.

## v7 behavior reproduced inside v8

v8 reproduces the seed-1729 v7 performance:

| Arm | Step-300 Q3 loss | Final loss | Final PPL |
|---|---:|---:|---:|
| Direct Q3 | 6.5589 | 5.8724 | 355.09 |
| Q9 -> Q3 | 9.2264 | **5.1938** | **180.15** |

Q9 again starts from the worse immediate Q3 checkpoint and finishes far better.

The Q3-vs-Q9 code Hamming distance grows from
**0.919%** at step 300 to
**1.869%** at the end of
continuation.

## Interpretation

v8 **falsifies the simple signed-preload explanation**:

> Q9 does not appear to win because it uniquely pushes future-changing weights
> in their eventual ternary-transition directions during preparation.

Instead, both Q3 and Q9 trajectories exhibit similar own-set directional
alignment once later-flip selection is taken into account.

The replicated v7 trainability / weight-selection effect therefore remains real,
but its mechanism is deeper than a simple per-weight "point it toward the
future threshold" story.

The next high-value causal test should target **which Q9-selected weights matter**
rather than assuming their benefit is explained by signed threshold direction.
A null/non-flipper analysis can still characterize the generic alignment effect,
but it is no longer likely to explain Q9's performance advantage by itself.
