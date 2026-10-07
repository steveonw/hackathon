# v12 — code identity vs continuous position on the Q9 disagreement mask

**HF job:** `6ac642a0df2184ac91ac018d`  
**Pinned code:** `22639f5b225e56be009886bf467a409910796eef`  
**Seed/order:** 1729

## Question

v11 replicated **where** the useful Q9 state resides: the ~6.3% of positions
where direct-Q3 and Q9 preparation choose different projected Q3 codes.

v12 asks what property of those positions matters:

- the Q9-selected Q3 code itself;
- depth/position inside the selected Q3 region;
- the exact continuous Q9-prepared master value;
- or merely making the same kinds of code changes at arbitrary matched positions.

## Validity checks

The true-mask arms `exact`, `proto`, and `minimal` all start from the
**same projected Q3 model**:

- exact vs prototype Q3 Hamming: **0**
- exact vs minimal Q3 Hamming: **0**
- paired pre-continuation validation losses: **exactly equal**

The matched-random control also passed its construction checks:

- changed-position count exactly equals |M|;
- D-vs-random Hamming exactly equals the true mask fraction;
- random positions are outside true M;
- instantiated random codes exactly match the constructed plan.

Thus differences among exact/prototype/minimal arise only from hidden continuous
master position while holding the ternary forward model fixed.

## Final held-out results

| Arm | Construction on M | Loss | PPL | Top-1 | KL | Gain vs D | Recovery vs exact |
|---|---|---:|---:|---:|---:|---:|---:|
| D | direct masters | 5.6136 | 274.13 | 29.72% | 2.6791 | — | — |
| exact | exact Q9 masters | 4.9329 | 138.78 | 36.07% | 1.9969 | **0.6807** | 100% |
| proto | Q3 prototype of Q9-selected code | **4.8884** | **132.74** | **37.66%** | **1.9428** | **0.7252** | **106.54%** |
| minimal | epsilon=0.01 inside Q9-selected region | 5.4979 | 244.18 | 31.09% | 2.5699 | 0.1157 | 17.00% |
| random | matched random reassignment, prototypes | 5.6793 | 292.75 | 28.99% | 2.7437 | -0.0657 | -9.65% |

Key comparisons:

- prototype is **-0.0445 nats better** than exact Q9 masters;
- prototype recovers **106.54%** of the exact-S
  positive-control gain (it over-recovers because it performs better);
- minimal crossing recovers only **17.00%**;
- the matched-random reassignment is **0.0657 nats worse than D**.

## What this means

The result rejects two simple stories.

### 1. Exact Q9 continuous values are not required

The standardized Q3 reconstruction prototype performs **better** than the exact
Q9 master values while starting from the identical ternary forward model.

Therefore the precise continuous Q9-prepared value on M is not necessary for
the observed trainability advantage on this seed.

### 2. Q9 code identity alone is not sufficient

The minimal-crossing arm has exactly the same Q9-selected projected Q3 code as
the exact and prototype arms, but recovers only ~17% of the exact-S gain.

So merely crossing the threshold into the Q9-selected code region is not enough.
**Boundary-relative depth / continuous position inside the region matters.**

The cleanest current interpretation is:

> Q9 identifies a highly useful set of **which ternary assignments to make at
> which positions**, but later Q3 optimization also benefits from placing those
> masters safely inside the selected Q3 regions rather than barely across the
> decision boundary. The canonical Q3 reconstruction prototypes are sufficient
> — and on seed 1729 slightly better than the exact Q9 master values.

The matched-random control strengthens the positional claim: reproducing the
same number, layer distribution, source-code mix, and target transition types at
different positions makes the model worse than direct. Thus the benefit is not
generic disruption or merely changing ~6.46% of codes.

## Mask composition

True mask size: **6.46%**
(20,315,352 positions).

Target-code split:

- target 0: **49.78%**
- target ±1 combined: **50.22%**

Transition geometry:

- adjacent transitions through zero: **99.95%**
- direct -1 <-> +1 sign flips: **0.05%**

So nearly the entire mask consists of one-boundary code changes rather than
direct sign reversals.

Descriptive nearest-boundary distances for the exact prepared D/S states are
very similar (~0.063 normalized units) in both target-zero and target-nonzero
groups. This does not conflict with the intervention: v12 deliberately moves
the prototype to the Q3 reconstruction level and the minimal arm to only 0.01
inside the boundary.

## Scope and next step

This is seed 1729 first. It is strong and interpretable enough to replicate
exactly on orders 271828 and 424242.

Do not yet claim the prototype result is three-order canonical.

If prototype again approaches or beats exact while minimal remains weak and
matched-random remains non-beneficial on both confirmatory orders, then the
mechanism can be summarized as:

1. Q9's **specific position/code selection** carries the useful discrete
   structure;
2. being **well inside** the selected Q3 region is important;
3. exact Q9 within-region continuous coordinates are unnecessary.
