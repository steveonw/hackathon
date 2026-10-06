# v4b confirmation — 9 -> 3 advantage across training orders

The v4 result has now been tested under two additional **independent training
orders**. The first attempt to vary only `torch.manual_seed` was correctly
discarded after it produced bit-for-bit identical traces; v4b instead uses the
seed to permute the same 1200 training chunks.

Within every run, direct and staged conditions see the **same examples in the
same order**. The only treatment difference is:

- direct: 1200 steps with a 3-state forward quantizer;
- staged: 300 steps with 9 states, then 900 steps with 3 states.

The held-out 8192-token evaluator is fixed.

## Results

| Training order | Direct PPL ↓ | 9->3 PPL ↓ | PPL reduction | Direct top-1 | 9->3 top-1 | KL direct | KL 9->3 |
|---|---:|---:|---:|---:|---:|---:|---:|
| reference v4 | 666.66 | **286.97** | **56.95%** | 22.84% | **30.49%** | 3.546 | **2.716** |
| seed 1729 | 352.77 | **178.85** | **49.30%** | 25.45% | **33.76%** | 2.950 | **2.238** |
| seed 271828 | 383.56 | **177.20** | **53.80%** | 25.44% | **32.08%** | 3.046 | **2.258** |

All three paired comparisons favor 9 -> 3 on all three held-out metrics.

Across the three orderings:

- mean paired perplexity reduction: **53.35%** (range 49.30%–56.95%);
- mean teacher top-1 gain: **+7.54 percentage points**;
- mean paired KL reduction: **24.47%**;
- mean held-out loss improvement: **0.765 nats/token**.

## Transition-shock replication

The first fully ternary loss was smaller after 9-state preparation in every run:

| Training order | Direct first ternary loss | After 9-state prep | Reduction |
|---|---:|---:|---:|
| reference v4 | 17.31 | 6.59 | 61.91% |
| seed 1729 | 13.12 | 7.50 | 42.81% |
| seed 271828 | 12.51 | 6.47 | 48.31% |

Mean reduction: **51.01%**.

## Mechanistic clue

The 9-state phase makes substantially larger discrete rearrangements before the
ternary landing. In the two shuffled-order replications, ~2.35% of 9-state
codes differed from their stage-start assignments after 300 steps. During the
following 900 ternary steps, only ~1.14% of ternary codes ended different from
their ternary-stage start.

Direct ternary, by comparison, ended with ~1.48% of codes displaced from its
initial ternary assignments.

This does **not** imply that fewer ternary flips are inherently better. It does
support the interpretation that the network is using the 9-state phase to make
larger preparatory rearrangements before entering the more restrictive
three-state space.

## Conclusion

The v4 staged advantage is **reproducible across the two tested shuffled
training orders**. That is substantially stronger evidence than the original
single run.

It is still not evidence of a production-ready conversion:

- only one model architecture/checkpoint has been tested;
- only one dataset/evaluator has been used;
- three paired orderings are too few for broad statistical claims;
- generated text is still qualitatively damaged.

The next useful experiment should increase training/data budget while keeping
this direct-vs-9->3 paired design fixed, rather than changing the schedule again.
