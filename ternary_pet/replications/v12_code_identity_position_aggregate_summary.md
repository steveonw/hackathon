# v12 — three-order code identity vs continuous-position aggregate

The exact v12 intervention was run on all three established training orders:
1729, 271828, and 424242.

All runs passed the construction checks:

- exact/prototype projected-Q3 Hamming = 0;
- exact/minimal projected-Q3 Hamming = 0;
- exact/prototype/minimal pre-continuation diagnostics match;
- matched-random changed-position count equals the true mask size;
- matched-random instantiated codes match the constructed transition plan.

## Results

| Seed | D | Exact Q9 | Q3 prototype | Minimal crossing | Matched random | Prototype recovery | Minimal recovery | Random recovery |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1729 | 5.6136 | 4.9329 | **4.8884** | 5.4979 | 5.6793 | **106.54%** | 17.00% | -9.65% |
| 271828 | 5.6524 | 4.9711 | **4.9187** | 5.5406 | 5.7208 | **107.70%** | 16.41% | -10.04% |
| 424242 | 5.6234 | 4.9791 | **4.9373** | 5.4911 | 5.7031 | **106.50%** | 20.54% | -12.37% |

## Aggregate

- mean true-mask size: **6.31%** of quantized weights;
- mean direct loss: **5.6298**;
- mean exact-Q9-on-mask loss: **4.9611**;
- mean prototype loss: **4.9148**;
- mean minimal-crossing loss: **5.5099**;
- mean matched-random loss: **5.7011**.

Relative to direct:

- exact Q9-on-mask mean gain: **0.6688 nats/token**;
- prototype mean gain: **0.7150 nats/token**;
- minimal-crossing mean gain: **0.1200 nats/token**;
- matched-random mean gain: **-0.0713 nats/token**.

Recovery relative to the exact-Q9 positive-control gain:

- prototype: **106.91%** on average;
- minimal crossing: **17.98%** on average;
- matched random: **-10.69%** on average.

Prototype beats exact Q9 masters by **0.0463 nats/token
on average**, and does so in **3/3 orders**.

Minimal crossing recovers only **17.98%** on average and
remains much weaker in **3/3 orders**.

Matched-random reassignment is worse than direct in **3/3 orders**.

## Canonical mechanism conclusion

For this SmolLM2/WikiText-2 matched-schedule setup, the mechanism sequence now
supports a replicated three-part conclusion:

1. **Q9 identifies the useful positions and ternary assignments.**
   v11 localized ~96% of the full trainability gain to the ~6.3% of positions
   where Q9 and direct preparation choose different projected Q3 codes.

2. **Exact Q9 within-region FP32 coordinates are unnecessary.**
   Replacing the exact Q9 masters on those positions with the standardized Q3
   reconstruction prototypes performs slightly better in all three orders.

3. **Merely crossing the Q9-selected ternary boundary is insufficient.**
   The minimal-crossing arm starts from the same ternary forward model as the
   exact/prototype arms but recovers only ~18% of the exact-Q9 gain on average.
   Thus boundary-relative depth / continuous placement inside the selected
   region matters materially.

The matched-random control strengthens the positional claim: reproducing the
same number and per-layer/source->target types of ternary transitions at
different positions is consistently harmful rather than beneficial.

## Scope

This is a replicated mechanism result for one model, dataset, quantizer family,
training budget, and objective. It is not yet a claim of architectural or
dataset generality.

The mechanism phase can reasonably stop here. Further experiments should be
framed as a new phase:

- cross-model / cross-dataset generalization;
- a cheaper practical recipe that predicts or learns the useful Q9-selected
  positions/codes without a full Q9 preparation phase;
- or production-oriented quantizer redesign.

Confirmatory HF jobs:

- seed 271828: `6ac64ccddf2184ac91ac092d`;
- seed 424242: `6ac64cd2df2184ac91ac092f`.

Pinned replication scripts:
`06c2df406d3e2029f742d64ad1065408b9a6209b`.
