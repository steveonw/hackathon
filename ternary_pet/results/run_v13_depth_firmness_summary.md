# v13 — seed-1729 depth sweep and firmness controls

The original monolithic v13 job timed out after all construction assertions had
passed. The experiment was rerun as two timeout-safe split jobs with the same
seed, preparation, schedule, optimizer semantics, and evaluation.

- v13A depth sweep: `6ac6a265df2184ac91ac410d`
- v13B firmness controls: `6ac6a272df2184ac91ac412c`
- pinned shared split commit:
  `9332a0a0b4063f7ed6786aa29fd049429cd50380`

Both completed successfully.

The direct baseline reproduced exactly in both split jobs:
**loss 5.6136**.

## Validity

All depth arms had:

- zero projected-Q3 Hamming to d=1.00;
- zero projected-Q3 Hamming to the intended Q9-selected projection;
- matching pre-continuation diagnostics.

B1 and B2 had:

- zero projected-Q3 Hamming to direct;
- matching pre-continuation diagnostics.

Thus depth-arm differences isolate hidden master placement while holding the
starting ternary forward model fixed, and B1/B2 isolate firmness while holding
direct's starting ternary forward model fixed.

## Depth sweep

| Depth | Final loss | PPL | Gain vs D | Recovery vs d=1 | Q9-code survival @100 | @300 | @900 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.03 | 5.4948 | 243.43 | 0.1188 | 16.38% | 55.28% | 51.82% | 51.76% |
| 0.25 | 4.9262 | 137.85 | 0.6875 | 94.79% | 95.82% | 89.39% | 86.48% |
| 0.50 | 4.8850 | 132.29 | 0.7286 | 100.47% | 99.49% | 98.30% | 97.32% |
| 0.75 | 4.8876 | 132.64 | 0.7260 | 100.11% | 99.84% | 99.42% | 99.13% |
| 1.00 | 4.8884 | 132.74 | 0.7252 | 100.00% | 99.92% | 99.64% | 99.45% |

Best point in this discrete sweep is **d=0.50**, loss **4.8850**.
It is only **0.0034 nats** better
than the d=1.00 prototype, so the curve is effectively flat from about d=0.5
through the prototype.

The main transition occurs much earlier:

- d=0.03: **16.38%** recovery;
- d=0.25: **94.79%** recovery;
- d=0.50: **100.47%** recovery.

This is a **threshold-like / saturating** pattern rather than "deeper is always
better."

## Survival dynamics

Q9-code survival on M strongly tracks the depth transition.

At continuation step 900 under the learned current Q3 scale:

- d=0.03: **51.76%**
- d=0.25: **86.48%**
- d=0.50: **97.32%**
- d=0.75: **99.13%**
- d=1.00: **99.45%**

Using the fixed original alpha0 gives essentially the same values; at step 900
the largest current-vs-fixed discrepancy among these arms is tiny. Therefore
the survival pattern is primarily a master-position effect rather than an
artifact of learned scale drift.

At step 900, target-zero vs target-nonzero survival is also similar within each
depth:

- d=0.03: target-zero **51.43%**, target-nonzero **52.09%**
- d=0.25: target-zero **86.98%**, target-nonzero **85.97%**
- d=0.50: target-zero **97.34%**, target-nonzero **97.29%**
- d=0.75: target-zero **99.01%**, target-nonzero **99.24%**
- d=1.00: target-zero **99.23%**, target-nonzero **99.67%**

So the large aggregate depth effect is not obviously driven by only one target
code class.

## Firmness without Q9 choices

| Arm | Construction | Final loss | Gain vs D |
|---|---|---:|---:|
| D | direct masters | 5.6136 | — |
| B1 | direct codes prototyped on Q9 mask M | 5.6403 | -0.0267 |
| B2 | direct codes prototyped on D's own changed set | 5.6195 | -0.0059 |

Both firmness-only controls are slightly worse than direct.

D's own changed set is **4.57%** of
quantized weights. Its Jaccard overlap with the Q9 disagreement mask is
**38.64%**.

## Interpretation

v13 sharpens v12:

> Q9's useful position/code choices need a **moderate interior commitment**, not
> merely a threshold crossing. Moving from depth 0.03 to 0.25 recovers almost
> all of the prototype benefit, and by depth 0.5 the effect has saturated.
> The corresponding Q9-code survival rises from roughly 52% to 97% at the end
> of continuation.

The survival result supports a concrete dynamical explanation: shallow
placements lose many of the Q9-selected assignments during later Q3 training,
while moderately deep placements preserve them long enough for the trainability
benefit to emerge.

However, survival should be described as strongly associated with the benefit,
not as independently proven to be the sole causal mediator.

B1/B2 show that "firmness" is not a generic trick for direct Q3 decisions.
Snapping direct's own assignments to prototypes does not help. Firmness matters
when attached to the **specific assignments selected by Q9**.

## Scope / stopping rule

This is seed 1729 only. The pattern is exceptionally clear and therefore meets
the preregistered criterion under which replication on orders 271828 and 424242
would be scientifically justified.

That replication is **optional**, because the v12 three-order mechanism result
already met the project's stop condition. Do not treat v13 replication as
required to support the canonical v12 mechanism claim.
