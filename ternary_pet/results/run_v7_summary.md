# Run v7 summary — equal-compute preparation geometry

**HF Job:** `6ac58e76fbc85ba6823bad78`  
**Pinned code:** `3aa494a4c8418052ed9e13e0de4f97c692a29fc7`

## Step-300 equal-compute result

All three preparations used the same first 300 chunks, then were projected
through the same original Q3 scales and scored on the same diagnostic set.

| Preparation | Native loss | Fixed-Q3 loss @300 ↓ |
|---|---:|---:|
| Direct Q3 | 6.5528 | **6.5589** |
| Q9 | **4.9320** in Q9 | 9.2264 |
| FP32/unquantized | **3.2053** native | 14.7346 |

Q9's immediate Q3 checkpoint is **2.6676 nats/token worse** than direct Q3@300.
FP32 is **8.1758 nats/token worse** than direct Q3@300.

Therefore the staged benefit is **not a better immediate ternary entry state**.

## Final result after identical 900-step Q3 continuation

All three arms then used original Q3 scales, fresh Adam, the same LR, and the
same remaining 900 chunks.

| Path | Final loss ↓ | PPL ↓ | Top-1 ↑ | KL ↓ |
|---|---:|---:|---:|---:|
| Q3 300 -> Q3 900 | 5.8724 | 355.09 | 25.73% | 2.951 |
| **Q9 300 -> Q3 900** | **5.1938** | **180.15** | **33.79%** | **2.256** |
| FP32 300 -> Q3 900 | 6.0311 | 416.19 | 23.66% | 3.124 |

Despite starting Q3 much worse at step 300, Q9 finishes **0.6786 nats/token
better** than direct, with **49.27% lower PPL** and **+8.06 pp** teacher top-1.

The FP32 warmup finishes worse than direct Q3, so generic less-constrained
warmup is not sufficient.

## Mechanistic diagnostics

Future Q3 assignments changed by step 300:

- direct Q3: **0.6933%**
- Q9: **0.6755%**
- FP32: **0.5809%**

Q3 and Q9 move nearly the same number of codes, but mostly different positions:

- Q3-vs-Q9 changed-set Jaccard: **19.64%**
- pairwise Q3 Hamming distance: **0.9194%**
- when both change the same weight, they choose the same final ternary code:
  **100%**

Global threshold-margin distributions are essentially identical. Fraction within
0.02 normalized distance of a Q3 threshold:

- initial: **3.5635%**
- Q3@300: **3.5646%**
- Q9@300: **3.5638%**
- FP32@300: **3.5636%**

So a simple "Q9 leaves more weights near thresholds" explanation is not
supported.

## Interpretation

v7 strongly favors a **trainability / weight-selection geometry** explanation:

> Q9 forward constraints cause a different subset of continuous FP32 masters to
> be repositioned. Their immediate Q3 projection is worse than direct Q3@300,
> but those masters are substantially more productive during the following Q3
> optimization.

This falsifies the simple "better ternary entry state" explanation and the
simple "any high-precision warmup works" explanation.

The strongest remaining clue is **which specific weights Q9 selects**, not how
many weights move or their global distance to Q3 thresholds.

Free-running generations remain poor; this is a training-geometry result, not a
usable ternary assistant.
