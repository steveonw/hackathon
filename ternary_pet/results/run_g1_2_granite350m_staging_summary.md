# G1-2 — Granite 4.0 350M staging gate, order 1729

Job: `6ac6f487df2184ac91ac693e`  
Pinned code: `f3a1f88890d1b70264f25f1be8c4d23d84594825`  
Hardware: A10G-small  
Direct-selected LR: constant `1e-4`  
Precision: FP32 masters + BF16-rounded source + BF16 autocast/BF16 teacher  
Status: **positive one-order cross-family staging result; G1-3 gate passed**

## Final held-out endpoint

| Metric | Direct Q3 | Q9→Q3 | Q9 advantage |
|---|---:|---:|---:|
| Loss | 5.66581 | **5.53495** | **0.13086 nats/token** |
| PPL | 288.82 | **253.40** | **12.27% lower** |
| Teacher top-1 | 27.53% | **28.04%** | **+0.51 pp** |
| KL to teacher | 2.49025 | **2.38307** | **0.10718 lower** |

Source held-out reference: loss 4.08876 / PPL 59.67.

## Equal-compute step-300 diagnostic

| State | Validation loss | PPL | Teacher top-1 | KL |
|---|---:|---:|---:|---:|
| Direct Q3, fixed original Q3 scales | **5.81745** | **336.11** | **22.95%** | **3.01815** |
| Q9 native | 5.65240 | 284.97 | 24.71% | 2.79601 |
| Q9-prepared masters projected to fixed Q3 | **7.74247** | **2304.15** | **8.24%** | **4.90649** |

The Q9-prepared state is **1.92502 nats/token worse** than direct when both
are projected through the same original Q3 scales after the same 300 updates.
Despite that severe immediate disadvantage, staged Q9→Q3 finishes 0.13086
nats/token better after the later Q3 continuation.

This reproduces the key **trainability-not-entry-quality** signature previously
seen on SmolLM2.

## Step-300 projected-code geometry

- direct changed vs source Q3: **5.3242%**
- Q9-prepared changed vs source Q3: **5.2723%**
- D-vs-S projected Q3 disagreement: **6.0246%**
  - 15,035,048 / 249,561,088 targeted weight positions
- changed-set Jaccard: **27.53%**
- when both changed a source code, they selected the same final code **99.87%**
  of the time.

Global D→S disagreement transitions were almost balanced:

- -1→0: 3,760,239
- 0→-1: 3,752,016
- 0→1: 3,748,625
- 1→0: 3,767,926
- rare direct sign flips: -1→1 = 3,141; 1→-1 = 3,101.

Thus raw code-movement quantity again does not explain the effect: D and S move
nearly the same fraction of weights while selecting different positions.

## Interpretation

On this first Granite order, the staged effect generalizes across model family
with the same qualitative signature as SmolLM2:

1. Q9 is substantially worse as an immediate ternary checkpoint;
2. Q9 nevertheless creates a state that optimizes better during later Q3;
3. the D/S disagreement set is again small, about 6% of targeted weights.

The effect size is notably smaller than the canonical SmolLM2 matched-schedule
effect (~0.67 nats/token), so do not claim scale-invariant magnitude.

## Caveats

- One Granite training order only.
- The causal localization / d=0.5 / matched-random mechanism has not yet been
  tested on Granite.
- Free-running generations remain degenerate/repetitive in both arms.
- This is still WikiText-2; model-family generalization is being tested before
  dataset generalization.

## Decision

The preregistered G1-2 gate **passes**. The 0.13086-nat held-out advantage is
technically clean and positive on all tracked endpoint metrics.

Proceed to G1-3: rebuild D/S step-300 states, define the true D-vs-S disagreement
mask, and compare D, S, M-exact, true-M d=0.5, and transition/layer/source-matched
random d=0.5 under a common Q3 continuation.
