# G1-1 — Granite 4.0 350M direct-Q3 schedule calibration

Job: `6ac6f18ae7a0dae8a2780246`  
Pinned code: `3cb0153be80d5e9fbe112460bde7bc26398d851f`  
Hardware: A10G-small  
Seed/order: 1729  
Status: **complete — direct schedule frozen for G1-2**

This was a direct-Q3-only validation screen. No Q9 arm was present and the
held-out test evaluator was not used.

## Validation result after 300 identical training chunks

| Candidate | Validation loss | PPL | Teacher top-1 | KL | Projected Q3 codes changed vs source |
|---|---:|---:|---:|---:|---:|
| **constant 1e-4** | **5.81281** | **334.56** | **22.98%** | **3.01163** | 5.323% |
| warm100 → 3e-4, cosine → 1e-4 | 6.08270 | 438.21 | 21.32% | 3.30245 | 11.616% |
| warm100 → 1e-3, cosine → 1e-4 | 6.74016 | 845.70 | 20.54% | 3.87338 | 27.231% |

All three candidates remained numerically finite under the Granite BF16 compute
path.

## Decision

Per the preregistered lowest-validation-loss rule, **constant LR 1e-4 wins**.

That schedule is now frozen for the Granite G1 scientific comparison:

- D: 1200 direct Q3 updates at constant 1e-4;
- S: 300 Q9 updates then 900 Q3 updates, also at constant 1e-4;
- no Q9-specific tuning is allowed;
- the Q9→Q3 switch restores original Q3 scales and uses fresh Adam while
  retaining the prepared FP32 masters.

The higher-LR schedules moved substantially more projected ternary assignments
but validated worse. This is a calibration observation, not evidence about the
Q9 staging hypothesis.
