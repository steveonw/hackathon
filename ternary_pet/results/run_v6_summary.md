# Run v6 summary — where does the 9 -> 3 advantage live?

**Hugging Face Job:** `6ac5816ffbc85ba6823ba8ec`  
**Pinned code:** `ad5376d4db03c0e467a47353293220474d6a8b42`  
**Model:** `HuggingFaceTB/SmolLM2-360M-Instruct`  
**Hardware:** A10G small

## Main result

v6 gives strong causal evidence that the staged advantage is stored primarily
in the **adapted FP32 master weights**, not in learned quantizer scales or Adam
optimizer state.

| Condition | Held-out loss ↓ | PPL ↓ | Top-1 ↑ | KL ↓ |
|---|---:|---:|---:|---:|
| Direct 1200@3 | 5.8747 | 355.90 | 25.55% | 2.954 |
| Carry prepared masters + scales + full Adam | 5.2059 | 182.34 | 32.80% | 2.281 |
| Reset Adam, keep prepared masters + scales | 5.2031 | 181.83 | 33.31% | 2.275 |
| **Prepared masters only; initial scales; fresh Adam** | **5.1990** | **181.09** | 32.74% | **2.262** |
| Prepared masters + master-weight Adam; initial scales | **5.1799** | **177.67** | 32.96% | **2.252** |

Every branch that keeps the prepared FP32 masters retains essentially the full
staged advantage.

The entire four-branch loss range is only **0.026 nats/token**, while the gap
from direct to the best branch is **0.695 nats/token**. The branch spread is
only about **3.7%** of that direct-vs-staged gap.

Therefore:

- resetting Adam does **not** destroy the advantage;
- restoring the original quantizer scales does **not** destroy the advantage;
- doing both simultaneously still leaves almost the full benefit.

The small ranking among branches should not be overinterpreted. The important
result is their near-equivalence.

## Clean same-data Q9 -> Q3 transition measurement

v6 finally measures the quantizer switch on the exact same diagnostic set with
no optimizer update between evaluations.

Before Q9 training, direct Q3 diagnostic loss was:

**15.4784**

After 300 Q9 updates, the checkpoint under Q9 scored:

**4.9320**

The same prepared FP32 weights, immediately projected to Q3, scored:

**9.1771**

So the clean Q9 -> Q3 switch itself costs:

**+4.2451 nats/token**

That is a real large quantization shock.

However, the crucial result is that the prepared Q3 projection is still
**6.3013 nats/token better than the original Q3 projection before any ternary
training occurs**.

Thus the 300-step Q9 phase does not merely warm up Adam. It changes the
continuous weights into a state that is intrinsically more compatible with the
later ternary projection.

## Future ternary decision boundaries

v6 directly compares ternary assignments before and after the Q9 preparation.

Using the learned prepared scales:

**0.6769%** of eventual ternary codes change.

Using the **original initial scales held fixed**:

**0.6755%** change.

Those numbers are nearly identical. Scale learning changes the aggregate
boundary-crossing measurement by only ~0.0014 percentage points.

This is the key mechanistic result:

> Q9 training moves the FP32 master weights themselves across future ternary
> decision boundaries.

The movement is concentrated in sensitive projections. With scales held fixed:

- layer 0 `v_proj`: **3.44%** future ternary assignments changed;
- layer 0 `o_proj`: **2.95%**;
- layer 0 `q_proj`: **1.57%**;
- layer 27 `k_proj`: **1.39%**.

Only ~0.68% of all targeted codes need to cross future ternary boundaries to
produce a large downstream recovery advantage, so the location of the changes
appears more important than their raw count.

## What v6 establishes

Within this SmolLM2 / WikiText / QAT setup, the best-supported mechanism is now:

```
Q9 adaptation
    ↓
FP32 master weights move
    ↓
some weights cross future Q3 decision boundaries
    ↓
the model enters ternary training from a substantially better discrete state
```

Prepared scale values and inherited Adam moments are secondary at most under
this budget.

This materially strengthens the "reorganization before constraint" hypothesis.

## What v6 does not establish

- It does not show that nine states are uniquely optimal.
- It does not prove the effect survives asymptotically; v5 still shows direct
  ternary catching up with more training.
- It does not fix the under-tuned constant-LR direct baseline.
- It does not solve the poor free-running generation quality.
- It remains one model family and one corpus.

The next scientific question is no longer "where is the advantage stored?"
v6 answers that fairly clearly: **mostly in the prepared master weights**.

The next question is whether a stronger direct-ternary optimizer/scheduler can
erase the benefit, and whether a better-calibrated intermediate codebook
changes its size.
