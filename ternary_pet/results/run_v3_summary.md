# Run v3 summary — transition schedules and the "step up, then down" hypothesis

**Hugging Face Job:** `6ac52df9404719ba37661fa1`  
**Model:** `HuggingFaceTB/SmolLM2-360M-Instruct`  
**Hardware:** Tesla T4 (`t4-small`)  
**Wall-clock runtime:** ~22m 35s  
**Status:** completed

## What v3 tested

Every path starts from the same BF16-rounded checkpoint. Trainable master/shadow
weights are then kept in FP32 (24-bit significand precision), with a learnable
per-output-row ternary scale and teacher-logit distillation.

This is the useful interpretation of "16 -> 24 -> 3": casting BF16 to FP32 does
not recover information that BF16 already lost, but gradient updates performed
in FP32 can create new fine-grained weight positions before the ternary switch.

Four schedules were compared:

| Variant | Schedule |
|---|---|
| Direct | 600 fully ternary QAT steps |
| Soft | 200-step 0->1 quantization-strength ramp + 400 ternary steps |
| Up/down, equal total | 200 FP32-forward adaptation + 400 ternary steps |
| Up/down, equal final | 200 FP32-forward adaptation + 600 ternary steps |

## Held-out results

| Variant | PPL ↓ | Top-1 agreement ↑ | KL to teacher ↓ |
|---|---:|---:|---:|
| BF16 source | 39.86 | 99.646% | 0.00004 |
| Direct ternary, 600 | **1978.94** | 7.690% | **4.539** |
| Soft 200 + ternary 400 | 2116.66 | 7.263% | 4.589 |
| Up/down 200 + ternary 400 | 2725.34 | 6.287% | 4.806 |
| Up/down 200 + ternary 600 | 2023.02 | **7.703%** | 4.563 |

The direct path wins narrowly on perplexity and KL. The up/down equal-final path
wins by a tiny amount on top-1 teacher agreement. Given a single run/seed, these
two should be treated as effectively tied rather than as evidence of superiority.

## Transition-shock result

The direct ternary path began at mixed training loss **15.9242**.

After 200 full-precision adaptation steps, the up/down path hit **15.2883** on
its first ternary step — about a **4% smaller immediate shock**.

So the higher-precision preparation phase did measurably change the landing.
It just did not produce a clear held-out quality win after recovery.

## "Death by a thousand cuts" result

The soft transition nearly eliminated the first-step cliff:

- first soft step: loss 1.129, teacher KL 0.003;
- around lambda=0.85: loss rose to 12.616;
- at full ternary after the ramp: loss 9.780;
- end of training: loss 5.501.

That is a concrete example of the user's hypothesis: spreading the transition
removes the single catastrophic discontinuity, but the damage can accumulate
as quantization strength increases. On this run, smoothing the journey did not
improve the final held-out destination.

## Major improvement over v2

v3's learnable ternary scale and longer training changed direct ternary
perplexity from catastrophic territory to **~1,979** on this evaluator.
It is still far from usable (the sample generations remain repetitive), but it
confirms that quantizer/training design matters far more than the simple state
count alone.

## v3 conclusion

The literal hypothesis "one high-precision adaptation step, then ternary is
better" is **not supported yet**, but it also did not fail in the same way as
the 27->9->3 hard staircase.

With equal final ternary recovery budget, the up/down path essentially converged
back to the direct-QAT result and slightly exceeded it on token agreement.
The useful clue is the ~4% smaller switch shock.

A stronger next test is therefore not another hard staircase. It is a hybrid:

```text
BF16 source
  -> FP32 master adaptation
  -> soft ternary phase-in
  -> full ternary QAT
```

That directly combines the user's high-precision preparation idea with the
research-supported gradual transition, while preserving FP32 shadow weights
through the entire process.
