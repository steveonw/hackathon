# Run v4 summary — persistent-shadow 9 -> 3

**Hugging Face Job:** `6ac539b5404719ba376621a2`  
**Model:** `HuggingFaceTB/SmolLM2-360M-Instruct`  
**Hardware:** T4 small  
**Status:** completed  
**Chosen LR:** `1e-4`

## Main result

For the first time in this project, the staged route beat direct ternary on the
fixed held-out evaluator while the discrete quantized codes were demonstrably
changing.

| Variant | Total steps | Ternary steps | PPL ↓ | Top-1 agreement ↑ | KL ↓ |
|---|---:|---:|---:|---:|---:|
| BF16 source | — | — | 39.86 | 99.65% | ~0 |
| Direct ternary | 1200 | 1200 | 666.66 | 22.84% | 3.546 |
| **9 -> 3 staged** | **1200** | **900** | **286.97** | **30.49%** | **2.716** |
| Direct ternary | 900 | 900 | 670.74 | 22.96% | 3.508 |

Against the clean equal-total-compute direct-1200 control, 9 -> 3 achieved:

- **56.95% lower perplexity**
- **+7.65 percentage points top-1 teacher agreement**
- **23.39% lower KL divergence**

## The transition shock was dramatically smaller

Direct ternary began at mixed training loss **17.312**.

After 300 steps in the 9-state representation, the first fully ternary step was
**6.594** — about a **61.9% smaller ternary-entry shock**.

The 9-state phase itself improved from 13.329 -> 2.991 before the switch.

## The codes were really moving

This run added exact discrete code-flip diagnostics, so the result cannot be
explained by a completely frozen quantized backbone.

The LR sweep showed increasing code movement as LR increased, and `1e-4`
gave the best short validation result.

At the end of direct-1200, about **1.67%** of ternary codes differed from their
stage-start assignments. During the staged run, the 9-state phase moved about
**2.54%** of its codes relative to its stage start, and the following ternary
phase ended with about **1.22%** of ternary codes different from its ternary
stage start.

The useful interpretation is not "more flips is always better." The staged
model made substantial rearrangements while it still had nine states, then
needed fewer ternary-state changes to reach a much better held-out model.

## The zero-fraction problem was fixed

The new absmean-based initialization produced approximately:

- -1: 34.1%
- 0: 31.7%
- +1: 34.1%

instead of v3's ~79% zeros.

That removed a major confound from the earlier experiments.

## What this does and does not show

This is the first positive evidence for the staircase hypothesis in this
project:

> allowing a pretrained model to reorganize in a ~3.17-bit / 9-state space
> before forcing it into ternary can improve the final ternary model.

The cleanest evidence is staged 300+900 vs direct 1200 because both consume the
same 1200 training chunks in the same global order and differ primarily in the
representation used for the first 300 updates.

However, this is still **one seed**, and the generated chat samples remain
degenerate/repetitive. The staged model is substantially less damaged, not yet
a usable converted assistant.

## Next replication

Before changing the recipe again, repeat the exact v4 protocol with at least
two additional seeds. If the 9 -> 3 advantage repeats, then increase the
training/data budget rather than introducing another schedule variable.
