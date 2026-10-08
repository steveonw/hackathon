# Stepping Down to Ternary: A 9-State Stage Makes Ternary QAT Train Better

*steveonw · October 2026 · SmolLM2-360M-Instruct · all code, raw results and job IDs in this repository*

---

## 1. Summary

**Question.** When converting a pretrained language model to ternary weights
({−1, 0, +1}) with quantization-aware training (QAT), does it help to first train
for a while with a 9-state weight grid and then switch to ternary, compared with
training in ternary from the start for the same number of steps?

**Answer, in this setup: yes, by a large and replicated margin, and we can say
where the benefit comes from.**

1. **The effect.** With the same data, the same 1,200 updates and the same tuned
   learning-rate schedule, 300 steps at 9 states followed by 900 at ternary ends
   with **~49% lower perplexity** than 1,200 steps of direct ternary training
   (0.67 nats/token lower held-out loss; 3/3 training orders).
2. **It is trainability, not a better starting point.** At the moment of the
   switch, the 9-state-prepared model is a *worse* ternary model than direct
   training at the same step, yet it trains much faster afterwards. A
   full-precision warm-up of the same length gives no comparable benefit.
3. **It is localized.** About **96%** of the benefit is carried by the **~6.3%**
   of weights where the two methods pick different ternary codes after 300 steps
   (causal swap test, 3/3 orders).
4. **What matters on those weights** is *which* code the 9-state phase chose,
   committed at least moderately firmly. Its exact continuous values are not
   needed (a clean standard value works slightly better), while placing the same
   code barely across the threshold recovers only ~18%. The same number of code
   changes at random positions makes things *worse*.

**Scope.** One 360M model, WikiText-2 only, short runs (1,200 steps, ~150k
tokens). The models remain far from usable as assistants. Progressive
quantization in general is not new (§3); the contribution here is the
ternary-specific result with matched controls and the causal localization of
the mechanism.

**Current generalization phase.** G1 has begun on
`ibm-granite/granite-4.0-350m`. The architecture/freezing smoke passed, but
the inherited FP16 path was numerically invalid even for the unquantized source.
A preregistered precision diagnostic showed FP32 CE **3.2354**, BF16 CE
**3.2406**, and FP16 non-finite on the same probe; one-step Q3 and Q9 training
were fully finite under BF16. BF16 is therefore locked as Granite's compute
path. G1-1 then selected **constant LR 1e-4** by direct-only validation
(5.8128 loss versus 6.0827 and 6.7402 for the higher-LR schedules), before any
Granite Q9 result existed. G1-2, the first scientific Granite D-vs-S staging
gate, is now active. There is **not yet a completed Granite Q9-vs-direct
scientific result**.

---

## 2. Results at a glance

All numbers are held-out WikiText-2 (fixed 8,192-token slice). The original
BF16 model scores perplexity **39.9**.

| # | Claim | Key numbers | Evidence | Runs |
|---|---|---|---|---|
| 1 | 9→3 beats direct ternary, tuned schedule for both | ppl ~139 vs ~273; −0.672 nats; +6.8 pp teacher top-1 | 3/3 orders | v10, v9 |
| 2 | Advantage survives tuning the baseline | Tuning direct closes ~47% of the old gap; schedule-matched 9→3 restores it | 3/3 orders | v9, v10 |
| 3 | 9→3 starts ternary *worse*, ends better | +0.51 nats worse at step 300, −0.67 nats at step 1200 | 3/3 orders | v10 (also v7) |
| 4 | Full-precision warm-up doesn't reproduce it | FP32 prep ends ≈ direct (mean +0.08 nats worse), far behind 9→3 | 3/3 orders | v7 |
| 5 | Benefit is stored in the master weights | Resetting Adam and/or scales at the switch changes ≤0.03 nats | 1 order | v6 |
| 6 | ~96% of benefit lives on ~6.3% of weights | Mask-only transfer recovers 95.5–97.1% | 3/3 orders | v11 |
| 7 | Code choice + firm placement is what matters | Prototype values: 107%; barely-across: 18%; random positions: −11% | 3/3 orders | v12 |
| 8 | "Firm enough" is reached early | Depth 0.25 → 95%, 0.5 → 100%; code survival tracks it | 1 order | v13 |
| 9 | Firmness alone doesn't help direct | Prototyping direct's own codes: −0.03 / −0.01 nats | 1 order | v13 |
| 10 | Advantage shrinks with longer training (old schedule) | 6,000 steps: ppl 70 vs 85, gap 0.20 nats | 1 order | v5 |
| 11 | Granite G1 engineering gate only | BF16 source CE 3.2406 vs FP32 3.2354; FP16 non-finite; Q3/Q9 BF16 steps finite | 1 technical smoke, no scientific order | G1-0/G1-0b |
| 12 | Granite direct schedule frozen before Q9 | const 1e-4 val loss 5.8128 vs 6.0827 / 6.7402 for higher-LR schedules | 1 calibration order; held-out test unused | G1-1 |

---

## 3. Prior work and what this adds

Lowering precision in stages is an established idea:

- **Progressive quantization for CNNs.** Zhuang et al., *Towards Effective
  Low-bitwidth Convolutional Neural Networks* (CVPR 2018,
  [arXiv:1711.00205](https://arxiv.org/abs/1711.00205)), trained networks by
  stepping bit-width down in stages and found this beat quantizing to the target
  bit-width directly.
- **Progressive QAT for LLMs.** *Bit-by-Bit: Progressive QAT Strategy with
  Outlier Channel Splitting for Stable Low-Bit LLMs*
  ([arXiv:2604.07888](https://arxiv.org/abs/2604.07888)) reduces LLM weights
  8 → 4 → 2 bits stage by stage, with nested grids for multi-precision
  deployment. It does not target ternary.
- **Ternary / 1.58-bit LLMs.** BitNet b1.58
  ([arXiv:2402.17764](https://arxiv.org/abs/2402.17764)) established ternary
  LLMs trained with full-precision latent weights. *Continual Quantization-Aware
  Pre-Training* ([arXiv:2502.11895](https://arxiv.org/abs/2502.11895)) studies
  when to switch from 16-bit to 1.58-bit pre-training. **ParetoQ**
  ([arXiv:2502.02631](https://arxiv.org/abs/2502.02631)) reports a qualitative
  change in learning behaviour below ~3 bits, consistent with the 9→3 step being
  the hard one here.

**What this project adds** (to our knowledge, from a limited literature search):

1. A matched comparison of a **9-state → ternary** schedule against direct
   ternary QAT on a pretrained LLM, with a tuned baseline, an equal schedule,
   and a full-precision warm-up control.
2. Evidence that the benefit is **trainability** rather than a better starting
   ternary model.
3. **Causal localization** by intervention: the benefit is carried by a small,
   specific set of ternary assignments, and requires those assignments to be
   committed with some margin.

---

## 4. Method (canonical protocol, v10–v13)

**Model and data.** `HuggingFaceTB/SmolLM2-360M-Instruct`, BF16-rounded before
every treatment. Training on 1,200 WikiText-2 chunks of 128 tokens, one chunk per
step, in a seeded shuffled order (orders 1729, 271828, 424242). Both arms of
every comparison see the same chunks in the same order. Evaluation on a fixed
8,192-token WikiText-2 test slice.

**What is quantized.** All linear layers except `lm_head`. Embeddings, norms and
`lm_head` stay full precision and are **frozen**, so recovery must come from the
quantized weights.

**Quantizer.** One persistent FP32 master weight per quantized parameter;
straight-through estimator. Learnable per-output-row scale α, initialized at
1.5 × mean|w| of the row.
- Ternary: values {0, ±⅔α}, zero threshold at ⅓α (≈32% zeros at init).
- 9-state: values {0, ±¼α, ±½α, ±¾α, ±α}.

**Training.** AdamW (β = 0.9/0.95, no weight decay), gradient clipping 1.0.
Loss = 0.35 × cross-entropy + 0.65 × KL to the untouched original model.
Learning rate: 100-step linear warmup to 1e-3, cosine decay to 1e-4 at step
1,200 (selected for *direct* ternary in v9; the 9→3 arm received no tuning of its
own).

**Treatments.**
- **Direct:** 1,200 ternary steps.
- **9→3:** 300 steps with the 9-state forward pass, then 900 ternary steps. At
  the switch, the master weights are kept, the scales are reset to their
  original ternary values, and Adam is reset. The LR schedule continues without
  restarting.

**Mechanism experiments (v11–v13)** rebuild both step-300 master states inside a
single run, construct hybrid models, and continue each for the same 900 ternary
steps. Wherever two arms are meant to start from the same ternary model, the
scripts **assert zero code differences and identical pre-continuation loss**
before training; all such checks passed.

---

## 5. Findings

### 5.1 The effect (v9, v10)

| Order | Tuned direct ppl | 9→3 ppl | Loss gain (nats) |
|---|---|---|---|
| 1729 | 269.3 | **134.4** | 0.695 |
| 271828 | 276.7 | **141.3** | 0.672 |
| 424242 | 272.2 | **142.1** | 0.651 |
| **Mean** | 272.7 | **139.3** | **0.672** |

Mean teacher top-1 agreement +6.8 points; mean KL to the original −0.68.

History matters here. With the earlier constant LR of 1e-4, the gap was ~0.69–0.77
nats (v4b, v7). Tuning direct training's schedule (v9) closed about half of
that gap, but giving 9→3 the *same* schedule (v10) restored it to 0.67. So the
advantage is not an artifact of a weak baseline.

### 5.2 Trainability, not a better entry point (v7, v10)

At step 300, both preparations are projected to ternary with the same scales and
scored on the same data. The 9-state-prepared model is **worse** in every order
(+0.51 nats under the tuned schedule; +2.59 under the old one), yet it finishes
**0.67–0.69 nats better** after identical continuation. In v7's matched
training batches (old schedule, order 1729) it caught up with direct within
about 100 ternary steps and then pulled steadily ahead.

A **full-precision warm-up** for the same 300 steps produces an excellent
full-precision model but no comparable ternary benefit: it finishes about level
with direct (mean 0.08 nats worse; slightly better in one order), far behind
9→3. The coarse intermediate grid, not "adapt first", is what helps.

v6 (one order) showed the benefit is carried by the **master weights**: keeping
or resetting Adam state and quantizer scales at the switch changes the result by
at most 0.03 nats against a ~0.7-nat effect.

### 5.3 Localization (v11)

At step 300, the two preparations disagree on the ternary code of **6.31%** of
weights (range 6.18–6.46%). Swapping 9→3's master values onto direct's model
**only at those positions** recovers **96.4%** of the full advantage
(95.5 / 97.1 / 96.6%). Swapping the other ~94% of weights alone gives a small
gain (0.11 nats) that becomes negligible (0.025 nats) once the disagreement
positions are swapped.

Both preparations change a similar number of codes, but mostly *different*
ones: their changed sets overlap with Jaccard ≈ 19% in all three orders.

### 5.4 What matters on those positions (v12, v13)

Holding the starting ternary model fixed and varying only where each master
weight sits inside its code's region:

| On the 6.3% positions | Share of gain (3 orders) |
|---|---|
| 9→3's exact master values | 100% (reference) |
| 9→3's code, at the standard value for that code | **107%** (106.5–107.7) |
| 9→3's code, placed just across the threshold | 18% (16.4–20.5) |
| Same kinds of code changes at random positions | −11% (worse than direct) |

The depth sweep (v13, one order) shows the benefit switches on quickly and then
saturates: depth 0.03 → 16%, 0.25 → 95%, 0.5 → 100%, with no further gain
deeper. The fraction of those positions still holding the 9-state-selected code
after 900 ternary steps tracks this closely (52% → 86% → 97% → 99%), consistent
with shallow placements being undone during later training. Survival is
associated with the benefit but not proven to be its only cause: at the
shallowest depth about half the codes survive but only 16% of the gain appears.

**Firmness alone does not help.** Snapping *direct's own* codes to their
standard values, either on the same positions or on direct's own recently
changed weights, makes results slightly worse (−0.03 and −0.01 nats).

### 5.5 Mechanism in one sentence

The 9-state phase selects a specific ~6% of ternary assignments that differ from
the ones direct training makes; committing those assignments with moderate
margin keeps them from being undone, and that accounts for essentially the whole
advantage, while the same commitment applied to other choices does not.

---

## 6. Limitations

- **One completed scientific model family and one dataset.** SmolLM2-360M,
  WikiText-2 for both training and evaluation. Granite-350M has passed the engineering/precision gate; direct-only G1-1
  selected constant 1e-4 and G1-2 is now testing D vs Q9→Q3. No completed
  cross-family result exists yet.
- **Short training.** 1,200 steps of single 128-token chunks (~150k tokens). In
  the one longer run (v5: 6,000 steps, old constant-LR schedule, one order) the
  gap **shrank** from ~0.7 to 0.20 nats (ppl 70 vs 85). Whether it persists
  asymptotically under the tuned schedule is unknown.
- **Usability.** Even the best models (ppl ~70–140) produce repetitive
  WikiText-style fragments and do not follow prompts.
- **Baselines.** The schedule was tuned for direct training only; 9→3 received
  no tuning, which should if anything understate its advantage. Other strong
  direct-ternary recipes (e.g., different scale schemes, longer warmup) were not
  tried.
- **Seeds.** Claims 5, 8 and 9 rest on one training order.
- **Design details.** Embeddings and `lm_head` are full precision; the 9-state
  and ternary grids are not nested (9-state ±α maps to ternary ±⅔α).

---

## 7. Possible next steps

- **Generalization (active):** complete the preregistered Granite-350M G1
  sequence: constant 1e-4 is now frozen from direct-only calibration; complete
  the active one-order D/S gate, then run the compressed causal mechanism
  battery only if staging is positive.
- **A cheaper recipe:** if the useful ~6% of assignments could be predicted
  without a full 9-state phase, the benefit could be had at lower cost.
- **Level count:** test other intermediate grids (e.g., 5 or 7 states, or a
  nested 9→3 grid).

---

## Appendix A. How the project got here (v1–v8)

The full chronological record, including all dead ends, is in
[`SHAREABLE_RESEARCH_REPORT.md`](SHAREABLE_RESEARCH_REPORT.md) and `results/`.
Short version:

- **v1–v3 (negative, due to setup problems):** naive 27→9→3 staging and
  transition schedules showed no benefit. Diagnosis: the quantizer zeroed ~79% of
  weights (scale initialized from the row maximum), the LR of 2e-5 was too low
  for any ternary code to change, and v2's hard commits between stages froze
  learning. With no training between stages, nested 27→9→3 rounding gives exactly
  the same ternary model as direct rounding, so any benefit must come from
  adaptation.
- **v4 (first positive result):** fixed quantizer (~32% zeros), LR 1e-4,
  persistent FP32 masters, code-flip logging. 9→3 beat direct at equal steps;
  extra direct steps (1,200 vs 900) did not help direct.
- **v4b (replication):** shuffled data orders; 9→3 won 3/3 (mean −0.77 nats).
- **v5 (scale-up):** 5× more training; advantage persisted but shrank (§6).
- **v6:** benefit carried by master weights, not Adam or scales.
- **v7:** equal-compute comparison; trainability effect and full-precision
  control (§5.2).
- **v8 (negative):** tested whether 9-state training pre-pushes weights toward
  their future thresholds. Not supported: each arm looked "aligned" only on its
  own future-flip set, the pattern expected from selection effects.
- **v9–v13:** §5.

## Appendix B. Reproducibility

Every run has a pinned commit, a Hugging Face Jobs ID and a raw JSON result in
`results/` or `replications/`; aggregate summaries for each replicated claim are
in `replications/*_aggregate_summary.md`. Runs used single T4 or A10G GPUs.

## Acknowledgments

Experiments were designed, run and reviewed with the help of AI assistants;
the original hypothesis and research direction are the author's.
