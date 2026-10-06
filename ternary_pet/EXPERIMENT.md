# Experiment protocol

## Question

Does the path into ternary space affect how much behavior a pretrained model
retains?

### v1 — independent 27 -> 9 -> 3

The first experiment used independently rescaled 27-, 9-, and 3-level weight
grids with a tiny recovery budget. It did not show a staged advantage and was
numerically unstable near ternary.

Remote run: \`6ac52475404719ba37661c8b\`.

### v2 — strict nested balanced-ternary ancestry

v2 used a literal balanced-ternary hierarchy with one inherited scale:

- 27 child states,
- 9 parent states,
- 3 ternary grandparent states.

Raw direct and nested projections ended identically, as expected for
deterministic nested projection without adaptation. Recovery became stable, but
direct ternary QAT still beat the staged route.

Remote run: \`6ac527e5404719ba37661dc9\`.


## v3 — transition schedules and high-precision preparation

v3 tests whether the path into ternary space matters more than a hard sequence
of discrete codebooks. All variants start from the same BF16-rounded checkpoint
and keep trainable FP32 master weights. The ternary forward quantizer uses a
learnable per-output-row scale, and recovery uses WikiText-2 with CE + teacher
KL distillation.

The schedules are:

- direct: 600 fully ternary steps;
- soft: a 200-step quantization-strength ramp followed by 400 ternary steps;
- up/down equal-total: 200 full-precision adaptation steps followed by 400
  ternary steps;
- up/down equal-final: 200 full-precision adaptation steps followed by 600
  ternary steps.

The up/down treatment is the practical version of a proposed 16->24->3 idea.
FP32 has a 24-bit significand; merely upcasting BF16 cannot recover discarded
bits, so the experimental variable is **adaptation while the master weights are
stored and updated at FP32 precision** before the ternary switch.

Remote run: Hugging Face Job `6ac52df9404719ba37661fa1`, T4 small.
See `results/run_v3_summary.md` for the outcome.


## v4 — persistent-shadow 9 -> 3 with flip-rate diagnostics

v4 is designed to answer whether staging helps **once the discrete weights are
actually moving**.

Changes from earlier runs:

- one FP32 shadow/master weight tensor is kept for the whole run;
- no hard commit at the 9 -> 3 transition;
- Adam optimizer state is preserved across the transition;
- non-quantized parameters are frozen;
- the ternary scale is initialized from rowwise absmean and remains learnable;
- actual discrete code flips are measured every 50 steps;
- one fixed WikiText-2 test slice (8192 tokens) is used.

Before the main comparison, a short direct-ternary LR sweep tests
`2e-5`, `5e-5`, and `1e-4`; the lowest held-out diagnostic loss is selected.

Main conditions:

1. direct ternary for 1200 steps;
2. 9-state QAT for 300 steps -> ternary QAT for 900 steps, with the same FP32
   shadows and optimizer state;
3. direct ternary for 900 steps.

This lets condition 2 be compared both against equal total compute (1200 steps)
and against an equal final ternary-stage budget (900 steps).

Remote run: Hugging Face Job `6ac539b5404719ba376621a2`, T4 small,
45-minute hard timeout.


## v5 — 5x training-volume test

v5 keeps the replicated v4/v4b recipe fixed and increases only training volume.

- model: `HuggingFaceTB/SmolLM2-360M-Instruct`
- fixed evaluator: WikiText-2 test, 8192 tokens
- training set: 6000 unique 128-token chunks, shuffled with seed 1729
- LR: `1e-4`
- non-quantized parameters frozen
- persistent FP32 shadow weights
- optimizer state preserved through the 9 -> 3 transition
- absmean-initialized learnable quantizer scale
- flip-rate logging every 250 steps

Equal-total-compute pair:

1. direct: 6000 ternary steps;
2. staged: 1500 9-state steps -> 4500 ternary steps.

Each treatment therefore processes ~768k training tokens. The successful 25/75
stage split from v4/v4b is unchanged.

Pinned Git commit:
`23157f7ce03f6bbbf944806574b18c133843f049`

Hugging Face Job:
`6ac56d99fbc85ba6823ba03f`

Hardware: A10G small. Hard timeout: 75 minutes.


### v5 outcome

Job `6ac56d99fbc85ba6823ba03f` completed successfully.

Held-out results:

- BF16 source: loss 3.6854 / PPL 39.86;
- direct 6000@3: loss 4.4463 / PPL 85.31 / top-1 41.44% / KL 1.522;
- staged 1500@9 -> 4500@3: loss 4.2493 / PPL 70.05 / top-1 43.64% / KL 1.330.

The staged treatment therefore retained a **0.197 nats/token** advantage and
17.88% lower PPL at 5x training volume, but the loss gap was about **74% smaller**
than the 0.765-nat mean seen in v4/v4b.

The immediate ternary-entry loss remained ~50.4% smaller after 9-state prep,
showing that transition smoothing is highly reproducible even as the final
quality gap narrows.

Interpretation: v5 supports a large early optimization/head-start effect with a
smaller residual advantage at 6000 updates. It does not establish a permanently
better asymptotic ternary basin.


## v6 — causal dissection of the 9 -> 3 transition

v6 returns to the v4-scale protocol to locate where the staged advantage is
stored rather than spending more compute on another scale-up.

Shared setup:

- model: \`HuggingFaceTB/SmolLM2-360M-Instruct\`
- seed/order: 1729
- 1200 shuffled 128-token training chunks
- fixed 8192-token WikiText-2 evaluator
- LR: \`1e-4\`
- frozen non-quantized parameters
- persistent FP32 shadow/master weights
- absmean-initialized learnable rowwise scales
- CE + teacher-KL objective

One common 9-state checkpoint is trained for 300 steps. That exact checkpoint
then branches into four 900-step ternary continuations:

1. \`carry_all\`: prepared masters + prepared scales + full Adam state;
2. \`reset_adam\`: prepared masters + prepared scales + fresh Adam;
3. \`masters_only\`: prepared masters + initial scales + fresh Adam;
4. \`masters_plus_weight_adam\`: prepared masters + initial scales + Adam state
   retained only for master weights; scale optimizer state is reset.

A fresh \`direct_1200\` ternary run remains the matched control.

### New transition diagnostics

Before any ternary update, the shared prepared checkpoint is evaluated on the
same fixed diagnostic chunks under Q9 and Q3. This gives a clean same-data
quantizer-switch penalty.

v6 also compares initial ternary codes against the prepared masters projected
to ternary in two ways:

- using the learned prepared scales;
- using the original initial scales.

The second comparison isolates how often 9-state training moved FP32 masters
across *future ternary decision boundaries* without allowing scale changes to
explain the crossing.

Pinned Git commit:
\`ad5376d4db03c0e467a47353293220474d6a8b42\`

Hugging Face Job:
\`6ac5816ffbc85ba6823ba8ec\`

Hardware: A10G small. Hard timeout: 50 minutes.
