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
