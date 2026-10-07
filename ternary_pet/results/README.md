# Results

Raw machine-readable and summarized results from each remote run belong here.

## Run registry

| Run | Model | Hardware | Status | Purpose |
|---|---|---|---|---|
| v1 / `6ac52475404719ba37661c8b` | SmolLM2-360M-Instruct | T4 small | completed | naive staged PTQ/QAT |
| v2 / `6ac527e5404719ba37661dc9` | SmolLM2-360M-Instruct | T4 small | completed | strict nested ancestry |
| v3 / `6ac52df9404719ba37661fa1` | SmolLM2-360M-Instruct | T4 small | completed | transition schedules |
| v4 / `6ac539b5404719ba376621a2` | SmolLM2-360M-Instruct | T4 small | completed | persistent-shadow 9->3 |
| v4b seed 1729 / `6ac546b5fbc85ba6823b8941` | SmolLM2-360M-Instruct | T4 small | completed | shuffled-order confirmation |
| v4b seed 271828 / `6ac546bafbc85ba6823b8946` | SmolLM2-360M-Instruct | T4 small | completed | shuffled-order confirmation |
| v5 / `6ac56d99fbc85ba6823ba03f` | SmolLM2-360M-Instruct | A10G small | completed | 5x training-volume test |
| v6 / `6ac5816ffbc85ba6823ba8ec` | SmolLM2-360M-Instruct | A10G small | completed | causal transition ablation |
| v7 / `6ac58e76fbc85ba6823bad78` | SmolLM2-360M-Instruct | A10G small | completed | equal-compute Q3/Q9/FP32 geometry |

## Current headline

v7 resolves the main v6 ambiguity.

At equal 300-step compute, Q9 projects to a **worse** Q3 checkpoint than direct
(9.226 vs 6.559 loss), but after the same fresh-Adam 900-step Q3 continuation it
finishes much better (5.194 vs 5.872; 180.15 vs 355.09 PPL).

FP32 warmup finishes worse than direct (6.031 loss / 416.19 PPL).

Q3 and Q9 change similar fractions of future Q3 codes but only **19.64% Jaccard**
overlap in which positions changed. The leading mechanism is now a
Q9-specific **trainability / weight-selection geometry** effect.

See `run_v7_summary.md`.