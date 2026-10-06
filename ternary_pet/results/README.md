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

## Current headline

At 1200 updates, staged 9->3 beat direct ternary by a mean **0.765 nats/token**
across three paired training orders.

At 6000 updates, staged still won, but by only **0.197 nats/token**:

- direct: PPL **85.31**, top-1 **41.44%**, KL **1.522**
- staged: PPL **70.05**, top-1 **43.64%**, KL **1.330**

This is evidence that much of the earlier staged advantage is an early
optimization/head-start effect. A smaller residual advantage remains at v5.

See `run_v5_summary.md` for interpretation.
