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
| v7 rep seed 271828 / `6ac59807fbc85ba6823bb060` | SmolLM2-360M-Instruct | A10G small | completed | v7 training-order replication |
| v7 rep seed 424242 / `6ac59809fbc85ba6823bb063` | SmolLM2-360M-Instruct | A10G small | completed | v7 training-order replication |
| v8 / `6ac5a0b1404719ba37664653` | SmolLM2-360M-Instruct | A10G small | completed | signed pre-loading mechanism diagnostic |
| v9 / `6ac5ad77fbc85ba6823bb71c` | SmolLM2-360M-Instruct | A10G small | completed | direct-Q3 LR/schedule tuning |
| v9 tuned rep seed 271828 / `6ac5b913404719ba37664b21` | SmolLM2-360M-Instruct | A10G small | completed | tuned direct-Q3 confirmation |
| v9 tuned rep seed 424242 / `6ac5b915fbc85ba6823bba5b` | SmolLM2-360M-Instruct | A10G small | completed | tuned direct-Q3 confirmation |

## Current headline

The v7 trainability mechanism replicated in **3/3 training orders**.

In every order, Q9 is worse than direct Q3 at the equal-compute step-300
fixed-Q3 diagnostic, yet finishes better after the identical 900-step Q3
continuation.

Aggregate across seeds 1729, 271828, 424242:

- mean final loss advantage: **0.688 nats/token**
- mean paired PPL reduction: **49.69%**
- mean teacher top-1 gain: **+6.86 pp**
- mean Q3-vs-Q9 changed-set Jaccard: **19.66%**

FP32 warmup never approaches the Q9 result, but it is only worse than direct in
2/3 orders; on seed 271828 it beats direct by a small 0.029 nats/token.

See `../replications/v7_aggregate_summary.md`.

### v8 mechanism update

v8 reproduced the seed-1729 v7 performance, but the proposed
**Q9-specific signed-preload** explanation failed its matched-set test.

- on Q9's later-flip set, Q9 prep was more aligned than Q3 prep by **0.001455**
  normalized units;
- on Q3's later-flip set, Q3 prep was more aligned than Q9 prep by **0.001691**;
- the reciprocal own-set advantages are similar, with the Q3 advantage slightly
  larger.

This matches the preregistered post-selection pattern rather than a
Q9-specific directional-preloading mechanism.

See `run_v8_summary.md`.

### v9 direct-baseline update

The historical direct-Q3 recipe was undertuned.

On seed 1729, a 100-step warmup plus cosine schedule (peak 1e-3, floor 1e-4)
improves direct-Q3 held-out loss from **5.8747** to **5.5957** and PPL from
**355.90** to **269.27**.

Historical Q9 -> Q3 remains better at **5.1938 loss / 180.15 PPL**.

The tuned schedule closes about **41%** of the original loss gap but leaves a
**0.402-nat/token** residual Q9 advantage. This new direct schedule must be
replicated on seeds 271828 and 424242 before becoming canonical.

See `run_v9_summary.md`.


### v9 tuned-direct replication aggregate

The v9-selected warmup+cosine direct-Q3 schedule was run without further tuning
on orders 271828 and 424242.

Q9 remains ahead of tuned direct in **3/3 orders**:

| Seed | Tuned direct loss | Q9 loss | Q9 advantage |
|---:|---:|---:|---:|
| 1729 | 5.5957 | **5.1938** | **0.4020** |
| 271828 | 5.6228 | **5.2417** | **0.3810** |
| 424242 | 5.6067 | **5.3008** | **0.3059** |

Aggregate:

- mean residual loss advantage: **0.3630 nats/token**;
- mean Q9 PPL reduction: **30.38%**;
- mean top-1 gain: **+2.52 pp**;
- mean KL advantage: **0.3871**;
- direct tuning closes about **47.27%** of the old loss gap on average.

The tuned schedule is now the canonical small-budget direct baseline.

See `../replications/v9_tuned_direct_aggregate_summary.md`.
