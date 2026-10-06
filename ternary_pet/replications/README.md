# v4 replication manifest

Reference protocol: `ternary_pet/smollm2_v4_fliprate_9to3.py`

Reference seed: `424242`  
Reference HF job: `6ac539b5404719ba376621a2`

Reference v4 equal-total result:

| Seed | Direct 1200 PPL | 9->3 PPL | Direct top-1 | 9->3 top-1 |
|---:|---:|---:|---:|---:|
| 424242 | 666.66 | **286.97** | 22.84% | **30.49%** |

Replication code is pinned to Git commit:

`f148eb024cd14ba28859718a9a9f59d55bf329d2`

## Replication jobs

| Seed | Script | HF job |
|---:|---|---|
| 1729 | `replications/smollm2_v4_seed1729.py` | `6ac5455e404719ba37662609` |
| 271828 | `replications/smollm2_v4_seed271828.py` | `6ac5455f404719ba3766260b` |

Only the seed differs from the v4 protocol. Both jobs use T4 small and a
45-minute hard timeout.

The replication criterion is simple: the staged 300@9 + 900@3 condition should
beat the direct 1200@3 control on the fixed 8192-token evaluator. Perplexity,
teacher top-1 agreement, KL, code-flip diagnostics, and generations will all be
recorded.
