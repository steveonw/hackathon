# v4 / v4b replication manifest

## Reference v4

Reference seed: `424242`  
Reference HF job: `6ac539b5404719ba376621a2`

| Treatment | PPL | Top-1 agreement | KL |
|---|---:|---:|---:|
| Direct 1200@3 | 666.66 | 22.84% | 3.546 |
| **Staged 300@9 + 900@3** | **286.97** | **30.49%** | **2.716** |

## Aborted seed-only replicas

Two initial replicas changed only `torch.manual_seed`:

- seed 1729: HF job `6ac5455e404719ba37662609`
- seed 271828: HF job `6ac5455f404719ba3766260b`

They produced bit-for-bit identical early training traces to each other and to
v4, revealing that v4's fixed data order and effectively deterministic model
path meant the nominal seed was not introducing meaningful stochasticity.
Both jobs were canceled rather than spending the full GPU budget.

This is retained as a methodological finding, not counted as replication.

## Confirmatory v4b replicas

v4b makes the seed meaningful by permuting the **same 1200 training chunks**.
The held-out evaluator, quantizer, frozen/non-frozen parameter choices,
optimizer, loss, and schedule are unchanged.

The v4-selected learning rate is locked at `1e-4` instead of being re-tuned
per seed. Only the preregistered primary comparison is run:

- direct ternary: 1200 steps
- staged: 300 steps at 9 states + 900 steps at 3 states

Within each seed, both treatments consume the same shuffled 1200 chunks in the
same order. Therefore the treatment difference is the precision schedule.

Pinned code commit:

`1e57c60a2c57fcd5880721420c2a6a11c2fc2680`

| Order seed | Script | HF job |
|---:|---|---|
| 1729 | `smollm2_v4b_orderseed1729.py` | `6ac54639fbc85ba6823b890e` |
| 271828 | `smollm2_v4b_orderseed271828.py` | `6ac5463bfbc85ba6823b8910` |

Replication criterion: staged 9->3 should beat direct 1200@3 on the fixed
8192-token evaluator, with lower perplexity/KL and higher teacher top-1
agreement.


## v4b launch correction

The first v4b launch attempt at commit
`1e57c60a2c57fcd5880721420c2a6a11c2fc2680` failed before the main training
loop because the generated scripts referenced the shuffled-order variable before
the insertion was actually present. No confirmatory training result from those
jobs is valid.

The scripts were patched and the corrected immutable commit is:

`f677c3065393582e43b77b255c267f17dd02e9cb`

Only jobs launched from this corrected commit count toward the replication.


### Corrected active jobs

Pinned code: `f677c3065393582e43b77b255c267f17dd02e9cb`

- order seed 1729: `6ac546b5fbc85ba6823b8941`
- order seed 271828: `6ac546bafbc85ba6823b8946`

The printed permutation heads differ, confirming these are genuinely different
training orders.
