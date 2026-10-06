# v4 / v4b replication record

## Reference v4

Reference HF job: `6ac539b5404719ba376621a2`.

The original paired result favored 9->3:

- direct 1200@3: PPL 666.66, top-1 22.84%, KL 3.546
- staged 300@9 + 900@3: PPL 286.97, top-1 30.49%, KL 2.716

## Methodological false start

Two attempted "seed replications" changed only `torch.manual_seed`. Their
early traces were bit-for-bit identical because the training order was fixed
and the relevant path was effectively deterministic. Those jobs were canceled
and are **not counted as replications**.

A first v4b script launch then failed before training because the generated
shuffle variable was missing. That failed launch is also excluded.

Keeping these failures documented is intentional.

## Valid confirmatory protocol

Corrected immutable code commit:

`f677c3065393582e43b77b255c267f17dd02e9cb`

The seed now permutes the same 1200 training chunks. Evaluation stays fixed.
The v4-selected LR is locked at `1e-4`; it is not re-tuned per seed.

| Order seed | HF job | Direct PPL | 9->3 PPL | Result |
|---:|---|---:|---:|---|
| 1729 | `6ac546b5fbc85ba6823b8941` | 352.77 | **178.85** | staged wins |
| 271828 | `6ac546bafbc85ba6823b8946` | 383.56 | **177.20** | staged wins |

Both completed successfully on T4 small.

See `v4b_aggregate_summary.md` for the three-ordering analysis.
