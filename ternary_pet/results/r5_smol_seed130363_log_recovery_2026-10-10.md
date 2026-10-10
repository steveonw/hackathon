# R5 SmolLM2 seed130363 — post-crash recovery note (2026-10-10)

## Provenance and status

- Original Hugging Face job: https://huggingface.co/jobs/codeflash85/6ac9b8cefee2c90070181ae3
- Original immutable training source: commit `4560d829f483386eb39411c2a9fbf204d4157c4d`, `ternary_pet/r5_smol_new_seed130363_fineweb_hybrid.py`.
- HF job ended `ERROR` (exit 1) at 2026-10-10 04:32:04 UTC. It did **not** emit `FINAL_JSON_BEGIN` / `FINAL_JSON_END`.
- This note preserves log-extracted observations, **not** a science-valid complete R5 result. No additional GPU runs or retries performed.

## Root cause

The original script's `code_hist(qs)` at lines 201–210 returns a dictionary of normalized fractions keyed by ternary code strings `"-1"`, `"0"`, `"1"` and no `total` member. At final aggregation, line 1018 accesses `a["final_q3_code_hist"]["total"]`, raising `KeyError: 'total'` before constructing/printing the complete final JSON. The existing separately recorded `quantized_master_weights` should still equal 314572800 per arm. The branch fix checks the histogram code set and its fractions sum to approximately one instead of expecting a nonexistent total member. It is a code fix for *future execution*, **not** post-hoc scientific validation of this failed job.

## Final arm summaries recovered from original job logs

Primary: 32 newly selected (to this QAT study) FineWeb-Edu documents, 16,384 tokens; **the original final per-document JSON was not emitted**, so document-paired validation cannot be independently performed from these logs.

| Arm | FineWeb NLL | WikiText validation NLL | historical WikiText test loss | effective updates | AMP skips |
|---|---:|---:|---:|---:|---:|
| D direct Q3 | 6.617191977798939 | 5.236532557755709 | 5.596514068543911 | 1193 | 7 |
| W wide Q9→Q3 | 5.775721933692694 | 4.506615519523621 | 4.892119951546192 | 1192 | 8 |
| N narrow Q9→Q3 | 6.591928333044052 | 5.235158897936344 | 5.599781956523657 | 1192 | 8 |
| H hybrid Q9→Q3 | 5.658845977857709 | 4.44468542560935 | 4.8923281244933605 | 1191 | 9 |

Descriptive primary differences: D−H = **0.958346** nats/token, N−H = **0.933082**, W−H = **0.116876**, D−W = **0.841470**. These do **not** establish `valid_for_science=true` because the precommitted post-run checks and per-document evidence are missing.

## Snapshot retention

H inference snapshot upload attempted; **`retained=false`**, failure HTTP **401 Unauthorized** on `https://huggingface.co/api/repos/create`. Locally constructed file reported **64,195,367 bytes**, SHA-256 `29e9f12623bb402fac15e96622fcdea405a03d2c17fcac33c6eed49076efd227`, intended private destination `codeflash85/ternary-pet-r5-checkpoints/smol_seed130363/H_final_compact.npz`. Its reported local existence inside ephemeral job storage does not establish a retained file. No actual model snapshot was recovered.

## Interpretation and next handling

- This is primarily an aggregation/reporting failure following logged completion of all four arms, not a recorded training-divergence result.
- It is **not possible** to recreate the missing per-document arrays or full scientific checks from aggregate log lines alone. Do not fabricate or retroactively mark the job valid.
- Granite R5 succeeded with full final JSON and `valid_for_science=true`; Granite's inference snapshot independently failed upload with the same 401.
- Do not automatically rerun compute. Preserve these original outcomes, fix snapshot authentication separately, and only schedule a new pre-registered validation if explicitly authorized.
