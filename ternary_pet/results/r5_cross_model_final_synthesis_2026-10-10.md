# R5 terminal synthesis — independent preregistered seeds and new-to-QAT FineWeb documents

**Date:** 2026-10-10 UTC. **Status:** Granite scientifically valid; SmolLM2 aggregate logs recovered but incomplete final verification. This report is postrun analysis, not a new preregistration, and no additional GPU compute is authorized by it.

## Provenance

- [Frozen R5 preregistration](../research_log/r5_independent_seed_new_fineweb_validation_prereg_2026-10-09.md) and [original job plan](../R5_FRESH_SEED_NEW_DOCS_ACTIVE_JOB_PLAN.md). These describe the state *before* completion and must be read as historical.
- **Granite-4.0-350M, seed104729**: [HF job](https://huggingface.co/jobs/codeflash85/6ac9b8c8095c57808930d94f), pinned executed source commit `c1f0d61aa69a85899203647b88277382275c2a3a`; [complete raw original FINAL_JSON](run_r5_granite_seed104729_freshdocs_2026-10-10.json), `valid_for_science=true`, **15/15 validity checks**.
- **SmolLM2-360M-Instruct, seed130363**: [HF job](https://huggingface.co/jobs/codeflash85/6ac9b8cefee2c90070181ae3), pinned executed source commit `4560d829f483386eb39411c2a9fbf204d4157c4d`; [log-recovery record](r5_smol_seed130363_log_recovery_2026-10-10.md), **job ERROR**, no final JSON, no original per-document vectors. The original reporting code attempted to access `final_q3_code_hist["total"]`, although that function returns fractions indexed by code; reporting logic was subsequently fixed on main, **not retroactively applied** to the pinned run.
- **Remote secret upload validation**: [HF CPU job](https://huggingface.co/jobs/codeflash85/6ac9ce67095c57808930e890), **COMPLETED**; log emitted `R5_REMOTE_CHECKPOINT_UPLOAD_OK`; private repo `codeflash85/ternary-pet-r5-checkpoints`, small upload and readback succeeded using `--secrets HF_TOKEN`. **Neither original R5 H snapshot was retained**: both original jobs ran before this fix and failed upload with HTTP 401.

## Primary: 32 FineWeb-Edu documents newly selected for this project's QAT validation

All values below are negative log-likelihood (nats/target token; **lower is better**). Within-family arms share the exact same 32 documents and 16,384 target-token window selection; document identities may **differ across models**, and pretrained model exposure to these documents is unknown.

| Family and evidence status | D direct Q3 | W wide Q9→Q3 | N narrow Q9→Q3 | H hybrid Q9→Q3 |
|---|---:|---:|---:|---:|
| Granite104729 **verified** | 6.490101 | 6.388855 | 6.506379 | **6.358198** |
| Smol130363 **log-recovered, not validated** | 6.617192 | 5.775722 | 6.591928 | **5.658846** |

### Granite: reproduced exact document-paired differences from archived original per-document records

| Contrast (positive means second arm better) | Mean difference (nats/token) | Documents positive /32 | Sample SD of document differences |
|---|---:|---:|---:|
| D−H | +0.131904 | 32 | 0.061012 |
| W−H | +0.030657 | 27 | 0.040488 |
| N−H | +0.148181 | 31 | 0.067395 |
| D−W | +0.101246 | 31 | 0.067625 |

Each document contributes precisely 512 target tokens, making the unweighted mean of 32 per-document differences equal to the corpus aggregate gap; all signs and aggregate gaps agree with the original FINAL_JSON. These are **descriptive within-one-seed** measures, not independent model/seed replication intervals or pretraining-disjoint evidence.

### Smol: only final-arm aggregate lines were recoverable

Within Smol's same-job arm logs, observed gaps were D−H = **+0.958346**, W−H = **+0.116876**, N−H = **+0.933082** nats/token. Final per-document signs, code histograms, all post-run checks and the complete FINAL_JSON are unavailable; **do not synthesize them or assign `valid_for_science=true`.** Smol also has 1191–1193 effective updates and 7–9 AMP skips across arms, as documented in the recovery note.

## Secondary: WikiText2 validation NLL (not the primary endpoint)

| Model / evidence | D | W | N | H |
|---|---:|---:|---:|---:|
| Granite validated | 5.395869 | 5.286142 | 5.372579 | **5.241634** |
| Smol recovered log | 5.236533 | 4.506616 | 5.235159 | **4.444685** |

## Synthesis and limits

The hybrid Q9 codebook, which retains the seven narrow interior reconstructions but restores the two outer levels to wide values, won the *observed aggregate comparisons* against D/W/N for both previously untested R5 seeds. This extends the R4 historical-seed results, but **only Granite contributes a complete new-seed validated R5 replication**. Do not pool model-family NLL numbers or infer universal superiority. Granite and Smol differ in optimization, precision and direct-Q3 reset behavior. A changed extreme reconstruction level can change downstream learned scales, master weights, and optimizer trajectories, so these data do not identify a single mediator. Only one new seed per model was run, selected by fixed arbitrary primes, not an unbiased seed sample. Both original H inference checkpoints are missing; this does **not** undo their logged losses, but prevents restoring the trained models.

## Next research gate (proposal only, no jobs scheduled)

1. Maintain frozen original scripts and source hashes; treat postrun reporting/authentication patches as different code versions.
2. If further paid training is approved, **freeze a separate, prospective R6 protocol before running**: independent new seeds, a genuinely distinct non-FineWeb evaluation corpus if feasible, exact paired per-document outputs and loss checks, and verified private inference-state retention using CLI-launched HF jobs with `--secrets HF_TOKEN`.
3. Put all final checks **before** the success marker, with robust serializable schema validation and terminal raw JSON. Preserve both positive and negative outcomes, avoid test-data hyperparameter search, and budget/authorize every GPU job explicitly.
4. Do not claim the remote CPU verification restored the previously lost snapshots or that the original Smol run was scientifically valid.
