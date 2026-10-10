# R3 — nonuniform Q9 outer-output recovery and unused per-document FineWeb evaluation

**Study intent:** user authorized follow-up with "do what you need to do" after valid R2 completed. **Current status:** R3 protocol/code frozen, CPU preflight under way; **do not assume GPU launched from this document until the HF job ID is appended**. [CURRENT_STATE.md](CURRENT_STATE.md) is authoritative.

**Frozen preregistration** [R3 nonuniform/QAT-fresh docs](research_log/r3_nonuniform_q9_outer_recovery_freshdoc_prereg_2026-10-09.md) (commit `bf89baef162f2a9e948ab702a1708c74b856d3a4`), committed before training source.  
**Experiment source** [`r3_nonuniform_q9_outer_freshdocs_seed1729.py`](r3_nonuniform_q9_outer_freshdocs_seed1729.py), pinned immutable Git commit **`a4abf256c0be9db771e3ec271c1f0a5e6d605816`**.  
**CPU preflight** [HF `6ac992d4095c57808930c1b0`](https://huggingface.co/jobs/codeflash85/6ac992d4095c57808930c1b0), checks AST, quantizer 9 codes, W/N/H exact nonuniform output and gradient formula, original Q3 post switch and actual streaming FineWeb 32-document selection. **Do not launch GPU until successful complete `R3_ALL_PREFLIGHT_OK`.**

## Controlled arms

**Seed1729, one model, one matched FineWeb training corpus and original hyperparameters.** Direct Q3 D continuous1200, original wide W Q9(300)→Q3(900), historical narrow N `T4,V6` Q9(300)→Q3(900), and nonuniform hybrid H Q9(300)→Q3(900). All Q9 arms use same code assignment `k=round(4*clamp(w/alpha,-.99,.99))`, exactly nine codes.

| Q9 k magnitude | W (wide) | N (narrow) | H (narrow inside, wide extreme) |
|---|---:|---:|---:|
| 0 | 0 | 0 | 0 |
| 1 | 1/4 | 1/6 | 1/6 |
| 2 | 1/2 | 2/6 | 2/6 |
| 3 | 3/4 | 3/6 | 3/6 |
| 4 | 1 | 4/6 | 1 |

Sign is applied symmetrically. Hybrid differs from narrow only on the **two outer ±4 reconstruction values** and differs from wide only in six interior nonzero outputs. Same STE z and code thresholds; all Q9 modes reset to original Q3 scales and fresh Adam/GradScaler after exactly300, then train original Q3 for900.

## Fresh evaluation contract

Old FineWeb QAT used only first340 streamed source docs in the pinned source order. Select **first 32 distinct FineWeb eval bucket0 document IDs at source row>340 with ≥516 tokenizer tokens**, excluding every first340 doc ID and duplicates. Each new document contributes first four nonoverlapping129-token windows (512 next-token target positions). **32 ×512=16384 paired evaluation tokens**, 32 hashed doc IDs, per-document CE sum/mean for each arm and aggregate; no training/dev tuning on this sample. New-to-QAT-evaluation docs may have been seen in model original pretraining; same FineWeb data family. Historic heldout FineWeb 21 docs and WikiText128 chunks are retained only as previously measured secondary endpoints and reproduction anchors.

Primary contrasts on **fresh docs**: `N−H` (outer output magnitude rescue), `H−W` (difference from original-wide interiors), `D−H`, and paired per-document sign and median descriptive stats. Reproduce old FineWeb/WikiText anchors for D/W/N within ±0.03; pass all technical checks before interpreting.

## Execution protocol

- Authorized GPU: **exactly one HF A10G-small, 2-hour max**, detached. No automatic extra seeds or jobs. Hold launch if preflight fails; debug and rerun CPU preflight rather than gambling paid compute.
- At terminal job: inspect exact HF status, read full `FINAL_JSON_BEGIN`...`FINAL_JSON_END`, ensure every check true; archive unmodified science record + HF provenance in `ternary_pet/results`, summarize fresh-doc primary, paired document contrasts, codebook results, old-anchor reproduction, AMP skips and limitations. Persist research state/report updates.
- No full-model checkpoints uploaded here due storage/budget. Save per-doc loss evidence and package versions; do not claim full reproducibility of model state.

**Prior R2:** [raw](results/run_r2_fineweb_q9_output_threshold_factorial_seed1729_2026-10-10.json), [summary](results/run_r2_fineweb_q9_output_threshold_factorial_seed1729_summary.md), valid 16/16 checks. Next AI should never duplicate this R3 GPU job once submitted.
