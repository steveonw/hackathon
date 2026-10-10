# R6 frozen operational preregistration — IMDb-test cross-corpus fresh seeds

**Frozen 2026-10-10 before ANY R6 GPU training.** Approved maximum **two** A10G-small jobs (one Granite, one Smol), **90 minutes each**. **No retries, extra arms, reseeding, retuning, or additional GPU jobs** without new instructions. Scripts not executed on GPU when this was written.

## Scientific purpose

Evaluate whether hybrid Q9(300)→Q3(900) improves ternary next-token NLL relative to direct Q3 (D), wide Q9(300)→Q3(900) (W), and narrow Q9(300)→Q3(900) (N) under frozen original training policies. Fixed new-to-project train-order seeds: Granite-4.0-350M **170141**, SmolLM2-360M-Instruct **190027**. No R6 outcome was inspected to select either seed or test source.

## Locked new primary evaluation

**Dataset `stanfordnlp/imdb`, test split, pinned Hub revision `e628166`**, separate movie-review distribution rather than FineWeb-Edu. Distinct text may still have been ingested during model pretraining; no pretraining-disjoint claim. Deterministic selector scans test split sequentially from row1, at most25,000 rows. For row number `i` and string `text`, derive `doc_id=f"{i}:{SHA256(text)}"`, `digest=SHA256(doc_id)`. Take first32 unique digest values where the first8 digest bytes interpreted unsigned big-endian mod20==7, and each *pinned model tokenizer* encodes at least516 tokens, without special tokens. Four 129-token windows per document give **512 targets/doc = 16,384 per arm/model**. Same selection for all D/W/N/H within model, potentially different sets across model tokenizers; hashes and source rows logged. Do not change dataset/subset according to results. IMDB license metadata is `other` (redistribution terms should be checked separately); no dataset text should be copied to repo.

## Frozen training policies and endpoints

Scripts: `ternary_pet/r6_granite_seed170141_imdb_hybrid.py` and `ternary_pet/r6_smol_seed190027_imdb_hybrid.py`. Derived without tuning from R5's historical validated/loss-recovered scripts. Granite: BF16 constant LR1e-4 with historical direct-Q3 step300 reset. Smol: FP16 + GradScaler, warmup100→1e-3 then cosine→1e-4, direct Q3 retains continuous Adam. W/N/H stages: Q9 300 opportunities, original-source Q3 scales restored and new Adam/GradScaler at transition, then Q3 900. Q9 hard codes -4..4; N and H have matching 7 inner reconstruction values; H's ±4 extremes equal W outer ±alpha. Each arm gets same model-specific training data and order and 1200 scheduled opportunities. Cross-model raw NLL not pooled.

Primary comparisons: D−H, W−H, N−H aggregate IMF test next-token NLL; report paired per-document signs/median/spread within each model only and all failures. Secondary first64 WikiText validation chunks, existing historical test, Q9 diagnostics and final Q3 code/scale histograms. Snapshot only H arm: compressed inference-only ternary codes/scales, **not** restorable FP32 master+Adam optimizer training state.

## CPU gate and upload

[CPU preflight job `6ac9d3defee2c90070183201`](https://huggingface.co/jobs/codeflash85/6ac9d3defee2c90070183201) **COMPLETED**, logs `R6_ALL_CPU_PREFLIGHT_OK`, both scripts parsed; checked expected new seeds/terminal markers/secret HfApi token injection; actually selected 32 documents for both pinned tokenizers: Granite scan to row5739, Smol to row4692. This test did **not** execute full GPU training or prove downstream metric schema at GPU-scale. Prior remote [CPU secret test](https://huggingface.co/jobs/codeflash85/6ac9ce67095c57808930e890) proved private repository write+readback using local CLI `--secrets HF_TOKEN`.

**Critical:** launch scientific jobs **only from the authenticated local CLI with `--secrets HF_TOKEN`**, since the ChatGPT Hugging Face app session does not have access to the Windows-stored write token. A job launched without the secret is not equivalent and may fail retaining a checkpoint. Record job IDs and exact executed Git SHA. Original experiment scripts were copied with historical `R5_` names for some metrics/keys; interpret `fresh_fineweb` as the R6 **IMDb** measurement only if `fineweb_new_doc_audit.dataset=="stanfordnlp/imdb"`. These legacy labels should be normalized in postprocessing, not used to misclassify the eval source.

## Validity / terminal handling

Report all 4 arms, 1200 opportunities, Q9 switch300, finite NLL, 32 matched-doc hashes/16,384 targets, expected pins, codebooks and full JSON checks; inspect `valid_for_science`, not job status alone. Both positive and negative results must be archived under results/, including failed job logs. Snapshot status is separate from science status. No rerun if a job errors. Do not infer sample-wide validity from two seeds.
