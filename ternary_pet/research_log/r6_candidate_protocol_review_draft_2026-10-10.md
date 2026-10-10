# R6 candidate protocol — review draft, NOT pre-registered or authorized

**Created 2026-10-10 after reviewing R5.** User subsequently approved proceeding without repeated permission requests, in context of the offered scope of **at most two A10G-small GPU jobs, one Granite and one Smol, 90-minute cap each**. This approves that ceiling, **not** a run with an unverified dataset, a missing secret, or an incomplete preflight. This remains a planning draft, not a locked preregistration or active experiment. Do not choose seeds, corpus, or endpoint based on observed R6 outcome.

## Why another experiment?

[R5 synthesis](../results/r5_cross_model_final_synthesis_2026-10-10.md) shows a complete Granite seed104729 result with 15/15 checks and hybrid outperforming D/W/N on newly selected FineWeb-Edu documents; Smol seed130363 ended in a post-evaluation output failure and has log-only aggregate outcomes. One new valid seed/model family and possible base pretraining overlap are not sufficient to generalize or identify the mechanism. Previous R4 seeds were historically selected.

## Recommended primary research question

Does hybrid Q9(300)→Q3(900) reliably lower final ternary held-out next-token NLL versus wide Q9→Q3 and direct Q3 across **new independent training seeds and a separate evaluation corpus** under the same within-model training policy?

## Candidate arms (freeze before launching)

- **D:** historical model-specific direct Q3, 1200 opportunities.
- **W:** original nine-level *wide* Q9 for 300, then Q3 for 900.
- **N:** nine-level *narrow* Q9 for 300, then Q3 for 900.
- **H:** nine-level *hybrid* Q9 with same seven interior reconstruction levels as N and original W outer extremes ±α, then Q3 for 900.
- Same tokenizer/source revision, 1200 train examples, fixed order, evaluation documents, Q3 scale reset, source BF16 rounding, LR and optimizer within a given family as the frozen R5/R4 historical protocols. Do not pool cross-family raw losses or treat family schedules as matched.

## Before locking R6 (gates requiring decisions)

1. **Budget and scope:** authorized ceiling is **two A10G-small 90-minute jobs total**, one per family. Confirm technical feasibility before launching; no silent retries, additional GPUs, or extended runtime.
2. **New training seeds:** select and publish fixed deterministic seeds *without testing outcomes*, document selection method, and do not reuse R1–R5 seeds.
3. **Primary corpus:** identify a suitable genuinely different distribution from FineWeb-Edu, check its license, stability, accessible pinned dataset revision, tokenizer-specific selection and document IDs, and likely pretraining contamination caveats. **PG-19 books** are a promising candidate, but the familiar HF dataset mirror uses legacy dataset scripts that may not load under current `datasets`; first identify a working pinned, appropriately licensed distribution and actually run a CPU selector test. Avoid picking corpus/subset after viewing model losses.
4. **Sample size / precision:** freeze number of independent documents, target tokens per doc, denominator, corpus-selection code, and intended confidence/uncertainty treatment. Document-level paired comparisons are not independent training-seed repetitions. If model tokenizers select different docs, keep comparisons within family.
5. **All checks before GPU:** source frozen by commit SHA, CPU AST parse and toy STE/codebook/state-transition checks, document selector audit for all targeted tokenizers, 1200-opportunity train schedule, nonfinite handling, snapshot and JSON serialization smoke test. Validate output-schema without indexing missing keys. Fully exercise construction of terminal `FINAL_JSON` on synthetic miniature arm data.
6. **Snapshot retention:** inject `HF_TOKEN` as a job secret with CLI `--secrets HF_TOKEN` (remote CPU proof [HF 6ac9ce67095c57808930e890](https://huggingface.co/jobs/codeflash85/6ac9ce67095c57808930e890) passed). Require private destination, upload + authenticated readback, checksum match; save inference-only packed Q3 codes, α/scales, tensor names/shapes and source revision, and explicitly distinguish from full training restart state. No token values in scripts, logs or git.
7. **Archival:** raw terminal JSON includes each arm's per-document NLL sums and losses, document hashes/source rows, NLL denominator, final code histogram and its schema, Q9 code occupancy, effective updates/skips, switches, optimizer and precision, dependency versions, preflight manifest, HF job ID, executed SHA, `valid_for_science`, and snapshot retention verified separately. All crashes preserve log data, without reconstructing unavailable per-document metrics.
8. **Stopping/interpretation:** no adaptive seed searches, no post-hoc tuning on the primary corpus, retain all negatives and technical failures, no automatic retries or additional GPU charges.

## Explicit proposed comparisons

Primary H relative to W and D, with N included as the matched inner-codebook mechanism control; report aggregate NLL differences and paired-document sign/median/spread **per training seed**, plus distribution across seeds when sufficient independent seeds exist. Secondary: WikiText2 validation, original historical test anchors (never as selection criterion), Q9 switch shock, and saved final Q3 occupancy/margins. Do not claim a pure causal role for outer magnitude alone; optimizer and scale trajectories may mediate results.

## Next operational decision

The two-job/90-minute ceiling is authorized. Pick corpus and seeds, build + run CPU-only preflight, then **freeze a separately dated R6 preregistration and immutable source commit**. Only after those gates may any expensive scientific job be submitted. Existing R5 scripts/results remain historical; never overwrite or re-label R5 evidence.
