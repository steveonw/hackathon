# F2/F3 preregistration — FineWeb-Edu paired Q9→Q3 replication, two new training orders

**Recorded:** 2026-10-09 EDT/UTC, **before** submitting the replication GPU jobs or observing their training outcomes.  
**Status:** Frozen replication plan; scientific outcomes not known.  
**User authorization:** "do the other 2 seeds on the fine and see the effects."  
**Original seed1729 study:** [F1 raw](../results/run_f1_fineweb_edu_direct_vs_staged_seed1729_2026-10-09.json) and [F1 summary](../results/run_f1_fineweb_edu_direct_vs_staged_seed1729_summary.md).  
**Frozen F1 source code SHA:** `64dcd7f20240b4c62e6ecac8df70a1336a57bb14`, [script](../f1_fineweb_edu_qat_seed1729.py).

## Research question, primary outcomes, interpretation

Can the F1 matched direct-Q3 versus Q9(300)→Q3(900) advantage on **FineWeb-Edu** be reproduced when **only the training-order / initialization RNG seed changes**, using **271828 and 424242**? This is a *two-order exact recipe replication* and should be reported together with the prior 1729 arm as a three-order aggregate. Unlike F1, the extra seeds are new to this FineWeb experiment but have appeared in earlier WikiText experiments.

**Primary effect per seed:** `FineWeb_eval_NLL(D) - FineWeb_eval_NLL(S300)` on the same fixed FineWeb-Edu document-heldout split, positive favors staged.  
**Secondary effect:** `WikiText2_validation_NLL(D) - WikiText2_validation_NLL(S300)`, positive favors staged, plus paired PPL ratios, top1/teacher-KL, optimizer skips, train-split learning curves.  
**Aggregate (planned before outcomes):** report all three seeds' raw paired gaps, mean, median, range, number positive, and descriptive standard deviation. With only three seeds and **shared held-out documents**, no claim of independent-data confirmation or narrow statistical confidence intervals. Explicitly print negative outcomes if any. No hypothesis threshold, hyperparameter search or early stopping.

## What stays unchanged from F1 (all seeds)

- Source model: `HuggingFaceTB/SmolLM2-360M-Instruct` revision `a10cc1512eabd3dde888204e902eca88bddb4951`, original tokenizer, BF16-rounded weights, FP32 trainable quantized masters and per-row scales, all other modules frozen, target all transformer linear layers except `lm_head`.
- Public QAT dataset: `HuggingFaceFW/fineweb-edu`, configuration `sample-10BT`, revision `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9`, `streaming=True`. Partition documents by `sha256(document id)` leading 8 bytes modulo20: bucket0 evaluation, bucket1 train-split/dev, buckets2..19 training; non-overlap across source-document IDs. Same **first 1,200 129-token train chunks from 180 docs**, **24 dev chunks from 3 docs**, **128 heldout chunks from 21 docs**, all as in F1; the same first 340 streamed documents must suffice. This **does not create new independent evaluation documents**; a future separate dataset-slice confirmation is necessary. Same EOS encoding and deterministic chunk concatenation.
- Separate WikiText-2 **validation** 128 chunks from `Salesforce/wikitext`, revision `b08601e04326c79dfdd32d625aee71d232d685c3`. Never use for recipe tuning.
- Both arms receive exactly the same **1,200 ordered train chunks** at a given seed, the same 100-step warmup up to 1e-3 then v10 global cosine to 1e-4 at step1,200, and same fixed FP16 teacher/0.35 CE+0.65 KL objective; AdamW (betas0.9/0.95, wd0), grad clip1, CUDA FP16 autocast and GradScaler, same quantizer/STE and source model.
- `D` direct Q3 for steps1..1200, with continuous row scales and optimizer. `S300` Q9 for steps1..300, original Q3 scale reset + fresh AdamW and GradScaler at switch, then Q3 steps301..1200.
- Same train-split/dev diagnostics at global steps300,600,900,1200, staged native-Q9→Q3 transition shock, final FineWeb heldout and WikiText validation NLL/PPL/teacher agreement/KL, final Q3 code movement/histogram and source-scale summary.
- Critical: F1's `same_order` check was implemented with a **seed1729-specific 16-index prefix**. New scripts MUST replace that assertion with *an independently precomputed expected prefix for each new seed*, while keeping the training permutation generation exactly `torch.randperm(1200, generator=torch.Generator().manual_seed(SEED))`. Otherwise a correct new-seed run would be flagged as technical failure.
- The **only changes** relative to F1 source are the literal numeric SEED, file identity/provenance strings, and the expected order prefix check used for correctness. Review an automated diff or scripted textual check before running.
- One A10G-small Hugging Face GPU job per seed, max **90 minutes** each. No new seeds, large sweeps or parameter retuning automatically. Do not interrupt the independent L1 6000-step WikiText job.

## Checks and technical failure rules

- Pin scripts to separate immutable Git commits before GPU submission; CPU-based source/syntax static checks must pass, including expected-order prefix and unchanged data/model SHA pins.
- Verify that `valid_for_science=true` and F1's five invariants (`both_1200`, `disjoint_fineweb_documents`, `correct_split_count`, `both_final_finite`, `same_order`) all hold for each seed.
- Require the FineWeb doc-partition source counts and first hashed source IDs to reproduce F1 exactly; if the revision stream is nondeterministic, mark a technical comparability discrepancy and do not silently adjust.
- Track actual optimizer updates and AMP skips separately. Scheduled equal1200 does not entail equal effective updates.
- If F1's source code fails under a new revision/environment, or numerical NaN, OOM, timeout, missing data, or mismatch arises, record failure and job logs; **do not** re-run with changed parameters without explicitly updating the plan and securing approval.
- Archive each full original `FINAL_JSON` with HF provenance under `ternary_pet/results/`, add a descriptive per-seed summary, then a three-seed aggregate. Update `CURRENT_STATE.md`, `AI_HANDOFF.md`, `EXPERIMENT.md`, `README.md`, living reports and [NEXT_JOBS_PLAN.md](../NEXT_JOBS_PLAN.md).

## Interpretation boundaries

The 271828 and 424242 replications assess **random training order and initialization RNG under fixed documents/model/data source**, not new data, architecture, pretraining-state exposure, or better standard ternary QAT baselines. The base model may have already seen FineWeb samples during pretraining. Statistical uncertainty from only three correlated seed runs should be reported cautiously. Independently frozen new document groups, more training orders and longer horizons remain separate questions, not authorized follow-up GPU jobs.
