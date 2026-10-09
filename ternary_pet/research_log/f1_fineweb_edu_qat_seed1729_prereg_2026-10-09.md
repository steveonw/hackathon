# F1 preregistration — source-pretraining-style FineWeb-Edu data generalization

**Date:** 2026-10-09 UTC / October 8 EDT. **Status:** frozen before GPU; one bounded exploratory job.  
**User idea:** "could we use the data set they were train on" — yes: the official SmolLM2-360M model card names FineWeb-Edu as a pretraining ingredient. We use an **available public sample**, not the full undisclosed/filtered 4T-token mixture or the model's SFT/DPO dataset.
**Primary comparison:** paired direct Q3 versus Q9→Q3 under the *same 1200-opportunity recipe*, training on FineWeb-Edu instead of WikiText-2. This isolates QAT data distribution within F1; changing this dataset and the 6000-step horizon simultaneously would be confounded.

## Data and provenance

- **Model:** `HuggingFaceTB/SmolLM2-360M-Instruct`, immutable revision `a10cc1512eabd3dde888204e902eca88bddb4951`, BF16-rounded source, original tokenizer.
- **Dataset:** `HuggingFaceFW/fineweb-edu`, public `sample-10BT` configuration, immutable revision `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9`, split `train` accessed via `streaming=True`. Preflight CPU job `6ac86d5e095c578089301e06` successfully accessed first documents and model tokenizer (`DATA_SOURCE_PREFLIGHT_OK`).
- **Document partition:** use `sha256(str(document["id"]))` mod20 deterministically: buckets `0` for **held-out FineWeb evaluation** (~5%), `1` for **train-split/dev** (~5%), and `2..19` for QAT training (90%). Concatenate text+EOS **within each bucket only**, then emit non-overlapping `SEQ+1=129` token chunks. Take first **1200 training chunks**, first **24 dev chunks**, and first **128 eval chunks** (16384 target tokens). Avoid shared documents across the three partitions; exact partition determined before looking at model outcomes. Do **not** optimize on eval. Log first hashed source IDs / content checksums without reproducing copyrighted text.
- **Independent second evaluation:** 128 WikiText-2 **validation** chunks (not the historically inspected test slice), dataset pinned revision `b08601e04326c79dfdd32d625aee71d232d685c3`. This is out-of-domain validation relative to FineWeb-Edu QAT, but training-domain overlap with source model's original pretraining is unknown.
- **Pretraining contamination caveat:** FineWeb-Edu was an ingredient of the base model's pretraining; sampled raw web documents **may already have appeared in pretraining**. Disjoint FineWeb QAT doc partitions are valid for comparing the two QAT arms' *adaptation*, but cannot establish wholly never-before-seen data to the pretrained model. Do not call the FineWeb eval independent of pretraining.

## Training and checks

- **D:** 1200 steps direct Q3, learned row scales and Adam continuous.
- **S300:** 300 steps Q9 then 900 Q3, restore frozen initial Q3 scales and fresh optimizer/scaler at transition.
- Same ordered QAT FineWeb chunks (shuffled with `torch.randperm(1200, generator=manual_seed(1729))`) and identical global LR schedule 100-step warmup to1e-3 then cosine to1e-4 by step1200. Same fixed FP16 teacher, 35% CE+65% KL, FP32 master/FP16 compute, AdamW betas(0.9,0.95), wd0, gradient clip1, same target linear modules/frozen others.
- Evaluate both arms once at final step1200 on BOTH FineWeb heldout and WikiText validation. Train-split/dev diagnostics at 300,600,900,1200 on 24 fixed FineWeb dev chunks, and S300 native Q9 vs initial-scale projected Q3 switch shock. Report PPL/teacher top1/KL alongside NLL; original Q3 code movement, amp skips and optimizer updates, model/dataset SHAs, sample bucket/source IDs and source-token count, data order.
- **Primary prereg outcome:** paired FineWeb held-out `LD-LS` NLL (positive = staged advantage), secondary WikiText validation paired gap, code diagnostics, and training diagnostics. No posthoc success threshold or parameter search. This is **one** seed/order, no CI claim, and not a reproduction of previous WikiText results because the training distribution deliberately changed.
- **Data integrity hard checks:** no document-ID overlap among train/dev/eval; chunks exactly 129 token IDs and a documented EOS convention; no duplicate QAT training chunks by index; 1200 scheduled opportunities each; no tuning on heldout; all final Q3 quantizers; losses finite. If HTTP/streaming failures prevent obtaining documented chunks, log a technical failure rather than silently switching data. Model revision and dataset revision must be pinned.
- Hardware **one A10G-small**, **90-minute timeout**, **one job**. Stop/no silent second run on OOM, timeout or degraded baseline. The paired WikiText longer-horizon L1 is a separate job and intentionally changes only training horizon in the established WikiText recipe.

## Results and handoff

Frozen preregistration before code/job. Verify CPU syntax and dataset smoke; pin experiment code SHA; record HF job ID, timing and GPU. On completion archive original full `FINAL_JSON` plus job provenance, summary, update [CURRENT_STATE.md](../CURRENT_STATE.md), [NEXT_EXPERIMENT_PLAN.md](../NEXT_EXPERIMENT_PLAN.md), [EXPERIMENT.md](../EXPERIMENT.md), living reports. Preserve null/negative results; no automatic repetition or further tuning.
