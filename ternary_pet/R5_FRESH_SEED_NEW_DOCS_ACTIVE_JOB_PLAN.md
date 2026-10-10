# R5 active handoff — two preregistered fresh-seed cross-model new-document jobs

**State at submission: BOTH R5 GPU jobs ACCEPTED; results PENDING.** Two A10G-small jobs, **90-minute timeout each**, no additional seeds, duplicate experiments or retries. Submitted 2026-10-10 UTC following explicit user "sure do it" authorization of independent seeds/new evaluation documents and preferably retained checkpoints.

## Scientific jobs and provenance

| Family | Frozen novel training seed | Exactly pinned source commit and Python file | Hugging Face job | Submission stage |
|---|---:|---|---|---|
| Granite-4.0-350m | **104729** | `c1f0d61aa69a85899203647b88277382275c2a3a`, [script](r5_granite_new_seed104729_fineweb_hybrid.py) | [`6ac9b8c8095c57808930d94f`](https://huggingface.co/jobs/codeflash85/6ac9b8c8095c57808930d94f) | SCHEDULING |
| SmolLM2-360M-Instruct | **130363** | `4560d829f483386eb39411c2a9fbf204d4157c4d`, [script](r5_smol_new_seed130363_fineweb_hybrid.py) | [`6ac9b8cefee2c90070181ae3`](https://huggingface.co/jobs/codeflash85/6ac9b8cefee2c90070181ae3) | SCHEDULING |

**Frozen prior-to-GPU preregistration:** [R5 protocol](research_log/r5_independent_seed_new_fineweb_validation_prereg_2026-10-09.md), committed `55786ecbd3baa63f5ded08287934d1b09d9e59a9`. New seeds 104729/130363 are fixed arbitrary prime integers, not cherry-picked after observing R5 results. R4 on two known seeds per family had H beating D/W/N but samples and seeds were repeatedly studied.

## CPU preflight (complete before GPU)

[HF `6ac9b87dfee2c90070181aa5`](https://huggingface.co/jobs/codeflash85/6ac9b87dfee2c90070181aa5) **COMPLETED 2026-10-10 04:01:40 UTC**, with logs `R5_QUANTIZER_QA_OK granite`, `R5_SOURCE_AST_QA_OK granite`, `R5_FRESH_DOC_SELECTOR_OK granite`, same for Smol, ending **`R5_ALL_CPU_PREFLIGHT_OK`**. Tests both exact immutable scripts' AST; nine Q9 codes, W/N/H same hard-code decisions at same inputs, H/N seven interior output values identical while H restores ±4 extreme outputs to ±α, FP32 master surrogate gradients including z-clipped weights, post-switch original Q3 forward values. New FineWeb source row selector was actually run using **each model's pinned tokenizer**: Granite found 32 docs in source rows **1487..3052**, Smol 32 in rows **1300..2984**. These sets may differ across tokenizers, and are **not** paired across the two model families.

## Frozen data and protocol

One **new-to-our-study training order** per model, same pinned original model revisions (Granite `bd8a1497065c0d6ba1ef19af6b0d2b14bacf71c2`, Smol `a10cc1512eabd3dde888204e902eca88bddb4951`), WikiText2 training data revision `b08601e04326c79dfdd32d625aee71d232d685c3`. Four independent-from-each-other arms **D direct Q3**, **W wide nine-state Q9→Q3**, **N narrow nine-state Q9→Q3 with original T4 thresholds**, and **H hybrid Q9 with exactly narrow levels 0,±1/6,±2/6,±3/6 and wide outer levels ±1 (times α)**. Q9 prep300 then final ternary continuation900 for W/N/H; model-specific D baseline as historically tested. Same original clipped-z STE and source Q3 scale/Adam reset on stage switch; Granite BF16 constantLR1e-4 and D reset at300, Smol FP16/GradScaler warmup+cosine and D continuous Adam, all pretrained BF16-roundtripped FP32 master QAT. **All four arms evaluated, no tuning.**

**New PRIMARY evaluation (held out from this QAT research to date):** FineWeb-Edu `sample-10BT` pinned revision `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9`, only rows **≥1223**, full-ID SHA256 first8 bytes big-endian **mod20==7**, first 32 distinct eligible documents with **≥516 tokenizer tokens**, bounded search row20k. Exactly **4 non-overlapping129-token windows/doc**, **512 next-token targets/doc**, 32 docs and16384 targets/model/arm. Save full doc SHA256, source row, per-doc NLL sum/mean; four arms within model see *exactly identical tokens and docs*, but because tokenizers differ, Granite and Smol doc sets may not match. Selection was frozen before training; these docs weren't used in earlier project's QAT evaluation; **underlying pretrained model exposure unknown, same FineWeb source family as earlier work**. Primary contrasts `N−H`, `W−H`, `D−H`, `D−W`, paired per-document descriptive signs/median only, not confidence in external population. Secondary WikiText2 validation64 chunks, historical test first64 chunks, train-split dev24 chunks. No exact historic numerical anchors exist for these new seeds, so don't invent them.

## Checkpoint attempt: distinguish restorable inference snapshot from training checkpoint

Each model's **H arm only** attempts private upload to HF **dataset `codeflash85/ternary-pet-r5-checkpoints`**, paths `granite_seed104729/H_final_compact.npz` and `smol_seed130363/H_final_compact.npz` respectively. 2-bit/weight-packed ternary codes + per-layer FP32 alpha, shapes, names, model revision and reconstruction manifest. If upload succeeds, this is restorable **inference-state effective Q3 linears** with the pinned unchanged pretrained nonquantized components; **NOT** a fully resumable training checkpoint and does not contain FP32 master weights/Adam moments/GradScaler. GPU JSON includes SHA256/size, `snapshot.retained` and remote path. Upload/auth failure must set `retained=false`, capture short error, **not** invalidate the science. No public uploads authorized: scripts create/check a **private** dataset repo only.

## Future handling and no automatic retries

1. Inspect only the exact two job IDs above. At terminal state, retrieve complete `FINAL_JSON_BEGIN`..`FINAL_JSON_END` and inspect `checks`, `valid_for_science`, all four primary FineWeb aggregate NLLs and paired 32-document records. Distinguish scientific negative/null result from technical failure. Preserve original logs on failures and **do not automatically re-launch**.
2. Verify that all four arms have 1200 scheduled opportunities, all final Q3, 9 Q9 codes, exact pinned revisions, 32 new docs/16384 tokens, unique/document hash selections, and checkpoint status. Check snapshots in private HF dataset only if `retained=true`; do not claim a retained artifact merely because a temporary `/tmp` filename existed.
3. Archive both full JSON documents with HF job provenance under `ternary_pet/results/`, prepare cross-model narrative without raw-NLL pooling and with **one previously untested seed/model** limitation, and update CURRENT_STATE/AI_HANDOFF/EXPERIMENT/RESEARCH_REPORT/SHAREABLE_RESEARCH_REPORT/README.
4. If private checkpoint files exist, record their exact repo URI, file sizes/hash, reconstruction notes. If not, say no durable snapshot retained. **No additional research GPU job is authorized from this instruction.**

All earlier R1–R4/F1–F3/G1/S1/L1 are completed and archived. [Previous R4 scientific aggregate](results/run_r4_granite_smol_easy_hard_four_job_aggregate_2026-10-10.md).
