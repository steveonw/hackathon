# R5 — fresh-seed cross-model hybrid Q9 validation with new FineWeb documents and compact inference snapshots

**Pre-registered 2026-10-09 EDT / 2026-10-10 UTC BEFORE scientific GPU submission or observing R5 losses.** User authorization: "sure do it", accepting recommendation of fresh seeds, new evaluation documents, and preferably retained checkpoints. Resource scope explicitly frozen: **TWO scientific A10G-small jobs total (one Granite, one SmolLM2), timeout90m each**; no extra seed sweeps, retries or changing outputs after seeing results. Existing R4 complete (4/4 valid). R5 is **one new seed per model**, not a population sample.

## Scientific goal and fixed choices

R4 on two *historically selected* seed orders per model found hybrid H beat W/N/D on WikiText2 validation for Granite-4.0-350M and SmolLM2-360M. To reduce seed cherry-picking and recycled-document evaluation, we commit to new, previously unexamined training-order seeds:
- **Granite model `ibm-granite/granite-4.0-350m`: seed `104729`**, one independent training order.
- **SmolLM2 `HuggingFaceTB/SmolLM2-360M-Instruct`: seed `130363`**, one independent training order.

These are **deterministic arbitrary preregistered primes**, not sampled uniformly/randomly or selected after inspecting outcomes. Same model revisions/data pins, original BF16-rounded pretrained source, same 1200 WikiText2 QAT train chunks followed by 24 train-split dev chunks; seed controls train-order permutation and stochasticity. Baseline frozen under each model's original R4 policy, including the meaningful confound: Granite BF16, constant LR1e-4 and D direct Q3 also reset at300; Smol FP16 + GradScaler, warmup100→1e-3 / cosine→1e-4 and D continuous Adam. NO cross-model pooling, quantitative comparisons paired within each family.

## Exact four training arms per scientific GPU

- `D` direct Q3 full1200, using original family-specific direct baseline/reset policy.
- `W` original wide Q9 for300 then Q3 for900, outputs `(0,±α/4,±α/2,±3α/4,±α)`.
- `N` nine-state narrow-T4 Q9 for300 then Q3 for900, outputs `(0,±α/6,±α/3,±α/2,±2α/3)`.
- `H` hybrid outer-recovered Q9 for300 then Q3 for900, outputs `(0,±α/6,±α/3,±α/2,±α)`.

All Q9 prepare `k=round(4clip(w/α,-.99,.99))`, exactly nine states, identical per-step STE `α(z+(h(k)-z).detach())`; H and N share all 7 central forward codebook outputs, differing only at two extreme *code values* ±4. At switch300 restore original-source Q3 row scales and fresh AdamW (plus GradScaler Smol), preserving FP32 master weights and original per-model LR policy. Final Q3 all arms.

## R5 primary evaluation: never-before-used FineWeb document IDs

Pinned public `HuggingFaceFW/fineweb-edu` `sample-10BT`, dataset SHA `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9`. Existing QAT FineWeb F1 and R3 used source positions ≤340 and R3 fresh QAT evaluation source rows343..1222, respectively; **R5 must scan source positions >1222 and use disjoint hash bucket**.

**Frozen deterministic selector:** stream source in dataset order; only rows with **1-based row number ≥1223** and SHA256 of UTF-8 doc `id` satisfying `int.from_bytes(bytes.fromhex(sha256(id))[:8],big) %20 == 7`, distinct IDs and **at least 516 tokenizer tokens** (each model applies its *own pinned tokenizer*, so different document-ID sets are possible and will be disclosed). Select first **32 qualifying** docs (bound scan to first 20,000 source rows). Each contributes its first **four distinct 129-token windows**, 512 next-token target positions/doc. **16,384** evaluated target tokens/arm; save exact document SHA256 hashes, row positions, per-doc cross entropy sum and average, common paired windows across D/W/N/H *within* model; never use this set in training, model selection, hyperparameter tuning or early stopping. Strict data checks: doc count32, 16384 targets, hash uniqueness, source row>1222, bucket7, sufficient token length; if source stream cannot yield32 within20k, mark technical failure, no relaxation or switching subsets after observing output.

These docs are **new to this project's QAT evaluation** and disjoint from prior scanned source rows/bucket, but they may have been part of either pretrained base model's original pretraining. They are also **from the same FineWeb dataset** that Smol FineWeb results used; not a wholly independent dataset. WikiText2 validation first64 chunks serves as SECONDARY matched evaluation; historical test first64 chunks gives context, with no exact NLL anchors for previously unseen seeds.

## Primary and secondary endpoints; interpretation

Primary aggregate fresh FineWeb per-token NLL for all D/W/N/H, with preregistered `N-H`, `W-H`, `D-H`, `D-W`; positive means H (or W) has lower NLL. Paired per-doc signs, mean/median, spread, fraction of all32 favoring H are descriptive only (repeated tokens per doc, one model and fixed set). **No data-dependent choice of which arm or seed to report.** Secondary WikiText validation/test, Q3 final code hist/scale stats, Q9 step300 outer-code fraction, native/projected dev NLL, training trace, effective updates/AMP skips, source version manifest.

### Restorable inference snapshots, best effort not a science pass gate

Save at least the **final H ternary *inference* state** for each model as a **compact 2-bit-packed per-layer code tensor plus source Q3 alpha/raw_alpha** (integer codes -1/0/+1 encoded 0/1/2 and packed 4 codes/byte), layer shapes/names, source model revision and reconstruction instructions. This allows rebuilding the final quantized effective linear weights with the pinned unchanged embedding/norm/head from base model. It does **not** restore optimizer state, Adam moments, FP32 *master* weights, training resumption, or exact AMP scaler. Preferred storage: **private Hugging Face dataset repository** `codeflash85/ternary-pet-r5-checkpoints`, path `r5_<family>_seed<seed>/final_H_q3_compact.npz`, via job credentials if authorized. Include SHA256 checksum and file size in science JSON. If repository upload/auth fails, **record an explicit checkpoint_saved=false and error**; finish science results without falsely claiming an accessible checkpoint. Do NOT silently make the repository public or upload confidential unrelated files. The job's temporary local file may be deleted on termination; do not call that retained.

For mechanism evidence independent of upload, save **final/step300 Q3 code histograms, source-vs-final changed-code fraction, step300 code occupancy, per-layer alpha and relative weight/threshold margin summary**, within bounded JSON.

## CPU/gpu gating and outcome validity

Before paid GPU: freeze source scripts at immutable Git commits; AST parse, exact 9 levels and H/N inner equality, Q3 switch and FP32 scale/weight surrogate toy-grad check; verify the selector yields 32 docs for both tokenizer revisions (using CPU only) or at least identical selector logic to a previously verified dataset scan. Two scientific jobs only; no retries/other variants. Require all arms1200 scheduled, final Q3, finite old+new NLL, exact fresh document accounting and model/data revision pins. There is **no numerical historical loss anchor possible for unseen seeds**, so do not invent one or reuse R4 seed results as target. Check checksum/order permutation deterministically and verify all within-seed training examples same across four arms.

If paid GPU terminates with invalid checks/failure: preserve logs and scientific raw evidence, do NOT claim successful causal replication or retry automatically. All jobs may complete with negative H effects; report honestly. Do not tune any result on these new docs. Interpret consistency only, not universal generalization from two new seeds. Old tests and R4 are background, not fresh confirmations.
