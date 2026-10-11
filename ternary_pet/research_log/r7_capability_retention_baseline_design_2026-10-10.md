# R7 preregistration draft — frozen capability-retention baseline (CPU first)

**Created 2026-10-10. Status: baseline design, not yet executed.** This is a fixed protocol for comparing the *pinned original base models* and *R6 H compact ternary checkpoints* **before tuning or creating R7 training candidates**. Do not change prompts, grading, or task counts based on scores. No GPU training is authorized or implied.

## Core question and units

How much of the original model's **instruction-following, answer correctness, fluency, and nonrepetition** is retained by its R6 hybrid-trained ternary weights? Primary unit is a **matched prompt/model pair**, not individual generated tokens; report results **separately by model family** and per task category. Do not pool unlike families into one success rate without qualification.

## Frozen inference arms and provenance

- **Granite original**: `ibm-granite/granite-4.0-350m`, revision `bd8a1497065c0d6ba1ef19af6b0d2b14bacf71c2`, FP32 CPU inference with original model source weights BF16-roundtripped as in R6.
- **Granite R6-H**: private snapshot `codeflash85/ternary-pet-r5-checkpoints/granite_seed170141/H_final_compact.npz`, SHA256 `3c30ab36a6518606db74a45411b6e96639171c063f0dd2b0f6cea384581f0888`, 168 quantized layers / 249,561,088 quantized weights.
- **SmolLM2 original**: `HuggingFaceTB/SmolLM2-360M-Instruct`, revision `a10cc1512eabd3dde888204e902eca88bddb4951`, same source BF16 roundtrip.
- **SmolLM2 R6-H**: private snapshot `codeflash85/ternary-pet-r5-checkpoints/smol_seed190027/H_final_compact.npz`, SHA256 `5d41638430f7a54fb553f2ad32d861386e5f38e42901c916fe8926172447fbfe`, 224 quantized layers / 314,572,800 weights.
- Reconstruct H exactly with prior verified 2-bit code decode (0→0,1→+1,2→−1), code×α/1.5, and pinned untouched source weights after BF16 roundtrip. Revalidate hashes and tensor dimensions on every independent run. This is a comparison to the R6 training regimen, **not a clean isolated compression-only causal experiment**.

## Fixed battery — 40 prompts per model, same text within families

Prompt set must be checked into the repository as a versioned JSONL or JSON file **before execution**. No prompt may have been tuned on R6 generation results or include the three previously probed prompts (scientist notebook, experimental baseline, 17+28). Four strata of 10 distinct prompts:

1. **Arithmetic/exact (10):** five 2–3 digit additions, five basic nonnegative subtractions. Deterministic expected integer; instruction `Answer with only the number.` Grade by matching first standalone integer token after stripping leading space; also report strict normalized exact match separately. Never count an answer embedded in unrelated prose as strict correct.
2. **Simple instruction/compliance (10):** five short deterministic formatting instructions with independently checkable responses (specific prefix, uppercase, fixed word count), and five fixed-choice factual/logic prompts whose exact expected answer is in the JSON dataset. Grade with per-prompt deterministic predicates and log any unexpected response format. No LLM-as-judge for the primary endpoint.
3. **Reading/extraction (10):** short fabricated self-contained snippets, each followed by a direct literal extraction question requiring one phrase, number, or name present in the snippet. Grade normalized strict answer plus a permissive exact-substring secondary. Avoid needing external factual knowledge.
4. **Open-ended continuity (10):** distinct natural passage starts, no direct instruction. Record generation, fraction of generated token IDs reused from previous 8-token context, longest repeated 1–4-token ngram runs, unique 2-gram ratio, EOS/newline-only/punctuation-only flags, and token count. Human-blind review may be a later **secondary** endpoint with recorded rubric; no subjective claims based on cherry-picked samples.

## Inference, scoring, and consistency

- **Primary deterministic path:** 64 newly generated tokens per prompt (or EOS), greedy decoding, no repetition penalty, no forced words or postprocessing beyond special-token stripping. For SmolLM2 instruction strata, use `tokenizer.apply_chat_template(..., add_generation_prompt=True)`; for continuation stratum use plain prefix. Granite is not labeled an instruction-tuned model; use a plain explicit `Question: ...\nAnswer:` wrapper for instruction strata and interpret answer rates relative to **its own original base** rather than expecting Smol-level performance.
- Fix prompt order and seed (20261010); set eval mode/use_cache, use the same full-precision *compute* path for both arms (CPU FP32) and identical tokenizer revisions. Limit threads for repeatability. Include prompt tokens, output token IDs, output text, raw output length, first-token EOS flag, and latency **as descriptive only** (unoptimized CPU runtime is not a speed comparison).
- **Secondary generation:** sample only the first *five* prompts per stratum with seed20261010, temperature0.8, top_p0.9, max_new_tokens64; seeded per (family,arm,prompt) with same random seed, but do not claim token-for-token randomness is a causal counterfactual.
- **Teacher-forced metric:** on the same 4 existing fixed R6 IMDb documents per model, report reconstructed H loss parity to archived original job, and pinned base NLL on exactly the same token windows. This is a **secondary** robustness anchor, not a substitute for ability retention. Already observed 4-doc H parity is close; do not use it to select prompts.

## Prespecified summaries

For every family × arm: show exact successes /10 for each of arithmetic, instruction, extraction, and separately strict/permissive; present paired base-versus-H outcomes per prompt as both-success, base-only, H-only, neither. Report H/base retention ratio only when base performance is nonzero, and always alongside absolute success counts. Open continuation: report median unique-bigram ratio and repetition rate, plus all 10 raw samples; include a few *predesignated* examples (first two per stratum) in executive summary, not post-hoc attractive cases. For the primary comparison, consider **base-only failures** (tasks base succeeds and H fails) as evidence of lost ability. No claims of statistical significance from 10 prompts/category; follow up with preregistered larger holdouts before publication claims.

## Validity gates

All 80 greedy records (2 families × 2 arms × 40 prompts) must be present; 40/40 matched prompt/format pairs within each family; 20 sampled prompts per arm as specified, finite logits, no unexpected crashes, all codebook tensor counts match, checksum matches. If any fails, mark the family incomplete and preserve partial output; do not silently reassign failures to incorrect answers. Record model and tokenizer revision, script SHA, dependency versions, CPU flavor, Hugging Face job ids, category breakdowns, immutable prompt-set SHA256. Separate **technical validity** from **ability success**, which may be very poor.

## Resource and privacy boundaries

Use CPU-only Hugging Face Jobs, one job/family, appropriate timeout based on preliminary runtime. The private snapshots need the account holder's existing Windows CLI `--secrets HF_TOKEN` (do not paste or commit the token). Do not launch more GPU scientific training or retune until baseline results are archived. Original model licenses and dataset-use rights remain applicable.

## Decision gate for any R7 training proposal

Only after this baseline: (1) inspect base success rate and H degradation per category, (2) identify whether instruction-format sensitivity or general next-token degeneration is dominant, (3) freeze a *new* train-vs-control proposal e.g. instruction-aware teacher distillation with matched training opportunities and identical eval battery, and (4) separately approve compute if needed. A capability-retention improvement means H increases paired instruction/extraction/arithmetic successes **without a material worsening of its relative Q3 loss**; exact thresholds and independent holdout must be frozen in the later experimental preregistration, not retrospectively set here.
