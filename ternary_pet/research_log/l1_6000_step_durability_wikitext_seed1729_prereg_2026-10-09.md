# L1 preregistration — 6000-step staged-vs-direct durability on WikiText-2

**Date:** 2026-10-09 UTC (October 8 EDT). **Status:** frozen before GPU, no results yet. **User authorization:** "set up the next jobs".  
**Scope:** one GPU job, `a10g-small`, max **2 hours**. Two sequential matched Q3 regimes, seed1729. No automatic seed expansion.  
**Purpose:** directly address "does direct Q3 eventually catch up?" while preserving existing quantizer/model/objective and initial global LR trajectory.

## Conditions

- `HuggingFaceTB/SmolLM2-360M-Instruct`, pinned model revision `a10cc1512eabd3dde888204e902eca88bddb4951`. BF16-rounded student source, FP32 trainable master weights + rowwise learned scale, frozen other modules; fixed FP16 reference teacher.
- Dataset: WikiText-2 raw-v1 `Salesforce/wikitext`, pinned revision `b08601e04326c79dfdd32d625aee71d232d685c3`. Original `stream_chunks` concatenates document tokens separated by EOS, then extracts nonoverlapping 129-token chunks for next-token 128 targets.
- **Reproduce exact original first1200 shuffled chunks:** load 6024 train chunks; reserve raw positions 1200..1223 for the same train-split diagnostic as v10; take raw first1200 chunks shuffled with seed1729 (same `torch.randperm(1200)` order), then use raw positions1224..6023 as **4800 new never-repeated QAT training chunks**; no learning from held-out data. Extra 4800 kept in deterministic sequential order for reproducibility.
- Training order for both arms identical; six thousand 128-target-token opportunities = **768,000 supervised training tokens per arm**, no intentional training-data cycles. Source model may have seen portions of WikiText-2 in pretraining; data novelty refers to **our QAT runs**.
- **D:** direct Q3 all6000 steps, Adam/scales continuous.
- **S:** Q9 for steps1..300 then Q3 for steps301..6000, restoring original Q3 scales and fresh Adam+GradScaler at switch. This is the v10 recipe carried forward.
- Objective and optimizer unchanged from v11/C1: 0.35 CE +0.65 fixed BF16-source-teacher KL, AdamW betas(0.9,0.95), wd0, FP16 CUDA autocast/GradScaler, clip1, exact same quantizer. Scheduled global LR for steps1..1200 **identical to v10** (100-step warmup to1e-3, cosine ending1e-4); **after1200 use constant1e-4** to avoid retrospectively retuning earlier recipe. This is a scientific continuation, not claim of optimal 6000-step LR for either method.
- **Evaluation:** fixed 128 chunks = **16,384 target tokens** from **WikiText-2 validation**, pinned same dataset revision. This was not used in the earlier T1/C1/M1 adaptive loop. Evaluate identical chunks at global steps1200,2400,3600,4800,6000; **do not use outcomes to change recipe or stop earlier**. Also measure historical WikiText-2 `test` 64 chunks at step1200 **only** as a technical reproduction check, not as a new confirmatory endpoint. Train-split 24 chunks at each checkpoint optional when budget allows; log actual AMP-skipped opportunities.
- **Primary outcome:** validation NLL gap `LD(6000)-LS(6000)` (positive favors staged). Secondary trajectory gap at each checkpoint, perplexity/teacher KL/top1, gap slope and direct/staged absolute losses. Reporting several intermediate points does not establish asymptotic convergence. **No hypothesis cutoff** or CI claimed from one order.
- Baseline reproduction: compare step1200 historical D test NLL **5.595722187310457** and S test **4.9009853675961494** with absolute tolerance **0.06** each. Train-only dev at step300 for D ~5.988094, S Q9 ~5.182719 if logged; tolerance0.05. On mismatches, mark technical discrepancy and preserve full raw logs, do not call it a scientific result.
- Both arms must complete6000 scheduled opportunities, record actual optimizer updates, Q3 code histogram and source-code movement at final, finite validation diagnostics, exact first1200 training-order head. No model generations or posthoc hyperparameter search.
- **Limits:** single already-studied seed1729, no inference speed, no quantizer/parameter update to modern BitNet, no portable claim. Validation split is untouched by *these QAT experiments* but may not be independent of the pretrained source corpus. L1 does not by itself answer changing to pretraining data.

## Operational and archival contract

Pin script SHA after committing it; CPU static smoke. Launch one code-pinned HF `a10g-small` job with `2h` timeout (may still cost GPU time). Record ID in `CURRENT_STATE.md`, `AI_HANDOFF.md`, `NEXT_EXPERIMENT_PLAN.md`, `EXPERIMENT.md`. On completion store entire `FINAL_JSON` results in `results/`, mark positive/negative/technical failures honestly, and propose future longer work only if justified. Do not silently launch extra seeds/hardware.
