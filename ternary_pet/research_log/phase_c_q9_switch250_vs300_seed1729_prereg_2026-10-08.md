# Phase C preregistration: Q9→Q3 switch at step 250 versus step 300

**Recorded:** October 8, 2026, Eastern time. **Status:** frozen exploratory one-order protocol before new GPU launch.  
**Provenance:** [T1 result](../results/run_t1_q9_discovery_timing_seed1729_summary.md), [v10 matched-schedule control](../replications/v10_schedule_matched_q9_aggregate_summary.md), [strategic note](2026-10-08_q9_assignment_discovery_timing.md).  
**Selection disclosure:** step 250 was selected *after* seeing T1 step-300 mask overlaps and prior v10 held-out outcomes. This test is **exploratory**, not independent confirmation. A future independent seed/data confirmatory preregistration is needed before broad scientific claims.

## Research question and falsifiable outcomes

Can 250 Q9 preparation steps followed by 950 Q3 steps retain most of the final benefit of 300 Q9 steps plus 900 Q3, with exactly 1,200 **scheduled** optimization opportunities per arm?

- **Primary quantitative comparison:** final held-out Q3 negative log likelihood, `L250 - L300` (positive = earlier switch is worse). **Exploratory preservation rule:** `L250-L300 <= +0.07` nats/token **AND** `(LD-L250)/(LD-L300) >= 0.90` when `LD>L300`. Neither threshold has a formal power guarantee. Report the raw paired difference and the ratio even if criteria fail.
- **Secondary outcomes:** PPL (exp of loss), top-1 agreement to teacher, teacher KL, train-split validation time course, quantized Q3 code movement, transition pre/post validation shock, optimizer-skip counts. Do not substitute training-batch loss for the held-out primary endpoint.
- **Competing explanations:** even with 69.4% mask recall at t250 (hindsight relative to t300), early extra Q3 updates may compensate for missing assignments; or missing last-50-step selections may be essential. T1 mask recall is not assumed equivalent to recovered benefit.

## Frozen conditions and arms

**Model:** `HuggingFaceTB/SmolLM2-360M-Instruct` (seed/order **1729**). Source is BF16-rounded. Quantized backbone targets the same `nn.Linear` modules except `lm_head`; frozen remaining params; FP32 learnable master weights and per-row scales. Fixed BF16 source teacher (loaded in FP16 for compute). WikiText-2 train corpus, 128 input-token blocks, identical 1,200 seed-shuffled training chunks per arm. Same parent data loader as v10/v11. Fixed 24 train-split diagnostic chunks; fixed separate **64 ×128 = 8192 held-out test tokens**, evaluated once after all training per arm. No prompt-based tuning.

**Objective:** 0.35 CE + 0.65 teacher KL. AdamW beta=(0.9,0.95), weight decay 0, grad clip 1.0, FP16 CUDA autocast + GradScaler. LR shared by global index 1..1200: linear warmup to 1e-3 at step100, cosine down to 1e-4 at step1200. **The LR schedule is never restarted at a switch.** All arms see the exact same training-chunk order.

Three arms in one bounded GPU job:
1. **D:** direct Q3 for 1,200 global step opportunities, optimizer and learned scale continuous throughout (existing tuned v9 baseline); no artificial restart at 250/300.
2. **S250:** Q9 for global steps 1–250; at the switch retain FP32 masters, reset per-row Q3 scales to their original BF16-rounded-source Q3 scale state, construct a fresh AdamW and GradScaler; use Q3 for 251–1200 (**950 Q3 steps**).
3. **S300:** Q9 for steps 1–300; same restoration/fresh optimizer at the switch; Q3 for 301–1200 (**900 Q3 steps**). This is the historical v10 positive arm re-run inside the job.

S250 and S300 differ **only** in which representation is used at global steps 251–300 and the consequent reset time and Q3 continuation length. Reset timing is part of the candidate recipe; this protocol does **not** independently isolate optimizer-reset timing as a causal variable.

Each arm uses the same seed reset before model construction, data source, teacher, learning-rate indices, quantizer and 1200 scheduled opportunities. A GradScaler skip is recorded if the scale decreases at an opportunity; actual successful optimizer updates may be fewer. Do not claim precisely 1200 effective updates if skips occur.

## Diagnostics and hard checks

- Record native-Q9 validation just before each Q9 switch, then after original-Q3-scale restoration on the **same train-split diagnostic chunks**, with **no update** between. Record transition loss shock; do not confuse with same-batch train loss.
- Record fixed train-split Q3 validation on each arm at global steps **250/300/600/900/1200** when applicable; at 250/300 include phase identity (Q9 vs Q3). These repeated dev diagnostics are diagnostic only, not used to choose more arms or change the LR.
- Record step/opportunity counts, skipped opportunities, native Q9 flip snapshots, Q3 movement after switch, complete finalized loss/PPL/top-1/KL, and code movement vs source in Q3.
- Validate same training order, unchanged scheduled global LR, source code identities, output Q3 codes at every final model, model finite metrics, completion of all three arms. Compare paired D final against historical `5.595722187310457` and S300 against historical `4.9009853675961494` on held-out; report any mismatch. Predesignate a **0.06-nats absolute reproducibility tolerance** for both. A larger mismatch means **technical discrepancy**, not a positive/negative result; preserve all logs, avoid silent tuning.
- **No early stop on observed improvement.** Predefine both preservation criteria above and report all outcomes, even when S250 is worse. Compare against the within-job D, not exclusively against historical baselines.
- **Scope:** one seed, fixed WikiText-2 heldout, previously observed. Outcomes must be labeled exploratory, even if numerical tests pass. No broader claims about architecture, asymptotics, generation quality, or use in production.

## Resource authorization, stop rule, recording

One Hugging Face `a10g-small` job under user `codeflash85`, maximum **90 minutes**. User explicitly authorized "sure do it" following the recommendation for this switch-time comparison. Do **not** automatically run seeds 271828/424242, new gridward/Gaussian arms, Phase B position-split causal tests, or further sweeps.

Pin immutable Git code commit before submission. After completion inspect status, save full `FINAL_JSON` unchanged with provenance, update `EXPERIMENT.md`, `AI_HANDOFF.md`, `RESEARCH_REPORT.md`, README and detailed result summary, preserve negative or technical outcomes, and propose independently confirmatory next steps without auto-running them.
