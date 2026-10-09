# M1 — frozen preregistration: identical Q3 continuation after Q9 steps 250 vs 300

**Registered:** 2026-10-08 Eastern, BEFORE job execution/results.  
**Status:** fixed exploratory seed1729 mechanistic comparison; one Hugging Face A10G-small GPU job at most, up to 90 minutes.  
**Authorizing request:** user instructed "do what you need to do" and specifically requested a copy of the plan in GitHub.  
**Planning context:** [NEXT_EXPERIMENT_PLAN.md](../NEXT_EXPERIMENT_PLAN.md) and [CURRENT_STATE.md](../CURRENT_STATE.md).  
**Parent implementations:** [v11](../smollm2_v11_hybrid_factorial.py), [C1](../c1_smol360m_q9_switch250_vs300_seed1729.py).

## Hypothesis and alternative explanations

C1 showed S250 Q9(250)+Q3(950) retained 95.75% of S300 Q9(300)+Q3(900) improvement over D at equal **total** 1200 step opportunities. T1 showed only 69.40% step300 disagreement-mask recall at 250. C1 changes **both Q9 and Q3 duration**, and lacks final position-level code comparisons.

M1 instead asks what the two Q9-prepared *states* do under the **same 900-step Q3 training operator**. It does not equalize total training opportunities or Q9 data exposure.

- H1 (prep difference matters): shorter Q9 preparation is worse than longer prep given an identical 900-step Q3 continuation.
- H2 (repair/rediscovery): Q3 continuing from the t250 state independently finds a substantial fraction of the ternary code identities visible at t300 Q9 preparation.
- H3 (multiple solutions): both endpoints perform similarly but differ in their final ternary code assignments.
- H4 (C1 extra-Q3 compensation): S250 benefits materially from the 50 additional Q3 steps in C1. M1 may support but does not **alone** prove this, because data exposures and post-switch schedules are deliberately different than C1.

## Methods fixed before job

- **Model:** `HuggingFaceTB/SmolLM2-360M-Instruct` source BF16-rounded, teacher fixed in FP16, student master weights FP32, targeting same `nn.Linear` except `lm_head`; other params frozen.
- **Data:** WikiText-2, deterministic training-order shuffle seed1729, 128-token input sequences from v11 loader. Fixed train-split LR validation chunks (24 x 128 tokens), held-out existing WikiText test chunks (64 x128 =8192 tokens) **not used for selecting protocol**. Evaluation on this previously seen test split is descriptive, not independent.
- **Quantizer:** v11 `SharedScaleQuant` quantization, rowwise initial alpha from source, STE; start Q9 nine states, restore saved **original initial Q3 scales** when constructing each Q3 continuation, preserve only FP32 masters. Do not carry Q9 Adam/scaler state.
- **Optimizer/objective:** AdamW betas(0.9,0.95) wd0, gradient clip1, 35% CE +65% fixed-teacher KL, FP16 CUDA autocast/GradScaler. Q9 prep uses shared v10 global LR formula: linear warmup through100 to1e-3, then cosine to1e-4 by global1200.
- **Single continuous Q9 trajectory:** run 300 preparation step opportunities, saving complete FP32 master weights and original-scale Q3 projected code assignments at t250 and t300. Validate/check native Q9 metrics at those t before deleting the model.
- **Common continuation:** instantiate two new Q3 models sequentially from these saved master states; for **both** continuations use exactly 900 `train_chunks[300+j]` batches for j=0..899, and same per-step `matched_lr(300+j)`. Restore original Q3 scales and create fresh optimizer/GradScaler before step1. This intentionally *skips* exposure to training chunks 251..300 in the P250 preparation trajectory: **P250 total 250+900=1150** and **P300 total 300+900=1200**. No train-chunk/lr restart inside either continuation. This differs from the previously tested equal-total-budget C1 S250 arm (Q3 chunks251..1200 at global LRs251..1200).
- **RNG:** reset seed before each constructed Q3 arm (but no user-facing stochastic sampling); same training batch order and optimizer hyperparameters. Generate no model samples. No additional Gaussian, gridward or prototype intervention.
- **Final heldout:** evaluate each endpoint once on the existing fixed 8192-token slice, after exactly 900 scheduled Q3 opportunities; log PPL/top1/KL, plus train-split val at Q3 continuation steps 0,300,600,900.

## Measurements, prespecified tests and technical acceptance

**Primary endpoint:** `loss(P250_900) - loss(P300_900)` on held-out CE. Report raw numeric difference. **No post-hoc quality threshold**: we are testing mechanism, not tuning early switch.

**Code redistribution/rediscovery (whole targeted backbone, position-level):**
- measure prep Q3 code arrays `S250` and `S300` against original source `S0`, and exact `S250 != S300` count.
- at each start-differing position, record final Q3 codes `F250`, `F300`, and report `P(F250==S300 | S250!=S300)`, `P(F300==S300 | S250!=S300)`, `P(F250==F300 | S250!=S300)`. Record source→late-Q9 transitions including newly selected, reverted, both-changed/different groups. Also final overall agreement, and top per-layer disagreement fractions.
- Explicitly **do not equate a final code match with causal importance** or infer an independently optimal ternary assignment.

**Checks:**
- 314,572,800 targeted weights in all snapshot/code comparisons; exact consistent layer names/shapes; original-scale projected codes in {-1,0,+1}, no unexpected unmatched source codes.
- t250/t300 native Q9 prep validation compares to historical approx 5.3908295 and 5.1827188 within 0.035; original-scale projected Q3 source-code movement t250 approx0.0438033962 and t300 approx0.0488410886 within 0.002 absolute.
- P300_900 final heldout should reproduce historic Q9(300)→Q3(900) loss **4.9009853675961494** within 0.06 nats. If not, mark technical discrepancy and do not interpret the primary contrast.
- 900 scheduled Q3 opportunities per arm; AMP skips explicitly counted (do **not** claim identical effective optimizer updates unless measured).
- Finite NLL, correct final Q3 code range, exact chunk/order and LR equality by construction. Record source code SHA, job ID, environment and raw machine-readable JSON.
- Predeclared outcome interpretation: Positive delta (P250 worse) supports preparation differences matter **under identical Q3 continuation**; near zero or negative weakens necessity of last50 Q9 steps for that operator. Final-code agreement/mismatch provides descriptive evidence for rediscovery vs alternative solutions, not a causal proof.
- On infrastructure/model/dataset/OOM/metric mismatch, preserve technical failure and do not change scientific settings quietly. One job only, no automatic replication or sweep.

## Archival and next gates

Frozen preregistration before GPU. Pin code SHA, static-check script, run one bounded A10G job in namespace `codeflash85` with max90m timeout. Record job ID and starting status in [CURRENT_STATE.md](../CURRENT_STATE.md), [AI_HANDOFF.md](../AI_HANDOFF.md), [EXPERIMENT.md](../EXPERIMENT.md). On completion save raw `FINAL_JSON` preserving scientific metrics verbatim and one interpretation report. Update research summary and all status documents. Any fresh-order confirmatory study or longer-horizon / new-dataset test requires new protocol and separately specified GPU authorization. **M1 is exploratory on the familiar seed/test set.**
