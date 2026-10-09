# G1 preregistration — Q3 depth-by-scale-trainability factorial

**Preregistered:** 2026-10-09 EDT, before new GPU submission or outcomes. **Authorization:** user explicitly approved launching the next two bounded GPU experiments (G1 and S1), not R1.  
**Status:** FIXED PROTOCOL, G1 outcomes unknown.  
**Parent code:** `ternary_pet/replications/smollm2_v13A_depth_sweep_seed1729.py` (v13A depth sweep), existing [v13 result](../results/run_v13_depth_firmness_summary.md). **Question:** is the within-Q3-bin depth benefit present **without** adapting the learned per-row scales?

## Fixed arms and recipe

- Single seed/order **1729**, `HuggingFaceTB/SmolLM2-360M-Instruct`, original WikiText-2 exact seed and original 8192-token heldout test. Original BF16 rounding, target same linear layers except lm_head, fixed FP16 teacher, 35% CE/65% KL, STE Q3 quantizer, AdamW betas (0.9,0.95), wd0, 1.0 norm clip, GradScaler, unchanged global v10 1200-step warmup100/cosine LR.
- **Preparation:** reproduce existing separate 300-step D direct Q3 and S Q9 master states, both driven by identical first300 training chunks/order/LR. Define `M={j | original-source-Q3-code(D_j) != original-source-Q3-code(S_j)}` using *common original* row scales. Keep D masters off M; on M put **Q9-chosen Q3 target code** at v13 normalized depths `d=0.03` (shallow) and `d=0.50` (moderately deep), using the original scale. Starting projected Q3 codes/forward diagnostics must be identical across all four arms and agree with S codes on M.
- **Four matched 900-step continuations**, steps301..1200 and identical LR, original Q3 row scales at start and fresh Adam/GradScaler:
  - `d003_learned` and `d050_learned`: FP32 masters **and row scales** trainable (v13 policy).
  - `d003_frozen` and `d050_frozen`: FP32 masters trainable, original `raw_alpha` row scales frozen by `requires_grad_(False)` and **omitted from Adam param groups** for *all 900 steps*.
- Each arm instantiated and continued sequentially on the same single GPU; exact same model/preparation/data, objective and quantizer, no extra unfrozen modules, no change in weight decay, LR, or evaluation tokens. No additional d=0.25/1.0 sweep here, no new seed unless separately authorized.

## Outcomes, checks and interpretation

- **Primary interaction:** `[(NLL_shallow - NLL_deep)_frozen] - [(NLL_shallow - NLL_deep)_learned]`. Also report both within-policy depth gains and raw final NLL/PPL.
- **Secondary:** Q9 target code survival on M at Q3 continuation100/300/900, measured both with each arm's current scales and common original scales, code histograms and source movements; alpha-state drift (exact zero for frozen) and AMP skips/effective optimizer updates. Per-level survival zero vs nonzero.
- **Technical acceptance:** d003 and d050 starting Q3 code projection and score identical within 1e-5; all start codes agree with S projected Q3 on M; scales at all ends equal original when frozen; frozen scale params have no gradient and are excluded from optimizer; 900 scheduled steps each; all final evals finite and Q3. Learned d003/d050 should reproduce v13 historical values (~5.4948 and ~4.8850 held-out CE) within **0.06 nats**, otherwise flag technical discrepancy and do not overinterpret. Verify full mask/target weight count matches original v13 scale and mask positions, and report any mismatch.
- **Interpretation:** If frozen-scale depth gain persists, learnable row-scale optimization is *not necessary* for the depth effect in this recipe. If it vanishes, row-scale training is a plausible mediator, not proven sole cause. Neither outcome explains Q9-vs-Q3 representable-range confound or establishes causality of any particular mask sites. The same familiar seed/eval is exploratory.
- **Compute:** exactly **one HF `a10g-small` job, 2h timeout**; syntax/static preflight before launch. Preserve originally parsed final JSON and job provenance, even if technical check fails. Do **not** launch R1 or extra seeds automatically.
