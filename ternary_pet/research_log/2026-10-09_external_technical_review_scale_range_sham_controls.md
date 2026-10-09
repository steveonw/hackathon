# External technical review — scale-gradient, range, reset and evidence controls

**Recorded:** 2026-10-09 (conversation date) as a *methodological critique / research plan*, **not** as experimentally observed findings. Another AI reviewed the repository at `e8e5ce1abd61a7848de8e330c834878b9e440358` without rerunning GPU work. User supplied its six specific critiques and requested an assessment implicitly by sharing the review. This entry preserves each concern and our independent verification against the code. **No new GPU jobs were requested or launched by this note.**

## What remains credible

The replicated finite-budget staged-versus-direct benefit on the original SmolLM2/WikiText quantizer and the transfer to a small sample of FineWeb-Edu are observations, with chronological negatives preserved. Mechanism/interpretation claims are narrower: the exact causes of depth/position advantage, gradient path, range extension, scale learning and optimizer-reset policy are not separately isolated by the archived studies. The critique does not negate the numerical staged benefit in those fixed recipes.

## 1. Highest priority: within-bin depth also changes learnable-scale gradient

**Verified implementation:** `smollm2_v12_code_identity_position.py` `SharedScaleQuant.forward`:
`a=softplus(raw_alpha)+1e-7`, `z=clamp(w/a,-0.99,0.99)`, `hard=round(n*z)/n`, `qcode=z+(hard-z).detach()`, `q=qcode*a`. Both FP32 master and `raw_alpha` are trainable. For `|w/a|<0.99`, with STE surrogate derivative `dq/da = hard-w/a`. If a master lies at `w=a*hard`, that one position contributes zero through `dq/da` (although loss weighting, other weights, softplus and clipping remain). **For clipped weights**, this formula does not apply in the same form; derivative of `z` wrt `a` vanishes.

**Interpretational correction:** v12/v13 depth changes can affect both interior code survival **and the scale-optimization trajectory**, so "depth is important exclusively because assignments persist" has not been proven. v13 notes fixed-original-alpha and learned-alpha final code-survival measurements are very close: this is evidence against *end-stage boundary drift being the main cause* of the survival statistic, not evidence against altered scale gradients during optimization.

**Proposed discriminating experiment G1 (UNRUN):** Reproduce v13 code-identity/depth arms at `d∈{0.03,0.25,0.50,1.00}`, original per-row Q3 scales restored identically and **frozen for all continuation steps** (`raw_alpha.requires_grad_(False)` / leave out optimizer), versus the historical learned-scale policy as a matched factorial. Same projected starting ternary forward output across depth arms, same saved preparation masters, same exact Q3 continuation batches, learning rates and reset semantics, same seed1729 diagnostic replication. Validate initial code/forward identity, fraction originally clipped, final code survival under **original and learned (when applicable) scales**, final held-out loss and train-split trajectories, scale gradient RMS in the learned-scale arm, training skip counts and input checks. If the depth curve persists with frozen scales, pure scale adaptation is *not required* to see it in this setup; if it collapses, scale optimization mediates/contributes but a collapsed result does not alone identify why. A frozen-scale configuration is a different trained baseline; report within-policy depth contrasts, not an unqualified cross-policy quality claim. Fix the original scales *before* continuation; do not re-estimate per-arm scales from different master states.

## 2. High priority: Q9 differs from Q3 in representable range as well as levels

**Verified:** with `n=1.5`, ternary outputs `{-2a/3,0,+2a/3}`; with `n=4`, nine-state outputs `{-a,-3a/4,-a/2,-a/4,0,a/4,a/2,3a/4,a}` for suitable non-clipped inputs. The Q9 max magnitude is 1.5 times Q3 at the same `a` (and weights are clipped internally). This is a **real confound** in interpreting nine-state preparation as *only* an increase in precision.

**Proposed R1 (UNRUN):** Paired Q9-wide (historical, ±a), Q9-range-matched (nine levels evenly spaced ±2a/3), and Q3 direct, with same source, data, optimizer, seed(s), fixed or controlled scale initialization and downstream Q3 projection. Preserve exactly **nine distinct output levels** and match saturation/STE derivative treatment carefully. Merely setting `n=6` with the current `clamp(z,-.99,.99)` is **not** a valid nine-state range match: that construction produces up to **13** integer codes (−6..6). Use explicit clipping of target integer code to −4..4 and a documented surrogate gradient/clipping interval. Check all output levels and saturation counts numerically. Because you cannot simultaneously hold range, spacing, threshold locations and clipping gradient identical across Q3/Q9, report which axes are controlled and measured instead of promising perfect isolation.

## 3. High priority: sham Q3 switch at step300 on the new FineWeb distribution

**Verified:** F1's `f1_run_arm` resets original source ternary row scales and fresh AdamW/GradScaler at step300 only if `q9_prep>0`; direct Q3 runs continuously. The observed F1 FineWeb effect is therefore the value of the **whole historical Q9-staged-with-reset recipe** versus **historical tuned direct-Q3 continuous**. Older v11 hybrid/continuation experiments used fresh optimizer/scale reset for both reconstructed paths, already supporting more than a trivial reset-only explanation **on the old WikiText dataset**. They do not prove that resetting is irrelevant on FineWeb-Edu.

**Proposed S1 (UNRUN):** Keep the frozen F1 FineWeb document split/seed1729/source/objective and all 1200 training chunks fixed. Add `D_sham`: direct Q3(1..300), then restore **original** Q3 scales and reset AdamW/GradScaler at step300, continue Q3(301..1200) at global LRs. Run a transparent 3-arm comparison using independently re-run D-continuous and Q9→Q3 as anchor arms if resources permit; otherwise compare against the archived matched F1 results with exact control reproduction checks and limitations. Distinguish **optimizer reset only**, **scale reset only**, and **both** in a factorial follow-up only if the initial sham result warrants the added cost. L1 also resets only S; do **not** retrospectively change its already-running/committed script. Analyze as an additional follow-up after L1.

## 4. Medium-high: random position controls are not boundary-margin matched

**Verified:** v12 `build_matched_random_plan` matches code transition counts per layer, drawing candidates from sites with matching source D codes outside the true D/S mask. It does **not** match original or prep-time signed decision-boundary distance, learned row scale, row identity, or sensitivity gradient. Hence the existing negative random control is strong against *naive random location* but not against more highly matched position-selection null hypotheses.

**Proposed P1 (UNRUN):** After freezing a target code source, calculate distance-to-nearest relevant Q3 threshold normalized by `a` for each candidate, row-ID/scale bucket and optionally absolute gradient norm proxy on a separate fixed train subset. Construct control masks matched by layer, source→target Q3 transition, margin bucket and row-scale bucket, log whether sufficient controls exist and their true balance/overlap. Matched-random seeds and dose/coverage checks. Compare identical prototype-depth construction and downstream Q3 continuation. Beware that gradient sensitivity estimated **after observing outcomes** introduces selection bias; freeze features/selection before the test.

## 5. Medium-high: M1 final code adoption compares different learned scale references

**Verified:** M1 preparation projected into original-source fixed Q3 scales; at the end `snapshot_codes(qs)` used each model's **learned current** row scale. Thus reported **44.16%** of late-Q9 choices in the early continuation, and final-finale concordance, combine master movement and scale-boundary movement. The statistic is descriptively correct under each arm's **actual inference codes**, but it is **not** a common-boundary explanation of assignment rediscovery.

**Proposed M2 instrumentation (UNRUN):** When rerunning a justified mechanism experiment, compute final Q3 codes **twice**: once at each learned end scale and again reprojected through the **same original** Q3 scales. Report rediscovery and all pairwise concordances under both references, including disagreement driven solely by scale choice and per-layer counts. Optionally report both code-identity and final FP32-master normalized margin. The archived M1 JSON does not contain full final masters or final arrays, so these exact additional statistics cannot simply be reconstructed without new data/snapshots.

## 6. High priority for future inference: keep per-document evidence, versions, checkpoints

**Verified:** F1's saved `FINAL_JSON` reports overall scores and aggregate code statistics; the model is deleted at arm completion. The FineWeb heldout uses just **21 distinct QAT-heldout documents**, 128×128 target tokens; the dev split is **3 documents**. Additional seeds on the **same documents** can test order/RNG robustness, **not** independent evaluation-document uncertainty.

**New future data artifact contract (UNRUN):** Preserve per-example/per-document token counts and summed CE / KL, stable hashed doc IDs, source data checksum/order, run metadata `torch`/`transformers`/`datasets`/`accelerate`/`huggingface_hub` versions, CUDA/GPU, model revision and code Git SHA, optimizer/curriculum/schedule including AMP skip details. For model resumption/generation, save quantized model **checkpoint in a proper Hugging Face model or artifact repository/bucket** with access, integrity and storage budget specified; do not put multi-hundred-MB model weights in GitHub `results/`. For small document samples, use **paired document-level** estimates with caveat on number of independent docs and pretraining overlap. Do not claim you can reconstruct missing checkpoints or per-document scores from aggregate archived results. Existing active F2/F3 scripts are immutable/pinned, so do not silently modify them while running.

## Scientific sequencing and limitations

**Recommendation:** Prioritize G1 fixed-vs-learned-scale *depth* discrimination for mechanism; add S1 sham-reset Q3 control before claiming independent FineWeb staging effect; next R1 range-match nine-state; then P1 boundary-margin-match and M2 common-scale destination instrumentation. For durability assess separately the already launched L1 and FineWeb F2/F3 before choosing more GPU. With limited budget G1 and S1 are the most targeted objections; run none until user approves specific cost and scientific choices.

**More rigorous conclusion:** measurable finite-budget Q9→Q3 improvement for specific quantizers and recipes; neither scale-gradient mediation, range-mediated benefit, optimally selected weight identities nor generalizable compression claims are established by existing data. Keep original positive and negative reports intact, make no posthoc reclassification of v12/v13 effects.

## References

- [v12 depth/position experiment](../smollm2_v12_code_identity_position.py), [v13 depth and firmness sweep](../results/run_v13_depth_firmness_summary.md).
- [F1 FineWeb script](../f1_fineweb_edu_qat_seed1729.py), [F1 raw JSON](../results/run_f1_fineweb_edu_direct_vs_staged_seed1729_2026-10-09.json).
- [M1 code dynamics script](../m1_smol360m_equal_q3_continuation_seed1729.py), [M1 result summary](../results/run_m1_equal_q3_continuation_seed1729_summary.md).
- [authoritative current state](../CURRENT_STATE.md), [working jobs](../NEXT_JOBS_PLAN.md).
