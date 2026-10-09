# Research history log — Q9 assignment discovery, timing, and next tests

**Recorded:** 2026-10-08 (America/New_York; some archived jobs/results are dated 2026-10-09 UTC)  
**Status:** retrospective interpretation + **draft experimental plan**, NOT completed tests or a frozen preregistration  
**Project:** Ternary Pet, SmolLM2-360M-Instruct and Granite-4.0-350M  
**Compute/repo state at recording:** No new GPU jobs launched for this investigation; existing experiment outputs remain unchanged.

> This document preserves a research discussion and its rationale for future AIs/researchers. It distinguishes reproducible observations, interpretations, and untested ideas. Do not promote hypotheses to results or silently launch experiments from this plan.

## 1. The motivating idea in the user's words

The original intent is **"a rough 9, but a better 3 afterwards"**: the intermediate nine-state network is not the product; it is a search mechanism for *which ternary assignments should eventually exist*. The user explicitly favors **assignment discovery** over a simple more-precision-is-better explanation.

Related user intuitions discussed with other AIs:

- **"Death by a thousand cuts":** many small adjustments can create a large cumulative effect. This is consistent with the v3 gradual-ramp observation (smaller first cliff, damage accumulated as quantization strength rose), but does not by itself prove benefit. In gridward, nine 10% pulls would contract distance to a stationary code target by 1 - 0.9^9 = 61.26% in the idealized no-gradient/no-code-change case.
- **"Gaussian process to a point":** possible exploration followed by increasingly focused commitment. **The exact intended meaning is not confirmed.** Existing G1-9/G1-10 Gaussian arms inject row-scale-normalized training-time noise before ternary quantization and anneal it; they are *not* a proven Gaussian-process method for selecting target codes. Gaussian and gridward have published WinQ-related prior art. Keep this an open idea, not a claimed invention or validated mechanism.

Working research question:

**Can Q9 discover the *right positions and Q3 code identities*, then allow moderately firm placement inside their Q3 regions, while avoiding premature commitment to direct-Q3 decisions? How early does that useful selection emerge?**

## 2. Established evidence already in the repository

1. **v10, three Smol training orders (1729, 271828, 424242):** under the same global warmup/cosine schedule, Q9 for steps 1–300 followed by Q3 for steps 301–1200 beats tuned direct Q3 by mean **0.6723 nats/token** (~**48.94%** paired perplexity reduction), despite Q9 projected into Q3 at step 300 scoring **0.5120 nats worse** on average than direct Q3 at equal compute. Therefore the supported finite-budget effect is *downstream trainability*, not a better immediate Q3 checkpoint. A longer v5 test retained a smaller advantage; permanent/asymptotic superiority is unproved.
2. **v11, three orders:** the step-300 disagreement mask M = {positions where projected Q3(Q9-prepared masters) differs from projected Q3(direct-prepared masters), both with original row scales} comprises mean **6.31%** of quantized weights. Transferring Q9 masters on M alone recovers mean **96.40%** of the full Q9 continuation benefit. This is a causal localization for the combined mask, *not* a causal split of its components.
3. **v12, three orders:** on M, replacing exact Q9 masters with standardized Q3-code prototypes slightly beats exact-master transfer in **3/3** orders (~**106.9%** recovery relative to exact M control); barely crossing Q3 boundaries recovers only ~**18.0%**. Matched-random code changes at different positions are harmful in **3/3** orders. Thus *which position, which code, and within-bin placement* matter.
4. **v13, one Smol order (1729):** a depth sweep gives ~**94.8%** benefit recovery at depth 0.25 and saturates near depth **0.5**; at d=0.5, **97.32%** of selected codes remain after 900 continuation steps. Firming *direct's own* code assignments does not help. Do not claim that survival is a proven sole causal mediator or that d=0.5 is universally optimal.
5. **Granite G1-9/G1-10:** nine 10% gridward pulls improved direct-Q3 in two Granite training orders. **Smol S1-1 negative:** the same intervention on Smol's separately tuned schedule worsened held-out loss **5.5957 → 5.8457** while suppressing sampled flips **78.4%**. Lower flip count does not guarantee quality; architecture, LR schedule, and optimizer reset differ between these experiments.

## 3. New retrospective decomposition: two kinds of decisions

Using the **archived v11 step-300 full-model aggregate overlap fields** (not a new GPU result), separate code-change sets relative to the original source Q3 code:

- **Q9-only:** Q9 changes the projected Q3 code from source; matched direct Q3 leaves it at source.
- **Direct-only:** matched direct Q3 changes the projected code; Q9 preserves the source code.
- **Both-changed/different:** both change a source code but end at different Q3 codes (tiny remainder).
- Positions where both change and agree do **not** belong to M.

| Training order | Q9-only (% of all target weights) | Direct-only (%) | Both-changed/different (%) | Entire D-vs-Q9 mask (%) |
|---|---:|---:|---:|---:|
| 1729 | 3.3849 | 3.0701 | 0.0031 | 6.4581 |
| 271828 | 3.3932 | 2.7849 | 0.0026 | 6.1807 |
| 424242 | 3.3981 | 2.9001 | 0.0028 | 6.3010 |

**New descriptive observation:** the Q9-only fraction is consistently about **3.4%** over these three orders, while direct-only changes contribute another **2.8–3.1%**. This suggests the Q9 advantage *could* depend on both useful changes and **avoiding some direct-Q3 changes**. These fractions are **not** effect sizes: v11/v12 tested the combined mask, and no causal experiment has separated Q9-only from direct-only positions. The small three-order range is not enough to assert a universal constant.

Derivation from v11: Q9-only = S_changed − intersection; Direct-only = D_changed − intersection; Both-changed/different = intersection × (1 − same_final_code_among_both_changed); their sum is D-vs-S projected-code Hamming. All codes are projected using comparable initial Q3 scales.

### Secondary clue from existing preparation traces

Mean **native Q9-code** net displacement since the start of Q9 training across seeds 1729/271828/424242:

| Q9 preparation step | Native Q9 codes changed (%) |
|---:|---:|
| 50 | 2.23 |
| 100 | 5.71 |
| 150 | 9.90 |
| 200 | 12.77 |
| 250 | 14.98 |
| 300 | 16.73 |

Native Q9 codes are still reorganizing at step 300. **Crucial limitation:** nine-state code movement is *not* projected Q3 movement and cannot tell us when the useful eventual ternary assignments first appeared. Archived data measure the exact D-vs-Q9 Q3 mask at step 300, not at earlier matched checkpoints. The logs do not establish an earlier optimal transition time. A short-run training-loss ranking is also not a reliable substitute for final Q3 trainability.

### Secondary gridward clue, not established mechanism

Granite's gridward sampled Q3 code-flipping recovers somewhat after its last pull; Smol's remains almost zero late. Smol has much larger typical row scales (~0.206 vs Granite ~0.0234) and a decaying LR; LR divided by row scale is only a **rough proxy**, not an actual Adam update-to-boundary calculation. Premature commitment is plausible but **unproved**, and this is separate from the primary Q9 timing question.

## 4. Proposed experimental work plan — gated and not yet launched

### Phase A — determine when Q9 selects projected Q3 assignments (diagnostics first)

**Aim:** find whether step-300 useful projected Q3 decisions are already present at earlier Q9 checkpoints, without requiring a full downstream continuation for every candidate.

- Start with **Smol seed 1729** as an explicitly exploratory run; after freezing the protocol, use seeds **271828 and 424242** for replication.
- Prepare **matched direct Q3 and Q9** for global steps 1–300 using the **v10 shared LR schedule**, identical pretrained BF16-rounded source, exact training chunks/order, FP32 masters, fixed teacher, and current quantizer semantics.
- At steps **0, 50, 100, 150, 200, 250, 300**, save compact **projected Q3 code snapshots of both arms using the original frozen Q3 row scales**, regardless of each arm's current learned scale. Also record native code movement separately and train-split validation diagnostics as defined in the parent protocol.
- For each checkpoint t, define M(t) = projected-code disagreement between matched Q9(t) and D(t). Decompose into **Q9-only(t)**, **D-only(t)**, and rare both-changed/different groups. Log counts by layer, source→target transition, and exact code identity.
- **Retrospective survival:** compare Q9(t)'s projected codes with Q9(300)'s codes *at identical positions*; report retention/precision/recall of the t-selected disagreements against M(300), including each disagreement category. This is hindsight analysis, **not** a deployable online oracle. Examine rank/order stability and whether the selections accumulate, reverse, or oscillate.
- **Online candidate predictors:** separately evaluate features knowable at t (distance to Q3 boundary normalized by alpha, per-weight gradient/update magnitude, gradient sign agreement over recent windows, change-history persistence). Measure prediction without using future labels at inference. Avoid inferring causality from correlations.
- Resource note: each full 314,572,800-position code snapshot is large if stored as int8 (~315 MB). **Bit-pack ternary codes (2 bits/code, ~79 MB/snapshot)** or stream/chunk snapshots; do not accidentally retain all full FP32 states on GPU or push giant binaries to Git.
- Do not look at the fixed held-out test to choose checkpoint, feature thresholds, mask rules, or training schedules.

**Deliverables:** raw diagnostics JSON plus compact snapshots if needed, stage-0 source pin, validation-only summary table, cross-checks that step-300 D/S masks reproduce archived v11 geometry, and clear separation of native Q9 vs projected Q3 trajectories.

**Phase A gate:** proceed to expensive Q3 continuation only if timing data reveal a plausible earlier interval or if decomposed mask stability is scientifically informative. Predeclare selection rules before validation-driven continuation. If the early masks are unstable, log that negative and do not claim a short Q9 phase is sufficient.

### Phase B — isolate the two roles of the disagreement mask

Use a frozen checkpoint t selected via **training-split validation / Phase A**, starting with **t=300** for mechanism comparability if necessary.

With matched D(t) and S(t) projected into original-scale Q3, construct common-continuation arms:

1. **D control** — direct-Q3 masters at t.
2. **Q9-only group** — on positions where S changed the source Q3 code and D did not, set the Q9-selected code with standardized moderate interior placement (candidate depth d=0.5); otherwise retain D masters.
3. **Direct-only group** — on positions where D changed the source code and S did not, restore the Q9-preserved source code with the same standardized interior placement; otherwise D masters.
4. **Union** of those two groups (plus a separately logged tiny both-changed/different category) at matching interior depth.
5. **Full Q9** prepared masters, and optionally a **matched-random** control preserving layer/transition counts.

Verify each intervention's intended Q3 codes, identical starting forward models for relevant equal-forward comparisons, matching pre-continuation diagnostics where applicable, original Q3 scale initialization, fresh Adam, no accidental changes to off-mask weights, and identical remaining training order. Continue in direct Q3 through **global step 1200** (thus equal total compute for comparisons at a given t). Report paired loss deltas and interactions versus D; only then distinguish **adding Q9-specific changes** from **preventing direct-only changes**.

**Important caveat:** The d=0.5 prototype is informed by viewed v13 results, so it is a specified reuse of prior evidence, *not* a new blind hypothesis. Pre-register the exact intervention and random control before compute.

### Phase C — validate discovery timing rather than only endpoint localization

Only after Phase A/B: compare at most a small preregistered set of Q9→Q3 switch times (candidate examples 100, 200, 300). Keep **1200 total update opportunities, one common global LR schedule, original Q3 scales at transition, fresh Adam, and identical ordered chunks**. Earlier Q9 switches necessarily receive more Q3 continuation; state that explicitly. Use validation to select and report the untouched held-out set once for final confirmation. Replicate on fresh seeds/orders, and reserve different datasets for later generalization.

## 5. Decision safeguards and history protocol

- **No GPU launch, paid-compute authorization, or repository/script change is implied by recording this draft.** Obtain explicit authorization for each compute budget/experiment after implementation + preregistration; do not auto-expand to more seeds or variants.
- Before running: pin an immutable Git commit, define primary outcomes, timing of final test access, failure criteria, hardware, memory/timeout budget, and stop conditions. Preserve prior completed outputs verbatim.
- At run completion: record HF job IDs/URLs, raw JSON, seed, code SHA, exact numerical outcomes, validation selection, construction assertions, negative findings, anomalous logs, and an updated handoff link.
- Interpret within-model matched comparisons before cross-model extrapolation. Cross-family comparisons remain confounded by architecture, LR, scale and optimizer schedule until controlled.
- Do not overclaim practical model quality: existing WikiText-2 pilots remain limited and generation is not yet healthy.

## 6. Existing supporting files

- [v3: gradual transition and “death by a thousand cuts”](../results/run_v3_summary.md)
- [v5: longer-training attenuation](../results/run_v5_summary.md)
- [v7: equal-compute immediate Q3 disadvantage](../replications/v7_aggregate_summary.md)
- [v10: shared LR, Q9→Q3 three-order replication](../replications/v10_schedule_matched_q9_aggregate_summary.md)
- [v11: causal localization and raw mask-overlap definition](../replications/v11_hybrid_factorial_aggregate_summary.md)
- [v11 raw seed 1729](../results/run_v11_hybrid_factorial_seed1729_2026-10-07.json)
- [v11 raw seed 271828](../results/run_v11_hybrid_factorial_seed271828_2026-10-07.json)
- [v11 raw seed 424242](../results/run_v11_hybrid_factorial_seed424242_2026-10-07.json)
- [v12: identity and interior-placement replications](../replications/v12_code_identity_position_aggregate_summary.md)
- [v13: interior-depth sweep and firmness controls](../results/run_v13_depth_firmness_summary.md)
- [G1-9/G1-10: Granite Gaussian × gridward results](../results/run_g1_10_granite350m_gaussian_pull_seed424242_summary.md)
- [S1-1: Smol gridward negative](../results/run_s1_1_smol360m_gridward_direct_q3_seed1729_summary.md)

**Hand-off headline:** Q9 appears to choose *different, specifically useful* ternary decisions, not just more of them. The underexplored causal split is **Q9-only changes versus direct-only changes Q9 avoids**. The undermeasured timing question is **when these eventual projected-Q3 choices first appear**. Do not mistake native Q9 flip traces for that timing measurement.
