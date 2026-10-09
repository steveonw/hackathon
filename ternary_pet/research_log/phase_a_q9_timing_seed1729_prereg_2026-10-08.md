# Phase A preregistration — when Q9 selects future ternary assignments

**Recorded:** October 8, 2026 (EDT), before job submission or new result inspection.
**Status:** frozen design for one exploratory pilot; all future expanded experiments require separate authorization.
**Related history:** [2026-10-08 research discussion](2026-10-08_q9_assignment_discovery_timing.md).

## Question and hypotheses

Q9 preparation is an assignment-search process rather than an attempt to minimize Q9-native training loss. Do useful eventual Q3 code disagreements between matched Q9 and Q3 training trajectories appear early (steps 50–250), or only toward step 300?

Working, *untested* hypotheses:

- H1: part of the final step-300 projected Q3 disagreement mask is present by step 100/200, even while native Q9 assignments continue changing.
- H2: Q9-only code changes and direct-only changes avoided by Q9 have different emergence/retention dynamics.
- Null: earlier masks have low agreement with step-300 Q9 selection; 300-step selection is not reliably predicted from the early state.

This pilot does **not** test the causal advantage of early switching or of the split disagreement subgroups.

## Exact pilot conditions

- SmolLM2-360M-Instruct, source BF16-rounded then FP32 trainable master weights, per-row learnable scales, target all transformer linear layers except lm_head, all other parameters frozen.
- Parent semantics/lineage: v11 matched-preparation script `ternary_pet/smollm2_v11_hybrid_factorial.py`. Paired arms: **D** (direct Q3) and **S** (Q9), **300 update opportunities each**, same deterministic training-order seed **1729** and same first 300 shuffled WikiText-2 train chunks.
- CE 35% + BF16-source-teacher KL 65%, FP16 autocast + GradScaler with persistent FP32 master/optimizer state, AdamW betas (0.9,0.95), weight decay 0, grad clip 1.0.
- Identical **v10 global schedule**: linear warmup 1e-5 to 1e-3 over 100 steps, then cosine decay toward 1e-4 by global step 1200. There is **no** Q9→Q3 continuation and no gridward or Gaussian intervention in this pilot.
- At fixed steps **0, 50, 100, 150, 200, 250, 300**, save in-memory **two-bit-packed** projected Q3 code assignments for both arms, using **original Q3 row scales**. Log native-code movement separately (different representation; not to be conflated with projected Q3).
- Use train-split validation chunks only for reproducibility diagnostics at the final step 300; no held-out test loss, no text generation, no hyperparameter search, and no selecting successful step t after seeing held-out data.
- GPU: **a10g-small**, one pilot job, explicit timeout around **3600 seconds**, Hugging Face namespace `codeflash85`. Do not automatically launch confirmatory seeds or continuation arms.
- Do not upload multi-GB snapshots; instead emit complete per-checkpoint aggregate stats, per-layer attribution summary, and a terminal machine-readable JSON object in job logs. Retrospective overlap uses *in-memory* packed snapshots and does not use future checkpoints in a training decision.

## Primary outputs (all descriptive; no cherry-picked endpoint)

At each t:
1. exact Q3-projected D-vs-S disagreement `M(t)` count/fraction, using frozen original Q3 scales;
2. partition into **Q9-only** (S differs from original but D does not), **D-only** (D differs from original but S does not), and both-changed/different;
3. Jaccard, precision and recall of `M(t)` against final `M(300)` (explicit hindsight oracle);
4. retention of the eventual S(300) code identities at the step-t positions, plus Q9-only/D-only group overlap vs final groups;
5. matched native quantizer movement trace and per-layer breakdown of the step-300 disagreement components.

## Hard checks and failure handling

- Same source, training order and LR across arms, and exactly 300 scheduled opportunities per arm.
- Q3 projected codes must be in {-1,0,+1}, unpacked bit-packing must roundtrip, and all tensor shapes/indexes must agree at all steps.
- At t=0 D and S projected codes must exactly equal the original Q3 source codes and M(0)=0.
- Step 300 code Hamming must reproduce historical v11 seed1729 M ~0.06458076, with tolerance <=0.002 absolute; D and S changed fractions and group decomposition checked for self-consistency.
- Native validation at 300 should roughly reproduce prior v11 D=5.988094 and S=5.182719. Fixed-Q3 projection validation D~5.988094 and S~6.542292; tolerance 0.035 nats. On mismatch, mark technical discrepancy, preserve logs, and **do not** interpret as scientific outcome.
- If GPU OOM/timeout/data/model error occurs, report technical failure. Do not silently change settings or submit more jobs.

## Stop rule and interpretation

One seed1729 pilot only. After completion, archive job ID, exact code SHA, raw FINAL_JSON and concise interpretation, including null or negative findings. Revisit whether Phase B (Q9-only vs direct-only causal interventions) or early transition checks are warranted. Any Phase B/C jobs and multi-seed confirmation require separate approval. Do not relabel these retrospective masks as predictive features available in an online training rule.

**Expected value:** This pilot can establish *when the trajectory resembles the step-300 Q9-selection geometry*, not whether an earlier checkpoint already captures the *full downstream Q3 quality gain*. That needs a distinct common-continuation experiment.
