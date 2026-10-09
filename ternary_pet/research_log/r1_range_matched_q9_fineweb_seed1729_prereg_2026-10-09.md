# R1 preregistration — Nine-state Q9 with Q3-matched representable range

**Date / timing:** 2026-10-09 EDT; saved **before GPU submission and before seeing R1 results**.  
**Authorization:** user: "ok next job", following the confirmed G1/S1 experiments and recommendation of a range-matched nine-state test.  
**Status:** frozen protocol; one scoped Hugging Face `a10g-small` GPU job, **2-hour timeout**, three sequential arms. NO additional seeds, curricula, sweeps, grid sizes or retrials auto-launched.

## Motivation and unresolved confound

Existing staged Q9→Q3 improves finite-budget held-out NLL on WikiText-2 and FineWeb-Edu, with 3/3 FineWeb training-order seeds positive; long horizon6000 retains a staged advantage. G1 demonstrated a depth effect when Q3 row scales were frozen, and S1 showed the FineWeb direct-Q3 optimizer/scale sham reset does not reproduce the Q9 gain. The original nine-state quantizer has output codes `k∈{-4,...,4}`, `Q9_wide=a*k/4` with max `|Q9_wide|=a`, whereas final ternary Q3 uses `k∈{-1,0,1}`, `Q3=a*(2/3)*k`. Thus **Q9 has 1.5× the output magnitude range at the same row scale**.

**Question:** Is that extra Q9 representable range *required* for the staged advantage, or can a **nine-state** preparation confined to Q3's output range retain it?

## Exact three-arm test

Reuse the successful [F1 FineWeb script](../f1_fineweb_edu_qat_seed1729.py), seed **1729**, pinned SmolLM2-360M-Instruct pretrained model revision `a10cc1512eabd3dde888204e902eca88bddb4951`, public `HuggingFaceFW/fineweb-edu` `sample-10BT` revision `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9`, and WikiText2 validation dataset revision `b08601e04326c79dfdd32d625aee71d232d685c3`. Same fixed document-hash partition and first1200 training chunks (180 FineWeb train docs), 24 dev chunks (3 docs), 128 FineWeb held-out chunks (21 docs) plus 128 WikiText validation chunks; the same seeds, order, teacher, tokenizer, BF16 roundtrip and FP32 masters; 35% CE + 65% fixed-teacher KL; frozen embeddings/norm/head, AdamW betas(0.9,0.95), wd0, clipping1, warmup100 up to 1e-3 then cosine LR to1e-4 over1200.

**Arms, same 1200 scheduled opportunities each:**
1. **D**: direct Q3 for1200, no switch or optimizer/scale reset. Repeat historical FineWeb F1 anchor.
2. **S_wide**: original Q9 for300 then Q3 for900; at300 restore original Q3 source row scales and reset AdamW+GradScaler; original `a*k/4` grid with k=round(4clip(w/a,-.99,.99)) ∈[-4,4]. Repeat F1 staged reference.
3. **S_range**: *range-matched nine-state Q9* for300 then identical Q3 for900, with same step300 original-Q3-scale and optimizer reset as S_wide. During the **Q9-only preparation**, integer code `k=clamp(round(6*clip(w/a,-.99,.99)),-4,4)`, 9 codes and output `q=a*k/6`; max `2a/3`, matches final Q3. The Q3 phase uses exactly original F1 Q3 quantizer, regardless of preparation type.

**STE / clipping:** maintain original per-weight prequantizer `z=clamp(w/a,-.99,.99)` and `q=a*(z+(hard-z).detach())`, with `hard=k/4` for wide, `hard=k/6` for range-matched, `hard=k/1.5` for Q3. This preserves the **same STE surrogate** `dq/dw` and clipping interval in normalized `w/a` for wide and range-matched, including inside regions where k is saturated at±4. In particular, the range-matched code clipping at `|k|=4` **must not** be implemented as `clamp(z,-2/3,2/3)` because that changes the weight/scale surrogate gradient above 2/3. Code clamp is on `round(6z)` only. No extra trainable params, activation-level changes, optimizer tweaks or hidden alternate clip thresholds.

**Interpretive limitation, registered up front:** Q9-wide and Q9-range have nine output states but different spacing and quantization thresholds. Neither exact code-threshold locations nor hard-output assignments can be matched simultaneously with range and state count. This experiment tests whether a *wider representable range is necessary*; it is **not a pure ablation of range alone** or a demonstration of an alternative general-purpose quantizer.

## Prespecified metrics

**Primary:** `NLL(S_range)−NLL(S_wide)` on the **same held-out FineWeb documents** at final step1200; positive means range-matched loses quality relative to wide. Also report `NLL(D)−NLL(S_range)` and `NLL(D)−NLL(S_wide)`; staged gains positive. The main scientific interpretation distinguishes whether S_range retains a meaningful staged advantage over D, not only whether it exactly ties S_wide.

**Secondary:** paired WikiText validation NLL/PPL, fixed FineWeb train-split dev trajectories (300,600,900,1200), Q9 native dev and immediately projected original-scale Q3 dev at switch, Q9 preparation stage code histogram, fraction of codes saturated at±4 and would-be raw rounding clipped, fraction of FP32 normalized weights exceeding Q3 range `|w/a|>2/3`, master-weight / raw-scale statistics, final Q3 code histogram and source-code change, AMP skipped/effective optimizer updates.

## Technical accept/reject rules

- Before paid compute, run CPU static AST check **and** CPU toy-autograd unit checks: range-matched Q9 has exactly nine reachable codes `-4..4` across a grid of `w/a`; forward amplitudes exactly `k/6`, wide `k/4` and Q3 `2k/3`, with no codes outside the desired range; compare gradient of FP32 master `w` and row scale `a` with the analytical STE for unclipped weights; separately check behavior after original normalized±.99 clipping and at code saturation `|k|=4`; verify switching to Q3 makes `range_matched` behavior irrelevant.
- Same model and FineWeb document-ID split, first16 training-order head `[221,850,89,747,685,1055,781,170,233,62,421,806,1031,1187,683,619]`, 1200 scheduled steps each; source/model/dataset revisions exact. Both S arms switch after300 Q9, reset original Q3 scales and fresh optimizer/GradScaler, both finish Q3. Final losses finite.
- Reproduce archived F1 historical FineWeb D NLL `5.7666598074138165` and S_wide NLL `4.996255073696375` within **0.03 nats/token**, and secondary WikiText validation D NLL `6.558238908648491`, S_wide `5.695225466042757` within0.03. If anchors fail, label R1 `valid_for_science=false` (technical comparability failure). Preserve all outputs/logs including negatives or errors. No alternative seed automatically.
- Require exactly nine code values reachable in R1 Q9-range unit tests; while training verify no Q9 integer code outside±4; `range_matched` flag present only during nine-state prep; after switch same historical Q3 function.
- Model may have previously encountered FineWeb source documents in pretrained-model pretraining; heldout 21 documents disjoint from this experiment's QAT training docs but not necessarily novel to underlying model. Seed1729 and heldout set previously viewed; single-seed explanatory experiment, not new independent evaluation.

## Run and result stewardship

Freeze code at a separate immutable Git commit before submission and link it in [CURRENT_STATE](../CURRENT_STATE.md). One A10G-small job max2h, no duplicates. When terminal: extract exact `FINAL_JSON`, verify all preregistered checks, archive raw JSON and job provenance, add summary to `ternary_pet/results`, and update `AI_HANDOFF`, `EXPERIMENT`, README and public reports. Do not conclude wider Q9 output range is the *sole* cause even if S_range loses; quantization cell spacing/thresholds also differ.
