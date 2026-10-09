# T1 Phase A — When Q9 makes its future Q3 assignment choices

**Experimental status:** COMPLETED, exploratory one-order pilot, passed preregistered reproducibility checks.  
**HF job:** [codeflash85/6ac85534fee2c90070172a41](https://huggingface.co/jobs/codeflash85/6ac85534fee2c90070172a41)  
**Start / completion (UTC):** 2026-10-09 02:45:08 / 02:51:49 (October 8 evening EDT)  
**Model:** `HuggingFaceTB/SmolLM2-360M-Instruct`; **seed/order:** 1729; GPU: A10G-small; code SHA: `2a982be6910d99eaf53d74f6e3ec3dae5807de17`.  
**Frozen prereg:** [Phase A timing test](../research_log/phase_a_q9_timing_seed1729_prereg_2026-10-08.md). **Full parsed FINAL_JSON:** [raw archive](run_t1_q9_discovery_timing_seed1729_2026-10-08.json).

## Scientific question

When do differences between Q9-prepared and matched direct-Q3-prepared **projected ternary assignments** first emerge and persist? This test examines only 300-step preparation in each arm; it does **not** perform 900-step ternary continuation, use Gaussian noise or gridward, evaluate held-out test, or generate text.

At each step 0,50,...,300, projected codes were computed with **identical original fixed Q3 row scales**, regardless of native Q9/Q3 training space. Full 314,572,800 targeted code positions in 224 tensors were represented by in-memory two-bit packs, 78,643,200 bytes per arm/checkpoint. Mask overlaps with step 300 are **retrospective/hindsight**, not future-aware online predictors.

## Primary outcome

| Step | Current D-vs-Q9 mask | Q9-only changed | Direct-only changed | Earlier-mask precision vs M300 | M300 recall (already identified) | Final Q9-only mask recall | Final direct-only mask recall |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 50 | 0.87% | 0.44% | 0.43% | 49.04% | 6.63% | 4.04% | 4.51% |
| 100 | 2.23% | 1.12% | 1.11% | 53.88% | 18.61% | 13.68% | 15.86% |
| 150 | 3.77% | 1.99% | 1.77% | 60.71% | 35.42% | 31.15% | 32.38% |
| 200 | 4.88% | 2.56% | 2.31% | 68.02% | 51.35% | 47.86% | 49.90% |
| 250 | 5.76% | 3.03% | 2.73% | 77.77% | 69.40% | 67.60% | 68.97% |
| 300 | 6.46% | 3.38% | 3.07% | 100.00% | 100.00% | 100.00% | 100.00% |

Definitions:
- `M(t)`: positions where D(t) and S(t) select different Q3 codes.
- **Q9-only:** Q9(t) differs from source Q3 at a position; D(t) retains the source code.
- **Direct-only:** D(t) differs from source; Q9(t) retains it.
- A tiny residual contains both-changed/different-code positions.
- **M300 recall:** fraction of final M(300) positions that are already in M(t); **precision:** fraction of M(t) positions also in final M(300).

At **step 200**, the mask has **4.88%** of all weights, **51.35%** final-mask recall, and **68.02%** precision.

At **step 250**, it has **5.76%** of all weights, **69.40%** final-mask recall, and **77.77%** precision.

**Interpretation:** the step-300 disagreement geometry emerges **gradually**. The early 100-step mask accounts for **18.61%** of M300. Step 250 already overlaps **69.40%** of M300 but misses about **30.60%**. There is no sharp plateau or proof that step 200/250 would yield the same downstream Q3 performance as step 300.

### What happens to both categories?

At step 200, **47.86%** of final Q9-only changes and **49.90%** of final direct-only changes are already visible. At step 250, those figures are **67.60%** and **68.97%**. Both types of selection develop in parallel.

At step 300, the complete disagreement fraction is **6.46%**: Q9-only **3.38%**, direct-only **3.07%**, and both-changed/different **0.00%**. These exactly reproduce the prior v11 seed1729 discrete-code geometry. This result **does not yet** isolate causal benefit between Q9-only and direct-only positions; the v11/v12 interventions treated their union.

## Checks and reproduction

| Check | Observed | Historical reference | Result |
|---|---:|---:|---|
| Direct native validation loss @300 | 5.988094 | 5.988094 | exact |
| Q9 native validation loss @300 | 5.182719 | 5.182719 | exact |
| Direct fixed-original-Q3 validation @300 | 6.004324 | 5.988094 | within 0.035 tolerance |
| Q9 fixed-original-Q3 validation @300 | 6.542292 | 6.542292 | exact |
| D/S disagreement fraction @300 | 0.064580765 | 0.064580765 | exact |
| Original-code agreement @0 | exact | exact | pass |

The **direct native vs fixed-original-scale Q3 loss** differs by **+0.016230 nats**: training updates the rowwise scales, and reloading initial scales for fixed projection changes the forward model slightly. The preregistered 0.035 tolerance passes, and the D/Q9 **discrete projected-code geometry** matches v11 exactly. Record this small difference, rather than claiming every validation score is bitwise identical.

**Final job validity flag:** `valid_for_science=true`. HF status completed with no reported job failure. The fixed held-out test was not evaluated. The full train-split validation protocol was preserved.

## Conclusion and proposed next decision

**Confirmed descriptively (seed 1729 only):** there is a staged accumulation of Q9-specific code choices *and* direct-Q3 code changes that Q9 avoids. M300 recall rises continuously from 6.6% at step 50 through 69.4% at step 250, with increasing early-mask precision.

**Not established:** which mask subgroup creates more of the eventual loss advantage; any online feature able to predict future useful choices; the best Q9→Q3 switch time; whether a 200- or 250-step Q9 phase retains final quality; cross-seed stability of timing.

**Next controlled experiment option, not automatically launched:** freeze Phase B (Q9-only vs direct-only 300-step intervention under common 900-step Q3 continuation) to resolve *what selection does*, or preregister a compact Phase C 200/250/300 switch-time ablation to resolve *when full downstream advantage emerges*. Use train-split validation for method selection and a fresh independent validation/test plan before confirmation. Preserve nulls and negatives. Explicitly approve each paid GPU allocation. **No additional jobs authorized or launched by this summary.**

References: [history / strategic work plan](../research_log/2026-10-08_q9_assignment_discovery_timing.md), [v11 mechanism](../replications/v11_hybrid_factorial_aggregate_summary.md), [v12 identity/depth](../replications/v12_code_identity_position_aggregate_summary.md), [v13 depth sweep](run_v13_depth_firmness_summary.md).
