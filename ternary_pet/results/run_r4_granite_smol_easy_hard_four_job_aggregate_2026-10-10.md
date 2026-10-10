# R4 results — Two model families × previously selected hard/easy seeds: Q9 hybrid outer recovery

**Final state:** FOUR Hugging Face A10G-small scientific jobs **COMPLETED**, all **48/48** reported technical checks passed (`valid_for_science=true` in each), both model-family historical Q3/Q9 anchors reproduced, all original `FINAL_JSON` outputs archived separately. No additional GPU experiments or duplicates launched during result analysis. All jobs ended **2026-10-10 UTC**. Frozen-before-submission [R4 preregistration](../research_log/r4_2026-10-09_granite_smol_easy_hard_two_batch_prereg.md) and original [2+2 batch/handoff](../R4_GRANITE_SMOL_EASY_HARD_JOBS.md).

## Executive finding

**The nonuniform hybrid nine-state preparatory Q9 (H) beats original wide Q9 (W), narrow Q9 (N) and direct Q3 (D) in all four within-model-and-seed evaluations.** Importantly **Granite seed271828 was previously staging-negative** for original wide Q9 vs direct, and remains negative here. The hybrid turns this into a **positive** staged gain.

**Interpretation:** restoring just the two extreme **codebook reconstruction values** to ±α, while leaving narrow Q9's seven central forward values intact and keeping original T4 code thresholds, works in two distinct model families and two preselected seeds per family *under each family's tested optimizer recipe*. It is not proof that only output magnitude matters mechanistically; row-scale gradients, master weights and final Q3 assignments respond to the perturbation during training. These are two **codebook levels**, not two individual weights.

## Primary WikiText-2 validation evaluation

Each model evaluated **exactly 64 aligned first validation chunks (8192 predicted tokens)**, previously unused for R4 model selection; validation chunks are contiguous token windows, **not 64 independent documents**. NLL is per predicted token, lower is better. Signed contrasts `D−H`, `W−H` and `N−H` are positive if H wins.

| Family | Seed | D direct Q3 ↓ | W wide Q9→Q3 ↓ | N narrow Q9→Q3 ↓ | H hybrid Q9→Q3 ↓ | D−H | W−H | N−H |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Granite | 271828 | 5.438636 | 5.577830 | 5.416681 | **5.370064** | +0.068572 | +0.207766 | +0.046617 |
| Granite | 424242 | 5.420453 | 5.349609 | 5.396033 | **5.295450** | +0.125003 | +0.054158 | +0.100583 |
| SmolLM2 | 271828 | 5.293719 | 4.567472 | 5.259229 | **4.480920** | +0.812799 | +0.086552 | +0.778308 |
| SmolLM2 | 424242 | 5.265099 | 4.553739 | 5.273762 | **4.520216** | +0.744883 | +0.033523 | +0.753546 |

For **Granite hard seed271828**, **D−W is negative**: 5.438636−5.577830 = **−0.139194 nats/token**, i.e. original wide Q9 *hurts* relative to direct Q3. Hybrid NLL **5.370064** gives **D−H +0.068572** and **W−H +0.207766**. Granite easy seed424242 hybrid gives **D−H +0.125003** and **W−H +0.054158**. These confirm the historical hard/easy heterogeneity without hiding the negative wide-Q9 arm.

For SmolLM2, hybrid advantages **D−H +0.812799** and **+0.744883**, while wide-Q9 also shows strong gains **D−W +0.726247** and **+0.711360**. H improves over W in both Smol seeds, by **+0.086552** and **+0.033523** respectively, despite the narrower central level spacing.

**No pooled NLL across model families:** their tokenizers, model distributions, LR curves, BF16/FP16 compute and optimizer reset policy differ; comparisons are paired **within** family and seed.

## Paired chunk diagnostics

Per-chunk contrasts preserve all 64 NLL sums and token counts in the raw JSON. Signs here indicate number of chunks where **H has lower NLL** than D, W, N respectively. No significance level or confidence interval claimed from these adjacent chunks.

| Family/seed | H better than D | H better than W | H better than N | Median per-chunk D−H gap | Median per-chunk H−W gap |
|---|---:|---:|---:|---:|---:|
| Granite 271828 | 54/64 | 62/64 | 44/64 | 0.068674 | -0.200114 |
| Granite 424242 | 59/64 | 48/64 | 58/64 | 0.132475 | -0.054909 |
| SmolLM2 271828 | 64/64 | 54/64 | 64/64 | 0.793967 | -0.079033 |
| SmolLM2 424242 | 64/64 | 44/64 | 64/64 | 0.712309 | -0.035391 |

All 64/64 validation chunks favor H over D and N in **both Smol seeds**. Granite hard seed: H improves over W on **62/64** chunks and over D on **54/64**; Granite easy seed: H improves over W on **48/64** and over D on **59/64**.

## Secondary historical WikiText test64-chunk anchors

These are the exact previously reported Granite/Smol test datasets; retained as a **protocol reproduction check**, not independent fresh observations. All four arms ended as original Q3.

| Family | Seed | D test NLL | W test NLL | N test NLL | H test NLL |
|---|---:|---:|---:|---:|---:|
| Granite | 271828 | 5.726725 | 5.841034 | 5.695813 | **5.653762** |
| Granite | 424242 | 5.801920 | 5.740574 | 5.735168 | **5.681365** |
| SmolLM2 | 271828 | 5.622788 | 4.950980 | 5.609046 | **4.888786** |
| SmolLM2 | 424242 | 5.606720 | 4.956250 | 5.619884 | **4.951714** |

All original D/W anchors reproduced prior studies within preregistered tolerances (Granite D ±0.04 W ±0.06; Smol D/W ±0.08). In fact, shown old test losses largely match previously archived baselines to many decimals. This demonstrates protocol comparability **within** each family, but is not a second independent confirmation data set.

## Exact experiments and technical provenance

- **Granite 271828:** HF [`6ac9a43d095c57808930ccfe`](https://huggingface.co/jobs/codeflash85/6ac9a43d095c57808930ccfe), `valid_for_science=true`, **12/12 pass**. Raw scientific record [JSON](run_r4_granite_hard_seed271828_2026-10-10.json).
- **Granite 424242:** HF [`6ac9a443095c57808930cd03`](https://huggingface.co/jobs/codeflash85/6ac9a443095c57808930cd03), `valid_for_science=true`, **12/12 pass**. Raw scientific record [JSON](run_r4_granite_easy_seed424242_2026-10-10.json).
- **SmolLM2 271828:** HF [`6ac9a5c2fee2c90070180ce3`](https://huggingface.co/jobs/codeflash85/6ac9a5c2fee2c90070180ce3), `valid_for_science=true`, **12/12 pass**. Raw scientific record [JSON](run_r4_smol_seed271828_2026-10-10.json).
- **SmolLM2 424242:** HF [`6ac9a5c9095c57808930cdde`](https://huggingface.co/jobs/codeflash85/6ac9a5c9095c57808930cdde), `valid_for_science=true`, **12/12 pass**. Raw scientific record [JSON](run_r4_smol_seed424242_2026-10-10.json).

- **Granite model:** `ibm-granite/granite-4.0-350m` pinned revision `bd8a1497065c0d6ba1ef19af6b0d2b14bacf71c2`, fixed WikiText2 revision `b08601e04326c79dfdd32d625aee71d232d685c3`, hard/easy seeds preselected from prior negative/positive original-Q9 results. Immutable source Git commit `33dfae38bd016946ad5d37bd633fdf02818fbacd`. Granite's established BF16 teacher + BF16 autocast/FP32 masters and fixed LR `1e-4`; original Granite D protocol uses a step300 row-scale and fresh optimizer reset matched to the staged arms. CPU [pin resolution](https://huggingface.co/jobs/codeflash85/6ac9a397fee2c90070180b8d), CPU [nine-state/STE preflight](https://huggingface.co/jobs/codeflash85/6ac9a40b095c57808930ccec) completed before GPU launch.
- **SmolLM2 model:** `HuggingFaceTB/SmolLM2-360M-Instruct` pinned `a10cc1512eabd3dde888204e902eca88bddb4951`, same pinned WikiText2 data revision; immutable source Git commit `65af3057b248f38a6d46ad8b7cd1bc1c52d2e654`. BF16-rounded source, FP16 teacher/autocast with GradScaler, FP32 masters; Smol's previous LR curve warms100 to `1e-3`, then cosines to `1e-4` at1200. Direct D uses original continuous Adam; staged W/N/H restore original Q3 row scales and reset optimizer/GradScaler exactly at300. CPU [Smol 2-script nine-code and gradient QA](https://huggingface.co/jobs/codeflash85/6ac9a578095c57808930cd9c) completed before both GPUs.
- Four arms in **each** seed/family, all at1200 scheduled step opportunities: D original direct Q3; W original nine-state wide `k/4`; N narrow nine-state `k/6` under original T4 decision thresholds; H hybrid seven interior N outputs plus wide extreme `k=±4 → ±α`. All three Q9 arms use the same `k=round(4*clamp(w/α,−.99,.99))` and clipped-z STE; all staged arms Q9 for300 then Q3 for900, restoring source Q3 row scales and resetting Adam. Only forward codebook levels differ, with learned training trajectories subsequently diverging.
- Both families used same cross-family CE35% / frozen-teacher KL65% teacher objective and source Q3 quantization approach, but **not identical precision, LR, or direct-baseline reset policies**.

## Scientific limits and next decisions

1. **Seed selection is post hoc and biased toward Granite's known historical easy/hard cases.** Two seeds per family cannot establish population-level reproducibility or universal architecture transfer, and repeated research on the same WikiText dataset can overfit analysis choices. The effect on Granite is numerically smaller than on Smol; raw NLL levels and gaps are not directly calibrated across tokenizers.
2. **Validation sample is only 64 contiguous chunks (8192 tokens).** These are not independent documents. No per-document bootstrap independent-sample CI should be inferred from token-window signs.
3. **The 2×2 design is a targeted codebook intervention, not a full factor isolation.** H vs N shares central forward outputs and starting discrete codes but changes extreme output reconstruction, which also changes gradients through `raw_alpha`, master evolution, saturation, and Q3 projection. The mechanism behind the benefit remains partially unresolved.
4. **Historically negative wide Q9 result is preserved rather than relabeled.** Granite hard seed shows W inferior to D on both primary validation and original test. Hybrid fixes this within experiment; further new random seeds are required before claiming robust reversal.
5. Existing code and full machine-readable numeric evidence are pinned and archived; **no final restorable weight checkpoints** were retained. Reproduction requires rerunning jobs.
6. Natural next work, **not launched**, is fresh independent model families or genuinely new-corpus/doc-heldout test, plus a mechanism test on saved checkpoints or gradients/code survival, and randomized seeds precommitted independently of previous outcomes. Do not auto-run more GPU jobs without user authorization.

See [CURRENT_STATE.md](../CURRENT_STATE.md) and [AI_HANDOFF.md](../AI_HANDOFF.md) for latest status.
