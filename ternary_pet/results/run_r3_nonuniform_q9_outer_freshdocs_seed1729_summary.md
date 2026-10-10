# R3 — restoring *only* Q9 extreme reconstruction values recovers nearly the full staged advantage

**Completed 2026-10-10 02:04:07 UTC** (Oct 9, 2026, 10:04 PM EDT). Hugging Face GPU [`6ac99666fee2c9007018011f`](https://huggingface.co/jobs/codeflash85/6ac99666fee2c9007018011f), A10G-small, terminal stage **COMPLETED**. **Scientific validity:** `valid_for_science=true`; **all 16/16 technical checks passed**.  
**Immutable source:** `a4abf256c0be9db771e3ec271c1f0a5e6d605816` [R3 script](../r3_nonuniform_q9_outer_freshdocs_seed1729.py).  
**Frozen preregistration:** `bf89baef162f2a9e948ab702a1708c74b856d3a4`, [R3 protocol](../research_log/r3_nonuniform_q9_outer_recovery_freshdoc_prereg_2026-10-09.md).  
**Complete scientific raw FINAL_JSON (including every per-doc loss) plus GPU job provenance:** [R3 JSON](run_r3_nonuniform_q9_outer_freshdocs_seed1729_2026-10-10.json).  
**Permanent execution handoff:** [R3 plan](../R3_ACTIVE_JOB_PLAN.md).

## Main result

**Changing just the two outermost reconstruction values of a narrow nine-state Q9 preparatory grid from ±2α/3 to ±α recovered most of the original wide Q9→Q3 staged advantage** on 32 FineWeb evaluation documents that had **never previously been used in this research's QAT experiments**.

The hybrid `H` has the **same nine integer decisions and same seven central forward output values as narrow `N`** conditional on the same FP32 master weights and row scales. During Q9 preparation (300 steps), only the **±4** code reconstruction magnitude differs (N ±2α/3 versus H ±α); the normalized code-assignment thresholds T4, outer clipped-z STE surrogate, 300-step preparation, scale/optimizer/GradScaler reset, 900-step ternary continuation, fixed source and data remain identical. All final models are **original ternary Q3**; no nine-state inference.

**Result:** on **new 32-document FineWeb evaluation**, `D−W=+0.903540` nats/token direct→original-wide gain, and `D−H=+0.872505` direct→hybrid gain. Hybrid retains **96.57%** of the original W gain relative to D and recovers **96.66%** of the N→W gap. Hybrid **beat N and D on all 32 documents**. W performed better than H on 23/32, with small aggregate loss difference `H−W=+0.031035`.

These outcomes are **strong evidence in this recipe** that high-magnitude extreme Q9 reconstruction values, or training dynamics they enable, drive the bulk of the staged gain; they **do not** prove code-index survival or physical magnitude capacity is the sole causal mediator. The output-dependent `raw_alpha` gradients, learned alpha trajectories, and subsequent master positions diverge once training starts despite exactly matched initial code decisions and fixed z STE surrogate.

## Four arms and fresh-document design

One SmolLM2-360M-Instruct seed **1729**, exact original source model, teacher, FineWeb-Edu sample-10BT source revision, FP32 masters+learned row scales, frozen other modules, 0.35 CE/0.65 teacher KL, BF16-rounded initialization, AdamW and global LR. One D direct Q3 1200 steps; three staged Q9 preparation300→normal Q3 continuation900, resetting original Q3 row scales and AdamW/GradScaler after step300. Each uses the same 1200 scheduled batches, 180 FineWeb training docs, 3 QAT train-split dev docs and the old21 FineWeb heldout docs only for historical checks.

| Arm | k0 | ±1 | ±2 | ±3 | ±4 | Role |
|---|---:|---:|---:|---:|---:|---|
| W | 0 | ±α/4 | ±α/2 | ±3α/4 | ±α | Original wide Q9 |
| N | 0 | ±α/6 | ±α/3 | ±α/2 | ±2α/3 | Historical narrow Q9 with same T4 |
| **H** | 0 | ±α/6 | ±α/3 | ±α/2 | **±α** | Narrow interior, original wide extremes |
| D | — | — | — | — | — | Direct Q3 baseline |

All Q9 arms use `k=round(4*clamp(w/α,-0.99,0.99))`, exactly integers -4..4; different h(k) values in `q=α*(z+(h(k)−z).detach())` during prep. At step300 all switch to the original ternary codebook with Q3 rows reset.

**Primary R3 evaluation chosen before any GPU training:** select the **first 32** distinct FineWeb-Edu eval-hash bucket0 documents occurring strictly after the 340 source rows scanned by the archived F1 loader, requiring **at least 516 tokenizer tokens**. Verified 32 unique doc IDs, source rows **343..1222**, all distinct from source IDs in first340; four nonoverlapping 129-token input blocks/doc, **512 evaluated prediction tokens/document**, 16,384 target tokens total. Each arm evaluated on **the identical per-document token windows**, with complete SHA256 doc ID, source row index, NLL sum, average NLL. These docs are *new to our QAT evaluation*, but **may have occurred in the pretrained base model's pretraining corpus**.

## Results: primary new 32-document evaluation

| Arm | New FineWeb NLL ↓ | New FineWeb PPL ↓ | Old FineWeb heldout NLL ↓ | WikiText validation NLL ↓ | Successful updates / AMP skips | Outer ±4 occupancy prep300 |
|---|---:|---:|---:|---:|---:|---|
| D | 5.789299 | 326.784 | 5.766660 | 6.558239 | 1194 / 6 | N/A |
| W | 4.885759 | 132.391 | 4.996255 | 5.695225 | 1192 / 8 | 29.30% |
| N | 5.813866 | 334.911 | 5.780473 | 6.593468 | 1190 / 10 | 29.22% |
| H | 4.916794 | 136.564 | 5.039273 | 5.730818 | 1191 / 9 | 29.34% |

**Predeclared fresh-document contrasts and per-document evidence** (gap defined as first arm minus second arm; positive means second model did better):

| Contrast | Aggregate NLL gap | Docs with positive gap | Median per-doc gap | Min–max per-doc gap |
|---|---:|---:|---:|---|
| N minus H | 0.897071 | 32/32 | 0.900884 | [0.574085, 1.156653] |
| H minus W | 0.031035 | 23/32 | 0.030385 | [-0.059231, 0.142908] |
| D minus H | 0.872505 | 32/32 | 0.891936 | [0.574462, 1.118086] |
| D minus W | 0.903540 | 32/32 | 0.909236 | [0.568733, 1.122449] |
| D minus N | -0.024567 | 13/32 | -0.020628 | [-0.137744, 0.044240] |

- N→H outer reconstruction restoration: `N−H=+0.897071` nats/token on newly selected docs; **32/32** favor H.
- H versus original W: `H−W=+0.031035`, **23/32** favor W (9 favor H). H is extremely close in final heldout loss compared to its large improvement over N.
- D versus H: `D−H=+0.872505`, **32/32** favor H.
- Narrow vs direct: `D−N=-0.024567`, narrow actually worse than D on new documents.
- **Historical endpoints** corroborate rather than determine the primary outcome: old FineWeb NLL D `5.766660`, W `4.996255`, N `5.780473`, H `5.039273`; WikiText validation D `6.558239`, W `5.695225`, N `6.593468`, H `5.730818`. Historical D/W/N anchors reproduced original R2/F1 values **exactly** to printed precision.

## Q9 preparation and numerical checks

| Preparation diagnostic at step300 | W original | N narrow | H hybrid |
|---|---:|---:|---:|
| Native Q9 train-split dev NLL | 5.428179 | 6.478196 | 5.365745 |
| Fraction of codes ±4 | 29.30% | 29.22% | 29.34% |
| Max output magnitude / α | 1.000000 | 0.666667 | 1.000000 |
| Immediate Q3 projected-dev shock | 1.514062 | 0.304707 | 1.817581 |

At prep300, H native-dev loss (**5.365745**) is comparable to W (**5.428179**), even though H's inner seven levels match N. H is also close to W in extreme-code occupancy, reinforcing that the extreme output magnitudes can materially affect the trajectory. Shock differs between stages but is not itself the 900-step final outcome.

**All 16/16 recorded checks:**

- `four_1200_opportunities`: **true**
- `three_Q9_switches_at300`: **true**
- `D_no_switch`: **true**
- `same_train_docs_chunks`: **true**
- `exact_training_order`: **true**
- `new32_docs_disjoint_old_scan`: **true**
- `new_docs_16384_paired_tokens`: **true**
- `finite_new_and_old`: **true**
- `all_nine_preparation_codes`: **true**
- `actual_preparation_ranges`: **true**
- `reproduce_D_old_fineweb`: **true**
- `reproduce_W_old_fineweb`: **true**
- `reproduce_N_old_fineweb`: **true**
- `reproduce_D_wikitext`: **true**
- `reproduce_W_wikitext`: **true**
- `reproduce_N_wikitext`: **true**

Every arm completed exactly1200 scheduled optimizer opportunities, but effective updates differed because AMP skipped 6/8/10/9 for D/W/N/H, respectively. This difference is reported as an implementation caveat, not silently equated.

**CPU QA history:** first CPU job [6ac992d4095c57808930c1b0](https://huggingface.co/jobs/codeflash85/6ac992d4095c57808930c1b0) printed all quantizer/data passes but ended with Python `PyGILState_Release` interpreter shutdown error (status ERROR), so it was not counted as a clean preflight. Independent finalization-safe CPU job [6ac99425095c57808930c23c](https://huggingface.co/jobs/codeflash85/6ac99425095c57808930c23c) COMPLETED and printed full quantizer/data QA success *before paid GPU launch*. The scientific R3 GPU job itself COMPLETED.

## Limitations and next decisions (NO new GPU job launched)

1. **One trained order/seed**, one 360M language model, same optimizer/teacher/QAT recipe and FineWeb source; the 32 docs were previously unseen by these QAT runs but are **not independent of the model's possible pretrained distribution** and not a second dataset. The old test set has been adaptively examined repeatedly.
2. **Nonuniform H changes only extreme output amplitudes at fixed per-step code assignments**, but because end-to-end trajectories change, scale gradients, code occupancy/margins, weight magnitudes and recovery histories may differ downstream. Avoid saying isolated "raw range magnitude alone" or "two individual weights" — the two outputs are *two codebook levels* shared by millions of weights, not two individual parameters.
3. Pair-wise per-document differences are descriptive for this fixed set; no claims of newly sampled documents being statistically independent of base pretraining, nor seed-level extrapolation or all architectures.
4. Full experiment JSON, dependency manifest, SHA256 doc IDs, losses and provenance are saved, **but no restorable trained weight checkpoint**. Reproduction requires rerunning pinned training.
5. Suggested next step is an **independent-model or new-corpus evaluation** of the now-targeted hybrid extreme-only Q9 result, or a preregistered direct measurement of the master/code survival mechanism — **not another same-seed variant tuned on these 32 docs**. These are suggestions only and not authorized as new GPU runs.

Read [CURRENT_STATE.md](../CURRENT_STATE.md) for up-to-date research status.
