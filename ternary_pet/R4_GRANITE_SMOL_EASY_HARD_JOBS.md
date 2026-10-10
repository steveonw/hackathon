# R4 — 2+2 hard/easy seeds on Granite and SmolLM2, one batch per user prompt

## FINAL — ALL FOUR JOBS COMPLETED, RAW RESULTS ARCHIVED

**R4 is COMPLETE and scientifically valid (2026-10-10 UTC).** All four HF A10G-small scientific jobs reached terminal **COMPLETED**, four separate `FINAL_JSON` records were recovered and `valid_for_science=true` in all. **48/48 reported checks passed** (12 per job), including 1200 matched training opportunities per arm, precise Q9 stage assignments, original ternary final models, and reproductions of historical D/W anchors. No further GPU jobs launched.

[**Full four-job aggregate, signed contrasts and limitations**](results/run_r4_granite_smol_easy_hard_four_job_aggregate_2026-10-10.md).

| Model / seed | HF job | Final stage, finish UTC | Full archived raw output |
|---|---|---|---|
| Granite hard 271828 | [6ac9a43d095c57808930ccfe](https://huggingface.co/jobs/codeflash85/6ac9a43d095c57808930ccfe) | COMPLETED, 2026-10-10 03:00:26 | [JSON](results/run_r4_granite_hard_seed271828_2026-10-10.json) |
| Granite easy 424242 | [6ac9a443095c57808930cd03](https://huggingface.co/jobs/codeflash85/6ac9a443095c57808930cd03) | COMPLETED, 2026-10-10 03:00:36 | [JSON](results/run_r4_granite_easy_seed424242_2026-10-10.json) |
| Smol 271828 | [6ac9a5c2fee2c90070180ce3](https://huggingface.co/jobs/codeflash85/6ac9a5c2fee2c90070180ce3) | COMPLETED, 2026-10-10 03:10:10 | [JSON](results/run_r4_smol_seed271828_2026-10-10.json) |
| Smol 424242 | [6ac9a5c9095c57808930cdde](https://huggingface.co/jobs/codeflash85/6ac9a5c9095c57808930cdde) | COMPLETED, 2026-10-10 03:09:31 | [JSON](results/run_r4_smol_seed424242_2026-10-10.json) |

**Primary WikiText2 validation NLL (lower better):**

| Family/seed | D direct Q3 | W original wide Q9 | N narrow Q9 | H hybrid (best in each case) |
|---|---:|---:|---:|---:|
| Granite 271828 HARD | 5.438636 | 5.577830 | 5.416681 | **5.370064** |
| Granite 424242 EASY | 5.420453 | 5.349609 | 5.396033 | **5.295450** |
| Smol 271828 | 5.293719 | 4.567472 | 5.259229 | **4.480920** |
| Smol 424242 | 5.265099 | 4.553739 | 5.273762 | **4.520216** |

**Key inference:** Hybrid H beats D, W and N on all four within-seed evaluations. **Granite hard seed** is especially diagnostic: original W remains **0.139194 worse than D**, while H becomes **0.068572 better than D** (H beats W by0.207766). Hybrid advantage over W is +0.054158 on Granite easy, +0.086552 and +0.033523 on the two Smol seeds. These seeds were **historically chosen**, not new randomized independent replications. Evaluation: first64 WikiText2 validation contiguous chunks (8192 tokens) per arm, with per-chunk losses; no valid independent 64-document inference. **Critical across-family confounds:** Granite BF16/constantLR1e-4 and D step300 reset; Smol FP16/warmup+cosine and D uninterrupted Adam. Interpret only paired **within-family** comparisons and generality of the qualitative direction, not a pooled causal model-family effect. Original test anchor D/W values reproduced in every job. No model checkpoints retained.

**All below sections are preserved submission/protocol HISTORY.** Previous references to "running", "no Smol jobs", or "results pending" are outdated. No more GPU jobs should be launched without fresh user authorization.

---


## Batch B EXECUTION RECORD — second user prompt (two Smol GPUs launched)

**Two and only two SmolLM2-360M-Instruct scientific GPU jobs accepted**, 2026-10-10 UTC:

| Smol training-order seed | Hugging Face GPU job | Initial stage | Cost cap |
|---|---|---|---|
| **271828** | [`6ac9a5c2fee2c90070180ce3`](https://huggingface.co/jobs/codeflash85/6ac9a5c2fee2c90070180ce3) | SCHEDULING | A10G-small, 90m |
| **424242** | [`6ac9a5c9095c57808930cdde`](https://huggingface.co/jobs/codeflash85/6ac9a5c9095c57808930cdde) | SCHEDULING | A10G-small, 90m |

**Identical source except seed**, immutable script SHA `65af3057b248f38a6d46ad8b7cd1bc1c52d2e654`:
- [Smol hard/order271828 source](r4_smol_q9_hybrid_seed271828.py)
- [Smol order424242 source](r4_smol_q9_hybrid_seed424242.py)

**CPU-only preflight:** [`6ac9a578095c57808930cd9c`](https://huggingface.co/jobs/codeflash85/6ac9a578095c57808930cd9c) **COMPLETED 2026-10-10 02:40:32 UTC** with `R4_SMOL_CPU_ARM_OK W/N/H` and `R4_SMOL_TWO_PINNED_SCRIPTS_PREFLIGHT_OK`. Whole AST, pinned model/data revisions, exactly 9 codes, code identities, **H and N identical interior outputs**, H-only outer output extension, STE weight and row-scale derivatives including clipped limit, original three-state Q3 after switching, and exact one-line seed difference all verified. No duplicate Smol GPUs found before submission.

**Smol source:** `HuggingFaceTB/SmolLM2-360M-Instruct` pinned `a10cc1512eabd3dde888204e902eca88bddb4951`, `Salesforce/wikitext` `wikitext-2-raw-v1` pinned `b08601e04326c79dfdd32d625aee71d232d685c3`. Each seed uses same established WikiText2 first1200 train chunks in seed-specific shuffled order, first24 subsequent train-split dev chunks; first64 original test chunks for legacy anchors, first64 WikiText validation chunks for additional primary paired evidence. **Smol FP16 teacher/autocast with GradScaler, FP32 masters and BF16-rounded source**, LR warmup100 to1e-3 then cosine to1e-4 at step1200; 35% CE+65% fixed teacher KL.

**Four 1200-step matched trajectories per Smol job:** D = original tuned continuous-Adam **direct Q3 all1200** (no sham optimizer reset, to allow direct historical v9 reproducibility); W = original Q9-wide300→Q3 900; N = narrow Q9 T4 300→Q3 900; H = hybrid Q9 T4 inner narrow and outer wide 300→Q3 900. **All W/N/H** restore source Q3 row scales and reset Adam/GradScaler at switch300, continuing unchanged global LR. These match source-level Granite/Smol *within-family* control practice; note Granite's batch A direct D did include a sham step300 scale/Adam reset, whereas Smol original D does not. Thus even the D intervention has family-specific protocol differences; cross-model comparisons should be **qualitative, within-family paired gains only**, not a two-family factorial holding optimizer policy constant.

**Predeclared technical reproduction anchors**, evaluated on original first64 WikiText test chunks:
- Smol seed271828 D≈`5.6228`, W≈`4.9510` NLL, ±0.08 nats tolerance.
- Smol seed424242 D≈`5.6067`, W≈`4.9563`, ±0.08 nats tolerance.

If these fail, archive original raw/check statuses and mark `valid_for_science=false` instead of editing results or auto-retrying. Primary **first64 WikiText validation chunks**, exactly8192 tokens, save per-chunk NLL sums/tokens and predeclared `N−H`, `H−W`, `D−H`, `D−W` effects. The validation slice is not 32 new FineWeb documents; it is contiguous token chunks from a familiar public dataset, and original base-model pretraining overlap unknown. No inferential n=64 document claims.

**Status:** Submitted; outcomes not known as of writing. No fifth GPU, extra seed, duplicate, or automatic rerun authorized. Both original Granite jobs may still be running concurrently; keep raw result handling independent. When terminal, inspect each HF job, parse `FINAL_JSON_BEGIN/END`, verify checks, archive entire JSON, report heterogeneity and update living project docs. **All four user-requested GPU submissions are now satisfied (two Granite + two Smol).**

---


**Current state: ALL FOUR authorized GPU jobs COMPLETED and all RAW results archived**; see FINAL section above. The older Batch A/B plan below is preserved as preregistration history.

## First prompt — Granite batch A, 2/2 GPU jobs

| Historical seed category | Model/order | Hugging Face GPU job | Stage at submission |
|---|---|---|---|
| **Hard** (Granite wide Q9 historically lost to direct) | `ibm-granite/granite-4.0-350m`, seed `271828` | [`6ac9a43d095c57808930ccfe`](https://huggingface.co/jobs/codeflash85/6ac9a43d095c57808930ccfe) | SCHEDULING |
| **Easy** (Granite wide Q9 historically beat direct) | `ibm-granite/granite-4.0-350m`, seed `424242` | [`6ac9a443095c57808930cd03`](https://huggingface.co/jobs/codeflash85/6ac9a443095c57808930cd03) | SCHEDULING |

Each is **one HF A10G-small, capped at 90 minutes**, from a **single source Git commit `33dfae38bd016946ad5d37bd633fdf02818fbacd`** containing:
- [Granite hard source](r4_granite_q9_hybrid_hard_seed271828.py)
- [Granite easy source](r4_granite_q9_hybrid_easy_seed424242.py)

Both scripts are verified **byte-identical except one `SEED` assignment**. Each job individually runs **four matched 1200-step arms**: D direct Q3, W original Q9(300)→Q3(900), N narrow Q9→Q3, and H hybrid Q9→Q3 with **only the ±4 outer code outputs restored to original wide ±α**. Q9 integer assignment `k=round(4clip(w/alpha,-.99,.99))` same for all, Q3 after300; all four arms reset original Q3 row scales and fresh AdamW at step300. No FP16/GradScaler; **Granite BF16 teacher + BF16 autocast, FP32 masters, original calibrated LR constant1e-4**. Frozen pretrained source revision `bd8a1497065c0d6ba1ef19af6b0d2b14bacf71c2`, pinned WikiText dataset revision `b08601e04326c79dfdd32d625aee71d232d685c3`.

**Before paid GPU:** CPU pin job [`6ac9a397fee2c90070180b8d`](https://huggingface.co/jobs/codeflash85/6ac9a397fee2c90070180b8d) COMPLETED and returned exact source/dataset commits. CPU quantizer/source preflight [`6ac9a40b095c57808930ccec`](https://huggingface.co/jobs/codeflash85/6ac9a40b095c57808930ccec) COMPLETED with `R4_GRANITE_QA_ARM_OK W/N/H` and `R4_GRANITE_TWO_SCRIPTS_PREFLIGHT_OK`; validated 9 codes, outer restoration only, torch STE gradients, original Q3 at switch, seed-only script change and BF16/no-GradScaler policy.

**Frozen study protocol before GPU:** [R4 cross-model 2+2 preregistration](research_log/r4_2026-10-09_granite_smol_easy_hard_two_batch_prereg.md), commit `8099384a95f82e0fa7c3f75f1e3f94971defddc2`. Granite seed choice is **previously observed hard/easy**, not randomly sampled seeds.

## Second prompt — SmolLM2 batch B, 0/2 jobs

**Historical before second prompt; both Smol seeds subsequently launched after explicit user request.** Planned:
- SmolLM2-360M-Instruct seed **271828** matched D/W/N/H.
- SmolLM2-360M-Instruct seed **424242** matched D/W/N/H.

On **next explicit user prompt**, first confirm that both Granite GPU jobs are terminal and archive scientific results. Then prepare/pin/CPU-check the two Smol WikiText2 scripts and submit exactly two jobs only if user wants the second pair. Use **Smol's established warmup100→1e-3/cosine→1e-4**, FP16 teacher/autocast and GradScaler, original Smol source revision and same first1200 chunk order per seed. Don't quietly use Granite constantLR1e-4 on Smol. **Cross-model comparison remains exploratory** because LR and precision recipes are architecture-specific. Per-seed source/validation evaluation tokenization differs by model; compare **within-model paired effects**, not raw cross-model NLL.

## Outcomes and archival checks when GPU jobs finish

Primary: **WikiText-2 validation** first64 129-token chunks, **8192 target tokens** each arm, 64 aligned **per-chunk NLL sums** for descriptive paired contrasts. Same validation split across both Granite seeds; fresh to this explicit R4 intervention, but no claim of independent pretraining docs. Predeclared comparisons `N−H`, `H−W`, `D−H`, and `D−W`. Secondary: original **WikiText test** first64 chunks, which serves to reproduce historical Granite D and W baselines and report standard metrics.

**Historical Granite seed271828:** D original heldout NLL `5.726725168526173`, W `~5.84103`. **Seed424242:** D `5.801920056343079`, W `~5.74057`. Technical tolerance D±0.04 / W±0.06 on that historical first64 test. If a check fails, mark study technically invalid instead of asserting a causal result; archive negative failures and source mismatch. Full JSON has per-arm scores, per-chunk sums, source revisions, Python package manifest and individual validity checks.

**Next session protocol:** inspect exact two HF jobs above, extract `FINAL_JSON_BEGIN`...`FINAL_JSON_END`, review `valid_for_science`, archival checks and negative outcomes; save exact raw JSON with provenance and summary in `ternary_pet/results`, update CURRENT_STATE.md, AI_HANDOFF.md, EXPERIMENT.md and reports. Do not launch any GPU duplicates/retries. **Wait for user next prompt to initiate Smol batch**, per request.
