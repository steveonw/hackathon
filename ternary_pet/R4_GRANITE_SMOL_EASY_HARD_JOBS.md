# R4 — 2+2 hard/easy seeds on Granite and SmolLM2, one batch per user prompt

**Authoritative as of 2026-10-09 EDT / 2026-10-10 UTC:** first two **GRANITE GPU jobs SUBMITTED**; **second two SMOL GPU jobs NOT SUBMITTED** and reserved for a future user prompt. Never silently launch all four in one turn.

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

**Not launched here; user explicitly requested two jobs per prompt and to wait for first pair.** Reserve:
- SmolLM2-360M-Instruct seed **271828** matched D/W/N/H.
- SmolLM2-360M-Instruct seed **424242** matched D/W/N/H.

On **next explicit user prompt**, first confirm that both Granite GPU jobs are terminal and archive scientific results. Then prepare/pin/CPU-check the two Smol WikiText2 scripts and submit exactly two jobs only if user wants the second pair. Use **Smol's established warmup100→1e-3/cosine→1e-4**, FP16 teacher/autocast and GradScaler, original Smol source revision and same first1200 chunk order per seed. Don't quietly use Granite constantLR1e-4 on Smol. **Cross-model comparison remains exploratory** because LR and precision recipes are architecture-specific. Per-seed source/validation evaluation tokenization differs by model; compare **within-model paired effects**, not raw cross-model NLL.

## Outcomes and archival checks when GPU jobs finish

Primary: **WikiText-2 validation** first64 129-token chunks, **8192 target tokens** each arm, 64 aligned **per-chunk NLL sums** for descriptive paired contrasts. Same validation split across both Granite seeds; fresh to this explicit R4 intervention, but no claim of independent pretraining docs. Predeclared comparisons `N−H`, `H−W`, `D−H`, and `D−W`. Secondary: original **WikiText test** first64 chunks, which serves to reproduce historical Granite D and W baselines and report standard metrics.

**Historical Granite seed271828:** D original heldout NLL `5.726725168526173`, W `~5.84103`. **Seed424242:** D `5.801920056343079`, W `~5.74057`. Technical tolerance D±0.04 / W±0.06 on that historical first64 test. If a check fails, mark study technically invalid instead of asserting a causal result; archive negative failures and source mismatch. Full JSON has per-arm scores, per-chunk sums, source revisions, Python package manifest and individual validity checks.

**Next session protocol:** inspect exact two HF jobs above, extract `FINAL_JSON_BEGIN`...`FINAL_JSON_END`, review `valid_for_science`, archival checks and negative outcomes; save exact raw JSON with provenance and summary in `ternary_pet/results`, update CURRENT_STATE.md, AI_HANDOFF.md, EXPERIMENT.md and reports. Do not launch any GPU duplicates/retries. **Wait for user next prompt to initiate Smol batch**, per request.
