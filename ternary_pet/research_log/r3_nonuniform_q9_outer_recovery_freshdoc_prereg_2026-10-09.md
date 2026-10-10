# R3 preregistration — nonuniform nine-state outer-recovery test + genuinely new FineWeb doc-heldout audit

**Frozen before R3 code/GPU submission: 2026-10-09 EDT (2026-10-10 UTC).** Authorized under user's "do what you need to do" after R2 review. Scope is **ONE A10G-small GPU job** with **2h maximum**, exactly four sequential seed1729 training arms. No seed sweep, second job, automatic retry, tuning or claim of new-pretraining-data generalization.

## Why this study

Earlier R2 held independent rounding thresholds T constant and found both *wide* Q9 output grids outperform direct Q3, while both *narrow* grids fail to materially beat direct. However even holding T=4 fixed, the range factor also changes evenly spaced **interior level spacing**. R3 tests **outer reconstruction amplitudes while holding all seven central nine-state output values fixed**, and evaluates on FineWeb documents **never previously used by this QAT project**.

## Exact Q9 codebooks and arms (frozen)

The same integer nine-state assignment rule for **every** Q9 preparation variant:
`z=clamp(w/alpha,-0.99,0.99)`, `k=round(4*z)` (exactly integer codes -4..+4), with no extra integer clipping needed at T4 (but safe clamp to ±4). This fixes quantization decision thresholds and code assignments *conditional on any given w,alpha*. The backward STE remains `q=alpha*(z+(h-z).detach())`; all forward Q9 preparations use `h=sgn(k)*level[abs(k)]` with nonuniform lookup choices:

| Name | k=0 | |k|=1 | |k|=2 | |k|=3 | |k|=4 | Meaning |
|---|---:|---:|---:|---:|---:|---|
| `W` historical wide | 0 | 1/4 | 1/2 | 3/4 | 1 | Original Q9, reproduces R2 W4T4 |
| `N` historical narrow | 0 | 1/6 | 2/6 | 3/6 | 4/6 | Reproduces R2 N6T4 |
| `H` hybrid outer restored | 0 | 1/6 | 2/6 | 3/6 | 1 | **ONLY** the two ±4 outer outputs differ from N; the seven interior states are identical |
| `D` direct | N/A | N/A | N/A | N/A | N/A | Original 1200-step direct Q3 anchor |

**Exactly four arms:** D, W, N, H in that fixed order (reference anchors D/W/N each run from scratch). Each Q9 arm prep300 with fixed T4, then reset original-source Q3 row scales and AdamW/GradScaler at step300, then Q3 continuation900. Same v10 matched global LR, first1200 shuffled QAT chunks, teacher/CE+KL weights, FP32 masters, scales, BF16-rounded starting weights and all model/dataset revisions as R2. **No other settings optimized.** This *nonuniform* H codebook is **not** an attempt to train the eventual ternary model at unequal ternary inference grid—every final model is exactly historical Q3.

### Important mechanism caveat

At the *same* FP32 master weights/scales, W/N/H have identical integer k decisions (T4) and the same gradient surrogate `z`; H versus N differs **only in output at |k|=4**. Once learning begins, model outputs/scale derivatives/master positions diverge, so different trained occupancy and code movement are consequences, not controlled constants. H versus N thus tests **the effect of restoring extreme code output magnitudes, holding central output values fixed**, within this recipe. H versus W tests the effect of restoring the central levels at the *same wide outer maximum*; cannot infer universal isolated causal effect of range vs spacing outside this codebook family.

## NEW untouched-by-QAT-evaluation FineWeb doc set: selection frozen a priori

Source unchanged: public `HuggingFaceFW/fineweb-edu`, `sample-10BT`, revision `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9`. Original F1 loader scans 340 ordered source rows to fill training, dev and original eval. Use **the same stable streaming order** but consider **only later rows, with 1-based stream row index > 340**, and only document IDs with `sha256(doc_id)` first eight bytes %20 == 0 (original eval bucket). To ensure single-document evaluation and avoid cross-document buffering, retain **the first 32 distinct qualifying document IDs after row340** with at least **516 tokens** from `tok.encode(text,add_special_tokens=False)` (4×129 tokens). For each, take only its first **four nonoverlapping 129-token windows**, evaluate 4×128 next-token targets = **512 tokens/document**, **128 windows, 16,384 tokens** in all, and record per-document token sum, negative-log-likelihood sum, loss, top1 if available, document-ID **SHA256 16-hex prefix only**, original row index and dataset source revision. Require exact 32 docs, 128 windows, 16,384 tokens, and uniqueness of full SHA256 IDs; if insufficient by scanning 20,000 rows, stop as **technical data failure** without silently relaxing rules. Skip duplicate IDs and empty/short text.

Because the original QAT train/dev/eval docs were selected within first340 source rows, all new docs are position-disjoint from those original docs. A document-ID repeat after row340 should be disallowed by explicitly recording the original first340 IDs and checking complete-ID hashes across candidates. The 32 new document IDs have **never been used for validation/selection in the archived project to date**. This is new heldout **QAT experiment evaluation**, *not* an independently sampled pretraining corpus or guaranteed absent from original pretrained-model data. It is still one source dataset, one fixed sample and single seed.

The old FineWeb doc-heldout128 chunks, original train dev24 chunks, and WikiText validation128 chunks are retained for **technical reproduction checks and secondary context**. For scientific interpretation, the **new 32-doc set is the R3 PRIMARY**; all historical samples have been examined repeatedly.

## Predeclared endpoints and interpretations

**Primary:** aggregate per-token NLL on the new 32-doc set. Contrast `new_NLL(N)-new_NLL(H)` (positive means restoring only extreme reconstruction magnitudes helps), `new_NLL(H)-new_NLL(W)` (positive means original wide interior levels provide further benefit), `new_NLL(D)-new_NLL(H)` (positive means H beats direct Q3). Report all four arms, PPL, and **paired per-document contrasts** (average doc loss gaps, count positive, median/range) as descriptive uncertainty context. No p-value or heldout tuning. **Do not rerun R3 on the same new docs after seeing results.**

**Secondary:** original FineWeb heldout/old WikiText validation aggregate NLL/PPL, step300 native prep dev and immediate original-scale Q3 projection shock, step300 code histogram ±4 fraction, FP32 master |w/alpha| ratio, outer z clipping, final Q3 hist/code movement and AMP skips. Show original control reproductions and any deviations.

**Historical anchor acceptance:** original FineWeb old heldout seed1729 direct D =5.7666598074138165, wide W =4.996255073696375, narrow N6T4=5.780473280698061, each tolerance ±0.03. Original WikiText validation D=6.558238908648491, W=5.695225466042757, N6T4=6.593468256294727, tolerance ±0.03. Correct old train order head `[221,850,89,747,685,1055,781,170,233,62,421,806,1031,1187,683,619]`, 180/3/21 source docs and 1200/24/128 old chunks. All four train 1200 scheduled opportunities with Q9→Q3 step300 switch only for W/N/H, identical final Q3. Validity fails if any source reproduction or new document accounting check fails. Preserve negative results and technical invalidity.

## Resource, preflight, evidence

- Before paid GPU: immutable Git commit, AST syntax check, CPU toy `SharedScaleQuant` check of nine code reachability, W/N/H exact level map, *identical* k at same w/alpha, H/N identical output for |k|<4, outer ±1 vs ±2/3, STE weight/alpha gradient (with clip edge), Q3 switch unaffected. Validate the new-doc stream selection on CPU when possible.
- **One HF GPU a10g-small with 2h ceiling**. Run the four arms sequentially, no parallel duplication, no unauthorised retry; do not queue another job if R3 is still running.
- Save `FINAL_JSON` with per-doc losses and hashed IDs, exact dependency versions, Git pinned SHA, data scan/exclusion audits, all technical checks and summary. No multi-GB checkpoint uploaded without separate storage cost approval; avoid pretending the saved aggregate and doc linked scores enable full restoration.
- Interpret loss trajectories cautiously: changes in learned scales and code occupancy arise during training even if starting code assignments are identical. No conclusion on other seed, model family or independent pretraining exposure.

**Decision gate:** if H recovers most of W's new-doc advantage, prioritize learning how large-magnitude extreme reconstruction stabilizes early ternary optimization; if H resembles N, interior spacing differences matter; if H is intermediate, both. Whichever result, cross-model evaluation and genuinely separate datasets are the next generalization priorities, not another adaptive retune of the same sample.
