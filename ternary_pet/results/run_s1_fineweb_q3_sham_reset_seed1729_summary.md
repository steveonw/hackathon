# S1 — FineWeb direct-Q3 sham optimizer and scale reset at step300

**Status:** COMPLETED, **2026-10-09 23:08:23 UTC**, `valid_for_science=true` and **9/9 construction/reproduction checks passed**.  
**Job:** [HF `6ac96a82fee2c9007017ea6b`](https://huggingface.co/jobs/codeflash85/6ac96a82fee2c9007017ea6b), A10G-small.  
**Pinned code:** `4cdab8ee73ddd804e96fac43992f91ab9a54c1b9`, [S1 source](../s1_fineweb_q3_sham_reset_seed1729.py).  
**Frozen preregistration:** [S1 control](../research_log/s1_fineweb_direct_q3_sham_reset_seed1729_prereg_2026-10-09.md).  
**Raw scientific JSON:** [S1 output](run_s1_fineweb_q3_sham_reset_seed1729_2026-10-09.json).

## Question

The established FineWeb-Edu direct-Q3 vs Q9→Q3 comparison reset optimizer/GradScaler and row scales at step300 only for the Q9-staged model. Could a **reset without Q9** explain most or all of the staged advantage?

S1 repeats the exact 1200-step F1 seed1729 FineWeb QAT data and objective. **D_continuous** trains direct Q3 all1200 with no resets, and **D_sham** trains direct Q3 first300, then restores *original Q3 row scales* and resets AdamW and GradScaler exactly at global step300, without changing FP32 masters, training batches, global LR or ternary level. The historical Q9→Q3 staged result remains a previously archived same-source reference rather than an arm rerun here.

Data: FineWeb `sample-10BT` pinned document ID hash split, 180 train documents →1200 chunks, 3 train-split dev documents →24 chunks, 21 heldout documents →128 chunks; independent-from-F1-QAT-training WikiText2 validation 128 chunks. This is **one seed, reused heldout documents**, and possible source-pretraining overlap is unknown.

## Results

| Arm | FineWeb heldout NLL ↓ | FineWeb PPL ↓ | WikiText validation NLL ↓ | WikiText PPL ↓ | Effective updates /1200 | AMP skips |
|---|---:|---:|---:|---:|---:|---:|
| D_continuous | 5.766659807 | 319.469 | 6.558238909 | 705.029 | 1194 | 6 |
| D_sham reset@300 | **5.789199110** | 326.751 | 6.560839813 | 706.865 | 1192 | 8 |
| **F1 historical Q9(300)→Q3(900)** | **4.996255074** | ~147.858 | **5.695225466** | ~297.444 | 1192 in historical F1 | 8 |

**Primary FineWeb contrast:** `D_continuous NLL − D_sham NLL = -0.022539303`. Negative: **sham reset slightly worsens direct Q3 by 0.022539 nats/token**.

**FineWeb sham minus historical staged:** **+0.792944036 nats** worse than Q9-staged.

**Secondary WikiText validation:** D-continuous − D-sham `-0.002600905` nats (nearly identical, sham slightly worse); D-sham − historical staged `+0.865614347` nats.

This **falsifies the simple reset-alone explanation** of F1's large Q9→Q3 advantage under these matched seed1729 conditions. It does *not* prove optimizer/scale resets contribute exactly zero to other model conditions, or isolate how Q9 preparation interacts with reset.

## Reset mechanics and checks

At global step300 the sham model's native Q3 train-dev loss was **6.563584129**, then with original Q3 scales restored **6.555334032** (shock **-0.008250097**, small and beneficial in the immediate dev diagnostic). Nevertheless, the final heldout loss slightly worsened compared with continuous Q3.

- The continuous Q3 anchor reproduced F1 exactly on both heldout benchmarks: FineWeb **5.766659807**, WikiText validation **6.558238909**.
- Preswitch train-split dev NLL after300 matched across independently run direct arms within 1e-5, the same train order used as F1.
- `D_sham` had exactly one step300 Q3 reset, no quantizer level switch, and no global LR restart; both arms had 1200 scheduled training opportunities.
- All checks: `same_initial_q3_dev_at300=true`; `historical_continuous_fineweb=true`; `historical_continuous_wikitext=true`; `same_1200_scheduled=true`; `one_sham_reset=true`; `document_split_disjoint=true`; `correct_chunks=true`; `all_finite=true`; `same_original_order_head=true`.
- Different effective optimizer updates (continuous1194, sham1192) are recorded and not assumed to be identical.

## Limits and next controls

Strong *single familiar seed* controlled evidence that the FineWeb staging benefit cannot be reproduced by copying the switch reset into direct Q3. Does not remove the Q9 larger representable-range confound; R1 range-matched Q9 remains the most direct unrun contrast. New-document evaluation, per-document loss and checkpoint saving are additional methodological gaps. No new GPU run is authorized from this completed report.

See [G1 complementary scale-freeze mechanism control](run_g1_depth_by_scale_freeze_seed1729_summary.md) and [CURRENT_STATE.md](../CURRENT_STATE.md).
