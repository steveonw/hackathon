# R8 two-family stage checkpoint outcome — verified completed

2026-10-10. Both user-launched A10G-small GPU jobs completed full H-only 1200-opportunity training; all 8 stage snapshots uploaded to private dataset `codeflash85/ternary-pet-r5-checkpoints`, then individually **read back with force_download=True and checksum verified in the producing GPU job**. Terminal markers `R8_STAGE_TRAINING_COMPLETE` present for both. **Reconstruction and full external audit of all 8 checkpoints remains pending**. Inference snapshots are NOT restartable optimizer training states. This is no longer a plan-only study.

| Family | Job | Stage | Levels | Size (bytes) | SHA256 |
|---|---|---|---:|---:|---|
| Granite | 6acadb04fee2c9007018f5be | C1 Q9 step300 | 9 | 99689409 | `02cf83d461ccf33ffa595956b3445ffb5c651303a952883d809e69387d3237db` |
| Granite | same | C2 projected Q3 step300 | 3 | 50932233 | `52ab128a5ad2c6d3afe4f88497bdf1a343ced3bd92a33e300e7b4e8732dffecf` |
| Granite | same | C3 Q3 step600 | 3 | 50920098 | `9a19dca77ef91de8791123167fbad0bdb9334772509f2922ec49e90a80a1714a` |
| Granite | same | C4 Q3 step1200 | 3 | 50895975 | `f38d054414c0ac99850cf8130ef65aabc0c73b6f317edf778064bc2037c26f4a` |
| SmolLM2 | 6acadb0dfee2c9007018f5c0 | C1 Q9 step300 | 9 | 125715467 | `b6be6a5d924fe95122d64cdee05fa3277fea4c509ce74e654045615b3c44722a` |
| SmolLM2 | same | C2 projected Q3 step300 | 3 | 64188410 | `6df947c00994e7d9a0322db26a6f77a966f1c4c9da30d28e26f82e7751018574` |
| SmolLM2 | same | C3 Q3 step600 | 3 | 64191825 | `7f89c6215c266af58d90fd3c671147edfdcab3a502a2a4285465e8e2fab9dbf4` |
| SmolLM2 | same | C4 Q3 step1200 | 3 | 64195170 | `468af19ee4a45535664703285e257372b2d3b81dfd2e74e2fafe319626bfc522` |

Remote files are `r8_stage_localization/{granite_seed170141|smol_seed190027}/R8_H_{stage}.npz`. Granite 168 layers / 249,561,088 quantized weights; Smol 224 layers / 314,572,800 quantized weights. Smol reported 1190 effective updates and 10 AMP skips across 1200 scheduled opportunities. Granite reported 1200 scheduled opportunities. Sources: Granite pinned `0881c4fc693232258d3148680ab6009137723310`; Smol `ecb76bfffec2ee0fc384e64661f4db12bdd99e7e`.

**Next**: independent CPU reconstruction audit and capability evaluation across C1/C2/C3/C4 against pinned original bases; **Q9 C1 needs its own decoder** (`bits_per_code=4`, `offset=4`, H codebook `k/6 + sign(k)/3` when `abs(k)==4`) and C2–C4 need Q3 decoder (`bits_per_code=2`, `offset=1`, `k/1.5`). Stage-specific R8 format differs from legacy R6 Q3 format. Use paired frozen prompts and report progression, not post-hoc cherry-picks. No need to rerun paid GPU training.
