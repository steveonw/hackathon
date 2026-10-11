# R8 two-family Q9→Q3 stage-checkpoint experiment — frozen launch handoff

**Prepared 2026-10-10. Not yet launched.** User authorized both models and is willing to allocate a larger GPU process. One matched H-only trajectory per family; existing R6 model/seed/dataset/optimizer and 300→900 opportunities are retained. This is a **stage-localization diagnostic**, not a replicate of four-arm comparative R6 science. No other D/W/N arms are run.

## Immutable source and stage capture

- Granite script `ternary_pet/r8_granite_stage_localization.py`, immutable commit `0881c4fc693232258d3148680ab6009137723310`. Based on `r6_granite_seed170141_imdb_hybrid.py`, seed170141.
- Smol script `ternary_pet/r8_smol_stage_localization.py`, immutable commit `ecb76bfffec2ee0fc384e64661f4db12bdd99e7e`. Based on `r6_smol_seed190027_imdb_hybrid.py`, seed190027.
- Dual-script CPU AST/structural remote preflight: HF job `6acada1cfee2c9007018f501`, returned `R8_SOURCE_PREFLIGHT_OK granite` and `R8_SOURCE_PREFLIGHT_OK smol`. This is **not** a full train/save/restore preflight: GPU outcomes still unknown.
- Both jobs capture (1) **C1_Q9_step300** before projection, (2) **C2_projected_Q3_step300** before any Q3 optimizer update, (3) **C3_Q3_step600** after 300 Q3 updates/opportunities, and (4) **C4_Q3_step1200** final.
- Each snapshot stores all quantized linear **forward-effective code and learned alpha scales** with pinned original source weights for other tensors; 4-bit packed signed Q9 codes are encoded offset+4 and 2-bit packed Q3 codes encoded offset+1. Manifest explicitly records Q9 H mapping `k/6 + sign(k)/3` on ±4 and Q3 map `k/1.5`. These files are **not** optimizer-resumable training states.
- Stored in **private** HF dataset `codeflash85/ternary-pet-r5-checkpoints` under `r8_stage_localization/{granite_seed170141,smol_seed190027}/R8_H_<stage>.npz`. Each upload is **mandatory** and independently read back with `force_download=True`, verified by full-file SHA256 and manifest; failure aborts training and is not silently marked successful.
- C0 original source already pinned (Granite `bd8a1497065c0d6ba1ef19af6b0d2b14bacf71c2`; Smol `a10cc1512eabd3dde888204e902eca88bddb4951`) and should not be redundantly stored. Original R6 C4 snapshot remains separately stored for historical comparison; the newly trained R8 C4 may vary if changed dependency resolution and must be compared empirically.

## Secure Windows PowerShell launch

Use the user's existing local `hf` CLI auth and `--secrets HF_TOKEN`. Do not paste token, don't run through connector without local secret. A10G-large is an optional upgrade if available; **A10G-small** is previously demonstrated to finish complete 4-arm R6 in ~90m, so one H-only arm per model is likely cheaper. Allow longer than 90m for mandatory private uploads/readbacks. These are GPU jobs and can incur charges.

```powershell
hf jobs uv run --flavor a10g-small --timeout 180m --secrets HF_TOKEN "https://raw.githubusercontent.com/steveonw/hackathon/0881c4fc693232258d3148680ab6009137723310/ternary_pet/r8_granite_stage_localization.py"
hf jobs uv run --flavor a10g-small --timeout 180m --secrets HF_TOKEN "https://raw.githubusercontent.com/steveonw/hackathon/ecb76bfffec2ee0fc384e64661f4db12bdd99e7e/ternary_pet/r8_smol_stage_localization.py"
```

## Validity gates after job completion

Do not equate Hugging Face `COMPLETED` with scientific/retention success. Verify exactly four `R8_STAGE_VERIFIED` markers for each family (8 total) with correct names, levels 9/3/3/3, expected layer count (168 Granite, 224 Smol), expected quantized weight count (249561088 Granite, 314572800 Smol), SHA256, bytes, private paths, and terminal `R8_STAGE_TRAINING_COMPLETE`. Download all eight files independently for checksum and test reconstruction, then run frozen capability battery at all stage points. Compare matched same-state C1→C2 losses and generation scores; C2→C3→C4 measures the Q3 optimization trajectory. Use new heldout prompts or treat the already used R7 prompts strictly as diagnostics, not a publication-quality untouched test.

Known limits: original R6 4-arm design included internal legacy R5 names and aux data loads; H-only R8 leaves those initialization procedures in place. Only static/AST and structural checks have passed, not an actual end-to-end GPU or private-write dry-run. The R8 Q9 packed format **differs** from the R6 final Q3 compact format and must be decoded using its R8 manifest. Secure dataset repo write access and adequate HF job runtime necessary.
