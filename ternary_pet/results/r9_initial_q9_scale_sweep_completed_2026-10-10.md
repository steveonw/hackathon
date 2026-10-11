# R9 CPU Q9 scaling and codebook sweep — completed 2026-10-10

Both public CPU jobs completed, emitted `R9_SCALE_SWEEP_FINAL valid:true`, 60 greedy records per family (15 reused R7 diagnostic prompts × four variants), four complete `R9_WEIGHT_DIAGNOSTICS` per family and no errors:
- Granite job `6acaf484095c578089319cab`
- Smol job `6acaf485fee2c900701906c0`

**All four configurations scored 0/12 permissive correct QA across 4 arithmetic, 4 instruction, 4 extraction prompts in both model families.** No claim that this is an untouched independent test (R7 subset).

| Variant | Granite rel RMSE | Smol rel RMSE | Granite cosine | Smol cosine | Granite bigram diversity (3 continuations) | Smol bigram diversity |
|---|---:|---:|---:|---:|---:|---:|
| H initial 1.0 | .377830 | .373640 | .960598 | .961584 | .021164 | .227513 |
| H scale 1.5 | **.310771** | **.309096** | **.972225** | **.972336** | .037037 | **.423280** |
| H scale 2.0 | .325898 | .326829 | .970886 | .970717 | .058201 | .296296 |
| W initial 1.0 | .347176 | .342259 | .953051 | .954392 | .042328 | .158730 |

Full targeted weights Granite 249561088 across168 target linears; Smol 314572800 across224. Original H zero fractions .123708 Granite/.122577 Smol and extreme code ±4 fractions .289712/.291031. At H×1.5, zero fractions .184318/.182648, extreme fractions .119311/.118339; at ×2, zero .243566/.241471, extreme .041961/.040300. H×1.5 improves RMSE 17.75% Granite, 17.28% Smol; despite improved weight similarity, QA score remains floor 0/12. Therefore naive uniform alpha scaling alone does **not** fix initial full-model functional collapse.

**Research implications:** Next useful inexpensive diagnosis is sensitivity by layer or per-module family (attention projections vs MLP), and activation/logit perturbation. Weight-space cosine averaged over all params is misleading about transformer compounded error. Use one-layer-at-a-time or one-block-at-a-time quantization with nonquantized remainder to localize fragile modules, plus teacher-logit agreement / NLL, before another costly optimization run. Do not launch GPU training on evidence from this sweep.

Source `ternary_pet/r9_cpu_initial_q9_scale_sweep.py`, immutable commit `9ea045d69fef088f4109c3cb8c08114b2b8a02be`. Full source preflight `6acaf47c095c578089319ca0`.
