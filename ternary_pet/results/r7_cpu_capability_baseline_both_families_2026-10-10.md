# R7 fixed capability baseline — two CPU jobs completed (2026-10-10)

**Source** pinned `c6a24b3c5ba12f97634bc65312cb8161fba1ebca`; frozen prompt file pinned `7956c5ee8a067efbab7a34ecca84dd287cf5cab9`. Both jobs returned `R7_BASELINE_FINAL` with `valid:true`, 120 generated records (80 greedy, 40 sample) each and prompt SHA256 `5180b1319cca0b3b65ffa01f0940d599747c9f9169bfde7d8b14d0f1dae6008b`.

- Granite CPU job `6acad549095c578089318d57`: base arithmetic **10/10**, instruction **9/10**, extraction **8/10**; reconstructed R6 H arithmetic **0/10**, instruction **0/10**, extraction **0/10**. Base total **27/30**; H **0/30** strict.
- Smol CPU job `6acad569fee2c9007018f35a`: base arithmetic **0/10**, instruction **4/10**, extraction **6/10**; H **0/10** in all three. Base total **10/30**; H **0/30** strict.

**Very important scoring caveat:** These are `strict` normalized exact-answer scores; not equivalent to semantic correctness. Smol original arithmetic 0/10 under strict scoring may reflect answer explanation/format rather than wrong arithmetic. Original script also grades `permissive`, but its `FINAL` aggregate includes only strict counts. The Hugging Face logs connector returned only 20 lines per job including the last 19 per-record lines and final aggregate: the full 120 per-case records are **not available in the retrieved log response**. Do not claim 120 records independently inspected or make any empirical claims about per-category permissive or full continuation/repetition summaries without retrieving a complete archive. Generation-specific R6 earlier showed Smol original answered 17+28 correctly, while ternary H did not. The tests are low-N designed diagnostics, not statistical or general benchmarks.

**Interpretation:** This is striking evidence of collapse in exact instruction/extraction response retention in both hybrid ternary R6 H snapshots versus pinned high-precision original models. It does *not* refute the previously established **within-ternary NLL advantage** of H against D/W/N, and it does not show causal attribution to H architecture versus ternary precision generally. Next: retrieve complete log or run a CPU-only scoring-audit focusing on permissive answers and decode / answer extraction; inspect sampled/continuation quality; plan instruction-retention control before additional GPU spend.

Both jobs CPU-only; no new GPU training.
