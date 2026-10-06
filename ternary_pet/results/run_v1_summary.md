# Run v1 summary — 2026-10-06

**Hugging Face Job:** `6ac52475404719ba37661c8b`  
**Model:** `HuggingFaceTB/SmolLM2-360M-Instruct`  
**Hardware:** `t4-small`  
**Job status:** completed

## Main metrics

| Variant | Loss | Perplexity | Top-1 agreement vs baseline |
|---|---:|---:|---:|
| Baseline | 4.6120 | 100.69 | 100.00% |
| Direct PTQ, 3 levels | 19.7328 | 371,414,153 | 0.00% |
| Staged PTQ, 27 -> 9 -> 3 | 20.6439 | 485,165,195 | 0.00% |
| Direct QAT, 3 levels, 30 steps | 13.2973 | 595,600 | 1.82% |
| Staged QAT, 10 + 10 + 10 steps | 17.5306 | 41,061,313 | 7.27% |

The absolute perplexities come from a tiny custom evaluation set and should
**not** be compared with published benchmark perplexities. The useful signal
here is the relative change between variants under the same evaluator.

## What happened

The raw 3-level jump destroyed useful language behavior. The staged raw
27 -> 9 -> 3 path did not protect it; by perplexity it was slightly worse.

Direct ternary recovery learned the tiny calibration examples aggressively:
training loss fell from 19.75 to 3.29 in 30 steps. That did **not** translate
into good held-out behavior, which is a strong sign of overfitting / insufficient
calibration coverage.

The staged path behaved differently:

- 27 levels: training loss improved from 6.15 to 5.67.
- 9 levels: training loss worsened from 5.11 to 10.04.
- 3 levels: training became unstable; the logged step-5 loss spiked to 58.20
  before ending at 17.10.

Its final perplexity was much worse than direct QAT, although its top-1 token
agreement with the baseline was higher (7.27% vs 1.82%). That disagreement
between metrics is worth retaining as a diagnostic, not treating as a win.

Qualitatively, both recovered models were still broken. Direct QAT produced
fragmented text; staged QAT largely collapsed into repeated "the".

## v1 conclusion

**v1 does not support the claim that the 27 -> 9 -> 3 staircase is better than
direct ternarization.**

It also does not cleanly falsify the underlying idea, because the experiment
used a very simple per-row max-absolute quantizer, a tiny calibration set, and
only 30 recovery steps. Most importantly, the state alphabets were independently
rescaled rather than enforcing a strict nested base-3 ancestry.

## What v2 should fix

1. Use a strict nested 27 -> 9 -> 3 codebook so state ancestry is actually
   preserved.
2. Tune ternary scale/threshold rather than using only row max-absolute scale.
3. Increase calibration diversity substantially.
4. Evaluate 27 and 9 levels before descending further, with an early-stop rule
   if a stage destabilizes.
5. Compare both equal-total-compute and equal-final-ternary-compute controls.
6. Add weight-state histograms, KL divergence, and layerwise sensitivity.
7. Run at least three seeds before interpreting a small metric advantage.

This is exactly the kind of failed first run we want in the record: it tells us
where the naive formulation breaks.
