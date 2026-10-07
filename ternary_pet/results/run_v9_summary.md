# Run v9 summary — tuned direct-Q3 baseline

**HF Job:** `6ac5ad77fbc85ba6823bb71c`  
**Pinned code:** `5c088e58539b2dede93df57ac3f72dbe0480a028`  
**Seed/order:** 1729  
**Hardware:** A10G-small

## Result

v9 confirms that the historical direct-Q3 baseline was **undertuned**, but a
substantially stronger direct schedule still does **not** match the historical
Q9 -> Q3 result on seed 1729.

The candidate screen used only the 24-chunk validation split for the first 300
updates.

### 300-step validation ranking

| Candidate | Validation loss | PPL | Top-1 | KL |
|---|---:|---:|---:|---:|
| const_1e-3 | 5.9758 | 393.8 | 28.03% | 3.0645 |
| warm100_cosine_1e-3 | 5.9881 | 398.7 | 25.91% | 3.1035 |
| const_5e-4 | 6.1401 | 464.1 | 25.78% | 3.1974 |
| const_3e-4 | 6.1935 | 489.5 | 23.60% | 3.2767 |
| const_3e-3 | 6.2196 | 502.5 | 23.54% | 3.2482 |
| warm100_cosine_3e-4 | 6.2606 | 523.6 | 23.08% | 3.3025 |
| const_1e-4 | 6.5528 | 701.2 | 22.20% | 3.5129 |

The preregistered full-run set was:

- constant `1e-4` historical reference;
- constant `1e-3` (best non-reference at 300);
- warmup100 + cosine, peak `1e-3`, floor `1e-4` (second-best non-reference).

## Full 1200-step held-out results

| Direct-Q3 schedule | Test loss | PPL | Top-1 | KL |
|---|---:|---:|---:|---:|
| constant 1e-4 | 5.8747 | 355.90 | 25.55% | 2.9536 |
| constant 1e-3 | 5.8356 | 342.26 | 26.68% | 2.8942 |
| **warm100 cosine 1e-3 -> 1e-4** | **5.5957** | **269.27** | **29.87%** | **2.6640** |
| historical Q9 -> Q3 | **5.1938** | **180.15** | **33.79%** | **2.2560** |

The tuned warmup+cosine direct baseline improves substantially over constant
1e-4:

- held-out loss improves by **0.2789 nats/token**;
- PPL falls from **355.90** to **269.27**;
- top-1 rises by **4.32 pp**;
- KL improves by **0.2896**.

This closes **41.0%** of the original seed-1729 Q9-vs-direct loss gap.

However, Q9 still remains ahead of the tuned direct baseline by:

- **0.4020 nats/token** held-out loss;
- **33.1%** lower PPL;
- **3.92 pp** top-1;
- **0.4080** KL.

Thus the narrow conclusion is:

> better direct optimization explains a meaningful part of the original small-
> budget Q9 advantage, but it does not erase it on seed/order 1729.

The new direct schedule must be replicated on training orders 271828 and 424242
before replacing constant 1e-4 as the canonical direct baseline.

## Important nuance

The 300-step validation winner (`const_1e-3`) is not the best 1200-step test
run. The second-ranked 300-step candidate (warmup+cosine 1e-3) finishes best.
That does not violate the preregistered selection rule because both were chosen
before held-out evaluation, but it shows that short-horizon ranking is an
imperfect predictor of the final schedule.

Free-running generations remain degenerate for all tuned direct variants.

## Implication for mechanism work

The hybrid-master causal intervention remains scientifically worthwhile, but
the comparison should now acknowledge that the historical direct recipe was
weak. The current best evidence is:

- Q9's advantage is not purely an LR/schedule artifact;
- schedule tuning reduces the seed-1729 loss gap from **0.6809**
  to **0.4020** nats/token;
- the remaining Q9 advantage should be checked against the tuned direct schedule
  on the two other v7 orders before becoming the canonical effect size.
