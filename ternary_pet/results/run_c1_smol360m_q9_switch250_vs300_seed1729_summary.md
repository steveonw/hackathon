# C1 — Switch Q9→Q3 after 250 versus 300 updates: exploratory seed1729

**Status:** HF job **COMPLETED** successfully; `valid_for_science=true`.  
**Completed:** 2026-10-09 03:24:11 UTC (Oct 8 evening EDT).  
**HF:** [6ac8597afee2c90070172c79](https://huggingface.co/jobs/codeflash85/6ac8597afee2c90070172c79).  
**Pinned script:** [`5577a771ed57409283690097a58bae0c8966fdb1`](https://github.com/steveonw/hackathon/blob/5577a771ed57409283690097a58bae0c8966fdb1/ternary_pet/c1_smol360m_q9_switch250_vs300_seed1729.py).  
**Pre-job prereg:** [Phase C exploratory switch comparison](../research_log/phase_c_q9_switch250_vs300_seed1729_prereg_2026-10-08.md).  
**Full parsed `FINAL_JSON`:** [raw results](run_c1_smol360m_q9_switch250_vs300_seed1729_2026-10-08.json).

## Primary result: both prespecified exploratory preservation criteria PASSED

T1 had found only **69.40%** retrospective recall of the eventual step-300 D/S disagreement mask at step 250 (one seed). C1 explicitly tested whether stopping Q9 preparation at 250 still preserves most downstream quality with equal **global 1200-step opportunity** budgets.

| Arm | Q9 steps | Q3 steps | Effective optimizer steps | AMP skips | Held-out test NLL ↓ | Held-out PPL ↓ | Top-1 agreement | Teacher KL ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| D | 0 | 1200 | 1194 | 6 | 5.595722 | 269.272 | 29.87% | 2.664031 |
| S250 | 250 | 950 | 1190 | 10 | 4.930515 | 138.451 | 36.83% | 1.997867 |
| S300 | 300 | 900 | 1190 | 10 | 4.900985 | 134.422 | 36.91% | 1.969979 |

- **S300 vs D gain:** `0.694736820` nats/token (established staged effect).
- **S250 vs D gain:** `0.665206932` nats/token, ~**48.58% lower perplexity** than direct Q3.
- **S250 minus S300:** **+`0.029529888` nats/token** (S250 **slightly worse**).
- **Retained fraction of S300's gain over D:** **95.75%**.
- Prespecified criterion A: `L250-L300 <= +0.07` nats: **PASS**.
- Prespecified criterion B: `(LD-L250)/(LD-L300) >= 0.90`: **PASS**.
- Joint exploratory decision: **PASS**.

Interpretation: under this exact matched 1200-step SmolLM2/WikiText-2 recipe, **250 Q9 steps retain nearly all the quality lift previously attained with 300**. But Q9-250 has 50 *more* subsequent Q3 steps than Q9-300. The trial does **not** show that the missing Q9 assignments are unimportant in isolation, nor that some universal 250-step optimal switch exists.

## Train-split validation time course — NOT held-out comparisons

| Global step | D dev NLL (space) | S250 dev NLL (space) | S300 dev NLL (space) |
|---:|---|---|---|
| 250 | 6.19610 (Q3) | 5.39083 (Q9) | 5.39083 (Q9) |
| 300 | 5.98809 (Q3) | 5.78881 (Q3) | 5.18272 (Q9) |
| 600 | 5.74032 (Q3) | 5.30526 (Q3) | 5.20442 (Q3) |
| 900 | 5.41510 (Q3) | 4.78606 (Q3) | 4.73523 (Q3) |
| 1200 | 5.25594 (Q3) | 4.57502 (Q3) | 4.52172 (Q3) |

At step 250 the Q9 arms share the **same** native-Q9 validation loss **5.390830**; S250's immediate original-scale ternary projection increases to **6.542107**. At step300 S300's native-Q9 validation is **5.182719**, but its immediate original-scale ternary projection is **6.542292**. The projected Q3 losses are surprisingly similar at the two switch times (difference -0.000185), despite distinct internal states.

Measured Q9→Q3 native-space switch shocks: S250 **+1.151278**, S300 **+1.359573** nats. These are *within-model phase* changes, not before/after optimizer updates. The difference in native Q9 performance is not a trustworthy guide to final Q3 outcome.

## Technical reproducibility and design checks

- `valid_for_science=true`; all three arms scheduled **1,200** global training opportunities with exactly the preregistered LR trajectory and matched training-chunk order.
- Within-run D **exactly** reproduced historical tuned direct Q3 test NLL **5.595722187310**. Within-run S300 **exactly** reproduced v10 staged test NLL **4.900985367596**.
- Each staged arm made 1190 actual optimizer updates and 10 AMP skipped opportunities; direct made 1194 actual and 6 skipped. Equal *scheduled* steps, not identical actual effective-update counts between direct and staged arms.
- Original Q3 row scales and fresh Adam+GradScaler applied at the S250/S300 switch, with no global LR restart. D kept continuous optimizer/scales as the tuned baseline.
- No generation-quality claims; this experiment used the existing held-out 8,192 tokens (and a fixed 3,072-token train-split diagnostic set). Both evaluation splits were known from previous experiments. Seed1729 was one previously studied order.
- Negative/limited findings retained: S250 is measurably worse than S300 by 0.02953 nats, and these results alone cannot establish independent generalization or a newly optimal timing law.

## Recommended follow-up, **not automatically launched**

If aiming to shorten Q9 at a fixed 1,200-step total compute budget, **independently confirm the fixed 250-vs-300 protocol on held-out new training orders** without further tuning on seed1729. The previous T1 selection and C1 outcome render seed1729 exploratory. If the specific science question is *what Q9 discovers rather than merely switch timing*, the more incisive new experiment remains separating causal effects of Q9-only changes and direct-only changes Q9 avoids at the step-300 mask.

**No additional GPU jobs or parameter sweeps authorized by this summary.** Any follow-up needs a separate frozen protocol and explicit authorization.
