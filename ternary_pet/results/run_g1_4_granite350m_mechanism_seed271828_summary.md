# Granite G1 confirmation — seed/order 271828

Job: `6ac70af0df2184ac91ac6ffc`  
Pinned shared commit: `6ed697a0ca5d0d19ed5ebff04b81a2ed0d259230`  
Status: **complete — negative staging order**

All preregistered construction assertions passed.

## Final held-out endpoint

| Arm | Loss | PPL | Gain vs D |
|---|---:|---:|---:|
| D | 5.72673 | 306.96 | — |
| S | 5.84103 | 344.14 | -0.11431 |
| M-exact | 5.80453 | 331.80 | -0.07780 |
| M-d50 | 5.78502 | 325.39 | -0.05829 |
| Random-d50 | 5.79268 | 327.89 | -0.06596 |

## Step-300 equal-forward diagnostic

- direct fixed-Q3 loss: **5.64945**
- Q9-prepared fixed-Q3 loss: **6.51453**
- Q9 immediate disadvantage: **0.86508 nats**
- D-vs-S disagreement mask: **6.9946%**
  (17,455,806 / 249,561,088 positions)

## Interpretation

This order is a **real negative for the staging claim**. Full S finishes
0.11431 nats worse than D. M-exact and M-d50 also remain worse than D.

However, the intervention ordering still contains useful structure:
M-d50 improves on full S by **0.05601 nats**
and improves on M-exact by **0.01951 nats**.
Matched-random is harmful versus D by **0.06596 nats**,
but is only 0.00766
nats worse than true-M d=0.5, so position specificity is weak on this order.

Do not describe this seed as supporting a positive Q9 staging effect.

## d=0.5 survival

At continuation step 900, true-M Q9-code survival is
**89.59%**
under learned scales and
**89.58%**
under fixed original alpha0.

This result must be interpreted together with orders 1729 and the other
confirmation seed; do not aggregate away sign differences.
