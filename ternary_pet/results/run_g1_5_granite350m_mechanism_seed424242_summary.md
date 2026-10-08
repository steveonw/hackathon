# Granite G1 confirmation — seed/order 424242

Job: `6ac70af2df2184ac91ac7000`  
Pinned shared commit: `6ed697a0ca5d0d19ed5ebff04b81a2ed0d259230`  
Status: **complete — positive staging order**

All preregistered construction assertions passed.

## Final held-out endpoint

| Arm | Loss | PPL | Gain vs D |
|---|---:|---:|---:|
| D | 5.80192 | 330.93 | — |
| S | 5.74057 | 311.24 | 0.06135 |
| M-exact | 5.74471 | 312.53 | 0.05721 |
| M-d50 | 5.71016 | 301.92 | 0.09176 |
| Random-d50 | 5.81984 | 336.92 | -0.01792 |

## Step-300 equal-forward diagnostic

- direct fixed-Q3 loss: **5.68631**
- Q9-prepared fixed-Q3 loss: **7.57531**
- Q9 immediate disadvantage: **1.88899 nats**
- D-vs-S disagreement mask: **6.0296%**
  (15,047,433 / 249,561,088 positions)

## Interpretation

This order cleanly reproduces the positive staging and mechanism pattern.
Full S beats D by **0.06135 nats**.
M-exact recovers **93.3%** of that gain.
M-d50 recovers **149.6%**, beats full S by
**0.03041 nats**, and matched-random
is harmful by **0.01792 nats** versus D.

## d=0.5 survival

At continuation step 900, true-M Q9-code survival is
**90.57%**
under learned scales and
**90.56%**
under fixed original alpha0.

This result must be interpreted together with orders 1729 and the other
confirmation seed; do not aggregate away sign differences.
