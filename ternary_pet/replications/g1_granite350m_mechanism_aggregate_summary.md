# Granite G1 mechanism confirmations — three-order aggregate

Model: `ibm-granite/granite-4.0-350m`  
Orders: 1729, 271828, 424242  
Shared confirmation pin: `6ed697a0ca5d0d19ed5ebff04b81a2ed0d259230`

## Individual orders

| Seed/order | D loss | S loss | S gain vs D | M-exact gain | M-d50 gain | Random-d50 gain | Mask |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1729 | 5.66205 | **5.53495** | **+0.12710** | +0.09680 | **+0.15071** | −0.07308 | 6.0246% |
| 271828 | **5.72673** | 5.84103 | **−0.11431** | −0.07780 | **−0.05829** | −0.06596 | 6.9946% |
| 424242 | 5.80192 | **5.74057** | **+0.06135** | +0.05721 | **+0.09176** | −0.01792 | 6.0296% |

Positive values mean lower loss than D.

## What replicated and what did not

### Full Q9 staging
- S beats D on **2/3** orders: 1729 and 424242.
- S loses to D on **1/3** orders: 271828 by 0.11431 nats/token.
- Mean paired S gain is only **+0.02471 nats/token**.
- Mean paired PPL reduction is **1.93%**.

Therefore Granite does **not** support a clean 3/3 replication of the staging
advantage. The family-level effect is heterogeneous across training order.

### Worse entry, later trainability
On **3/3** orders, the Q9-prepared state is substantially worse than direct as
an immediate fixed-Q3 model at step 300:

- 1729: 7.74247 vs 5.81281
- 271828: 6.51453 vs 5.64945
- 424242: 7.57531 vs 5.68631

But only 2/3 orders convert that worse entry into a final S>D advantage.

### True-mask d=0.5 intervention
M-d50 beats full S on **3/3** orders:

- 1729: 5.51134 vs 5.53495
- 271828: 5.78502 vs 5.84103
- 424242: 5.71016 vs 5.74057

Its mean improvement over full S is **0.03668 nats/token**.

M-d50 beats D on **2/3** orders, not 3/3. On order 271828 it still loses to D
by 0.05829 nats.

Mean paired M-d50 gain vs D is **+0.06139 nats/token**, with mean paired PPL
reduction **5.58%**.

### Matched-random control
Random-d50 is harmful versus D on **3/3** orders:

- 1729: −0.07308-nat gain
- 271828: −0.06596
- 424242: −0.01792

Mean effect: **−0.05232 nats/token**.

M-d50 also beats matched-random on **3/3** orders, although the separation is
tiny on seed 271828 (0.00766 nats). Thus position specificity is strongly
supported on 1729 and 424242, but only weakly separated on 271828.

### Disagreement-mask size
Mask sizes:
- 1729: 6.0246%
- 271828: 6.9946%
- 424242: 6.0296%

Mean: **6.3496%**.

The ~6% scale is therefore fairly stable, with seed 271828 somewhat larger.

### d=0.5 code survival
Q9-selected-code survival after 900 Q3 continuation steps:
- 1729: 90.70%
- 271828: 89.59%
- 424242: 90.57%

Mean: **90.28%**.

Learned-scale and fixed-alpha0 survival remain nearly identical on every order,
again arguing against scale drift as the main explanation.

## Conclusion

Granite gives a **mixed staging replication** but a more consistent intervention
pattern.

Safe claim:

> On Granite-350M, Q9 preparation always produces a worse immediate ternary
> checkpoint and a ~6% D-vs-S disagreement set, but the final full-staging
> advantage is order-dependent (2/3 positive). Placing the Q9-selected codes at
> moderate interior depth on the true disagreement set consistently improves
> over full S (3/3), while the matched-random intervention is harmful (3/3).

Do **not** claim that Q9 staging itself replicates 3/3 on Granite.

Do **not** average away seed 271828. It is a scientifically meaningful negative
order and materially weakens any universal cross-family staging claim.

The intervention result suggests that the reusable signal may be more robust
than the raw staged trajectory itself: Q9 can identify useful commitments even
on an order where carrying the full Q9-prepared master state is ultimately
counterproductive. This is a hypothesis-level interpretation, not yet a proven
mechanism for why seed 271828 reverses sign.
