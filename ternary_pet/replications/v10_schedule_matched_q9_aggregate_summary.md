# v10 — schedule-matched Q9 -> Q3 control

Pinned code: `4050f42cf226a300082178b2fda475ebcf31664e`

Valid HF jobs:

- seed 1729: `6ac5c252fbc85ba6823bbd6c`
- seed 271828: `6ac5c254fbc85ba6823bbd6e`
- seed 424242: `6ac5c256404719ba37664cf1`

## Protocol

Q9 -> Q3 received the exact global LR curve previously selected for tuned
direct Q3:

- steps 1-100: linear warmup to 1e-3;
- steps 101-1200: cosine decay to 1e-4;
- Q9 forward weights on global steps 1-300;
- at step 300, discard Q9 scales, restore original Q3 scales, and reset Adam;
- Q3 on global steps 301-1200;
- do not restart the LR curve at the transition.

No Q9-specific hyperparameter search was performed.

## Final held-out results

| Seed | Tuned direct Q3 | Historical Q9 -> Q3 | **Schedule-matched Q9 -> Q3** | Matched-Q9 gain vs direct |
|---:|---:|---:|---:|---:|
| 1729 | 5.5957 | 5.1938 | **4.9010** | **0.6947** |
| 271828 | 5.6228 | 5.2417 | **4.9510** | **0.6718** |
| 424242 | 5.6067 | 5.3008 | **4.9563** | **0.6505** |

Schedule-matched Q9 beats tuned direct in **3/3 orders**.

Aggregate versus tuned direct:

- mean held-out loss advantage: **0.6723 nats/token**;
- mean paired PPL reduction: **48.94%**;
- mean teacher top-1 gain: **+6.82 percentage points**;
- mean KL advantage: **0.6817**.

Applying the shared schedule also improves Q9 itself versus the historical
constant-1e-4 staged runs by **0.3094
nats/token on average**.

## Equal-compute step-300 diagnostic still points the other way

| Seed | Tuned direct Q3 @300 loss | Matched-Q9 fixed-Q3 @300 loss | Q9 immediate disadvantage |
|---:|---:|---:|---:|
| 1729 | 5.9881 | 6.5423 | **+0.5542** |
| 271828 | 6.1710 | 6.7168 | **+0.5458** |
| 424242 | 6.0572 | 6.4933 | **+0.4361** |

Mean step-300 fixed-Q3 disadvantage for Q9: **+0.5120
nats/token**.

So the stronger schedule preserves the central trainability pattern:

> after equal 300-step compute, Q9-prepared masters are still a worse immediate
> ternary model than direct-Q3-prepared masters, yet after the remaining 900 Q3
> updates they finish far better.

## Preparation geometry changes under the stronger schedule

At step 300, the fraction of projected Q3 codes changed versus the initial
source is:

| Seed | Tuned direct | Matched Q9 |
|---:|---:|---:|
| 1729 | 4.57% | 4.88% |
| 271828 | 4.27% | 4.88% |
| 424242 | 4.38% | 4.87% |

Means:

- tuned direct: **4.41%**;
- matched Q9: **4.88%**.

This is much larger than the ~0.68% code displacement in the old constant-1e-4
v7 regime. Therefore the v7 code-overlap/Jaccard mechanism diagnostics should
not be assumed to transfer unchanged to the tuned regime.

## Interpretation

v10 closes the equal-global-schedule fairness loophole.

The strongest current empirical statement is:

> On SmolLM2-360M-Instruct under this 1200-step CE/KL QAT setup, Q9 -> Q3 with
> the same warmup+cosine global LR schedule as tuned direct Q3 finishes better
> in all three training orders by **0.6723 nats/token
> on average**, despite being **0.5120 nats/token
> worse as an immediate fixed-Q3 checkpoint after the first 300 updates.

This is still **not** a best-tuned-vs-best-tuned comparison: Q9 did not receive
an independent equal-budget hyperparameter search.

The next mechanism experiment should use this stronger matched schedule and
rebuild the direct/Q9 step-300 states inside the same run before constructing
the four-arm hybrid-master factorial.
