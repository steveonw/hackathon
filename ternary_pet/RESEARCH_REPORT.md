# Stepping Down to Ternary: A 9-State Stage Makes Ternary QAT Train Better

*steveonw · October 2026 · SmolLM2-360M-Instruct · all code, raw results and job IDs in this repository*

---

## 1. Summary

**Question.** When converting a pretrained language model to ternary weights
({−1, 0, +1}) with quantization-aware training (QAT), does it help to first train
for a while with a 9-state weight grid and then switch to ternary, compared with
training in ternary from the start for the same number of steps?

**Answer, in this setup: yes, by a large and replicated margin, and we can say
where the benefit comes from.**

1. **The effect.** With the same data, the same 1,200 updates and the same tuned
   learning-rate schedule, 300 steps at 9 states followed by 900 at ternary ends
   with **~49% lower perplexity** than 1,200 steps of direct ternary training
   (0.67 nats/token lower held-out loss; 3/3 training orders).
2. **It is trainability, not a better starting point.** At the moment of the
   switch, the 9-state-prepared model is a *worse* ternary model than direct
   training at the same step, yet it trains much faster afterwards. A
   full-precision warm-up of the same length gives no comparable benefit.
3. **It is localized.** About **96%** of the benefit is carried by the **~6.3%**
   of weights where the two methods pick different ternary codes after 300 steps
   (causal swap test, 3/3 orders).
4. **What matters on those weights** is *which* code the 9-state phase chose,
   committed at least moderately firmly. Its exact continuous values are not
   needed (a clean standard value works slightly better), while placing the same
   code barely across the threshold recovers only ~18%. The same number of code
   changes at random positions makes things *worse*.

**Scope.** One 360M model, WikiText-2 only, short runs (1,200 steps, ~150k
tokens). The models remain far from usable as assistants. Progressive
quantization in general is not new (§3); the contribution here is the
ternary-specific result with matched controls and the causal localization of
the mechanism.

**Current generalization phase.** Granite-4.0-350M is now complete across the
three preregistered orders, and the result is **mixed**. Full Q9→Q3 staging beats
direct on **2/3** orders (+0.1271, −0.1143, +0.0613 nats), so Granite does not
replicate Smol's 3/3 staging superiority. Yet several structural signals are
stable: Q9 is a worse immediate fixed-Q3 checkpoint on **3/3**, the D-vs-S
disagreement mask remains ~6% (mean **6.35%**), true-mask `d=0.5` beats full S
on **3/3**, and matched-random `d=0.5` is harmful on **3/3**. Thus the
position/code + interior-placement mechanism appears more robust than the raw
full-staging trajectory, but even M-d50 fails to beat D on seed 271828.

**Post-confirmation protocol sensitivity.** A deliberately transferred
Smol v10–v13 LR schedule (100-step warmup to `1e-3`, cosine to `1e-4`)
was tested on **two** targeted Granite orders, 271828 and 424242, after the
three-order constant-`1e-4` confirmation. Both D and full S became worse in
absolute held-out loss on both orders, and D-vs-S projected ternary disagreement
grew from 6–7% to about **30–33%**. On 271828, S fell 0.3603 nats behind D
and M-d50 also lost to D; on 424242, S was nearly tied with D while M-d50
**still beat D by 0.0285 nats**. The same schedule can therefore damage
Granite's endpoint and assignment stability without uniformly eliminating
Q9's useful code commitments. These are targeted exploratory comparisons,
not new independent seeds or a reason to revise the original Smol 3/3
matched-schedule result.


**G1-9 and G1-10: direct ternary gridward-pull improvement reproduced
on two targeted Granite orders.** Two preregistered equal-budget 2×2
WinQ-inspired factorials compared direct-Q3 (D), Gaussian only (G),
nine 10%-toward-current-code latent-master pulls (P), and both (GP),
at fixed constant `1e-4` LR. On known-hard order 271828,
D=5.72673, P=**5.48971** (gain **+0.23702** nats,
PPL −21.10%); on previously staging-positive order 424242,
D=5.80192, P=**5.52811** (gain **+0.27381** nats,
PPL −23.95%). Direct arms exactly reproduce both historical baselines.
P reduces sampled continuation ternary flip frequency by ~87% on both
orders. Gaussian only and the incremental GP-over-P effect **change sign**
between the two orders. This is strong *within-Granite training-order*
replication, **not cross-architecture confirmation** or proof that
reduced flips causally mediate improvement. Basic gridward interpolation
and Gaussian latent perturbation have published WinQ prior art.

**S1-1 cross-family negative (SmolLM2-360M).** The **same nine
10%-gridward pulls** were applied to Smol seed1729 under its previously
tuned continuous-Adam warmup-to-`1e-3`/cosine-to-`1e-4`
direct-Q3 schedule. **D held-out loss 5.595722, P 5.845746:
gridward makes Smol worse by 0.250024 nats and raises PPL
28.41%.** D exactly reproduces its historical tuned baseline,
all recorded tests pass and both arms had the same six AMP-skipped
updates. Yet P reduces sampled per-step code flips 78.41%;
thus suppressing transitions alone is **not a universal indicator
of better low-bit learning**. At step300/600 P is slightly better
on validation; by 900/1200 it is worse. These tests differ in
*architecture and training schedule*, not just model architecture.
Smol's previously replicated 3/3 positive **Q9→Q3 staging**
results are unaffected because this was a direct-Q3 gridward test.

---

## 2. Results at a glance

All numbers are held-out WikiText-2 (fixed 8,192-token slice). The original
BF16 model scores perplexity **39.9**.

| # | Claim | Key numbers | Evidence | Runs |
|---|---|---|---|---|
| 1 | 9→3 beats direct ternary, tuned schedule for both | ppl ~139 vs ~273; −0.672 nats; +6.8 pp teacher top-1 | 3/3 orders | v10, v9 |
| 2 | Advantage survives tuning the baseline | Tuning direct closes ~47% of the old gap; schedule-matched 9→3 restores it | 3/3 orders | v9, v10 |
| 3 | 9→3 starts ternary *worse*, ends better | +0.51 nats worse at step 300, −0.67 nats at step 1200 | 3/3 orders | v10 (also v7) |
| 4 | Full-precision warm-up doesn't reproduce it | FP32 prep ends ≈ direct (mean +0.08 nats worse), far behind 9→3 | 3/3 orders | v7 |
| 5 | Benefit is stored in the master weights | Resetting Adam and/or scales at the switch changes ≤0.03 nats | 1 order | v6 |
| 6 | ~96% of benefit lives on ~6.3% of weights | Mask-only transfer recovers 95.5–97.1% | 3/3 orders | v11 |
| 7 | Code choice + firm placement is what matters | Prototype values: 107%; barely-across: 18%; random positions: −11% | 3/3 orders | v12 |
| 8 | "Firm enough" is reached early | Depth 0.25 → 95%, 0.5 → 100%; code survival tracks it | 1 order | v13 |
| 9 | Firmness alone doesn't help direct | Prototyping direct's own codes: −0.03 / −0.01 nats | 1 order | v13 |
| 10 | Advantage shrinks with longer training (old schedule) | 6,000 steps: ppl 70 vs 85, gap 0.20 nats | 1 order | v5 |
| 11 | Granite G1 engineering gate only | BF16 source CE 3.2406 vs FP32 3.2354; FP16 non-finite; Q3/Q9 BF16 steps finite | 1 technical smoke, no scientific order | G1-0/G1-0b |
| 12 | Granite direct schedule frozen before Q9 | const 1e-4 val loss 5.8128 vs 6.0827 / 6.7402 for higher-LR schedules | 1 calibration order; held-out test unused | G1-1 |
| 13 | Granite staging generalizes on first order | Q9→Q3 −0.1309 nats; 12.27% lower PPL; Q9 is +1.925 nats worse immediately at step 300 | 1 order | G1-2 |
| 14 | Granite first-order mechanism result | exact-M 76.2%; true-M d=0.5 118.6%; matched-random harmful | order 1729 | G1-3 |
| 15 | Granite confirmations are mixed | full S wins 2/3; M-d50 > S 3/3; random harmful 3/3; mean mask 6.35% | 3 orders | G1-3/G1-4/G1-5 |
| 16 | Smol schedule transfer fails on Granite 271828 | D 6.0089 vs 5.7267; S 6.3692 vs 5.8410; M 32.71% vs 6.99%; d50 below D | 1 known-hard order, 4 arms, no new random control | G1-7b |
| 17 | Smol schedule transfer also degrades previously positive Granite 424242 | D 6.0571 vs 5.8019; S 6.0624 vs 5.7406; mask 29.49% vs 6.03%; d50 still beats D by 0.0285 | 1 selected weaker positive order, 4 arms, no new random control | G1-8 |
| 18 | Gridward latent-weight interpolation substantially improves direct Granite Q3 | D 5.72673; P **5.48971** (−0.23702 nat, −21.10% PPL); GP 5.48785; G-only 5.74337; sampled flips P ~87.2% lower | single historically hard order, four equal-budget arms, validation held out for diagnostics, no new random arm | G1-9 |
| 19 | Gridward pull effect replicates on second targeted Granite order | Seed424242 D 5.80192, **P 5.52811** (+0.27381 nat / −23.95% PPL); G 5.75636; GP 5.53534; sampled flips ~86.9% lower for P | two selected Granite orders (271828,424242), 4 equal-budget arms each, no independent family transfer | G1-9/G1-10 |
| 20 | Fixed gridward pull FAILS cross-family on Smol tuned direct ternary | Smol seed1729 D **5.59572**, P **5.84575** (P +0.25002 nats WORSE, +28.41% PPL); yet sampled code-flip frequency falls **78.41%**; within-job D matches historical D exactly | single previously known Smol order, two direct-Q3 arms, Smol LR1e-3 warmup/cosine and continuous Adam, architecture and protocol differences confounded | S1-1 |

---

## 3. Prior work and what this adds

Lowering precision in stages is an established idea:

- **Progressive quantization for CNNs.** Zhuang et al., *Towards Effective
  Low-bitwidth Convolutional Neural Networks* (CVPR 2018,
  [arXiv:1711.00205](https://arxiv.org/abs/1711.00205)), trained networks by
  stepping bit-width down in stages and found this beat quantizing to the target
  bit-width directly.
- **Progressive QAT for LLMs.** *Bit-by-Bit: Progressive QAT Strategy with
  Outlier Channel Splitting for Stable Low-Bit LLMs*
  ([arXiv:2604.07888](https://arxiv.org/abs/2604.07888)) reduces LLM weights
  8 → 4 → 2 bits stage by stage, with nested grids for multi-precision
  deployment. It does not target ternary.
- **Ternary / 1.58-bit LLMs.** BitNet b1.58
  ([arXiv:2402.17764](https://arxiv.org/abs/2402.17764)) established ternary
  LLMs trained with full-precision latent weights. *Continual Quantization-Aware
  Pre-Training* ([arXiv:2502.11895](https://arxiv.org/abs/2502.11895)) studies
  when to switch from 16-bit to 1.58-bit pre-training. **ParetoQ**
  ([arXiv:2502.02631](https://arxiv.org/abs/2502.02631)) reports a qualitative
  change in learning behaviour below ~3 bits, consistent with the 9→3 step being
  the hard one here.

**What this project adds** (to our knowledge, from a limited literature search):

1. A matched comparison of a **9-state → ternary** schedule against direct
   ternary QAT on a pretrained LLM, with a tuned baseline, an equal schedule,
   and a full-precision warm-up control.
2. Evidence that the benefit is **trainability** rather than a better starting
   ternary model.
3. **Causal localization** by intervention: the benefit is carried by a small,
   specific set of ternary assignments, and requires those assignments to be
   committed with some margin.

---

## 4. Method (canonical protocol, v10–v13)

**Model and data.** `HuggingFaceTB/SmolLM2-360M-Instruct`, BF16-rounded before
every treatment. Training on 1,200 WikiText-2 chunks of 128 tokens, one chunk per
step, in a seeded shuffled order (orders 1729, 271828, 424242). Both arms of
every comparison see the same chunks in the same order. Evaluation on a fixed
8,192-token WikiText-2 test slice.

**What is quantized.** All linear layers except `lm_head`. Embeddings, norms and
`lm_head` stay full precision and are **frozen**, so recovery must come from the
quantized weights.

**Quantizer.** One persistent FP32 master weight per quantized parameter;
straight-through estimator. Learnable per-output-row scale α, initialized at
1.5 × mean|w| of the row.
- Ternary: values {0, ±⅔α}, zero threshold at ⅓α (≈32% zeros at init).
- 9-state: values {0, ±¼α, ±½α, ±¾α, ±α}.

**Training.** AdamW (β = 0.9/0.95, no weight decay), gradient clipping 1.0.
Loss = 0.35 × cross-entropy + 0.65 × KL to the untouched original model.
Learning rate: 100-step linear warmup to 1e-3, cosine decay to 1e-4 at step
1,200 (selected for *direct* ternary in v9; the 9→3 arm received no tuning of its
own).

**Treatments.**
- **Direct:** 1,200 ternary steps.
- **9→3:** 300 steps with the 9-state forward pass, then 900 ternary steps. At
  the switch, the master weights are kept, the scales are reset to their
  original ternary values, and Adam is reset. The LR schedule continues without
  restarting.

**Mechanism experiments (v11–v13)** rebuild both step-300 master states inside a
single run, construct hybrid models, and continue each for the same 900 ternary
steps. Wherever two arms are meant to start from the same ternary model, the
scripts **assert zero code differences and identical pre-continuation loss**
before training; all such checks passed.

---

## 5. Findings

### 5.1 The effect (v9, v10)

| Order | Tuned direct ppl | 9→3 ppl | Loss gain (nats) |
|---|---|---|---|
| 1729 | 269.3 | **134.4** | 0.695 |
| 271828 | 276.7 | **141.3** | 0.672 |
| 424242 | 272.2 | **142.1** | 0.651 |
| **Mean** | 272.7 | **139.3** | **0.672** |

Mean teacher top-1 agreement +6.8 points; mean KL to the original −0.68.

History matters here. With the earlier constant LR of 1e-4, the gap was ~0.69–0.77
nats (v4b, v7). Tuning direct training's schedule (v9) closed about half of
that gap, but giving 9→3 the *same* schedule (v10) restored it to 0.67. So the
advantage is not an artifact of a weak baseline.

### 5.2 Trainability, not a better entry point (v7, v10)

At step 300, both preparations are projected to ternary with the same scales and
scored on the same data. The 9-state-prepared model is **worse** in every order
(+0.51 nats under the tuned schedule; +2.59 under the old one), yet it finishes
**0.67–0.69 nats better** after identical continuation. In v7's matched
training batches (old schedule, order 1729) it caught up with direct within
about 100 ternary steps and then pulled steadily ahead.

A **full-precision warm-up** for the same 300 steps produces an excellent
full-precision model but no comparable ternary benefit: it finishes about level
with direct (mean 0.08 nats worse; slightly better in one order), far behind
9→3. The coarse intermediate grid, not "adapt first", is what helps.

v6 (one order) showed the benefit is carried by the **master weights**: keeping
or resetting Adam state and quantizer scales at the switch changes the result by
at most 0.03 nats against a ~0.7-nat effect.

### 5.3 Localization (v11)

At step 300, the two preparations disagree on the ternary code of **6.31%** of
weights (range 6.18–6.46%). Swapping 9→3's master values onto direct's model
**only at those positions** recovers **96.4%** of the full advantage
(95.5 / 97.1 / 96.6%). Swapping the other ~94% of weights alone gives a small
gain (0.11 nats) that becomes negligible (0.025 nats) once the disagreement
positions are swapped.

Both preparations change a similar number of codes, but mostly *different*
ones: their changed sets overlap with Jaccard ≈ 19% in all three orders.

### 5.4 What matters on those positions (v12, v13)

Holding the starting ternary model fixed and varying only where each master
weight sits inside its code's region:

| On the 6.3% positions | Share of gain (3 orders) |
|---|---|
| 9→3's exact master values | 100% (reference) |
| 9→3's code, at the standard value for that code | **107%** (106.5–107.7) |
| 9→3's code, placed just across the threshold | 18% (16.4–20.5) |
| Same kinds of code changes at random positions | −11% (worse than direct) |

The depth sweep (v13, one order) shows the benefit switches on quickly and then
saturates: depth 0.03 → 16%, 0.25 → 95%, 0.5 → 100%, with no further gain
deeper. The fraction of those positions still holding the 9-state-selected code
after 900 ternary steps tracks this closely (52% → 86% → 97% → 99%), consistent
with shallow placements being undone during later training. Survival is
associated with the benefit but not proven to be its only cause: at the
shallowest depth about half the codes survive but only 16% of the gain appears.

**Firmness alone does not help.** Snapping *direct's own* codes to their
standard values, either on the same positions or on direct's own recently
changed weights, makes results slightly worse (−0.03 and −0.01 nats).

### 5.5 Mechanism in one sentence

The 9-state phase selects a specific small set of ternary assignments that differ
from the ones direct training makes; committing those assignments with moderate
margin keeps them from being undone, and that accounts for most or all of the
advantage, while the same commitment applied to other choices does not.

### 5.6 Cross-family Granite result (G1, order 1729)

Granite-4.0-350M reproduces the central trainability signature: after 300 equal
updates, Q9-prepared masters projected into the common Q3 system are much worse
than direct (**7.7425 vs 5.8174 loss**), yet Q9→Q3 finishes better
(**5.5350 vs 5.6658**, 12.27% lower PPL).

The projected D-vs-S disagreement mask is **6.0246%**, again close to the ~6%
scale seen on SmolLM2.

Under the common fresh-Adam continuation used for G1-3:

| Arm | Held-out loss | Gain vs D | Recovery of full S |
|---|---:|---:|---:|
| D | 5.66205 | — | — |
| S | 5.53495 | 0.12710 | 100% |
| M-exact | 5.56525 | 0.09680 | **76.2%** |
| M-d50 | **5.51134** | **0.15071** | **118.6%** |
| Random-d50 | 5.73514 | −0.07308 | −57.5% |

All S / M-exact / M-d50 arms begin with the same projected ternary forward
weights and diagnostics. The matched-random arm preserves mask size, layer
allocation and source→target transition counts but uses different positions.

This supports the **qualitative** cross-family mechanism: Q9's useful information
is in specific position/code choices plus useful interior placement, not exact
Q9 continuous values or generic code snapping. But exact-mask localization is
weaker on Granite (76.2%) than on SmolLM2 (~96.4%), so the quantitative
partition of the effect appears family-dependent.

For M-d50, Q9-selected-code survival is 99.68% after 100 continuation steps,
97.98% after 300, and **90.70%** after 900; learned-scale and fixed-alpha0
survival are nearly identical.

### 5.7 Why Granite seed 271828 fails: post-hoc forensics

The one negative Granite order is not obviously endpoint noise. Its Q9
preparation is already qualitatively different before the switch.

| diagnostic | 1729 (+) | 271828 (−) | 424242 (+) |
|---|---:|---:|---:|
| native Q9−D loss @300 | −0.160 | **+0.278** | −0.138 |
| Q9/D grad ratio @100 | 0.82× | **3.33×** | 0.76× |
| D-vs-S mask | 6.02% | **6.99%** | 6.03% |
| changed-set Jaccard proxy | ~27.5% | **~19.7%** | ~27.5% |
| true d50 − random advantage | 0.224 | **0.0077** | 0.110 |

On the two positive orders, native Q9 validation is already better than native
direct at step 300 and Q9 gradients have settled to direct-like magnitudes.
On 271828, native Q9 is 0.278 nats worse, Q9 gradients remain much larger, and
Q9 training loss is worse than direct at every logged prep checkpoint.

The negative order also has a harmful same-code/off-mask Q9 contribution.
Replacing Q9's off-mask masters with direct masters improves loss by 0.0365
nats. Standardizing the true-mask Q9-selected codes at d=0.5 repairs another
0.0195 nats, but the arm remains worse than direct and only 0.0077 nats better
than matched random.

Global Q9 histograms and scale statistics are nearly identical across seeds,
as are d50 survival (~90%) and continuation code churn. The failure therefore
looks more like **poor Q9 assignment discovery** than global scale drift,
excessive late churn, or an evaluation accident.

A notable but incomplete geometric clue is greater late-layer disagreement:
layer 27 contains 4.56% of the negative seed's mask versus ~3.1% on the positive
orders, driven mainly by shared-MLP input/output weights. This explains only a
minority of the extra disagreement.

Working hypothesis: Q9 staging succeeds when the 300-step Q9 phase itself
reaches a settled native-Q9 state and discovers useful alternative ternary
commitments. This is post-hoc n=3 evidence, not a validated predictor.

Detailed analysis:
`replications/g1_granite350m_seed271828_forensics.md`.

### 5.8 Transferring Smol's v10–v13 schedule to Granite (G1-7b)

This was a deliberate protocol-sensitivity test after Granite's three-order
constant-`1e-4` result, conducted **only on the known-hard order 271828**.

Smol v10–v13 used a 100-step warmup to `1e-3`, followed by global cosine
decay to `1e-4` at step 1,200. Prior Granite G1 instead used constant
`1e-4`, chosen by direct-only G1-1 calibration. G1-7 copied the Smol global
LR curve while preserving Granite's BF16-safe compute, FP32 masters, data,
300/900 split, original Q3 scale reset and fresh Adam. The first attempt
(`6ac722dfdf2184ac91ac75e9`) failed technically after 300-step preparation:
the enlarged disagreement mask made a **full-size layer/source→target matched
random control impossible**. The amended, preregistered four-arm retry
(`6ac7248adf2184ac91ac768d`) completed successfully; all equal-forward
assertions passed.

| Arm | Constant-LR 271828 final loss | Transferred Smol schedule loss |
|---|---:|---:|
| D | **5.72673** | **6.00895** |
| S | 5.84103 | 6.36920 |
| M-exact | 5.80453 | 6.35442 |
| M-d50 | 5.78502 | **6.30402** |

The full S deficit versus D worsens from **0.11431** to **0.36025**
nats/token. D itself gets worse by 0.28222, so the transferred schedule hurts
both routes, but S more. M-d50 still improves on full S by 0.06517 nats, yet
remains 0.29508 nats worse than D.

The sharpest change is in the **discrete assignment regime**:

- D moved **27.11%** of source Q3 codes by step 300; S moved **23.43%**,
  compared with only ~5% movement each at the old constant LR.
- D/S projected-Q3 disagreement grew from **6.995%** to **32.715%**
  (81,642,667 of 249,561,088 targeted weights).
- Native Q9 validation was still 0.38936 nats worse than native D at step 300,
  with a **5.15×** preclip gradient-norm ratio.
- Selected Q9 code survival with d=0.5 after 900 further Q3 steps fell from
  **89.59%** to **60.75%**. Fixed-alpha0 and learned-scale survival still
  nearly coincide (60.70% vs 60.75%).

The original immediate fixed-Q3 Q9 disadvantage actually *shrinks* under the
new schedule (0.865 to 0.428 nats), while final staged performance deteriorates.
This reinforces that **entry quality alone does not predict later trainability**.

**Interpretation:** the Smol-tuned `1e-3` peak does not transfer cleanly
to this Granite order. The vast mask expansion, different code churn, and weaker
survival are associated with the worsened endpoint, but do not by themselves
identify the causal explanation. No random-control specificity claim is made
under this schedule. This is one development order, not a general negative
for all Granite seeds. The earlier mixed three-order results remain intact.

Raw:
`results/run_g1_7b_granite350m_v10schedule_seed271828_2026-10-08.json`.
Summary:
`results/run_g1_7b_granite350m_v10schedule_summary.md`.

### 5.9 A second Granite order confirms aggressive LR sensitivity, but preserves d50 signal (G1-8)

The previously positive order **424242** was selected before launch as the
weaker of the two positive Granite orders (historical S−D benefit 0.06135 nats
versus 1729's 0.12710). The G1-8 script is a **one-line seed-only change**
from the successfully completed four-arm G1-7b schedule-transfer harness.
Smol v10–v13's global warmup-to-`1e-3`/cosine-to-`1e-4` LR curve was
unchanged; Granite kept its BF16 student/teacher path, FP32 masters, original
Q3 scales, fixed data/evaluation protocol and 300/900 updates. All four arms
completed, and S, M-exact and M-d50 passed zero-code-difference and
equal-forward checks at the switch.

| Arm | Constant-LR Granite 424242 loss | Smol schedule Granite 424242 loss |
|---|---:|---:|
| D | **5.80192** | 6.05708 |
| S | 5.74057 | 6.06241 |
| M-exact | 5.74471 | 6.04609 |
| M-d50 | **5.71016** | **6.02857** |

**All four methods degrade in absolute loss** under the transferred
schedule. The full S advantage over D changes from **+0.06135** to
**−0.00534** nats (essentially a near-tie in this one run). But the M-d50
arm **still improves on the new direct baseline by 0.02850 nats**, and on
the new full S by 0.03384 nats. The stage's full FP32 state and its discrete
Q9-selected commitments thus remain distinct interventions.

The high peak LR again changes discrete assignment geometry dramatically:

- Q3 direct code movement from the original source increases to **27.280%**;
  projected Q9 movement increases to **27.124%**, up from ~5.3% each.
- D-vs-S projected-code disagreement increases from **6.0296%** to
  **29.4904%**: 15,047,433 to **73,596,548** code disagreements.
- At continuation step 900, M-d50 selected-code survival falls from
  **90.57%** to **57.14%**. Learned-scale and fixed-reference-scale survival
  remain nearly the same (57.14% vs 57.12%).

**Crucially, Q9 preparation did not fail in the same way on both seeds.**
On 424242 under the high-LR schedule, *native Q9* validation loss at
step300 was **6.21319**, better than direct's **6.30551**. Q9's preclip
gradient norm was **1.257** versus D **1.941**. On 271828 under the
same schedule Q9's native loss was **0.38936 nats worse than D** and its
gradient norm was **5.15× D**. The adverse **schedule sensitivity** appears
on both tested orders even when native Q9 has successfully trained.

Thus the data support a narrower cross-order interpretation: directly copying
Smol's LR peak **makes both Granite D and S worse on two targeted orders**
and pushes the model into a different ternary code-movement regime. It does
**not** prove Q9 is never useful under this schedule (M-d50 helps on 424242),
nor prove that mask inflation causes the final loss. The earlier three-order
constant-LR confirmation remains mixed but valid. Both G1-7b and G1-8 omit a
matched-random arm after the documented infeasibility of the old full-size
control; there is no new position-specificity comparison under high LR.

Completed HF job: `6ac79551e7a0dae8a2788f0b`.
Frozen code: `39399baff38b15f961e9571f142e193a783c925b`.
Raw result:
`results/run_g1_8_granite350m_v10schedule_seed424242_2026-10-08.json`.
Interpretation:
`results/run_g1_8_granite350m_v10schedule_summary.md`.

### 5.10 A different route: stabilize direct ternary training by gridward interpolation (G1-9)

After Smol's successful staging and Granite's mixed staging and negative
high-LR transfers, a preregistered **WinQ-inspired 2×2 factorial** on
Granite's known-hard order 271828 isolated two interventions in
**direct ternary QAT**, without any Q9 preparation:

- **D:** ordinary hard-Q3 forward with FP32 masters and STE.
- **G:** add row-scale-normalized Gaussian noise before training-time hard Q3
  quantization (`σ=0.04×α_row` until step900, linearly decaying to zero
  by step1200); evaluation remains hard, noiseless ternary.
- **P:** after optimizer updates every 100 steps through step900, set
  `W ← 0.9W+0.1Q3(W)` for the FP32 masters, at their current row scales.
- **GP:** both.

All four arms see the **same 1,200 ordered training chunks**, objective,
fixed teacher, BF16 compute, AdamW, constant `1e-4` LR, 300-step scale reset
and fresh-Adam boundary, and 8,192-token held-out evaluation. All
predeclared checks passed. The D arm reproduced its independent prior
seed271828 held-out result **exactly** (5.726725).

| Arm | Held-out loss | PPL | Gain vs D | Teacher top-1 |
|---|---:|---:|---:|---:|
| D | 5.726725 | 306.96 | — | 27.66% |
| G | 5.743370 | 312.11 | −0.016645 | 27.10% |
| **P** | **5.489709** | **242.19** | **+0.237016** | **31.08%** |
| GP | **5.487853** | **241.74** | **+0.238872** | 31.04% |

**Main result:** periodic gridward interpolation improves held-out loss
by **0.237 nats**, lowers PPL **21.10%**, and improves teacher-top1
agreement by **3.42 percentage points** at equal optimizer-step budget.
Gaussian alone fails to improve D. Combining Gaussian with pull gives
only an additional **0.001856-nat** improvement beyond P (no meaningful
claim of added noise benefit from one order).

A deterministic sample of 32,768 weights across 16 layers monitored their
noise-free ternary codes after every optimizer step. During the 900-step
continuation, sampled code-change probability per weight-update was
**0.000736 (D)**, **0.000741 (G)**, **0.0000945 (P)** and
**0.0000893 (GP)**. The gridward-only arm shows **~87.2% fewer**
sampled code flips, alongside a lower final fraction of source-code changes
(D 10.335% vs P 5.281%). But the immediate two-step reversal **share**
among flips decreased only modestly; this does not show that reversal
oscillation alone mediates the result. Moving masters deeper into quantization
cells changes future STE dynamics, so additional controls would be needed
to separate the effects.

At 300 steps, the gridward-only arm's native Q3 validation loss already
improved from D **5.64539** to **5.50481**, preceding the final 900-step
benefit. These are same protocol validation slices, not final held-out scores.

**Boundaries:** The positive 271828 development outcome does not undo
earlier Granite mixed three-order staging results, Smol three-order staging
benefits, or G1-7b/G1-8 aggressive-LR failures. This is a different,
direct-Q3 training algorithm tested on one known order. It does not
establish a new Q9 mask-specificity result or a win across models.
Gridward interpolation and Gaussian latent noise have prior art
(e.g. WinQ, ICML 2026); novelty would require an additional contribution
such as generalizable boundary-aware control, not merely implementing
the pull rule. A frozen-parameter replication is the next priority.

Completed GPU job: `6ac821ec095c5780892ff7f9`.
Code pin: `6ce4c4056e8cfd3292c54f34bedc68bf7691688f`.
Raw:
`results/run_g1_9_granite350m_gaussian_pull_seed271828_2026-10-08.json`.
Full analysis:
`results/run_g1_9_granite350m_gaussian_pull_summary.md`.

### 5.11 G1-10: a seed-only replication confirms the direct ternary gridward-pull gain on Granite 424242

Following the strong G1-9 gridward-pull result on historically negative Granite
order 271828, **424242** was prospectively selected as a previously
full-staging-positive order for an otherwise **identical four-arm
factorial**. The implementation was committed and the experiment was
preregistered before the GPU run. The source diff from G1-9 has **only
one line changed**, `SEED=271828` to `SEED=424242`. All D/G/P/GP
arms trained 1,200 identical-length updates per arm, same 300-step
scale/Adam reset, constant LR `1e-4`, noise and gridward settings,
hard ternary inference, and the same 8,192-token WikiText-2 test slice.
All saved construction/step/finite/pull-count checks passed.

| Arm | G1-9: 271828 loss | G1-10: 424242 loss | G1-10 PPL | G1-10 D−arm gain |
|---|---:|---:|---:|---:|
| D | 5.726725 | **5.801920** | 330.93 | — |
| G (Gaussian only) | 5.743370 | 5.756362 | 316.20 | +0.045558 |
| **P (gridward only)** | **5.489709** | **5.528112** | **251.67** | **+0.273808** |
| GP | 5.487853 | 5.535340 | 253.49 | +0.266580 |

The G1-10 D arm again **exactly reproduces** its corresponding historical
constant-LR baseline, 5.801920056. P lowers held-out NLL by
**0.273808 nats** and PPL by **23.95%** on 424242; the original
271828 P gain was **0.237016 nats / 21.10% lower PPL**.
The two targeted-order mean D−P gain is **+0.255412 nats** (not an
unbiased population estimate or a formal cross-seed confidence interval).
This supports a repeatable **within-Granite** advantage for the
fixed gridward-pull training method, including one order previously
unfavorable to Q9 staging and another favorable to Q9 staging.

Gaussian is **not independently validated as essential**. G-only
worsened D by 0.016645 on 271828 but improved D by 0.045558 on
424242. GP was 0.001856 better than P on 271828 but **0.007228
worse** than P on 424242. Thus the consistent improvement is P,
while G and GP interactions change signs across these orders.

Clean sampled per-step Q3 transitions (32,768 sampled weights,
16 layers) for continuation: 424242 D **0.000728421** flips per
weight-update, P **0.000095486**, a **~86.9% reduction**;
271828 D **0.000735745**, P **0.000094469**, a **~87.2% reduction**.
Final code movement from initial 424242 D **10.327%**, P
**5.369%**. The fraction of immediate two-step reversals among
flip events changes less dramatically than the total rate, and
interpolation alters latent-master positions and future optimizer
trajectories. We have **not** isolated why lower transition activity
and lower validation loss travel together.

**Raw-data provenance caveat:** Because only the seed line changed,
G1-10 retained the G1-9 internal `kind` / event labels and
stale `historic_D_loss_271828=5.726725` /
`D_minus_historic_D_loss=+0.075195` payload fields.
Those two historical-comparator fields should **not** be used as
a 424242 baseline check. We preserved the raw JSON unchanged and
verified independently that G1-10 D=5.801920056 is identical to
previous archived Granite 424242 D=5.801920056. Identify G1-10
by HF job ID, pinned code SHA, and recorded seed424242.

**Scope and prior art:** Gridward latent-master interpolation
has relevant precedent in WinQ (ICML 2026). This is an adaptation
with consistent performance on two *already observed* Granite
orders under a short fixed recipe, not an original method,
cross-model success, or a new Q9-vs-random specificity test.
Original Granite three-order Q9 staging outcomes (2/3 positive)
and Smol staging 3/3, including all earlier negative outcomes,
remain valid as separate interventions.

Job: `6ac82c93fee2c900701711db` (completed 2026-10-09 00:21:46 UTC).
Code pin: `e08659a51fd0d72ed85f01f8e7ce739283ca6c61`.
Raw: `results/run_g1_10_granite350m_gaussian_pull_seed424242_2026-10-08.json`.
Full summary: `results/run_g1_10_granite350m_gaussian_pull_seed424242_summary.md`.
No new GPU jobs started after G1-10.

### 5.12 S1-1: fixed gridward-pull rule does not transfer to Smol's tuned direct Q3

After both targeted Granite orders benefited from nine periodic
10%-gridward pulls, the cross-family **S1-1** test was
preregistered before implementation or GPU use. We held the
**pull rule** fixed but used the independently validated
**SmolLM2-360M-Instruct direct-Q3 baseline**, not Granite's
optimizer. Seed1729 is a previously studied development order,
so the test is not a blind new-order draw.

Arms: **D** ordinary direct Q3 and **P** identical direct Q3
plus `W←0.9W+0.1Q3(W)` after optimizer steps100,200,...900.
Both use continuous AdamW for all 1,200 step opportunities,
Smol's global warmup100 peak `1e-3` followed by cosine decay
to `1e-4`, FP32 master weights and Smol FP16 AMP,
the same CE/KL distillation and ordered WikiText-2 train
chunks, with no Gaussian and no Q9 prep. Unlike Granite,
there is **no scale restoration/Adam reset at step300**.

| Smol direct-Q3 method | Final heldout NLL | Perplexity | Teacher top-1 | Teacher KL |
|---|---:|---:|---:|---:|
| **D, tuned direct** | **5.595722** | **269.27** | **29.87%** | **2.66403** |
| P, tuned direct + gridward | 5.845746 | 345.76 | 27.28% | 2.93324 |

**Primary result: P is worse by 0.250024 nats/token,
perplexity +28.41%.** The D control exactly reproduced
the independent v9 tuned-direct seed1729 endpoint
`5.595722187310457`. All nine recorded structural
and schedule checks passed. Both arms took 1,200
step opportunities with **six AMP-skipped optimizer
updates each** (1,194 effective Adam updates apiece),
and P executed all nine gridward pulls after successful
optimizer steps.

**Code-transition diagnostics:** a fixed 32,768-position
sample across 16 layers, monitored after each step,
changed ternary codes with probability **0.000435054**
per weight/step for D versus **0.000093918** for P.
Gridward suppressed sampled per-step transitions by
**78.41%**, while the final full-network fraction
of ternary codes differing from initialization
decreased from **6.960% (D)** to **3.445% (P)**.
**This is a useful counterexample to the claim that
reducing quantized-code changes alone improves quality.**

The validation timeline reveals a reversal in relative
performance (these are training-heldout diagnostics,
not checkpoint choices from test data):

| Global step | D validation NLL | P validation NLL | Relative P−D |
|---|---:|---:|---:|
| 300 | 5.98809 | 5.97667 | −0.01142 (P slightly better) |
| 600 | 5.74032 | 5.67064 | −0.06968 (P better) |
| 900 | **5.41510** | 5.58253 | +0.16743 (P worse) |
| 1200 | **5.25594** | 5.57738 | +0.32144 (P worse) |

Sampled code-flip frequency in the P arm approaches
zero during the final few hundred steps while
the clean D arm continues improving and retaining
some assignment changes. **Premature commitment is
a plausible hypothesis, not a mechanism proof**;
the pull also shifts the location of FP32 masters
and changes downstream STE dynamics.

**Interpretation boundary:** Prior G1-9/G1-10
Granite +0.237/+0.274-nat gains at constant
`1e-4`, with a step300 optimizer/scale reset,
remain valid but do **not** translate into a
universal direct-Q3 recipe. The cross-model comparison
changes architecture **and** family-specific LR/reset
trajectory, and does not identify which difference
drives opposite effects. It also does not test
Gaussian smoothing or Q9→Q3, so the earlier replicated
Smol staged-Q9 advantages remain intact.

G1-9/G1-10 and S1-1 are **WinQ-inspired
adaptations of published gridward concepts**, not
proof of a new method. Next hypothesis would be to
**stabilize selectively** according to boundary
margins, consistent useful code motion and training
phase, while avoiding hard commitment of uncertain
weights. Any adaptive-pull tuning must be
preregistered and validated independently, not
retuned against the examined Smol held-out test.

Completed job: `6ac83c3bfee2c90070171b1a`,
2026-10-09 01:18:20 UTC.
Code pin: `a5634bce459092a503e7534e6b89b6d3a019548b`.
Raw JSON:
`results/run_s1_1_smol360m_gridward_direct_q3_seed1729_2026-10-09.json`.
Summary:
`results/run_s1_1_smol360m_gridward_direct_q3_seed1729_summary.md`.
No additional GPU jobs launched.

---

## 6. Limitations

- **One clean replicated family plus one mixed cross-family result.**
  SmolLM2-360M is positive 3/3. Granite-350M is positive for full staging on
  2/3 orders, with one genuine negative order. The structural d=0.5/random
  intervention pattern is more consistent than full staging, but M-d50 still
  fails to beat D on seed 271828. Both currently use WikiText-2.
- **Cross-family LR sensitivity.** The Smol v10–v13 warmup-to-1e-3
  schedule degrades *both D and S absolute endpoints* on two selected Granite
  orders (271828, 424242); the disagreement fraction expands from ~6–7% to
  ~30–33% and M-d50 survival decreases sharply. It does not, however,
  uniformly erase the d50 intervention advantage. Neither the Smol-tuned peak
  nor Granite's original 300-step calibration proves a universally optimal
  schedule. The two orders are historically selected development tests, not
  independent confirmations.
- **G1-9/G1-10 pull effect replicated only within Granite.** P improved
  direct Q3 on previously seen seeds271828 (+0.237 nats) and 424242
  (+0.274 nats) with identical settings, while Gaussian-only and
  GP-over-P effects changed sign. Neither seed is an independent new
  architecture or a blind confirmatory draw. Improvements coincide
  with ~87% fewer sampled per-update code flips, but do not prove
  boundary-transition suppression caused the loss gain. The core
  interpolation method already has prior art (WinQ).
- **Cross-model gridward negative (S1-1).** The same nine 10% pulls
  improve tuned direct ternary on two selected Granite orders under
  Granite's constant 1e-4/step300-reset protocol, but **harm** tuned
  direct Smol (loss +0.250 nats, PPL +28.41%) under Smol's
  warmup1e-3/cosine continuous-Adam protocol. Smol monitored
  78.41% fewer code transitions despite degradation, directly
  contradicting the naive "fewer flips always better" rule.
  Architecture vs LR/reset interactions are not isolated;
  all tests are short and historically observed orders.
- **Short training.** 1,200 steps of single 128-token chunks (~150k tokens). In
  the one longer run (v5: 6,000 steps, old constant-LR schedule, one order) the
  gap **shrank** from ~0.7 to 0.20 nats (ppl 70 vs 85). Whether it persists
  asymptotically under the tuned schedule is unknown.
- **Usability.** Even the best models (ppl ~70–140) produce repetitive
  WikiText-style fragments and do not follow prompts.
- **Baselines.** The schedule was tuned for direct training only; 9→3 received
  no tuning, which should if anything understate its advantage. Other strong
  direct-ternary recipes (e.g., different scale schemes, longer warmup) were not
  tried.
- **Seeds.** Claims 5, 8 and 9 rest on one training order.
- **Design details.** Embeddings and `lm_head` are full precision; the 9-state
  and ternary grids are not nested (9-state ±α maps to ternary ±⅔α).

---

## 7. Possible next steps

- **Investigate selective or time-limited commitment, not flip minimization:**
  S1-1 has now tested the identical nine 10%-gridward pulls on
  Smol's tuned direct-Q3 training. It **failed badly** despite
  78.41% fewer sampled code transitions. Before further GPU
  runs, compare existing boundary-margin distributions,
  cross-phase transition stability, gradient directions and
  checkpoint validation trajectories of Granite G1-9/G1-10
  versus Smol S1-1. Consider preregistered
  confidence- or schedule-conditioned pulls that permit
  beneficial late code changes. Do not choose λ or
  early-stop rules based on the already-viewed Smol test.
  Independent validation and new seeds are required for
  any tuned method.
- **Generalization and schedule sensitivity:** Granite confirmation is
  mixed (2/3 positive at constant 1e-4). Copying Smol's higher-peak LR made
  both tested orders substantially worse and drove ~30–33% disagreement
  masks. A future schedule experiment should use preregistered,
  model-aware calibration on validation-only data rather than assuming
  Smol's selected peak transfers. The code-choice/placement benefit persists
  on one schedule-transfer order, so do not conflate overall LR degradation
  with complete rejection of the Q9 commitment mechanism.
- **A cheaper recipe:** if the useful ~6% of assignments could be predicted
  without a full 9-state phase, the benefit could be had at lower cost.
- **Level count:** test other intermediate grids (e.g., 5 or 7 states, or a
  nested 9→3 grid).

---

## Appendix A. How the project got here (v1–v8)

The full chronological record, including all dead ends, is in
[`SHAREABLE_RESEARCH_REPORT.md`](SHAREABLE_RESEARCH_REPORT.md) and `results/`.
Short version:

- **v1–v3 (negative, due to setup problems):** naive 27→9→3 staging and
  transition schedules showed no benefit. Diagnosis: the quantizer zeroed ~79% of
  weights (scale initialized from the row maximum), the LR of 2e-5 was too low
  for any ternary code to change, and v2's hard commits between stages froze
  learning. With no training between stages, nested 27→9→3 rounding gives exactly
  the same ternary model as direct rounding, so any benefit must come from
  adaptation.
- **v4 (first positive result):** fixed quantizer (~32% zeros), LR 1e-4,
  persistent FP32 masters, code-flip logging. 9→3 beat direct at equal steps;
  extra direct steps (1,200 vs 900) did not help direct.
- **v4b (replication):** shuffled data orders; 9→3 won 3/3 (mean −0.77 nats).
- **v5 (scale-up):** 5× more training; advantage persisted but shrank (§6).
- **v6:** benefit carried by master weights, not Adam or scales.
- **v7:** equal-compute comparison; trainability effect and full-precision
  control (§5.2).
- **v8 (negative):** tested whether 9-state training pre-pushes weights toward
  their future thresholds. Not supported: each arm looked "aligned" only on its
  own future-flip set, the pattern expected from selection effects.
- **v9–v13:** §5.

## Appendix B. Reproducibility

Every run has a pinned commit, a Hugging Face Jobs ID and a raw JSON result in
`results/` or `replications/`; aggregate summaries for each replicated claim are
in `replications/*_aggregate_summary.md`. Runs used single T4 or A10G GPUs.

## Acknowledgments

Experiments were designed, run and reviewed with the help of AI assistants;
the original hypothesis and research direction are the author's.

## T1 update — how early are the Q9-selected future ternary decisions visible? (October 8, 2026 EDT)

A preregistered **single-order timing diagnostic**, T1 (seed1729), completed on Hugging Face (job [6ac85534fee2c90070172a41](https://huggingface.co/jobs/codeflash85/6ac85534fee2c90070172a41), pinned code `2a982be6910d99eaf53d74f6e3ec3dae5807de17`). D and S were trained with matched v10 Q3/Q9 preparation conditions for 300 steps each, without Q3 continuation; their FP32 masters were projected through original Q3 scales at checkpoints 0/50/100/150/200/250/300. No held-out test set was evaluated.

The projected Q3 **D/S disagreement mask grows gradually**. Recall of the final step300 mask is **18.61% at step100**, **35.42% at150**, **51.35% at200**, and **69.40% at250**. The step250 mask precision against step300 is **77.77%**; it is thus still missing **30.60%** of final step300 disagreements. The same progressive timing applies to both the **Q9-only changes** and the **direct-only changes that Q9 avoids**. The job passed numerical reproducibility checks and exactly reproduces the prior **6.45808%** step300 mask on seed1729. The fixed-original-scale direct validation loss was 0.01623 higher than direct native loss, within the frozen tolerance; diagnostics are distinct.

**What it does not establish:** earlier mask overlap is not the final Q3 loss advantage, an online prediction rule, or a demonstration that earlier switches are sufficient. The precision/recall values use hindsight knowledge of the step300 disagreement set; one seed does not establish generality. The separate contributions of Q9-only and direct-only code assignments remain causally unresolved.

**Next experiment decision:** choose and preregister a bounded causal-split study at step300, or a 200/250/300 Q9→Q3 continuation study that holds total training opportunities fixed. No follow-up GPU job was launched by T1 analysis.

Records: [T1 raw JSON](results/run_t1_q9_discovery_timing_seed1729_2026-10-08.json), [T1 detailed summary](results/run_t1_q9_discovery_timing_seed1729_summary.md), [preregistration](research_log/phase_a_q9_timing_seed1729_prereg_2026-10-08.md), and [research-history motivation](research_log/2026-10-08_q9_assignment_discovery_timing.md).

## C1 update — Q9 for 250 steps preserves 95.75% of the 300-step staging benefit (October 8, 2026 EDT)

A frozen exploratory follow-up to the one-seed T1 timing study tested an **actual training-time Q9→Q3 switch** 50 steps earlier, not merely the resemblance of code masks. This used the same 1200 scheduled step budget and global LR schedule in all arms, with the same seed/order 1729, BF16-rounded SmolLM2-360M source, WikiText-2 teacher objective, FP32 masters and FP16 autocast.

| Arm | Schedule | Final held-out NLL ↓ | Final PPL ↓ |
|---|---|---:|---:|
| D | Direct Q3 for 1200 | 5.595722 | 269.272 |
| S250 | Q9 250 → Q3 950 | **4.930515** | **138.451** |
| S300 | Q9 300 → Q3 900 | **4.900985** | **134.422** |

At the Q9→Q3 transition, the original Q3 scales and a fresh AdamW/GradScaler were restored; the global LR schedule was **not restarted**. D used continuous AdamW and learned scales. D and S300 matched historical held-out losses *exactly* and technical checks passed. Equal **1200 update opportunities** did not mean equal actual optimizer updates (direct 1194; staged 1190 after AMP skips).

**Predeclared exploratory outcome:** S250 is **+0.0295299 nats** worse than S300 but preserves **95.7495%** of the S300 improvement over D. Both nominal test thresholds, extra loss <=0.07 and preserved gain >=90%, passed. S250 remains **0.665207 nats/token** better than direct Q3.

**Interpretation:** under these finite-budget conditions, cutting **50 Q9 steps**, replaced by **50 additional ternary Q3 steps**, sacrifices comparatively little end quality. T1 nevertheless found that step250 includes only ~69.4% of the eventual projected-code disagreement mask. It is therefore **not proven** that all Q9 assignments are learned by step250, or that the last 50 Q9 steps are useless; instead, some benefit may be recoverable or replaceable through later Q3 updates. The two Q9→Q3 transition checkpoints have nearly equal immediate fixed-Q3 projection losses (6.542107 vs 6.542292), even though Q9 native-loss differs (5.390830 vs 5.182719). Immediate projected loss alone still does not establish later trainability.

This was **one familiar seed/order** and a previously evaluated held-out slice. Step250 was selected after inspecting T1 masks and prior historical outcomes, so this is *exploratory evidence*, not independent statistical confirmation, a general optimal schedule, or a production-quality model. A fixed-protocol replication on new orders is needed. No additional GPU jobs launched after C1.

Records: [frozen protocol](research_log/phase_c_q9_switch250_vs300_seed1729_prereg_2026-10-08.md); [HF job](https://huggingface.co/jobs/codeflash85/6ac8597afee2c90070172c79); [raw parsed C1 JSON](results/run_c1_smol360m_q9_switch250_vs300_seed1729_2026-10-08.json); [detailed result and dev trajectories](results/run_c1_smol360m_q9_switch250_vs300_seed1729_summary.md).

## M1 update — matched Q3 continuation and final code destinations (October 9, 2026 UTC)

M1 tested the remaining ambiguity from C1. **M1 is deliberately different from C1's equal-total-training-budget recipe test.** It saves a single Q9 trajectory at steps250 and300, then uses **identical 900-step Q3 continuation batches and learning rates** for both saved FP32 master states. Both continuations use the original source Q3 scales and fresh AdamW/GradScaler, with 897 successful Q3 updates and 3 AMP skips each.

| Q9 preparation | Shared Q3 continuation | Total update opportunities | Held-out NLL ↓ | PPL ↓ |
|---|---:|---:|---:|---:|
| Step250 | 900 | **1150** | **4.950432** | 141.236 |
| Step300 | 900 | **1200** | **4.900985** | 134.422 |

**The extra 50 Q9 preparation steps confer a +0.0494467 nats/token advantage when the downstream Q3 training operator is held fixed.** All seven prespecified technical/reproduction checks passed, including the exact historical Q9(300)→Q3(900) endpoint. The prior C1 test instead showed 250 Q9 +950 Q3 preserved 95.75% of Q9(300)+900 Q3's improvement at *equal total* 1200-step budget. M1 adds evidence that Q9 preparation itself affects downstream trainability; it does **not** settle optimal resource allocation between Q9 and Q3, because total steps differ.

M1 also records **actual final ternary codes by position**, previously unavailable from C1. Original-Q3-scale projected Q9 codes at steps250 and300 disagree at **6,245,131 of 314,572,800 positions (1.9853%)**. After shared Q3 continuation:
- P250 final codes match the step300 Q9-prep code on **44.16%** of initially differing positions.
- P300 final codes retain the step300 prep code on **59.30%** of those positions.
- The final two models agree on **60.00%** of those initially differing positions and **93.858%** across all targeted weights.

These observations show **partial convergence toward later Q9 choices and substantial reorganization/remaining disagreement**, not a proof that specific early weights are disproportionately important or that one late-prep code set is universally correct. The late Q9 code set is itself not preserved at many positions by its own Q3 continuation. The two different starting geometries may also produce differences outside their original 1.985% code-disagreement positions.

**Outstanding threats:** familiar seed 1729, repeatedly viewed 8192-token WikiText heldout, short training horizon, degraded text generation, nonstandard internal QAT baseline, model-family sensitivity and absence of fresh-data confirmation. The highest-value next independent tests are durability versus direct Q3 under a longer fixed recipe (multiple common checkpoints) and genuinely new evaluation data. No further GPU run launched.

M1 records: [full raw JSON](results/run_m1_equal_q3_continuation_seed1729_2026-10-09.json), [detailed technical interpretation](results/run_m1_equal_q3_continuation_seed1729_summary.md), [frozen preregistration](research_log/m1_equal_q3_continuation_seed1729_prereg_2026-10-08.md), [completed HF job](https://huggingface.co/jobs/codeflash85/6ac86712fee2c900701734bb).

## F1 — FineWeb-Edu pretraining-style QAT data: paired Q9→Q3 gain survives (October 9, 2026 UTC)

The user proposed evaluating our staged quantization method with data closer to SmolLM2's original pretraining distribution. We tested **FineWeb-Edu sample-10BT**, a public subset of **one ingredient** in SmolLM2 pretraining, without claiming to reconstruct the original multi-dataset pretraining or Instruct tuning. **F1 is an exploratory one-seed test of QAT-data distribution transfer, not cross-architecture transfer or full pretraining-data replication.**

One paired job compared direct Q3 all1200 training opportunities with Q9 for300 then Q3 for900. Both used the same BF16-rounded SmolLM2-360M-Instruct source, trainable target linear master weights/rowwise scales, frozen other modules, fixed teacher, 35% CE+65% KL, AdamW and same v10 global LR schedule. Training chunks were fixed and paired between treatments.

| Final evaluation | Direct Q3 NLL ↓ | Staged Q9→Q3 NLL ↓ | Advantage D−S ↑ | Direct PPL | Staged PPL |
|---|---:|---:|---:|---:|---:|
| FineWeb-Edu doc-heldout (primary) | 5.766660 | **4.996255** | **+0.770405** | 319.469 | **147.858** |
| WikiText2 validation (secondary) | 6.558239 | **5.695225** | **+0.863013** | 705.029 | **297.444** |

**HF job [6ac86eb0fee2c900701738dd](https://huggingface.co/jobs/codeflash85/6ac86eb0fee2c900701738dd) COMPLETED on 2026-10-09 04:48:30 UTC**, scientific `valid_for_science=true` with **5/5 design checks passed**. Pinned source commit `64dcd7f20240b4c62e6ecac8df70a1336a57bb14`. D actual training updates1194 (6 skipped), S1192 (8 skipped), from 1200 scheduled each.

**Evaluation split caution:** FineWeb sample used first 340 streamed docs, with hash-bucket document-disjoint sets: **180 QAT train docs** supplying1200 chunks, **3 train-dev docs** supplying24 chunks, **21 heldout docs** supplying128 chunks. The QAT train and heldout data did not share source document IDs, but the pretrained language model **may have already encountered some sampled FineWeb documents during pretraining**. Token chunks within a document are correlated. This is a positive signal on a small source-distribution sample, not comprehensive evidence on independently unseen pretraining content. WikiText2 validation data were not used to choose F1 hyperparameters and were distinct from prior WikiText test samples.

The stage transition incurs a large one-step Q3 projection shock on the tiny FineWeb train-split dev set (native Q9 NLL5.428179; initial-scale Q3 NLL6.942240); staged Q3 then recovers strongly. This is consistent with the earlier optimization-path hypothesis, but does **not** prove code assignment causality, model quality after generation or permanence of the gain at long horizons.

**L1 longer-horizon WikiText job is separate** ([job 6ac86e1f095c578089301e53](https://huggingface.co/jobs/codeflash85/6ac86e1f095c578089301e53)) and was still running during F1 archival. Results here neither imply nor predict whether direct Q3 catches up by6000 steps.

[F1 full result JSON](results/run_f1_fineweb_edu_direct_vs_staged_seed1729_2026-10-09.json) · [F1 detailed scientific summary](results/run_f1_fineweb_edu_direct_vs_staged_seed1729_summary.md) · [frozen preregistration](research_log/f1_fineweb_edu_qat_seed1729_prereg_2026-10-09.md) · [live state](CURRENT_STATE.md). No additional GPU jobs were launched to analyze F1.

## Latest completed studies and external mechanism-review follow-up (2026-10-09 UTC)

**All F1/F2/F3 FineWeb seeds and L1 WikiText6000 job have COMPLETED**, all final scientific validity flags passed. No research jobs currently active in the last verified HF listing. **FineWeb aggregate:** Q9→Q3 beats direct Q3 in **3/3 orders**, mean held-out advantage **+0.763898 nats/token** (range +0.742353..+0.778936), WikiText validation mean +0.801917. Identical 21 FineWeb evaluation documents across orders, *not independent validation samples*. [3-seed aggregate](results/run_f1_f3_fineweb_edu_three_seed_aggregate_summary.md) · [F2 raw](results/run_f2_fineweb_edu_direct_vs_staged_seed271828_2026-10-09.json) · [F3 raw](results/run_f3_fineweb_edu_direct_vs_staged_seed424242_2026-10-09.json).

**L1 six-thousand-step:** [result summary](results/run_l1_6000step_wikitext_durability_seed1729_summary.md) · [raw](results/run_l1_6000step_wikitext_durability_seed1729_2026-10-09.json). On fixed WikiText validation, staged advantage declines **+0.716087 @1200 → +0.616040 nats/token @6000** (nonmonotonic), so direct Q3 hasn't caught up by6000. Reproduced both step1200 historical tests exactly. This is one long training order, not a convergence proof.

**Separate independent review of unresolved mechanism controls:** [frozen-scale depth / Q9 range match / FineWeb sham switch / margin-matched positions / common-scale code comparisons / document-linked evidence](research_log/2026-10-09_external_technical_review_scale_range_sham_controls.md). Reviewing `smollm2_v12_code_identity_position.py` confirms scale-gradient confounding is real; v13's nearly-equal fixed/learned alpha *measurement* of code survival does not eliminate a scale-*training-gradient* mechanism. Next recommended controls **G1 frozen-vs-learned-scale depth** and **S1 FineWeb direct-Q3 sham step300 reset**, followed by range-matched Q9. No control GPU jobs launched as part of this review. [Authoritative current state](CURRENT_STATE.md) takes precedence over historical “running” sections.

---

## G1/S1 COMPLETE — controls on scale gradients and sham resets (2026-10-09 UTC)

G1 HF [6ac96a7efee2c9007017ea64](https://huggingface.co/jobs/codeflash85/6ac96a7efee2c9007017ea64) COMPLETED, 6/6 checks, pinned source SHA `059a0bb91aa55fea99ed56b1d0d980eb21fd31ae`. Final WikiText heldout NLL: d0.03 learned **5.494818**, d0.50 learned **4.884970**, shallow-minus-deep gain **+0.609848**; d0.03 frozen **5.516548**, d0.50 frozen **4.896080**, gain **+0.620468**; frozen-minus-learned depth interaction **+0.010620**. Identical initial Q3 codes/scores; raw scales frozen exactly and excluded from optimizer, all 900 opportunities (898 effective/2 AMP skips). **Depth effect persists without row-scale learning**, supporting interior assignment placement and survival rather than requiring scale-gradient learning. Does not isolate Q9's wider range or fully prove mediation.

S1 HF [6ac96a82fee2c9007017ea6b](https://huggingface.co/jobs/codeflash85/6ac96a82fee2c9007017ea6b) COMPLETED, 9/9 checks, pinned SHA `4cdab8ee73ddd804e96fac43992f91ab9a54c1b9`. On matched seed1729 FineWeb, D Q3 continuous loss **5.766660**, D Q3 with original-scale+Adam/GradScaler sham reset@300 **5.789199** (sham worse **0.022539**), historical same-seed Q9→Q3 **4.996255** (**0.792944** better than sham). WikiText validation likewise sham 6.560840, continuous6.558239, staged5.695225. Anchor D reproduces original F1 exactly. **Reset alone is insufficient** to explain staged FineWeb benefit. One seed, same 21-doc heldout sample.

**Raw evidence:** [G1 JSON](results/run_g1_depth_by_scale_freeze_seed1729_2026-10-09.json) · [G1 summary](results/run_g1_depth_by_scale_freeze_seed1729_summary.md) · [S1 JSON](results/run_s1_fineweb_q3_sham_reset_seed1729_2026-10-09.json) · [S1 summary](results/run_s1_fineweb_q3_sham_reset_seed1729_summary.md). Original frozen protocol/job handoff [G1_S1_ACTIVE_JOB_PLAN.md](G1_S1_ACTIVE_JOB_PLAN.md).

**Next:** strongest unrun mechanistic confound is Q9's broader output range (range-matched nine levels versus original Q9, correct STE/clipping); boundary-margin matched controls and truly new evaluation docs follow. No R1 or other paid follow-up job automatically launched. [CURRENT_STATE.md](CURRENT_STATE.md) is authoritative; old “active/submitted” headings elsewhere are historical.

---

## R1 COMPLETED — range-matched nine-state Q9 nearly erases staging gain (2026-10-10 UTC)

**Valid scientific R1**: [HF job](https://huggingface.co/jobs/codeflash85/6ac97c30095c57808930b904), finished **2026-10-10 00:05:57 UTC**, source SHA `f28f1d4814ab0a61f2b529e2add007ce142972d5`. **12/12 technical/reproduction checks passed**, including *exact* archived FineWeb F1 direct-Q3 and original wide Q9→Q3 heldout NLL reproductions. [R1 full raw JSON](results/run_r1_q9_range_matched_fineweb_seed1729_2026-10-10.json) · [R1 detailed summary](results/run_r1_q9_range_matched_fineweb_seed1729_summary.md) · [original prereg](research_log/r1_range_matched_q9_fineweb_seed1729_prereg_2026-10-09.md) · [R1 handoff](R1_ACTIVE_JOB_PLAN.md).

**R1 question:** is having nine intermediate levels enough, or is the originally wider Q9 representation (max ±α) important compared with Q3's max ±2α/3? A new nine-state range-matched Q9 uses `q=α*clamp(round(6*clip(w/α,-.99,.99)),-4,4)/6`, preserving nine codes, normalized `z` outer clip and surrogate gradient formula, but deliberately compressing output range to ±2α/3. All arms use the same seed1729 FineWeb `sample-10BT`, original model source, 300 Q9 +900 Q3 continuation/switch, and fixed heldout evaluation.

| Endpoint | Direct Q3 | Original wide Q9→Q3 | Range-matched Q9→Q3 |
|---|---:|---:|---:|
| FineWeb doc-heldout NLL ↓ | 5.766660 | **4.996255** | 5.751036 |
| WikiText validation NLL ↓ | 6.558239 | **5.695225** | 6.566215 |

FineWeb wide-Q9 gain `D−wide` **+0.770405 nats/token**; range-matched gain `D−range` **+0.015624** (only **2.03%** of original). Range-matched was 0.0080 nats **worse than direct** on WikiText validation; original wide gain there **+0.863013**.

**Crucial confound still present:** A uniformly spaced nine-state grid constrained to Q3's range also changes quantization cell width, internal thresholds and code saturation. At Q9 preparation300 the proportion of extreme code ±4 was **47.83%** range-matched vs **29.30%** wide; **36.45%** of range-matched would-be raw round(6z) codes exceed ±4 and are clipped. We can state **the wider output range and/or its coupled code geometry is necessary for *this particular range-matched implementation* to retain the effect**, not that range alone caused it or any narrow-range nine-level method must fail. The new S_range already has worse Q9-native train-dev loss, not merely larger switch shock.

**Scope:** single familiar seed1729 and 21 small FineWeb heldout source documents, pretrained-model document exposure unknown, no restored model checkpoint, no production-generation claim. Previous G1 frozen-scale and S1 sham-reset controls passed and support a meaningful optimization-path/assignment-stability story but do not render R1 a pure isolation of range. **No additional GPU job was launched after R1.** [CURRENT_STATE.md](CURRENT_STATE.md) is authoritative; older “R1 submitted” sections are historical.

---

