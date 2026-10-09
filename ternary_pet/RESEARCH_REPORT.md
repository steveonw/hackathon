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

- **Next verify cross-model transfer of P, not another Granite tune:**
  The frozen nine 10%-pull setting improved direct Q3 on both selected
  Granite orders (G1-9 and G1-10). A preregistered, equal-budget
  **SmolLM2 direct-Q3 vs P** (or a full four-arm factorial if budget
  allows) would test whether this direct ternary stabilization extends
  beyond Granite, using that family's established schedule rather
  than importing Granite's LR uncritically. Keep independent
  confirmation data unavailable during hyperparameter selection.
  Do not retune Gaussian σ or λ using these held-out results.
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
