# Results

Raw machine-readable and summarized results from each remote run belong here.

## Run registry

| Run | Model | Hardware | Status | Purpose |
|---|---|---|---|---|
| v1 / `6ac52475404719ba37661c8b` | SmolLM2-360M-Instruct | T4 small | completed | naive staged PTQ/QAT |
| v2 / `6ac527e5404719ba37661dc9` | SmolLM2-360M-Instruct | T4 small | completed | strict nested ancestry |
| v3 / `6ac52df9404719ba37661fa1` | SmolLM2-360M-Instruct | T4 small | completed | transition schedules |
| v4 / `6ac539b5404719ba376621a2` | SmolLM2-360M-Instruct | T4 small | completed | persistent-shadow 9->3 |
| v4b seed 1729 / `6ac546b5fbc85ba6823b8941` | SmolLM2-360M-Instruct | T4 small | completed | shuffled-order confirmation |
| v4b seed 271828 / `6ac546bafbc85ba6823b8946` | SmolLM2-360M-Instruct | T4 small | completed | shuffled-order confirmation |
| v5 / `6ac56d99fbc85ba6823ba03f` | SmolLM2-360M-Instruct | A10G small | completed | 5x training-volume test |
| v6 / `6ac5816ffbc85ba6823ba8ec` | SmolLM2-360M-Instruct | A10G small | completed | causal transition ablation |
| v7 / `6ac58e76fbc85ba6823bad78` | SmolLM2-360M-Instruct | A10G small | completed | equal-compute Q3/Q9/FP32 geometry |
| v7 rep seed 271828 / `6ac59807fbc85ba6823bb060` | SmolLM2-360M-Instruct | A10G small | completed | v7 training-order replication |
| v7 rep seed 424242 / `6ac59809fbc85ba6823bb063` | SmolLM2-360M-Instruct | A10G small | completed | v7 training-order replication |
| v8 / `6ac5a0b1404719ba37664653` | SmolLM2-360M-Instruct | A10G small | completed | signed pre-loading mechanism diagnostic |
| v9 / `6ac5ad77fbc85ba6823bb71c` | SmolLM2-360M-Instruct | A10G small | completed | direct-Q3 LR/schedule tuning |
| v9 tuned rep seed 271828 / `6ac5b913404719ba37664b21` | SmolLM2-360M-Instruct | A10G small | completed | tuned direct-Q3 confirmation |
| v9 tuned rep seed 424242 / `6ac5b915fbc85ba6823bba5b` | SmolLM2-360M-Instruct | A10G small | completed | tuned direct-Q3 confirmation |
| v10 seed 1729 / `6ac5c252fbc85ba6823bbd6c` | SmolLM2-360M-Instruct | A10G small | completed | schedule-matched Q9 control |
| v10 seed 271828 / `6ac5c254fbc85ba6823bbd6e` | SmolLM2-360M-Instruct | A10G small | completed | schedule-matched Q9 control |
| v10 seed 424242 / `6ac5c256404719ba37664cf1` | SmolLM2-360M-Instruct | A10G small | completed | schedule-matched Q9 control |
| v11 / `6ac5c7befbc85ba6823bbef0` | SmolLM2-360M-Instruct | A10G small | completed | hybrid-master causal localization |
| v11 rep seed 271828 / `6ac638c0c656c912b4ffae8a` | SmolLM2-360M-Instruct | A10G small | completed | hybrid causal replication |
| v11 rep seed 424242 / `6ac638c3f0d78b8017af0d5a` | SmolLM2-360M-Instruct | A10G small | completed | hybrid causal replication |
| v12 / `6ac642a0df2184ac91ac018d` | SmolLM2-360M-Instruct | A10G small | completed | code identity vs continuous position |
| v12 rep seed 271828 / `6ac64ccddf2184ac91ac092d` | SmolLM2-360M-Instruct | A10G small | completed | code/position mechanism confirmation |
| v12 rep seed 424242 / `6ac64cd2df2184ac91ac092f` | SmolLM2-360M-Instruct | A10G small | completed | code/position mechanism confirmation |
| v13 original / `6ac65927e7a0dae8a277c24c` | SmolLM2-360M-Instruct | A10G small | timed out | monolithic depth/firmness launch; technical failure only |
| v13A / `6ac6a265df2184ac91ac410d` | SmolLM2-360M-Instruct | A10G small | completed | depth sweep on Q9-selected mask |
| v13B / `6ac6a272df2184ac91ac412c` | SmolLM2-360M-Instruct | A10G small | completed | direct-firmness controls |
| G1-0 / `6ac6eacfdf2184ac91ac658a` | Granite-4.0-350M | A10G small | technical gate failed | architecture smoke; FP16 source/teacher path non-finite |
| G1-0b / `6ac6eec9df2184ac91ac67ca` | Granite-4.0-350M | A10G small | completed | precision diagnostic; BF16 passed, FP16 failed |
| G1-1 / `6ac6f18ae7a0dae8a2780246` | Granite-4.0-350M | A10G small | completed | direct-Q3 calibration; constant 1e-4 selected |
| G1-2 / `6ac6f487df2184ac91ac693e` | Granite-4.0-350M | A10G small | completed | positive one-order staging gate; Q9→Q3 +0.1309 nat / 12.27% lower PPL |
| G1-3 / `6ac6fe5fdf2184ac91ac6c1f` | Granite-4.0-350M | A10G small | completed | mechanism positive: exact-M 76.2%, d50 118.6%, matched-random harmful |

## Current headline

The canonical SmolLM2-360M / WikiText-2 mechanism result is now v12, replicated
across **3/3 training orders**.

- schedule-matched Q9→Q3 beats tuned direct Q3 by **0.6723 nats/token** on
  average with **48.94% lower PPL** (v10);
- **96.4%** of the full trainability gain localizes to the ~**6.31%** of
  positions where direct and Q9 choose different projected Q3 codes (v11);
- placing Q9's selected codes at standardized Q3 prototypes recovers **106.9%**
  of the exact-Q9 mask gain, while minimal crossing recovers **18.0%** and
  transition-matched random positions are harmful in **3/3 orders** (v12).

v13 is an optional one-order refinement: the placement effect saturates around
**d≈0.5**, with Q9-code survival rising to ~97%, while firmness applied to
direct's own choices does not help.

See `../RESEARCH_REPORT.md` for the living current-state interpretation.

### G1 generalization status

The Family-B small-model phase has started on `ibm-granite/granite-4.0-350m`.
G1-0 found clean target/freezing behavior but exposed a numerical blocker in the
inherited FP16 path: even the unquantized source evaluation was non-finite, and
Q3/Q9 teacher-KL steps produced non-finite loss/gradients. This is a technical
precision failure, **not** a negative Q9-vs-Q3 result.

G1-0b resolved the blocker cleanly: FP32 and BF16 were finite, FP16 was
non-finite, and one-step Q3/Q9 BF16 training had finite losses and gradients.
BF16 is now locked as the Granite compute path before any scientific D-vs-S
result. G1-1 selected **constant 1e-4** by direct-only validation (loss 5.8128
vs 6.0827 and 6.7402 for the two higher-LR schedules). That schedule is frozen.
G1-2 completed positively on order 1729: Q9→Q3 improves held-out loss by
**0.1309 nats/token** and lowers PPL by **12.27%**, despite being **1.925 nats
worse** than direct as an immediate fixed-Q3 checkpoint at step 300. The
D-vs-S projected-Q3 disagreement set is **6.0246%**.

G1-3 is now active, testing whether that ~6% set carries the benefit and whether
true-mask d=0.5 placement beats a per-layer/source→target matched random control.

See:
- `run_g1_0_granite350m_smoke_summary.md`
- `run_g1_0b_granite350m_precision_summary.md`

### v8 mechanism update

v8 reproduced the seed-1729 v7 performance, but the proposed
**Q9-specific signed-preload** explanation failed its matched-set test.

- on Q9's later-flip set, Q9 prep was more aligned than Q3 prep by **0.001455**
  normalized units;
- on Q3's later-flip set, Q3 prep was more aligned than Q9 prep by **0.001691**;
- the reciprocal own-set advantages are similar, with the Q3 advantage slightly
  larger.

This matches the preregistered post-selection pattern rather than a
Q9-specific directional-preloading mechanism.

See `run_v8_summary.md`.

### v9 direct-baseline update

The historical direct-Q3 recipe was undertuned.

On seed 1729, a 100-step warmup plus cosine schedule (peak 1e-3, floor 1e-4)
improves direct-Q3 held-out loss from **5.8747** to **5.5957** and PPL from
**355.90** to **269.27**.

Historical Q9 -> Q3 remains better at **5.1938 loss / 180.15 PPL**.

The tuned schedule closes about **41%** of the original loss gap but leaves a
**0.402-nat/token** residual Q9 advantage. This new direct schedule must be
replicated on seeds 271828 and 424242 before becoming canonical.

See `run_v9_summary.md`.


### v9 tuned-direct replication aggregate

The v9-selected warmup+cosine direct-Q3 schedule was run without further tuning
on orders 271828 and 424242.

Q9 remains ahead of tuned direct in **3/3 orders**:

| Seed | Tuned direct loss | Q9 loss | Q9 advantage |
|---:|---:|---:|---:|
| 1729 | 5.5957 | **5.1938** | **0.4020** |
| 271828 | 5.6228 | **5.2417** | **0.3810** |
| 424242 | 5.6067 | **5.3008** | **0.3059** |

Aggregate:

- mean residual loss advantage: **0.3630 nats/token**;
- mean Q9 PPL reduction: **30.38%**;
- mean top-1 gain: **+2.52 pp**;
- mean KL advantage: **0.3871**;
- direct tuning closes about **47.27%** of the old loss gap on average.

The tuned schedule is now the canonical small-budget direct baseline.

See `../replications/v9_tuned_direct_aggregate_summary.md`.


### v10 equal-schedule Q9 aggregate

The direct-selected warmup+cosine global LR curve was applied unchanged to
Q9 -> Q3 on all three established orders.

| Seed | Tuned direct | Historical Q9 | **Matched-schedule Q9** | Q9 gain vs direct |
|---:|---:|---:|---:|---:|
| 1729 | 5.5957 | 5.1938 | **4.9010** | **0.6947** |
| 271828 | 5.6228 | 5.2417 | **4.9510** | **0.6718** |
| 424242 | 5.6067 | 5.3008 | **4.9563** | **0.6505** |

Aggregate against tuned direct:

- mean loss advantage: **0.6723 nats/token**;
- mean PPL reduction: **48.94%**;
- mean top-1 gain: **+6.82 pp**;
- mean KL advantage: **0.6817**.

At the equal-compute step-300 fixed-Q3 diagnostic, matched Q9 is still worse
than tuned direct by **0.5120 nats/token on average**.

The shared schedule improves Q9 by **0.3094 nats/token on average** versus the
historical constant-1e-4 staged runs.

See `../replications/v10_schedule_matched_q9_aggregate_summary.md`.


### v11 hybrid-master causal localization

Seed/order 1729:

- D-vs-S step-300 projected-Q3 disagreement mask: **6.458%** of weights;
- 00 direct endpoint: **5.6136**;
- 10 Q9-on-mask-only: **4.9329**;
- 01 Q9-on-same-code-only: **5.5020**;
- 11 full Q9: **4.9010**.

The 10 arm recovers **95.5%** of the full 00->11 loss gain. The 01 arm recovers
only a small fraction. Pair-equality assertions passed exactly before
continuation.

Interpretation: on seed 1729, the dominant causal carrier is the Q9-prepared
master state on the code-disagreement positions, not the distributed same-code
majority.

See `run_v11_hybrid_factorial_summary.md`.


### v11 three-order causal aggregate

| Seed | Mask size | L00 | L10 | L01 | L11 | Mask recovery |
|---:|---:|---:|---:|---:|---:|---:|
| 1729 | 6.46% | 5.6136 | **4.9329** | 5.5020 | **4.9010** | **95.5%** |
| 271828 | 6.18% | 5.6524 | **4.9711** | 5.5620 | **4.9510** | **97.1%** |
| 424242 | 6.30% | 5.6234 | **4.9791** | 5.4979 | **4.9563** | **96.6%** |

Mean mask size: **6.31%**.  
Mean mask-only recovery: **96.4%**.

The causal localization replicates in **3/3 orders**.

See `../replications/v11_hybrid_factorial_aggregate_summary.md`.


### v12 seed-1729 code-vs-position intervention

- D: 5.6136
- exact Q9-on-M: 4.9329
- Q3 prototype on M: **4.8884**
- minimal crossing on M: 5.4979
- matched-random reassignment: 5.6793

Prototype recovery relative to exact positive control: **106.5%**.  
Minimal-crossing recovery: **17.0%**.  
Matched-random recovery: **-9.7%**.

Exact/prototype/minimal begin from the same projected Q3 forward model.

See `run_v12_code_identity_position_summary.md`.


### v12 three-order code-vs-position aggregate

| Seed | Direct | Exact Q9 | Prototype | Minimal | Random | Prototype recovery |
|---:|---:|---:|---:|---:|---:|---:|
| 1729 | 5.6136 | 4.9329 | **4.8884** | 5.4979 | 5.6793 | 106.5% |
| 271828 | 5.6524 | 4.9711 | **4.9187** | 5.5406 | 5.7208 | 107.7% |
| 424242 | 5.6234 | 4.9791 | **4.9373** | 5.4911 | 5.7031 | 106.5% |

Mean prototype recovery: **106.9%**.  
Mean minimal recovery: **18.0%**.  
Mean matched-random recovery: **-10.7%**.

Pattern replicates in **3/3 orders**.

See `../replications/v12_code_identity_position_aggregate_summary.md`.


### v13 optional closing experiment — seed 1729

- depth job: `6ac6a265df2184ac91ac410d`
- firmness job: `6ac6a272df2184ac91ac412c`
- summary: `run_v13_depth_firmness_summary.md`

Best discrete depth tested: **d=0.50**, loss **4.8850**.

The transition is sharply saturating:
d=0.03 gives 16.4% recovery, d=0.25 gives 94.8%, and d=0.50 gives 100.5%.

B1/B2 firmness-only controls are both slightly worse than direct.


### Active Granite confirmation jobs

| Job | Seed/order | Status | Purpose |
|---|---:|---|---|
| G1-4 / `6ac70af0df2184ac91ac6ffc` | 271828 | completed | negative staging order; M-d50 improves on S but remains worse than D |
| G1-5 / `6ac70af2df2184ac91ac7000` | 424242 | completed | positive staging/mechanism confirmation |

Shared pinned commit:
`6ed697a0ca5d0d19ed5ebff04b81a2ed0d259230`.

Only the seed/order differs from G1-3.


### Granite three-order conclusion

The frozen confirmations are **mixed**.

- Full Q9 staging beats D on **2/3** orders.
- Q9 is worse as an immediate fixed-Q3 checkpoint on **3/3** orders.
- Mean disagreement mask: **6.35%**.
- True-M d=0.5 beats full S on **3/3** orders.
- Matched-random d=0.5 is harmful versus D on **3/3** orders.
- True-M d=0.5 beats D on **2/3**, failing on seed 271828.

Do not call Granite a clean 3/3 staging replication. The stronger recurring
signal is that Q9-selected position/code commitments at moderate interior depth
are more useful than the full Q9-prepared state itself.

Canonical aggregate:
`../replications/g1_granite350m_mechanism_aggregate_summary.md`.


### Active G1-6 trajectory-alignment development job

| Job | Seed/order | Status | Purpose |
|---|---:|---|---|
| G1-6 / `6ac71a23df2184ac91ac73ca` | 271828 | running/scheduling | test seed-conditioned trajectory alignment of Q9 commitments |

Pinned implementation:
`10110ffba0ce9b1069f015cd32b235958e37f6c9`.

Arms:
D, All-d50, AlignTop50-d50, AlignBottom50-d50.

The top/bottom selector halves are exactly matched within every layer and
D→S transition class.
