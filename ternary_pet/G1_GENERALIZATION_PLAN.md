# G1 — Cross-family small-model generalization plan

Status: **preregistered roadmap; no G1 scientific result yet**

This begins a new phase after the SmolLM2-360M / WikiText-2 mechanism stop
condition. It is intentionally **not v14**. The question is now which parts of
the established effect generalize across model families and later across scale.

## Generalization matrix

| Family | Small cell | Large cell |
|---|---|---|
| A — SmolLM2 | `HuggingFaceTB/SmolLM2-360M-Instruct` — established | `HuggingFaceTB/SmolLM2-1.7B-Instruct` — future G2 |
| B — Granite 4.0 | `ibm-granite/granite-4.0-350m` — **G1** | larger Granite 4.0 dense sibling — future G3 |

The Family-B choice remains provisional until G1-0 confirms that the established
quantization semantics can be ported cleanly.

Intended sequence:
1. Family-B small (G1)
2. SmolLM2 large (G2)
3. Family-B large (G3)

A failure in one quadrant does not automatically cancel the others.

## G1-0 — architecture / quantization smoke test

Purpose: establish that the Smol protocol has a faithful Granite implementation
before spending on a scientific comparison.

Required audit:
- exact model identifier and loaded model class;
- total parameter count;
- target module classes and names;
- number of quantized tensors and quantized parameters;
- excluded tensors;
- whether embeddings and `lm_head` are tied or share storage;
- initial Q3 zero fraction and code histogram;
- initial Q9 code histogram;
- source held-out loss / PPL;
- one Q3 forward/backward step;
- one Q9 forward/backward step;
- peak GPU memory;
- confirmation that non-quantized parameters remain frozen.

No scientific conclusion may be drawn from this smoke test.

**Stop rule:** if "quantize all linear weights except `lm_head`" is not a
comparable intervention on Granite, stop and preregister an architecture-specific
target mapping before proceeding.

## G1-1 — direct-Q3 schedule calibration

Seed/order: 1729.

Tune **direct Q3 only** using validation / LR-selection chunks. Q9 must not be
used to choose the schedule.

Initial candidates:
1. constant LR 1e-4;
2. 100-step warmup to 3e-4, cosine to 1e-4;
3. 100-step warmup to 1e-3, cosine to 1e-4.

A short validation-only screen chooses one schedule. Once selected, that global
schedule is frozen for the rest of G1 and is given unchanged to Q9 -> Q3.

## G1-2 — Family-B staging gate

Seed/order: 1729 first.

Arms:
- **D:** direct Q3 for 1200 updates;
- **S:** Q9 for global steps 1-300, then Q3 for steps 301-1200.

Keep the established semantics wherever G1-0 confirms they transfer:
- common source checkpoint / dtype treatment;
- persistent FP32 masters;
- rowwise learnable quantizer scales;
- non-quantized parameters frozen;
- CE35 + teacher-KL65 objective;
- same 1200 chunks in the same shuffled order;
- same fixed held-out evaluator;
- same global LR schedule selected by G1-1;
- at Q9 -> Q3: restore original Q3 scales, fresh Adam, keep masters, continue
  the global LR curve without restarting it.

Required step-300 diagnostics:
- D native Q3 quality;
- S native Q9 quality;
- D and S projected through the same original Q3 scales on the same data;
- D/S projected-Q3 Hamming;
- code movement vs source;
- disagreement-mask size and transition counts.

Required final held-out metrics:
- loss;
- PPL;
- teacher top-1 agreement;
- KL to teacher;
- fixed generation probes as descriptive-only output.

### G1-2 interpretation

The strongest replication of the Smol phenomenon is:
1. Q9 is equal or worse as an immediate fixed-Q3 checkpoint after equal
   300-step compute; and
2. Q9 finishes better after the common later Q3 optimization budget.

However, immediate-Q3 inferiority is **not required** for staging itself to
count as a positive cross-family result. If Granite gives a better immediate Q3
entry state and a better final endpoint, report staging generalization with a
different mechanism signature.

**Gate to G1-3:** proceed only if S has a material, technically clean final
held-out advantage over D. Do not retune after inspecting S.

If S is tied or worse, record the negative/mixed result and do not automatically
launch the mechanism battery.

## G1-3 — compressed mechanism test

Run only if G1-2 passes.

Rebuild matched step-300 states:
- D = direct-Q3 preparation;
- S = Q9 preparation;
- M = positions where D and S, projected with the same original Q3 scales,
  choose different Q3 codes.

Continuation arms:
1. **D** — direct masters everywhere;
2. **S** — Q9 masters everywhere;
3. **M-exact** — S masters only on M, D elsewhere;
4. **M-d50** — D outside M; on M use S's selected Q3 code at normalized
   depth d=0.5;
5. **Random-d50** — reproduce M's layer/source->target transition counts at
   different positions, also using d=0.5.

Required equality/construction assertions must pass before continuation.
Where arms are intended to have the same ternary forward model, require zero
projected-Q3 Hamming and matching diagnostics.

Log Q9-selected-code survival for M-d50 after 100, 300, and 900 continuation
steps under current learned Q3 scales and fixed original alpha0.

Questions:
- Does full S still beat D?
- Does M-exact recover most of the S-vs-D advantage?
- Does M-d50 recover a substantial fraction of M-exact?
- Does matched Random-d50 fail or perform much worse than true-M d=0.5?
- Does assignment survival show the same qualitative persistence pattern?

Do **not** require Granite to match Smol's exact 6.3% mask size, 96.4% recovery,
106.9% prototype recovery, or v13 survival percentages. The cross-family target
is the within-model causal pattern.

## G1-4 / G1-5 — confirmatory orders

Run orders 271828 and 424242 only if seed 1729 is scientifically interpretable
and strong enough to justify replication.

Confirmatory runs must lock:
- architecture mapping;
- direct-selected schedule;
- quantizer semantics;
- d=0.5 definition;
- matched-random construction;
- training/evaluation construction;
- continuation semantics.

Only the training-order seed changes.

Before any aggregate conclusion, inspect every order individually and report
heterogeneity plainly.

## Deliberately excluded from the first G1 batch

Do not automatically repeat the full Smol mechanism excavation:
- no FP32 warmup control unless G1-2 creates a reason to reopen it;
- no Adam/scale carryover factorial;
- no signed-preload test;
- no full depth sweep;
- no target-zero / target-nonzero submask intervention;
- no state-count sweep.

Reopen one only if Granite creates a specific contradiction or ambiguity.

## Planned phase progression

If Family-B small is positive, next run **G2: SmolLM2-1.7B** as the
within-family scale test, then the analogous larger Family-B model as G3.

Eventual matrix interpretation:
- both sizes work in both families -> strong generality evidence;
- Smol small+large work but Granite fails -> family/training dependence;
- both small models work but larger models fail -> scale dependence;
- one quadrant differs -> family-by-scale interaction.

This matrix is generalization evidence, not a clean causal factorial for
architecture, because family changes tokenizer, pretraining, post-training, and
implementation details too.

## Documentation protocol

For every G1 launch:
1. preregister in `EXPERIMENT.md`;
2. pin exact code;
3. record HF job ID and hardware;
4. retain failures/cancellations;
5. save raw JSON plus run summary;
6. update `results/README.md`;
7. update replication aggregates when applicable;
8. edit living `RESEARCH_REPORT.md` in place;
9. append chronology to `SHAREABLE_RESEARCH_REPORT.md`;
10. update `AI_HANDOFF.md` and README headline when state changes.

Chat result format:
key held-out table -> 2-3 sentence interpretation -> caveats/order coverage ->
one recommended next step.

This roadmap records intent only; compute launches are separate actions.
