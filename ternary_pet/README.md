# Ternary Pet Experiments

> ## 🤖 AI / researcher handoff — START HERE
>
> New AI/researcher: read **[AI_HANDOFF.md](AI_HANDOFF.md)** before changing an
> experiment or launching compute.

Experiments on whether a pretrained language model can enter ternary weight
space more effectively after adapting on an intermediate discrete grid.

## Current headline

Matched-schedule **Q9→Q3 beats tuned direct Q3 in 3/3 training orders**, with a
mean held-out loss advantage of **0.672 nats/token** and **48.94% lower PPL**.
The Q9-prepared state is actually a **worse immediate Q3 checkpoint**, so the
effect is better later trainability rather than better entry quality.

The dominant benefit localizes to the **~6.3% of weights** where Q9 and direct
choose different Q3 codes: transferring only those choices recovers **96.4%**
of the full gain, and standardized interior placement recovers **106.9%** across
3/3 orders while transition-matched random positions are harmful.

A seed-1729 depth sweep further shows that the placement effect is
**threshold-like and saturating**: about (d=0.25) recovers ~95% and
(d=0.5) reaches the plateau, with Q9-code survival rising to ~97%.
This v13 depth refinement is **one order only**; the canonical mechanism result
remains the three-order v12 replication.

**Living current-state report:** [RESEARCH_REPORT.md](RESEARCH_REPORT.md)  
**Append-only chronological research log:** [SHAREABLE_RESEARCH_REPORT.md](SHAREABLE_RESEARCH_REPORT.md)

## Current interpretation

Q9 does not simply preserve more information or create a better ternary model at
the switch. It discovers a small, specific set of useful position/code
assignments; those assignments need enough interior margin to survive later Q3
optimization. Exact Q9 continuous values are unnecessary, and generic
prototype snapping does not help direct Q3.

The mechanism phase on SmolLM2-360M / WikiText-2 has reached its stop condition.
The next phase is **cross-model generalization**, followed by scale-up only if
the cheaper second-family test justifies it.

## Evidence hierarchy

- **Replicated across 3/3 orders:** matched-schedule staging advantage (v10),
  disagreement-mask localization (v11), code-choice/interior-placement result
  and matched-random failure (v12).
- **One-order supporting mechanism:** master-weight carryover (v6), v13 depth
  saturation and firmness controls.
- **Important negative result:** generic FP32 warm-up does not reproduce Q9,
  but it is **not worse than direct in every order**; order 271828 gives FP32 a
  small 0.0292-nat improvement over direct.
- **Scope:** one 360M model and WikiText-2 so far; free-running generation remains
  poor, and longer-run asymptotics are unresolved.

## Repository map

- [RESEARCH_REPORT.md](RESEARCH_REPORT.md) — **living current state**; edit in
  place after each experiment
- [SHAREABLE_RESEARCH_REPORT.md](SHAREABLE_RESEARCH_REPORT.md) — append-only
  chronological narrative
- [AI_HANDOFF.md](AI_HANDOFF.md) — operational handoff, current protocol and
  claim boundaries
- [EXPERIMENT.md](EXPERIMENT.md) — preregistrations and chronological
  experiment record
- `results/` — raw JSON and per-run summaries
- `replications/` — replication scripts and aggregate summaries
- `smollm2_v*.py` — experiment scripts

## Research workflow

`preregister → pin code → launch job → record job ID → save raw JSON → write run summary → update aggregates → update RESEARCH_REPORT.md → update AI_HANDOFF.md / README as needed`

Before making a replicated conclusion, inspect **every seed/order individually**,
not only the mean, and say plainly when a result is mixed.
