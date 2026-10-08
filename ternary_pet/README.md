# Ternary Pet Experiments

> ## 🤖 AI / researcher handoff — START HERE
>
> New AI/researcher: read **[AI_HANDOFF.md](AI_HANDOFF.md)** before changing an
> experiment or launching compute.

Experiments on whether a pretrained language model can enter ternary weight
space more effectively after adapting on an intermediate discrete grid.

## Current headline

Matched-schedule **Q9→Q3 beats tuned direct Q3 in 3/3 training orders**: mean held-out loss improves by **0.672 nats/token** and PPL by **48.94%**.
The benefit localizes to the **~6.3%** of weights where Q9 and direct choose different Q3 codes; that subset recovers **96.4%** of the full gain across 3/3 orders.
A standardized interior placement for those Q9-selected codes recovers **106.9%** across 3/3 orders; v13 further shows the depth effect saturates around **d≈0.5** on seed 1729.
**Living current-state report:** [RESEARCH_REPORT.md](RESEARCH_REPORT.md) · **Chronological log:** [SHAREABLE_RESEARCH_REPORT.md](SHAREABLE_RESEARCH_REPORT.md)

## Current interpretation

Q9 does not simply preserve more information or create a better ternary model at
the switch. It discovers a small, specific set of useful position/code
assignments; those assignments need enough interior margin to survive later Q3
optimization. Exact Q9 continuous values are unnecessary, and generic
prototype snapping does not help direct Q3.

The mechanism phase on SmolLM2-360M / WikiText-2 has reached its stop condition.
**Cross-model generalization is now active on Granite-4.0-350M.** Its BF16
engineering gate passed after FP16 was found numerically invalid, and the
direct-only G1-1 schedule calibration is the current live step.

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
