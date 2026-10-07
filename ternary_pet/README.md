# Ternary Pet Experiments

> ## 🤖 AI / researcher handoff — START HERE
>
> New AI/researcher: read **[AI_HANDOFF.md](AI_HANDOFF.md)** before changing an
> experiment or launching compute.

Experiments on whether a pretrained language model can enter ternary weight
space more gracefully through an intermediate representation.

## Current headline

**v7 shows that Q9's advantage is a trainability effect, not a better immediate
ternary checkpoint.**

After equal 300-step preparation and projection through the same original Q3
scales:

- direct Q3@300 diagnostic loss: **6.559**
- Q9-prepared Q3@300: **9.226**
- FP32-prepared Q3@300: **14.735**

So Q9 is actually a *worse* immediate Q3 model than direct after equal compute.

But after all three receive the same 900-step Q3 continuation with fresh Adam
and original scales:

| Path | Final loss ↓ | PPL ↓ |
|---|---:|---:|
| Q3 -> Q3 | 5.872 | 355.09 |
| **Q9 -> Q3** | **5.194** | **180.15** |
| FP32 -> Q3 | 6.031 | 416.19 |

Q9 finishes **0.679 nats/token better** than direct and **49.27% lower PPL**.
FP32 warmup is worse than direct.

Q3 and Q9 change nearly the same number of future ternary codes by step 300
(~0.69% vs ~0.68%), but their changed-position sets overlap weakly
(**19.64% Jaccard**). Global distance-to-threshold distributions are almost
identical.

## Current interpretation

> Q9's forward constraint appears to create a different **weight-selection /
> optimization geometry**: it repositions a different subset of FP32 masters.
> The resulting immediate Q3 model is poor, yet those masters are much more
> trainable during later Q3 optimization.

This is stronger evidence that the intermediate discrete representation itself
matters; a generic FP32 warmup does not reproduce the effect.

v5 still shows that direct Q3 catches up substantially with much more training,
so no asymptotic superiority has been established.

## Repository layout

- `AI_HANDOFF.md` — start here
- `EXPERIMENT.md` — chronological protocol/outcomes
- `SHAREABLE_RESEARCH_REPORT.md` — external-review narrative
- `smollm2_v7_equal_compute_geometry.py` — current completed v7 script
- `results/run_v7_summary.md` — v7 interpretation
- `results/run_v7_2026-10-07.json` — complete v7 machine-readable result
- `replications/` — v4b confirmation
- `results/` — all run records
