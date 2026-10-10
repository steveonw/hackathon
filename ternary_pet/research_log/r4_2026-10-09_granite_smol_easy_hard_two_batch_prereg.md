# R4 prereg — four-job Granite/Smol easy-hard matrix, two GPU jobs per user prompt

**Frozen before any new scientific GPU submission, 2026-10-09 EDT (UTC Oct10).** User explicitly authorized **four total jobs split across two prompts**: first prompt **two Granite jobs** (historically hard and easy order); next prompt **two SmolLM2 jobs** using the same two seeds. **Do not start second batch during this prompt, or automatically run retries, extra seeds or follow-on jobs.** User will ask in a later turn once first pair finishes.

## Why and seed selection (pre-existing observations)

R3 on SmolLM2 seed1729 found nonuniform hybrid H, with the **narrow grid's seven interior outputs and original wide ±α extreme values**, retained 96.6% of original wide Q9 advantage on 32 new-to-QAT FineWeb docs. We need cross-model/order falsification. Existing Granite results under its *own, direct-Q3-calibrated constant LR1e-4* were mixed: historical **hard seed271828** D heldout WikiText NLL **5.726725168526173**, S original Q9→Q3 **~5.84103 (worse)**; historical **easy/positive seed424242** D **5.801920056343079**, S **~5.74057 (better)**. These are **previously selected known outcomes**, not random or blind replication seeds.

**Batch A, this prompt (exactly two jobs):** `ibm-granite/granite-4.0-350m` seed271828 (hard) and seed424242 (easy), one A10G-small job **per seed**, maximum90m each (increase only with separate approval). Four independently reset training trajectories per job, so within-seed paired arm comparisons are valid. Hard/easy seed label refers ONLY to Granite's historical *full staging* outcome, not problem difficulty or model difficulty.

**Batch B, NEXT user prompt (NOT authorized to launch now):** `HuggingFaceTB/SmolLM2-360M-Instruct`, seeds271828 and424242, matched four D/W/N/H arms per seed, WikiText2 training/test/validation with the **model's existing tuned global warmup100/cosine schedule** (1e-3 peak, 1e-4 floor), original Smol FP16 teacher/autocast and GradScaler, and original Q9 switch resets; do not silently transfer Granite constant1e-4 to Smol. New prereg execution scripts to be pinned and CPU QA'd in the *next* turn before those two jobs. This matrix tests **qualitative transfer** under each family’s already established recipe, NOT an architecture-isolated causal effect of family difference because LR/autocast implementations differ.

## Batch A source / precision / data

Model `ibm-granite/granite-4.0-350m` pinned HF revision **`bd8a1497065c0d6ba1ef19af6b0d2b14bacf71c2`**, dataset `Salesforce/wikitext` `wikitext-2-raw-v1` pinned revision **`b08601e04326c79dfdd32d625aee71d232d685c3`**, resolved in a pre-run HF CPU metadata job `6ac9a397fee2c90070180b8d`. Same tokenizer, original fixed WikiText2 train1200×129 chunks + subsequent train-split dev24×129 chunks; shuffle the **1200 train** with seed271828 or424242. BF16-rounded original source, FP32 master weights, per-row learned softplus scales; nonquantized modules frozen; BF16 teacher and student autocast; CE35% + teacher KL65%, AdamW betas (0.9,0.95), wd0, grad clip1. **No FP16 or GradScaler** on Granite because previous FP16 failed; Granite reference constant learning rate `1e-4` all1200 steps.

**Training arms per seed (exactly four)**:
- `D`: direct original ternary Q3 all1200; a scale/Adam reset at300 to match existing historical Granite baseline protocol.
- `W`: original wide nine-state Q9 prep300 with outputs `{0,±α/4,±α/2,±3α/4,±α}`, then original Q3 900.
- `N`: original-threshold nine-state narrow Q9 prep300 with outputs `{0,±α/6,±α/3,±α/2,±2α/3}`, then Q3 900.
- `H`: nonuniform hybrid Q9 prep300 with outputs `{0,±α/6,±α/3,±α/2,±α}`, then Q3 900.

All preparatory quantizers use same integer decision `k=round(4*clamp(w/α,-0.99,0.99))`; H and N identical at k=0,±1,±2,±3; H differs only at extreme code ±4 reconstructed values. Same unclipped-z STE surrogate for Q9 and Q3. **Every arm** restores original Q3 source row scales, reinitializes student with its own captured FP32 masters, and resets optimizer precisely at step300, matching Granite G1 mechanism continuation semantics. Same first300 and final900 training chunks/order and constant LR per seed. Every final model is original Q3; 249,561,088 target linears expected.

## Evaluation and outcomes

**Primary within-seed Q9 hybrid generalization:** use **WikiText2 validation** first **64 nonoverlapping 129-token chunks / 8192 next-token targets** (not Granite's repeatedly consulted test slice); evaluate each arm with this identical *non-training* corpus, ideally save per-chunk CE sums/losses for descriptives. No tuning on this set. Primary signed `N−H`, `H−W`, `D−H`, `D−W` NLL differences. Report both hard and easy seeds separately; do not hide negative seed.

**Secondary and anchors:** WikiText2 historical **test first64 chunks / 8192 targets** and train-split dev24 chunks (old scientific endpoints). Historical D must reproduce seed271828 5.726725168526173, seed424242 5.801920056343079 within ±0.04 nats; historical W must reproduce 5.84103 and5.74057 respectively within ±0.06 nats; failure makes new experiment non-comparable, archived as technical check failure. Predeclare broad tolerances to detect gross data drift; do not retune. Compare H and N effects only after verifying anchors. Validation group previously not used to select Granite LR, but cannot guarantee dataset absent from pretrained model pretraining; already-known research model/order recipes are exploratory.

**Primary evidence:** each fresh-validation chunk CE sum/tokens, paired per-chunk signs/descriptives; not independent documents. Secondary held-out WikiText test and PPL. Also each arm’s step300 native Q9 dev NLL, projected Q3 dev NLL, outer ±4 code usage, final Q3 code histogram/flip fraction, matched training100/300/600/900/1200 summaries, optimizer update counts and BF16 finite checks. Include resolved model/dataset SHAs and package versions.

## Technical gate and resource budget

- Prereg BEFORE committing final executable scripts; run CPU static/quantizer toy audit BEFORE paid GPU. Ensure all nine integer codes reachable, W/N/H initial code identity at same w/alpha, H/N center equal with extreme ±4 differing by α/3, gradients correct for FP32 master and row-scale STE including clipping, original Q3 after switch, and numeric source revision pins. CPU QA on pinned two scripts should pass (scientific GPU may still fail; archive failure).
- **Exactly two A10G-small jobs** for Granite hard/easy in this turn, **90m timeout each**. GPU execution from immutable repo SHA, not moving main. Never launch Smol second-batch jobs automatically.
- If any job fails protocol/reproduction checks, preserve terminal FINAL_JSON/raw logs and mark invalid for scientific interpretation; never hide null or negative results, re-run, alter thresholds or pick convenient seeds. After completion update reports, checks and honest limitations; user can prompt later to request archive and next two.

**Not claimed:** original Granite staged advantage is robust (it was mixed), a mechanism universally works across architectures, codebook output level change isolates scale derivatives or master/code survival, or per-chunk scores are statistically independent document replicates.
