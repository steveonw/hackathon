# R7 post-hoc scoring and repetition audit — 2026-10-10

**No model inference rerun or GPU used.** Retrieved both original Hugging Face job logs with `hf_jobs logs tail:500`, obtaining **all 120 individual records per family** (contrary to initial default log tail of 20). Jobs: Granite `6acad549095c578089318d57`, Smol `6acad569fee2c9007018f35a`. Original R7 inference script revision `c6a24b3c5ba12f97634bc65312cb8161fba1ebca` and preregistered 40-prompt file `7956c5ee8a067efbab7a34ecca84dd287cf5cab9`, prompt SHA256 `5180b1319cca0b3b65ffa01f0940d599747c9f9169bfde7d8b14d0f1dae6008b`. Re-evaluated only the frozen *greedy* responses (10 each per arm/family/category), retaining strict exact match and a post-hoc corrected permissive match.

## Corrected score table

| Family | Category | Base strict | Base corrected permissive | H strict | H corrected permissive |
|---|---|---:|---:|---:|---:|
| Granite | Arithmetic | 10/10 | 10/10 | 0/10 | 0/10 |
| Granite | Instructions | 9/10 | 9/10 | 0/10 | 0/10 |
| Granite | Extraction | 8/10 | 10/10 | 0/10 | 0/10 |
| SmolLM2 | Arithmetic | 0/10 | **10/10** | 0/10 | 0/10 |
| SmolLM2 | Instructions | 4/10 | 6/10 | 0/10 | 0/10 |
| SmolLM2 | Extraction | 6/10 | 8/10 | 0/10 | 0/10 |

**Total corrected permissive hits out of 30:** Granite base **29**, H **0**; Smol base **24**, H **0**. Thus the exact-format scoring issue did *not* hide significant hybrid success on this fixed battery.

### Specific scoring bug

R7 original `score` function uses `re.search(r"(?<!\\w)\\d+(?!\\w)",output)`, which searches for literal backslash sequences, not numeric tokens, so the original saved `permissive` field was **false even for base outputs `63`, `85`, etc.** Original aggregate final emitted only strict counts. Corrected arithmetic audit enumerated integer tokens with `/(?<!\\w)[+-]?\\d+(?!\\w)/g` and checked that the expected integer occurred. This permissive method **may give credit to a number embedded in unrelated text** and should never be interpreted as strict instruction compliance; here no H output contained its target answer. E.g. Smol base `24 + 39 = 63` correctly receives permissive credit but failed its prompt's `only the number` format, so strict 0/10 stands.

## Continuation repetition

Mean unique bigram ratio over 10 greedy continuation cases (higher indicates less repetition; not an independent semantic-quality metric):

| Family | Base | R6 H ternary |
|---|---:|---:|
| Granite | **0.684600** | **0.100000** |
| SmolLM2 | **0.912698** | **0.393651** |

Examples: Granite H repeated `the city of the city` or `tree tree tree`; Smol H repeated a building description, or newlines followed by `1888 – 1888 – 1988`. Original base continuations were much more varied.

## Scientific implications and next action

The corrected audit **strengthens**, rather than weakens, the conclusion that both R6 H ternary checkpoints have severe *functional* degradation relative to their high-precision bases, despite the independently validated relative NLL gains over direct/wide/narrow Q3 QAT. The original preannounced *strict* endpoint was still valid as strict output compliance; the permissive arithmetic implementation was broken and is transparently corrected here. Do not change this frozen benchmark or tune on its prompts. For a future R7 training run, add a truly independent holdout and fix scoring in the preregistered evaluator. Before proposing more costly GPU experiments, diagnose teacher/student language retention during Q9 and Q3 transitions or compare strong compressed baselines; this audit alone does not isolate whether degradation is due to ternary precision, training budget, model size, or choice of loss/objective.
