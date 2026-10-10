# R6 controlled generation — Smol completed, coherent continuation but instruction following degraded

**2026-10-10 UTC.** [HF CPU job 6ac9f61ffee2c90070184f2b](https://huggingface.co/jobs/codeflash85/6ac9f61ffee2c90070184f2b) **COMPLETED** with `R6_GENERATION_CONTROL_COMPLETE` and 12/12 output cases. Corrected source SHA `38954d695e5f9f9a72dcd8fe19055edcea921704`. This is a CPU-only comparison between the pinned higher-precision SmolLM2-360M-Instruct base and R6 H ternary inference snapshot. Source logs were accessed directly through the connected Hugging Face app.

| Task / decoder | Original base output | R6 H ternary output |
|---|---|---|
| Plain scientist/notebook continuation, greedy | ` down the date, the location, and the name of the person who had been missing. She then began to search the` | ` that the scientist was not aware of the fact that the scientists had not noticed.\n\n\n\n\n\n \n\n` |
| Plain continuation, sample (temperature 0.8, top-p 0.9) | ` down a new research question.\n\nThe scientist opened the notebook and wrote down a new research question. The new research` | ` a letter to the researcher, Dr. Douglas , on 23 November 2011 . The two parties` |
| Chat-template baseline explanation, greedy | `A baseline is useful in an experiment as it provides a reference point for comparison, allowing researchers to accurately measure changes or effects` | `\n\n\n\n\n\n\n\n\n\n\n\n\n ( 2008 ) is a long bar` |
| Chat-template baseline explanation, sample | `A baseline serves as a reference point to compare results and determine if the changes observed in the experiment are statistically significant.` | ` Slayer is one of the most prolific experimental performances of a variety of heavy rock rock and heavy rock bands . \n` |
| Chat-template 17 + 28, greedy | `17 + 28 = 45` | `\n\n\n\n ( 27 – 2008 ) is a former member of the company 's` |
| Chat-template 17 + 28, sample | `17 + 28 = 45.` | ` 13: 170, , 1934 \n` |

All 12 calls reported finite next-token logits; both models used same 24-token generation cap, same formatting per task, and identical sample seed.

**Interpretation:** This is genuine limited success: Smol R6 hybrid can generate coherent **plain continuations** (albeit with signs of degeneracy), disproving a blanket statement that ternary inference produces only punctuation. However, proper-chat-template instruction following **failed** on both examples even though the pretrained base passed. This points to large instruction-following degradation in this ternary training setup and is consistent with the independently verified R6 H loss/weight reconstruction parity. It does not establish a universal failure for all prompts or isolate precision from other training choices. Prior Granite full controls [HF job 6ac9ee4bfee2c90070184acf](https://huggingface.co/jobs/codeflash85/6ac9ee4bfee2c90070184acf) found similarly degraded, repetitive H output versus original base on some prompts. Future focus: instruction-aware diagnostics, benchmark retention and distillation / staged transition mechanisms before any deployment claim. No new GPUs launched.
