# R6 controlled generation: Granite complete; Smol tokenizer API failure and fix

**2026-10-10.** No new GPU training. Both source checksums and R6 training results remain valid.

- **Granite** CPU generation controls [HF 6ac9ee4bfee2c90070184acf](https://huggingface.co/jobs/codeflash85/6ac9ee4bfee2c90070184acf) **COMPLETED**, `R6_GENERATION_CONTROL_COMPLETE` with all 12 base/H × three prompts × greedy/sampling cases. Original base model gave coherent continuation for the scientist/notebook prompt; hybrid repeated `story` and `character`. On explanation/arithmetic prompts the base sometimes stopped immediately with EOS, so those cases are not strong evidence of successful base instruction following. Hybrid generated repetitive `character` or `1.` and sampling remained low quality. Thus Granite's learned Q3 model has a demonstrable generation-quality problem despite H's relative ternary NLL win and checkpoint-vs-training loss parity.
- **Smol** CPU generation controls [HF 6ac9ee51095c578089310104](https://huggingface.co/jobs/codeflash85/6ac9ee51095c578089310104) **ERROR** before comparison due `TypeError: embedding(): argument 'indices' must be Tensor, not BatchEncoding`. The Smol chat template under this tokenizer/version returns a `BatchEncoding` object; original code passed it directly into the language model. This is test code failure, not evidence of snapshot failure.
- [Corrected controlled-generation script](../r6_cpu_generation_base_vs_hybrid.py) pinned SHA `38954d695e5f9f9a72dcd8fe19055edcea921704` now extracts `input_ids` and normalizes to a 2D long tensor for Smol chat-template input. [CPU-only regression](https://huggingface.co/jobs/codeflash85/6ac9f58d095c57808931046c) produced `R6_SMOL_CHAT_TENSOR_REGRESSION_OK [1, 40]`. Smol's full base/H generation comparison has **not yet been re-executed**. Run only Smol, not Granite again, via authenticated local CLI `--secrets HF_TOKEN`.

**Secure Smol retry:**

`hf jobs uv run --flavor cpu-basic --timeout 40m --secrets HF_TOKEN "https://raw.githubusercontent.com/steveonw/hackathon/38954d695e5f9f9a72dcd8fe19055edcea921704/ternary_pet/r6_cpu_generation_base_vs_hybrid.py" -- --family smol`

Interpretation: The Granite evidence supports post-quantization generative degradation for these prompts relative to the higher-precision base, but not yet a universal failure across use cases. A direct Q3 or W-generation control would be needed to attribute generation degradation specifically to H rather than ternary precision generally. No new GPU training should be launched.
