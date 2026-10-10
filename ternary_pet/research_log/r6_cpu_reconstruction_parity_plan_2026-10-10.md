# R6 reconstructed-inference loss parity investigation — CPU diagnostic ready

**2026-10-10.** R6 both H compact archives passed independent private SHA and structural checks, and both CPU reconstructions finished with finite logits, but generated only repetitive punctuation/newlines. This is *not* evidence that NLL is preserved between original training and reconstructed inference. Do not claim parity yet.

## Frozen diagnostic

[CPU reconstruction NLL parity script](../r6_cpu_reconstruction_nll_parity.py), pinned commit `ddcb84cc4566b989429d5cd131199f264c5b6d81`, reuses the checksum-verified packed Q3 decoder, BF16-rounded base-model source parameters, original pinned tokenizers/revisions and exactly **the first four R6 IMDb test documents per model**, identified by their original row numbers and SHA256 hashes; *no adaptive selection*. Uses four 129-token windows per doc, 512 evaluated targets each (2,048 targets/model), scoring the reconstructed H on CPU in FP32. Original H per-document NLL values are embedded verbatim from frozen R6 raw FINAL_JSON. Emits `R6_PARITY_DOC` four times and `R6_PARITY_DIAGNOSTIC_COMPLETE` with differences and finite checks. Does not launch training or use GPU.

**Interpretive caution:** R6 original Granite GPU eval used BF16 autocast, Smol GPU eval used FP16 autocast. CPU FP32 parity does **not** guarantee bit-identical model logits, so tiny differences alone do not establish a restoration error. Large gaps could indicate reconstructed dequantization/scale mismatch or unintended nonquantized state differences. The original compact snapshots only save quantized linears; they cannot restore any nonquantized tensors that training altered, so exact parity is not guaranteed by manifest design. Test may diagnose, but cannot prove cause without further controlled comparison.

## Secure launch prerequisite

Script downloads a private `codeflash85/ternary-pet-r5-checkpoints` snapshot, so it must be run through authenticated local Hugging Face CLI with `--secrets HF_TOKEN`; connector cannot safely supply the user's locally stored credential. No CPU parity job has been submitted at the time of this record. No extra GPU scientific jobs authorized or needed.

### Execute after local CLI authentication

`hf jobs uv run --flavor cpu-basic --timeout 40m --secrets HF_TOKEN "https://raw.githubusercontent.com/steveonw/hackathon/ddcb84cc4566b989429d5cd131199f264c5b6d81/ternary_pet/r6_cpu_reconstruction_nll_parity.py" -- --family granite`

`hf jobs uv run --flavor cpu-basic --timeout 40m --secrets HF_TOKEN "https://raw.githubusercontent.com/steveonw/hackathon/ddcb84cc4566b989429d5cd131199f264c5b6d81/ternary_pet/r6_cpu_reconstruction_nll_parity.py" -- --family smol`
