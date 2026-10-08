# G1-0 — Granite 4.0 350M architecture / quantization smoke

Job: `6ac6eacfdf2184ac91ac658a`  
Pinned code: `1c85fdb1bdccd70c5c925ea2807bf773381739fd`  
Hardware: A10G-small  
Status: **completed technical gate; G1-1 blocked pending precision fix**

## Key audit

| Check | Result |
|---|---:|
| Total parameters (derived from target + excluded) | 352,379,904 |
| Targeted weight parameters | 249,561,088 |
| Excluded parameters | 102,818,816 |
| Target linear modules (derived from 336 trainable objects / 2) | 168 |
| Non-quantized parameters frozen | yes |
| Unexpected trainable tensors | none |
| Initial Q3 zero fraction | 31.99% |
| Initial Q9 zero fraction | 12.37% |
| Q3 peak CUDA memory | 6.30 GiB |
| Q9 peak CUDA memory | 8.17 GiB |

## Numerical result

The smoke exposed a compute-path problem before any scientific experiment:

- the **unquantized source evaluation itself** returned NaN loss/PPL under the
  inherited FP16-autocast path;
- the Q3 student cross-entropy was finite (21.55), but teacher KL, total loss,
  and gradient norm were NaN;
- the Q9 student cross-entropy was finite (26.62), but teacher KL, total loss,
  and gradient norm were NaN.

Because the source/teacher path is already non-finite, this is **not evidence
that Q3 or Q9 fails on Granite**. It is a technical precision incompatibility in
the inherited Smol compute path.

## Decision

Do **not** launch G1-1 direct-Q3 schedule calibration yet.

Run a narrow G1-0b precision diagnostic comparing source/teacher behavior under
FP32, BF16 autocast / BF16 teacher, and FP16 autocast. If BF16 is finite and
tracks FP32 closely, preregister BF16 mixed precision as the Granite-specific
compute setting before any scientific D-vs-S result is observed.

This job remains a technical gate only and contributes no scientific
generalization result.
