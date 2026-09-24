# Training summary

The submitted base checkpoint was trained from random initialization. It did not use pretrained weights, distillation, or model merging.

| Item | Value |
|---|---:|
| Training tokens | 5,000,000,000 |
| Steps | 76,294 |
| Sequence length | 1,024 |
| Batch size | 64 |
| Precision | BF16 |
| Hardware | 1 × NVIDIA A100-SXM4 80 GB |
| Wall time | 42,409 s (11 h 46 m 49 s) |
| Average throughput | 117,899 tokens/s |
| Best internal validation loss | 2.334379 |

The end-to-end pipeline—data construction, tokenizer training, architecture experiments, 5B run, and official evaluation—is preserved in the eight output-free notebooks under `notebooks/`.

The final model is the **base 5B checkpoint**. Later experimental Stage-2 runs are not used in this release.
