# ANKA-50M C0 baseline

This directory preserves the first full-size control model used before Reflect, Morph, and WorldPairs were introduced.

## Locked configuration

| Item | Value |
|---|---:|
| Parameters | 49,960,320 |
| Layers | 28 |
| Hidden size | 384 |
| Attention | 6 query / 2 KV heads |
| Head size | 64 |
| SwiGLU width | 1,024 |
| Vocabulary | 15,360 |
| Context | 1,024 |
| Position encoding | RoPE |
| Normalization | pre-RMSNorm |
| Weight tying | Yes |

The control model intentionally contains none of the final ANKA mechanisms. Its purpose was to provide an interpretable baseline for architecture experiments.

## Parameter derivation

| Component | Parameters |
|---|---:|
| Tied embedding/output | 5,898,240 |
| Attention per block | 393,216 |
| SwiGLU per block | 1,179,648 |
| Two RMSNorms per block | 768 |
| One block | 1,573,632 |
| 28 blocks | 44,061,696 |
| Final RMSNorm | 384 |
| **Total** | **49,960,320** |

## Validation

```bash
python -m unittest -v tests/test_model.py tests/test_training.py
python verify_baseline.py --device auto --seq-len 32
```

The tests cover parameter counting, RoPE norm preservation, GQA shapes, forward/backward execution, causal masking, optimizer grouping, and tied weights.

## Throughput smoke test

```bash
python train_smoke.py \
  --device cuda \
  --sequence-length 1024 \
  --micro-batch-size 2 \
  --global-batch-tokens 8192 \
  --warmup-steps 2 \
  --measure-steps 5 \
  --json-out results/baseline_eager.json
```

The smoke test reuses a GPU-resident random batch, so it measures the model and optimizer without disk, tokenizer, or DataLoader overhead.
