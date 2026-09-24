# C0 baseline audit

## Result

The architecture, independent parameter derivation, causal computation, and optimizer mathematics were consistent at source level. The accompanying unit and GPU smoke tests were used before the project advanced to Reflect and Morph experiments.

## Verified properties

- Decoder-only causal Transformer
- 28 layers with width 384
- 6 query heads, 2 key/value heads, head dimension 64
- SwiGLU width 1,024
- RoPE context 1,024
- learned RMSNorm scales
- no Linear biases
- tied input/output embedding
- 49,960,320 unique trainable parameters
- 39,680 parameters below the 50M limit

## Default pilot mathematics

```text
micro batch                 8
sequence length         1,024
tokens per micro step    8,192
gradient accumulation       32
tokens per update       262,144
optimizer updates           977
total seen tokens    256,114,688
```

The loss is divided by the accumulation count before each backward call. AdamW uses betas `(0.9, 0.95)`, weight decay on matrices but not normalization scales, gradient clipping at `1.0`, and fused updates when supported.

## Precision paths

| Hardware | Precision path |
|---|---|
| A100 / L4 | BF16 autocast, no scaler |
| T4 | FP16 autocast with GradScaler |
| CPU | FP32 |

## Reproduce the acceptance checks

```bash
python -m unittest -v tests/test_model.py tests/test_training.py
python verify_baseline.py --device cuda --seq-len 32 --backward
python train_smoke.py \
  --device cuda \
  --sequence-length 1024 \
  --micro-batch-size 2 \
  --global-batch-tokens 8192 \
  --warmup-steps 2 \
  --measure-steps 5 \
  --json-out results/baseline_eager.json
```
