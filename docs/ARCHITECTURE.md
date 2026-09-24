# RMW3 architecture

ANKA-50M RMW3 has 13 physical Transformer blocks that are applied twice. Parameters are shared across the two passes, giving 26 effective block applications without doubling the backbone parameter count.

## Backbone

- 512-dimensional token states
- pre-RMSNorm decoder blocks
- grouped-query attention with 8 query heads and 2 key/value heads
- rotary position embeddings
- 1,280-dimensional SwiGLU feed-forward layers
- tied token embedding and language-model output weights

## Reflect

Before a physical block is used in the second pass, its current state is compared with the saved state produced by the same block in the first pass. A rank-32 transform converts this discrepancy into a learned correction, controlled by a sigmoid gate.

The released checkpoint's ablation showed that disabling Reflect increased validation loss by 0.2671 on the audit sample. All 13 units had a positive individual ablation contribution. These measurements are recorded in `results/architecture_audits/reflect_ablation_calibration.json`.

## Morph

Four sparse Morph stages are placed across the 26 effective block applications. Each stage uses a learned router to select one of four rank-368 experts per token. The selected expert output is added through a learned sigmoid scale; a straight-through probability term keeps the router trainable.

Disabling Morph increased validation loss by 0.1678 on the same audit sample. Shifting router/expert assignments increased loss by 0.0266, providing a direct test that learned routing mattered. Details are in `results/architecture_audits/morph_ablation_calibration.json`.

## Parameter budget

The exact unique trainable parameter count is 48,944,657. Run:

```bash
python scripts/verify_parameter_count.py
```

The checkpoint serializes both names of the tied embedding/output tensor for compatibility. Runtime tying ensures this matrix is counted once.
