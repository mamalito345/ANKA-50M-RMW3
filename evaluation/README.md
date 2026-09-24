# Evaluation

Run the four zero-shot competition benchmarks after downloading the Hugging Face model files:

```bash
python -m evaluation.run_official_benchmarks \
  --model-dir /path/to/ANKA-50M-RMW3 \
  --output reports/official.json
```

Use `--limit 32` for a quick smoke test. The published full-run environment used `lm-eval==0.4.12`, batch size 16, seed 3407, and no few-shot examples. The WikiText-103 implementation and full training/evaluation sequence are preserved in `notebooks/ANKA_RMW3_Full_5B_Training_and_Evaluation.ipynb`.

WikiText-103:

```bash
python -m evaluation.evaluate_wikitext103 \
  --model-dir /path/to/ANKA-50M-RMW3 \
  --output reports/wikitext103.json
```
