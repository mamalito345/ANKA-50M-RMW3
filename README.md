# ANKA-50M RMW3

ANKA-50M RMW3 is a 48,944,657-parameter English causal base language model designed and trained from scratch for the GIBC 2026 first-model track. Its 13-block Transformer backbone is reused for two recurrent passes and augmented with learned Reflect corrections and sparse Morph routing.

> This release is a base completion model, not an instruction-tuned chatbot.

[Download the model from Hugging Face](https://huggingface.co/Melih1234/ANKA-50M-RMW3)

## Highlights

- Trained from random initialization on 5 billion tokens.
- 13 physical blocks reused for 26 effective block applications.
- Reflect and Morph both show positive measured ablation contributions.
- Sustained 117,899 training tokens/s on one NVIDIA A100-SXM4 80 GB.
- Completed the released run in 11 h 46 m 49 s, or 11.78 GPU-hours.
- Includes the exact architecture, evaluation adapter, audit reports, and eight cleaned experiment notebooks.

## Official benchmark results

All multiple-choice results are zero-shot and were measured with `lm-evaluation-harness==0.4.12`.

| Task | Metric | Score |
|---|---|---:|
| HellaSwag | normalized accuracy | 28.46% |
| ARC-Easy | normalized accuracy | 42.26% |
| PIQA | normalized accuracy | 56.80% |
| WinoGrande | accuracy | 51.46% |
| WikiText-103 | perplexity | 25.967 |

Machine-readable reports are available in `results/official_benchmark_summary.json`, `results/official_benchmark_summary.csv`, and `results/wikitext103_test_perplexity.json`.

## Model configuration

| Item | Value |
|---|---:|
| Unique trainable parameters | 48,944,657 |
| Vocabulary | 16,384 |
| Hidden size | 512 |
| Physical / effective blocks | 13 / 26 |
| Attention | 8 query / 2 KV heads |
| Feed-forward width | 1,280 |
| Context length | 1,024 tokens |
| Reflect rank | 32 |
| Morph stages / experts | 4 / 4 per stage |
| Morph expert rank | 368 |

## Released-run efficiency

| Item | Value |
|---|---:|
| Training tokens | 5,000,000,000 |
| Training steps | 76,294 |
| Precision | BF16 |
| Hardware | 1 x NVIDIA A100-SXM4 80 GB |
| Average throughput | 117,899 tokens/s |
| Wall time | 42,409 s (11 h 46 m 49 s) |
| GPU time | 11.78 GPU-hours |
| Efficiency | about 424.4M tokens/GPU-hour |
| Best internal validation loss | 2.334379 |

These figures describe the submitted base checkpoint. Later Stage-2 experiments are not included in this release.

## Architecture evidence

The architecture audits use the same 4,096-token evaluation sample for comparable ablations.

| Configuration | Loss | Increase vs. full model |
|---|---:|---:|
| Full RMW3 | 2.629027 | — |
| Reflect disabled | 2.896107 | +0.267081 |
| Morph disabled | 2.796865 | +0.167838 |
| Morph router/expert assignments shifted | 2.655662 | +0.026635 |

All 13 Reflect units and all four Morph stages had positive individual ablation contributions on this audit sample. These results demonstrate contribution on the measured sample; they are not a claim of universal superiority.

## Repository map

- `anka/`: exact RMW3 architecture and inference helpers
- `evaluation/`: `lm-evaluation-harness` adapter and official task runner
- `scripts/`: parameter verification and text-generation entry points
- `notebooks/`: eight English, output-free experiment notebooks
- `experiments/c0_baseline/`: reproducible C0 baseline package
- `results/`: official measurements, samples, and architecture audits
- `docs/`: architecture, data, training, and AI-assistance documentation
- `PUBLISH_TO_GITHUB_COLAB.ipynb`: one-cell Colab utility for the initial GitHub publication

## Setup

Clone this repository, enter its directory, and create an environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Download the model files from [Hugging Face](https://huggingface.co/Melih1234/ANKA-50M-RMW3), then run:

```bash
python scripts/verify_parameter_count.py
python scripts/generate.py \
  --model-dir /path/to/ANKA-50M-RMW3 \
  --prompt "Science is important because"
```

## Reproduce the official tasks

```bash
python -m evaluation.run_official_benchmarks \
  --model-dir /path/to/ANKA-50M-RMW3 \
  --output reports/official.json
```

Add `--limit 32` for a smoke test. See `evaluation/README.md` for the exact harness version, task list, and expected metrics.

## Experiment notebooks

The notebooks preserve the research path from the baseline through the final RMW3 run. They are historical, path-dependent Colab experiments rather than a single sequential `Run all` workflow. Read `notebooks/README.md` before running them.

## Documentation

- [Architecture](docs/ARCHITECTURE.md)
- [Training](docs/TRAINING.md)
- [Data and tokenizer](docs/DATA.md)
- [AI assistance disclosure](docs/AI_ASSISTANCE.md)
- [Model weights](MODEL_WEIGHTS.md)

## Limitations

ANKA-50M RMW3 can hallucinate, repeat phrases, produce weak long-range reasoning, and reflect biases in its training data. It is not aligned for dialogue, safety refusals, tool use, or high-stakes decisions. Its 1,024-token context and 50M-scale capacity also limit instruction following and factual recall compared with larger models.

## License

Code and model weights are released under Apache License 2.0. Dataset components retain their upstream licenses and terms.
