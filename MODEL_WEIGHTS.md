# Model weights

The public Git repository intentionally excludes the model checkpoint. Download the release from:

https://huggingface.co/Melih1234/ANKA-50M-RMW3

Required files:

- `model.safetensors`
- `tokenizer.json`
- `config.json`
- `generation_config.json`
- `modeling_anka.py`

Keep these files in one local directory and pass that directory to the generation or evaluation command with `--model-dir`.

The Hugging Face release contains BF16 weights. The safetensors file is therefore smaller than an FP32 serialization of the same 48,944,657-parameter model.
