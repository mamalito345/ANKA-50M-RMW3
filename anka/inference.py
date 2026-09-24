"""Loading and generation helpers for ANKA-50M RMW3."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import load_file
from tokenizers import Tokenizer

from .modeling_anka import RMW3, RMW3Config, count_parameters


def load_model(model_dir: str | Path, device: str | None = None):
    model_dir = Path(model_dir)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    raw_config = json.loads((model_dir / "config.json").read_text())
    config = RMW3Config(
        **{field: raw_config[field] for field in RMW3Config.__dataclass_fields__}
    )
    model = RMW3(config)
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    model = model.to(device=device, dtype=dtype)
    model.load_state_dict(
        load_file(str(model_dir / "model.safetensors"), device="cpu"),
        strict=True,
    )
    model.eval()
    if count_parameters(model) != 48_944_657:
        raise RuntimeError("Unexpected parameter count")
    tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    return model, tokenizer


@torch.inference_mode()
def generate(
    model: RMW3,
    tokenizer: Tokenizer,
    prompt: str,
    max_new_tokens: int = 128,
    temperature: float = 0.8,
    top_p: float = 0.95,
    seed: int = 3407,
) -> str:
    device = next(model.parameters()).device
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False).ids or [1]
    generated: list[int] = []
    generator = torch.Generator(device=device).manual_seed(seed)

    for _ in range(max_new_tokens):
        active = (prompt_ids + generated)[-model.config.max_seq_len :]
        inputs = torch.tensor([active], device=device, dtype=torch.long)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            logits, _ = model(inputs)
        next_logits = logits[0, -1].float()
        if temperature <= 0:
            next_id = int(next_logits.argmax())
        else:
            probabilities = torch.softmax(next_logits / temperature, dim=-1)
            if top_p < 1.0:
                sorted_probs, sorted_ids = torch.sort(probabilities, descending=True)
                remove = (torch.cumsum(sorted_probs, -1) - sorted_probs) > top_p
                sorted_probs[remove] = 0
                sorted_probs /= sorted_probs.sum().clamp_min(1e-12)
                position = torch.multinomial(sorted_probs, 1, generator=generator)
                next_id = int(sorted_ids[position].item())
            else:
                next_id = int(torch.multinomial(probabilities, 1, generator=generator))
        if next_id == 2:
            break
        generated.append(next_id)

    return prompt + tokenizer.decode(generated, skip_special_tokens=True)
