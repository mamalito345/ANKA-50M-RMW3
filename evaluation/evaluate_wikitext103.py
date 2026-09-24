#!/usr/bin/env python3
"""Evaluate ANKA on the WikiText-103 raw test split."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset
from tqdm.auto import tqdm

from anka.inference import load_model


@torch.inference_mode()
def evaluate(model, tokenizer, batch_size: int = 8) -> dict:
    dataset = load_dataset(
        "Salesforce/wikitext", "wikitext-103-raw-v1", split="test"
    )
    text = "\n\n".join(row["text"].strip() for row in dataset if row["text"].strip())
    text_ids = tokenizer.encode(text, add_special_tokens=False).ids
    stream = [1, *text_ids]
    windows = []
    for target_begin in range(1, len(stream), 512):
        target_end = min(target_begin + 512, len(stream))
        context_begin = max(0, target_end - 1025)
        sequence = stream[context_begin:target_end]
        score_start = max(0, target_begin - (context_begin + 1))
        windows.append((sequence[:-1], sequence[1:], score_start))

    device = next(model.parameters()).device
    total_nll = 0.0
    total_tokens = 0
    for start in tqdm(range(0, len(windows), batch_size), desc="WikiText-103"):
        items = windows[start : start + batch_size]
        longest = max(len(item[0]) for item in items)
        inputs = torch.zeros((len(items), longest), device=device, dtype=torch.long)
        for row, (input_ids, _, _) in enumerate(items):
            inputs[row, : len(input_ids)] = torch.tensor(input_ids, device=device)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            logits, _ = model(inputs)
        for row, (_, targets, score_start) in enumerate(items):
            selected = logits[row, score_start : len(targets)].float()
            target = torch.tensor(targets[score_start:], device=device)
            total_nll += float(F.cross_entropy(selected, target, reduction="sum"))
            total_tokens += target.numel()

    loss = total_nll / total_tokens
    return {
        "dataset": "Salesforce/wikitext",
        "config": "wikitext-103-raw-v1",
        "split": "test",
        "scored_tokens": total_tokens,
        "context_length": 1024,
        "stride": 512,
        "mean_loss": loss,
        "perplexity": math.exp(loss),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("reports/wikitext103.json"))
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    model, tokenizer = load_model(args.model_dir)
    result = evaluate(model, tokenizer, args.batch_size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
