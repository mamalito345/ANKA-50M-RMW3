#!/usr/bin/env python3
"""Run the four official zero-shot multiple-choice benchmarks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import lm_eval

from anka.inference import load_model
from evaluation.lm_eval_adapter import ANKAHarnessLM


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("reports/official.json"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    model, tokenizer = load_model(args.model_dir)
    adapter = ANKAHarnessLM(model, tokenizer, args.batch_size)
    results = lm_eval.simple_evaluate(
        model=adapter,
        tasks=["hellaswag", "arc_easy", "piqa", "winogrande"],
        num_fewshot=0,
        batch_size=args.batch_size,
        limit=args.limit,
        device=str(adapter.device),
        random_seed=3407,
        numpy_random_seed=3407,
        torch_random_seed=3407,
        fewshot_random_seed=3407,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, default=str))
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
