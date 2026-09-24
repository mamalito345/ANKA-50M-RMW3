#!/usr/bin/env python3
"""Generate a completion from a downloaded Hugging Face model directory."""

import argparse
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from anka.inference import generate, load_model


parser = argparse.ArgumentParser()
parser.add_argument("--model-dir", required=True)
parser.add_argument("--prompt", default="The future of artificial intelligence")
parser.add_argument("--max-new-tokens", type=int, default=128)
parser.add_argument("--temperature", type=float, default=0.8)
parser.add_argument("--top-p", type=float, default=0.95)
parser.add_argument("--seed", type=int, default=3407)
args = parser.parse_args()

model, tokenizer = load_model(args.model_dir)
print(
    generate(
        model,
        tokenizer,
        args.prompt,
        args.max_new_tokens,
        args.temperature,
        args.top_p,
        args.seed,
    )
)
