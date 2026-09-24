"""Instantiate the full ANKA-50M baseline and run a short smoke test."""

from __future__ import annotations

import argparse
from contextlib import nullcontext

import torch

from anka import ANKAConfig, ANKAForCausalLM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--backward", action="store_true")
    return parser.parse_args()


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but no CUDA device is available")
    return torch.device(requested)


def main() -> None:
    args = parse_args()
    torch.manual_seed(1234)

    config = ANKAConfig()
    if not 2 <= args.seq_len <= config.max_seq_len:
        raise ValueError(f"seq-len must be between 2 and {config.max_seq_len}")
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")

    device = choose_device(args.device)
    model = ANKAForCausalLM(config).to(device)
    model.train(mode=args.backward)
    tokens = torch.randint(
        0,
        config.vocab_size,
        (args.batch_size, args.seq_len),
        dtype=torch.long,
        device=device,
    )

    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    precision_context = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if use_bf16
        else nullcontext()
    )
    grad_context = nullcontext() if args.backward else torch.no_grad()

    with grad_context, precision_context:
        output = model(tokens, targets=tokens)

    if args.backward:
        output.loss.backward()

    print(f"device: {device}")
    print(f"precision: {'BF16 autocast' if use_bf16 else 'FP32'}")
    print(f"parameters: {model.num_trainable_parameters():,}")
    print(f"logits: {tuple(output.logits.shape)}")
    print(f"loss: {output.loss.item():.6f}")
    print("backward: PASS" if args.backward else "forward: PASS")


if __name__ == "__main__":
    main()
