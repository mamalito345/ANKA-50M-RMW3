"""Benchmark the full ANKA-50M training step without a data-loader bottleneck."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import itertools
import json
from pathlib import Path
import statistics

import torch

from anka import (
    ANKAForCausalLM,
    TrainConfig,
    TrainingEngine,
    format_step_metrics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--precision",
        choices=("auto", "bf16", "fp16", "fp32"),
        default="auto",
    )
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--micro-batch-size", type=int, default=2)
    parser.add_argument("--global-batch-tokens", type=int, default=2_048)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--measure-steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--no-fused-adamw", action="store_true")
    parser.add_argument("--json-out", type=Path)
    return parser.parse_args()


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but no CUDA device is available")
    return torch.device(requested)


def main() -> None:
    args = parse_args()
    if args.warmup_steps < 0:
        raise ValueError("warmup-steps cannot be negative")
    if args.measure_steps <= 0:
        raise ValueError("measure-steps must be positive")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    device = choose_device(args.device)
    model = ANKAForCausalLM().to(device)

    # The LR schedule is intentionally much longer than this smoke run. Its
    # purpose here is mechanics and throughput, not meaningful learning.
    config = TrainConfig(
        total_steps=1_000,
        warmup_steps=50,
        global_batch_tokens=args.global_batch_tokens,
        micro_batch_size=args.micro_batch_size,
        sequence_length=args.sequence_length,
        precision=args.precision,
        use_fused_adamw=not args.no_fused_adamw,
    )
    engine = TrainingEngine(model, config)
    if args.compile:
        # Compile only the forward/backward model. The optimizer keeps the same
        # Parameter objects and remains outside the compiled graph for now.
        engine.model = torch.compile(model, dynamic=False)

    # Reusing one already-resident random batch removes tokenization, disk, CPU,
    # and host-to-device transfer from this model-speed measurement.
    fixed_batch = torch.randint(
        0,
        model.config.vocab_size,
        (config.micro_batch_size, config.sequence_length),
        dtype=torch.long,
        device=device,
    )
    batches = itertools.repeat(fixed_batch)

    print("=" * 72)
    print("ANKA-50M TRAINING SMOKE BENCHMARK")
    print("=" * 72)
    if device.type == "cuda":
        print(f"gpu                  : {torch.cuda.get_device_name(device)}")
    print(f"device               : {device}")
    print(f"precision            : {engine.precision.name}")
    print(f"parameters           : {model.num_trainable_parameters():,}")
    print(f"sequence length      : {config.sequence_length:,}")
    print(f"micro batch          : {config.micro_batch_size}")
    print(f"accumulation steps   : {config.gradient_accumulation_steps}")
    print(f"global batch tokens  : {config.global_batch_tokens:,}")
    print(f"fused AdamW          : {engine.optimizer_summary.fused}")
    print(
        f"decayed parameters   : "
        f"{engine.optimizer_summary.decay_parameters:,}"
    )
    print(
        f"no-decay parameters  : "
        f"{engine.optimizer_summary.no_decay_parameters:,}"
    )
    print(f"torch.compile        : {args.compile}")
    print("-" * 72)

    for _ in range(args.warmup_steps):
        engine.train_step(batches)
    if args.warmup_steps:
        print(f"warmup               : {args.warmup_steps} steps complete")

    measured = []
    for _ in range(args.measure_steps):
        metrics = engine.train_step(batches)
        measured.append(metrics)
        print(format_step_metrics(metrics))

    throughputs = [metrics.tokens_per_second for metrics in measured]
    durations = [metrics.elapsed_seconds for metrics in measured]
    duration_mean = statistics.fmean(durations)
    duration_cv = (
        statistics.pstdev(durations) / duration_mean if duration_mean else 0.0
    )

    print("-" * 72)
    print(f"median tokens/s      : {statistics.median(throughputs):,.0f}")
    print(f"step-time CV         : {duration_cv:.2%}")
    print(f"last peak memory     : {measured[-1].peak_memory_gib:.2f} GiB")

    if args.json_out is not None:
        report = {
            "kind": "anka50m_training_smoke",
            "torch_version": torch.__version__,
            "device": str(device),
            "gpu": (
                torch.cuda.get_device_name(device)
                if device.type == "cuda"
                else None
            ),
            "precision": engine.precision.name,
            "parameters": model.num_trainable_parameters(),
            "sequence_length": config.sequence_length,
            "micro_batch_size": config.micro_batch_size,
            "gradient_accumulation_steps": (
                config.gradient_accumulation_steps
            ),
            "global_batch_tokens": config.global_batch_tokens,
            "fused_adamw": engine.optimizer_summary.fused,
            "compiled": args.compile,
            "warmup_steps": args.warmup_steps,
            "measure_steps": args.measure_steps,
            "median_tokens_per_second": statistics.median(throughputs),
            "step_time_cv": duration_cv,
            "last_peak_memory_gib": measured[-1].peak_memory_gib,
            "steps": [asdict(metrics) for metrics in measured],
        }
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        print(f"json report          : {args.json_out}")


if __name__ == "__main__":
    main()
