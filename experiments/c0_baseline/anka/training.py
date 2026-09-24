"""Single-GPU pretraining utilities for the first ANKA-50M experiment.

This module deliberately separates model correctness from data loading. It can
train on any iterator that yields a token tensor or a mapping containing
``input_ids`` and optional ``targets``/``labels`` tensors.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import inspect
import math
import time
from typing import Any, Iterator, Literal, Mapping, Optional, TypeAlias

import torch
import torch.nn as nn


Batch: TypeAlias = torch.Tensor | Mapping[str, torch.Tensor]
PrecisionName: TypeAlias = Literal["auto", "bf16", "fp16", "fp32"]


# -----------------------------------------------------------------------------
# 1. Training configuration and effective batch size
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class TrainConfig:
    """Optimizer-step settings for one single-GPU experiment arm.

    The defaults describe our initial 256M-token screening run: 977 optimizer
    updates at 262,144 tokens per update, or 256,114,688 seen tokens.
    Learning-rate values remain pilot defaults until the LR range test.
    """

    total_steps: int = 977
    warmup_steps: int = 49
    global_batch_tokens: int = 262_144
    micro_batch_size: int = 8
    sequence_length: int = 1_024

    max_learning_rate: float = 3e-3
    min_learning_rate: float = 3e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0

    precision: PrecisionName = "auto"
    use_fused_adamw: bool = True

    def __post_init__(self) -> None:
        if self.total_steps <= 0:
            raise ValueError("total_steps must be positive")
        if not 0 <= self.warmup_steps < self.total_steps:
            raise ValueError("warmup_steps must be in [0, total_steps)")
        if self.global_batch_tokens <= 0:
            raise ValueError("global_batch_tokens must be positive")
        if self.micro_batch_size <= 0:
            raise ValueError("micro_batch_size must be positive")
        if self.sequence_length < 2:
            raise ValueError("sequence_length must be at least two")
        if self.global_batch_tokens % self.tokens_per_micro_step != 0:
            raise ValueError(
                "global_batch_tokens must be divisible by "
                "micro_batch_size * sequence_length"
            )
        if self.max_learning_rate <= 0.0:
            raise ValueError("max_learning_rate must be positive")
        if not 0.0 <= self.min_learning_rate <= self.max_learning_rate:
            raise ValueError(
                "min_learning_rate must be in [0, max_learning_rate]"
            )
        if self.weight_decay < 0.0:
            raise ValueError("weight_decay cannot be negative")
        if not 0.0 <= self.beta1 < 1.0 or not 0.0 <= self.beta2 < 1.0:
            raise ValueError("AdamW betas must be in [0, 1)")
        if self.adam_epsilon <= 0.0:
            raise ValueError("adam_epsilon must be positive")
        if self.max_grad_norm <= 0.0:
            raise ValueError("max_grad_norm must be positive")
        if self.precision not in {"auto", "bf16", "fp16", "fp32"}:
            raise ValueError(f"unsupported precision: {self.precision}")

    @property
    def tokens_per_micro_step(self) -> int:
        return self.micro_batch_size * self.sequence_length

    @property
    def gradient_accumulation_steps(self) -> int:
        return self.global_batch_tokens // self.tokens_per_micro_step

    def learning_rate_at(self, optimizer_step: int) -> float:
        """Linear warmup followed by cosine decay to min_learning_rate."""

        if optimizer_step < 0:
            raise ValueError("optimizer_step cannot be negative")
        if self.warmup_steps and optimizer_step < self.warmup_steps:
            return self.max_learning_rate * (
                (optimizer_step + 1) / self.warmup_steps
            )

        if optimizer_step >= self.total_steps - 1:
            return self.min_learning_rate

        decay_updates = self.total_steps - self.warmup_steps
        if decay_updates <= 1:
            return self.min_learning_rate

        progress = (optimizer_step - self.warmup_steps) / (decay_updates - 1)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_learning_rate + cosine * (
            self.max_learning_rate - self.min_learning_rate
        )


# -----------------------------------------------------------------------------
# 2. AdamW parameter groups
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class OptimizerGroupSummary:
    decay_names: tuple[str, ...]
    no_decay_names: tuple[str, ...]
    decay_parameters: int
    no_decay_parameters: int
    fused: bool


def _parameter_groups(
    model: nn.Module, weight_decay: float
) -> tuple[list[dict[str, Any]], OptimizerGroupSummary]:
    """Decay matrices; do not decay one-dimensional RMSNorm scales."""

    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    decay_names: list[str] = []
    no_decay_names: list[str] = []
    seen_parameter_ids: set[int] = set()

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        parameter_id = id(parameter)
        if parameter_id in seen_parameter_ids:
            raise RuntimeError(f"parameter appears more than once: {name}")
        seen_parameter_ids.add(parameter_id)

        if parameter.ndim >= 2:
            decay.append(parameter)
            decay_names.append(name)
        else:
            no_decay.append(parameter)
            no_decay_names.append(name)

    if not decay or not no_decay:
        raise RuntimeError("expected both decay and no-decay parameter groups")

    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    summary = OptimizerGroupSummary(
        decay_names=tuple(decay_names),
        no_decay_names=tuple(no_decay_names),
        decay_parameters=sum(parameter.numel() for parameter in decay),
        no_decay_parameters=sum(parameter.numel() for parameter in no_decay),
        fused=False,
    )
    return groups, summary


def build_adamw(
    model: nn.Module,
    config: TrainConfig,
    device: torch.device,
) -> tuple[torch.optim.AdamW, OptimizerGroupSummary]:
    groups, summary = _parameter_groups(model, config.weight_decay)

    supports_fused = "fused" in inspect.signature(torch.optim.AdamW).parameters
    use_fused = config.use_fused_adamw and device.type == "cuda" and supports_fused
    optimizer_kwargs: dict[str, Any] = {
        "lr": config.max_learning_rate,
        "betas": (config.beta1, config.beta2),
        "eps": config.adam_epsilon,
    }
    if supports_fused:
        optimizer_kwargs["fused"] = use_fused

    optimizer = torch.optim.AdamW(groups, **optimizer_kwargs)
    summary = OptimizerGroupSummary(
        decay_names=summary.decay_names,
        no_decay_names=summary.no_decay_names,
        decay_parameters=summary.decay_parameters,
        no_decay_parameters=summary.no_decay_parameters,
        fused=use_fused,
    )
    return optimizer, summary


# -----------------------------------------------------------------------------
# 3. Mixed-precision policy
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class PrecisionPolicy:
    name: Literal["bf16", "fp16", "fp32"]
    autocast_dtype: Optional[torch.dtype]
    use_grad_scaler: bool

    def context(self, device: torch.device):
        if self.autocast_dtype is None:
            return nullcontext()
        return torch.autocast(
            device_type=device.type,
            dtype=self.autocast_dtype,
        )


def resolve_precision(
    requested: PrecisionName, device: torch.device
) -> PrecisionPolicy:
    if device.type != "cuda":
        if requested not in {"auto", "fp32"}:
            raise ValueError("BF16/FP16 training in this project requires CUDA")
        return PrecisionPolicy("fp32", None, False)

    if requested == "auto":
        requested = "bf16" if torch.cuda.is_bf16_supported() else "fp16"

    if requested == "bf16":
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("this CUDA device does not support BF16")
        return PrecisionPolicy("bf16", torch.bfloat16, False)
    if requested == "fp16":
        return PrecisionPolicy("fp16", torch.float16, True)
    if requested == "fp32":
        return PrecisionPolicy("fp32", None, False)
    raise ValueError(f"unsupported precision: {requested}")


# -----------------------------------------------------------------------------
# 4. One optimizer update with gradient accumulation
# -----------------------------------------------------------------------------


@dataclass
class TrainState:
    optimizer_step: int = 0
    attempted_steps: int = 0
    tokens_seen: int = 0


@dataclass(frozen=True)
class StepMetrics:
    optimizer_step: int
    attempted_steps: int
    tokens_seen: int
    step_tokens: int
    loss: float
    learning_rate: float
    grad_norm: float
    elapsed_seconds: float
    data_seconds: float
    tokens_per_second: float
    peak_memory_gib: float
    reserved_memory_gib: float
    update_skipped: bool

    @property
    def data_fraction(self) -> float:
        if self.elapsed_seconds == 0.0:
            return 0.0
        return self.data_seconds / self.elapsed_seconds


class TrainingEngine:
    """A small, explicit single-GPU training engine for controlled experiments."""

    def __init__(self, model: nn.Module, config: TrainConfig) -> None:
        self.model = model
        self.config = config
        self.device = next(model.parameters()).device

        if config.sequence_length > model.config.max_seq_len:
            raise ValueError(
                "training sequence_length exceeds model.config.max_seq_len"
            )

        for parameter in model.parameters():
            if parameter.device != self.device:
                raise RuntimeError("all model parameters must be on one device")

        self.precision = resolve_precision(config.precision, self.device)
        self.optimizer, self.optimizer_summary = build_adamw(
            model, config, self.device
        )
        self.scaler = (
            torch.amp.GradScaler("cuda", enabled=True)
            if self.precision.use_grad_scaler
            else None
        )
        self.state = TrainState()

    @staticmethod
    def _unpack_batch(batch: Batch) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(batch, torch.Tensor):
            return batch, batch
        if "input_ids" not in batch:
            raise KeyError("batch mapping must contain input_ids")

        input_ids = batch["input_ids"]
        targets = batch.get("targets", batch.get("labels", input_ids))
        return input_ids, targets

    def _prepare_batch(self, batch: Batch) -> tuple[torch.Tensor, torch.Tensor]:
        input_ids, targets = self._unpack_batch(batch)
        same_tensor = targets is input_ids
        input_ids = input_ids.to(
            self.device, dtype=torch.long, non_blocking=True
        )
        targets = (
            input_ids
            if same_tensor
            else targets.to(self.device, dtype=torch.long, non_blocking=True)
        )

        expected_shape = (
            self.config.micro_batch_size,
            self.config.sequence_length,
        )
        if tuple(input_ids.shape) != expected_shape:
            raise ValueError(
                f"input batch has shape {tuple(input_ids.shape)}; "
                f"expected {expected_shape}"
            )
        if targets.shape != input_ids.shape:
            raise ValueError("targets must have the same shape as input_ids")
        return input_ids, targets

    def _set_learning_rate(self, learning_rate: float) -> None:
        for parameter_group in self.optimizer.param_groups:
            parameter_group["lr"] = learning_rate

    def _cuda_synchronize(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def train_step(self, batches: Iterator[Batch]) -> StepMetrics:
        """Consume one effective batch and attempt one AdamW update."""

        if self.state.optimizer_step >= self.config.total_steps:
            raise RuntimeError(
                "training is complete; refusing to exceed total_steps"
            )

        self.model.train()
        learning_rate = self.config.learning_rate_at(self.state.optimizer_step)
        self._set_learning_rate(learning_rate)
        self.optimizer.zero_grad(set_to_none=True)

        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        self._cuda_synchronize()
        started_at = time.perf_counter()

        loss_sum = torch.zeros((), dtype=torch.float32, device=self.device)
        step_tokens = 0
        data_seconds = 0.0

        for _ in range(self.config.gradient_accumulation_steps):
            data_started_at = time.perf_counter()
            batch = next(batches)
            data_seconds += time.perf_counter() - data_started_at
            input_ids, targets = self._prepare_batch(batch)

            with self.precision.context(self.device):
                output = self.model(input_ids, targets=targets)
                if output.loss is None:
                    raise RuntimeError("model did not return a training loss")
                micro_loss = output.loss
                backward_loss = (
                    micro_loss / self.config.gradient_accumulation_steps
                )

            loss_sum += micro_loss.detach().float()
            if self.scaler is None:
                backward_loss.backward()
            else:
                self.scaler.scale(backward_loss).backward()

            step_tokens += input_ids.numel()
            del output, micro_loss, backward_loss, input_ids, targets

        if step_tokens != self.config.global_batch_tokens:
            raise RuntimeError(
                f"step consumed {step_tokens:,} tokens; expected "
                f"{self.config.global_batch_tokens:,}"
            )

        if self.scaler is not None:
            self.scaler.unscale_(self.optimizer)

        grad_norm_tensor = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            max_norm=self.config.max_grad_norm,
            error_if_nonfinite=self.scaler is None,
        )

        update_skipped = False
        if self.scaler is None:
            self.optimizer.step()
        else:
            scale_before = self.scaler.get_scale()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            update_skipped = self.scaler.get_scale() < scale_before

        self.state.attempted_steps += 1
        self.state.tokens_seen += step_tokens
        if not update_skipped:
            self.state.optimizer_step += 1

        self._cuda_synchronize()
        elapsed_seconds = time.perf_counter() - started_at
        mean_loss = (
            loss_sum / self.config.gradient_accumulation_steps
        ).item()
        grad_norm = grad_norm_tensor.detach().float().item()

        if self.device.type == "cuda":
            peak_memory_gib = torch.cuda.max_memory_allocated(self.device) / 2**30
            reserved_memory_gib = torch.cuda.max_memory_reserved(self.device) / 2**30
        else:
            peak_memory_gib = 0.0
            reserved_memory_gib = 0.0

        return StepMetrics(
            optimizer_step=self.state.optimizer_step,
            attempted_steps=self.state.attempted_steps,
            tokens_seen=self.state.tokens_seen,
            step_tokens=step_tokens,
            loss=mean_loss,
            learning_rate=learning_rate,
            grad_norm=grad_norm,
            elapsed_seconds=elapsed_seconds,
            data_seconds=data_seconds,
            tokens_per_second=step_tokens / elapsed_seconds,
            peak_memory_gib=peak_memory_gib,
            reserved_memory_gib=reserved_memory_gib,
            update_skipped=update_skipped,
        )


def format_step_metrics(metrics: StepMetrics) -> str:
    skipped = " SKIPPED" if metrics.update_skipped else ""
    return (
        f"step={metrics.optimizer_step:05d} "
        f"loss={metrics.loss:.4f} "
        f"lr={metrics.learning_rate:.3e} "
        f"grad={metrics.grad_norm:.3f} "
        f"tok/s={metrics.tokens_per_second:,.0f} "
        f"data={metrics.data_fraction:.1%} "
        f"peak={metrics.peak_memory_gib:.2f}GiB{skipped}"
    )
