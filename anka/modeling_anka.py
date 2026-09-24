"""ANKA-50M RMW3 model architecture.

This module contains the exact architecture used by the submitted Base 5B
checkpoint. It intentionally depends only on PyTorch.
"""

from dataclasses import asdict, dataclass
from typing import Optional
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class RMW3Config:
    vocab_size: int = 16_384
    d_model: int = 512
    n_physical_layers: int = 13
    recurrent_passes: int = 2
    n_heads: int = 8
    n_kv_heads: int = 2
    d_ff: int = 1_280
    reflect_rank: int = 32
    morph_rank: int = 368
    morph_stages: int = 4
    morph_experts: int = 4
    max_seq_len: int = 1_024
    rope_theta: float = 10_000.0
    norm_eps: float = 1e-5

    def to_dict(self) -> dict:
        return asdict(self)


class RMSNorm(nn.Module):
    def __init__(self, dimension: int, epsilon: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dimension))
        self.epsilon = epsilon

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        inverse_rms = torch.rsqrt(
            x.float().pow(2).mean(dim=-1, keepdim=True) + self.epsilon
        )
        normalized = (x.float() * inverse_rms).to(x.dtype)
        return normalized * self.weight


def build_rope_cache(
    maximum_sequence_length: int,
    head_dimension: int,
    theta: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    inverse_frequency = 1.0 / (
        theta
        ** (
            torch.arange(0, head_dimension, 2, dtype=torch.float32)
            / head_dimension
        )
    )
    positions = torch.arange(maximum_sequence_length, dtype=torch.float32)
    angles = torch.outer(positions, inverse_frequency)
    return angles.cos(), angles.sin()


def apply_rope(
    x: torch.Tensor,
    cosine: torch.Tensor,
    sine: torch.Tensor,
) -> torch.Tensor:
    sequence_length = x.shape[-2]
    cosine = cosine[:sequence_length].to(device=x.device, dtype=x.dtype)[
        None, None, :, :
    ]
    sine = sine[:sequence_length].to(device=x.device, dtype=x.dtype)[
        None, None, :, :
    ]
    even = x[..., 0::2]
    odd = x[..., 1::2]
    rotated = torch.stack(
        (even * cosine - odd * sine, even * sine + odd * cosine), dim=-1
    )
    return rotated.flatten(-2)


class GroupedQueryAttention(nn.Module):
    def __init__(self, config: RMW3Config):
        super().__init__()
        self.number_of_heads = config.n_heads
        self.number_of_kv_heads = config.n_kv_heads
        self.head_dimension = config.d_model // config.n_heads

        if config.d_model % config.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if config.n_heads % config.n_kv_heads != 0:
            raise ValueError("n_heads must be divisible by n_kv_heads")

        self.query_projection = nn.Linear(
            config.d_model,
            config.n_heads * self.head_dimension,
            bias=False,
        )
        self.key_projection = nn.Linear(
            config.d_model,
            config.n_kv_heads * self.head_dimension,
            bias=False,
        )
        self.value_projection = nn.Linear(
            config.d_model,
            config.n_kv_heads * self.head_dimension,
            bias=False,
        )
        self.output_projection = nn.Linear(
            config.n_heads * self.head_dimension,
            config.d_model,
            bias=False,
        )

        cosine, sine = build_rope_cache(
            config.max_seq_len,
            self.head_dimension,
            config.rope_theta,
        )
        self.register_buffer("rope_cosine", cosine, persistent=False)
        self.register_buffer("rope_sine", sine, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, _ = x.shape
        query = self.query_projection(x).view(
            batch_size,
            sequence_length,
            self.number_of_heads,
            self.head_dimension,
        ).transpose(1, 2)
        key = self.key_projection(x).view(
            batch_size,
            sequence_length,
            self.number_of_kv_heads,
            self.head_dimension,
        ).transpose(1, 2)
        value = self.value_projection(x).view(
            batch_size,
            sequence_length,
            self.number_of_kv_heads,
            self.head_dimension,
        ).transpose(1, 2)

        query = apply_rope(query, self.rope_cosine, self.rope_sine)
        key = apply_rope(key, self.rope_cosine, self.rope_sine)
        output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=0.0,
            is_causal=True,
            enable_gqa=True,
        )
        output = output.transpose(1, 2).contiguous().view(
            batch_size, sequence_length, -1
        )
        return self.output_projection(output)


class SwiGLU(nn.Module):
    def __init__(self, config: RMW3Config):
        super().__init__()
        self.gate_projection = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.up_projection = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.down_projection = nn.Linear(config.d_ff, config.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_projection(
            F.silu(self.gate_projection(x)) * self.up_projection(x)
        )


class RMW3Block(nn.Module):
    def __init__(self, config: RMW3Config):
        super().__init__()
        self.attention_norm = RMSNorm(config.d_model, config.norm_eps)
        self.attention = GroupedQueryAttention(config)
        self.ffn_norm = RMSNorm(config.d_model, config.norm_eps)
        self.ffn = SwiGLU(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.attention_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x


class ReflectUnit(nn.Module):
    def __init__(self, config: RMW3Config):
        super().__init__()
        self.down_projection = nn.Linear(
            config.d_model, config.reflect_rank, bias=False
        )
        self.up_projection = nn.Linear(
            config.reflect_rank, config.d_model, bias=False
        )
        self.gate_logit = nn.Parameter(torch.tensor(-2.1972246, dtype=torch.float32))

    def forward(
        self,
        current: torch.Tensor,
        previous: torch.Tensor,
    ) -> torch.Tensor:
        discrepancy = current - previous
        correction = self.up_projection(
            F.silu(self.down_projection(discrepancy))
        )
        gate = torch.sigmoid(self.gate_logit).to(current.dtype)
        return current + gate * correction


class MorphExpert(nn.Module):
    def __init__(self, config: RMW3Config):
        super().__init__()
        self.down_projection = nn.Linear(
            config.d_model, config.morph_rank, bias=False
        )
        self.up_projection = nn.Linear(
            config.morph_rank, config.d_model, bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up_projection(F.silu(self.down_projection(x)))


class SparseMorphStage(nn.Module):
    def __init__(self, config: RMW3Config):
        super().__init__()
        self.number_of_experts = config.morph_experts
        self.router = nn.Linear(
            config.d_model, config.morph_experts, bias=False
        )
        self.experts = nn.ModuleList(
            [MorphExpert(config) for _ in range(config.morph_experts)]
        )
        self.scale_logit = nn.Parameter(torch.tensor(-4.0, dtype=torch.float32))
        self.last_loads: Optional[torch.Tensor] = None
        self.last_entropy: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        original_shape = x.shape
        flat = x.reshape(-1, original_shape[-1])
        probabilities = torch.softmax(self.router(flat).float(), dim=-1)
        choices = probabilities.argmax(dim=-1)
        output = torch.zeros_like(flat)
        loads = []

        for expert_index, expert in enumerate(self.experts):
            selected_mask = choices == expert_index
            loads.append(selected_mask.float().mean())
            indices = torch.nonzero(selected_mask, as_tuple=False).flatten()
            if indices.numel() == 0:
                continue

            selected_input = flat.index_select(0, indices)
            expert_output = expert(selected_input).to(output.dtype)
            selected_probability = probabilities.index_select(0, indices)[
                :, expert_index
            ].to(output.dtype)
            straight_through = (
                1.0 + selected_probability - selected_probability.detach()
            )
            output.index_copy_(
                0,
                indices,
                expert_output * straight_through.unsqueeze(-1),
            )

        load_tensor = torch.stack(loads)
        importance = probabilities.mean(dim=0)
        balance_loss = self.number_of_experts * torch.sum(
            load_tensor * importance
        )
        entropy = -(
            probabilities * probabilities.clamp_min(1e-9).log()
        ).sum(dim=-1).mean()
        self.last_loads = load_tensor.detach()
        self.last_entropy = entropy.detach()
        scale = torch.sigmoid(self.scale_logit).to(output.dtype)
        return x + scale * output.view(original_shape), balance_loss


class RMW3(nn.Module):
    def __init__(self, config: Optional[RMW3Config] = None):
        super().__init__()
        self.config = config if config is not None else RMW3Config()
        config = self.config

        self.token_embeddings = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList(
            [RMW3Block(config) for _ in range(config.n_physical_layers)]
        )
        self.reflect_units = nn.ModuleList(
            [ReflectUnit(config) for _ in range(config.n_physical_layers)]
        )
        self.morph_stages = nn.ModuleList(
            [SparseMorphStage(config) for _ in range(config.morph_stages)]
        )
        self.final_norm = RMSNorm(config.d_model, config.norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.apply(self._initialize_weights)
        self.lm_head.weight = self.token_embeddings.weight
        self._scale_residual_outputs()

    @staticmethod
    def _initialize_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _scale_residual_outputs(self) -> None:
        residual_scale = 1.0 / math.sqrt(
            2
            * self.config.n_physical_layers
            * self.config.recurrent_passes
        )
        with torch.no_grad():
            for block in self.blocks:
                block.attention.output_projection.weight.mul_(residual_scale)
                block.ffn.down_projection.weight.mul_(residual_scale)
            for reflect_unit in self.reflect_units:
                reflect_unit.up_projection.weight.mul_(residual_scale)
            for stage in self.morph_stages:
                for expert in stage.experts:
                    expert.up_projection.weight.mul_(residual_scale)

    def forward(
        self,
        input_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        if input_ids.shape[1] > self.config.max_seq_len:
            raise ValueError(
                f"Sequence length {input_ids.shape[1]} exceeds "
                f"the limit {self.config.max_seq_len}"
            )

        x = self.token_embeddings(input_ids)
        first_pass_states = []
        balance_losses = []
        effective_layer_index = 0
        total_effective_layers = (
            self.config.n_physical_layers * self.config.recurrent_passes
        )

        for recurrent_pass in range(self.config.recurrent_passes):
            for layer_index, block in enumerate(self.blocks):
                if recurrent_pass == 1:
                    x = self.reflect_units[layer_index](
                        x, first_pass_states[layer_index]
                    )

                x = block(x)
                if recurrent_pass == 0:
                    first_pass_states.append(x)

                current_stage = min(
                    self.config.morph_stages - 1,
                    effective_layer_index
                    * self.config.morph_stages
                    // total_effective_layers,
                )
                next_stage = min(
                    self.config.morph_stages - 1,
                    (effective_layer_index + 1)
                    * self.config.morph_stages
                    // total_effective_layers,
                )
                stage_finished = (
                    next_stage != current_stage
                    or effective_layer_index == total_effective_layers - 1
                )
                if stage_finished:
                    x, balance_loss = self.morph_stages[current_stage](x)
                    balance_losses.append(balance_loss)

                effective_layer_index += 1

        logits = self.lm_head(self.final_norm(x))
        auxiliary_loss = (
            torch.stack(balance_losses).mean()
            if balance_losses
            else logits.new_zeros(())
        )
        return logits, auxiliary_loss


RMW3_CONFIG = RMW3Config()


def count_parameters(model: nn.Module) -> int:
    """Count unique trainable parameters, including tied embeddings once."""
    return sum(parameter.numel() for parameter in model.parameters())


__all__ = ["RMW3", "RMW3Config", "RMW3_CONFIG", "count_parameters"]
