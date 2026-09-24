"""ANKA-50M: a compact decoder-only causal language model.

The file is intentionally ordered as a teaching document:

1. Configuration and invariants
2. RMSNorm
3. Rotary position embeddings (RoPE)
4. Grouped-query causal self-attention (GQA)
5. SwiGLU feed-forward network
6. Pre-normalized Transformer block
7. Full tied-embedding language model

The baseline contains no ANKA-Reflect, ANKA-Morph, or ANKA-WorldPairs
mechanism. It is the control model against which those interventions are tested.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# 1. Configuration
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class ANKAConfig:
    """All architectural constants for the 49,960,320-parameter baseline."""

    vocab_size: int = 15_360
    max_seq_len: int = 1_024
    n_layers: int = 28
    d_model: int = 384
    n_heads: int = 6
    n_kv_heads: int = 2
    d_ff: int = 1_024
    rope_theta: float = 10_000.0
    rms_norm_eps: float = 1e-6
    dropout: float = 0.0
    init_std: float = 0.02

    def __post_init__(self) -> None:
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if self.max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")
        if self.n_layers <= 0:
            raise ValueError("n_layers must be positive")
        if self.d_model <= 0:
            raise ValueError("d_model must be positive")
        if self.n_heads <= 0:
            raise ValueError("n_heads must be positive")
        if self.n_kv_heads <= 0:
            raise ValueError("n_kv_heads must be positive")
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        if self.head_dim % 2 != 0:
            raise ValueError("RoPE requires an even head_dim")
        if self.d_ff <= 0:
            raise ValueError("d_ff must be positive")
        if self.rope_theta <= 0.0:
            raise ValueError("rope_theta must be positive")
        if self.rms_norm_eps <= 0.0:
            raise ValueError("rms_norm_eps must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.init_std <= 0.0:
            raise ValueError("init_std must be positive")

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @property
    def kv_dim(self) -> int:
        return self.n_kv_heads * self.head_dim

    @property
    def queries_per_kv(self) -> int:
        return self.n_heads // self.n_kv_heads

    def expected_parameter_count(self) -> int:
        """Analytical count for this exact baseline implementation.

        Every RMSNorm has one learnable scale vector. Input embeddings and
        output logits share the same matrix, so the vocabulary matrix is
        counted once.
        """

        embedding = self.vocab_size * self.d_model

        attention_per_layer = (
            self.d_model * self.d_model  # q_proj
            + self.d_model * self.kv_dim  # k_proj
            + self.d_model * self.kv_dim  # v_proj
            + self.d_model * self.d_model  # o_proj
        )
        swiglu_per_layer = 3 * self.d_model * self.d_ff
        norms_per_layer = 2 * self.d_model
        transformer_stack = self.n_layers * (
            attention_per_layer + swiglu_per_layer + norms_per_layer
        )
        final_norm = self.d_model

        return embedding + transformer_stack + final_norm


# -----------------------------------------------------------------------------
# 2. RMSNorm
# -----------------------------------------------------------------------------


class RMSNorm(nn.Module):
    """Root-mean-square normalization with optional learned scaling.

    ANKA uses affine=True in every block and in the final normalization. The
    affine=False option remains useful for later controlled ablations.
    """

    def __init__(self, dim: int, eps: float = 1e-6, affine: bool = False) -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight: Optional[nn.Parameter]
        if affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_parameter("weight", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Accumulate the variance in float32. This is safer for BF16/FP16,
        # while the returned tensor keeps the input dtype.
        input_dtype = x.dtype
        x_float = x.float()
        normalized = x_float * torch.rsqrt(
            x_float.pow(2).mean(dim=-1, keepdim=True) + self.eps
        )
        normalized = normalized.to(dtype=input_dtype)
        if self.weight is not None:
            normalized = normalized * self.weight.to(dtype=input_dtype)
        return normalized


# -----------------------------------------------------------------------------
# 3. Rotary position embeddings (RoPE)
# -----------------------------------------------------------------------------


class RotaryEmbedding(nn.Module):
    """Precomputed rotary angles for query and key vectors.

    RoPE changes activations, not parameter count. Sine/cosine tables are
    registered as non-persistent buffers, so they move with the model but are
    regenerated instead of being stored in checkpoints.
    """

    def __init__(self, dim: int, max_seq_len: int, theta: float = 10_000.0) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("RotaryEmbedding requires an even dimension")

        inv_freq = 1.0 / (
            theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
        )
        positions = torch.arange(max_seq_len, dtype=torch.float32)
        angles = torch.outer(positions, inv_freq)

        self.dim = dim
        self.max_seq_len = max_seq_len
        self.register_buffer("cos_cached", angles.cos(), persistent=False)
        self.register_buffer("sin_cached", angles.sin(), persistent=False)

    @staticmethod
    def _rotate(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """Rotate adjacent feature pairs while accumulating in float32."""

        input_dtype = x.dtype
        x_pairs = x.float().reshape(*x.shape[:-1], -1, 2)
        x_even = x_pairs[..., 0]
        x_odd = x_pairs[..., 1]

        rotated_even = x_even * cos - x_odd * sin
        rotated_odd = x_even * sin + x_odd * cos
        rotated = torch.stack((rotated_even, rotated_odd), dim=-1)
        return rotated.flatten(start_dim=-2).to(dtype=input_dtype)

    def forward(
        self, query: torch.Tensor, key: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply RoPE to tensors shaped [batch, heads, sequence, head_dim]."""

        seq_len = query.size(-2)
        if key.size(-2) != seq_len:
            raise ValueError("query and key sequence lengths must match")
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"sequence length {seq_len} exceeds RoPE cache {self.max_seq_len}"
            )

        # [1, 1, sequence, head_dim / 2], broadcast across batch and heads.
        cos = self.cos_cached[:seq_len].view(1, 1, seq_len, -1)
        sin = self.sin_cached[:seq_len].view(1, 1, seq_len, -1)
        return self._rotate(query, cos, sin), self._rotate(key, cos, sin)


# -----------------------------------------------------------------------------
# 4. Grouped-query causal self-attention
# -----------------------------------------------------------------------------


class GroupedQueryAttention(nn.Module):
    """Causal self-attention with six query heads and two KV heads by default."""

    def __init__(self, config: ANKAConfig) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.head_dim
        self.queries_per_kv = config.queries_per_kv
        self.dropout = config.dropout

        # No projection uses a bias; every parameter therefore contributes to
        # a learned direction instead of a per-feature offset.
        self.q_proj = nn.Linear(
            config.d_model, config.n_heads * config.head_dim, bias=False
        )
        self.k_proj = nn.Linear(config.d_model, config.kv_dim, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.kv_dim, bias=False)
        self.o_proj = nn.Linear(
            config.n_heads * config.head_dim, config.d_model, bias=False
        )
        self.rope = RotaryEmbedding(
            dim=config.head_dim,
            max_seq_len=config.max_seq_len,
            theta=config.rope_theta,
        )

    def _split_heads(
        self, x: torch.Tensor, n_heads: int
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        return x.view(batch_size, seq_len, n_heads, self.head_dim).transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        query = self._split_heads(self.q_proj(x), self.n_heads)
        key = self._split_heads(self.k_proj(x), self.n_kv_heads)
        value = self._split_heads(self.v_proj(x), self.n_kv_heads)

        query, key = self.rope(query, key)

        # PyTorch versions differ in native GQA kernel support. Explicitly
        # repeating only the activations keeps the architecture and parameter
        # count identical and provides a dependable correctness baseline.
        key = key.repeat_interleave(self.queries_per_kv, dim=1)
        value = value.repeat_interleave(self.queries_per_kv, dim=1)

        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )

        attended = attended.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.n_heads * self.head_dim
        )
        return self.o_proj(attended)


# -----------------------------------------------------------------------------
# 5. SwiGLU feed-forward network
# -----------------------------------------------------------------------------


class SwiGLU(nn.Module):
    """Gated feed-forward network: down(silu(gate(x)) * up(x))."""

    def __init__(self, config: ANKAConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.up_proj = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.down_proj = nn.Linear(config.d_ff, config.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gated = F.silu(self.gate_proj(x)) * self.up_proj(x)
        return self.down_proj(gated)


# -----------------------------------------------------------------------------
# 6. Pre-normalized Transformer block
# -----------------------------------------------------------------------------


class ANKABlock(nn.Module):
    """One attention residual followed by one SwiGLU residual."""

    def __init__(self, config: ANKAConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(
            config.d_model, eps=config.rms_norm_eps, affine=True
        )
        self.attention = GroupedQueryAttention(config)
        self.ffn_norm = RMSNorm(
            config.d_model, eps=config.rms_norm_eps, affine=True
        )
        self.feed_forward = SwiGLU(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.attn_norm(x))
        x = x + self.feed_forward(self.ffn_norm(x))
        return x


# -----------------------------------------------------------------------------
# 7. Full causal language model
# -----------------------------------------------------------------------------


@dataclass
class ANKAOutput:
    """Output kept deliberately small for training and evaluation adapters."""

    logits: torch.Tensor
    loss: Optional[torch.Tensor] = None


class ANKAForCausalLM(nn.Module):
    """The complete ANKA-50M baseline with tied input/output embeddings."""

    PARAMETER_LIMIT = 50_000_000

    def __init__(self, config: Optional[ANKAConfig] = None) -> None:
        super().__init__()
        self.config = config if config is not None else ANKAConfig()

        self.token_embedding = nn.Embedding(
            self.config.vocab_size, self.config.d_model
        )
        self.layers = nn.ModuleList(
            ANKABlock(self.config) for _ in range(self.config.n_layers)
        )
        self.final_norm = RMSNorm(
            self.config.d_model,
            eps=self.config.rms_norm_eps,
            affine=True,
        )

        self.reset_parameters()
        self._verify_parameter_count()

    def reset_parameters(self) -> None:
        """Initialize ordinary and residual-output matrices separately."""

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=self.config.init_std)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=self.config.init_std)
            elif isinstance(module, RMSNorm) and module.weight is not None:
                nn.init.ones_(module.weight)

        # Each block adds two residual branches. Scaling their final projections
        # prevents the initial residual stream variance from growing with depth.
        residual_std = self.config.init_std / math.sqrt(2 * self.config.n_layers)
        for block in self.layers:
            nn.init.normal_(
                block.attention.o_proj.weight, mean=0.0, std=residual_std
            )
            nn.init.normal_(
                block.feed_forward.down_proj.weight, mean=0.0, std=residual_std
            )

    def num_trainable_parameters(self) -> int:
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )

    def _verify_parameter_count(self) -> None:
        actual = self.num_trainable_parameters()
        expected = self.config.expected_parameter_count()
        if actual != expected:
            raise RuntimeError(
                f"parameter-count mismatch: implementation={actual:,}, "
                f"analytical={expected:,}"
            )
        if actual > self.PARAMETER_LIMIT:
            raise RuntimeError(
                f"model has {actual:,} trainable parameters; "
                f"competition limit is {self.PARAMETER_LIMIT:,}"
            )

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Project with the input embedding matrix; no second vocab matrix."""

        return F.linear(hidden_states, self.token_embedding.weight)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> ANKAOutput:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        if input_ids.dtype != torch.long:
            raise TypeError("input_ids must use torch.long token IDs")
        if input_ids.size(1) > self.config.max_seq_len:
            raise ValueError(
                f"sequence length {input_ids.size(1)} exceeds "
                f"max_seq_len {self.config.max_seq_len}"
            )
        if targets is not None and targets.shape != input_ids.shape:
            raise ValueError("targets must have the same shape as input_ids")
        if targets is not None and targets.dtype != torch.long:
            raise TypeError("targets must use torch.long token IDs")

        hidden_states = self.token_embedding(input_ids)
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        hidden_states = self.final_norm(hidden_states)
        logits = self.compute_logits(hidden_states)

        loss = None
        if targets is not None:
            if input_ids.size(1) < 2:
                raise ValueError(
                    "at least two tokens are required to compute next-token loss"
                )
            # Position t predicts target t+1. A target value of -100 is ignored.
            shift_logits = logits[:, :-1, :].contiguous()
            shift_targets = targets[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_targets.view(-1),
                ignore_index=-100,
            )

        return ANKAOutput(logits=logits, loss=loss)
