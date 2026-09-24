"""Correctness tests for the ANKA-50M architecture."""

import unittest

import torch

from anka.model import (
    ANKABlock,
    ANKAConfig,
    ANKAForCausalLM,
    GroupedQueryAttention,
    RMSNorm,
    RotaryEmbedding,
    SwiGLU,
)


class ConfigAndNormTests(unittest.TestCase):
    def test_default_config_has_the_locked_parameter_budget(self) -> None:
        config = ANKAConfig()
        self.assertEqual(config.head_dim, 64)
        self.assertEqual(config.kv_dim, 128)
        self.assertEqual(config.expected_parameter_count(), 49_960_320)

    def test_parameter_free_rmsnorm_has_no_trainable_parameters(self) -> None:
        norm = RMSNorm(dim=384, affine=False)
        self.assertEqual(sum(parameter.numel() for parameter in norm.parameters()), 0)

    def test_baseline_block_has_two_learned_norm_scales(self) -> None:
        config = ANKAConfig(
            vocab_size=128,
            max_seq_len=32,
            n_layers=2,
            d_model=32,
            n_heads=4,
            n_kv_heads=2,
            d_ff=64,
        )
        block = ANKABlock(config)
        self.assertEqual(block.attn_norm.weight.numel(), config.d_model)
        self.assertEqual(block.ffn_norm.weight.numel(), config.d_model)

    def test_rmsnorm_shape_dtype_and_scale(self) -> None:
        norm = RMSNorm(dim=8, affine=False)
        x = torch.randn(2, 5, 8, dtype=torch.float32)
        y = norm(x)

        self.assertEqual(y.shape, x.shape)
        self.assertEqual(y.dtype, x.dtype)
        expected_rms = torch.ones_like(y[..., 0])
        actual_rms = y.pow(2).mean(dim=-1).sqrt()
        self.assertTrue(
            torch.allclose(actual_rms, expected_rms, atol=2e-5, rtol=2e-5)
        )


class PositionAndAttentionTests(unittest.TestCase):
    def test_rope_preserves_shape_and_vector_norm(self) -> None:
        rope = RotaryEmbedding(dim=8, max_seq_len=16)
        query = torch.randn(2, 3, 7, 8)
        key = torch.randn(2, 1, 7, 8)
        rotated_query, rotated_key = rope(query, key)

        self.assertEqual(rotated_query.shape, query.shape)
        self.assertEqual(rotated_key.shape, key.shape)
        self.assertTrue(
            torch.allclose(
                rotated_query.norm(dim=-1), query.norm(dim=-1), atol=1e-5
            )
        )

    def test_gqa_output_shape(self) -> None:
        config = ANKAConfig(
            vocab_size=128,
            max_seq_len=32,
            n_layers=2,
            d_model=32,
            n_heads=4,
            n_kv_heads=2,
            d_ff=64,
        )
        attention = GroupedQueryAttention(config).eval()
        x = torch.randn(2, 11, 32)
        y = attention(x)
        self.assertEqual(y.shape, x.shape)


class FeedForwardAndBlockTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = ANKAConfig(
            vocab_size=128,
            max_seq_len=32,
            n_layers=2,
            d_model=32,
            n_heads=4,
            n_kv_heads=2,
            d_ff=64,
        )

    def test_swiglu_output_shape(self) -> None:
        ffn = SwiGLU(self.config)
        x = torch.randn(2, 9, self.config.d_model)
        self.assertEqual(ffn(x).shape, x.shape)

    def test_block_output_shape(self) -> None:
        block = ANKABlock(self.config).eval()
        x = torch.randn(2, 9, self.config.d_model)
        self.assertEqual(block(x).shape, x.shape)


class FullModelTests(unittest.TestCase):
    @staticmethod
    def tiny_config() -> ANKAConfig:
        return ANKAConfig(
            vocab_size=128,
            max_seq_len=32,
            n_layers=2,
            d_model=32,
            n_heads=4,
            n_kv_heads=2,
            d_ff=64,
        )

    def test_full_anka_parameter_count_is_exact(self) -> None:
        model = ANKAForCausalLM(ANKAConfig())
        self.assertEqual(model.num_trainable_parameters(), 49_960_320)
        self.assertLessEqual(model.num_trainable_parameters(), 50_000_000)

    def test_forward_loss_and_backward(self) -> None:
        torch.manual_seed(7)
        config = self.tiny_config()
        model = ANKAForCausalLM(config)
        tokens = torch.randint(0, config.vocab_size, (2, 13))

        output = model(tokens, targets=tokens)
        self.assertEqual(output.logits.shape, (2, 13, config.vocab_size))
        self.assertIsNotNone(output.loss)
        self.assertTrue(torch.isfinite(output.loss))

        output.loss.backward()
        self.assertIsNotNone(model.token_embedding.weight.grad)
        self.assertTrue(torch.isfinite(model.token_embedding.weight.grad).all())

    def test_causal_mask_blocks_future_information(self) -> None:
        torch.manual_seed(11)
        config = self.tiny_config()
        model = ANKAForCausalLM(config).eval()

        prefix = torch.tensor([[1, 2, 3, 4, 5]])
        suffix_a = torch.tensor([[6, 7, 8]])
        suffix_b = torch.tensor([[20, 21, 22]])
        tokens_a = torch.cat((prefix, suffix_a), dim=1)
        tokens_b = torch.cat((prefix, suffix_b), dim=1)

        with torch.no_grad():
            logits_a = model(tokens_a).logits[:, : prefix.size(1)]
            logits_b = model(tokens_b).logits[:, : prefix.size(1)]

        self.assertTrue(torch.allclose(logits_a, logits_b, atol=1e-6, rtol=1e-6))

    def test_only_one_vocabulary_matrix_exists(self) -> None:
        config = self.tiny_config()
        model = ANKAForCausalLM(config)
        vocabulary_matrices = [
            name
            for name, parameter in model.named_parameters()
            if tuple(parameter.shape) == (config.vocab_size, config.d_model)
        ]
        self.assertEqual(vocabulary_matrices, ["token_embedding.weight"])


if __name__ == "__main__":
    unittest.main()
