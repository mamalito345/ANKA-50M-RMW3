"""Tests for optimizer grouping, scheduling, and gradient accumulation."""

import itertools
import unittest

import torch

from anka.model import ANKAConfig, ANKAForCausalLM
from anka.training import TrainConfig, TrainingEngine, build_adamw


def tiny_model() -> ANKAForCausalLM:
    config = ANKAConfig(
        vocab_size=128,
        max_seq_len=32,
        n_layers=2,
        d_model=32,
        n_heads=4,
        n_kv_heads=2,
        d_ff=64,
    )
    return ANKAForCausalLM(config)


class TrainConfigTests(unittest.TestCase):
    def test_effective_batch_math(self) -> None:
        config = TrainConfig(
            total_steps=10,
            warmup_steps=2,
            global_batch_tokens=64,
            micro_batch_size=2,
            sequence_length=8,
        )
        self.assertEqual(config.tokens_per_micro_step, 16)
        self.assertEqual(config.gradient_accumulation_steps, 4)

    def test_invalid_effective_batch_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            TrainConfig(
                total_steps=10,
                warmup_steps=2,
                global_batch_tokens=65,
                micro_batch_size=2,
                sequence_length=8,
            )

    def test_warmup_and_cosine_endpoints(self) -> None:
        config = TrainConfig(
            total_steps=10,
            warmup_steps=2,
            global_batch_tokens=64,
            micro_batch_size=2,
            sequence_length=8,
            max_learning_rate=1e-3,
            min_learning_rate=1e-4,
        )
        self.assertAlmostEqual(config.learning_rate_at(0), 5e-4)
        self.assertAlmostEqual(config.learning_rate_at(1), 1e-3)
        self.assertAlmostEqual(config.learning_rate_at(2), 1e-3)
        self.assertAlmostEqual(config.learning_rate_at(9), 1e-4)


class OptimizerTests(unittest.TestCase):
    def test_parameter_groups_are_complete_and_disjoint(self) -> None:
        model = tiny_model()
        config = TrainConfig(
            total_steps=4,
            warmup_steps=1,
            global_batch_tokens=32,
            micro_batch_size=2,
            sequence_length=8,
            use_fused_adamw=False,
        )
        _, summary = build_adamw(model, config, torch.device("cpu"))

        self.assertEqual(
            summary.decay_parameters + summary.no_decay_parameters,
            model.num_trainable_parameters(),
        )
        self.assertTrue(set(summary.decay_names).isdisjoint(summary.no_decay_names))
        self.assertTrue(all("norm" in name for name in summary.no_decay_names))
        self.assertFalse(summary.fused)

    def test_one_accumulated_step_updates_the_model(self) -> None:
        torch.manual_seed(17)
        model = tiny_model()
        config = TrainConfig(
            total_steps=4,
            warmup_steps=1,
            global_batch_tokens=32,
            micro_batch_size=2,
            sequence_length=8,
            precision="fp32",
            use_fused_adamw=False,
        )
        engine = TrainingEngine(model, config)
        tokens = torch.randint(0, model.config.vocab_size, (2, 8))
        batches = itertools.repeat(tokens)
        before = model.token_embedding.weight.detach().clone()

        metrics = engine.train_step(batches)

        self.assertTrue(torch.isfinite(torch.tensor(metrics.loss)))
        self.assertEqual(metrics.step_tokens, 32)
        self.assertEqual(metrics.tokens_seen, 32)
        self.assertEqual(metrics.optimizer_step, 1)
        self.assertFalse(metrics.update_skipped)
        self.assertFalse(torch.equal(before, model.token_embedding.weight.detach()))

    def test_training_cannot_silently_exceed_total_steps(self) -> None:
        model = tiny_model()
        config = TrainConfig(
            total_steps=1,
            warmup_steps=0,
            global_batch_tokens=16,
            micro_batch_size=2,
            sequence_length=8,
            precision="fp32",
            use_fused_adamw=False,
        )
        engine = TrainingEngine(model, config)
        tokens = torch.randint(0, model.config.vocab_size, (2, 8))
        batches = itertools.repeat(tokens)

        engine.train_step(batches)
        with self.assertRaises(RuntimeError):
            engine.train_step(batches)


if __name__ == "__main__":
    unittest.main()
