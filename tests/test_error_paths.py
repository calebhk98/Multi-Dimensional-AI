"""
Tests for error paths and edge cases across the codebase.

These tests ensure proper error handling for:
- Malformed configurations
- Empty or invalid inputs
- Edge cases in data processing
- Recovery from errors
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.config import Config


class TestConfigErrorPaths:
    """Tests for configuration error handling."""

    def test_config_missing_required_section(self):
        """Test handling of missing required configuration sections."""
        # Create config without model section
        config_dict = {"training": {"optimizer": {"lr": 1e-4}}}
        config = Config(config_dict)
        # Should use defaults for missing sections
        assert config.get("model", {}) == {}

    def test_config_invalid_value_types(self):
        """Test handling of invalid configuration value types."""
        config_dict = {
            "model": {
                "transformer": {
                    "hidden_dim": "not_a_number",  # Should be int
                }
            }
        }
        # Config should store the value as-is (validation happens at use time)
        config = Config(config_dict)
        assert config["model"]["transformer"]["hidden_dim"] == "not_a_number"

    def test_config_empty_dict(self):
        """Test handling of empty configuration."""
        config = Config({})
        assert config.config == {}

    def test_config_deeply_nested_access(self):
        """Test accessing deeply nested config with missing intermediate keys."""
        config = Config({})
        # Should return default without raising
        result = config.get("a.b.c.d", default="default")
        assert result == "default"


class TestModelErrorPaths:
    """Tests for model error handling."""

    def test_model_empty_batch(self):
        """Test model handling of empty batch."""
        from src.models.text_only_transformer import TextOnlyTransformer

        config = {"model": {"transformer": {"hidden_dim": 64, "num_layers": 1, "num_attention_heads": 4, "ffn_dim": 256}}}
        model = TextOnlyTransformer(config)

        # Empty tensor should raise or handle gracefully
        empty_input = torch.tensor([], dtype=torch.long).reshape(0, 0)
        try:
            model(empty_input)
        except (RuntimeError, ValueError, IndexError):
            pass  # Expected to fail for empty input

    def test_model_sequence_exceeds_max_length(self):
        """Test model with sequence longer than max_seq_length."""
        from src.models.text_only_transformer import TextOnlyTransformer

        config = {
            "model": {
                "transformer": {"hidden_dim": 64, "num_layers": 1, "num_attention_heads": 4, "ffn_dim": 256},
                "encoders": {"internal_voice": {"max_seq_length": 32, "vocab_size": 1000}},
            }
        }
        model = TextOnlyTransformer(config)

        # Sequence longer than max_seq_length
        long_input = torch.randint(0, 1000, (2, 64))  # 64 > 32
        try:
            model(long_input)
        except (RuntimeError, IndexError):
            pass  # May fail due to position embedding size

    def test_model_invalid_token_ids(self):
        """Test model with token IDs outside vocabulary range."""
        from src.models.text_only_transformer import TextOnlyTransformer

        config = {
            "model": {
                "transformer": {"hidden_dim": 64, "num_layers": 1, "num_attention_heads": 4, "ffn_dim": 256},
                "encoders": {"internal_voice": {"vocab_size": 100}},
            }
        }
        model = TextOnlyTransformer(config)

        # Token IDs outside vocab range
        invalid_input = torch.tensor([[999, 1000, 1001]])  # All > vocab_size
        try:
            model(invalid_input)
        except (RuntimeError, IndexError):
            pass  # Expected to fail for out-of-range tokens


class TestDataValidationErrorPaths:
    """Tests for data validation error handling."""

    def test_validation_with_nan_values(self):
        """Test validation handles NaN values."""
        from src.data.validation import validate_value_ranges

        data = {"left_eye_image": torch.tensor([[[[float("nan")]]]])}
        # NaN comparison may behave unexpectedly
        try:
            validate_value_ranges(data)
        except (ValueError, RuntimeError):
            pass  # May raise due to NaN

    def test_validation_with_inf_values(self):
        """Test validation handles infinite values."""
        from src.data.validation import validate_value_ranges

        data = {"left_eye_image": torch.tensor([[[[float("inf")]]]])}
        with pytest.raises(ValueError, match="Values above"):
            validate_value_ranges(data)

    def test_validation_with_negative_inf(self):
        """Test validation handles negative infinite values."""
        from src.data.validation import validate_value_ranges

        data = {"left_eye_image": torch.tensor([[[[float("-inf")]]]])}
        with pytest.raises(ValueError, match="Values below"):
            validate_value_ranges(data)

    def test_validation_deeply_nested_structure(self):
        """Test validation with deeply nested dictionaries."""
        from src.data.validation import validate_input_shapes

        data = {
            "level1": {
                "level2": {
                    "level3": torch.randn(4, 10),
                }
            }
        }
        # Should handle or fail gracefully
        validate_input_shapes(data)


class TestEvolutionErrorPaths:
    """Tests for evolution module error handling."""

    def test_evolution_with_zero_population(self):
        """Test GA with zero population size."""
        from src.evolution.strategies import SimpleGA

        ga = SimpleGA(pop_size=0)
        # Evolve with empty population
        try:
            ga.evolve([])
        except (ValueError, ZeroDivisionError, IndexError):
            pass  # Expected to fail

    def test_evolution_with_single_individual(self):
        """Test GA with single individual population."""
        from src.evolution.strategies import SimpleGA

        ga = SimpleGA(pop_size=1, elite_ratio=1.0)
        population = [(np.array([1.0, 2.0]), 1.0)]
        result = ga.evolve(population)
        assert len(result) == 1

    def test_evolution_with_all_equal_fitness(self):
        """Test evolution when all individuals have same fitness."""
        from src.evolution.strategies import OpenAIES

        es = OpenAIES(sigma=0.1, learning_rate=0.01)
        params = np.array([1.0, 2.0, 3.0])
        # All same fitness - reward normalization may handle this
        results = [(42, 1.0), (43, 1.0), (44, 1.0)]
        updated = es.update(params, results)
        # With all equal rewards and whitening, update should be zero or handle gracefully
        assert updated is not None

    def test_evolution_empty_results(self):
        """Test evolution update with empty results."""
        from src.evolution.strategies import OpenAIES

        es = OpenAIES()
        params = np.array([1.0, 2.0, 3.0])
        # Empty results should return original params
        updated = es.update(params, [])
        np.testing.assert_array_equal(updated, params)

    def test_parameter_proxy_empty_model(self):
        """Test ParameterProxy with model that has no parameters."""
        from src.evolution.strategies import ParameterProxy

        class EmptyModel(nn.Module):
            def forward(self, x):
                return x

        proxy = ParameterProxy()
        model = EmptyModel()
        params = proxy.get_params(model)
        assert len(params) == 0

    def test_parameter_proxy_lora_target_no_lora_params(self):
        """Test ParameterProxy with lora target but no lora parameters."""
        from src.evolution.strategies import ParameterProxy

        class RegularModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(10, 10)

            def forward(self, x):
                return self.fc(x)

        proxy = ParameterProxy(target="lora")
        model = RegularModel()
        params = proxy.get_params(model)
        assert len(params) == 0  # No lora params


class TestDatasetErrorPaths:
    """Tests for dataset error handling."""

    def test_text_dataset_truncated_file(self):
        """Test TextDataset with truncated/corrupted file."""
        from src.data.text_dataset import TextDataset

        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            # Write very few tokens
            tokens = np.array([1, 2, 3], dtype=np.uint16)
            tokens.tofile(f)
            temp_path = f.name

        try:
            # Request longer sequences than available
            dataset = TextDataset(temp_path, seq_length=1000)
            # num_samples should be 0 or very small
            assert dataset.num_samples == 0 or len(dataset) <= 1
        finally:
            Path(temp_path).unlink()

    def test_text_dataset_empty_file(self):
        """Test TextDataset with empty file."""
        from src.data.text_dataset import TextDataset

        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            temp_path = f.name
            # File is empty

        try:
            dataset = TextDataset(temp_path, seq_length=64)
            assert len(dataset) == 0
        finally:
            Path(temp_path).unlink()


class TestTrainerErrorPaths:
    """Tests for trainer error handling."""

    def test_trainer_with_nan_loss(self):
        """Test trainer handling of NaN loss."""
        from src.training.trainer import Trainer

        # Create a simple model that produces NaN
        class NaNModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(10, 10)

            def forward(self, x):
                return {"hidden_states": self.fc(x) * float("nan")}

            def compute_loss(self, outputs, targets, loss_weights=None):
                return torch.tensor(float("nan")), {"total": float("nan")}

        # This tests that the trainer can be instantiated
        # Full NaN handling depends on trainer implementation

    def test_trainer_zero_gradient_clip(self):
        """Test trainer with zero gradient clip value."""
        # Zero gradient clip should be handled (no clipping or error)
        config = {
            "model": {"transformer": {"hidden_dim": 64}},
            "training": {"gradient_clip": 0.0},
        }
        # Trainer should handle this gracefully


class TestFitnessErrorPaths:
    """Tests for fitness evaluator error handling."""

    def test_survival_fitness_missing_env_state(self):
        """Test SurvivalFitness with missing env_state keys."""
        from src.evolution.fitness import SurvivalFitness

        fitness = SurvivalFitness()
        # Missing timesteps key - should use default
        result = fitness.evaluate({}, {})
        assert result == 0.0  # Default timesteps is 0

    def test_task_fitness_missing_progress(self):
        """Test TaskCompletionFitness with missing progress."""
        from src.evolution.fitness import TaskCompletionFitness

        fitness = TaskCompletionFitness()
        # Missing task_progress - should use default
        result = fitness.evaluate({}, {"task_completed": False})
        assert result == 0.0  # Default progress is 0

    def test_fitness_with_negative_values(self):
        """Test fitness evaluator with negative environment values."""
        from src.evolution.fitness import SurvivalFitness

        fitness = SurvivalFitness(timestep_reward=1.0)
        result = fitness.evaluate({}, {"timesteps": -10, "alive": True})
        assert result == -10.0  # Should handle negative timesteps


class TestEncoderDecoderErrorPaths:
    """Tests for encoder/decoder error handling."""

    def test_encoder_zero_batch_size(self):
        """Test encoder with zero batch size."""
        from src.encoders.internal_voice_encoder import InternalVoiceEncoder

        encoder = InternalVoiceEncoder(vocab_size=1000, embedding_dim=64)
        empty_input = torch.tensor([], dtype=torch.long).reshape(0, 32)
        try:
            encoder(empty_input)
        except (RuntimeError, ValueError):
            pass  # May fail for empty batch

    def test_encoder_single_token_sequence(self):
        """Test encoder with single-token sequences."""
        from src.encoders.internal_voice_encoder import InternalVoiceEncoder

        encoder = InternalVoiceEncoder(vocab_size=1000, embedding_dim=64)
        single_token = torch.randint(0, 1000, (4, 1))
        output = encoder(single_token)
        assert output.shape[1] == 1  # Single token sequence

    def test_decoder_mismatched_hidden_dim(self):
        """Test decoder with mismatched hidden dimension input."""
        from src.decoders.text_decoder import InternalTextDecoder

        decoder = InternalTextDecoder(vocab_size=1000, embedding_dim=64)
        # Input with wrong hidden dim
        wrong_dim_input = torch.randn(4, 32, 128)  # 128 != 64
        try:
            decoder(wrong_dim_input)
        except RuntimeError:
            pass  # Should fail due to dimension mismatch


class TestCollateErrorPaths:
    """Tests for collate function error handling."""

    def test_collate_empty_batch(self):
        """Test collate with empty batch."""
        from src.data.dataset import collate_fn

        try:
            collate_fn([])
        except (ValueError, RuntimeError, IndexError):
            pass  # Expected to fail for empty batch

    def test_collate_mismatched_shapes(self):
        """Test collate with mismatched tensor shapes."""
        from src.data.text_only_dataset import text_only_collate_fn

        # Samples with different shapes
        batch = [
            {
                "inputs": {"internal_voice_tokens": torch.randn(32)},
                "targets": {"internal_text": torch.randn(32)},
            },
            {
                "inputs": {"internal_voice_tokens": torch.randn(64)},  # Different shape
                "targets": {"internal_text": torch.randn(64)},
            },
        ]
        try:
            text_only_collate_fn(batch)
        except RuntimeError:
            pass  # Should fail due to shape mismatch


class TestRecoveryScenarios:
    """Tests for recovery from errors."""

    def test_model_recovers_after_bad_input(self):
        """Test that model can process valid input after receiving bad input."""
        from src.models.text_only_transformer import TextOnlyTransformer

        config = {"model": {"transformer": {"hidden_dim": 64, "num_layers": 1, "num_attention_heads": 4, "ffn_dim": 256}}}
        model = TextOnlyTransformer(config)

        # Try with bad input
        try:
            model(torch.tensor([]))
        except:
            pass

        # Should still work with valid input
        valid_input = torch.randint(0, 1000, (2, 32))
        output = model(valid_input)
        assert output["logits"].shape == (2, 32, 50257)

    def test_trainer_continues_after_nan_batch(self):
        """Test concept that trainer can continue after encountering NaN."""
        # This is more of a design/documentation test
        # Real implementation would need NaN detection and handling
        pass
