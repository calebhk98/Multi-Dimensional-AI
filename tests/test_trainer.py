"""
Tests for training module (src/training/trainer.py).
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
from src.training.trainer import Trainer


class SimpleModel(nn.Module):
    """Simple model for testing."""

    def __init__(self, input_dim=10, hidden_dim=20, output_dim=5):
        super().__init__()
        self.fc = nn.Linear(input_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, output_dim)
        self.loss_fn = nn.MSELoss()

    def forward(self, **kwargs):
        """Forward pass that handles multi-modal inputs."""
        # For testing, just use internal_voice_tokens if provided
        x = kwargs.get('internal_voice_tokens', torch.randn(2, 10))
        if x.dim() > 2:
            x = x.view(x.size(0), -1)  # Flatten
        h = self.fc(x)
        out = self.output(h)
        return {"hidden_states": h, "output": out}

    def compute_loss(self, outputs, targets):
        """Compute loss for testing."""
        loss = self.loss_fn(outputs["output"], targets.get("internal_text", torch.zeros_like(outputs["output"])))
        loss_dict = {"total": loss, "mse": loss}
        return loss, loss_dict


class TestTrainerInitialization:
    """Tests for Trainer initialization."""

    def test_trainer_basic_initialization(self):
        """Test basic trainer initialization."""
        model = SimpleModel()
        config = {
            "training": {
                "optimizer": {"lr": 3e-4, "betas": [0.9, 0.95], "weight_decay": 0.01},
                "max_steps": 1000,
                "log_interval": 10,
                "save_interval": 100,
                "checkpointing": {"save_dir": "checkpoints"}
            }
        }

        # Create dummy data
        dataset = TensorDataset(torch.randn(10, 10), torch.randn(10, 5))
        train_loader = DataLoader(dataset, batch_size=2)

        trainer = Trainer(model, config, train_loader, device="cpu")

        assert trainer.model is not None
        assert trainer.config == config
        assert trainer.train_loader == train_loader
        assert trainer.device == "cpu"
        assert trainer.max_steps == 1000
        assert trainer.log_interval == 10
        assert trainer.save_interval == 100

    def test_trainer_with_validation_loader(self):
        """Test trainer initialization with validation loader."""
        model = SimpleModel()
        config = {"training": {}}

        dataset = TensorDataset(torch.randn(10, 10), torch.randn(10, 5))
        train_loader = DataLoader(dataset, batch_size=2)
        val_loader = DataLoader(dataset, batch_size=2)

        trainer = Trainer(model, config, train_loader, val_loader=val_loader, device="cpu")

        assert trainer.val_loader == val_loader

    def test_trainer_default_config_values(self):
        """Test that trainer uses default values for missing config."""
        model = SimpleModel()
        config = {"training": {}}  # Empty training config

        dataset = TensorDataset(torch.randn(10, 10), torch.randn(10, 5))
        train_loader = DataLoader(dataset, batch_size=2)

        trainer = Trainer(model, config, train_loader, device="cpu")

        assert trainer.max_steps == 1000
        assert trainer.log_interval == 10
        assert trainer.save_interval == 100

    def test_trainer_custom_optimizer_config(self):
        """Test trainer with custom optimizer configuration."""
        model = SimpleModel()
        config = {
            "training": {
                "optimizer": {
                    "lr": 1e-3,
                    "betas": [0.8, 0.999],
                    "weight_decay": 0.05
                }
            }
        }

        dataset = TensorDataset(torch.randn(10, 10), torch.randn(10, 5))
        train_loader = DataLoader(dataset, batch_size=2)

        trainer = Trainer(model, config, train_loader, device="cpu")

        # Check optimizer parameters
        assert trainer.optimizer.defaults['lr'] == 1e-3
        assert trainer.optimizer.defaults['betas'] == (0.8, 0.999)
        assert trainer.optimizer.defaults['weight_decay'] == 0.05

    def test_trainer_creates_save_directory(self, tmp_path):
        """Test that trainer creates checkpoint save directory."""
        model = SimpleModel()
        save_dir = tmp_path / "test_checkpoints"
        config = {
            "training": {
                "checkpointing": {"save_dir": str(save_dir)}
            }
        }

        dataset = TensorDataset(torch.randn(10, 10), torch.randn(10, 5))
        train_loader = DataLoader(dataset, batch_size=2)

        trainer = Trainer(model, config, train_loader, device="cpu")

        assert save_dir.exists()
        assert trainer.save_dir == save_dir

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_trainer_cuda_device(self):
        """Test trainer initialization with CUDA device."""
        model = SimpleModel()
        config = {"training": {}}

        dataset = TensorDataset(torch.randn(10, 10), torch.randn(10, 5))
        train_loader = DataLoader(dataset, batch_size=2)

        trainer = Trainer(model, config, train_loader, device="cuda")

        assert trainer.device == "cuda"
        assert next(trainer.model.parameters()).is_cuda


class TestTrainerTrainStep:
    """Tests for Trainer.train_step() method."""

    def test_train_step_basic(self):
        """Test basic training step execution."""
        model = SimpleModel()
        config = {"training": {}}

        dataset = TensorDataset(torch.randn(10, 10), torch.randn(10, 5))
        train_loader = DataLoader(dataset, batch_size=2)

        trainer = Trainer(model, config, train_loader, device="cpu")

        # Create batch matching expected format
        batch = {
            "inputs": {"internal_voice_tokens": torch.randn(2, 10)},
            "targets": {"internal_text": torch.randn(2, 5)}
        }

        loss, loss_dict = trainer.train_step(batch)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # Scalar
        assert isinstance(loss_dict, dict)
        assert "total" in loss_dict

    def test_train_step_backward_pass(self):
        """Test that train_step performs backward pass correctly."""
        model = SimpleModel()
        config = {"training": {}}

        dataset = TensorDataset(torch.randn(10, 10), torch.randn(10, 5))
        train_loader = DataLoader(dataset, batch_size=2)

        trainer = Trainer(model, config, train_loader, device="cpu")

        batch = {
            "inputs": {"internal_voice_tokens": torch.randn(2, 10)},
            "targets": {"internal_text": torch.randn(2, 5)}
        }

        # Check gradients are None before
        for param in model.parameters():
            assert param.grad is None or torch.all(param.grad == 0)

        loss, loss_dict = trainer.train_step(batch)

        # Check gradients exist after backward
        for param in model.parameters():
            assert param.grad is not None

    def test_train_step_optimizer_step(self):
        """Test that train_step updates model parameters."""
        model = SimpleModel()
        config = {"training": {}}

        dataset = TensorDataset(torch.randn(10, 10), torch.randn(10, 5))
        train_loader = DataLoader(dataset, batch_size=2)

        trainer = Trainer(model, config, train_loader, device="cpu")

        # Store initial parameters
        initial_params = [p.clone() for p in model.parameters()]

        batch = {
            "inputs": {"internal_voice_tokens": torch.randn(2, 10)},
            "targets": {"internal_text": torch.randn(2, 5)}
        }

        loss, loss_dict = trainer.train_step(batch)

        # Check that parameters have changed
        for initial, current in zip(initial_params, model.parameters()):
            assert not torch.equal(initial, current)

    def test_train_step_gradient_clipping(self):
        """Test that gradient clipping is applied."""
        model = SimpleModel()
        config = {"training": {}}

        dataset = TensorDataset(torch.randn(10, 10), torch.randn(10, 5))
        train_loader = DataLoader(dataset, batch_size=2)

        trainer = Trainer(model, config, train_loader, device="cpu")

        # Create batch that produces large gradients
        batch = {
            "inputs": {"internal_voice_tokens": torch.randn(2, 10) * 100},
            "targets": {"internal_text": torch.randn(2, 5) * 100}
        }

        with patch('torch.nn.utils.clip_grad_norm_') as mock_clip:
            loss, loss_dict = trainer.train_step(batch)
            mock_clip.assert_called_once()

    def test_train_step_with_nested_touch_data(self):
        """Test train_step with nested touch data structure."""
        model = SimpleModel()
        config = {"training": {}}

        dataset = TensorDataset(torch.randn(10, 10), torch.randn(10, 5))
        train_loader = DataLoader(dataset, batch_size=2)

        trainer = Trainer(model, config, train_loader, device="cpu")

        batch = {
            "inputs": {
                "internal_voice_tokens": torch.randn(2, 10),
                "touch_data": {
                    "positions": torch.randn(2, 5, 3),
                    "forces": torch.randn(2, 5, 3)
                }
            },
            "targets": {"internal_text": torch.randn(2, 5)}
        }

        loss, loss_dict = trainer.train_step(batch)
        assert isinstance(loss, torch.Tensor)

    def test_train_step_with_animation_targets(self):
        """Test train_step with nested animation targets."""
        model = SimpleModel()
        config = {"training": {}}

        dataset = TensorDataset(torch.randn(10, 10), torch.randn(10, 5))
        train_loader = DataLoader(dataset, batch_size=2)

        trainer = Trainer(model, config, train_loader, device="cpu")

        batch = {
            "inputs": {"internal_voice_tokens": torch.randn(2, 10)},
            "targets": {
                "animation": {
                    "joint_rotations": torch.randn(2, 24, 4),
                    "blend_shapes": torch.randn(2, 52)
                }
            }
        }

        loss, loss_dict = trainer.train_step(batch)
        assert isinstance(loss, torch.Tensor)


class TestTrainerSaveCheckpoint:
    """Tests for Trainer.save_checkpoint() method."""

    def test_save_checkpoint_basic(self, tmp_path):
        """Test basic checkpoint saving."""
        model = SimpleModel()
        config = {
            "training": {
                "checkpointing": {"save_dir": str(tmp_path)}
            }
        }

        dataset = TensorDataset(torch.randn(10, 10), torch.randn(10, 5))
        train_loader = DataLoader(dataset, batch_size=2)

        trainer = Trainer(model, config, train_loader, device="cpu")
        trainer.save_checkpoint(step=100)

        checkpoint_path = tmp_path / "model_step_100.pt"
        assert checkpoint_path.exists()

    def test_save_checkpoint_final(self, tmp_path):
        """Test saving final checkpoint."""
        model = SimpleModel()
        config = {
            "training": {
                "checkpointing": {"save_dir": str(tmp_path)}
            }
        }

        dataset = TensorDataset(torch.randn(10, 10), torch.randn(10, 5))
        train_loader = DataLoader(dataset, batch_size=2)

        trainer = Trainer(model, config, train_loader, device="cpu")
        trainer.save_checkpoint(step=1000, final=True)

        checkpoint_path = tmp_path / "model_final.pt"
        assert checkpoint_path.exists()

    def test_checkpoint_contents(self, tmp_path):
        """Test that checkpoint contains all necessary information."""
        model = SimpleModel()
        config = {
            "training": {
                "checkpointing": {"save_dir": str(tmp_path)},
                "max_steps": 1000
            }
        }

        dataset = TensorDataset(torch.randn(10, 10), torch.randn(10, 5))
        train_loader = DataLoader(dataset, batch_size=2)

        trainer = Trainer(model, config, train_loader, device="cpu")
        trainer.save_checkpoint(step=500)

        checkpoint_path = tmp_path / "model_step_500.pt"
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        assert 'step' in checkpoint
        assert checkpoint['step'] == 500
        assert 'model_state_dict' in checkpoint
        assert 'optimizer_state_dict' in checkpoint
        assert 'config' in checkpoint
        assert checkpoint['config'] == config

    def test_load_from_checkpoint(self, tmp_path):
        """Test loading model from checkpoint."""
        model = SimpleModel()
        config = {
            "training": {
                "checkpointing": {"save_dir": str(tmp_path)}
            }
        }

        dataset = TensorDataset(torch.randn(10, 10), torch.randn(10, 5))
        train_loader = DataLoader(dataset, batch_size=2)

        trainer = Trainer(model, config, train_loader, device="cpu")

        # Save checkpoint
        trainer.save_checkpoint(step=100)

        # Create new model and load checkpoint
        new_model = SimpleModel()
        checkpoint = torch.load(tmp_path / "model_step_100.pt", map_location="cpu")
        new_model.load_state_dict(checkpoint['model_state_dict'])

        # Compare parameters
        for p1, p2 in zip(model.parameters(), new_model.parameters()):
            assert torch.equal(p1, p2)

    def test_multiple_checkpoints(self, tmp_path):
        """Test saving multiple checkpoints."""
        model = SimpleModel()
        config = {
            "training": {
                "checkpointing": {"save_dir": str(tmp_path)}
            }
        }

        dataset = TensorDataset(torch.randn(10, 10), torch.randn(10, 5))
        train_loader = DataLoader(dataset, batch_size=2)

        trainer = Trainer(model, config, train_loader, device="cpu")

        # Save multiple checkpoints
        for step in [100, 200, 300]:
            trainer.save_checkpoint(step=step)

        assert (tmp_path / "model_step_100.pt").exists()
        assert (tmp_path / "model_step_200.pt").exists()
        assert (tmp_path / "model_step_300.pt").exists()


class TestTrainerFullTraining:
    """Tests for Trainer.train() method and full training loop."""

    def test_training_loop_completes(self, tmp_path):
        """Test that training loop completes successfully."""
        model = SimpleModel()
        config = {
            "training": {
                "max_steps": 10,
                "log_interval": 5,
                "save_interval": 10,
                "checkpointing": {"save_dir": str(tmp_path)}
            }
        }

        # Create dataset with enough samples
        dataset = TensorDataset(torch.randn(100, 10), torch.randn(100, 5))
        train_loader = DataLoader(dataset, batch_size=2)

        trainer = Trainer(model, config, train_loader, device="cpu")

        # Mock the train_step to return predictable values
        original_train_step = trainer.train_step

        def mock_train_step(batch):
            loss = torch.tensor(1.0, requires_grad=True)
            loss_dict = {"total": loss, "mse": loss}
            # Still do gradient updates
            return original_train_step(batch)

        with patch.object(trainer, 'train_step', side_effect=mock_train_step):
            trainer.train()

        # Check that final checkpoint was saved
        assert (tmp_path / "model_final.pt").exists()

    def test_training_loop_stops_at_max_steps(self, tmp_path):
        """Test that training stops at max_steps."""
        model = SimpleModel()
        max_steps = 5
        config = {
            "training": {
                "max_steps": max_steps,
                "log_interval": 2,
                "save_interval": 100,  # Don't save during training
                "checkpointing": {"save_dir": str(tmp_path)}
            }
        }

        dataset = TensorDataset(torch.randn(100, 10), torch.randn(100, 5))
        train_loader = DataLoader(dataset, batch_size=2)

        trainer = Trainer(model, config, train_loader, device="cpu")

        # Count train_step calls
        call_count = 0
        original_train_step = trainer.train_step

        def counting_train_step(batch):
            nonlocal call_count
            call_count += 1
            return original_train_step(batch)

        trainer.train_step = counting_train_step
        trainer.train()

        assert call_count == max_steps

    def test_training_saves_checkpoint_at_interval(self, tmp_path):
        """Test that checkpoints are saved at specified intervals."""
        model = SimpleModel()
        config = {
            "training": {
                "max_steps": 25,
                "log_interval": 5,
                "save_interval": 10,
                "checkpointing": {"save_dir": str(tmp_path)}
            }
        }

        dataset = TensorDataset(torch.randn(100, 10), torch.randn(100, 5))
        train_loader = DataLoader(dataset, batch_size=2)

        trainer = Trainer(model, config, train_loader, device="cpu")
        trainer.train()

        # Check that checkpoints were saved at steps 10 and 20
        assert (tmp_path / "model_step_10.pt").exists()
        assert (tmp_path / "model_step_20.pt").exists()
        # Final checkpoint at step 25
        assert (tmp_path / "model_final.pt").exists()

    def test_training_with_small_dataset(self, tmp_path):
        """Test training with dataset smaller than max_steps."""
        model = SimpleModel()
        config = {
            "training": {
                "max_steps": 100,  # More steps than data
                "log_interval": 10,
                "save_interval": 50,
                "checkpointing": {"save_dir": str(tmp_path)}
            }
        }

        # Small dataset - only 5 batches
        dataset = TensorDataset(torch.randn(10, 10), torch.randn(10, 5))
        train_loader = DataLoader(dataset, batch_size=2)

        trainer = Trainer(model, config, train_loader, device="cpu")
        trainer.train()

        # Should complete successfully and iterate through dataset multiple times
        assert (tmp_path / "model_final.pt").exists()


class TestTrainerEdgeCases:
    """Tests for edge cases and error handling."""

    def test_trainer_with_empty_config(self):
        """Test trainer with completely empty config."""
        model = SimpleModel()
        config = {}

        dataset = TensorDataset(torch.randn(10, 10), torch.randn(10, 5))
        train_loader = DataLoader(dataset, batch_size=2)

        # Should not raise error, use defaults
        trainer = Trainer(model, config, train_loader, device="cpu")
        assert trainer.max_steps == 1000

    def test_trainer_gradient_nan_detection(self):
        """Test behavior when gradients become NaN."""
        model = SimpleModel()
        config = {"training": {}}

        dataset = TensorDataset(torch.randn(10, 10), torch.randn(10, 5))
        train_loader = DataLoader(dataset, batch_size=2)

        trainer = Trainer(model, config, train_loader, device="cpu")

        # Create batch that could produce NaN
        batch = {
            "inputs": {"internal_voice_tokens": torch.tensor([[float('nan')] * 10, [float('nan')] * 10])},
            "targets": {"internal_text": torch.randn(2, 5)}
        }

        # Should handle without crashing (though loss may be NaN)
        loss, loss_dict = trainer.train_step(batch)
        # Loss might be NaN, which is expected behavior

    def test_model_mode_after_training(self, tmp_path):
        """Test that model is in train mode during training."""
        model = SimpleModel()
        config = {
            "training": {
                "max_steps": 5,
                "checkpointing": {"save_dir": str(tmp_path)}
            }
        }

        dataset = TensorDataset(torch.randn(100, 10), torch.randn(100, 5))
        train_loader = DataLoader(dataset, batch_size=2)

        trainer = Trainer(model, config, train_loader, device="cpu")

        # Verify model is set to train mode
        original_train_step = trainer.train_step

        def check_mode_train_step(batch):
            assert trainer.model.training
            return original_train_step(batch)

        trainer.train_step = check_mode_train_step
        trainer.train()
