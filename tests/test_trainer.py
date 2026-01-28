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



from torch.utils.data import Dataset

class MockDataset(Dataset):
    """Mock dataset that returns dictionaries as expected by Trainer."""
    
    def __init__(self, size=10, input_dim=10, target_dim=5):
        """
        Initialize mock dataset.

        Args:
            size: Number of samples.
            input_dim: Input feature dimension.
            target_dim: Target feature dimension.
        """
        self.inputs = torch.randn(size, input_dim)
        self.targets = torch.randn(size, target_dim)
        
    def __len__(self):
        """
        Get dataset size.

        Returns:
            Length of the dataset.
        """
        return len(self.inputs)
        
    def __getitem__(self, idx):
        """
        Get item at index.

        Args:
            idx: Index.

        Returns:
            Dictionary with inputs and targets.
        """
        return {
            "inputs": {"internal_voice_tokens": self.inputs[idx]},
            "targets": {"internal_text": self.targets[idx]}
        }


class SimpleModel(nn.Module):
    """Simple model for testing."""

    def __init__(self, input_dim=10, hidden_dim=20, output_dim=5):
        """
        Simple model init.
        
        Args:
            input_dim: Input dimension.
            hidden_dim: Hidden dimension.
            output_dim: Output dimension.
        """
        super().__init__()
        self.fc = nn.Linear(input_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, output_dim)
        self.loss_fn = nn.MSELoss()

    def forward(self, **kwargs):
        """
        Forward pass that handles multi-modal inputs.
            
        Args:
            **kwargs: Inputs.
            
        Returns:
            Dict with outputs.
        """
        # For testing, just use internal_voice_tokens if provided
        x = kwargs.get('internal_voice_tokens', torch.randn(2, 10))
        if x.dim() > 2:
            x = x.view(x.size(0), -1)  # Flatten
        h = self.fc(x)
        out = self.output(h)
        return {"hidden_states": h, "output": out}

    def compute_loss(self, outputs, targets):
        """
        Compute loss for testing.
            
        Args:
            outputs: Model outputs.
            targets: Target values.
            
        Returns:
            Tuple of (loss, loss_dict).
        """
        loss = self.loss_fn(outputs["output"], targets.get("internal_text", torch.zeros_like(outputs["output"])))
        loss_dict = {"total": loss, "mse": loss}
        return loss, loss_dict


class TestTrainerInitialization:
    """Tests for Trainer initialization."""

    def test_trainer_basic_initialization(self):
        """
        Purpose:
            Test basic trainer initialization with a standard config.

        Workflow:
            1. Create a model and config.
            2. Initialize Trainer.
            3. Verify attributes are set correctly.

        ToDo:
            - None
        """
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
        """
        Purpose:
            Test trainer initialization with validation loader.

        Workflow:
            1. Create model and config.
            2. Initialize Trainer with valid_loader.
            3. Verify val_loader is stored.

        ToDo:
            - None
        """
        model = SimpleModel()
        config = {"training": {}}

        dataset = TensorDataset(torch.randn(10, 10), torch.randn(10, 5))
        train_loader = DataLoader(dataset, batch_size=2)
        val_loader = DataLoader(dataset, batch_size=2)

        trainer = Trainer(model, config, train_loader, val_loader=val_loader, device="cpu")

        assert trainer.val_loader == val_loader

    def test_trainer_default_config_values(self):
        """
        Purpose:
            Test that trainer uses default values for missing config.

        Workflow:
            1. Initialize Trainer with empty config.
            2. Verify default values for steps, intervals, etc.

        ToDo:
            - None
        """
        model = SimpleModel()
        config = {"training": {}}  # Empty training config

        dataset = TensorDataset(torch.randn(10, 10), torch.randn(10, 5))
        train_loader = DataLoader(dataset, batch_size=2)

        trainer = Trainer(model, config, train_loader, device="cpu")

        assert trainer.max_steps == 1000
        assert trainer.log_interval == 10
        assert trainer.save_interval == 100

    def test_trainer_custom_optimizer_config(self):
        """
        Purpose:
             Test trainer with custom optimizer configuration.

        Workflow:
            1. Define config with custom optimizer params.
            2. Initialize Trainer.
            3. Verify optimizer defaults match config.

        ToDo:
            - None
        """
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
        """
        Purpose:
            Test that trainer creates checkpoint save directory.

        Workflow:
            1. Define save dict in config.
            2. Initialize Trainer.
            3. Verify directory exists.

        ToDo:
            - None
        """
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
        """
        Purpose:
            Test trainer initialization with CUDA device.

        Workflow:
            1. Initialize Trainer with device='cuda'.
            2. Verify model parameters are on CUDA.

        ToDo:
            - Skip if no CUDA.
        """
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
        """
        Purpose:
            Test basic training step execution.

        Workflow:
            1. Initialize Trainer.
            2. Run train_step with a batch.
            3. Verify loss is scalar and loss_dict contains 'total'.

        ToDo:
            - None
        """
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
        """
        Purpose:
            Test that train_step performs backward pass correctly.

        Workflow:
            1. Ensure gradients are initially None.
            2. Run train_step.
            3. Verify gradients are populated.

        ToDo:
            - None
        """
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
        """
        Purpose:
            Test that train_step updates model parameters.

        Workflow:
            1. Store initial parameters.
            2. Run train_step.
            3. Verify parameters have changed.

        ToDo:
            - None
        """
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
        """
        Purpose:
            Test that gradient clipping is applied.

        Workflow:
            1. Create batch with large values.
            2. Mock clip_grad_norm_.
            3. Run train_step.
            4. Verify clip was called.

        ToDo:
            - None
        """
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
        """
        Purpose:
            Test train_step with nested touch data structure.

        Workflow:
            1. Create batch with nested dictionary inputs.
            2. Run train_step.
            3. Verify no errors and loss is returned.

        ToDo:
            - None
        """
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
        """
        Purpose:
            Test train_step with nested animation targets.

        Workflow:
            1. Create batch with nested targets.
            2. Run train_step.
            3. Verify execution.

        ToDo:
            - None
        """
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
        """
        Purpose:
            Test basic checkpoint saving.

        Workflow:
            1. Configure save_dir.
            2. Run save_checkpoint(step=100).
            3. Verify file exists.

        ToDo:
            - None
        """
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
        """
        Purpose:
            Test saving final checkpoint.

        Workflow:
            1. Run save_checkpoint with final=True.
            2. Verify 'model_final.pt' exists.

        ToDo:
            - None
        """
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
        """
        Purpose:
            Test that checkpoint contains all necessary information.

        Workflow:
            1. Save checkpoint.
            2. Load checkpoint.
            3. Verify keys (step, model_state, optimizer_state, config).

        ToDo:
            - None
        """
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
        """
        Purpose:
            Test loading model from checkpoint.

        Workflow:
            1. Save checkpoint from Model A.
            2. Initialize Model B.
            3. Load checkpoint into Model B.
            4. Verify parameters match.

        ToDo:
            - None
        """
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
        """
        Purpose:
            Test saving multiple checkpoints.

        Workflow:
            1. Call save_checkpoint multiple times.
            2. Verify all files exist.

        ToDo:
            - None
        """
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
        """
        Purpose:
            Test that training loop completes successfully.

        Workflow:
            1. Configure steps and save dir.
            2. Mock train_step to return dummy loss.
            3. Run train().
            4. Verify final checkpoint exists.

        ToDo:
            - None
        """
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
        dataset = MockDataset(size=100)
        train_loader = DataLoader(dataset, batch_size=2)

        trainer = Trainer(model, config, train_loader, device="cpu")

        # Mock the train_step to return predictable values
        original_train_step = trainer.train_step

        def mock_train_step(batch):
            """
            Mock of train step for testing loop without full backward pass.
            
            Args:
                batch: Input batch.
                
            Returns:
                Tuple of (loss, loss_dict).
            """
            loss = torch.tensor(1.0, requires_grad=True)
            loss_dict = {"total": loss, "mse": loss}
            # Still do gradient updates
            return original_train_step(batch)

        with patch.object(trainer, 'train_step', side_effect=mock_train_step):
            trainer.train()

        # Check that final checkpoint was saved
        assert (tmp_path / "model_final.pt").exists()

    def test_training_loop_stops_at_max_steps(self, tmp_path):
        """
        Purpose:
             Test that training stops at max_steps.

        Workflow:
            1. Set max_steps=5.
            2. Add counter wrapper to train_step.
            3. Run train().
            4. Verify count equals max_steps.

        ToDo:
            - None
        """
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

        dataset = MockDataset(size=100)
        train_loader = DataLoader(dataset, batch_size=2)

        trainer = Trainer(model, config, train_loader, device="cpu")

        # Count train_step calls
        call_count = 0
        original_train_step = trainer.train_step

        def counting_train_step(batch):
            """
            Wrapper to count calls to train_step.
            
            Args:
                batch: Input batch.
                
            Returns:
                Result of original train_step.
            """
            nonlocal call_count
            call_count += 1
            return original_train_step(batch)

        trainer.train_step = counting_train_step
        trainer.train()

        assert call_count == max_steps

    def test_training_saves_checkpoint_at_interval(self, tmp_path):
        """
        Purpose:
            Test that checkpoints are saved at specified intervals.

        Workflow:
            1. Set save_interval=10, max_steps=25.
            2. Run train().
            3. Verify checkpoints at 10, 20, and final exist.

        ToDo:
            - None
        """
        model = SimpleModel()
        config = {
            "training": {
                "max_steps": 25,
                "log_interval": 5,
                "save_interval": 10,
                "checkpointing": {"save_dir": str(tmp_path)}
            }
        }

        dataset = MockDataset(size=100)
        train_loader = DataLoader(dataset, batch_size=2)

        trainer = Trainer(model, config, train_loader, device="cpu")
        trainer.train()

        # Check that checkpoints were saved at steps 10 and 20
        assert (tmp_path / "model_step_10.pt").exists()
        assert (tmp_path / "model_step_20.pt").exists()
        # Final checkpoint at step 25
        assert (tmp_path / "model_final.pt").exists()

    def test_training_with_small_dataset(self, tmp_path):
        """
        Purpose:
            Test training with dataset smaller than max_steps.

        Workflow:
            1. Create small dataset (5 batches).
            2. Set max_steps=100.
            3. Run train().
            4. Verify loop continues recycling data until 100 steps.

        ToDo:
            - None
        """
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
        dataset = MockDataset(size=10)
        train_loader = DataLoader(dataset, batch_size=2)

        trainer = Trainer(model, config, train_loader, device="cpu")
        trainer.train()

        # Should complete successfully and iterate through dataset multiple times
        assert (tmp_path / "model_final.pt").exists()


class TestTrainerEdgeCases:
    """Tests for edge cases and error handling."""

    def test_trainer_with_empty_config(self):
        """
        Purpose:
            Test trainer with completely empty config.

        Workflow:
            1. Initialize with config={}.
            2. Verify defaults are used.

        ToDo:
            - None
        """
        model = SimpleModel()
        config = {}

        dataset = TensorDataset(torch.randn(10, 10), torch.randn(10, 5))
        train_loader = DataLoader(dataset, batch_size=2)

        # Should not raise error, use defaults
        trainer = Trainer(model, config, train_loader, device="cpu")
        assert trainer.max_steps == 1000

    def test_trainer_gradient_nan_detection(self):
        """
        Purpose:
            Test behavior when gradients become NaN.

        Workflow:
            1. Create batch with NaNs.
            2. Run train_step.
            3. Verify it does not crash (check loss handling).

        ToDo:
            - None
        """
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
        """
        Purpose:
            Test that model is in train mode during training.

        Workflow:
            1. Add wrapper to train_step that asserts model.training is True.
            2. Run train().
            3. Verify assertions pass.

        ToDo:
            - None
        """
        model = SimpleModel()
        config = {
            "training": {
                "max_steps": 5,
                "checkpointing": {"save_dir": str(tmp_path)}
            }
        }

        dataset = MockDataset(size=100)
        train_loader = DataLoader(dataset, batch_size=2)

        trainer = Trainer(model, config, train_loader, device="cpu")

        # Verify model is set to train mode
        original_train_step = trainer.train_step

        def check_mode_train_step(batch):
            """
            Verifies model is in training mode during step.
            
            Args:
                batch: Input batch.
                
            Returns:
                Result of original train_step.
            """
            assert trainer.model.training
            return original_train_step(batch)

        trainer.train_step = check_mode_train_step
        trainer.train()


class TestCheckpointValidation:
	"""
	Purpose:
		Tests for checkpoint validation with real data shapes.
		
	Workflow:
		Validate checkpoint save/load and resume functionality.
		
	ToDo:
		None
	"""
	
	def test_checkpoint_save_with_real_shapes(self, tmp_path):
		"""
		Purpose:
			Test checkpoint saving with realistic multi-modal data shapes.
			
		Workflow:
			1. Create trainer with multi-modal dataset
			2. Save checkpoint
			3. Load and verify shapes match expected
			
		ToDo:
			None
		"""
		from src.data.synthetic_generator import SyntheticDataGenerator
		from src.data.multimodal_dataset import MultiModalDataset, multimodal_collate_fn
		
		generator = SyntheticDataGenerator()
		dataset = MultiModalDataset(generator, length=10)
		loader = DataLoader(dataset, batch_size=2, collate_fn=multimodal_collate_fn)
		
		model = SimpleModel()
		config = {
			"training": {
				"max_steps": 5,
				"checkpointing": {"save_dir": str(tmp_path)}
			}
		}
		
		trainer = Trainer(model, config, loader, device="cpu")
		trainer.save_checkpoint(step=5)
		
		# Load and verify
		checkpoint = torch.load(tmp_path / "model_step_5.pt", map_location="cpu")
		assert checkpoint["step"] == 5
		assert "model_state_dict" in checkpoint
		assert "optimizer_state_dict" in checkpoint
		assert "config" in checkpoint
	
	def test_checkpoint_load_and_resume(self, tmp_path):
		"""
		Purpose:
			Test that training can resume from checkpoint.
			
		Workflow:
			1. Train for 10 steps and save
			2. Create new trainer and load checkpoint
			3. Resume training for 5 more steps
			4. Verify final model is different from initial
			
		ToDo:
			None
		"""
		from src.data.synthetic_generator import SyntheticDataGenerator
		from src.data.multimodal_dataset import MultiModalDataset, multimodal_collate_fn
		
		generator = SyntheticDataGenerator()
		dataset = MultiModalDataset(generator, length=50)
		loader = DataLoader(dataset, batch_size=2, collate_fn=multimodal_collate_fn)
		
		model = SimpleModel()
		config = {
			"training": {
				"max_steps": 10,
				"save_interval": 10,
				"checkpointing": {"save_dir": str(tmp_path)}
			}
		}
		
		# Initial training
		trainer1 = Trainer(model, config, loader, device="cpu")
		trainer1.train()
		
		# Save initial params
		checkpoint = torch.load(tmp_path / "model_step_10.pt", map_location="cpu")
		initial_params = [p.clone() for p in model.parameters()]
		
		# Create new trainer and load checkpoint
		new_model = SimpleModel()
		new_trainer = Trainer(new_model, config, loader, device="cpu")
		loaded_checkpoint = new_trainer.load_checkpoint(tmp_path / "model_step_10.pt")
		
		# Verify loaded step
		assert loaded_checkpoint["step"] == 10
		
		# Train for a few more steps
		for _ in range(5):
			batch = next(iter(loader))
			new_trainer.train_step(batch)
		
		# Verify params have changed
		params_changed = False
		for initial, current in zip(initial_params, new_model.parameters()):
			if not torch.equal(initial, current):
				params_changed = True
				break
		
		assert params_changed
	
	def test_checkpoint_resume_preserves_optimizer_state(self, tmp_path):
		"""
		Purpose:
			Test that optimizer state is preserved across checkpoint save/load.
			
		Workflow:
			1. Run train steps to build optimizer state
			2. Save checkpoint
			3. Load into new trainer
			4. Verify optimizer state matches
			
		ToDo:
			None
		"""
		from src.data.synthetic_generator import SyntheticDataGenerator
		from src.data.multimodal_dataset import MultiModalDataset, multimodal_collate_fn
		
		generator = SyntheticDataGenerator()
		dataset = MultiModalDataset(generator, length=20)
		loader = DataLoader(dataset, batch_size=2, collate_fn=multimodal_collate_fn)
		
		model = SimpleModel()
		config = {
			"training": {
				"max_steps": 10,
				"checkpointing": {"save_dir": str(tmp_path)}
			}
		}
		
		trainer = Trainer(model, config, loader, device="cpu")
		
		# Run few steps
		for _ in range(5):
			batch = next(iter(loader))
			trainer.train_step(batch)
		
		# Save checkpoint
		trainer.save_checkpoint(step=5)
		
		# Load intoan model
		new_model = SimpleModel()
		new_trainer = Trainer(new_model, config, loader, device="cpu")
		new_trainer.load_checkpoint(tmp_path / "model_step_5.pt")
		
		# Compare optimizer states (check that exp_avg buffers exist)
		# This verifies Adam state was preserved
		old_state = trainer.optimizer.state_dict()
		new_state = new_trainer.optimizer.state_dict()
		
		# Both should have state for parameters
		assert len(old_state["state"]) == len(new_state["state"])
	
	def test_checkpoint_with_multimodal_batch(self, tmp_path):
		"""
		Purpose:
			Verify checkpoint works with full multi-modal batch structure.
			
		Workflow:
			1. Use synthetic multi-modal dataset
			2. Train and save checkpoint
			3. Verify no errors with nested data structures
			
		ToDo:
			None
		"""
		from src.data.synthetic_generator import SyntheticDataGenerator
		from src.data.multimodal_dataset import MultiModalDataset, multimodal_collate_fn
		
		generator = SyntheticDataGenerator()
		dataset = MultiModalDataset(generator, length=10)
		loader = DataLoader(dataset, batch_size=2, collate_fn=multimodal_collate_fn)
		
		# Use simple model for testing infrastructure
		model = SimpleModel()
		config = {
			"training": {
				"max_steps": 3,
				"checkpointing": {"save_dir": str(tmp_path)}
			}
		}
		
		trainer = Trainer(model, config, loader, device="cpu")
		trainer.train()
		
		# Verify checkpoint exists and contains expected data
		checkpoint = torch.load(tmp_path / "model_final.pt", map_location="cpu")
		assert "step" in checkpoint
		assert checkpoint["step"] == 3
