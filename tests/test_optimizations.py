
"""
Tests for training optimizations.
"""

import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock, patch
from src.training.trainer import Trainer
from src.config import Config

class SimpleModel(nn.Module):
    def __init__(self):
        """Initialize simple linear model."""
        super().__init__()
        self.linear = nn.Linear(10, 10)
    
    def forward(self, x, **kwargs):
        """
        Forward pass.
        
        Args:
            x: Input tensor.
            **kwargs: Additional args.
            
        Returns:
            Output tensor.
        """
        return self.linear(x)

    def compute_loss(self, outputs, targets):
        """
        Compute mean loss.
        
        Args:
            outputs: Model outputs.
            targets: Target values.
            
        Returns:
            Tuple of (loss_tensor, loss_dict).
        """
        loss = outputs.mean()
        return loss, {"total_loss": loss}

@pytest.fixture
def mock_config():
    """
    Create a mock configuration dictionary.

    Returns:
        Dictionary with training configuration.
    """
    return {
        "training": {
            "optimizer": {"lr": 1e-3},
            "scheduler": {"type": "cosine", "warmup_steps": 10},
            "gradient_accumulation_steps": 2,
            "mixed_precision": "no", # Disable for CPU test
            "max_grad_norm": 1.0,
            "max_steps": 100,
            "log_interval": 1,
            "save_interval": 100,
            "checkpointing": {"save_dir": "tmp_checkpoints"}
        }
    }

def test_gradient_accumulation_steps(mock_config):
    """
    Test gradient accumulation logic.

    Purpose:
        Verify that optimizer steps are skipped/performed according to accumulation config.

    Workflow:
        1. Setup model and mock optimizer.
        2. Run training steps loop manually or via mocked trainer.
        3. Assert call counts of scaler.step/optimizer.step.

    Args:
        mock_config: Config fixture.
        
    ToDo:
        None
    """
    # Setup
    model = SimpleModel()
    optimizer_mock = MagicMock(spec=torch.optim.AdamW)
    # We strip the optimizer creation in Trainer or mock it?
    # Easier to just instantiate Trainer then replace optimizer
    
    # We need a train_loader mock
    loader = [
        {"inputs": {"x": torch.randn(2, 10)}, "targets": {"y": torch.randn(2, 10)}},
        {"inputs": {"x": torch.randn(2, 10)}, "targets": {"y": torch.randn(2, 10)}},
        {"inputs": {"x": torch.randn(2, 10)}, "targets": {"y": torch.randn(2, 10)}},
        {"inputs": {"x": torch.randn(2, 10)}, "targets": {"y": torch.randn(2, 10)}},
    ]
    
    trainer = Trainer(model, mock_config, loader)
    # Mock optimizer and scheduler
    trainer.optimizer = MagicMock()
    trainer.optimizer.param_groups = [{'lr': 1e-3}]
    trainer.scheduler = MagicMock()
    trainer.base_scheduler = MagicMock()
    
    # Mock scaler
    trainer.scaler = MagicMock()
    
    # Run 4 steps (batches)
    # accumulation=2 should trigger step 2 times (at step 1 and 3, 0-indexed?)
    # Logic: if (step + 1) % accum == 0: step()
    # Step 0: (1)%2 != 0. No step.
    # Step 1: (2)%2 == 0. Step.
    # Step 2: (3)%2 != 0. No step.
    # Step 3: (4)%2 == 0. Step.
    
    trainer.max_steps = 4
    trainer.train()
    
    # Check optimizer calls
    # scaler.step(optimizer) is called instead of optimizer.step()
    assert trainer.scaler.step.call_count == 2
    assert trainer.scaler.update.call_count == 2
    assert trainer.scheduler.step.call_count == 2
    
def test_scheduler_initialization(mock_config):
    """
    Test scheduler setup.

    Purpose:
        Verify that scheduler is initialized with correct warmup steps.

    Workflow:
        1. Modify config for warmup.
        2. Initialize Trainer.
        3. Check trainer.scheduler attribute.

    Args:
        mock_config: Config fixture.

    ToDo:
        None
    """
    model = SimpleModel()
    mock_config["training"]["scheduler"]["warmup_steps"] = 5
    trainer = Trainer(model, mock_config, [])
    
    assert trainer.warmup_steps == 5
    assert trainer.scheduler is not None

def test_config_flags(mock_config):
    """
    Test configuration flags are correctly applied.

    Purpose:
        Verify mixed precision, compile, and checkpointing flags are respected.

    Workflow:
        1. Set config flags.
        2. Initialize Trainer.
        3. Assert trainer attributes match config.
        4. Assert model methods were called (e.g. enable_gradient_checkpointing).

    Args:
        mock_config: Config fixture.

    ToDo:
        None
    """
    mock_config["training"]["mixed_precision"] = "fp16"
    mock_config["training"]["gradient_checkpointing"] = True
    mock_config["training"]["compile_model"] = False # Avoid compile overhead in test
    
    model = SimpleModel()
    model.transformer = MagicMock() # Mock transformer submodule
    
    trainer = Trainer(model, mock_config, [])
    
    assert trainer.mixed_precision == "fp16"
    assert trainer.gradient_checkpointing is True
    # Verify enable_gradient_checkpointing was called
    model.transformer.enable_gradient_checkpointing.assert_called_with(True)

def test_mfu_tracking_logic(mock_config):
    """
    Test MFU calculation logic.

    Purpose:
        Ensure MFU and token metrics are calculated without error.

    Workflow:
        1. Run a minimal training step.
        2. Check that metrics history contains MFU and TFLOPS keys.

    Args:
        mock_config: Config fixture.

    ToDo:
        None
    """
    # Verify MFU calc doesn't crash
    model = SimpleModel()
    loader = [{"inputs": {"x": torch.randn(2, 10)}, "targets": {"y": torch.randn(2, 10)}}]
    trainer = Trainer(model, mock_config, loader)
    trainer.max_steps = 1
    
    # Mock methods to avoid complex logic
    trainer.optimizer = MagicMock()
    trainer.optimizer.param_groups = [{'lr': 0.1}]
    trainer.scaler = MagicMock()
    
    trainer.train()
    
    # Check metrics history
    assert len(trainer.metrics_history) > 0
    last_metric = trainer.metrics_history[-1]
    assert "mfu" in last_metric
    assert "tflops" in last_metric
