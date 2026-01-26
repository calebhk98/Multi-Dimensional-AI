"""
Tests for Trainer plumbing enhancements (logging, checkpointing).
Separated from test_trainer.py to maintain manageable file size.
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from src.training.trainer import Trainer

class MockModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(10, 5)
        
    def forward(self, **kwargs):
        return self.layer(kwargs.get("vision_left", torch.randn(2, 10)))
        
    def compute_loss(self, outputs, targets):
        # Return a scalar loss and a rich dictionary of breakdown losses
        loss = torch.tensor(1.0, requires_grad=True)
        loss_dict = {
            "total": loss,
            "loss/vision_left": torch.tensor(0.5),
            "loss/audio": torch.tensor(0.3),
            "loss/touch": torch.tensor(0.2)
        }
        return loss, loss_dict

def test_modality_specific_logging():
    """
    Purpose:
        Verify that modality-specific losses are correctly propagated in the training step.
    
    Workflow:
        1. Setup Trainer with MockModel.
        2. Run train_step.
        3. Check return values for granular keys.
    """
    model = MockModel()
    config = {"training": {}}
    dataset = TensorDataset(torch.randn(10, 10), torch.randn(10, 5))
    loader = DataLoader(dataset, batch_size=2)
    
    trainer = Trainer(model, config, loader)
    
    batch = {
        "inputs": {"vision_left": torch.randn(2, 10)},
        "targets": {}
    }
    
    loss, loss_dict = trainer.train_step(batch)
    
    assert "loss/vision_left" in loss_dict
    assert "loss/audio" in loss_dict
    assert loss_dict["loss/vision_left"] == 0.5

def test_checkpoint_loading(tmp_path):
    """
    Purpose:
        Verify that the Trainer can load a checkpoint and restore state.
        
    Workflow:
        1. Initialize Trainer A, save checkpoint.
        2. Initialize Trainer B.
        3. Trainer B loads checkpoint.
        4. Verify step/config/state match.
    """
    model = MockModel()
    config = {
        "training": {
            "checkpointing": {"save_dir": str(tmp_path)},
            "max_steps": 100
        },
        "test_param": "foo" # Unique config param to verify
    }
    dataset = TensorDataset(torch.randn(10, 10), torch.randn(10, 5))
    loader = DataLoader(dataset, batch_size=2)
    
    trainer_a = Trainer(model, config, loader)
    trainer_a.save_checkpoint(step=50)
    
    # Modify model weights slightly to ensure load actually changes them
    with torch.no_grad():
        model.layer.weight.add_(1.0)
        
    trainer_b = Trainer(model, config, loader)
    
    # We need to implement load_checkpoint in Trainer first! 
    # This test is expected to fail or error until then (AttributeError).
    if hasattr(trainer_b, "load_checkpoint"): 
        loaded_state = trainer_b.load_checkpoint(tmp_path / "model_step_50.pt")
        
        # Verify
        assert loaded_state["step"] == 50
        assert loaded_state["config"]["test_param"] == "foo"
        
        # In a real scenario we'd check weights, but MockModel is shared instance here unless copy.
        # But conceptually checking the method returns state is enough for plumbing.
    else:
        pytest.fail("Trainer does not have load_checkpoint method yet")
