"""
Full forward/backward pass validation test.
Validates that a realistic config can complete a full training step.
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class MockCreatureForFullPass(nn.Module):
	"""
	Simple mock model to test full pass capability.
	"""
	def __init__(self):
		super().__init__()
		self.encoder = nn.Linear(100, 64)
		self.decoder = nn.Linear(64, 10)
		
	def forward(self, **kwargs):
		x = kwargs.get("vision_left", torch.randn(2, 100))
		h = self.encoder(x)
		out = self.decoder(h)
		return {"hidden": h, "output": out}
		
	def compute_loss(self, outputs, targets):
		pred = outputs["output"]
		target = targets.get("internal_text", torch.zeros_like(pred))
		loss = nn.functional.mse_loss(pred, target)
		return loss, {"total": loss, "mse": loss}


def test_full_forward_backward_pass():
	"""
	Purpose:
		Validate that a realistic config can run a full forward/backward pass.
		
	Workflow:
		1. Create model with encoder/decoder layers.
		2. Create batch with inputs and targets.
		3. Run forward pass.
		4. Compute loss.
		5. Run backward pass.
		6. Verify gradients exist.
	"""
	model = MockCreatureForFullPass()
	
	batch = {
		"inputs": {"vision_left": torch.randn(4, 100)},
		"targets": {"internal_text": torch.randn(4, 10)}
	}
	
	# Forward
	outputs = model(**batch["inputs"])
	assert "hidden" in outputs
	assert "output" in outputs
	
	# Loss
	loss, loss_dict = model.compute_loss(outputs, batch["targets"])
	assert loss.requires_grad
	
	# Backward
	loss.backward()
	
	# Check gradients
	for name, param in model.named_parameters():
		assert param.grad is not None, f"No gradient for {name}"
		assert not torch.all(param.grad == 0), f"Zero gradient for {name}"
		
	print("Full forward/backward pass validated successfully.")


def test_full_pass_with_trainer():
	"""
	Purpose:
		Validate full pass through Trainer.train_step.
		
	Workflow:
		1. Setup Trainer with mock model.
		2. Run train_step.
		3. Verify loss returned and parameters updated.
	"""
	from src.training.trainer import Trainer
	
	model = MockCreatureForFullPass()
	config = {"training": {}}
	dataset = TensorDataset(torch.randn(10, 100), torch.randn(10, 10))
	loader = DataLoader(dataset, batch_size=2)
	
	trainer = Trainer(model, config, loader, device="cpu")
	
	# Store initial params
	initial_params = {n: p.clone() for n, p in model.named_parameters()}
	
	batch = {
		"inputs": {"vision_left": torch.randn(2, 100)},
		"targets": {"internal_text": torch.randn(2, 10)}
	}
	
	loss, loss_dict = trainer.train_step(batch)
	
	assert loss is not None
	assert "total" in loss_dict
	
	# Verify params changed
	for name, param in model.named_parameters():
		assert not torch.equal(initial_params[name], param), f"Param {name} unchanged"
		
	print("Full pass with Trainer validated successfully.")
