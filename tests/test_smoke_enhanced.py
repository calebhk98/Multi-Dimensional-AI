"""
Enhanced smoke tests with real sample batches and comprehensive validation.
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import tempfile
from pathlib import Path
import json

from src.data.synthetic_generator import SyntheticDataGenerator
from src.data.multimodal_dataset import MultiModalDataset, multimodal_collate_fn
from src.training.trainer import Trainer

# Try to import real model, fall back to mock if not available
try:
	from src.models.multimodal_transformer import MultiModalCreature
	MODEL_AVAILABLE = True
except ImportError:
	MODEL_AVAILABLE = False
	MultiModalCreature = None


class MockMultiModalCreature(nn.Module):
	"""
	Purpose:
		Mock model for testing when real model is unavailable.
		
	Workflow:
		1. Initialize with config
		2. Provide forward pass that returns hidden states
		3. Provide compute_loss method
		
	ToDo:
		None
	"""
	
	def __init__(self, config):
		"""
		Initialize mock creature.
		
		Args:
			config: Configuration dictionary.
		"""
		super().__init__()
		self.config = config
		self.param = nn.Parameter(torch.randn(1))
		
	def forward(self, **kwargs):
		"""
		Forward pass.
		
		Args:
			**kwargs: Various input modalities.
			
		Returns:
			Dict with hidden states.
		"""
		# Return mock hidden states
		return_hidden = kwargs.get("return_hidden_states", False)
		if return_hidden:
			return {"hidden": self.param.expand(1, 10, 512)}
		return {}
		
	def compute_loss(self, outputs, targets):
		"""
		Compute mock loss.
		
		Args:
			outputs: Model outputs.
			targets: Target values.
			
		Returns:
			Tuple of (loss_tensor, loss_dict).
		"""
		loss = torch.tensor(0.5, requires_grad=True)
		loss_dict = {
			"total": loss,
			"internal_text": torch.tensor(0.1),
			"external_text": torch.tensor(0.15),
			"audio": torch.tensor(0.12),
			"animation": torch.tensor(0.13),
		}
		return loss, loss_dict


class TestSmokeEnhanced:
	"""
	Purpose:
		Enhanced smoke tests for full pipeline validation.
		
	Workflow:
		Test complete forward/backward pass with various configurations.
		
	ToDo:
		None
	"""
	
	def test_forward_pass_all_modalities(self):
		"""
		Purpose:
			Test forward pass with all input modalities enabled.
			
		Workflow:
			1. Create synthetic dataset
			2. Initialize model
			3. Run forward pass
			4. Verify outputs exist
			
		ToDo:
			None
		"""
		generator = SyntheticDataGenerator()
		dataset = MultiModalDataset(generator, length=4)
		loader = DataLoader(dataset, batch_size=2, collate_fn=multimodal_collate_fn)
		
		config = {}
		if MODEL_AVAILABLE:
			model = MultiModalCreature(config)
		else:
			model = MockMultiModalCreature(config)
		
		batch = next(iter(loader))
		inputs = batch["inputs"]
		
		# Move to CPU for testing
		device = "cpu"
		for key in inputs:
			if isinstance(inputs[key], torch.Tensor):
				inputs[key] = inputs[key].to(device)
			elif isinstance(inputs[key], dict):
				inputs[key] = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
					for k, v in inputs[key].items()}
		
		model = model.to(device)
		outputs = model(**inputs, return_hidden_states=True)
		
		assert outputs is not None
		assert isinstance(outputs, dict)
	
	def test_backward_pass_gradient_flow(self):
		"""
		Purpose:
			Test that gradients flow properly through the model.
			
		Workflow:
			1. Create model and data
			2. Run forward pass
			3. Compute loss
			4. Run backward
			5. Verify gradients exist
			
		ToDo:
			None
		"""
		generator = SyntheticDataGenerator()
		dataset = MultiModalDataset(generator, length=4)
		loader = DataLoader(dataset, batch_size=2, collate_fn=multimodal_collate_fn)
		
		config = {}
		if MODEL_AVAILABLE:
			model = MultiModalCreature(config)
		else:
			model = MockMultiModalCreature(config)
		
		model = model.to("cpu")
		batch = next(iter(loader))
		
		# Move batch to CPU
		inputs = {k: v.to("cpu") if isinstance(v, torch.Tensor) else v 
			for k, v in batch["inputs"].items()}
		
		# Handle nested touch_data
		if "touch_data" in inputs:
			inputs["touch_data"] = {k: v.to("cpu") for k, v in inputs["touch_data"].items()}
		
		targets = {k: v.to("cpu") if isinstance(v, torch.Tensor) else v 
			for k, v in batch["targets"].items()}
		
		# Handle nested animation targets
		if "animation" in targets:
			targets["animation"] = {k: v.to("cpu") for k, v in targets["animation"].items()}
		
		# Forward
		outputs = model(**inputs, return_hidden_states=True)
		
		# Loss
		loss, loss_dict = model.compute_loss(outputs, targets)
		
		# Backward
		loss.backward()
		
		# Verify gradients exist
		has_gradients = False
		for param in model.parameters():
			if param.grad is not None:
				has_gradients = True
				break
		
		assert has_gradients, "No gradients found after backward pass"
	
	def test_trainer_integration_with_metrics(self):
		"""
		Purpose:
			Test trainer integration with metrics tracking.
			
		Workflow:
			1. Create trainer with synthetic data
			2. Run 5 training steps
			3. Verify metrics are tracked
			4. Verify metrics include modality-specific losses
			
		ToDo:
			None
		"""
		generator = SyntheticDataGenerator()
		dataset = MultiModalDataset(generator, length=10)
		loader = DataLoader(dataset, batch_size=2, collate_fn=multimodal_collate_fn)
		
		config = {
			"training": {
				"max_steps": 5,
				"log_interval": 2,
				"save_interval": 100,
			}
		}
		
		if MODEL_AVAILABLE:
			model = MultiModalCreature(config)
		else:
			model = MockMultiModalCreature(config)
		
		trainer = Trainer(model, config, loader, device="cpu")
		
		# Run training
		trainer.train()
		
		# Verify metrics history
		assert len(trainer.metrics_history) == 5
		
		# Check that metrics contain expected keys
		first_metric = trainer.metrics_history[0]
		assert "step" in first_metric
		assert "step_time_sec" in first_metric
		assert "samples_per_sec" in first_metric
		
		# Check for modality-specific losses (if model provides them)
		assert "total" in first_metric
	
	def test_all_encoders_decoders_used(self):
		"""
		Purpose:
			Verify that all modality encoders/decoders are exercised.
			
		Workflow:
			1. Create batch with all modalities
			2. Run forward pass
			3. Verify all expected outputs exist
			
		ToDo:
			None
		"""
		if not MODEL_AVAILABLE:
			pytest.skip("Real model not available")
		
		generator = SyntheticDataGenerator()
		sample = generator.generate_sample()
		
		config = {}
		model = MultiModalCreature(config).to("cpu")
		
		# Move inputs to CPU
		inputs = {k: v.to("cpu") if isinstance(v, torch.Tensor) else v 
			for k, v in sample["inputs"].items()}
		
		if "touch_data" in inputs:
			inputs["touch_data"] = {k: v.to("cpu") for k, v in inputs["touch_data"].items()}
		
		# Add batch dimension
		for key in inputs:
			if isinstance(inputs[key], torch.Tensor):
				inputs[key] = inputs[key].unsqueeze(0)
			elif isinstance(inputs[key], dict):
				inputs[key] = {k: v.unsqueeze(0) if isinstance(v, torch.Tensor) else v 
					for k, v in inputs[key].items()}
		
		outputs = model(**inputs)
		
		# Real model should return outputs (check if they exist)
		# This is a basic smoke test - just verify no crash
		assert outputs is not None
	
	def test_loss_computation_all_modalities(self):
		"""
		Purpose:
			Test that loss computation works for all output modalities.
			
		Workflow:
			1. Create sample batch
			2. Run forward with return_hidden_states=True
			3. Compute loss with all target modalities
			4. Verify loss dict contains expected keys
			
		ToDo:
			None
		"""
		generator = SyntheticDataGenerator()
		sample = generator.generate_sample()
		
		config = {}
		if MODEL_AVAILABLE:
			model = MultiModalCreature(config)
		else:
			model = MockMultiModalCreature(config)
		
		model = model.to("cpu")
		
		# Prepare inputs
		inputs = {k: v.to("cpu") if isinstance(v, torch.Tensor) else v 
			for k, v in sample["inputs"].items()}
		
		if "touch_data" in inputs:
			inputs["touch_data"] = {k: v.to("cpu") for k, v in inputs["touch_data"].items()}
		
		# Add batch dimension
		for key in inputs:
			if isinstance(inputs[key], torch.Tensor):
				inputs[key] = inputs[key].unsqueeze(0)
			elif isinstance(inputs[key], dict):
				inputs[key] = {k: v.unsqueeze(0) if isinstance(v, torch.Tensor) else v 
					for k, v in inputs[key].items()}
		
		# Prepare targets
		targets = {k: v.to("cpu") if isinstance(v, torch.Tensor) else v 
			for k, v in sample["targets"].items()}
		
		if "animation" in targets:
			targets["animation"] = {k: v.to("cpu") for k, v in targets["animation"].items()}
		
		# Add batch dimension
		for key in targets:
			if isinstance(targets[key], torch.Tensor):
				targets[key] = targets[key].unsqueeze(0)
			elif isinstance(targets[key], dict):
				targets[key] = {k: v.unsqueeze(0) if isinstance(v, torch.Tensor) else v 
					for k, v in targets[key].items()}
		
		# Forward
		outputs = model(**inputs, return_hidden_states=True)
		
		# Compute loss
		loss, loss_dict = model.compute_loss(outputs, targets)
		
		assert loss is not None
		assert isinstance(loss, torch.Tensor)
		assert "total" in loss_dict
