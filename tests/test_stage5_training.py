"""
Tests for Stage 5: Backpropagation (End-to-End Training).
Verifies MultiModalDataset and full training loop integration.
"""

import pytest
import torch
from torch.utils.data import DataLoader
from src.config import Config
from src.models.multimodal_transformer import MultiModalCreature
from src.data.synthetic_generator import SyntheticDataGenerator
# This import will fail until we create the file
from src.data.multimodal_dataset import MultiModalDataset, multimodal_collate_fn

class TestStage5Training:
	"""Test suite for Stage 5 end-to-end training."""

	@pytest.fixture
	def config(self):
		"""Create a minimal config for testing."""
		return {
			"model": {
				"encoders": {
					"internal_voice": {"vocab_size": 100, "embedding_dim": 32},
					"external_voice": {"vocab_size": 100, "embedding_dim": 32},
					"audio": {"embedding_dim": 32},
					"vision": {"embedding_dim": 32, "image_size": 32},
					"proprioception": {"embedding_dim": 32},
					"touch": {"embedding_dim": 32}
				},
				"decoders": {
					"internal_text": {"vocab_size": 100, "embedding_dim": 32},
					"external_text": {"vocab_size": 100, "embedding_dim": 32},
					"audio": {"embedding_dim": 32},
					"animation": {"embedding_dim": 32}
				},
				"transformer": {
					"num_layers": 2,
					"hidden_dim": 32,
					"num_attention_heads": 4,
					"ffn_dim": 64
				},
				"fusion": {
					"strategy": "concatenate",
					"modality_embeddings": True
				}
			},
			"training": {
				"batch_size": 2,
				"loss_weights": {
					"internal_text": 1.0,
					"external_text": 1.0,
					"audio": 1.0,
					"animation": 1.0
				}
			}
		}

	@pytest.fixture
	def generator(self):
		"""Create a generator with small dimensions for speed."""
		return SyntheticDataGenerator(
			vocab_size=100,
			image_size=32,
			temporal_window=5,
			codebook_size=50
		)

	def test_dataset_instantiation(self, generator):
		"""Test that MultiModalDataset can be instantiated."""
		dataset = MultiModalDataset(generator, length=10)
		assert len(dataset) == 10
		
		sample = dataset[0]
		assert "inputs" in sample
		assert "targets" in sample
		
		# Check specific nested keys exist
		assert "vision" in sample["inputs"] or "left_eye_image" in sample["inputs"]
		# Note: Generator produces 'left_eye_image', not 'vision' directly unless dataset maps it?
		# Expectation: Dataset returns raw generator output usually
		assert "left_eye_image" in sample["inputs"]
		assert "internal_text" in sample["targets"]

	def test_dataloader_collation(self, generator):
		"""Test that DataLoader correctly batches samples using collate_fn."""
		dataset = MultiModalDataset(generator, length=10)
		loader = DataLoader(
			dataset, 
			batch_size=2, 
			collate_fn=multimodal_collate_fn
		)
		
		batch = next(iter(loader))
		
		# Check batch dimensions
		# Inputs
		assert batch["inputs"]["internal_voice_tokens"].shape[0] == 2
		assert batch["inputs"]["left_eye_image"].shape[0] == 2
		# Touch data is nested, check it collated correctly
		assert "touch_data" in batch["inputs"]
		assert batch["inputs"]["touch_data"]["contact_active"].shape[0] == 2
		
		# Targets
		assert batch["targets"]["internal_text"].shape[0] == 2
		assert "animation" in batch["targets"]
		assert batch["targets"]["animation"]["rotations"].shape[0] == 2

	def test_training_step(self, config, generator):
		"""Test a full forward/backward pass with the model."""
		# 1. Setup Model
		model = MultiModalCreature(config)
		
		# 2. Setup Data
		dataset = MultiModalDataset(generator, length=4)
		loader = DataLoader(
			dataset, 
			batch_size=2, 
			collate_fn=multimodal_collate_fn
		)
		
		batch = next(iter(loader))
		
		# 3. Forward Pass
		# We need to manually move to device or use Trainer logic. 
		# For unit test, just keep on CPU.
		
		# Model forward expects unpacked args or matching keys.
		# trainer.py handles this unpacking. Let's simulate trainer logic.
		inputs = batch["inputs"]
		targets = batch["targets"]
		
		outputs = model(
			**inputs,
			return_hidden_states=True
		)
		
		# 4. Compute Loss
		loss, loss_dict = model.compute_loss(outputs, targets)
		
		# 5. Backward
		loss.backward()
		
		assert loss.item() > 0
		assert "total_loss" in loss_dict
		assert "internal_text" in loss_dict
		# Animation loss should correspond to configured loss weights
		assert "animation" in loss_dict

