"""
Integration test for the full MultiModalCreature model.
"""

import pytest
import torch
from src.models import MultiModalCreature
from src.data.synthetic_generator import SyntheticDataGenerator

def test_model_forward_pass():
	"""
	Integration Test: Full Model Forward Pass.
	
	Verifies that the MultiModalCreature model can be instantiated with a minimal 
	config, accept inputs from all 6 modalities, and produce outputs for all 
	4 decoding streams without crashing.
	"""
	# Setup minimal config
	config = {
		"model": {
			"transformer": {
				"hidden_dim": 64, # Small for fast test
				"num_layers": 2,
				"num_attention_heads": 4,
				"ffn_dim": 256
			},
			"encoders": {
				"internal_voice": {"vocab_size": 100},
				"audio": {"embedding_dim": 64},
				"vision": {"embedding_dim": 64, "image_size": 32}, # Small image
				"proprioception": {"embedding_dim": 64},
				"touch": {"embedding_dim": 64}
			},
			"decoders": {
				"audio": {"embedding_dim": 64},
				"animation": {"embedding_dim": 64}
			},
			"fusion": {"strategy": "concatenate"}
		}
	}
	
	model = MultiModalCreature(config)
	
	# Generate a single sample
	# Note: Generator makes large defaults (image 224), we should probably make small inputs manually
	# Or just use the model's actual dim expectations if generator is adjustable.
	# We'll make manual small inputs consistent with the small config above.
	
	bs = 2
	inputs = {
		"internal_voice_tokens": torch.randint(0, 100, (bs, 10)),
		"external_voice_tokens": torch.randint(0, 100, (bs, 10)),
		"audio_waveform": torch.randn(bs, 16000),
		"left_eye_image": torch.randn(bs, 3, 32, 32),
		"right_eye_image": torch.randn(bs, 3, 32, 32),
		"joint_positions": torch.randn(bs, 10, 24, 3),
		"joint_rotations": torch.randn(bs, 10, 24, 4),
		"touch_data": {
			"contact_active": torch.ones(bs, 2, dtype=torch.bool),
			"contact_points": torch.zeros(bs, 2, dtype=torch.long),
			"contact_forces": torch.randn(bs, 2, 1),
			"contact_positions": torch.randn(bs, 2, 3),
			"surface_types": torch.zeros(bs, 2, dtype=torch.long),
			"temperatures": torch.randn(bs, 2, 1) # Added because generator has it now
		}
	}
	
	# Forward pass
	outputs = model(**inputs, return_hidden_states=True)
	
	assert 'internal_text' in outputs
	assert 'external_text' in outputs
	assert 'audio' in outputs
	assert 'animation' in outputs
	assert 'hidden_states' in outputs
	
	# Check output shapes
	# Output length depends on fusion length (total seq len of all inputs)
	# Just checking keys exist and are tensors is good enough for basic integration
	assert isinstance(outputs['internal_text']['tokens'], torch.Tensor)

def test_model_loss_computation():
	"""
	Integration Test: Full Model Loss Computation.
	
	Verifies that the model's compute_loss method can process outputs and targets
	to produce a scalar loss and a dictionary of individual losses.
	"""
	# Similar setup as above
	config = {
		"model": {
			"transformer": {"hidden_dim": 32, "num_layers": 1, "num_attention_heads": 2, "ffn_dim": 64},
			"encoders": {"internal_voice": {"vocab_size": 50}},
			"fusion": {"strategy": "concatenate"}
		},
		"loss_weights": {"internal_text": 1.0}
	}
	model = MultiModalCreature(config)
	
	# Fake outputs from model
	bs, seq_len = 2, 20
	outputs = {
		"hidden_states": torch.randn(bs, seq_len, 32)
	}
	
	# Fake targets
	targets = {
		"internal_text": torch.randint(0, 50, (bs, seq_len))
	}
	
	loss, loss_dict = model.compute_loss(outputs, targets)
	
	assert isinstance(loss, torch.Tensor)
	assert not torch.isnan(loss)
	assert 'internal_text' in loss_dict
