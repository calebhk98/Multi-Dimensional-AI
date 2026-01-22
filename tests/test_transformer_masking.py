
import pytest
import torch
import torch.nn as nn
from src.models.transformer_backbone import TransformerBackbone, TransformerLayer

class TestTransformerMasking:
	"""
	Tests for TransformerBackbone masking behavior.
	"""

	@pytest.fixture
	def model_config(self):
		return {
			"num_layers": 2,
			"hidden_dim": 32,
			"num_heads": 4,
			"ffn_dim": 64,
			"dropout": 0.0,  # Disable dropout for deterministic testing
			"attention_dropout": 0.0
		}

	def test_backbone_runs_with_padding_mask(self, model_config):
		"""
		Happy Path: Ensure the backbone runs without error when a padding mask is provided.
		This validates the fix for the 4D mask crash.
		"""
		model = TransformerBackbone(**model_config)
		bs, seq_len = 2, 10
		x = torch.randn(bs, seq_len, model_config["hidden_dim"])
		
		# Create a mask where the last few tokens are padding (0)
		# Sequence 1: 10 valid tokens
		# Sequence 2: 7 valid tokens, 3 padding
		mask = torch.ones(bs, seq_len)
		mask[1, 7:] = 0 
		
		# This previously crashed
		output = model(x, attention_mask=mask)
		
		assert output.shape == x.shape
		assert not torch.isnan(output).any()

	def test_padding_is_effectively_ignored(self, model_config):
		"""
		Logic Check: Ensure that padding tokens do not affect the output of valid tokens.
		If attention is properly masked, changing the content of padded tokens should not change 
		the output valid tokens.
		"""
		# Single layer to pinpoint attention effect
		model = TransformerBackbone(num_layers=1, hidden_dim=32, num_heads=4, ffn_dim=64, dropout=0.0, attention_dropout=0.0)
		# Set to eval mode
		model.eval()

		bs, seq_len = 1, 5
		x = torch.randn(bs, seq_len, 32)
		
		# Mask: last 2 tokens are padding
		mask = torch.tensor([[1.0, 1.0, 1.0, 0.0, 0.0]])
		
		# Run 1: Normal input
		with torch.no_grad():
			output1 = model(x, attention_mask=mask)
		
		# Run 2: Change the padded values in input
		x_modified = x.clone()
		x_modified[0, 3:] += 100.0 # Add huge noise to pad positions
		
		with torch.no_grad():
			output2 = model(x_modified, attention_mask=mask)
			
		# The outputs for the VALID positions (0, 1, 2) should be identical
		# The outputs for PADDED positions (3, 4) will be different because they process their own changed self-features
		# But valid tokens shouldn't attend to the padded garbage.
		
		valid_diff = (output1[0, :3] - output2[0, :3]).abs().max()
		
		# Tolerance for float precision
		assert valid_diff < 1e-6, f"Valid tokens were affected by changes in masked padding! Diff: {valid_diff}"

	def test_key_padding_mask_propagation(self):
		"""
		Integration: Verify parameters are passed correctly to the underlying layer.
		"""
		layer = TransformerLayer(hidden_dim=32, num_heads=4, ffn_dim=64)
		x = torch.randn(2, 10, 32)
		
		# Case 1: No mask
		out = layer(x, attention_mask=None)
		assert out.shape == x.shape
		
		# Case 2: Key padding mask (boolean: True for padding)
		# Note: In PyTorch notation for MultiheadAttention key_padding_mask, True = Padding
		# In our TransformerBackbone API, input mask is 1=Valid, 0=Padding.
		# We need to verify our conversion matches.
		
		# Let's test the layer directly with what we expect to pass it
		key_padding_mask = torch.zeros(2, 10, dtype=torch.bool) # All valid
		key_padding_mask[0, 5:] = True # First seq has 5 padding tokens
		
		out_masked = layer(x, key_padding_mask=key_padding_mask)
		assert out_masked.shape == x.shape

if __name__ == "__main__":
	pytest.main([__file__])
