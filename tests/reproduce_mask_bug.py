
"""
Test script to reproduce the TransformerBackbone mask bug.
"""
import torch
from src.models.transformer_backbone import TransformerBackbone

def test_mask_bug():
	"""
	Test function to reproduce the 4D tensor mask error.
	"""
	print("Testing TransformerBackbone mask...")
	try:
		model = TransformerBackbone(
			num_layers=1,
			hidden_dim=32,
			num_heads=4,
			ffn_dim=128
		)
		
		bs = 2
		seq_len = 10
		x = torch.randn(bs, seq_len, 32)
		# Mask: 1 for valid, 0 for padding
		mask = torch.ones(bs, seq_len)
		mask[:, -2:] = 0 # Last 2 tokens are padding
		
		output = model(x, attention_mask=mask)
		print("Success! Output shape:", output.shape)
		
	except Exception as e:
		print(f"Caught expected error: {e}")
		# Identify if it is the shape error
		if "4-D" in str(e):
			print("Reproduction Confirmed: 4-D tensor error found.")
		else:
			print("Unexpected error message.")
			raise e

if __name__ == "__main__":
	test_mask_bug()
