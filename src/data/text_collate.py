"""
Collate function for batching text samples.
"""

import torch


def text_collate_fn(batch):
	"""
	Collate text samples into batched tensors.

	Args:
		batch: List of samples from TextDataset

	Returns:
		dict with 'inputs' [B, T] and 'targets' [B, T]
	"""
	inputs = torch.stack([item["input"] for item in batch])
	targets = torch.stack([item["target"] for item in batch])

	return {
		"inputs": inputs,
		"targets": targets
	}
