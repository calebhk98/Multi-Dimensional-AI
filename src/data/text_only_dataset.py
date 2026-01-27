"""
Text-only dataset adapter for MultiModalCreature model.
Converts text batches into the model's expected multi-modal batch format.
"""

import torch
from torch.utils.data import Dataset


class TextOnlyDataset(Dataset):
	"""
	Adapter that wraps TextDataset and returns model-compatible batch format.

	Converts simple text batches into the multi-modal input/target structure
	expected by MultiModalCreature, using only internal_voice_tokens as input
	and internal_text as target.
	"""

	def __init__(self, text_dataset):
		"""
		Initialize TextOnlyDataset adapter.

		Args:
			text_dataset: Instance of TextDataset
		"""
		self.text_dataset = text_dataset

	def __len__(self):
		return len(self.text_dataset)

	def __getitem__(self, idx):
		"""
		Get a sample in model-compatible format.

		Returns:
			dict with 'inputs' and 'targets' in multi-modal format
		"""
		sample = self.text_dataset[idx]

		# Convert to model's expected format
		return {
			"inputs": {
				"internal_voice_tokens": sample["input"]
			},
			"targets": {
				"internal_text": sample["target"]
			}
		}


def text_only_collate_fn(batch):
	"""
	Collate text-only batches into model-compatible format.

	Args:
		batch: List of samples from TextOnlyDataset

	Returns:
		dict with nested 'inputs' and 'targets' dictionaries
	"""
	inputs = torch.stack([item["inputs"]["internal_voice_tokens"] for item in batch])
	targets = torch.stack([item["targets"]["internal_text"] for item in batch])

	return {
		"inputs": {
			"internal_voice_tokens": inputs
		},
		"targets": {
			"internal_text": targets
		}
	}
