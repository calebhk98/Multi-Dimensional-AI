"""
Dataset wrapper for data generation.
"""

import torch
from torch.utils.data import Dataset
from typing import Dict, Any
from src.data.synthetic_generator import SyntheticDataGenerator

class CreatureDataset(Dataset):
	"""
	PyTorch Dataset that yields multi-modal creature data.
	Can wrap real data (future) or use SyntheticDataGenerator (current).
	"""
	
	def __init__(
		self,
		length: int = 1000,
		synthetic: bool = True,
		config: Dict[str, Any] = None,
	):
		"""
		Args:
			length: Virtual length of the dataset (number of samples to generate)
			synthetic: Whether to use synthetic data
			config: Configuration dict
		"""
		self.length = length
		self.synthetic = synthetic
		self.generator = SyntheticDataGenerator() # Could pass config params here
		
	def __len__(self) -> int:
		"""
		Get the length of the dataset.

		Returns:
			Number of samples in the dataset
		"""
		return self.length
	
	def __getitem__(self, idx: int) -> Dict[str, Any]:
		"""
		Get a single sample.

		Args:
			idx: Index of sample to retrieve

		Returns:
			Dictionary of inputs and targets
		
		Raises:
			NotImplementedError: If synthetic is False (real data not implemented)
		"""
		if self.synthetic:
			return self.generator.generate_sample()
		else:
			raise NotImplementedError("Real data loading not implemented yet")

def collate_fn(batch):
	"""
	Custom collate function to handle dictionary structure and variable lengths.

	Args:
		batch: List of samples (dictionaries) from the dataset

	Returns:
		Batched dictionary with stacked tensors
	"""
	# Simplified collate that assumes generator produces fixed sizes or we let PyTorch default handle simple cases
	# For a robust implementation, we'd need detailed padding logic for text sequences.
	# For now, we will rely on PyTorch's default_collate but manually handle the nested dicts if needed.
	# Actually, standard default_collate works well with nested dicts of tensors.
	# The only issue is variable length sequences (text).
	# We will assume fixed length or pad in generator for now to keep baseline simple.
	
	from torch.utils.data.dataloader import default_collate
	return default_collate(batch)
