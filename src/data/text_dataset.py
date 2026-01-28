"""
Text dataset loader for tokenized corpora.
Loads pre-tokenized data from disk and produces fixed-length sequences.
"""

from torch.utils.data import Dataset
import torch
import numpy as np
from pathlib import Path


class TextDataset(Dataset):
	"""
	Dataset for pre-tokenized text data.

	Loads token IDs from disk (.bin, .pt, .npy formats) and returns
	fixed-length sequences for language modeling.
	"""

	def __init__(self, token_file: str, seq_length: int = 512):
		"""
		Initialize TextDataset.

		Args:
			token_file: Path to tokenized data file (.bin, .pt, or .npy)
			seq_length: Context window size (default: 512)
		"""
		token_file = Path(token_file)

		if not token_file.exists():
			raise FileNotFoundError(f"Token file not found: {token_file}")

		# Load pre-tokenized data based on file extension
		if token_file.suffix == '.bin':
			self.tokens = np.fromfile(token_file, dtype=np.uint16)
		elif token_file.suffix == '.pt':
			self.tokens = torch.load(token_file)
			if isinstance(self.tokens, torch.Tensor):
				self.tokens = self.tokens.numpy()
		elif token_file.suffix == '.npy':
			self.tokens = np.load(token_file)
		else:
			raise ValueError(f"Unsupported format: {token_file.suffix}. Use .bin, .pt, or .npy")

		self.seq_length = seq_length

		# Calculate number of samples
		self.num_samples = len(self.tokens) // self.seq_length

		print(f"Loaded {len(self.tokens):,} tokens from {token_file}")
		print(f"Creating {self.num_samples:,} samples of length {seq_length}")

	def __len__(self):
		"""
		Return the number of samples in the dataset.

		Returns:
			int: Number of samples.
		"""
		return self.num_samples

	def __getitem__(self, idx):
		"""
		Get a training sample.

		Args:
			idx: Index of the sample to retrieve.

		Returns:
			dict with 'input' [seq_length] and 'target' [seq_length]
		"""
		start = idx * self.seq_length
		end = start + self.seq_length + 1  # +1 for target shift

		# Handle edge case at end of dataset
		if end > len(self.tokens):
			end = len(self.tokens)
			start = end - (self.seq_length + 1)

		chunk = self.tokens[start:end]

		# Input: tokens[0:seq_length], Target: tokens[1:seq_length+1]
		return {
			"input": torch.tensor(chunk[:-1], dtype=torch.long),
			"target": torch.tensor(chunk[1:], dtype=torch.long)
		}
