"""
Tests for the HuggingFace dataset preparation script.
"""

import pytest
import numpy as np
import tempfile
import os
from pathlib import Path
from unittest.mock import MagicMock, patch
from scripts.prepare_hf_data import prepare_dataset

class TestPrepareHFData:
	
	@pytest.fixture
	def mock_dataset(self):
		"""
		Create a mock HuggingFace dataset.

		Returns:
			List of dicts representing dataset rows.
		"""
		mock_ds = [
			{"text": "Hello world"},
			{"text": "This is a test"},
			{"text": ""} # Empty line handling
		]
		return mock_ds

	@patch('scripts.prepare_hf_data.load_dataset')
	@patch('scripts.prepare_hf_data.GPT2Tokenizer')
	def test_prepare_dataset(self, mock_tokenizer_cls, mock_load_dataset, mock_dataset):
		"""
		Test the preparation logic with mocked dataset and tokenizer.

		Purpose:
			Verify that dataset is correctly loaded, mapped, and saved to binary format.

		Workflow:
			1. Mock dataset loading and mapping.
			2. Mock tokenizer encoding.
			3. Run prepare_dataset.
			4. Verify output file exists and has correct content.

		Args:
			mock_tokenizer_cls: Mocked tokenizer class.
			mock_load_dataset: Mocked load_dataset function.
			mock_dataset: Fixture dataset.

		ToDo:
			None
		"""
		# Setup mocks
		# partial mock for dataset to support map
		mock_ds_obj = MagicMock()
		# map should return the processed dataset (iterable of dicts with 'tokens')
		# We simulate what batch_tokenize produces
		# "Hello world" -> [72, ...]
		
		def fake_map(function, batched=False, **kwargs):
			"""
			Simulate dataset.map functionality.
			
			Args:
				function: Function to apply.
				batched: Whether to batch.
				**kwargs: Additional args.
				
			Returns:
				List of processed examples.
			"""
			# Apply function to our mock data
			# map in script is batched=True
			if batched:
				# function expects dict of lists
				texts = [x['text'] for x in mock_dataset]
				result = function({"text": texts})
				# result is {"tokens": [[...], [...]]}
				# map returns a dataset, which is iterable. 
				# We return a list of dicts: [{'tokens': [...]}, ...]
				return [{"tokens": t} for t in result['tokens']]
			return []

		mock_ds_obj.map.side_effect = fake_map
		mock_ds_obj.column_names = ["text"] # required for remove_columns
		mock_load_dataset.return_value = mock_ds_obj
		
		# Mock tokenizer encode to just return ASCII values for simplicity
		mock_tokenizer = MagicMock()
		mock_tokenizer.encode.side_effect = lambda x: [ord(c) for c in x]
		mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer
		
		# Create temp output file
		with tempfile.TemporaryDirectory() as temp_dir:
			output_file = Path(temp_dir) / "output.bin"
			
			prepare_dataset(
				dataset_name="test_dataset",
				config_name="test_config",
				cache_dir="test_cache",
				output_file=str(output_file),
				max_tokens=None
			)
			
			# Verify file exists
			assert output_file.exists()
			
			# Read back and verify content
			# "Hello world" -> [72, 101, 108, 108, 111, 32, 119, 111, 114, 108, 100]
			# "This is a test" -> [84, 104, 105, 115, 32, 105, 115, 32, 97, 32, 116, 101, 115, 116]
			data = np.fromfile(output_file, dtype=np.uint16)
			
			expected_len = len("Hello world") + len("This is a test")
			assert len(data) == expected_len
			assert data[0] == ord('H')
			assert data[-1] == ord('t')

	@patch('scripts.prepare_hf_data.load_dataset')
	def test_max_tokens_limit(self, mock_load_dataset):
		"""
		Test that max_tokens argument stops processing early.

		Purpose:
			Ensure the script respects the max_tokens limit and stops outputting data.

		Workflow:
			1. Create infinite/large data source.
			2. Mock tokenizer.
			3. Run prepare_dataset with max_tokens limit.
			4. Verify output size is within limit.

		Args:
			mock_load_dataset: Mocked load_dataset function.

		ToDo:
			None
		"""
		# Infinite generator of data
		def infinite_data():
			"""
			Generate infinite stream of dummy data.
			
			Returns:
				Yields dummy data dicts.
			"""
			while True:
				yield {"text": "A" * 10} # 10 tokens per row
				
		# We need a finite slice for the mock to iterate over usually, 
		# but the script iterates over the returned object.
		# Let's just give it a large list.
		large_ds = [{"text": "A" * 10} for _ in range(100)]
		mock_load_dataset.return_value = large_ds
		
		with tempfile.TemporaryDirectory() as temp_dir:
			output_file = Path(temp_dir) / "output_limit.bin"
			
			with patch('scripts.prepare_hf_data.GPT2Tokenizer') as mock_tok_cls:
				mock_tok = MagicMock()
				mock_tok.encode.return_value = [65] * 10
				mock_tok_cls.from_pretrained.return_value = mock_tok
				
				prepare_dataset(
					dataset_name="test",
					config_name="test",
					cache_dir="test",
					output_file=str(output_file),
					max_tokens=55 # Should parse 6 rows: 60 tokens (checked after row) OR check inside? 
					# Implementation checks: if token_count >= max_tokens: break.
					# 10 -> 20 -> 30 -> 40 -> 50 -> 60 (Break)
				)
				
			data = np.fromfile(output_file, dtype=np.uint16)
			# The script writes the full buffer OF THE LAST ROW even if it goes slightly over, 
			# Or breaks exactly.
			# Code: 
			# current_buffer.extend(tokens)
			# if count >= max: break
			# So it will include the full last row.
			
			assert len(data) >= 55
			assert len(data) <= 60 # 6 rows * 10 tokens
