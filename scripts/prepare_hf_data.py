"""
Prepare HuggingFace Dataset for Training.
Loads a dataset (from cache or download), tokenizes it, and saves it as a binary file.
"""

import argparse
import numpy as np
import os
from pathlib import Path
from datasets import load_dataset
from transformers import GPT2Tokenizer
from tqdm import tqdm

def prepare_dataset(dataset_name: str, config_name: str, cache_dir: str, output_file: str, max_tokens: int = None, split: str = "train"):
	"""
	Load, tokenize, and save dataset.
	
	Purpose:
		Downloads (or loads from cache) a HuggingFace dataset, tokenizes it using GPT-2 tokenizer,
		and saves the result as a raw binary file of uint16 tokens.

	Workflow:
		1. Load dataset using HuggingFace `datasets`.
		2. Initialize GPT-2 tokenizer.
		3. Map `batch_tokenize` over the dataset in parallel.
		4. Iterate over tokenized dataset and write chunks to binary file.

	Args:
		dataset_name: Name of the HF dataset (e.g. 'wikimedia/wikipedia')
		config_name: Config name (e.g. '20231101.en')
		cache_dir: Directory where dataset is cached
		output_file: Output .bin file path
		max_tokens: Optional limit on number of tokens
		split: Dataset split to load
		
	Returns:
		None
	"""
	print(f"Loading dataset {dataset_name} ({config_name}) from {cache_dir}...")
	
	# Ensure cache dir exists as a path object, providing absolute path if needed
	if cache_dir:
		cache_path = Path(cache_dir).resolve()
	else:
		cache_path = None

	try:
		dataset = load_dataset(dataset_name, config_name, cache_dir=str(cache_path) if cache_path else None, split=split)
	except Exception as e:
		print(f"Error loading dataset: {e}")
		print("Note: If the dataset is not fully cached, internet access may be required.")
		return

	print(f"Dataset loaded. Size: {len(dataset)} rows.")
	
	print("Loading tokenizer...")
	tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
	# Silence tokenizer warnings and allow arbitrary length processing
	if tokenizer.pad_token is None:
		tokenizer.pad_token = tokenizer.eos_token
	tokenizer.model_max_length = 100_000_000 # Suppress "token indices sequence length is longer than..." warning
	
	def batch_tokenize(examples):
		"""
		Tokenize a batch of text examples.
		
		Args:
			examples: Dict of lists of text.
			
		Returns:
			Dict containing list of token lists.
		"""
		return {"tokens": [tokenizer.encode(text) for text in examples["text"] if text]}

	print(f"Tokenizing with {os.cpu_count()} processes...")
	# Use batched=True and num_proc for parallel processing
	tokenized_ds = dataset.map(
		batch_tokenize,
		batched=True,
		num_proc=max(1, os.cpu_count() - 1), # Leave one core free
		remove_columns=dataset.column_names, # Drop original text to save memory
		desc="Tokenizing"
	)

	token_count = 0
	output_path = Path(output_file)
	output_path.parent.mkdir(parents=True, exist_ok=True)
	
	print(f"Writing to {output_path}...")
	
	# Open file in write binary mode
	with open(output_path, "wb") as f:
		# Iterate over the pre-tokenized dataset
		buffer = []
		BATCH_WRITE_SIZE = 1_000_000
		
		for batch in tqdm(tokenized_ds, desc="Saving"):
			tokens = batch['tokens']
			buffer.extend(tokens)
			token_count += len(tokens)
			
			if len(buffer) >= BATCH_WRITE_SIZE:
				arr = np.array(buffer, dtype=np.uint16)
				f.write(arr.tobytes())
				buffer = []
				
			if max_tokens and token_count >= max_tokens:
				print(f"Reached max tokens limit ({max_tokens}).")
				break
		
		# Write remaining
		if buffer:
			arr = np.array(buffer, dtype=np.uint16)
			f.write(arr.tobytes())
			
	print(f"\nPreparation complete!")
	print(f"Total tokens processed: {token_count:,}")
	print(f"Saved to: {output_path}")

def main():
	"""
	Main entry point.
	
	Purpose:
		Parse command line arguments and run dataset preparation.
		
	Workflow:
		1. Define arguments.
		2. Parse arguments.
		3. Call prepare_dataset.
	"""
	parser = argparse.ArgumentParser(description="Convert HuggingFace dataset to binary for training")
	parser.add_argument("--dataset", type=str, default="wikimedia/wikipedia", help="Dataset name")
	parser.add_argument("--config", type=str, default="20231101.en", help="Dataset config")
	parser.add_argument("--cache_dir", type=str, default=None, help="Path to HF cache directory")
	parser.add_argument("--output", type=str, required=True, help="Output .bin file path")
	parser.add_argument("--max_tokens", type=int, default=None, help="Stop after N tokens")
	parser.add_argument("--split", type=str, default="train", help="Dataset split")
	
	args = parser.parse_args()
	
	prepare_dataset(args.dataset, args.config, args.cache_dir, args.output, args.max_tokens, args.split)

if __name__ == "__main__":
	main()
