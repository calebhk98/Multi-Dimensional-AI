"""
Generate dummy tokenized data for testing/dry-runs.
Creates a random sequence of tokens saved as a .bin file.
"""

import argparse
import numpy as np
from pathlib import Path


def generate_dummy_data(output_file: str, num_tokens: int = 100000, vocab_size: int = 50257):
	"""
	Generate random tokens and save to binary file.
	
	Purpose:
		Create a dummy dataset for testing the training pipeline without real data.

	Workflow:
		1. Generate random integers within vocabulary range.
		2. Ensure output directory exists.
		3. Save as raw binary file (.bin).

	Args:
		output_file: Path to save .bin file
		num_tokens: Number of tokens to generate
		vocab_size: Size of vocabulary (high limit for random tokens)
		
	Returns:
		None
	"""
	output_path = Path(output_file)
	
	print(f"Generating {num_tokens:,} random tokens (vocab={vocab_size})...")
	
	# Generate random tokens [0, vocab_size)
	# Use uint16 (max 65535, fits GPT-2 vocab 50257)
	tokens = np.random.randint(0, vocab_size, size=num_tokens, dtype=np.uint16)
	
	# Create directory
	output_path.parent.mkdir(parents=True, exist_ok=True)
	
	# Save
	print(f"Saving to {output_path}...")
	tokens.tofile(output_path)
	
	print(f"Done! File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")


def main():
	"""
	Main entry point.

	Purpose:
		Parse arguments and run dummy data generation.

	Workflow:
		1. Parse command line arguments.
		2. Call generate_dummy_data.
	"""
	parser = argparse.ArgumentParser(description="Generate dummy training data")
	parser.add_argument("--output", type=str, default="data/dummy_corpus.bin", help="Output path")
	parser.add_argument("--tokens", type=int, default=1000000, help="Number of tokens (default: 1M)")
	parser.add_argument("--vocab", type=int, default=50257, help="Vocab size (default: 50257)")
	
	args = parser.parse_args()
	generate_dummy_data(args.output, args.tokens, args.vocab)


if __name__ == "__main__":
	main()
