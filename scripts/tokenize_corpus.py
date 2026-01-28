"""
Tokenize a text corpus for training.
Converts raw text to token IDs using GPT-2 tokenizer.
"""

import argparse
import numpy as np
from pathlib import Path
from transformers import GPT2Tokenizer


def tokenize_corpus(input_file: str, output_file: str):
	"""
	Tokenize a text file and save as binary token IDs.

	Args:
		input_file: Path to input .txt file
		output_file: Path to output .bin file
	"""
	input_path = Path(input_file)
	output_path = Path(output_file)

	if not input_path.exists():
		raise FileNotFoundError(f"Input file not found: {input_path}")

	print(f"Loading tokenizer...")
	tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

	print(f"Reading text from {input_path}...")
	with open(input_path, 'r', encoding='utf-8') as f:
		text = f.read()

	print(f"Tokenizing {len(text):,} characters...")
	tokens = tokenizer.encode(text)

	print(f"Saving {len(tokens):,} tokens to {output_path}...")
	tokens_np = np.array(tokens, dtype=np.uint16)

	# Create output directory if it doesn't exist
	output_path.parent.mkdir(parents=True, exist_ok=True)

	# Save as binary file
	tokens_np.tofile(output_path)

	print(f"\nTokenization complete!")
	print(f"  Input: {len(text):,} characters")
	print(f"  Output: {len(tokens):,} tokens")
	print(f"  Compression ratio: {len(text) / len(tokens):.2f} chars/token")
	print(f"  File size: {output_path.stat().st_size / 1024:.2f} KB")


def main():
	"""Main entry point."""
	parser = argparse.ArgumentParser(description="Tokenize text corpus for training")
	parser.add_argument(
		"--input",
		type=str,
		default="data/corpus.txt",
		help="Input text file (default: data/corpus.txt)"
	)
	parser.add_argument(
		"--output",
		type=str,
		default="data/corpus.bin",
		help="Output token file (default: data/corpus.bin)"
	)
	args = parser.parse_args()

	tokenize_corpus(args.input, args.output)


if __name__ == "__main__":
	main()
