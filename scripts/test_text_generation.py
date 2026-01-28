"""
Test text generation with trained model.

Simple script to validate that text-only training worked by generating
text from a prompt using the trained MultiModalCreature model.

Usage:
	python scripts/test_text_generation.py --checkpoint checkpoints/text_only/model_final.pt
	python scripts/test_text_generation.py --prompt "The quick brown fox"
"""

import argparse
import torch
import yaml
from pathlib import Path
from transformers import GPT2Tokenizer


def load_model(checkpoint_path: str, device: str = "cuda"):
	"""
	Load trained model from checkpoint.
	
	Args:
		checkpoint_path: Path to .pt checkpoint file
		device: Device to load model on
	
	Returns:
		Loaded model in eval mode
	"""
	checkpoint_path = Path(checkpoint_path)
	
	if not checkpoint_path.exists():
		raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
	
	print(f"Loading model from {checkpoint_path}...")
	checkpoint = torch.load(checkpoint_path, map_location=device)
	
	# Import model class
	from src.models.multimodal_transformer import MultiModalCreature
	
	# Initialize model
	config = checkpoint.get("config", {})
	model = MultiModalCreature(config)
	model.load_state_dict(checkpoint["model_state_dict"])
	model.to(device)
	model.eval()
	
	print(f"Model loaded successfully on {device}")
	return model, config


def generate_text(
	model,
	tokenizer,
	prompt: str,
	max_new_tokens: int = 50,
	temperature: float = 0.8,
	top_k: int = 50,
	device: str = "cuda"
):
	"""
	Generate text from a prompt.
	
	Args:
		model: Trained MultiModalCreature model
		tokenizer: GPT2Tokenizer instance
		prompt: Input text prompt
		max_new_tokens: Number of tokens to generate
		temperature: Sampling temperature (higher = more random)
		top_k: Top-k sampling parameter
		device: Device to run inference on
	
	Returns:
		Generated text string
	"""
	# Encode prompt
	input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
	
	print(f"\nPrompt: {prompt}")
	print(f"Generating {max_new_tokens} tokens...\n")
	print(prompt, end="", flush=True)
	
	with torch.no_grad():
		for step in range(max_new_tokens):
			# Forward pass
			outputs = model(
				internal_voice_tokens=input_ids,
				return_hidden_states=True,
				temperature=temperature,
				top_k=top_k,
			)
			
			# Get next token logits
			next_token_logits = outputs["logits_internal_text"][0, -1, :]
			
			# Apply temperature
			if temperature != 1.0:
				next_token_logits = next_token_logits / temperature
			
			# Top-k sampling
			if top_k > 0:
				indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
				next_token_logits[indices_to_remove] = float('-inf')
			
			# Sample next token
			probs = torch.softmax(next_token_logits, dim=-1)
			next_token = torch.multinomial(probs, num_samples=1)
			
			# Append to sequence
			input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
			
			# Decode and print new token
			new_text = tokenizer.decode([next_token.item()])
			print(new_text, end="", flush=True)
	
	# Get full generated text
	generated_text = tokenizer.decode(input_ids[0])
	
	print("\n\nGeneration complete!")
	return generated_text


def main():
	"""Main entry point."""
	parser = argparse.ArgumentParser(description="Test text generation with trained model")
	parser.add_argument(
		"--checkpoint",
		type=str,
		default="checkpoints/text_only/model_final.pt",
		help="Path to model checkpoint (default: checkpoints/text_only/model_final.pt)"
	)
	parser.add_argument(
		"--prompt",
		type=str,
		default="Once upon a time",
		help="Text prompt to generate from"
	)
	parser.add_argument(
		"--max-tokens",
		type=int,
		default=50,
		help="Number of tokens to generate (default: 50)"
	)
	parser.add_argument(
		"--temperature",
		type=float,
		default=0.8,
		help="Sampling temperature (default: 0.8)"
	)
	parser.add_argument(
		"--top-k",
		type=int,
		default=50,
		help="Top-k sampling (default: 50)"
	)
	parser.add_argument(
		"--device",
		type=str,
		default="cuda" if torch.cuda.is_available() else "cpu",
		help="Device to use (default: cuda if available)"
	)
	args = parser.parse_args()
	
	print("=" * 80)
	print("Text Generation Test")
	print("=" * 80)
	
	# Load tokenizer
	print("\nLoading GPT-2 tokenizer...")
	tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
	
	# Load model
	model, config = load_model(args.checkpoint, args.device)
	
	# Generate text
	generated_text = generate_text(
		model=model,
		tokenizer=tokenizer,
		prompt=args.prompt,
		max_new_tokens=args.max_tokens,
		temperature=args.temperature,
		top_k=args.top_k,
		device=args.device
	)
	
	print("\n" + "=" * 80)
	print("Full Generated Text:")
	print("=" * 80)
	print(generated_text)
	print("=" * 80)


if __name__ == "__main__":
	main()
