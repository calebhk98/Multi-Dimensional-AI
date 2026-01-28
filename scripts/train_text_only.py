"""
Text-only training script for Multi-Dimensional AI Creature.
Simple LLM-style training: text in, text out.
"""

import argparse
import yaml
import torch
from torch.utils.data import DataLoader
from pathlib import Path

from src.models.multimodal_transformer import MultiModalCreature
from src.training.trainer import Trainer
from src.data.text_dataset import TextDataset
from src.data.text_only_dataset import TextOnlyDataset, text_only_collate_fn


def load_config(config_path: str):
	"""
	Load YAML configuration file.

	Args:
		config_path: Path to config file.

	Returns:
		dict: Loaded configuration.
	"""
	with open(config_path, 'r') as f:
		return yaml.safe_load(f)


def main():
	"""Main entry point."""
	parser = argparse.ArgumentParser(description="Text-only training for Multi-Dimensional AI")
	parser.add_argument(
		"--config",
		type=str,
		required=True,
		help="Path to configuration YAML file"
	)
	parser.add_argument(
		"--data",
		type=str,
		required=True,
		help="Path to tokenized data file (.bin, .pt, or .npy)"
	)
	parser.add_argument(
		"--dry-run",
		action="store_true",
		help="Run only 10 steps for quick validation"
	)
	args = parser.parse_args()

	# Load configuration
	print(f"Loading config from {args.config}...")
	config = load_config(args.config)

	# Setup device
	requested_device = config.get("defaults", {}).get("device", "cuda")
	if requested_device == "cuda" and not torch.cuda.is_available():
		print("Warning: CUDA requested but not available. Falling back to CPU.")
		device = "cpu"
	else:
		device = requested_device

	print(f"Using device: {device}")

	# Create dataset
	print(f"\nCreating dataset from {args.data}...")
	seq_length = config["model"]["encoders"]["internal_voice"]["max_seq_length"]
	text_dataset = TextDataset(args.data, seq_length=seq_length)
	dataset = TextOnlyDataset(text_dataset)

	# Create dataloader
	batch_size = config["training"]["batch_size"]
	print(f"Creating DataLoader with batch_size={batch_size}...")
	train_loader = DataLoader(
		dataset,
		batch_size=batch_size,
		shuffle=True,
		collate_fn=text_only_collate_fn,
		num_workers=0  # Set to 0 for debugging, increase for performance
	)

	# Create model
	print("\nInitializing MultiModalCreature model...")
	model = MultiModalCreature(config)

	# Count parameters
	total_params = sum(p.numel() for p in model.parameters())
	trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
	print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")

	# Dry run override
	if args.dry_run:
		print("\n*** DRY RUN MODE: Training for only 10 steps ***")
		config["training"]["max_steps"] = 10
		config["training"]["log_interval"] = 1
		config["training"]["save_interval"] = 10
		config["defaults"]["save_dir"] = "checkpoints/dry_run"

	# Create trainer
	print("\nInitializing Trainer...")
	trainer = Trainer(
		model=model,
		config=config,
		train_loader=train_loader,
		device=device
	)

	# Start training
	print("\n" + "="*80)
	print("Starting training...")
	print("="*80 + "\n")

	trainer.train()

	print("\n" + "="*80)
	print("Training complete!")
	print("="*80)


if __name__ == "__main__":
	main()
