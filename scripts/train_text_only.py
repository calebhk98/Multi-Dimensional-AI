"""
Text-only training script for Multi-Dimensional AI Creature.
Simple LLM-style training: text in, text out.
"""

import argparse

import torch
from torch.utils.data import DataLoader
from pathlib import Path
import sys
import os

# Add project root to path so we can import src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.multimodal_transformer import MultiModalCreature
from src.training.trainer import Trainer
from src.data.text_dataset import TextDataset
from src.data.text_only_dataset import TextOnlyDataset, text_only_collate_fn
from src.config import Config





def main():
	"""Main entry point."""
	# Enable TF32 for ~22% speedup on RTX 30xx/40xx GPUs
	torch.set_float32_matmul_precision("high")
	if torch.cuda.is_available():
		torch.backends.cuda.matmul.allow_tf32 = True
		torch.backends.cudnn.allow_tf32 = True
		torch.backends.cudnn.benchmark = True

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
	parser.add_argument(
		"--model-config",
		type=str,
		default="configs/model_1b.yaml",
		help="Path to model configuration YAML file"
	)
	args = parser.parse_args()

	# Load configuration
	print(f"Loading config from {args.config} and {args.model_config}...")
	config_obj = Config.from_files(
		training_config_path=args.config,
		model_config_path=args.model_config
	)
	
	# Flatten config into a single dictionary for compatibility with current codebase
	config = {}
	if config_obj.model:
		config.update(config_obj.model)
	if config_obj.training:
		config.update(config_obj.training)
	if config_obj.inference:
		config.update(config_obj.inference)


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
	data_cfg = config.get("data", {})
	num_workers = data_cfg.get("num_workers", 4)

	# Windows doesn't handle multiprocessing well
	if os.name == 'nt':
		num_workers = 0
		print("Warning: Windows detected, using num_workers=0")

	print(f"Creating DataLoader with batch_size={batch_size}, num_workers={num_workers}...")
	train_loader = DataLoader(
		dataset,
		batch_size=batch_size,
		shuffle=True,
		collate_fn=text_only_collate_fn,
		num_workers=num_workers,
		pin_memory=data_cfg.get("pin_memory", True),
		prefetch_factor=data_cfg.get("prefetch_factor", 2) if num_workers > 0 else None,
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
		print("Running in DRY RUN mode")
		print("Overriding config for safety/speed on local machine:")

		# Force safe limits to prevent RAM explosition
		config.setdefault("training", {})
		config["training"]["batch_size"] = 2
		config["training"]["max_steps"] = 5
		config["training"]["log_interval"] = 1
		config["training"]["save_interval"] = 10000000 # Disable saving
		# Also reduce gradient accumulation to avoid confusion
		config["training"]["gradient_accumulation_steps"] = 1
		
		print("- Batch size: 2")
		print("- Max steps: 5")
		print("- Saving disabled")
		
		# Set dry run save dir in multiple places to be safe
		# 1. Where Trainer looks: config["training"]["checkpointing"]["save_dir"]
		# 2. Where original code looked: config["defaults"]["save_dir"]
		
		dry_run_dir = "checkpoints/dry_run"
		
		# Trainer path
		config.setdefault("training", {}).setdefault("checkpointing", {})["save_dir"] = dry_run_dir
		
		# Defaults path (just in case)
		config.setdefault("defaults", {})["save_dir"] = dry_run_dir
		
		# Root checkpointing (config file structure)
		config.setdefault("checkpointing", {})["save_dir"] = dry_run_dir

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
