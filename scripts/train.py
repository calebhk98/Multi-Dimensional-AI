"""
Unified training script for Multi-Dimensional AI Creature.
Supports various phases (Single Modality, Pairwise Integration, etc.) via configuration.
"""

import argparse
import yaml
import torch
from torch.utils.data import DataLoader
from typing import Dict, Any
from pathlib import Path

from src.models.multimodal_transformer import MultiModalCreature
from src.training.trainer import Trainer
from src.data.synthetic_generator import SyntheticDataGenerator
from src.data.pairwise_dataset import PairwiseDataset, pairwise_collate_fn
from src.data.multimodal_dataset import MultiModalDataset, multimodal_collate_fn

def load_config(config_path: str) -> Dict[str, Any]:
	"""Load configuration from YAML file."""
	with open(config_path, "r") as f:
		return yaml.safe_load(f)

def create_dataset(config: Dict[str, Any]):
	"""
	Create dataset based on configuration.
	Currently defaults to Synthetic data generator.
	"""
	# 1. Setup Generator
	# Check if 'pairwise' or 'single_modality' specific configs exist, else use defaults
	if "pairwise" in config:
		dataset_cfg = config["pairwise"].get("dataset", {})
	else:
		# Fallback to general model config or defaults
		dataset_cfg = {} 
		# Extract from model.encoders if possible? 
		# tailored for generator init signature
	
	# Initialize Generator with safe defaults or config values
	generator = SyntheticDataGenerator(
		vocab_size=dataset_cfg.get("vocab_size", 
			config.get("model", {}).get("encoders", {}).get("internal_voice", {}).get("vocab_size", 50257)
		),
		sample_rate=dataset_cfg.get("sample_rate", 16000),
		image_size=dataset_cfg.get("image_size", 224),
		num_joints=dataset_cfg.get("num_joints", 24),
		temporal_window=dataset_cfg.get("temporal_window", 10),
		codebook_size=dataset_cfg.get("codebook_size", 
			config.get("model", {}).get("encoders", {}).get("audio", {}).get("codebook_size", 
				config.get("model", {}).get("decoders", {}).get("audio", {}).get("codebook_size", 1024)
			)
		)
	)
	
	# 2. Setup Dataset Wrapper
	training_cfg = config.get("training", {})
	max_steps = training_cfg.get("max_steps", 1000)
	batch_size = training_cfg.get("batch_size", 4)
	virtual_length = max_steps * batch_size
	
	if "pairwise" in config:
		print("Initializing PairwiseDataset...")
		pairs = config["pairwise"]["pairs"]
		# Convert list of lists to list of tuples if needed
		pairs = [tuple(p) for p in pairs]
		return PairwiseDataset(generator, pairs, length=virtual_length)
		
	elif "multimodal" in config or "stage5" in config:
		print("Initializing MultiModalDataset...")
		dataset_cfg = config.get("multimodal", {}).get("dataset", {})
		# Re-init generator with multimodal specific params if needed, 
		# or assuming the default one created above is sufficient.
		# Ideally we might want to pass config to generator creation earlier if params differ.
		return MultiModalDataset(generator, length=virtual_length)

	else:
		# Default or Single Modality fallback (could import SyntheticDataset from train_single_modality if needed)
		# For now, let's assume this script targets Phase 4+ or we implement a GenericSyntheticDataset if needed.
		# But for Phase 4, we only need PairwiseDataset support.
		# raise NotImplementedError("This script currently only supports 'pairwise' configuration section.")
		print("Warning: No specific dataset config found. Defaulting to MultiModalDataset (Stage 5).")
		return MultiModalDataset(generator, length=virtual_length)

def main():
	parser = argparse.ArgumentParser(description="Multi-Dimensional AI Creature Training")
	parser.add_argument("--config", type=str, required=True, help="Path to configuration file")
	parser.add_argument("--dry-run", action="store_true", help="Run a short training loop for verification")
	args = parser.parse_args()
	
	# Load Config
	config = load_config(args.config)
	
	# Setup Device
	requested_device = config.get("defaults", {}).get("device", "cuda")
	if requested_device == "cuda" and not torch.cuda.is_available():
		print("Warning: CUDA requested but not available. Falling back to CPU.")
		device = "cpu"
	else:
		device = requested_device
		
	print(f"Using device: {device}")
	
	# Setup Model
	print("Initializing MultiModalCreature...")
	model = MultiModalCreature(config)
	
	# Setup Data
	print("Creating Dataset...")
	train_dataset = create_dataset(config)
	
	training_cfg = config.get("training", {})
	batch_size = training_cfg.get("batch_size", 4)
	
	collate_fn = None
	if "pairwise" in config:
		collate_fn = pairwise_collate_fn
	else:
		collate_fn = multimodal_collate_fn

	train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
	
	# Handle Dry Run
	if args.dry_run:
		print("Dry Run: Overriding max_steps to 10")
		config["training"]["max_steps"] = 10
		config["training"]["save_interval"] = 10 # Save once
		config["defaults"]["save_dir"] = "checkpoints/dry_run"
	
	# Setup Trainer
	print("Starting Trainer...")
	trainer = Trainer(
		model=model,
		config=config,
		train_loader=train_loader,
		device=device
	)
	
	# Train
	trainer.train()

if __name__ == "__main__":
	main()
