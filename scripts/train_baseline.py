"""
Script to train a single modality baseline.
"""

import argparse
import torch
from torch.utils.data import DataLoader
from src.config import Config
from src.models import MultiModalCreature
from src.data import CreatureDataset, collate_fn  # Need to ensure dataset.py has collate_fn
from src.training import Trainer

def main():
	"""
	==============================================================================
	Function: main
	==============================================================================
	Purpose:  Main entry point for the training script. Parses command-line arguments,
			initializes the configuration, model, and dataset, and launches the
			training loop for a specified single modality baseline.

	Parameters:
		- None

	Returns:
		None

	Dependencies:
		- argparse
		- src.config.Config
		- src.models.MultiModalCreature
		- src.data.CreatureDataset
		- src.training.Trainer
		- torch.utils.data.DataLoader

	Processing Workflow:
		1.  Parse command-line arguments (modality, config, steps).
		2.  Load training configuration from file.
		3.  Override max steps and save interval for baseline testing.
		4.  Initialize the `MultiModalCreature` model.
		5.  Freezing logic explanation (note: we do not freeze, but mask loss).
		6.  Create synthetic `CreatureDataset` and `DataLoader`.
		7.  Configure loss weights to isolate the target modality:
			- text: internal and external text loss = 1.0
			- audio: audio loss = 1.0
			- proprioception: animation loss = 1.0
			- vision: internal text loss = 1.0 (proxy)
		8.  Inject loss weights into config/model.
		9.  Initialize `Trainer` and start training.

	ToDo:
		- Implement actual parameter freezing for stricter baselines.

	Usage:
		python scripts/train_baseline.py --modality audio --steps 50
	==============================================================================
	"""
	parser = argparse.ArgumentParser(description="Train single modality baseline")
	parser.add_argument("--modality", type=str, required=True, 
			choices=["text", "audio", "vision", "proprioception"],
			help="Modality to train")
	parser.add_argument("--config", type=str, default="configs/training_config.yaml",
			help="Path to training config")
	parser.add_argument("--steps", type=int, default=100, help="Number of training steps")
	args = parser.parse_args()
	
	# Load config
	# For now, creating a dummy config if files don't exist, or loading default
	# Assuming the config files we created exist
	config = Config.from_files(training_config_path=args.config)
	
	# Override max steps for quick baseline test
	if config.training is None: config.training = {}
	config.training["max_steps"] = args.steps
	config.training["save_interval"] = args.steps # Save only at end
	
	# Initialize model
	print("Initializing model...")
	model = MultiModalCreature(config.__dict__) # Pass all configs
	
	# Freeze parameters not related to the target modality
	# This is a simplification. For a true baseline we might just want to 
	# zero out other inputs or ignore their losses.
	# But freezing ensures we are strictly testing the gradient flow of relevant components.
	
	# However, since everything goes through the shared transformer, 
	# we can't freeze the transformer if we want to train!
	# So "Baseline" here means: feed ONLY this modality, compute loss ONLY for this modality.
	
	# Create dataset (Synthetic)
	# Create dataset (Synthetic)
	print("Creating dataset...")
	dataset = CreatureDataset(length=args.steps * 16, synthetic=True) # Ensure enough data
	dataloader = DataLoader(
		dataset, 
		batch_size=8, # Training config might say 16, using small for local test
		shuffle=True, 
		collate_fn=collate_fn
	)
	
	# Configure loss weights based on modality
	loss_weights = _configure_loss_weights(args.modality)
	
	# Inject loss weights into config/model
	if config.model is None: config.model = {}
	config.model["loss_weights"] = loss_weights
	
	# Ensure model config is updated
	if hasattr(model, 'config'):
		model.config["loss_weights"] = loss_weights
	else:
		print("Warning: Could not update model config with loss weights")
	
	print(f"Starting training for modality: {args.modality}")
	trainer = Trainer(model, config.__dict__, dataloader)
	trainer.train()

def _configure_loss_weights(modality):
	"""
	==============================================================================
	Function: _configure_loss_weights
	==============================================================================
	Purpose:  Sets up loss weights dictionary based on target modality.

	Parameters:
		- modality: str
			The target modality ("text", "audio", "proprioception", "vision").

	Returns:
		Dict[str, float] - Loss weights dictionary.
	==============================================================================
	"""
	loss_weights = {
		"internal_text": 0.0,
		"external_text": 0.0,
		"audio": 0.0,
		"animation": 0.0,
	}
	
	if modality == "text":
		loss_weights["internal_text"] = 1.0
		loss_weights["external_text"] = 1.0 
	elif modality == "audio":
		loss_weights["audio"] = 1.0
	elif modality == "proprioception":
		loss_weights["animation"] = 1.0
	elif modality == "vision":
		# Vision has no direct decoder in current architecture (it informs other actions)
		# To verify it trains, we can attach a dummy auxiliary loss or just verify forward pass.
		# For this baseline, we'll just enable internal text loss driven by vision input
		# to ensure gradients flow back to the visual encoder.
		loss_weights["internal_text"] = 1.0
		
	return loss_weights

if __name__ == "__main__":
	main()
