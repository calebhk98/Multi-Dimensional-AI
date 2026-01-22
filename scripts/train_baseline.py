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
    print("Creating dataset...")
    dataset = CreatureDataset(length=args.steps * 16, synthetic=True) # Ensure enough data
    dataloader = DataLoader(
        dataset, 
        batch_size=8, # Training config might say 16, using small for local test
        shuffle=True, 
        collate_fn=collate_fn
    )
    
    # Filter inputs/targets logic logic could go into a custom Collate 
    # OR we just modify the Trainer/loss weights dynamically.
    
    # Let's adjust loss weights to ignore other modalities
    loss_weights = {
        "internal_text": 0.0,
        "external_text": 0.0,
        "audio": 0.0,
        "animation": 0.0,
    }
    
    if args.modality == "text":
        loss_weights["internal_text"] = 1.0
        loss_weights["external_text"] = 1.0 
    elif args.modality == "audio":
        loss_weights["audio"] = 1.0
    elif args.modality == "proprioception":
        loss_weights["animation"] = 1.0
    elif args.modality == "vision":
        # Vision has no direct decoder in current architecture (it informs other actions)
        # To verify it trains, we can attach a dummy auxiliary loss or just verify forward pass.
        # For this baseline, we'll just enable internal text loss driven by vision input
        # to ensure gradients flow back to the visual encoder.
        loss_weights["internal_text"] = 1.0
    
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
    
if __name__ == "__main__":
    main()
