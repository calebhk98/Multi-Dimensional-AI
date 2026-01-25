"""
Script to train single modalities (Audio, Voice, Motion) through the central Brain.
Uses masked inputs to train specific pathways in the MultiModalCreature.
"""

import argparse
import yaml
import torch
from torch.utils.data import DataLoader, Dataset
from typing import Dict, Any

from src.training.trainer import Trainer
from src.models.multimodal_transformer import MultiModalCreature
from src.data.synthetic_generator import SyntheticDataGenerator


class SyntheticDataset(Dataset):
    """Simple wrapper to satisfy DataLoader interface for synthetic generator."""
    def __init__(self, generator: SyntheticDataGenerator, modality: str, length: int = 1000):
        """
        Initialize SyntheticDataset.

        Args:
            generator (SyntheticDataGenerator): Generator instance.
            modality (str): Modality to generate data for.
            length (int): Virtual length of the dataset.
        """
        self.generator = generator
        self.modality = modality
        self.length = length
        
    def __len__(self):
        """
        Get dataset length.

        Returns:
            int: Virtual length of the dataset.
        """
        return self.length
        
    def __getitem__(self, idx):
        """
        Generate a single sample.

        Args:
            idx (int): Index (unused for synthetic data).

        Returns:
            Dict[str, Any]: Dictionary containing inputs and targets.
        """
        # Generate fresh sample every time
        sample = self.generator.generate_sample()
        inputs = sample["inputs"]
        targets = sample["targets"]
        
        # Filter for specific modality and map to MultiModalCreature keys
        if self.modality == "audio":
            return {
                "inputs": {
                    "audio_waveform": inputs["audio_waveform"].squeeze(0)
                },
                "targets": {
                    "audio": targets["audio"]
                }
            }
        elif self.modality == "voice_internal":
            return {
                "inputs": {
                    "internal_voice_tokens": inputs["internal_voice_tokens"]
                },
                "targets": {
                    "internal_text": targets["internal_text"]
                }
            }
        elif self.modality == "voice_external":
            return {
                "inputs": {
                    "external_voice_tokens": inputs["external_voice_tokens"]
                },
                "targets": {
                    "external_text": targets["external_text"]
                }
            }
        elif self.modality == "motion":
            return {
                "inputs": {
                    "joint_positions": inputs["joint_positions"],
                    "joint_rotations": inputs["joint_rotations"]
                },
                "targets": {
                    "animation": targets["animation"]
                }
            }
        else:
            raise ValueError(f"Unknown modality: {self.modality}")


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path (str): Path to the config file.

    Returns:
        Dict[str, Any]: Loaded configuration dictionary.
    """
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    """
    Main entry point for single momentum training.
    Parses arguments, sets up model and data, and runs training loop.
    """
    parser = argparse.ArgumentParser(description="Train single modality components through the Brain")
    parser.add_argument("--modality", type=str, required=True, 
                        choices=["audio", "voice_internal", "voice_external", "motion"],
                        help="Modality to train")
    parser.add_argument("--config", type=str, default="configs/single_modality_config.yaml",
                        help="Path to configuration file")
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    # Merge specific modality config into main Training config for convenience
    modality_cfg = config.get("modalities", {}).get(args.modality, {})
    training_cfg = config.get("training", {})
    # Override batch size if present in modality config
    if "batch_size" in modality_cfg:
        training_cfg["batch_size"] = modality_cfg["batch_size"]
        
    device = config.get("defaults", {}).get("device", "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training {args.modality} on {device}")
    
    # 1. Setup Model (Full Brain)
    # Ensure config has 'model' section with necessary params
    # We rely on configs/single_modality_config.yaml having a 'model' key
    
    model = MultiModalCreature(config)
    
    # 2. Setup Data
    generator = SyntheticDataGenerator(
        vocab_size=modality_cfg.get("vocab_size", 50257),
        sample_rate=modality_cfg.get("sample_rate", 16000),
        num_joints=modality_cfg.get("num_joints", 24),
        temporal_window=modality_cfg.get("temporal_window", 10),
        codebook_size=modality_cfg.get("codebook_size", 1024)
    )
    
    train_dataset = SyntheticDataset(generator, args.modality, length=training_cfg.get("max_steps", 1000) * training_cfg.get("batch_size", 4))
    train_loader = DataLoader(train_dataset, batch_size=training_cfg.get("batch_size", 4), shuffle=True)
    
    # 3. Trainer
    full_config = config
    
    trainer = Trainer(
        model=model,
        config=full_config,
        train_loader=train_loader,
        device=device
    )
    
    trainer.train()


if __name__ == "__main__":
    main()
