
"""
Script to run training on real dataset.
"""
import sys
import logging
from pathlib import Path
import torch
from torch.utils.data import DataLoader

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.config import Config
from src.data.real_dataset import RealMultiModalDataset
from src.data.multimodal_dataset import multimodal_collate_fn
from src.models.multimodal_transformer import MultiModalCreature
from src.training.trainer import Trainer

def main():
    """
    Main entry point for real-data training.

    Purpose:
        Orchestrate the training process using real-world multi-modal data.
        Loads configuration, dataset, model, and starts the trainer.

    Workflow:
        1. Load real training configuration
        2. Initialize RealMultiModalDataset
        3. Create DataLoader
        4. Initialize MultiModalCreature model
        5. Initialize and run Trainer

    ToDo:
        - Add checkpoint resumption argument
    """
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # 1. Load Config
    config_path = "configs/real_training_config.yaml"
    config = Config.load("configs/training_config.yaml") # Load base first
    real_config = Config.load(config_path) # Then specific
    
    # Merge for training (simple merge for now)
    # real_config typically contains 'dataset' and specific 'training' params
    real_ds_config = real_config.get("dataset", {})
    training_config = real_config.get("training", {})
    model_config = config.model # Model config still comes from the base config
    
    logger.info(f"Loading dataset from {real_ds_config.get('root_dir')}")
    
    # 2. Setup Dataset & Loader
    dataset = RealMultiModalDataset(
        root_dir=real_ds_config.get("root_dir", "dataset_raw")
    )
    
    if len(dataset) == 0:
        logger.warning(f"No data found in {real_ds_config.get('root_dir')}. Training will exit early.")
        
    loader = DataLoader(
        dataset,
        batch_size=real_ds_config.get("batch_size", 4),
        shuffle=real_ds_config.get("shuffle", True),
        num_workers=real_ds_config.get("num_workers", 0),
        collate_fn=multimodal_collate_fn
    )
    
    # 3. Setup Model
    # MultiModalCreature expects a dict containing a "model" key
    full_config = {"model": model_config, "training": training_config}
    
    try:
        model = MultiModalCreature(config=full_config)
    except TypeError as e:
        logger.error(f"Failed to init model: {e}")
        return

    # 4. Setup Trainer
    trainer = Trainer(
        model=model,
        config={"training": training_config}, # Wrap to match trainer expectations
        train_loader=loader,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # 5. Train
    logger.info("Starting training...")
    trainer.train()

if __name__ == "__main__":
    main()
