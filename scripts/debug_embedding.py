"""
Debug script to investigate embedding index out of bounds error.
"""
import torch
import yaml
import numpy as np
from src.models.multimodal_transformer import MultiModalCreature
from src.data.text_dataset import TextDataset
from src.data.text_only_dataset import TextOnlyDataset, text_only_collate_fn
from torch.utils.data import DataLoader

def log(msg):
    """
    Log message to console and file.
    
    Args:
        msg: Message to log.
    """
    print(msg)
    with open("debug_log.txt", "a", encoding="utf-8") as f:
        f.write(msg + "\n")

def main():
    """
    Main entry point for debugging.
    
    Purpose:
        Investigate embedding index out of bounds errors.
    
    Workflow:
        1. Load config and model.
        2. Check embedding sizes.
        3. Load data and check ranges.
        4. Run forward pass with a single sample.
    """
    with open("debug_log.txt", "w") as f: f.write("") # Clear log
    
    log("Loading config...")
    with open("configs/text_only_config.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    log("Initializing model...")
    model = MultiModalCreature(config)
    log("Model initialized.")
    
    # Check embedding sizes
    log("Checking encoder embeddings...")
    if hasattr(model, "internal_voice_encoder"):
        enc = model.internal_voice_encoder
        # Check for various embedding attribute names
        if hasattr(enc, "token_embedding"):
             log(f" Encoder internal_voice token_embedding: {enc.token_embedding.num_embeddings}")
        if hasattr(enc, "position_embedding"):
             log(f" Encoder internal_voice position_embedding: {enc.position_embedding.num_embeddings}")
    else:
        log(" Model has no internal_voice_encoder")
             
    log("Loading data...")
    ds = TextDataset("data/dummy_corpus.bin", seq_length=2048)
    log(f"Dataset size: {len(ds)}")
    
    # Check data range
    if len(ds.tokens) > 0:
        log(f"Data min: {ds.tokens.min()}, Max: {ds.tokens.max()}")
    
    # Get a batch
    log("Getting batch...")
    if len(ds) == 0:
        log("Dataset empty!")
        return

    # Manual fetch
    item = ds[0]
    inp = item["input"]
    log(f"Sample 0 input min: {inp.min()}, max: {inp.max()}, shape: {inp.shape}")
    
    # Run model forward with just this input
    log("Running forward pass...")
    # Wrap in batch dim
    inputs = {
        "internal_voice_tokens": inp.unsqueeze(0)
    }
    
    try:
        model(internal_voice_tokens=inputs["internal_voice_tokens"])
        log("Forward pass success!")
    except Exception as e:
        log(f"Forward pass failed: {e}")
        import traceback
        with open("debug_log.txt", "a", encoding="utf-8") as f:
            traceback.print_exc(file=f)

if __name__ == "__main__":
    main()
