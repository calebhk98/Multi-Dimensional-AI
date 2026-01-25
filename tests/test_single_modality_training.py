"""
Tests for single-modality training infrastructure using the Full Brain (MultiModalCreature).
"""

import pytest
import torch
from torch.utils.data import DataLoader

from src.models.multimodal_transformer import MultiModalCreature
from scripts.train_single_modality import SyntheticDataset
from src.data.synthetic_generator import SyntheticDataGenerator
from src.training.trainer import Trainer

@pytest.fixture
def generator():
    return SyntheticDataGenerator(
        vocab_size=100,
        max_seq_length=20,
        sample_rate=16000,
        audio_duration=0.1,
        num_joints=4,
        temporal_window=5,
        image_size=32,
        codebook_size=64
    )

@pytest.fixture
def model_config():
    # Miniature config for testing speed
    return {
        "model": {
            "transformer": {
                "num_layers": 2,
                "hidden_dim": 32,
                "num_attention_heads": 4,
                "ffn_dim": 64,
                "dropout": 0.0
            },
            "fusion": {
                "strategy": "concatenate",
                "modality_embeddings": True
            },
            "encoders": {
                "internal_voice": {"vocab_size": 100},
                "audio": {
                    "sample_rate": 16000,
                    "num_conv_layers": 2,
                    "conv_channels": 64,
                    "codebook_size": 64
                },
                "vision": {"image_size": 32, "patch_size": 8},
                "proprioception": {"num_joints": 4, "temporal_window": 5},
                "touch": {"num_contact_points": 5}
            },
            "decoders": {
                "audio": {"codebook_size": 64},
                "animation": {"num_joints": 4}
            }
        },
        "training": {
            "max_steps": 1,
            "log_interval": 1, 
            "save_interval": 100
        },
        "loss_weights": {
            "internal_text": 1.0,
            "external_text": 1.0,
            "audio": 1.0,
            "animation": 1.0
        }
    }

def test_audio_training_step(generator, model_config):
    """Test one training step for Audio through full brain."""
    model = MultiModalCreature(model_config)
    
    dataset = SyntheticDataset(generator, "audio", length=2)
    loader = DataLoader(dataset, batch_size=2)
    
    trainer = Trainer(model, model_config, loader, device="cpu")
    
    batch = next(iter(loader))
    loss, _ = trainer.train_step(batch)
    
    assert loss.item() > 0
    assert not torch.isnan(loss)

def test_voice_training_step(generator, model_config):
    """Test one training step for Voice (Internal) through full brain."""
    model = MultiModalCreature(model_config)
    
    dataset = SyntheticDataset(generator, "voice_internal", length=2)
    loader = DataLoader(dataset, batch_size=2)
    
    trainer = Trainer(model, model_config, loader, device="cpu")
    
    batch = next(iter(loader))
    loss, _ = trainer.train_step(batch)
    
    assert loss.item() > 0
    assert not torch.isnan(loss)

def test_motion_training_step(generator, model_config):
    """Test one training step for Motion through full brain."""
    model = MultiModalCreature(model_config)
    
    dataset = SyntheticDataset(generator, "motion", length=2)
    loader = DataLoader(dataset, batch_size=2)
    
    trainer = Trainer(model, model_config, loader, device="cpu")
    
    batch = next(iter(loader))
    loss, _ = trainer.train_step(batch)
    
    assert loss.item() > 0
    assert not torch.isnan(loss)
