"""
Tests for pairwise training integration (Phase 4).
"""

import pytest
import torch
from torch.utils.data import DataLoader
import yaml
import tempfile
import os

from src.models.multimodal_transformer import MultiModalCreature
from src.data.synthetic_generator import SyntheticDataGenerator
from src.data.pairwise_dataset import PairwiseDataset
from src.training.trainer import Trainer

@pytest.fixture
def generator():
    """
    Fixture for SyntheticDataGenerator.

    Purpose:
        Provides test generator instance.

    Workflow:
        1. Create generator with small params

    ToDo:
        None

    Returns:
        SyntheticDataGenerator: Generator instance.
    """
    return SyntheticDataGenerator(
        vocab_size=100,
        max_seq_length=32,
        sample_rate=16000,
        audio_duration=0.1,
        image_size=32,
        codebook_size=64
    )

@pytest.fixture
def model_config():
    """
    Fixture for model configuration.

    Purpose:
        Provides minimal model config for testing.

    Workflow:
        1. Return config dict

    ToDo:
        None

    Returns:
        dict: Model configuration dictionary.
    """
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
                "external_voice": {"vocab_size": 100}, # Needed for text->speech
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
            "external_text": 1.0,
            "audio": 1.0,
        }
    }

def test_pairwise_dataset_vision_to_text(generator):
    """
    Purpose:
        Verify PairwiseDataset correctly yields only vision inputs and text targets.
    
    Workflow:
        1. create PairwiseDataset with mapping vision->external_text
        2. get item
        3. assert inputs has 'left_eye_image', 'right_eye_image'
        4. assert inputs does NOT have 'audio', etc.
        5. assert targets has 'external_text'

    ToDo:
        None
    """
    dataset = PairwiseDataset(
        generator=generator,
        pairs=[("vision", "external_text")],
        length=10
    )
    
    item = dataset[0]
    inputs = item["inputs"]
    targets = item["targets"]
    
    assert "left_eye_image" in inputs
    assert "right_eye_image" in inputs
    assert "audio_waveform" not in inputs
    assert "external_text" in targets
    assert "audio" not in targets

def test_pairwise_dataset_mixed(generator):
    """
    Purpose:
        Verify PairwiseDataset correctly handles multiple pairs in list.

    Workflow:
        1. Create dataset with two pairs
        2. Verify items have valid structure

    ToDo:
        None
    """
    dataset = PairwiseDataset(
        generator=generator,
        pairs=[("vision", "external_text"), ("external_text", "audio")],
        length=10
    )
    
    # Just check we can get items without error and they have valid structure
    item = dataset[0]
    assert "inputs" in item
    assert "targets" in item

def test_vision_to_text_training_step(generator, model_config):
    """
    Purpose:
        Test one training step for Vision -> Text pair.

    Workflow:
        1. Create model and dataset
        2. Run one training step
        3. Verify loss is valid

    ToDo:
        None
    """
    model = MultiModalCreature(model_config)
    
    # Pair: Vision (Input) -> External Text (Output)
    # Note: SyntheticGenerator 'vision' keys are left_eye_image/right_eye_image
    # 'external_text' is the target key.
    dataset = PairwiseDataset(
        generator=generator, 
        pairs=[("vision", "external_text")], 
        length=2
    )
    loader = DataLoader(dataset, batch_size=2)
    
    trainer = Trainer(model, model_config, loader, device="cpu")
    
    batch = next(iter(loader))
    loss, _ = trainer.train_step(batch)
    
    assert loss.item() > 0
    assert not torch.isnan(loss)

def test_text_to_speech_training_step(generator, model_config):
    """
    Purpose:
        Test one training step for Text (External) -> Speech (Audio) pair.

    Workflow:
        1. Create model and dataset
        2. Run training step
        3. Verify loss is positive and not NaN

    ToDo:
        None
    """
    model = MultiModalCreature(model_config)
    
    # Pair: External Text (Input) -> Audio (Output)
    dataset = PairwiseDataset(
        generator=generator, 
        pairs=[("external_text", "audio")], 
        length=2
    )
    loader = DataLoader(dataset, batch_size=2)
    
    trainer = Trainer(model, model_config, loader, device="cpu")
    
    batch = next(iter(loader))
    loss, _ = trainer.train_step(batch)
    
    assert loss.item() > 0
    assert not torch.isnan(loss)
