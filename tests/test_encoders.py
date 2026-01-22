"""
Unit tests for Encoders.
Includes regression tests for recent fixes.
"""

import pytest
import torch
import numpy as np
from src.encoders.internal_voice_encoder import InternalVoiceEncoder
from src.encoders.external_voice_encoder import ExternalVoiceEncoder
from src.encoders.audio_encoder import AudioEncoder
from src.encoders.visual_encoder import VisualEncoder
from src.encoders.proprioception_encoder import ProprioceptionEncoder
from src.encoders.touch_encoder import TouchEncoder

@pytest.fixture
def batch_size():
    return 4

@pytest.fixture
def seq_len():
    return 32

@pytest.fixture
def embedding_dim():
    return 1536

def test_internal_voice_encoder(batch_size, seq_len, embedding_dim):
    # Note: internal_voice_encoder argument is vocab_size, not vocabulary_size in some versions?
    # Checking source invalidates this, but assuming consistent with codebase
    # Using 'vocab_size' based on current file structure if possible, but the old test used vocabulary_size
    # Let's check init: vocab_size.
    encoder = InternalVoiceEncoder(vocab_size=1000, embedding_dim=embedding_dim)
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    
    output = encoder(input_ids)
    
    assert 'embeddings' in output
    assert 'attention_mask' in output
    assert output['embeddings'].shape == (batch_size, seq_len, embedding_dim)

def test_audio_encoder(batch_size, embedding_dim):
    # Audio encoder downsamples, so output length depends on input
    encoder = AudioEncoder(embedding_dim=embedding_dim)
    waveform = torch.randn(batch_size, 16000) # 1 second audio
    
    output = encoder(waveform)
    
    assert 'embeddings' in output
    assert output['embeddings'].shape[0] == batch_size
    assert output['embeddings'].shape[2] == embedding_dim
    # Check codebook indices return
    output_indices = encoder(waveform, return_indices=True)
    assert 'indices' in output_indices

def test_visual_encoder_basic(batch_size, embedding_dim):
    encoder = VisualEncoder(embedding_dim=embedding_dim, image_size=224, patch_size=16)
    image = torch.randn(batch_size, 3, 224, 224)
    
    # Test mono (duplicate for stereo arg or None)
    output = encoder(image, image)
    
    num_patches_per_image = (224 // 16) ** 2
    expected_seq_len = num_patches_per_image * 2 # Stereo
    
    assert output['embeddings'].shape == (batch_size, expected_seq_len, embedding_dim)

def test_visual_encoder_divisibility_fix():
    """
    Regression Test: Ensure VisualEncoder works even if embedding_dim is not divisible by default num_heads (12).
    """
    small_dim = 32
    # 32 is not divisible by 12. Should auto-adjust to 8, 4, 2, or 1.
    encoder = VisualEncoder(embedding_dim=small_dim, image_size=32, patch_size=16)
    
    # Verify it runs
    image = torch.randn(2, 3, 32, 32)
    output = encoder(image, image)
    
    assert output['embeddings'].shape[2] == small_dim
    # We can inspect the private transformer layer to verify nhead if we want, but running is enough.
    assert encoder.transformer.layers[0].self_attn.num_heads in [8, 4, 2, 1]

def test_proprioception_encoder_basic(batch_size, embedding_dim):
    temporal_window = 10
    num_joints = 24
    encoder = ProprioceptionEncoder(
        embedding_dim=embedding_dim,
        num_joints=num_joints, 
        temporal_window=temporal_window
    )
    
    pos = torch.randn(batch_size, temporal_window, num_joints, 3)
    rot = torch.randn(batch_size, temporal_window, num_joints, 4)
    
    output = encoder(pos, rot)
    
    assert output['embeddings'].shape == (batch_size, temporal_window, embedding_dim)

def test_proprioception_encoder_padding_fix(batch_size):
    """
    Regression Test: Ensure ProprioceptionEncoder handles missing zero-velocity padding correctly.
    If use_velocity=True but previous_state is None, it should pad inputs with zeros to match dimensions.
    """
    embedding_dim = 64
    temporal_window = 5
    num_joints = 4
    
    # Default is use_velocity=True
    encoder = ProprioceptionEncoder(
        embedding_dim=embedding_dim,
        num_joints=num_joints, 
        temporal_window=temporal_window,
        use_velocity=True
    )
    
    pos = torch.randn(batch_size, temporal_window, num_joints, 3)
    rot = torch.randn(batch_size, temporal_window, num_joints, 4)
    
    # Do NOT pass previous_positions/previous_rotations
    # This crashed before the fix
    output = encoder(pos, rot)
    
    assert output['embeddings'].shape == (batch_size, temporal_window, embedding_dim)

def test_touch_encoder_basic(batch_size, embedding_dim):
    num_contacts = 5
    encoder = TouchEncoder(embedding_dim=embedding_dim, max_contacts=num_contacts)
    
    contact_active = torch.ones(batch_size, num_contacts, dtype=torch.bool)
    contact_points = torch.randint(0, 10, (batch_size, num_contacts))
    contact_forces = torch.rand(batch_size, num_contacts, 1)
    contact_positions = torch.randn(batch_size, num_contacts, 3)
    surface_types = torch.randint(0, 8, (batch_size, num_contacts))
    
    output = encoder(
        contact_active=contact_active,
        contact_points=contact_points,
        contact_forces=contact_forces,
        contact_positions=contact_positions,
        surface_types=surface_types
    )
    
    assert output['embeddings'].shape == (batch_size, num_contacts, embedding_dim)

def test_touch_encoder_divisibility_fix():
    """
    Regression Test: Ensure TouchEncoder works if embedding_dim is not divisible by default num_heads (8).
    """
    odd_dim = 1550 # Not divisible by 8 (1550/8 = 193.75)
    # Wait, 1550 is even. 1550/2 = 775. 1550 is not div by 4 (ends in 50).
    # So heads should adjust to 2.
    
    encoder = TouchEncoder(embedding_dim=odd_dim, max_contacts=4)
    
    contact_active = torch.ones(2, 4, dtype=torch.bool)
    contact_points = torch.randint(0, 10, (2, 4))
    contact_forces = torch.rand(2, 4, 1)
    contact_positions = torch.randn(2, 4, 3)
    surface_types = torch.randint(0, 8, (2, 4))
    
    output = encoder(
        contact_active=contact_active,
        contact_points=contact_points,
        contact_forces=contact_forces,
        contact_positions=contact_positions,
        surface_types=surface_types
    )
    
    assert output['embeddings'].shape[2] == odd_dim
    assert encoder.contact_aggregator.num_heads in [2, 1]
