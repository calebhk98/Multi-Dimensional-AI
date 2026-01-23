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
    """
    Standard batch size to use across encoder tests.

    Returns:
        Batch size integer
    """
    return 4

@pytest.fixture
def seq_len():
    """
    Standard sequence length to use across encoder tests.

    Returns:
        Sequence length integer
    """
    return 32

@pytest.fixture
def embedding_dim():
    """
    Standard embedding dimension to use across encoder tests.

    Returns:
        Embedding dimension size
    """
    return 1536

def test_internal_voice_encoder(batch_size, seq_len, embedding_dim):
    """
    Purpose:
        Test Internal Voice Encoder (Happy Path).
        Verifies that the encoder properly processes a batch of token IDs and returns 
        embeddings of the expected shape [batch, seq_len, embedding_dim].
        
    Workflow:
        1. Initialize InternalVoiceEncoder.
        2. Pass random input IDs.
        3. Verify output keys and shape.
        
    ToDo:
        - None
    """
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

def test_internal_voice_encoder_error_handling(batch_size, seq_len, embedding_dim):
    """
    Purpose:
        Test Internal Voice Encoder error handling (Bad Path).
        Verifies error raised when token ID is out of vocabulary range.
        
    Workflow:
        1. Initialize InternalVoiceEncoder with small vocab.
        2. Pass input ID out of range.
        3. Verify IndexError.
        
    ToDo:
        - None
    """
    encoder = InternalVoiceEncoder(vocab_size=10, embedding_dim=embedding_dim)
    # Token ID 10 is out of bounds (0-9)
    input_ids = torch.tensor([[10]])
    with pytest.raises(IndexError):
        encoder(input_ids)

def test_audio_encoder(batch_size, embedding_dim):
    """
    Purpose:
        Test Audio Encoder functionality (Happy Path).
        Verifies that the encoder processes raw audio waveforms and returns embeddings.
        
    Workflow:
        1. Initialize AudioEncoder.
        2. Pass random waveform.
        3. Verify output shape (batch, tokens, dim).
        4. Verify indices are returned if requested.
        
    ToDo:
        - None
    """
    # Audio encoder downsamples, so output length depends on input
    encoder = AudioEncoder(embedding_dim=embedding_dim)
    waveform = torch.randn(batch_size, 16000) # 1 second audio
    
    output = encoder(waveform)
    
    assert 'embeddings' in output
    # For default hop_length=320, 16000/320 = 50 tokens
    assert output['embeddings'].shape[0] == batch_size
    assert output['embeddings'].shape[2] == embedding_dim
    # Check codebook indices return
    output_indices = encoder(waveform, return_indices=True)
    assert 'indices' in output_indices

def test_visual_encoder_basic(batch_size, embedding_dim):
    """
    Purpose:
        Test Visual Encoder functionality (Happy Path).
        Verifies encoding of stereo images using a ViT-based architecture.
        
    Workflow:
        1. Initialize VisualEncoder.
        2. Pass random stereo images.
        3. Verify output shape matches expected number of patches * 2.
        
    ToDo:
        - None
    """
    encoder = VisualEncoder(embedding_dim=embedding_dim, image_size=224, patch_size=16)
    image = torch.randn(batch_size, 3, 224, 224)
    
    # Test mono (duplicate for stereo arg or None)
    output = encoder(image, image)
    
    num_patches_per_image = (224 // 16) ** 2
    expected_seq_len = num_patches_per_image * 2 # Stereo
    
    assert output['embeddings'].shape == (batch_size, expected_seq_len, embedding_dim)

def test_visual_encoder_error_handling(batch_size, embedding_dim):
    """
    Purpose:
        Test Visual Encoder error handling (Bad Path).
        Verifies errors on invalid input shapes.
        
    Workflow:
        1. Initialize VisualEncoder.
        2. Pass image with wrong channel count -> Verify RuntimeError.
        3. Pass image with wrong spatial dimensions -> Verify RuntimeError.
        
    ToDo:
        - None
    """
    encoder = VisualEncoder(embedding_dim=embedding_dim, image_size=224, patch_size=16)
    
    # Wrong channel count (1 instead of 3) - conv2d should fail
    bad_image = torch.randn(batch_size, 1, 224, 224)
    with pytest.raises(RuntimeError):
        encoder(bad_image)
        
    # Wrong image size - might pass through convolution but mismatch embeddings?
    # If spatial dimensions (224) change, num_patches changes.
    # Positional embedding is constant size.
    # So if we pass 225x225, (225//16)=14. ...
    # Wait, 1/num_patches embedding...
    # If image size is wrong, positional embedding addition will broadcast failure or size mismatch.
    # Let's test that.
    bad_size = torch.randn(batch_size, 3, 300, 300) 
    # 300 // 16 = 18. 18*18 = 324 patches. Default 224->14*14=196 patches.
    # pos_embed: [1, 196, dim] vs [B, 324, dim] -> Error.
    with pytest.raises(RuntimeError):
        encoder(bad_size)

def test_visual_encoder_divisibility_fix():
    """
    Purpose:
        Regression Test: Ensure VisualEncoder works even if embedding_dim is not divisible by default num_heads (12).
        
    Workflow:
        1. Initialize VisualEncoder with dim=32 (not div by 12).
        2. Pass valid image.
        3. Verify forward pass succeeds and output dim is correct.
        4. Verify num_heads adjusted.
        
    ToDo:
        - None
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
    """
    Purpose:
        Test Proprioception Encoder functionality (Happy Path).
        Verifies processing of body joint positions and rotations over a temporal window.
        
    Workflow:
        1. Initialize ProprioceptionEncoder.
        2. Pass random positions and rotations.
        3. Verify output shape [batch, time, dim].
        
    ToDo:
        - None
    """
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
    Purpose:
        Regression Test: Ensure ProprioceptionEncoder handles missing zero-velocity padding correctly.
        
    Workflow:
        1. Initialize with use_velocity=True.
        2. Pass current pos/rot without previous state.
        3. Verify no crash (previously crashed).
        
    ToDo:
        - None
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
    """
    Purpose:
        Test Touch Encoder functionality (Happy Path).
        Verifies processing of sparse contact points.
        
    Workflow:
        1. Initialize TouchEncoder.
        2. Pass random contact data (active, points, forces, pos, types).
        3. Verify output shape.
        
    ToDo:
        - None
    """
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
    Purpose:
        Regression Test: Ensure TouchEncoder works if embedding_dim is not divisible by default num_heads (8).
        
    Workflow:
        1. Initialize TouchEncoder with odd dimension.
        2. Pass valid input.
        3. Verify forward pass succeeds.
        
    ToDo:
        - None
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
