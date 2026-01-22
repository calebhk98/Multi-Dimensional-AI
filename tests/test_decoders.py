"""
Unit tests for Decoders.
"""

import pytest
import torch
from src.decoders.text_decoder import InternalTextDecoder, ExternalTextDecoder
from src.decoders.audio_decoder import AudioDecoder
from src.decoders.animation_decoder import AnimationDecoder

@pytest.fixture
def batch_size():
    return 4

@pytest.fixture
def seq_len():
    return 10

@pytest.fixture
def embedding_dim():
    return 1536

def test_text_decoder(batch_size, seq_len, embedding_dim):
    decoder = InternalTextDecoder(vocab_size=1000, embedding_dim=embedding_dim)
    hidden_states = torch.randn(batch_size, seq_len, embedding_dim)
    
    # Test generation
    output = decoder(hidden_states)
    assert 'tokens' in output
    assert output['tokens'].shape == (batch_size, seq_len)
    
    # Test loss computation
    targets = torch.randint(0, 1000, (batch_size, seq_len))
    loss = decoder.compute_loss(hidden_states, targets)
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0 # Scalar

def test_audio_decoder(batch_size, seq_len, embedding_dim):
    decoder = AudioDecoder(codebook_size=256, embedding_dim=embedding_dim)
    hidden_states = torch.randn(batch_size, seq_len, embedding_dim)
    
    output = decoder(hidden_states)
    assert 'tokens' in output
    
    targets = torch.randint(0, 256, (batch_size, seq_len))
    loss = decoder.compute_loss(hidden_states, targets)
    assert not torch.isnan(loss)

def test_animation_decoder(batch_size, seq_len, embedding_dim):
    decoder = AnimationDecoder(embedding_dim=embedding_dim, num_joints=24)
    hidden_states = torch.randn(batch_size, seq_len, embedding_dim)
    
    output = decoder(hidden_states)
    assert 'joint_rotations' in output
    assert 'blend_shapes' in output
    assert output['joint_rotations'].shape == (batch_size, seq_len, 24, 4)
    
    # Test loss
    # Needs dummy targets for all 3 components
    sample_rot = torch.randn(batch_size, seq_len, 24, 4)
    sample_shape = torch.rand(batch_size, seq_len, 51)
    sample_eyes = torch.rand(batch_size, seq_len, 8)
    
    loss, loss_dict = decoder.compute_loss(
        hidden_states, 
        sample_rot, 
        sample_shape, 
        sample_eyes
    )
    assert isinstance(loss, torch.Tensor)
    assert 'total_animation_loss' in loss_dict
