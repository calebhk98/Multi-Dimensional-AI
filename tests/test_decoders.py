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
	"""
	Fixture for batch size used in tests.

	Returns:
		Batch size integer
	"""
	return 4

@pytest.fixture
def seq_len():
	"""
	Fixture for sequence length used in tests.

	Returns:
		Sequence length integer
	"""
	return 10

@pytest.fixture
def embedding_dim():
	"""
	Fixture for embedding dimension used in tests.

	Returns:
		Embedding dimension size
	"""
	return 1536

def test_text_decoder(batch_size, seq_len, embedding_dim):
	"""
	Happy Path: Text Decoder.
	
	Verifies generation of token IDs from hidden states and scalar loss computation.
	
	Args:
		batch_size: Test batch size
		seq_len: Test sequence length
		embedding_dim: Test embedding dimension
	"""
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
	assert isinstance(loss, torch.Tensor)
	assert loss.ndim == 0 # Scalar

def test_text_decoder_error_handling(embedding_dim):
	"""
	Bad Path: Input dimension mismatch.
	
	Args:
		embedding_dim: Test embedding dimension
	"""
	decoder = InternalTextDecoder(vocab_size=1000, embedding_dim=embedding_dim)
	# Pass wrong embedding dim (e.g. +1)
	bad_states = torch.randn(2, 10, embedding_dim + 1)
	with pytest.raises(RuntimeError):
		decoder(bad_states)

def test_audio_decoder(batch_size, seq_len, embedding_dim):
	"""
	Happy Path: Audio Decoder.
	
	Verifies decoding of hidden states to discrete audio codes.
	
	Args:
		batch_size: Test batch size
		seq_len: Test sequence length
		embedding_dim: Test embedding dimension
	"""
	decoder = AudioDecoder(codebook_size=256, embedding_dim=embedding_dim)
	hidden_states = torch.randn(batch_size, seq_len, embedding_dim)
	
	output = decoder(hidden_states)
	assert 'tokens' in output
	
	targets = torch.randint(0, 256, (batch_size, seq_len))
	loss = decoder.compute_loss(hidden_states, targets)
	assert not torch.isnan(loss)

def test_animation_decoder(batch_size, seq_len, embedding_dim):
	"""
	Happy Path: Animation Decoder.
	
	Verifies prediction of joint rotations, blend shapes, and eye params.
	Tests complex loss calculation combining rotation, position, and specialized losses.
	
	Args:
		batch_size: Test batch size
		seq_len: Test sequence length
		embedding_dim: Test embedding dimension
	"""
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
