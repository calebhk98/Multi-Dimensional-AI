"""
Audio encoder - processes environmental sounds.
Converts raw audio waveforms into discrete tokens.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class AudioEncoder(nn.Module):
	"""
	Audio encoder using CNN + Transformer architecture.
	
	Converts raw audio waveforms (16kHz) into discrete tokens representing
	sounds, not words (phonemes, breathing, environmental sounds, etc.).
	
	Similar to Wav2Vec 2.0 or EnCodec approach.
	"""
	
	def __init__(
		self,
		sample_rate: int = 16000,
		hop_length: int = 320,  # ~50 tokens/second
		embedding_dim: int = 1536,
		num_conv_layers: int = 7,
		conv_channels: int = 512,
		codebook_size: int = 1024,
		dropout: float = 0.1,
		**kwargs,  # Absorb extra config args (e.g. num_codebooks, encoder_type)
	):
		"""
		==============================================================================
		Function: __init__
		==============================================================================
		Purpose:  Initializes the AudioEncoder module. Sets up the CNN feature
		          extractor, codebook for quantization, and transformer-ready
		          embedding layers for raw audio processing.

		Parameters:
		    - sample_rate: int
		        Audio sample rate in Hz (default: 16000).
		    - hop_length: int
		        Hop length for frame extraction (default: 320 for ~50 tokens/sec).
		    - embedding_dim: int
		        Output embedding dimension (default: 1536).
		    - num_conv_layers: int
		        Number of convolutional layers for downsampling (default: 7).
		    - conv_channels: int
		        Number of channels in convolutional layers (default: 512).
		    - codebook_size: int
		        Size of discrete audio codebook (default: 1024).
		    - dropout: float
		        Dropout probability (default: 0.1).

		Returns:
		    None

		Dependencies:
		    - torch.nn.Conv1d
		    - torch.nn.GroupNorm
		    - torch.nn.GELU
		    - torch.nn.Embedding
		    - torch.nn.Linear
		    - torch.nn.Dropout
		    - torch.nn.LayerNorm

		Processing Workflow:
		    1.  Store configuration parameters.
		    2.  Construct `feature_extractor` using a stack of Conv1d, GroupNorm, and GELU layers.
		    3.  Define `projection` layer to map to embedding dimension.
		    4.  Initialize `codebook` for vector quantization.
		    5.  Initialize `positional_encoding` and `modality_embedding`.
		    6.  Initialize `dropout` and `layer_norm`.

		ToDo:
		    - None

		Usage:
		    model = AudioEncoder(sample_rate=16000, embedding_dim=512)
		==============================================================================
		"""
		super().__init__()
		
		self.sample_rate = sample_rate
		self.hop_length = hop_length
		self.embedding_dim = embedding_dim
		self.codebook_size = codebook_size
		
		# CNN feature extractor
		# Stack of 1D convolutions to downsample audio
		conv_layers = []
		in_channels = 1  # Mono audio
		
		for i in range(num_conv_layers):
			out_channels = conv_channels if i > 0 else conv_channels // 2
			conv_layers.extend([
				nn.Conv1d(
					in_channels=in_channels,
					out_channels=out_channels,
					kernel_size=10,
					stride=5 if i == 0 else 2,
					padding=5,
				),
				nn.GroupNorm(num_groups=out_channels // 16, num_channels=out_channels),
				nn.GELU(),
			])
			in_channels = out_channels
		
		self.feature_extractor = nn.Sequential(*conv_layers)
		
		# Project to embedding dimension
		self.projection = nn.Linear(conv_channels, embedding_dim)
		
		# Codebook for quantization (vector quantization)
		self.codebook = nn.Embedding(codebook_size, embedding_dim)
		
		# Positional encoding
		self.positional_encoding = nn.Parameter(
			torch.randn(1, 1000, embedding_dim)  # Max 1000 frames
		)
		
		# Modality embedding
		self.modality_embedding = nn.Parameter(
			torch.randn(1, 1, embedding_dim)
		)
		
		# Dropout and layer norm
		self.dropout = nn.Dropout(dropout)
		self.layer_norm = nn.LayerNorm(embedding_dim)
	
	def quantize(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
		"""
		==============================================================================
		Function: quantize
		==============================================================================
		Purpose:  Quantizes continuous features into discrete codes using a codebook
		          (Vector Quantization). Uses the Straight-Through Estimator for
		          backpropagation.

		Parameters:
		    - features: torch.Tensor
		        Continuous features tensor [batch, seq_len, embed_dim].

		Returns:
		    Tuple[torch.Tensor, torch.Tensor] - A tuple containing:
		        - quantized: Quantized features tensor [batch, seq_len, embed_dim].
		        - indices: Codebook indices tensor [batch, seq_len].

		Dependencies:
		    - self.codebook
		    - torch.cdist

		Processing Workflow:
		    1.  Flatten input features.
		    2.  Compute pairwise distances between features and codebook vectors.
		    3.  Find indices of nearest codebook vectors.
		    4.  Retrieve quantized vectors from codebook.
		    5.  Reshape quantized vectors and indices back to [batch, seq_len].
		    6.  Apply Straight-Through Estimator: `quantized = features + (quantized - features).detach()`.

		ToDo:
		    - None

		Usage:
		    quantized, indices = self.quantize(features)
		==============================================================================
		"""
		batch_size, seq_len, embed_dim = features.shape
		
		# Flatten
		flat_features = features.reshape(-1, embed_dim)
		
		# Compute distances to codebook vectors
		distances = torch.cdist(
			flat_features.unsqueeze(0),
			self.codebook.weight.unsqueeze(0)
		).squeeze(0)
		
		# Get nearest codebook indices
		indices = torch.argmin(distances, dim=-1)
		
		# Get quantized vectors
		quantized = self.codebook(indices)
		
		# Reshape back
		quantized = quantized.reshape(batch_size, seq_len, embed_dim)
		indices = indices.reshape(batch_size, seq_len)
		
		# Straight-through estimator for backprop
		quantized = features + (quantized - features).detach()
		
		return quantized, indices
	
	def forward(
		self,
		waveform: torch.Tensor,
		return_indices: bool = False,
	) -> Dict[str, torch.Tensor]:
		"""
		==============================================================================
		Function: forward
		==============================================================================
		Purpose:  Encodes raw audio waveforms into discrete token embeddings.

		Parameters:
		    - waveform: torch.Tensor
		        Raw audio tensor [batch_size, num_samples].
		    - return_indices: bool
		        Calculates and returns discrete codebook indices if True (default: False).

		Returns:
		    Dict[str, torch.Tensor] - Dictionary containing:
		        - "embeddings": Encoded embeddings [batch, seq_len, embed_dim].
		        - "attention_mask": Attention mask [batch, seq_len].
		        - "indices": Discrete code indices [batch, seq_len] (if return_indices is True).

		Dependencies:
		    - self.feature_extractor
		    - self.projection
		    - self.quantize
		    - self.positional_encoding
		    - self.modality_embedding

		Processing Workflow:
		    1.  Add channel dimension to waveform if needed.
		    2.  Extract features using CNN `feature_extractor`.
		    3.  Transpose frames to sequence dimension.
		    4.  Project features to embedding dimension.
		    5.  Quantize features to get discrete codes and indices.
		    6.  Add `positional_encoding` (truncated to sequence length).
		    7.  Add `modality_embedding`.
		    8.  Apply layer norm and dropout.
		    9.  Generate attention mask.
		    10. Return dictionary result.

		ToDo:
		    - None

		Usage:
		    output = model(waveform)
		==============================================================================
		"""
		# Add channel dimension if needed
		if waveform.dim() == 2:
			waveform = waveform.unsqueeze(1)  # [B, 1, T]
		
		# CNN feature extraction
		features = self.feature_extractor(waveform)  # [B, C, T']
		
		# Transpose to [B, T', C]
		features = features.transpose(1, 2)
		
		# Project to embedding dimension
		features = self.projection(features)  # [B, T', D]
		
		batch_size, seq_len, _ = features.shape
		
		# Quantize to discrete codes
		quantized, indices = self.quantize(features)
		
		# Add positional encoding
		pos_encoding = self.positional_encoding[:, :seq_len, :]
		quantized = quantized + pos_encoding
		
		# Add modality embedding
		modal_embeds = self.modality_embedding.expand(batch_size, seq_len, -1)
		embeddings = quantized + modal_embeds
		
		# Apply layer norm and dropout
		embeddings = self.layer_norm(embeddings)
		embeddings = self.dropout(embeddings)
		
		# Create attention mask (all valid)
		attention_mask = torch.ones(
			batch_size, seq_len,
			dtype=torch.long,
			device=waveform.device
		)
		
		output = {
			"embeddings": embeddings,
			"attention_mask": attention_mask,
		}
		
		if return_indices:
			output["indices"] = indices
		
		return output
	
	def get_output_dim(self) -> int:
		"""
		==============================================================================
		Function: get_output_dim
		==============================================================================
		Purpose:  Returns the size of the output embeddings produced by this encoder.

		Parameters:
		    - None

		Returns:
		    int - Embedding dimension size (e.g., 1536).

		Dependencies:
		    - None

		Processing Workflow:
		    1.  Return `self.embedding_dim`.

		ToDo:
		    - None

		Usage:
		    dim = model.get_output_dim()
		==============================================================================
		"""
		return self.embedding_dim
