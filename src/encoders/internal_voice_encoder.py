"""
Internal voice encoder - processes the creature's internal thoughts.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional
from src.encoders.voice_tokenizer import VoiceTokenizer


class InternalVoiceEncoder(nn.Module):
	"""
	Encoder for internal voice (creature's thoughts).
	
	Converts internal thought text into embeddings that feed into
	the transformer backbone.
	"""
	
	def __init__(
		self,
		vocab_size: int = 50257,
		embedding_dim: int = 1536,
		max_seq_length: int = 512,
		dropout: float = 0.1,
	):
		"""
		==============================================================================
		Function: __init__
		==============================================================================
		Purpose:  Initializes the InternalVoiceEncoder module. Sets up embeddings for
		          internal thought tokens, positions, and modality, along with
		          normalization and dropout layers.

		Parameters:
		    - vocab_size: int
		        Size of the vocabulary (default: 50257).
		    - embedding_dim: int
		        Dimension of token embeddings (default: 1536).
		    - max_seq_length: int
		        Maximum sequence length supported (default: 512).
		    - dropout: float
		        Dropout probability (default: 0.1).

		Returns:
		    None

		Dependencies:
		    - torch.nn.Embedding
		    - torch.nn.Dropout
		    - torch.nn.LayerNorm

		Processing Workflow:
		    1.  Store configuration parameters.
		    2.  Initialize `token_embedding` layer.
		    3.  Initialize `position_embedding` layer.
		    4.  Initialize `modality_embedding` parameter.
		    5.  Initialize `dropout` and `layer_norm`.

		ToDo:
		    - None

		Usage:
		    model = InternalVoiceEncoder(vocab_size=50257, embedding_dim=512)
		==============================================================================
		"""
		super().__init__()
		
		self.vocab_size = vocab_size
		self.embedding_dim = embedding_dim
		self.max_seq_length = max_seq_length
		
		# Token embedding layer
		self.token_embedding = nn.Embedding(
			num_embeddings=vocab_size,
			embedding_dim=embedding_dim,
			padding_idx=0,  # Assuming 0 is padding
		)
		
		# Positional embedding
		self.position_embedding = nn.Embedding(
			num_embeddings=max_seq_length,
			embedding_dim=embedding_dim,
		)
		
		# Modality embedding (distinguishes internal from external voice)
		self.modality_embedding = nn.Parameter(
			torch.randn(1, 1, embedding_dim)
		)
		
		# Dropout
		self.dropout = nn.Dropout(dropout)
		
		# Layer norm
		self.layer_norm = nn.LayerNorm(embedding_dim)
	
	def forward(
		self,
		input_ids: torch.Tensor,
		attention_mask: Optional[torch.Tensor] = None,
	) -> Dict[str, torch.Tensor]:
		"""
		==============================================================================
		Function: forward
		==============================================================================
		Purpose:  Processes input token IDs from internal thoughts into embeddings,
		          adding positional and modality information.

		Parameters:
		    - input_ids: torch.Tensor
		        Token IDs tensor [batch_size, seq_len].
		    - attention_mask: Optional[torch.Tensor]
		        Attention mask tensor [batch_size, seq_len].

		Returns:
		    Dict[str, torch.Tensor] - Dictionary containing:
		        - "embeddings": Encoded embeddings [batch_size, seq_len, embed_dim]
		        - "attention_mask": Attention mask [batch_size, seq_len]

		Dependencies:
		    - self.token_embedding
		    - self.position_embedding
		    - self.modality_embedding

		Processing Workflow:
		    1.  Generate position IDs based on sequence length.
		    2.  Get token embeddings from `input_ids`.
		    3.  Get position embeddings from `position_ids`.
		    4.  Expand `modality_embedding` to match batch and sequence size.
		    5.  Sum token, position, and modality embeddings.
		    6.  Apply layer norm and dropout.
		    7.  Generate attention mask if one is not provided.

		ToDo:
		    - None

		Usage:
		    output = model(input_ids)
		==============================================================================
		"""
		batch_size, seq_len = input_ids.shape
		
		# Create position IDs
		position_ids = torch.arange(
			seq_len,
			dtype=torch.long,
			device=input_ids.device
		).unsqueeze(0).expand(batch_size, -1)
		
		# Get embeddings
		token_embeds = self.token_embedding(input_ids)  # [B, L, D]
		pos_embeds = self.position_embedding(position_ids)  # [B, L, D]
		
		# Add modality embedding (broadcast across batch and sequence)
		modal_embeds = self.modality_embedding.expand(batch_size, seq_len, -1)
		
		# Combine embeddings
		embeddings = token_embeds + pos_embeds + modal_embeds
		
		# Apply layer norm and dropout
		embeddings = self.layer_norm(embeddings)
		embeddings = self.dropout(embeddings)
		
		# Create attention mask if not provided
		if attention_mask is None:
			attention_mask = torch.ones_like(input_ids)
		
		return {
			"embeddings": embeddings,
			"attention_mask": attention_mask,
		}
	
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
