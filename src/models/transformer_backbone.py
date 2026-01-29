"""
Transformer backbone - the core of the multi-modal creature model.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
from torch.utils.checkpoint import checkpoint


class TransformerBackbone(nn.Module):
	"""
	Core transformer backbone that processes fused multi-modal tokens.
	
	Standard transformer architecture with:
	- Multi-head self-attention
	- Feed-forward networks
	- Layer normalization
	- Residual connections
	"""
	
	def __init__(
		self,
		num_layers: int = 24,
		hidden_dim: int = 1536,
		num_heads: int = 16,
		ffn_dim: int = 6144,  # 4x hidden_dim
		dropout: float = 0.1,
		attention_dropout: float = 0.1,
		max_seq_length: int = 4096,
	):
		"""
		Initialize transformer backbone.
		
		Args:
			num_layers: Number of transformer layers
			hidden_dim: Hidden dimension
			num_heads: Number of attention heads
			ffn_dim: Feed-forward network dimension
			dropout: Dropout probability
			attention_dropout: Attention dropout probability
			max_seq_length: Maximum sequence length
		"""
		super().__init__()
		
		self.num_layers = num_layers
		self.hidden_dim = hidden_dim
		self.num_heads = num_heads
		self.max_seq_length = max_seq_length
		self.gradient_checkpointing = False
		
		# Transformer encoder layers
		self.layers = nn.ModuleList([
			TransformerLayer(
				hidden_dim=hidden_dim,
				num_heads=num_heads,
				ffn_dim=ffn_dim,
				dropout=dropout,
				attention_dropout=attention_dropout,
			)
			for _ in range(num_layers)
		])
		
		# Final layer norm
		self.final_layer_norm = nn.LayerNorm(hidden_dim)
		
	def enable_gradient_checkpointing(self, value: bool = True):
		"""
		Enable or disable gradient checkpointing.
		
		Args:
			value: Whether to enable checkpointing.
		"""
		self.gradient_checkpointing = value
	
	def forward(
		self,
		hidden_states: torch.Tensor,
		attention_mask: Optional[torch.Tensor] = None,
		return_all_layers: bool = False,
	) -> torch.Tensor:
		"""
		Forward pass through transformer.
		
		Args:
			hidden_states: Input embeddings [batch, seq_len, hidden_dim]
			attention_mask: Attention mask [batch, seq_len]
			return_all_layers: Return outputs from all layers
			
		Returns:
			Output hidden states [batch, seq_len, hidden_dim]
			or list of outputs if return_all_layers=True
		"""
		# Convert attention mask to key_padding_mask if provided
		# Input mask: 1 for valid, 0 for padding
		# PyTorch key_padding_mask: False for valid, True for padding (boolean)
		key_padding_mask = None
		if attention_mask is not None:
			# First ensure it's bool: 0 -> True (Pad), 1 -> False (Valid)
			key_padding_mask = (attention_mask == 0).bool()
		
		all_hidden_states = []
		
		# Pass through each transformer layer
		for layer in self.layers:
			if self.gradient_checkpointing and self.training:
				# Use checkpointing
				# Note: checkpoint requires one tensor with requires_grad=True
				# hidden_states usually has it.
				# We wrap the layer call in a lambda or pass args directly
				hidden_states = checkpoint(
					layer,
					hidden_states,
					key_padding_mask,
					use_reentrant=False
				)
			else:
				hidden_states = layer(
					hidden_states, 
					key_padding_mask=key_padding_mask
				)
			
			if return_all_layers:
				all_hidden_states.append(hidden_states)
		
		# Final layer norm
		hidden_states = self.final_layer_norm(hidden_states)
		
		if return_all_layers:
			all_hidden_states.append(hidden_states)
			return all_hidden_states
		
		return hidden_states


class TransformerLayer(nn.Module):
	"""Single transformer layer."""
	
	def __init__(
		self,
		hidden_dim: int,
		num_heads: int,
		ffn_dim: int,
		dropout: float = 0.1,
		attention_dropout: float = 0.1,
	):
		"""
		Initialize transformer layer.
		
		Args:
			hidden_dim: Hidden dimension
			num_heads: Number of attention heads
			ffn_dim: FFN dimension
			dropout: Dropout probability
			attention_dropout: Attention dropout
		"""
		super().__init__()
		
		# Multi-head self-attention
		self.self_attention = nn.MultiheadAttention(
			embed_dim=hidden_dim,
			num_heads=num_heads,
			dropout=attention_dropout,
			batch_first=True,
		)
		
		# Layer norms
		self.attention_layer_norm = nn.LayerNorm(hidden_dim)
		self.ffn_layer_norm = nn.LayerNorm(hidden_dim)
		
		# Feed-forward network
		self.ffn = nn.Sequential(
			nn.Linear(hidden_dim, ffn_dim),
			nn.GELU(),
			nn.Dropout(dropout),
			nn.Linear(ffn_dim, hidden_dim),
			nn.Dropout(dropout),
		)
		
		# Dropout
		self.dropout = nn.Dropout(dropout)
	
	def forward(
		self,
		hidden_states: torch.Tensor,
		attention_mask: Optional[torch.Tensor] = None,
		key_padding_mask: Optional[torch.Tensor] = None,
	) -> torch.Tensor:
		"""
		Forward pass through layer.
		
		Args:
			hidden_states: Input [batch, seq_len, hidden_dim]
			attention_mask: Attention mask
			
		Returns:
			Output [batch, seq_len, hidden_dim]
		"""
		# Self-attention with residual
		residual = hidden_states
		
		hidden_states, _ = self.self_attention(
			query=hidden_states,
			key=hidden_states,
			value=hidden_states,
			attn_mask=attention_mask,
			key_padding_mask=key_padding_mask,
			need_weights=False,
		)
		
		hidden_states = self.dropout(hidden_states)
		hidden_states = residual + hidden_states
		hidden_states = self.attention_layer_norm(hidden_states)
		
		# FFN with residual
		residual = hidden_states
		hidden_states = self.ffn(hidden_states)
		hidden_states = residual + hidden_states
		hidden_states = self.ffn_layer_norm(hidden_states)
		
		return hidden_states
