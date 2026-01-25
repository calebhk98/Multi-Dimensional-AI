"""
Token fusion module for combining multi-modal inputs.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any, Tuple


class TokenFusionModule(nn.Module):
	"""
	Fuses tokens from multiple input modalities into a single sequence.
	
	Supports different fusion strategies:
	- Concatenation (simple)
	- Cross-attention fusion (more complex)
	- Learned fusion (trainable)
	"""
	
	def __init__(
		self,
		embedding_dim: int = 1536,
		fusion_strategy: str = "concatenate",  # or "cross_attention", "learned"
		use_modality_embeddings: bool = True,
		dropout: float = 0.1,
	):
		"""
		Initialize token fusion module.
		
		Args:
			embedding_dim: Embedding dimension (must match encoder outputs)
			fusion_strategy: How to fuse tokens
			use_modality_embeddings: Add learned modality embeddings
			dropout: Dropout probability
		"""
		super().__init__()
		
		self.embedding_dim = embedding_dim
		self.fusion_strategy = fusion_strategy
		self.use_modality_embeddings = use_modality_embeddings
		
		# Modality type embeddings (if not already in encoders)
		if use_modality_embeddings:
			# 6 modalities: int_voice, ext_voice, audio, vision, proprio, touch
			self.modality_type_embeddings = nn.Parameter(
				torch.randn(6, embedding_dim)
			)
		
		# Cross-attention fusion (if using that strategy)
		if fusion_strategy == "cross_attention":
			self._setup_cross_attention(embedding_dim, dropout)
		
		# Learned fusion (small transformer)
		elif fusion_strategy == "learned":
			self._setup_learned_fusion(embedding_dim, dropout)
			
		# Dropout (re-initialized or used in methods)
		self.dropout = nn.Dropout(dropout)

	def _setup_cross_attention(self, embedding_dim, dropout):
		"""
		Setup components for cross-attention fusion strategy.
		
		Args:
			embedding_dim: Dimension of embeddings
			dropout: Dropout probability for attention layers
		
		Returns:
			None
		"""
		# Default to 8 heads if not specified/inferable, or ensuring divisibility
		# Use a safe default for num_heads that divides embedding_dim
		num_heads = 16
		if embedding_dim % num_heads != 0:
			# Find largest power of 2 divisor
			for h in [8, 4, 2, 1]:
				if embedding_dim % h == 0:
					num_heads = h
					break
		
		self.fusion_attention = nn.MultiheadAttention(
			embed_dim=embedding_dim,
			num_heads=num_heads,
			dropout=dropout,
			batch_first=True,
		)

	def _setup_learned_fusion(self, embedding_dim, dropout):
		"""
		Setup components for learned fusion strategy.
		
		Args:
			embedding_dim: Dimension of embeddings
			dropout: Dropout probability for transformer layers
		
		Returns:
			None
		"""
		self.fusion_transformer = nn.TransformerEncoder(
			nn.TransformerEncoderLayer(
				d_model=embedding_dim,
				nhead=16,
				dim_feedforward=embedding_dim * 4,
				dropout=dropout,
				batch_first=True,
			),
			num_layers=2,
		)
	
	def forward(
		self,
		encoder_outputs: Dict[str, Dict[str, torch.Tensor]],
	) -> Dict[str, Any]:
		"""
		Fuse multi-modal tokens.
		
		Args:
			encoder_outputs: Dictionary of encoder outputs
				Keys: "internal_voice", "external_voice", "audio", 
					"vision", "proprioception", "touch"
				Values: Dict with "embeddings" and "attention_mask"
				
		Returns:
			Dictionary with:
				- embeddings: Fused embeddings [batch, total_seq_len, dim]
				- attention_mask: Combined mask [batch, total_seq_len]
				- modality_ranges: Dict of (start, end) indices per modality
		"""
		batch_size = None
		all_embeddings = []
		all_masks = []
		modality_ranges = {}
		current_pos = 0
		
		# Define order of modalities
		modality_order = [
			"internal_voice",
			"external_voice", 
			"audio",
			"vision",
			"proprioception",
			"touch",
		]
		
		# Concatenate tokens from all modalities
		for i, modality_name in enumerate(modality_order):
			# Skip modalities that aren't present in the encoder outputs
			if modality_name not in encoder_outputs:
				continue
			
			output = encoder_outputs[modality_name]
			embeddings = output["embeddings"]
			mask = output["attention_mask"]
				
			if batch_size is None:
				batch_size = embeddings.shape[0]
			
			seq_len = embeddings.shape[1]
				
			# Add modality type embedding if enabled
			if self.use_modality_embeddings:
				modality_embed = self.modality_type_embeddings[i].unsqueeze(0).unsqueeze(0)
				embeddings = embeddings + modality_embed
			
			all_embeddings.append(embeddings)
			all_masks.append(mask)
				
			# Track boundaries
			modality_ranges[modality_name] = (current_pos, current_pos + seq_len)
			current_pos += seq_len
		
		# Concatenate along sequence dimension
		if self.fusion_strategy == "concatenate":
			fused_embeddings = torch.cat(all_embeddings, dim=1)
			fused_mask = torch.cat(all_masks, dim=1)
		
		# Cross-attention fusion
		elif self.fusion_strategy == "cross_attention":
			# Concatenate first
			fused_embeddings = torch.cat(all_embeddings, dim=1)
			fused_mask = torch.cat(all_masks, dim=1)
			
			# Apply cross-attention
			fused_embeddings, _ = self.fusion_attention(
				query=fused_embeddings,
				key=fused_embeddings,
				value=fused_embeddings,
				key_padding_mask=(fused_mask == 0),
				need_weights=False,
			)
		
		# Learned fusion with small transformer
		elif self.fusion_strategy == "learned":
			fused_embeddings = torch.cat(all_embeddings, dim=1)
			fused_mask = torch.cat(all_masks, dim=1)
			
			# Create attention mask for transformer
			attn_mask = fused_mask.unsqueeze(1).unsqueeze(2)
			attn_mask = (1.0 - attn_mask) * -10000.0
			
			fused_embeddings = self.fusion_transformer(
				fused_embeddings,
				src_key_padding_mask=(fused_mask == 0),
			)
		
		else:
			raise ValueError(f"Unknown fusion strategy: {self.fusion_strategy}")
		
		# Apply dropout
		fused_embeddings = self.dropout(fused_embeddings)
		
		return {
			"embeddings": fused_embeddings,
			"attention_mask": fused_mask,
			"modality_ranges": modality_ranges,
		}
	
	def get_modality_tokens(
		self,
		fused_embeddings: torch.Tensor,
		modality_ranges: Dict[str, Tuple[int, int]],
		modality_name: str,
	) -> torch.Tensor:
		"""
		Extract tokens for a specific modality from fused sequence.
		
		Args:
			fused_embeddings: Fused embeddings [batch, seq_len, dim]
			modality_ranges: Dict of (start, end) tuples
			modality_name: Name of modality to extract
			
		Returns:
			Modality tokens [batch, modality_seq_len, dim]
		"""
		start, end = modality_ranges[modality_name]
		return fused_embeddings[:, start:end, :]
