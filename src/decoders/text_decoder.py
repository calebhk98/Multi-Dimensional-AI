"""
Text decoders for internal and external text generation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class InternalTextDecoder(nn.Module):
	"""
	Decoder for internal text (creature's thoughts).
	
	Generates text tokens representing the creature's private reasoning.
	Can output NULL token to represent no internal thought.
	"""
	
	def __init__(
		self,
		vocab_size: int = 50257,
		embedding_dim: int = 1536,
		dropout: float = 0.1,
		use_null_token: bool = True,
	):
		"""
		Initialize internal text decoder.
		
		Args:
			vocab_size: Vocabulary size
			embedding_dim: Input embedding dimension
			dropout: Dropout probability
			use_null_token: Whether to support NULL token generation
		"""
		super().__init__()
		
		self.vocab_size = vocab_size
		self.embedding_dim = embedding_dim
		self.use_null_token = use_null_token
		
		# Projection head
		self.output_projection = nn.Linear(embedding_dim, vocab_size)
		
		# Layer norm before projection
		self.layer_norm = nn.LayerNorm(embedding_dim)
		
		# Dropout
		self.dropout = nn.Dropout(dropout)
		
		# NULL token index (assuming last token in vocab)
		self.null_token_id = vocab_size - 1 if use_null_token else None
	
	def forward(
		self,
		hidden_states: torch.Tensor,
		temperature: float = 1.0,
		top_k: Optional[int] = None,
		top_p: Optional[float] = None,
		return_logits: bool = False,
	) -> Dict[str, torch.Tensor]:
		"""
		Generate internal text tokens.
		
		Args:
			hidden_states: Hidden states from transformer [batch, seq_len, dim]
			temperature: Sampling temperature
			top_k: Top-k sampling
			top_p: Nucleus sampling threshold
			return_logits: Return raw logits instead of sampling
			
		Returns:
			Dictionary containing:
				- tokens: Generated token IDs [batch, seq_len] (if not return_logits)
				- logits: Raw logits [batch, seq_len, vocab_size] (if return_logits)
				- probabilities: Token probabilities [batch, seq_len, vocab_size]
		"""
		# Apply layer norm and dropout
		hidden_states = self.layer_norm(hidden_states)
		hidden_states = self.dropout(hidden_states)
		
		# Project to vocabulary
		logits = self.output_projection(hidden_states)
		# [batch, seq_len, vocab_size]
		
		if return_logits:
			probabilities = F.softmax(logits / temperature, dim=-1)
			return {
				"logits": logits,
				"probabilities": probabilities,
			}
		
		# Apply temperature
		logits = logits / temperature
		
		# Apply top-k filtering
		if top_k is not None:
			indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
			logits[indices_to_remove] = float('-inf')
		
		# Apply top-p (nucleus) filtering
		if top_p is not None:
			sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
			cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
			
			# Remove tokens with cumulative probability above threshold
			sorted_indices_to_remove = cumulative_probs > top_p
			sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
			sorted_indices_to_remove[..., 0] = 0
			
			# Scatter to original indexing
			indices_to_remove = sorted_indices_to_remove.scatter(
				-1, sorted_indices, sorted_indices_to_remove
			)
			logits[indices_to_remove] = float('-inf')
		
		# Sample from distribution
		probabilities = F.softmax(logits, dim=-1)
		
		# Ensure valid probabilities for multinomial
		# 1. Replace NaN with 0
		probabilities = torch.nan_to_num(probabilities, nan=0.0)
		
		# 2. Add epsilon to avoid all-zero rows if everything was masked (highly unlikely but possible)
		# or if precision issues caused underflow
		if (probabilities.sum(dim=-1) == 0).any():
			# Fallback: uniform distribution for failed rows, or just force argmax behavior
			# Let's add a tiny epsilon to all
			probabilities = probabilities + 1e-8
			probabilities = probabilities / probabilities.sum(dim=-1, keepdim=True)
			
		tokens = torch.multinomial(
			probabilities.view(-1, self.vocab_size),
			num_samples=1
		).view(probabilities.shape[:-1])
		
		return {
			"tokens": tokens,
			"probabilities": probabilities,
		}
	
	def compute_loss(
		self,
		hidden_states: torch.Tensor,
		target_tokens: torch.Tensor,
		attention_mask: Optional[torch.Tensor] = None,
	) -> torch.Tensor:
		"""
		Compute cross-entropy loss for training.
		
		Args:
			hidden_states: Hidden states [batch, seq_len, dim]
			target_tokens: Target token IDs [batch, seq_len]
			attention_mask: Attention mask [batch, seq_len]
			
		Returns:
			Loss scalar
		"""
		# Get logits
		hidden_states = self.layer_norm(hidden_states)
		logits = self.output_projection(hidden_states)
		
		# Flatten for loss computation
		logits_flat = logits.view(-1, self.vocab_size)
		targets_flat = target_tokens.view(-1)
		
		# Compute cross-entropy loss
		loss = F.cross_entropy(
			logits_flat,
			targets_flat,
			reduction='none'
		)
		
		# Apply attention mask if provided
		if attention_mask is not None:
			mask_flat = attention_mask.view(-1).float()
			loss = (loss * mask_flat).sum() / mask_flat.sum()
		else:
			loss = loss.mean()
		
		return loss


class ExternalTextDecoder(nn.Module):
	"""
	Decoder for external text (creature's speech).
	
	Generates text tokens representing what the creature says aloud.
	Separate from internal thoughts - can say different things than thinking.
	"""
	
	def __init__(
		self,
		vocab_size: int = 50257,
		embedding_dim: int = 1536,
		dropout: float = 0.1,
		use_null_token: bool = True,
	):
		"""
		Initialize external text decoder.
		
		Args:
			vocab_size: Vocabulary size
			embedding_dim: Input embedding dimension
			dropout: Dropout probability
			use_null_token: Whether to support NULL token (silence)
		"""
		super().__init__()
		
		self.vocab_size = vocab_size
		self.embedding_dim = embedding_dim
		self.use_null_token = use_null_token
		
		# Separate projection head from internal text
		self.output_projection = nn.Linear(embedding_dim, vocab_size)
		
		# Layer norm
		self.layer_norm = nn.LayerNorm(embedding_dim)
		
		# Dropout
		self.dropout = nn.Dropout(dropout)
		
		# NULL token for silence
		self.null_token_id = vocab_size - 1 if use_null_token else None
	
	def forward(
		self,
		hidden_states: torch.Tensor,
		temperature: float = 1.0,
		top_k: Optional[int] = None,
		top_p: Optional[float] = None,
		return_logits: bool = False,
	) -> Dict[str, torch.Tensor]:
		"""
		Generate external text tokens (same interface as internal decoder).
		
		Args:
			hidden_states: Hidden states from transformer
			temperature: Sampling temperature
			top_k: Top-k sampling
			top_p: Nucleus sampling threshold
			return_logits: Return raw logits
			
		Returns:
			Dictionary with tokens/logits and probabilities
		"""
		# Apply layer norm and dropout
		hidden_states = self.layer_norm(hidden_states)
		hidden_states = self.dropout(hidden_states)
		
		# Project to vocabulary
		logits = self.output_projection(hidden_states)
		
		if return_logits:
			probabilities = F.softmax(logits / temperature, dim=-1)
			return {
				"logits": logits,
				"probabilities": probabilities,
			}
		
		# Apply temperature
		logits = logits / temperature
		
		# Apply top-k filtering
		if top_k is not None:
			indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
			logits[indices_to_remove] = float('-inf')
		
		# Apply top-p filtering
		if top_p is not None:
			sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
			cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
			
			sorted_indices_to_remove = cumulative_probs > top_p
			sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
			sorted_indices_to_remove[..., 0] = 0
			
			indices_to_remove = sorted_indices_to_remove.scatter(
				-1, sorted_indices, sorted_indices_to_remove
			)
			logits[indices_to_remove] = float('-inf')
		
		# Sample
		probabilities = F.softmax(logits, dim=-1)
		
		# Ensure valid probabilities
		probabilities = torch.nan_to_num(probabilities, nan=0.0)
		
		if (probabilities.sum(dim=-1) == 0).any():
			probabilities = probabilities + 1e-8
			probabilities = probabilities / probabilities.sum(dim=-1, keepdim=True)

		tokens = torch.multinomial(
			probabilities.view(-1, self.vocab_size),
			num_samples=1
		).view(probabilities.shape[:-1])
		
		return {
			"tokens": tokens,
			"probabilities": probabilities,
		}
	
	def compute_loss(
		self,
		hidden_states: torch.Tensor,
		target_tokens: torch.Tensor,
		attention_mask: Optional[torch.Tensor] = None,
	) -> torch.Tensor:
		"""
		Compute cross-entropy loss.
		
		Args:
			hidden_states: Hidden states
			target_tokens: Target tokens
			attention_mask: Attention mask
			
		Returns:
			Loss scalar
		"""
		hidden_states = self.layer_norm(hidden_states)
		logits = self.output_projection(hidden_states)
		
		logits_flat = logits.view(-1, self.vocab_size)
		targets_flat = target_tokens.view(-1)
		
		loss = F.cross_entropy(logits_flat, targets_flat, reduction='none')
		
		if attention_mask is not None:
			mask_flat = attention_mask.view(-1).float()
			loss = (loss * mask_flat).sum() / mask_flat.sum()
		else:
			loss = loss.mean()
		
		return loss
