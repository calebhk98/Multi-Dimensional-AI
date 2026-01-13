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
        Initialize internal voice encoder.
        
        Args:
            vocab_size: Vocabulary size
            embedding_dim: Dimension of token embeddings
            max_seq_length: Maximum sequence length
            dropout: Dropout probability
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
        Encode internal voice tokens.
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            Dictionary containing:
                - embeddings: Encoded embeddings [batch_size, seq_len, embed_dim]
                - attention_mask: Attention mask [batch_size, seq_len]
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
        """Get output embedding dimension."""
        return self.embedding_dim
