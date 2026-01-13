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
    ):
        """
        Initialize audio encoder.
        
        Args:
            sample_rate: Audio sample rate in Hz
            hop_length: Hop length for frame extraction
            embedding_dim: Output embedding dimension
            num_conv_layers: Number of convolutional layers
            conv_channels: Number of channels in conv layers
            codebook_size: Size of discrete audio codebook
            dropout: Dropout probability
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
        Quantize continuous features to discrete codes.
        
        Args:
            features: Continuous features [batch, seq_len, embed_dim]
            
        Returns:
            Tuple of:
                - quantized: Quantized features [batch, seq_len, embed_dim]
                - indices: Codebook indices [batch, seq_len]
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
        Encode audio waveform.
        
        Args:
            waveform: Raw audio [batch_size, num_samples]
            return_indices: Whether to return discrete indices
            
        Returns:
            Dictionary containing:
                - embeddings: Encoded embeddings [batch, seq_len, embed_dim]
                - indices: Discrete codes [batch, seq_len] (if return_indices=True)
                - attention_mask: All ones [batch, seq_len]
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
        """Get output embedding dimension."""
        return self.embedding_dim
