"""
Audio decoder for generating vocalizations.
Generates discrete audio tokens that are decoded to waveforms.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class AudioDecoder(nn.Module):
    """
    Audio decoder for generating vocalizations.
    
    Generates discrete audio tokens representing:
    - Speech sounds
    - Emotional vocalizations (gasps, laughter)
    - Breathing sounds
    
    Uses similar architecture to audio encoder's codebook.
    """
    
    def __init__(
        self,
        codebook_size: int = 1024,
        embedding_dim: int = 1536,
        dropout: float = 0.1,
        use_null_token: bool = True,
    ):
        """
        Initialize audio decoder.
        
        Args:
            codebook_size: Size of audio token vocabulary
            embedding_dim: Input embedding dimension
            dropout: Dropout probability
            use_null_token: Support NULL token (silence)
        """
        super().__init__()
        
        self.codebook_size = codebook_size
        self.embedding_dim = embedding_dim
        self.use_null_token = use_null_token
        
        # Projection to codebook size
        self.output_projection = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, codebook_size),
        )
        
        # Layer norm
        self.layer_norm = nn.LayerNorm(embedding_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # NULL token (silence)
        self.null_token_id = codebook_size - 1 if use_null_token else None
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        return_logits: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Generate audio tokens.
        
        Args:
            hidden_states: Hidden states [batch, seq_len, dim]
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling
            return_logits: Return logits instead of sampling
            
        Returns:
            Dictionary with audio tokens/logits
        """
        # Normalize and dropout
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)
        
        # Project to codebook
        logits = self.output_projection(hidden_states)
        # [batch, seq_len, codebook_size]
        
        if return_logits:
            probabilities = F.softmax(logits / temperature, dim=-1)
            return {
                "logits": logits,
                "probabilities": probabilities,
            }
        
        # Apply temperature
        logits = logits / temperature
        
        # Top-k filtering
        if top_k is not None:
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = float('-inf')
        
        # Top-p filtering
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
        tokens = torch.multinomial(
            probabilities.view(-1, self.codebook_size),
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
            target_tokens: Target audio tokens
            attention_mask: Attention mask
            
        Returns:
            Loss scalar
        """
        hidden_states = self.layer_norm(hidden_states)
        logits = self.output_projection(hidden_states)
        
        logits_flat = logits.view(-1, self.codebook_size)
        targets_flat = target_tokens.view(-1)
        
        loss = F.cross_entropy(logits_flat, targets_flat, reduction='none')
        
        if attention_mask is not None:
            mask_flat = attention_mask.view(-1).float()
            loss = (loss * mask_flat).sum() / mask_flat.sum()
        else:
            loss = loss.mean()
        
        return loss
    
    def decode_to_waveform(
        self,
        audio_tokens: torch.Tensor,
        vocoder: nn.Module,
    ) -> torch.Tensor:
        """
        Decode audio tokens to waveform using vocoder.
        
        Args:
            audio_tokens: Audio token indices [batch, seq_len]
            vocoder: Neural vocoder (e.g., EnCodec decoder)
            
        Returns:
            Waveform [batch, num_samples]
        """
        # This would use the vocoder to convert tokens → audio
        # Placeholder - actual implementation depends on vocoder choice
        # (EnCodec, HiFi-GAN, etc.)
        
        with torch.no_grad():
            waveform = vocoder.decode(audio_tokens)
        
        return waveform
