"""
Lightweight text-only transformer for efficient LLM training.
This is a pure GPT-2 style decoder-only transformer without multi-modal overhead.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import math


class TextOnlyTransformer(nn.Module):
    """
    Pure text-only transformer (GPT-2 style).

    No multi-modal encoders/decoders - just embeddings -> transformer -> LM head.
    This achieves similar parameter counts to standard GPT models.
    """

    def __init__(self, config: Dict):
        """
        Initialize text-only transformer.

        Args:
            config: Configuration dictionary with model settings
        """
        super().__init__()

        self.config = config
        model_config = config.get("model", {})
        transformer_config = model_config.get("transformer", {})
        encoder_config = model_config.get("encoders", {}).get("internal_voice", {})

        # Core dimensions
        self.vocab_size = encoder_config.get("vocab_size", 50257)
        self.hidden_dim = transformer_config.get("hidden_dim", 768)
        self.num_layers = transformer_config.get("num_layers", 12)
        self.num_heads = transformer_config.get("num_attention_heads", 12)
        self.ffn_dim = transformer_config.get("ffn_dim", 3072)
        self.max_seq_length = encoder_config.get("max_seq_length", 512)
        self.dropout = transformer_config.get("dropout", 0.1)

        # Token and position embeddings
        self.token_embedding = nn.Embedding(self.vocab_size, self.hidden_dim)
        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_dim)
        self.dropout_layer = nn.Dropout(self.dropout)

        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(
                hidden_dim=self.hidden_dim,
                num_heads=self.num_heads,
                ffn_dim=self.ffn_dim,
                dropout=self.dropout,
            )
            for _ in range(self.num_layers)
        ])

        # Final layer norm
        self.ln_f = nn.LayerNorm(self.hidden_dim)

        # LM head (no weight tying for simplicity, can add later)
        self.lm_head = nn.Linear(self.hidden_dim, self.vocab_size, bias=False)

        # Initialize weights
        self.apply(self._init_weights)

        # Gradient checkpointing flag
        self._gradient_checkpointing = False

    def _init_weights(self, module):
        """Initialize weights with GPT-2 style initialization."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def enable_gradient_checkpointing(self, enable: bool = True):
        """Enable or disable gradient checkpointing."""
        self._gradient_checkpointing = enable

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_hidden_states: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            input_ids: Token IDs [batch, seq_len]
            attention_mask: Optional attention mask [batch, seq_len]
            return_hidden_states: Whether to return hidden states

        Returns:
            Dictionary with 'logits' and optionally 'hidden_states'
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Create position IDs
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)

        # Embeddings
        token_emb = self.token_embedding(input_ids)
        pos_emb = self.position_embedding(position_ids)
        hidden_states = self.dropout_layer(token_emb + pos_emb)

        # Create causal mask
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
            diagonal=1
        )

        # Apply transformer layers
        for layer in self.layers:
            if self._gradient_checkpointing and self.training:
                hidden_states = torch.utils.checkpoint.checkpoint(
                    layer,
                    hidden_states,
                    causal_mask,
                    use_reentrant=False,
                )
            else:
                hidden_states = layer(hidden_states, causal_mask)

        # Final layer norm
        hidden_states = self.ln_f(hidden_states)

        # LM head
        logits = self.lm_head(hidden_states)

        outputs = {"logits": logits}
        if return_hidden_states:
            outputs["hidden_states"] = hidden_states

        return outputs

    def compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        loss_weights: Optional[Dict[str, float]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute cross-entropy loss.

        Args:
            outputs: Model outputs with 'logits' or 'hidden_states'
            targets: Target dictionary with 'internal_text'
            loss_weights: Ignored (for API compatibility)

        Returns:
            Tuple of (loss, loss_dict)
        """
        # Get logits - either directly or compute from hidden states
        if "logits" in outputs:
            logits = outputs["logits"]
        else:
            hidden_states = outputs["hidden_states"]
            logits = self.lm_head(hidden_states)

        # Get targets
        target_ids = targets.get("internal_text", targets.get("target"))

        # Flatten for cross entropy
        logits_flat = logits.view(-1, self.vocab_size)
        targets_flat = target_ids.view(-1)

        loss = F.cross_entropy(logits_flat, targets_flat, ignore_index=-100)

        return loss, {"total_loss": loss.item(), "cross_entropy": loss.item()}


class TransformerBlock(nn.Module):
    """Single transformer block with pre-norm architecture."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.ln1 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.ln2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        causal_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass with pre-norm residual connections.

        Args:
            x: Input tensor [batch, seq, hidden]
            causal_mask: Boolean causal attention mask [seq, seq]

        Returns:
            Output tensor [batch, seq, hidden]
        """
        # Self-attention with pre-norm
        normed = self.ln1(x)
        attn_out, _ = self.attn(
            normed, normed, normed,
            attn_mask=causal_mask,
            is_causal=True,
        )
        x = x + attn_out

        # FFN with pre-norm
        x = x + self.ffn(self.ln2(x))

        return x
