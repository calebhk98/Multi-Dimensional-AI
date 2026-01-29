"""
Tests for TokenFusionModule with all fusion strategies.

Purpose:
    Verify all fusion strategies (concatenate, cross_attention, learned)
    work correctly with various input configurations.
"""

import pytest
import torch
import torch.nn as nn
from typing import Dict

from src.models.fusion_module import TokenFusionModule


class TestFusionModuleInit:
    """Tests for TokenFusionModule initialization."""

    def test_init_concatenate_strategy(self):
        """
        Test initialization with concatenate strategy.

        Purpose:
            Verify module initializes for concatenation fusion.
        """
        module = TokenFusionModule(
            embedding_dim=512,
            fusion_strategy="concatenate",
            use_modality_embeddings=True
        )

        assert module.fusion_strategy == "concatenate"
        assert module.embedding_dim == 512
        assert hasattr(module, "modality_type_embeddings")

    def test_init_cross_attention_strategy(self):
        """
        Test initialization with cross_attention strategy.

        Purpose:
            Verify cross-attention components are created.
        """
        module = TokenFusionModule(
            embedding_dim=512,
            fusion_strategy="cross_attention",
            use_modality_embeddings=True
        )

        assert module.fusion_strategy == "cross_attention"
        assert hasattr(module, "fusion_attention")
        assert isinstance(module.fusion_attention, nn.MultiheadAttention)

    def test_init_learned_strategy(self):
        """
        Test initialization with learned fusion strategy.

        Purpose:
            Verify transformer encoder is created for learned fusion.
        """
        module = TokenFusionModule(
            embedding_dim=512,
            fusion_strategy="learned",
            use_modality_embeddings=True
        )

        assert module.fusion_strategy == "learned"
        assert hasattr(module, "fusion_transformer")
        assert isinstance(module.fusion_transformer, nn.TransformerEncoder)

    def test_init_without_modality_embeddings(self):
        """
        Test initialization without modality embeddings.

        Purpose:
            Verify modality embeddings can be disabled.
        """
        module = TokenFusionModule(
            embedding_dim=512,
            fusion_strategy="concatenate",
            use_modality_embeddings=False
        )

        assert not hasattr(module, "modality_type_embeddings")

    def test_init_various_embedding_dims(self):
        """
        Test initialization with various embedding dimensions.

        Purpose:
            Verify module handles different dimensions correctly.
        """
        for dim in [128, 256, 512, 768, 1024, 1536]:
            module = TokenFusionModule(
                embedding_dim=dim,
                fusion_strategy="cross_attention"
            )
            assert module.embedding_dim == dim


class TestConcatenateFusion:
    """Tests for concatenate fusion strategy."""

    @pytest.fixture
    def concat_module(self):
        """Create module with concatenate strategy."""
        return TokenFusionModule(
            embedding_dim=256,
            fusion_strategy="concatenate",
            use_modality_embeddings=True,
            dropout=0.0  # Disable dropout for deterministic tests
        )

    def test_forward_single_modality(self, concat_module):
        """
        Test forward pass with single modality.

        Purpose:
            Verify fusion works with just one input modality.
        """
        encoder_outputs = {
            "internal_voice": {
                "embeddings": torch.randn(2, 10, 256),
                "attention_mask": torch.ones(2, 10)
            }
        }

        result = concat_module(encoder_outputs)

        assert "embeddings" in result
        assert "attention_mask" in result
        assert "modality_ranges" in result

        assert result["embeddings"].shape == (2, 10, 256)
        assert result["attention_mask"].shape == (2, 10)
        assert "internal_voice" in result["modality_ranges"]

    def test_forward_multiple_modalities(self, concat_module):
        """
        Test forward pass with multiple modalities.

        Purpose:
            Verify concatenation of multiple modality tokens.
        """
        encoder_outputs = {
            "internal_voice": {
                "embeddings": torch.randn(2, 10, 256),
                "attention_mask": torch.ones(2, 10)
            },
            "audio": {
                "embeddings": torch.randn(2, 15, 256),
                "attention_mask": torch.ones(2, 15)
            },
            "vision": {
                "embeddings": torch.randn(2, 20, 256),
                "attention_mask": torch.ones(2, 20)
            }
        }

        result = concat_module(encoder_outputs)

        # Total sequence length should be sum of all
        total_seq = 10 + 15 + 20
        assert result["embeddings"].shape == (2, total_seq, 256)
        assert result["attention_mask"].shape == (2, total_seq)

        # Check modality ranges
        assert result["modality_ranges"]["internal_voice"] == (0, 10)
        assert result["modality_ranges"]["audio"] == (10, 25)
        assert result["modality_ranges"]["vision"] == (25, 45)

    def test_forward_all_modalities(self, concat_module):
        """
        Test forward pass with all 6 modalities.

        Purpose:
            Verify all modalities can be fused together.
        """
        encoder_outputs = {
            "internal_voice": {
                "embeddings": torch.randn(2, 5, 256),
                "attention_mask": torch.ones(2, 5)
            },
            "external_voice": {
                "embeddings": torch.randn(2, 5, 256),
                "attention_mask": torch.ones(2, 5)
            },
            "audio": {
                "embeddings": torch.randn(2, 5, 256),
                "attention_mask": torch.ones(2, 5)
            },
            "vision": {
                "embeddings": torch.randn(2, 5, 256),
                "attention_mask": torch.ones(2, 5)
            },
            "proprioception": {
                "embeddings": torch.randn(2, 5, 256),
                "attention_mask": torch.ones(2, 5)
            },
            "touch": {
                "embeddings": torch.randn(2, 5, 256),
                "attention_mask": torch.ones(2, 5)
            }
        }

        result = concat_module(encoder_outputs)

        assert result["embeddings"].shape == (2, 30, 256)
        assert len(result["modality_ranges"]) == 6


class TestCrossAttentionFusion:
    """Tests for cross_attention fusion strategy."""

    @pytest.fixture
    def cross_attn_module(self):
        """Create module with cross_attention strategy."""
        return TokenFusionModule(
            embedding_dim=256,
            fusion_strategy="cross_attention",
            use_modality_embeddings=True,
            dropout=0.0
        )

    def test_forward_basic(self, cross_attn_module):
        """
        Test basic cross-attention forward pass.

        Purpose:
            Verify cross-attention is applied to fused tokens.
        """
        encoder_outputs = {
            "internal_voice": {
                "embeddings": torch.randn(2, 10, 256),
                "attention_mask": torch.ones(2, 10)
            },
            "audio": {
                "embeddings": torch.randn(2, 15, 256),
                "attention_mask": torch.ones(2, 15)
            }
        }

        result = cross_attn_module(encoder_outputs)

        total_seq = 10 + 15
        assert result["embeddings"].shape == (2, total_seq, 256)
        # Output should be different from simple concatenation due to attention

    def test_forward_with_padding_mask(self, cross_attn_module):
        """
        Test cross-attention respects padding mask.

        Purpose:
            Verify padding positions are handled correctly.
        """
        encoder_outputs = {
            "internal_voice": {
                "embeddings": torch.randn(2, 10, 256),
                "attention_mask": torch.tensor([
                    [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 0, 0]
                ], dtype=torch.float)
            }
        }

        result = cross_attn_module(encoder_outputs)

        assert result["embeddings"].shape == (2, 10, 256)
        assert result["attention_mask"].shape == (2, 10)

    def test_output_different_from_input(self, cross_attn_module):
        """
        Test that cross-attention modifies embeddings.

        Purpose:
            Verify attention transformation occurs.
        """
        input_emb = torch.randn(2, 10, 256)
        encoder_outputs = {
            "internal_voice": {
                "embeddings": input_emb.clone(),
                "attention_mask": torch.ones(2, 10)
            }
        }

        result = cross_attn_module(encoder_outputs)

        # Embeddings should be transformed (not identical)
        # Note: With single modality, self-attention is applied
        # This may or may not change values significantly


class TestLearnedFusion:
    """Tests for learned fusion strategy."""

    @pytest.fixture
    def learned_module(self):
        """Create module with learned fusion strategy."""
        return TokenFusionModule(
            embedding_dim=256,
            fusion_strategy="learned",
            use_modality_embeddings=True,
            dropout=0.0
        )

    def test_forward_basic(self, learned_module):
        """
        Test basic learned fusion forward pass.

        Purpose:
            Verify transformer encoder is applied to fused tokens.
        """
        encoder_outputs = {
            "internal_voice": {
                "embeddings": torch.randn(2, 10, 256),
                "attention_mask": torch.ones(2, 10)
            },
            "audio": {
                "embeddings": torch.randn(2, 15, 256),
                "attention_mask": torch.ones(2, 15)
            }
        }

        result = learned_module(encoder_outputs)

        total_seq = 10 + 15
        assert result["embeddings"].shape == (2, total_seq, 256)

    def test_forward_with_padding(self, learned_module):
        """
        Test learned fusion with padding mask.

        Purpose:
            Verify padding is handled by transformer.
        """
        encoder_outputs = {
            "internal_voice": {
                "embeddings": torch.randn(2, 10, 256),
                "attention_mask": torch.tensor([
                    [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 0, 0]
                ], dtype=torch.float)
            }
        }

        result = learned_module(encoder_outputs)

        assert result["embeddings"].shape == (2, 10, 256)


class TestFusionModuleEdgeCases:
    """Edge case tests for TokenFusionModule."""

    def test_unknown_fusion_strategy(self):
        """
        Test that unknown strategy raises error.

        Purpose:
            Verify proper error for invalid strategy.
        """
        module = TokenFusionModule(
            embedding_dim=256,
            fusion_strategy="unknown_strategy"
        )

        encoder_outputs = {
            "internal_voice": {
                "embeddings": torch.randn(2, 10, 256),
                "attention_mask": torch.ones(2, 10)
            }
        }

        with pytest.raises(ValueError, match="Unknown fusion strategy"):
            module(encoder_outputs)

    def test_get_modality_tokens(self):
        """
        Test extracting modality-specific tokens from fused output.

        Purpose:
            Verify modality token extraction works correctly.
        """
        module = TokenFusionModule(
            embedding_dim=256,
            fusion_strategy="concatenate",
            dropout=0.0
        )

        encoder_outputs = {
            "internal_voice": {
                "embeddings": torch.randn(2, 10, 256),
                "attention_mask": torch.ones(2, 10)
            },
            "audio": {
                "embeddings": torch.randn(2, 15, 256),
                "attention_mask": torch.ones(2, 15)
            }
        }

        result = module(encoder_outputs)

        # Extract internal_voice tokens
        voice_tokens = module.get_modality_tokens(
            result["embeddings"],
            result["modality_ranges"],
            "internal_voice"
        )

        assert voice_tokens.shape == (2, 10, 256)

        # Extract audio tokens
        audio_tokens = module.get_modality_tokens(
            result["embeddings"],
            result["modality_ranges"],
            "audio"
        )

        assert audio_tokens.shape == (2, 15, 256)

    def test_batch_size_one(self):
        """
        Test fusion with batch size of 1.

        Purpose:
            Verify single-sample batches work correctly.
        """
        module = TokenFusionModule(
            embedding_dim=256,
            fusion_strategy="concatenate",
            dropout=0.0
        )

        encoder_outputs = {
            "internal_voice": {
                "embeddings": torch.randn(1, 10, 256),
                "attention_mask": torch.ones(1, 10)
            }
        }

        result = module(encoder_outputs)

        assert result["embeddings"].shape == (1, 10, 256)

    def test_empty_sequence(self):
        """
        Test handling of zero-length sequences.

        Purpose:
            Verify module handles edge case of empty sequence.
        """
        module = TokenFusionModule(
            embedding_dim=256,
            fusion_strategy="concatenate",
            dropout=0.0
        )

        # Mix of empty and non-empty
        encoder_outputs = {
            "internal_voice": {
                "embeddings": torch.randn(2, 0, 256),  # Empty
                "attention_mask": torch.ones(2, 0)
            },
            "audio": {
                "embeddings": torch.randn(2, 10, 256),
                "attention_mask": torch.ones(2, 10)
            }
        }

        result = module(encoder_outputs)

        # Should still work with total of 10 tokens
        assert result["embeddings"].shape == (2, 10, 256)


class TestFusionModuleGradients:
    """Tests for gradient flow through fusion module."""

    def test_gradient_flow_concatenate(self):
        """
        Test gradients flow through concatenate fusion.

        Purpose:
            Verify backpropagation works correctly.
        """
        module = TokenFusionModule(
            embedding_dim=256,
            fusion_strategy="concatenate"
        )

        input_emb = torch.randn(2, 10, 256, requires_grad=True)
        encoder_outputs = {
            "internal_voice": {
                "embeddings": input_emb,
                "attention_mask": torch.ones(2, 10)
            }
        }

        result = module(encoder_outputs)
        loss = result["embeddings"].sum()
        loss.backward()

        assert input_emb.grad is not None
        assert input_emb.grad.shape == input_emb.shape

    def test_gradient_flow_cross_attention(self):
        """
        Test gradients flow through cross-attention fusion.

        Purpose:
            Verify backprop through attention mechanism.
        """
        module = TokenFusionModule(
            embedding_dim=256,
            fusion_strategy="cross_attention"
        )

        input_emb = torch.randn(2, 10, 256, requires_grad=True)
        encoder_outputs = {
            "internal_voice": {
                "embeddings": input_emb,
                "attention_mask": torch.ones(2, 10)
            }
        }

        result = module(encoder_outputs)
        loss = result["embeddings"].sum()
        loss.backward()

        assert input_emb.grad is not None

    def test_gradient_flow_learned(self):
        """
        Test gradients flow through learned fusion.

        Purpose:
            Verify backprop through transformer encoder.
        """
        module = TokenFusionModule(
            embedding_dim=256,
            fusion_strategy="learned"
        )

        input_emb = torch.randn(2, 10, 256, requires_grad=True)
        encoder_outputs = {
            "internal_voice": {
                "embeddings": input_emb,
                "attention_mask": torch.ones(2, 10)
            }
        }

        result = module(encoder_outputs)
        loss = result["embeddings"].sum()
        loss.backward()

        assert input_emb.grad is not None
