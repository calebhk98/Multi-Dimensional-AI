"""
Tests for TextOnlyTransformer model.
"""

import pytest
import torch
import torch.nn as nn

from src.models.text_only_transformer import TextOnlyTransformer, TransformerBlock


@pytest.fixture
def minimal_config():
    """Minimal configuration for TextOnlyTransformer."""
    return {
        "model": {
            "transformer": {
                "hidden_dim": 64,
                "num_layers": 2,
                "num_attention_heads": 4,
                "ffn_dim": 256,
                "dropout": 0.1,
            },
            "encoders": {
                "internal_voice": {
                    "vocab_size": 1000,
                    "max_seq_length": 128,
                }
            },
        }
    }


@pytest.fixture
def model(minimal_config):
    """Create TextOnlyTransformer instance."""
    return TextOnlyTransformer(minimal_config)


@pytest.fixture
def batch_size():
    """Standard batch size for tests."""
    return 4


@pytest.fixture
def seq_len():
    """Standard sequence length for tests."""
    return 32


@pytest.fixture
def input_ids(batch_size, seq_len):
    """Generate dummy input token IDs."""
    return torch.randint(0, 1000, (batch_size, seq_len))


class TestTextOnlyTransformerInit:
    """Tests for TextOnlyTransformer initialization."""

    def test_init_with_config(self, minimal_config):
        """Test initialization with config dict."""
        model = TextOnlyTransformer(minimal_config)
        assert model.hidden_dim == 64
        assert model.num_layers == 2
        assert model.num_heads == 4
        assert model.ffn_dim == 256
        assert model.vocab_size == 1000
        assert model.max_seq_length == 128

    def test_init_default_values(self):
        """Test initialization with empty config uses defaults."""
        config = {"model": {}}
        model = TextOnlyTransformer(config)
        assert model.vocab_size == 50257  # GPT-2 default
        assert model.hidden_dim == 768
        assert model.num_layers == 12
        assert model.num_heads == 12
        assert model.ffn_dim == 3072
        assert model.max_seq_length == 512
        assert model.dropout == 0.1

    def test_token_embedding_created(self, model):
        """Test that token embedding is created with correct dimensions."""
        assert isinstance(model.token_embedding, nn.Embedding)
        assert model.token_embedding.num_embeddings == 1000
        assert model.token_embedding.embedding_dim == 64

    def test_position_embedding_created(self, model):
        """Test that position embedding is created with correct dimensions."""
        assert isinstance(model.position_embedding, nn.Embedding)
        assert model.position_embedding.num_embeddings == 128
        assert model.position_embedding.embedding_dim == 64

    def test_transformer_layers_created(self, model):
        """Test that correct number of transformer layers are created."""
        assert len(model.layers) == 2
        for layer in model.layers:
            assert isinstance(layer, TransformerBlock)

    def test_layer_norm_created(self, model):
        """Test that final layer norm is created."""
        assert isinstance(model.ln_f, nn.LayerNorm)
        assert model.ln_f.normalized_shape == (64,)

    def test_lm_head_created(self, model):
        """Test that LM head is created with correct dimensions."""
        assert isinstance(model.lm_head, nn.Linear)
        assert model.lm_head.in_features == 64
        assert model.lm_head.out_features == 1000
        assert model.lm_head.bias is None

    def test_gradient_checkpointing_off_by_default(self, model):
        """Test that gradient checkpointing is off by default."""
        assert model._gradient_checkpointing is False


class TestForwardPass:
    """Tests for forward pass."""

    def test_forward_pass_basic(self, model, input_ids):
        """Test basic forward pass returns correct output shape."""
        outputs = model(input_ids)
        assert "logits" in outputs
        assert outputs["logits"].shape == (4, 32, 1000)

    def test_forward_pass_returns_dict(self, model, input_ids):
        """Test that forward pass returns a dictionary."""
        outputs = model(input_ids)
        assert isinstance(outputs, dict)

    def test_forward_pass_different_batch_sizes(self, model):
        """Test forward pass with different batch sizes."""
        for batch_size in [1, 2, 8, 16]:
            input_ids = torch.randint(0, 1000, (batch_size, 32))
            outputs = model(input_ids)
            assert outputs["logits"].shape == (batch_size, 32, 1000)

    def test_forward_pass_different_seq_lengths(self, model):
        """Test forward pass with different sequence lengths."""
        for seq_len in [10, 32, 64, 128]:
            input_ids = torch.randint(0, 1000, (4, seq_len))
            outputs = model(input_ids)
            assert outputs["logits"].shape == (4, seq_len, 1000)

    def test_forward_pass_with_attention_mask(self, model, input_ids):
        """Test forward pass with attention mask (currently not used but should not error)."""
        attention_mask = torch.ones(4, 32)
        outputs = model(input_ids, attention_mask=attention_mask)
        assert outputs["logits"].shape == (4, 32, 1000)

    def test_forward_returns_hidden_states_when_requested(self, model, input_ids):
        """Test that hidden states are returned when requested."""
        outputs = model(input_ids, return_hidden_states=True)
        assert "hidden_states" in outputs
        assert outputs["hidden_states"].shape == (4, 32, 64)

    def test_forward_no_hidden_states_by_default(self, model, input_ids):
        """Test that hidden states are not returned by default."""
        outputs = model(input_ids, return_hidden_states=False)
        assert "hidden_states" not in outputs

    def test_forward_pass_no_nan(self, model, input_ids):
        """Test that forward pass produces no NaN values."""
        outputs = model(input_ids)
        assert not torch.isnan(outputs["logits"]).any()

    def test_forward_pass_no_inf(self, model, input_ids):
        """Test that forward pass produces no infinite values."""
        outputs = model(input_ids)
        assert not torch.isinf(outputs["logits"]).any()


class TestCausalMasking:
    """Tests for causal attention masking."""

    def test_causal_mask_prevents_future_attention(self, model, input_ids):
        """Test that causal mask prevents attending to future tokens."""
        # Run forward pass and check outputs are different for different positions
        model.eval()
        with torch.no_grad():
            outputs = model(input_ids)

        # If causal masking works, changing future tokens shouldn't affect past predictions
        input_ids_modified = input_ids.clone()
        input_ids_modified[:, -1] = 999  # Change last token

        with torch.no_grad():
            outputs_modified = model(input_ids_modified)

        # First n-1 positions should have same logits
        assert torch.allclose(
            outputs["logits"][:, :-1, :],
            outputs_modified["logits"][:, :-1, :],
            atol=1e-5,
        )

    def test_position_embedding_sequence_handling(self, model):
        """Test that position embeddings handle different sequence lengths."""
        for seq_len in [1, 16, 64, 128]:
            input_ids = torch.randint(0, 1000, (2, seq_len))
            outputs = model(input_ids)
            assert outputs["logits"].shape == (2, seq_len, 1000)


class TestComputeLoss:
    """Tests for compute_loss method."""

    def test_compute_loss_returns_tuple(self, model, input_ids):
        """Test that compute_loss returns a tuple of (loss, loss_dict)."""
        outputs = model(input_ids)
        targets = {"internal_text": torch.randint(0, 1000, (4, 32))}
        loss, loss_dict = model.compute_loss(outputs, targets)

        assert isinstance(loss, torch.Tensor)
        assert isinstance(loss_dict, dict)

    def test_compute_loss_cross_entropy(self, model, input_ids):
        """Test that compute_loss computes cross-entropy loss."""
        outputs = model(input_ids)
        targets = {"internal_text": torch.randint(0, 1000, (4, 32))}
        loss, loss_dict = model.compute_loss(outputs, targets)

        assert loss.dim() == 0  # Scalar
        assert loss.item() > 0  # Loss should be positive

    def test_compute_loss_dict_contains_keys(self, model, input_ids):
        """Test that loss_dict contains expected keys."""
        outputs = model(input_ids)
        targets = {"internal_text": torch.randint(0, 1000, (4, 32))}
        _, loss_dict = model.compute_loss(outputs, targets)

        assert "total_loss" in loss_dict
        assert "cross_entropy" in loss_dict

    def test_compute_loss_with_hidden_states(self, model, input_ids):
        """Test compute_loss when outputs contain hidden_states instead of logits."""
        outputs = model(input_ids, return_hidden_states=True)
        # Remove logits to test the hidden_states path
        outputs_no_logits = {"hidden_states": outputs["hidden_states"]}
        targets = {"internal_text": torch.randint(0, 1000, (4, 32))}

        loss, loss_dict = model.compute_loss(outputs_no_logits, targets)
        assert loss.item() > 0

    def test_compute_loss_with_target_key(self, model, input_ids):
        """Test compute_loss with 'target' key instead of 'internal_text'."""
        outputs = model(input_ids)
        targets = {"target": torch.randint(0, 1000, (4, 32))}
        loss, _ = model.compute_loss(outputs, targets)
        assert loss.item() > 0

    def test_compute_loss_ignores_padding(self, model, input_ids):
        """Test that compute_loss ignores padding tokens (index -100)."""
        outputs = model(input_ids)
        targets = torch.randint(0, 1000, (4, 32))
        targets[:, -10:] = -100  # Last 10 tokens are padding
        loss, _ = model.compute_loss(outputs, {"internal_text": targets})
        assert loss.item() > 0

    def test_compute_loss_loss_weights_ignored(self, model, input_ids):
        """Test that loss_weights parameter is ignored (for API compatibility)."""
        outputs = model(input_ids)
        targets = {"internal_text": torch.randint(0, 1000, (4, 32))}
        loss1, _ = model.compute_loss(outputs, targets, loss_weights=None)
        loss2, _ = model.compute_loss(outputs, targets, loss_weights={"text": 2.0})
        assert loss1.item() == loss2.item()


class TestGradientCheckpointing:
    """Tests for gradient checkpointing."""

    def test_enable_gradient_checkpointing(self, model):
        """Test enabling gradient checkpointing."""
        model.enable_gradient_checkpointing(True)
        assert model._gradient_checkpointing is True

    def test_disable_gradient_checkpointing(self, model):
        """Test disabling gradient checkpointing."""
        model.enable_gradient_checkpointing(True)
        model.enable_gradient_checkpointing(False)
        assert model._gradient_checkpointing is False

    def test_gradient_checkpointing_in_training(self, model, input_ids):
        """Test that gradient checkpointing works in training mode."""
        model.train()
        model.enable_gradient_checkpointing(True)

        outputs = model(input_ids)
        targets = {"internal_text": torch.randint(0, 1000, (4, 32))}
        loss, _ = model.compute_loss(outputs, targets)

        # Should be able to backward through checkpointed layers
        loss.backward()

        # Gradients should exist
        assert model.token_embedding.weight.grad is not None

    def test_gradient_checkpointing_not_used_in_eval(self, model, input_ids):
        """Test that gradient checkpointing is not used in eval mode."""
        model.eval()
        model.enable_gradient_checkpointing(True)

        # Should run without error (checkpointing only affects training)
        with torch.no_grad():
            outputs = model(input_ids)
        assert outputs["logits"].shape == (4, 32, 1000)


class TestTransformerBlock:
    """Tests for TransformerBlock component."""

    @pytest.fixture
    def block(self):
        """Create a TransformerBlock."""
        return TransformerBlock(
            hidden_dim=64,
            num_heads=4,
            ffn_dim=256,
            dropout=0.1,
        )

    def test_block_init(self, block):
        """Test TransformerBlock initialization."""
        assert isinstance(block.ln1, nn.LayerNorm)
        assert isinstance(block.ln2, nn.LayerNorm)
        assert isinstance(block.attn, nn.MultiheadAttention)
        assert isinstance(block.ffn, nn.Sequential)

    def test_block_forward(self, block):
        """Test TransformerBlock forward pass."""
        x = torch.randn(4, 32, 64)
        causal_mask = torch.triu(torch.ones(32, 32, dtype=torch.bool), diagonal=1)

        output = block(x, causal_mask)
        assert output.shape == (4, 32, 64)

    def test_block_residual_connections(self, block):
        """Test that block uses residual connections."""
        x = torch.randn(4, 32, 64)
        causal_mask = torch.triu(torch.ones(32, 32, dtype=torch.bool), diagonal=1)

        # With residual connections, output should be correlated with input
        output = block(x, causal_mask)

        # The output should not be identical to input (transformations applied)
        assert not torch.allclose(output, x)

        # But should be in similar range due to residual connections
        assert output.abs().mean() < x.abs().mean() * 10

    def test_block_no_nan_output(self, block):
        """Test that block produces no NaN values."""
        x = torch.randn(4, 32, 64)
        causal_mask = torch.triu(torch.ones(32, 32, dtype=torch.bool), diagonal=1)

        output = block(x, causal_mask)
        assert not torch.isnan(output).any()


class TestWeightInitialization:
    """Tests for weight initialization."""

    def test_linear_weights_initialized(self, model):
        """Test that linear weights are initialized with GPT-2 style init."""
        # GPT-2 uses normal(0, 0.02) for linear weights
        for name, param in model.named_parameters():
            if "weight" in name and "ln" not in name and "embedding" not in name:
                # Weights should be roughly normal with std ~0.02
                # Allow some tolerance for small layers
                assert param.std() < 0.1, f"Param {name} has std {param.std()}"

    def test_layer_norm_initialized(self, model):
        """Test that layer norms are initialized correctly."""
        assert torch.allclose(model.ln_f.weight, torch.ones_like(model.ln_f.weight))
        assert torch.allclose(model.ln_f.bias, torch.zeros_like(model.ln_f.bias))


class TestIntegration:
    """Integration tests for TextOnlyTransformer."""

    def test_training_step(self, model, input_ids):
        """Test a full training step."""
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        targets = {"internal_text": torch.randint(0, 1000, (4, 32))}

        # Forward
        outputs = model(input_ids)
        loss, _ = model.compute_loss(outputs, targets)

        # Backward
        optimizer.zero_grad()
        loss.backward()

        # Check gradients exist
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None

        # Update
        optimizer.step()

    def test_inference_mode(self, model, input_ids):
        """Test inference mode."""
        model.eval()
        with torch.no_grad():
            outputs = model(input_ids)

        assert outputs["logits"].shape == (4, 32, 1000)

    def test_model_can_overfit_single_example(self, minimal_config):
        """Test that model can overfit a single example (sanity check)."""
        model = TextOnlyTransformer(minimal_config)
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        input_ids = torch.randint(0, 1000, (1, 16))
        targets = {"internal_text": torch.randint(0, 1000, (1, 16))}

        initial_loss = None
        for _ in range(100):
            outputs = model(input_ids)
            loss, _ = model.compute_loss(outputs, targets)

            if initial_loss is None:
                initial_loss = loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        final_loss = loss.item()
        assert final_loss < initial_loss, "Model should be able to reduce loss"
