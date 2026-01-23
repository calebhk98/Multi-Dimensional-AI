"""
End-to-end workflow tests.
Tests complete workflows from configuration to inference.
"""

import pytest
import torch
import yaml
from pathlib import Path
from src.config import Config


class TestConfigurationWorkflow:
    """Test configuration loading and modification workflow."""

    def test_load_modify_save_config_workflow(self, tmp_path):
        """Test complete config workflow: load → modify → save → reload."""
        # Step 1: Create initial config
        initial_config = {
            "model": {"hidden_dim": 768},
            "training": {"lr": 1e-3}
        }

        config_path = tmp_path / "model_config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(initial_config, f)

        # Step 2: Load config
        config = Config.from_files(model_config_path=str(config_path))
        assert config.get("model.hidden_dim") == 768

        # Step 3: Modify config
        config.set("model.hidden_dim", 1536)
        config.set("model.num_layers", 24)
        assert config.get("model.hidden_dim") == 1536

        # Step 4: Save modified config
        output_dir = tmp_path / "modified"
        config.save(output_dir=str(output_dir))

        # Step 5: Reload and verify
        reloaded = Config.from_files(
            model_config_path=str(output_dir / "model_config.yaml")
        )
        assert reloaded.get("model.hidden_dim") == 1536
        assert reloaded.get("model.num_layers") == 24

    def test_config_merge_workflow(self, tmp_path):
        """Test merging multiple config files."""
        # Create separate configs
        model_config = {"transformer": {"hidden_dim": 1536}}
        training_config = {"optimizer": {"lr": 3e-4}}

        model_path = tmp_path / "model.yaml"
        training_path = tmp_path / "training.yaml"

        with open(model_path, 'w') as f:
            yaml.dump(model_config, f)
        with open(training_path, 'w') as f:
            yaml.dump(training_config, f)

        # Load both
        config = Config.from_files(
            model_config_path=str(model_path),
            training_config_path=str(training_path)
        )

        # Both should be accessible
        assert config.get("model.transformer.hidden_dim") == 1536
        assert config.get("training.optimizer.lr") == 3e-4


@pytest.mark.skip(reason="Model implementation not complete")
class TestTrainingWorkflow:
    """Test complete training workflow."""

    def test_training_checkpoint_resume_workflow(self, tmp_path):
        """Test training → checkpoint → resume workflow."""
        # This would test:
        # 1. Initialize model and trainer
        # 2. Train for N steps
        # 3. Save checkpoint
        # 4. Create new model and trainer
        # 5. Load checkpoint
        # 6. Continue training
        # 7. Verify continuity
        pass

    def test_training_validation_workflow(self):
        """Test training with periodic validation."""
        # This would test:
        # 1. Train for K steps
        # 2. Run validation
        # 3. Log metrics
        # 4. Continue training
        pass

    def test_hyperparameter_tuning_workflow(self):
        """Test hyperparameter search workflow."""
        # This would test trying different hyperparameters
        pass


@pytest.mark.skip(reason="Model implementation not complete")
class TestInferenceWorkflow:
    """Test inference workflow."""

    def test_load_model_and_infer_workflow(self):
        """Test loading model and running inference."""
        # This would test:
        # 1. Load checkpoint
        # 2. Set model to eval mode
        # 3. Run inference on test data
        # 4. Verify output format
        pass

    def test_batch_inference_workflow(self):
        """Test batch inference on multiple inputs."""
        pass

    def test_streaming_inference_workflow(self):
        """Test streaming inference for real-time use."""
        pass


class TestEncoderDecoderIntegration:
    """Test encoder-decoder integration."""

    def test_encode_decode_cycle(self):
        """Test encoding inputs and decoding outputs."""
        from src.encoders.internal_voice_encoder import InternalVoiceEncoder
        from src.decoders.text_decoder import TextDecoder

        embedding_dim = 512
        batch_size = 2
        seq_len = 10

        # Encode
        encoder = InternalVoiceEncoder(vocab_size=1000, embedding_dim=embedding_dim)
        tokens = torch.randint(0, 1000, (batch_size, seq_len))
        encoded = encoder(tokens)

        # Decode
        decoder = TextDecoder(vocab_size=1000, embedding_dim=embedding_dim)
        decoded = decoder(encoded["embeddings"])

        assert decoded["tokens"].shape == (batch_size, seq_len)

    def test_multi_encoder_fusion(self):
        """Test combining multiple encoder outputs."""
        from src.encoders.internal_voice_encoder import InternalVoiceEncoder
        from src.encoders.audio_encoder import AudioEncoder

        embedding_dim = 512
        batch_size = 2

        # Encode from multiple sources
        voice_encoder = InternalVoiceEncoder(vocab_size=1000, embedding_dim=embedding_dim)
        audio_encoder = AudioEncoder(embedding_dim=embedding_dim)

        voice_tokens = torch.randint(0, 1000, (batch_size, 10))
        audio_waveform = torch.randn(batch_size, 16000)

        voice_encoded = voice_encoder(voice_tokens)
        audio_encoded = audio_encoder(audio_waveform)

        # Concatenate (simple fusion)
        fused = torch.cat([
            voice_encoded["embeddings"],
            audio_encoded["embeddings"]
        ], dim=1)

        # Should have combined sequence length
        assert fused.shape[0] == batch_size
        assert fused.shape[2] == embedding_dim


class TestDataFlowIntegration:
    """Test data flow through system."""

    def test_batch_creation_and_processing(self):
        """Test creating and processing a batch."""
        # Create dummy batch
        batch = {
            "inputs": {
                "internal_voice_tokens": torch.randint(0, 1000, (4, 10)),
            },
            "targets": {
                "internal_text": torch.randint(0, 1000, (4, 10)),
            }
        }

        # Move to device
        device = "cpu"
        batch["inputs"] = {
            k: v.to(device) for k, v in batch["inputs"].items()
        }
        batch["targets"] = {
            k: v.to(device) for k, v in batch["targets"].items()
        }

        # Verify
        assert batch["inputs"]["internal_voice_tokens"].device.type == device

    def test_attention_mask_propagation(self):
        """Test that attention masks flow correctly through pipeline."""
        from src.encoders.internal_voice_encoder import InternalVoiceEncoder

        encoder = InternalVoiceEncoder(vocab_size=1000, embedding_dim=512)

        # Create input with padding
        tokens = torch.randint(0, 1000, (2, 10))
        tokens[:, 7:] = 0  # Simulate padding

        output = encoder(tokens)

        # Attention mask should be present
        assert "attention_mask" in output
        assert output["attention_mask"].shape == tokens.shape


class TestLossComputation:
    """Test loss computation workflows."""

    def test_single_decoder_loss(self):
        """Test computing loss for single decoder."""
        from src.decoders.text_decoder import TextDecoder

        decoder = TextDecoder(vocab_size=1000, embedding_dim=512)
        hidden_states = torch.randn(2, 10, 512)
        target_tokens = torch.randint(0, 1000, (2, 10))

        loss = decoder.compute_loss(hidden_states, target_tokens)

        assert loss.dim() == 0  # Scalar
        assert loss > 0

    def test_multi_decoder_loss_combination(self):
        """Test combining losses from multiple decoders."""
        from src.decoders.text_decoder import TextDecoder
        from src.decoders.audio_decoder import AudioDecoder

        hidden_states = torch.randn(2, 10, 512)

        # Text decoder loss
        text_decoder = TextDecoder(vocab_size=1000, embedding_dim=512)
        text_targets = torch.randint(0, 1000, (2, 10))
        text_loss = text_decoder.compute_loss(hidden_states, text_targets)

        # Audio decoder loss
        audio_decoder = AudioDecoder(codebook_size=1024, embedding_dim=512)
        audio_targets = torch.randint(0, 1024, (2, 10))
        audio_loss = audio_decoder.compute_loss(hidden_states, audio_targets)

        # Combined loss
        loss_weights = {"text": 1.0, "audio": 0.5}
        total_loss = (
            loss_weights["text"] * text_loss +
            loss_weights["audio"] * audio_loss
        )

        assert total_loss > 0
        assert total_loss.requires_grad

    def test_animation_loss_components(self):
        """Test animation decoder's multi-component loss."""
        from src.decoders.animation_decoder import AnimationDecoder

        decoder = AnimationDecoder(embedding_dim=512)
        hidden_states = torch.randn(2, 10, 512)

        # Generate targets
        target_rotations = torch.randn(2, 10, 24, 4)
        target_rotations = decoder.normalize_quaternion(target_rotations)
        target_blend_shapes = torch.rand(2, 10, 51)
        target_eye_params = torch.rand(2, 10, 8)

        loss, loss_dict = decoder.compute_loss(
            hidden_states,
            target_rotations,
            target_blend_shapes,
            target_eye_params
        )

        # Should have multiple loss components
        assert "body_loss" in loss_dict
        assert "face_loss" in loss_dict
        assert "eye_loss" in loss_dict
        assert "total_animation_loss" in loss_dict


class TestGradientFlow:
    """Test gradient flow through components."""

    def test_encoder_gradient_flow(self):
        """Test that gradients flow through encoder."""
        from src.encoders.internal_voice_encoder import InternalVoiceEncoder

        encoder = InternalVoiceEncoder(vocab_size=1000, embedding_dim=512)
        tokens = torch.randint(0, 1000, (2, 10))

        output = encoder(tokens)
        embeddings = output["embeddings"]

        # Compute dummy loss
        loss = embeddings.mean()
        loss.backward()

        # Check that encoder has gradients
        for param in encoder.parameters():
            if param.requires_grad:
                assert param.grad is not None

    def test_decoder_gradient_flow(self):
        """Test that gradients flow through decoder."""
        from src.decoders.text_decoder import TextDecoder

        decoder = TextDecoder(vocab_size=1000, embedding_dim=512)
        hidden_states = torch.randn(2, 10, 512, requires_grad=True)
        target_tokens = torch.randint(0, 1000, (2, 10))

        loss = decoder.compute_loss(hidden_states, target_tokens)
        loss.backward()

        # Check gradients
        assert hidden_states.grad is not None
        for param in decoder.parameters():
            if param.requires_grad:
                assert param.grad is not None

    def test_end_to_end_gradient_flow(self):
        """Test gradient flow through encoder-decoder chain."""
        from src.encoders.internal_voice_encoder import InternalVoiceEncoder
        from src.decoders.text_decoder import TextDecoder

        encoder = InternalVoiceEncoder(vocab_size=1000, embedding_dim=512)
        decoder = TextDecoder(vocab_size=1000, embedding_dim=512)

        # Forward pass
        tokens = torch.randint(0, 1000, (2, 10))
        encoded = encoder(tokens)
        loss = decoder.compute_loss(encoded["embeddings"], tokens)

        # Backward pass
        loss.backward()

        # Check gradients in both encoder and decoder
        for param in encoder.parameters():
            if param.requires_grad:
                assert param.grad is not None

        for param in decoder.parameters():
            if param.requires_grad:
                assert param.grad is not None


class TestMemoryEfficiency:
    """Test memory efficiency of operations."""

    def test_batch_size_scaling(self):
        """Test that memory scales reasonably with batch size."""
        from src.encoders.internal_voice_encoder import InternalVoiceEncoder

        encoder = InternalVoiceEncoder(vocab_size=1000, embedding_dim=512)

        # Small batch
        small_tokens = torch.randint(0, 1000, (2, 10))
        small_output = encoder(small_tokens)

        # Large batch
        large_tokens = torch.randint(0, 1000, (16, 10))
        large_output = encoder(large_tokens)

        # Should complete without OOM
        assert small_output["embeddings"].shape[0] == 2
        assert large_output["embeddings"].shape[0] == 16

    def test_sequence_length_scaling(self):
        """Test that memory scales with sequence length."""
        from src.encoders.internal_voice_encoder import InternalVoiceEncoder

        encoder = InternalVoiceEncoder(vocab_size=1000, embedding_dim=512)

        # Short sequence
        short_tokens = torch.randint(0, 1000, (2, 10))
        short_output = encoder(short_tokens)

        # Long sequence
        long_tokens = torch.randint(0, 1000, (2, 100))
        long_output = encoder(long_tokens)

        # Should complete
        assert short_output["embeddings"].shape[1] == 10
        assert long_output["embeddings"].shape[1] == 100


class TestErrorHandling:
    """Test error handling in workflows."""

    def test_invalid_input_shapes(self):
        """Test handling of invalid input shapes."""
        from src.encoders.visual_encoder import VisualEncoder

        encoder = VisualEncoder(
            image_size=224,
            patch_size=16,
            embedding_dim=512,
            use_stereo=False
        )

        # Wrong shape (missing batch dimension)
        with pytest.raises((RuntimeError, ValueError)):
            wrong_shape = torch.randn(3, 224, 224)
            encoder(wrong_shape)

    def test_out_of_vocab_tokens(self):
        """Test handling of out-of-vocabulary tokens."""
        from src.encoders.internal_voice_encoder import InternalVoiceEncoder

        encoder = InternalVoiceEncoder(vocab_size=1000, embedding_dim=512)

        # Tokens out of range
        with pytest.raises((IndexError, RuntimeError)):
            invalid_tokens = torch.tensor([[1000, 1001, 1002]])  # >= vocab_size
            encoder(invalid_tokens)

    def test_mismatched_dimensions(self):
        """Test handling of dimension mismatches."""
        from src.decoders.text_decoder import TextDecoder

        decoder = TextDecoder(vocab_size=1000, embedding_dim=512)

        # Wrong embedding dimension
        with pytest.raises((RuntimeError, ValueError)):
            wrong_dim = torch.randn(2, 10, 256)  # Should be 512
            decoder(wrong_dim)


@pytest.mark.skip(reason="Model implementation not complete")
class TestModelCheckpointing:
    """Test model checkpointing workflows."""

    def test_save_and_load_checkpoint(self, tmp_path):
        """Test saving and loading model checkpoint."""
        # Would test:
        # 1. Create and train model
        # 2. Save checkpoint
        # 3. Create new model
        # 4. Load checkpoint
        # 5. Verify parameters match
        pass

    def test_checkpoint_includes_metadata(self, tmp_path):
        """Test that checkpoint includes necessary metadata."""
        # Should include:
        # - Model state dict
        # - Optimizer state dict
        # - Training step
        # - Configuration
        # - Timestamp
        pass

    def test_backwards_compatibility(self, tmp_path):
        """Test loading checkpoints from older versions."""
        pass


class TestReproducibility:
    """Test reproducibility of results."""

    def test_same_seed_same_results(self):
        """Test that same seed produces same results."""
        from src.encoders.internal_voice_encoder import InternalVoiceEncoder

        seed = 42
        embedding_dim = 512
        tokens = torch.randint(0, 1000, (2, 10))

        # Run 1
        torch.manual_seed(seed)
        encoder1 = InternalVoiceEncoder(vocab_size=1000, embedding_dim=embedding_dim)
        output1 = encoder1(tokens)

        # Run 2
        torch.manual_seed(seed)
        encoder2 = InternalVoiceEncoder(vocab_size=1000, embedding_dim=embedding_dim)
        output2 = encoder2(tokens)

        # Should be identical
        assert torch.allclose(output1["embeddings"], output2["embeddings"], atol=1e-6)

    def test_deterministic_forward_pass(self):
        """Test that forward pass is deterministic in eval mode."""
        from src.encoders.internal_voice_encoder import InternalVoiceEncoder

        encoder = InternalVoiceEncoder(vocab_size=1000, embedding_dim=512)
        encoder.eval()

        tokens = torch.randint(0, 1000, (2, 10))

        # Multiple forward passes
        with torch.no_grad():
            output1 = encoder(tokens)
            output2 = encoder(tokens)

        # Should be identical
        assert torch.equal(output1["embeddings"], output2["embeddings"])
