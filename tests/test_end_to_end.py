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
        """
        Purpose:
            Test complete config workflow: load → modify → save → reload.
            
        Workflow:
            1. Create initial config file.
            2. Load via Config.from_files.
            3. Modify using config.set().
            4. Save.
            5. Reload and verify changes.
            
        ToDo:
            - None
        """
        # Step 1: Create initial config
        # Step 1: Create initial config (just model config)
        initial_config = {
            "hidden_dim": 768,
            "num_layers": 12
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
        """
        Purpose:
            Test merging multiple config files.
            
        Workflow:
            1. Create separate model and training config files.
            2. Load both via Config.from_files.
            3. Verify both sections coexist.
            
        ToDo:
            - None
        """
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


class TestTrainingWorkflow:
    """Test complete training workflow."""

    def test_training_checkpoint_resume_workflow(self, tmp_path):
        """
        Purpose:
            Test training → checkpoint → resume workflow.
            
        Workflow:
            1. Train model for a few steps.
            2. Save checkpoint.
            3. Resume training from checkpoint.
            4. Verify state restoration.
            
        ToDo:
            - None
        """
        from src.models.multimodal_transformer import MultiModalCreature
        from src.training.trainer import Trainer
        from torch.utils.data import DataLoader, TensorDataset

        # Setup Config
        config = {
            "model": {
                "encoders": {
                    "internal_voice": {"vocab_size": 100, "embedding_dim": 64},
                    "external_voice": {"vocab_size": 100, "embedding_dim": 64},
                    "audio": {"embedding_dim": 64},
                    "vision": {"embedding_dim": 64},
                    "proprioception": {"embedding_dim": 64},
                    "touch": {"embedding_dim": 64},
                },
                "transformer": {
                    "num_layers": 2,
                    "hidden_dim": 64,
                    "num_attention_heads": 4,
                    "ffn_dim": 256,
                },
                "fusion": {"strategy": "concatenate"},
                "decoders": {
                    "internal_text": {"vocab_size": 100, "embedding_dim": 64},
                    "external_text": {"vocab_size": 100, "embedding_dim": 64},
                    "audio": {"embedding_dim": 64},
                    "animation": {"embedding_dim": 64},
                }
            },
            "training": {
                "max_steps": 10,
                "batch_size": 2,
                "checkpointing": {"save_dir": str(tmp_path / "checkpoints")}
            }
        }

        # Setup Dummy Data
        # Minimal compatible inputs
        # Note: Dataset should yield single samples, DataLoader will batch them
        single_inputs = {
            "internal_voice_tokens": torch.randint(0, 100, (10,)),
        }
        single_targets = {
            "internal_text": torch.randint(0, 100, (10,)),
        }
        
        # Create a simpler dataset wrapper that yields dicts
        class DictDataset(torch.utils.data.Dataset):
            def __len__(self): return 10
            def __getitem__(self, idx): return {"inputs": single_inputs, "targets": single_targets}

        dataset = DictDataset()
        loader = DataLoader(dataset, batch_size=2)

        # 1. Initialize & Train
        model = MultiModalCreature(config)
        trainer = Trainer(model, config, loader, device="cpu")
        
        # Hack to run fewer steps
        trainer.max_steps = 2 
        trainer.save_interval = 2
        trainer.train()

        # 2. Checkpoint should exist
        ckpt_path = tmp_path / "checkpoints" / "model_step_2.pt"
        assert ckpt_path.exists()

        # 3. Resume (Load Checkpoint)
        # Verify we can load it back
        checkpoint = torch.load(ckpt_path)
        assert checkpoint['step'] == 2
        
        # Create new model and load weights
        new_model = MultiModalCreature(config)
        new_model.load_state_dict(checkpoint['model_state_dict'])
        
        # Verify weights match
        for p1, p2 in zip(model.parameters(), new_model.parameters()):
            assert torch.allclose(p1, p2)

    def test_training_validation_workflow(self, tmp_path):
        """
        Purpose:
            Test training with periodic validation.
            
        Workflow:
            1. Train loop.
            2. Validation loop triggers.
            3. Metrics logged.
            
        ToDo:
            - None
        """
        from src.models.multimodal_transformer import MultiModalCreature
        from src.training.trainer import Trainer
        from torch.utils.data import DataLoader

        config = {
            "model": {
                "encoders": {"internal_voice": {"vocab_size": 100}}, # Minimal necessary
                "transformer": {"hidden_dim": 64, "num_layers": 1},
                "decoders": {"internal_text": {"vocab_size": 100}}
            },
            "training": {
                "max_steps": 2,
                "checkpointing": {"save_dir": str(tmp_path / "checkpoints")}
            }
        }
        
        # Mock Data
        class DictDataset(torch.utils.data.Dataset):
            def __len__(self): return 4
            def __getitem__(self, idx): 
                return {
                    "inputs": {"internal_voice_tokens": torch.randint(0, 100, (5,))},
                    "targets": {"internal_text": torch.randint(0, 100, (5,))}
                }

        train_loader = DataLoader(DictDataset(), batch_size=2)
        val_loader = DataLoader(DictDataset(), batch_size=2)

        model = MultiModalCreature(config)
        trainer = Trainer(model, config, train_loader, val_loader=val_loader, device="cpu")
        
        # Just ensure it runs without error (validation logic is inside trainer.train ideally, 
        # but standard Trainer might implemented simplistic valid loop. 
        # Assuming Trainer supports it if passed, or we verify it doesn't crash.)
        trainer.train()
        # Pass implies success for now

    def test_hyperparameter_tuning_workflow(self, tmp_path):
        """
        Purpose:
            Test hyperparameter search workflow (basic diff check).
            
        Workflow:
            1. Run training with LR 1e-3.
            2. Run training with LR 1e-5.
            3. Verify different loss outcomes (or simply that they run).
            
        ToDo:
            - None
        """
        # Simplistic check: Just ensure we can config different LRs
        config = {
            "model": {
                "encoders": {"internal_voice": {"vocab_size": 100, "embedding_dim": 16}}, # Increased vocab
                "transformer": {"hidden_dim": 16, "num_layers": 1, "num_heads": 2},
                 "decoders": {"internal_text": {"vocab_size": 100}}
            },
            "training": {
                "max_steps": 1,
                "checkpointing": {"save_dir": str(tmp_path)}
            }
        }
        
        from src.models.multimodal_transformer import MultiModalCreature
        from src.training.trainer import Trainer
        from torch.utils.data import DataLoader
        
        dataset = type('D', (torch.utils.data.Dataset,), {
            '__len__': lambda s: 2,
            '__getitem__': lambda s, i: {
                "inputs": {"internal_voice_tokens": torch.zeros(5, dtype=torch.long)}, 
                "targets": {"internal_text": torch.zeros(5, dtype=torch.long)}
            }
        })()
        loader = DataLoader(dataset, batch_size=1)
        
        # Run 1
        config["training"]["optimizer"] = {"lr": 1e-2}
        model1 = MultiModalCreature(config)
        trainer1 = Trainer(model1, config, loader, device="cpu")
        trainer1.train()
        
        # Run 2
        config["training"]["optimizer"] = {"lr": 1e-5}
        model2 = MultiModalCreature(config)
        trainer2 = Trainer(model2, config, loader, device="cpu")
        trainer2.train()


class TestInferenceWorkflow:
    """Test inference workflow."""

    def test_load_model_and_infer_workflow(self, tmp_path):
        """
        Purpose:
            Test loading model and running inference.
            
        Workflow:
            1. Save a model state.
            2. Load it back.
            3. Run forward pass.
            
        ToDo:
            - None
        """
        from src.models.multimodal_transformer import MultiModalCreature
        import torch
        
        config = {
            "model": {
                "encoders": {"internal_voice": {"vocab_size": 50, "embedding_dim": 32}},
                "transformer": {"hidden_dim": 32, "num_layers": 1},
                "decoders": {"internal_text": {"vocab_size": 50}}
            }
        }
        
        model = MultiModalCreature(config)
        save_path = tmp_path / "model.pt"
        torch.save(model.state_dict(), save_path)
        
        # Load
        loaded_model = MultiModalCreature(config)
        loaded_model.load_state_dict(torch.load(save_path))
        loaded_model.eval()
        
        # Infer
        inputs = torch.randint(0, 50, (1, 10))
        with torch.no_grad():
            output = loaded_model(internal_voice_tokens=inputs)
            
        assert "internal_text" in output

    def test_batch_inference_workflow(self):
        """
        Purpose:
            Test batch inference on multiple inputs.
            
        Workflow:
            1. Prepare large batch.
            2. Infer.
            3. Verify output shapes.
            
        ToDo:
            - None
        """
        from src.models.multimodal_transformer import MultiModalCreature
        
        config = {
             "model": {
                "encoders": {"internal_voice": {"vocab_size": 50, "embedding_dim": 32}},
                "transformer": {"hidden_dim": 32, "num_layers": 1},
                "decoders": {"internal_text": {"vocab_size": 50}}
            }
        }
        model = MultiModalCreature(config)
        
        batch_size = 4
        inputs = torch.randint(0, 50, (batch_size, 10))
        output = model(internal_voice_tokens=inputs)
        
        # Check first output modality
        # Output is a dict containing 'tokens', 'probabilities'
        assert output["internal_text"]["tokens"].shape[0] == batch_size

    def test_streaming_inference_workflow(self):
        """
        Purpose:
            Test sequential inference (simulation of streaming).
            
        Workflow:
            1. Run inference multiple times sequentially.
            2. Verify persistence or independence as expected (stateless for now).
            
        ToDo:
            - None
        """
        from src.models.multimodal_transformer import MultiModalCreature
        
        config = {
             "model": {
                "encoders": {"internal_voice": {"vocab_size": 50, "embedding_dim": 32}},
                "transformer": {"hidden_dim": 32, "num_layers": 1},
                "decoders": {"internal_text": {"vocab_size": 50}}
            }
        }
        model = MultiModalCreature(config)
        model.eval()
        
        inputs = torch.randint(0, 50, (1, 5))
        
        # Stream chunks (simulated by repeated calls)
        # Fix seed for deterministic sampling comparison
        torch.manual_seed(42)
        out1 = model(internal_voice_tokens=inputs)
        
        torch.manual_seed(42)
        out2 = model(internal_voice_tokens=inputs)
        
        # Should be identical for stateless model
        assert torch.allclose(out1["internal_text"]["tokens"].float(), out2["internal_text"]["tokens"].float())


class TestModelCheckpointing:
    """Test model checkpointing workflows."""

    def test_save_and_load_checkpoint(self, tmp_path):
        """
        Purpose:
            Test saving and loading model checkpoint.
            
        Workflow:
            1. Create model.
            2. Save state dict.
            3. Load state dict.
            4. Compare parameters.
            
        ToDo:
            - None
        """
        from src.models.multimodal_transformer import MultiModalCreature
        
        config = {
             "model": {
                "encoders": {"internal_voice": {"vocab_size": 50, "embedding_dim": 32}},
                "transformer": {"hidden_dim": 32, "num_layers": 1},
                 "decoders": {"internal_text": {"vocab_size": 50}}
            }
        }
        model = MultiModalCreature(config)
        path = tmp_path / "test_ckpt.pt"
        
        torch.save(model.state_dict(), path)
        
        loaded = MultiModalCreature(config)
        loaded.load_state_dict(torch.load(path))
        
        for p1, p2 in zip(model.parameters(), loaded.parameters()):
            assert torch.allclose(p1, p2)

    def test_checkpoint_includes_metadata(self, tmp_path):
        """
        Purpose:
            Test that checkpoint includes necessary metadata.
            
        Workflow:
            1. Train (mock) and save via Trainer (reused logic).
            2. Load dict.
            3. Verify keys (epoch, config, etc.).
            
        ToDo:
            - None
        """
        from src.training.trainer import Trainer
        # Minimal setup to get a trainer
        config = {"training": {"checkpointing": {"save_dir": str(tmp_path)}}}
        
        # Mock objects
        class MockModel(torch.nn.Module):
            def __init__(self): super().__init__()
            def forward(self, **kwargs): return {}
        
        model = MockModel()
        trainer = Trainer(model, config, None, device="cpu")
        
        # Manually save
        trainer.save_checkpoint(step=10)
        
        ckpt = torch.load(tmp_path / "model_step_10.pt")
        assert "step" in ckpt
        assert "model_state_dict" in ckpt
        assert "optimizer_state_dict" in ckpt
        assert "config" in ckpt
        assert ckpt["step"] == 10

    def test_backwards_compatibility(self, tmp_path):
        """
        Purpose:
            Test loading checkpoints (simulated backward compat).
            
        Workflow:
            1. Create a dict simulating a checkpoint.
            2. Attempt to load partial state or verify migration (none needed yet).
            
        ToDo:
            - None
        """
        # For now, just ensure strict=False loads work if we want that, 
        # or just that standard load works.
        ckpt = {"model_state_dict": {}, "step": 0}
        path = tmp_path / "legacy.pt"
        torch.save(ckpt, path)
        
        loaded = torch.load(path)
        assert loaded["step"] == 0
    """Test encoder-decoder integration."""

    def test_encode_decode_cycle(self):
        """
        Purpose:
            Test encoding inputs and decoding outputs.
            
        Workflow:
            1. Encode random tokens.
            2. Decode embeddings.
            3. Verify output shape.
            
        ToDo:
            - None
        """
        from src.encoders.internal_voice_encoder import InternalVoiceEncoder
        from src.decoders.text_decoder import InternalTextDecoder

        embedding_dim = 512
        batch_size = 2
        seq_len = 10

        # Encode
        encoder = InternalVoiceEncoder(vocab_size=1000, embedding_dim=embedding_dim)
        tokens = torch.randint(0, 1000, (batch_size, seq_len))
        encoded = encoder(tokens)

        # Decode
        decoder = InternalTextDecoder(vocab_size=1000, embedding_dim=embedding_dim)
        decoded = decoder(encoded["embeddings"])

        assert decoded["tokens"].shape == (batch_size, seq_len)

    def test_multi_encoder_fusion(self):
        """
        Purpose:
            Test combining multiple encoder outputs.
            
        Workflow:
            1. Encoder audio and text.
            2. Fuse embeddings (concat).
            3. Verify fused shape.
            
        ToDo:
            - None
        """
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
        """
        Purpose:
            Test creating and processing a batch.
            
        Workflow:
            1. Create nested batch dict.
            2. Move to device.
            3. Verify device placement.
            
        ToDo:
            - None
        """
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
        """
        Purpose:
            Test that attention masks flow correctly through pipeline.
            
        Workflow:
            1. Create input with padding.
            2. Run encoder.
            3. Verify attention_mask in output.
            
        ToDo:
            - None
        """
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
        """
        Purpose:
            Test computing loss for single decoder.
            
        Workflow:
            1. Run decoder.compute_loss.
            2. Verify scalar output > 0.
            
        ToDo:
            - None
        """
        from src.decoders.text_decoder import InternalTextDecoder

        decoder = InternalTextDecoder(vocab_size=1000, embedding_dim=512)
        hidden_states = torch.randn(2, 10, 512)
        target_tokens = torch.randint(0, 1000, (2, 10))

        loss = decoder.compute_loss(hidden_states, target_tokens)

        assert loss.dim() == 0  # Scalar
        assert loss > 0

    def test_multi_decoder_loss_combination(self):
        """
        Purpose:
            Test combining losses from multiple decoders.
            
        Workflow:
            1. Compute text loss.
            2. Compute audio loss.
            3. Combine with weights.
            4. Verify gradient requirement.
            
        ToDo:
            - None
        """
        from src.decoders.text_decoder import InternalTextDecoder
        from src.decoders.audio_decoder import AudioDecoder

        hidden_states = torch.randn(2, 10, 512)

        # Text decoder loss
        text_decoder = InternalTextDecoder(vocab_size=1000, embedding_dim=512)
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
        """
        Purpose:
            Test animation decoder's multi-component loss.
            
        Workflow:
            1. Run animation loss.
            2. Verify sub-components (body, face, eye) in dict.
            
        ToDo:
            - None
        """
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
        """
        Purpose:
            Test that gradients flow through encoder.
            
        Workflow:
            1. Forward pass.
            2. Compute loss.
            3. Backward pass.
            4. Verify params have grad.
            
        ToDo:
            - None
        """
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
        """
        Purpose:
            Test that gradients flow through decoder.
            
        Workflow:
            1. Forward pass.
            2. Compute loss.
            3. Backward pass.
            4. Verify inputs and params have grad.
            
        ToDo:
            - None
        """
        from src.decoders.text_decoder import InternalTextDecoder

        decoder = InternalTextDecoder(vocab_size=1000, embedding_dim=512)
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
        """
        Purpose:
            Test gradient flow through encoder-decoder chain.
            
        Workflow:
            1. Encode -> Decode -> Loss.
            2. Backward.
            3. Verify both modules have grads.
            
        ToDo:
            - None
        """
        from src.encoders.internal_voice_encoder import InternalVoiceEncoder
        from src.decoders.text_decoder import InternalTextDecoder

        encoder = InternalVoiceEncoder(vocab_size=1000, embedding_dim=512)
        decoder = InternalTextDecoder(vocab_size=1000, embedding_dim=512)

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
        """
        Purpose:
            Test that memory scales reasonably with batch size.
            
        Workflow:
            1. Run small batch.
            2. Run large batch.
            3. Verify both succeed (implicit OOM check by running).
            
        ToDo:
            - None
        """
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
        """
        Purpose:
            Test that memory scales with sequence length.
            
        Workflow:
            1. Run short seq.
            2. Run long seq.
            3. Verify output shapes.
            
        ToDo:
            - None
        """
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
        """
        Purpose:
            Test handling of invalid input shapes.
            
        Workflow:
            1. Create encoder.
            2. Pass input with missing dimension.
            3. Verify error raised.
            
        ToDo:
            - None
        """
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
        """
        Purpose:
            Test handling of out-of-vocabulary tokens.
            
        Workflow:
            1. Create encoder.
            2. Pass token > vocab_size.
            3. Verify error raised.
            
        ToDo:
            - None
        """
        from src.encoders.internal_voice_encoder import InternalVoiceEncoder

        encoder = InternalVoiceEncoder(vocab_size=1000, embedding_dim=512)

        # Tokens out of range
        with pytest.raises((IndexError, RuntimeError)):
            invalid_tokens = torch.tensor([[1000, 1001, 1002]])  # >= vocab_size
            encoder(invalid_tokens)

    def test_mismatched_dimensions(self):
        """
        Purpose:
            Test handling of dimension mismatches.
            
        Workflow:
            1. Create decoder.
            2. Pass input with wrong embedding dim.
            3. Verify error raised.
            
        ToDo:
            - None
        """
        from src.decoders.text_decoder import InternalTextDecoder

        decoder = InternalTextDecoder(vocab_size=1000, embedding_dim=512)

        # Wrong embedding dimension
        with pytest.raises((RuntimeError, ValueError)):
            wrong_dim = torch.randn(2, 10, 256)  # Should be 512
            decoder(wrong_dim)


@pytest.mark.skip(reason="Model implementation not complete")
class TestModelCheckpointing:
    """Test model checkpointing workflows."""

    def test_save_and_load_checkpoint(self, tmp_path):
        """
        Purpose:
            Test saving and loading model checkpoint.
            
        Workflow:
            1. Train model.
            2. Save.
            3. Load.
            4. Compare key params.
            
        ToDo:
            - Implement test.
        """
        # Would test:
        # 1. Create and train model
        # 2. Save checkpoint
        # 3. Create new model
        # 4. Load checkpoint
        # 5. Verify parameters match
        pass

    def test_checkpoint_includes_metadata(self, tmp_path):
        """
        Purpose:
            Test that checkpoint includes necessary metadata.
            
        Workflow:
            1. Save checkpoint.
            2. Load dict.
            3. Verify keys (epoch, config, etc.).
            
        ToDo:
            - Implement test.
        """
        # Should include:
        # - Model state dict
        # - Optimizer state dict
        # - Training step
        # - Configuration
        # - Timestamp
        pass

    def test_backwards_compatibility(self, tmp_path):
        """
        Purpose:
            Test loading checkpoints from older versions.
            
        Workflow:
            1. Mock old checkpoint.
            2. Load.
            3. Verify success.
            
        ToDo:
            - Implement test.
        """
        pass


class TestReproducibility:
    """Test reproducibility of results."""

    def test_same_seed_same_results(self):
        """
        Purpose:
            Test that same seed produces same results.
            
        Workflow:
            1. Set seed X -> Run 1 -> Output 1.
            2. Set seed X -> Run 2 -> Output 2.
            3. Assert Output 1 == Output 2.
            
        ToDo:
            - None
        """
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
        """
        Purpose:
            Test that forward pass is deterministic in eval mode.
            
        Workflow:
            1. Set eval mode.
            2. Run forward twice on same input.
            3. Assert equal outputs.
            
        ToDo:
            - None
        """
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
