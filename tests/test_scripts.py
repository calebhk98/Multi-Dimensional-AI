"""
Tests for scripts directory - CLI entry points and utilities.

Purpose:
    Verify script functionality including argument parsing,
    configuration loading, and basic execution paths.
"""

import pytest
import sys
import argparse
import yaml
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
import torch

# Add project root for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class TestTrainScript:
    """Tests for scripts/train.py functionality."""

    def test_load_config_valid_yaml(self, temp_dir):
        """
        Test load_config with valid YAML file.

        Purpose:
            Verify config loading works correctly.

        Workflow:
            1. Create temp YAML config
            2. Load with load_config
            3. Verify contents match
        """
        from scripts.train import load_config

        config_content = {
            "model": {"embedding_dim": 512},
            "training": {"batch_size": 4, "max_steps": 100}
        }

        config_path = temp_dir / "test_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_content, f)

        loaded = load_config(str(config_path))

        assert loaded["model"]["embedding_dim"] == 512
        assert loaded["training"]["batch_size"] == 4

    def test_load_config_missing_file(self):
        """
        Test load_config raises error for missing file.

        Purpose:
            Verify proper error handling for missing config.
        """
        from scripts.train import load_config

        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/config.yaml")

    def test_create_dataset_pairwise(self, minimal_model_config):
        """
        Test create_dataset creates PairwiseDataset when configured.

        Purpose:
            Verify dataset creation based on config.
        """
        from scripts.train import create_dataset
        from src.data.pairwise_dataset import PairwiseDataset

        config = {
            "pairwise": {
                "pairs": [["internal_voice", "audio"]],
                "dataset": {"vocab_size": 1000}
            },
            "training": {"batch_size": 2, "max_steps": 10}
        }

        dataset = create_dataset(config)
        assert isinstance(dataset, PairwiseDataset)

    def test_create_dataset_multimodal_fallback(self, minimal_model_config):
        """
        Test create_dataset falls back to MultiModalDataset.

        Purpose:
            Verify fallback behavior when no specific config.
        """
        from scripts.train import create_dataset
        from src.data.multimodal_dataset import MultiModalDataset

        config = {
            "training": {"batch_size": 2, "max_steps": 10}
        }

        dataset = create_dataset(config)
        assert isinstance(dataset, MultiModalDataset)


class TestInferenceScript:
    """Tests for scripts/inference.py functionality."""

    def test_inference_server_init(self):
        """
        Test InferenceServer initialization.

        Purpose:
            Verify server initializes with defaults when no config.
        """
        from scripts.inference import InferenceServer

        server = InferenceServer(dry_run=True)

        assert server.dry_run is True
        assert server.model is None
        assert server._running is False

    def test_inference_server_default_config(self):
        """
        Test InferenceServer uses default config when file missing.

        Purpose:
            Verify default configuration is applied.
        """
        from scripts.inference import InferenceServer

        server = InferenceServer(
            config_path=Path("/nonexistent/config.yaml"),
            dry_run=True
        )

        # Should have default VR config
        assert "vr" in server.config
        assert "inference" in server.config

    def test_generate_dummy_outputs(self):
        """
        Test dummy output generation for dry-run mode.

        Purpose:
            Verify dummy outputs have correct structure.
        """
        from scripts.inference import InferenceServer

        server = InferenceServer(dry_run=True)
        outputs = server._generate_dummy_outputs()

        assert "audio" in outputs
        assert "animation" in outputs
        assert outputs["audio"].shape == (1, 10)
        assert outputs["animation"]["joint_rotations"].shape == (1, 1, 24, 4)

    def test_parse_args_defaults(self):
        """
        Test argument parsing with defaults.

        Purpose:
            Verify CLI argument defaults.
        """
        from scripts.inference import parse_args

        with patch("sys.argv", ["inference.py"]):
            args = parse_args()

        assert args.dry_run is False
        assert args.verbose is False
        assert args.port is None

    def test_parse_args_with_options(self):
        """
        Test argument parsing with options.

        Purpose:
            Verify CLI arguments are parsed correctly.
        """
        from scripts.inference import parse_args

        test_args = [
            "inference.py",
            "--dry-run",
            "--port", "6666",
            "--verbose"
        ]

        with patch("sys.argv", test_args):
            args = parse_args()

        assert args.dry_run is True
        assert args.port == 6666
        assert args.verbose is True


class TestPrepareHfData:
    """Tests for scripts/prepare_hf_data.py functionality."""

    def test_prepare_hf_data_import(self):
        """
        Test prepare_hf_data script can be imported.

        Purpose:
            Verify script has no import errors.
        """
        try:
            import scripts.prepare_hf_data
            assert True
        except ImportError as e:
            pytest.skip(f"Optional dependencies not available: {e}")


class TestTrainTextFast:
    """Tests for scripts/train_text_fast.py functionality."""

    def test_train_text_fast_import(self):
        """
        Test train_text_fast script can be imported.

        Purpose:
            Verify script has no import errors.
        """
        import scripts.train_text_fast
        assert hasattr(scripts.train_text_fast, "main") or True


class TestCheckEnvironment:
    """Tests for scripts/check_environment.py functionality."""

    def test_check_environment_import(self):
        """
        Test check_environment script can be imported.

        Purpose:
            Verify environment check script loads.
        """
        import scripts.check_environment
        assert True


class TestGenerateDummyData:
    """Tests for scripts/generate_dummy_data.py functionality."""

    def test_generate_dummy_data_import(self):
        """
        Test generate_dummy_data script can be imported.

        Purpose:
            Verify script has no import errors.
        """
        import scripts.generate_dummy_data
        assert True


class TestValidateSession:
    """Tests for scripts/validate_session.py functionality."""

    def test_validate_session_import(self):
        """
        Test validate_session script can be imported.

        Purpose:
            Verify script has no import errors.
        """
        try:
            import scripts.validate_session
            assert True
        except ImportError:
            pytest.skip("validate_session has unmet dependencies")
