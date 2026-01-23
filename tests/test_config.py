"""
Tests for configuration module (src/config.py).
"""

import pytest
import yaml
import tempfile
from pathlib import Path
from src.config import Config


class TestConfigInitialization:
    """Tests for Config initialization."""

    def test_default_initialization(self):
        """Test that Config initializes with empty dicts."""
        config = Config()
        assert config.model == {}
        assert config.training == {}
        assert config.inference == {}
        assert config.evolution == {}

    def test_manual_initialization(self):
        """Test Config initialization with provided values."""
        model_cfg = {"transformer": {"num_layers": 12}}
        training_cfg = {"optimizer": {"lr": 0.001}}

        config = Config(model=model_cfg, training=training_cfg)
        assert config.model == model_cfg
        assert config.training == training_cfg
        assert config.inference == {}
        assert config.evolution == {}


class TestConfigFromFiles:
    """Tests for loading configuration from YAML files."""

    def test_load_all_configs(self, tmp_path):
        """Test loading all configuration files."""
        # Create temporary config files
        model_config = {"transformer": {"hidden_dim": 1536, "num_layers": 24}}
        training_config = {"optimizer": {"lr": 3e-4, "weight_decay": 0.01}}
        inference_config = {"batch_size": 32, "max_length": 512}
        evolution_config = {"population_size": 50, "generations": 100}

        model_path = tmp_path / "model_config.yaml"
        training_path = tmp_path / "training_config.yaml"
        inference_path = tmp_path / "inference_config.yaml"
        evolution_path = tmp_path / "evolution_config.yaml"

        with open(model_path, 'w') as f:
            yaml.dump(model_config, f)
        with open(training_path, 'w') as f:
            yaml.dump(training_config, f)
        with open(inference_path, 'w') as f:
            yaml.dump(inference_config, f)
        with open(evolution_path, 'w') as f:
            yaml.dump(evolution_config, f)

        # Load configs
        config = Config.from_files(
            model_config_path=str(model_path),
            training_config_path=str(training_path),
            inference_config_path=str(inference_path),
            evolution_config_path=str(evolution_path)
        )

        assert config.model == model_config
        assert config.training == training_config
        assert config.inference == inference_config
        assert config.evolution == evolution_config

    def test_load_partial_configs(self, tmp_path):
        """Test loading when only some config files exist."""
        model_config = {"transformer": {"hidden_dim": 768}}
        model_path = tmp_path / "model_config.yaml"

        with open(model_path, 'w') as f:
            yaml.dump(model_config, f)

        # Load with only model config existing
        config = Config.from_files(
            model_config_path=str(model_path),
            training_config_path=str(tmp_path / "nonexistent_training.yaml"),
            inference_config_path=str(tmp_path / "nonexistent_inference.yaml"),
            evolution_config_path=str(tmp_path / "nonexistent_evolution.yaml")
        )

        assert config.model == model_config
        assert config.training == {}
        assert config.inference == {}
        assert config.evolution == {}

    def test_load_no_configs(self, tmp_path):
        """Test loading when no config files exist."""
        config = Config.from_files(
            model_config_path=str(tmp_path / "nonexistent1.yaml"),
            training_config_path=str(tmp_path / "nonexistent2.yaml"),
            inference_config_path=str(tmp_path / "nonexistent3.yaml"),
            evolution_config_path=str(tmp_path / "nonexistent4.yaml")
        )

        assert config.model == {}
        assert config.training == {}
        assert config.inference == {}
        assert config.evolution == {}

    def test_load_empty_yaml(self, tmp_path):
        """Test loading empty YAML files."""
        model_path = tmp_path / "model_config.yaml"
        with open(model_path, 'w') as f:
            f.write("")  # Empty file

        config = Config.from_files(model_config_path=str(model_path))
        # Empty YAML loads as None, which becomes None not {}
        assert config.model is None or config.model == {}

    def test_load_malformed_yaml(self, tmp_path):
        """Test handling of malformed YAML files."""
        model_path = tmp_path / "model_config.yaml"
        with open(model_path, 'w') as f:
            f.write("invalid: yaml: content: [[[")

        with pytest.raises(yaml.YAMLError):
            Config.from_files(model_config_path=str(model_path))

    def test_load_nested_structure(self, tmp_path):
        """Test loading complex nested configuration."""
        model_config = {
            "transformer": {
                "hidden_dim": 1536,
                "num_layers": 24,
                "attention": {
                    "num_heads": 12,
                    "dropout": 0.1,
                    "head_dim": 128
                }
            },
            "encoders": {
                "visual": {
                    "image_size": 224,
                    "patch_size": 16
                }
            }
        }

        model_path = tmp_path / "model_config.yaml"
        with open(model_path, 'w') as f:
            yaml.dump(model_config, f)

        config = Config.from_files(model_config_path=str(model_path))
        assert config.model == model_config
        assert config.model["transformer"]["attention"]["num_heads"] == 12


class TestConfigGet:
    """Tests for Config.get() method with dot notation."""

    def test_get_top_level(self):
        """Test getting top-level configuration."""
        config = Config(model={"hidden_dim": 1536})
        assert config.get("model") == {"hidden_dim": 1536}

    def test_get_nested_value(self):
        """Test getting nested configuration value."""
        config = Config(model={
            "transformer": {
                "num_layers": 24,
                "hidden_dim": 1536
            }
        })

        assert config.get("model.transformer.num_layers") == 24
        assert config.get("model.transformer.hidden_dim") == 1536

    def test_get_deeply_nested_value(self):
        """Test getting deeply nested configuration value."""
        config = Config(training={
            "optimizer": {
                "adam": {
                    "betas": {
                        "beta1": 0.9,
                        "beta2": 0.95
                    }
                }
            }
        })

        assert config.get("training.optimizer.adam.betas.beta1") == 0.9
        assert config.get("training.optimizer.adam.betas.beta2") == 0.95

    def test_get_nonexistent_key_returns_default(self):
        """Test that getting nonexistent key returns default value."""
        config = Config(model={"hidden_dim": 1536})

        assert config.get("model.nonexistent") is None
        assert config.get("model.nonexistent", default=42) == 42
        assert config.get("nonexistent.deeply.nested", default="default") == "default"

    def test_get_with_none_value(self):
        """Test getting a key that has None as value."""
        config = Config(model={"option": None})

        # Should return None (the actual value), not the default
        assert config.get("model.option") is None
        assert config.get("model.option", default="default") is None

    def test_get_with_various_types(self):
        """Test getting values of different types."""
        config = Config(model={
            "string": "value",
            "integer": 42,
            "float": 3.14,
            "boolean": True,
            "list": [1, 2, 3],
            "dict": {"nested": "value"}
        })

        assert config.get("model.string") == "value"
        assert config.get("model.integer") == 42
        assert config.get("model.float") == 3.14
        assert config.get("model.boolean") is True
        assert config.get("model.list") == [1, 2, 3]
        assert config.get("model.dict") == {"nested": "value"}

    def test_get_empty_string_key(self):
        """Test get with empty string key."""
        config = Config(model={"hidden_dim": 1536})
        # Empty string splits to [''], which should not match anything
        assert config.get("") is None


class TestConfigSet:
    """Tests for Config.set() method with dot notation."""

    def test_set_top_level_new_key(self):
        """Test setting a new top-level key."""
        config = Config()
        config.set("custom_field", {"value": 42})

        assert config.custom_field == {"value": 42}

    def test_set_nested_value_existing_path(self):
        """Test setting nested value in existing structure."""
        config = Config(model={"transformer": {"num_layers": 24}})
        config.set("model.transformer.num_layers", 48)

        assert config.model["transformer"]["num_layers"] == 48

    def test_set_nested_value_new_path(self):
        """Test setting nested value creating new structure."""
        config = Config(model={})
        config.set("model.transformer.hidden_dim", 1536)

        assert config.model["transformer"]["hidden_dim"] == 1536

    def test_set_deeply_nested_new_path(self):
        """Test setting deeply nested value creating new structure."""
        config = Config()
        config.set("training.optimizer.adam.betas.beta1", 0.9)

        assert config.training["optimizer"]["adam"]["betas"]["beta1"] == 0.9

    def test_set_overwrite_existing_value(self):
        """Test overwriting an existing value."""
        config = Config(model={"hidden_dim": 768})
        config.set("model.hidden_dim", 1536)

        assert config.model["hidden_dim"] == 1536

    def test_set_various_types(self):
        """Test setting values of different types."""
        config = Config()

        config.set("model.string", "value")
        config.set("model.integer", 42)
        config.set("model.float", 3.14)
        config.set("model.boolean", False)
        config.set("model.list", [1, 2, 3])
        config.set("model.dict", {"nested": "value"})

        assert config.model["string"] == "value"
        assert config.model["integer"] == 42
        assert config.model["float"] == 3.14
        assert config.model["boolean"] is False
        assert config.model["list"] == [1, 2, 3]
        assert config.model["dict"] == {"nested": "value"}

    def test_set_none_value(self):
        """Test setting None as a value."""
        config = Config(model={})
        config.set("model.option", None)

        assert config.model["option"] is None


class TestConfigSave:
    """Tests for Config.save() method."""

    def test_save_all_configs(self, tmp_path):
        """Test saving all configuration files."""
        config = Config(
            model={"transformer": {"hidden_dim": 1536}},
            training={"optimizer": {"lr": 3e-4}},
            inference={"batch_size": 32},
            evolution={"population_size": 50}
        )

        config.save(output_dir=str(tmp_path))

        # Check that files were created
        assert (tmp_path / "model_config.yaml").exists()
        assert (tmp_path / "training_config.yaml").exists()
        assert (tmp_path / "inference_config.yaml").exists()
        assert (tmp_path / "evolution_config.yaml").exists()

        # Verify contents
        with open(tmp_path / "model_config.yaml", 'r') as f:
            loaded_model = yaml.safe_load(f)
        assert loaded_model == config.model

    def test_save_partial_configs(self, tmp_path):
        """Test saving when only some configs are populated."""
        config = Config(model={"hidden_dim": 768}, training={})
        config.save(output_dir=str(tmp_path))

        assert (tmp_path / "model_config.yaml").exists()
        # Empty dict might not create file or creates file with "{}" or "null"
        # The implementation saves if self.training evaluates to truthy
        # Empty dict is falsy in Python, so training_config.yaml should not be created
        assert not (tmp_path / "training_config.yaml").exists()

    def test_save_creates_directory(self, tmp_path):
        """Test that save creates output directory if it doesn't exist."""
        output_dir = tmp_path / "new_configs"
        assert not output_dir.exists()

        config = Config(model={"test": "value"})
        config.save(output_dir=str(output_dir))

        assert output_dir.exists()
        assert (output_dir / "model_config.yaml").exists()

    def test_save_overwrites_existing_files(self, tmp_path):
        """Test that save overwrites existing configuration files."""
        config = Config(model={"version": 1})
        config.save(output_dir=str(tmp_path))

        # Modify and save again
        config.model["version"] = 2
        config.save(output_dir=str(tmp_path))

        with open(tmp_path / "model_config.yaml", 'r') as f:
            loaded = yaml.safe_load(f)
        assert loaded["version"] == 2

    def test_save_roundtrip(self, tmp_path):
        """Test that saved configs can be loaded back correctly."""
        original_config = Config(
            model={
                "transformer": {
                    "hidden_dim": 1536,
                    "num_layers": 24,
                    "attention": {"num_heads": 12}
                }
            },
            training={"optimizer": {"lr": 3e-4, "betas": [0.9, 0.95]}},
            inference={"batch_size": 32, "temperature": 0.7}
        )

        # Save
        original_config.save(output_dir=str(tmp_path))

        # Load back
        loaded_config = Config.from_files(
            model_config_path=str(tmp_path / "model_config.yaml"),
            training_config_path=str(tmp_path / "training_config.yaml"),
            inference_config_path=str(tmp_path / "inference_config.yaml"),
            evolution_config_path=str(tmp_path / "evolution_config.yaml")
        )

        assert loaded_config.model == original_config.model
        assert loaded_config.training == original_config.training
        assert loaded_config.inference == original_config.inference
        # evolution was not set, so it should be empty/None
        assert loaded_config.evolution == {} or loaded_config.evolution is None


class TestConfigIntegration:
    """Integration tests for Config class."""

    def test_get_set_integration(self):
        """Test that get and set work together correctly."""
        config = Config()

        # Set a value
        config.set("model.transformer.num_layers", 24)

        # Get it back
        assert config.get("model.transformer.num_layers") == 24

        # Modify it
        config.set("model.transformer.num_layers", 48)
        assert config.get("model.transformer.num_layers") == 48

    def test_load_modify_save(self, tmp_path):
        """Test loading, modifying, and saving configuration."""
        # Create initial config
        initial_config = {"transformer": {"hidden_dim": 768}}
        model_path = tmp_path / "model_config.yaml"
        with open(model_path, 'w') as f:
            yaml.dump(initial_config, f)

        # Load
        config = Config.from_files(model_config_path=str(model_path))

        # Modify
        config.set("model.transformer.hidden_dim", 1536)
        config.set("model.transformer.num_layers", 24)

        # Save
        output_dir = tmp_path / "modified"
        config.save(output_dir=str(output_dir))

        # Load back and verify
        modified_config = Config.from_files(
            model_config_path=str(output_dir / "model_config.yaml")
        )
        assert modified_config.get("model.transformer.hidden_dim") == 1536
        assert modified_config.get("model.transformer.num_layers") == 24

    def test_multiple_operations(self):
        """Test multiple get/set operations in sequence."""
        config = Config()

        # Set multiple values
        config.set("model.hidden_dim", 1536)
        config.set("model.num_layers", 24)
        config.set("training.lr", 3e-4)
        config.set("training.batch_size", 32)

        # Get them back
        assert config.get("model.hidden_dim") == 1536
        assert config.get("model.num_layers") == 24
        assert config.get("training.lr") == 3e-4
        assert config.get("training.batch_size") == 32

        # Modify some
        config.set("model.num_layers", 48)
        config.set("training.batch_size", 64)

        # Verify changes
        assert config.get("model.num_layers") == 48
        assert config.get("training.batch_size") == 64
        # Others unchanged
        assert config.get("model.hidden_dim") == 1536
        assert config.get("training.lr") == 3e-4
