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
        """
        Purpose:
            Test that Config initializes with empty dicts.
            
        Workflow:
            1. Initialize Config().
            2. Verify all attributes are empty dicts.
            
        ToDo:
            - None
        """
        config = Config()
        assert config.model == {}
        assert config.training == {}
        assert config.inference == {}
        assert config.evolution == {}

    def test_manual_initialization(self):
        """
        Purpose:
            Test Config initialization with provided values.
            
        Workflow:
            1. Initialize Config with valid dicts.
            2. Verify attributes match inputs.
            
        ToDo:
            - None
        """
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
        """
        Purpose:
            Test loading all configuration files.
            
        Workflow:
            1. Create dummy YAML files for all sections.
            2. Calls Config.from_files with all paths.
            3. Verify attributes match file content.
            
        ToDo:
            - None
        """
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
        """
        Purpose:
            Test loading when only some config files exist.
            
        Workflow:
            1. Create one YAML file.
            2. Call Config.from_files with mixed valid/invalid paths.
            3. Verify valid file loads, invalid ones result in empty dicts (or don't crash).
            
        ToDo:
            - None
        """
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
        """
        Purpose:
            Test loading when no config files exist.
            
        Workflow:
            1. Call Config.from_files with all invalid paths.
            2. Verify all attributes are empty.
            
        ToDo:
            - None
        """
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
        """
        Purpose:
            Test loading empty YAML files.
            
        Workflow:
            1. Create an empty YAML file.
            2. Load it.
            3. Verify it results in None or empty dict.
            
        ToDo:
            - None
        """
        model_path = tmp_path / "model_config.yaml"
        with open(model_path, 'w') as f:
            f.write("")  # Empty file

        config = Config.from_files(model_config_path=str(model_path))
        # Empty YAML loads as None, which becomes None not {}
        assert config.model is None or config.model == {}

    def test_load_malformed_yaml(self, tmp_path):
        """
        Purpose:
            Test handling of malformed YAML files.
            
        Workflow:
            1. Create a file with invalid YAML.
            2. Assert that loading it raises YAMLError.
            
        ToDo:
            - None
        """
        model_path = tmp_path / "model_config.yaml"
        with open(model_path, 'w') as f:
            f.write("invalid: yaml: content: [[[")

        with pytest.raises(yaml.YAMLError):
            Config.from_files(model_config_path=str(model_path))

    def test_load_nested_structure(self, tmp_path):
        """
        Purpose:
            Test loading complex nested configuration.
            
        Workflow:
            1. Create YAML with deep nesting.
            2. Load it.
            3. Verify nested values are accessible.
            
        ToDo:
            - None
        """
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
        """
        Purpose:
            Test getting top-level configuration.
            
        Workflow:
            1. Create Config with flat value.
            2. Get it.
            3. Verify value.
            
        ToDo:
            - None
        """
        config = Config(model={"hidden_dim": 1536})
        assert config.get("model") == {"hidden_dim": 1536}

    def test_get_nested_value(self):
        """
        Purpose:
            Test getting nested configuration value.
            
        Workflow:
            1. Create Config with nested dict.
            2. Get via dot notation 'a.b'.
            3. Verify value.
            
        ToDo:
            - None
        """
        config = Config(model={
            "transformer": {
                "num_layers": 24,
                "hidden_dim": 1536
            }
        })

        assert config.get("model.transformer.num_layers") == 24
        assert config.get("model.transformer.hidden_dim") == 1536

    def test_get_deeply_nested_value(self):
        """
        Purpose:
            Test getting deeply nested configuration value.
            
        Workflow:
            1. Create deeply nested dict.
            2. Get via 'a.b.c.d'.
            3. Verify value.
            
        ToDo:
            - None
        """
        adam_config = {
            "betas": {
                "beta1": 0.9,
                "beta2": 0.95
            }
        }
        optimizer_config = {"adam": adam_config}
        config = Config(training={"optimizer": optimizer_config})

        assert config.get("training.optimizer.adam.betas.beta1") == 0.9
        assert config.get("training.optimizer.adam.betas.beta2") == 0.95

    def test_get_nonexistent_key_returns_default(self):
        """
        Purpose:
            Test that getting nonexistent key returns default value.
            
        Workflow:
            1. Create Config.
            2. Get nonexistent key.
            3. Verify returns None or specified default.
            
        ToDo:
            - None
        """
        config = Config(model={"hidden_dim": 1536})

        assert config.get("model.nonexistent") is None
        assert config.get("model.nonexistent", default=42) == 42
        assert config.get("nonexistent.deeply.nested", default="default") == "default"

    def test_get_with_none_value(self):
        """
        Purpose:
            Test getting a key that has None as value.
            
        Workflow:
            1. Set a key to None.
            2. Get it with a default provided.
            3. Verify it returns None (explicit value) not default.
            
        ToDo:
            - None
        """
        config = Config(model={"option": None})

        # Should return None (the actual value), not the default
        assert config.get("model.option") is None
        assert config.get("model.option", default="default") is None

    def test_get_with_various_types(self):
        """
        Purpose:
            Test getting values of different types.
            
        Workflow:
            1. Create Config with int, float, list, bool.
            2. Get each.
            3. Verify types are preserved.
            
        ToDo:
            - None
        """
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
        """
        Purpose:
            Test get with empty string key.
            
        Workflow:
            1. Call config.get("").
            2. Verify returns None.
            
        ToDo:
            - None
        """
        config = Config(model={"hidden_dim": 1536})
        # Empty string splits to [''], which should not match anything
        assert config.get("") is None


class TestConfigSet:
    """Tests for Config.set() method with dot notation."""

    def test_set_top_level_new_key(self):
        """
        Purpose:
            Test setting a new top-level key.
            
        Workflow:
            1. Call config.set("key", val).
            2. Verify config.key == val.
            
        ToDo:
            - None
        """
        config = Config()
        config.set("custom_field", {"value": 42})

        assert config.custom_field == {"value": 42}

    def test_set_nested_value_existing_path(self):
        """
        Purpose:
            Test setting nested value in existing structure.
            
        Workflow:
            1. Create nested config.
            2. Set existing path 'a.b' to new value.
            3. Verify update.
            
        ToDo:
            - None
        """
        config = Config(model={"transformer": {"num_layers": 24}})
        config.set("model.transformer.num_layers", 48)

        assert config.model["transformer"]["num_layers"] == 48

    def test_set_nested_value_new_path(self):
        """
        Purpose:
            Test setting nested value creating new structure.
            
        Workflow:
            1. Create Config.
            2. Set 'a.b' where a doesn't exist.
            3. Verify 'a' is created as dict and 'b' is set.
            
        ToDo:
            - None
        """
        config = Config(model={})
        config.set("model.transformer.hidden_dim", 1536)

        assert config.model["transformer"]["hidden_dim"] == 1536

    def test_set_deeply_nested_new_path(self):
        """
        Purpose:
            Test setting deeply nested value creating new structure.
            
        Workflow:
            1. Set 'a.b.c.d' on empty config.
            2. Verify structure is created.
            
        ToDo:
            - None
        """
        config = Config()
        config.set("training.optimizer.adam.betas.beta1", 0.9)

        assert config.training["optimizer"]["adam"]["betas"]["beta1"] == 0.9

    def test_set_overwrite_existing_value(self):
        """
        Purpose:
            Test overwriting an existing value.
            
        Workflow:
            1. Set 'a' to 1.
            2. Set 'a' to 2.
            3. Verify 'a' is 2.
            
        ToDo:
            - None
        """
        config = Config(model={"hidden_dim": 768})
        config.set("model.hidden_dim", 1536)

        assert config.model["hidden_dim"] == 1536

    def test_set_various_types(self):
        """
        Purpose:
            Test setting values of different types.
            
        Workflow:
            1. Set various types (int, list, dict).
            2. Verify they are stored correctly.
            
        ToDo:
            - None
        """
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
        """
        Purpose:
            Test setting None as a value.
            
        Workflow:
            1. Set key to None.
            2. Verify key exists and is None.
            
        ToDo:
            - None
        """
        config = Config(model={})
        config.set("model.option", None)

        assert config.model["option"] is None


class TestConfigSave:
    """Tests for Config.save() method."""

    def test_save_all_configs(self, tmp_path):
        """
        Purpose:
             Test saving all configuration files.
             
        Workflow:
            1. Create Config with all sections.
            2. Save to tmp_path.
            3. Verify all 4 YAML files exist and contain correct data.
            
        ToDo:
            - None
        """
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
        """
        Purpose:
            Test saving when only some configs are populated.
            
        Workflow:
            1. Config with only model config.
            2. Save.
            3. Verify model_config.yaml exists, others do not.
            
        ToDo:
            - None
        """
        config = Config(model={"hidden_dim": 768}, training={})
        config.save(output_dir=str(tmp_path))

        assert (tmp_path / "model_config.yaml").exists()
        # Empty dict might not create file or creates file with "{}" or "null"
        # The implementation saves if self.training evaluates to truthy
        # Empty dict is falsy in Python, so training_config.yaml should not be created
        assert not (tmp_path / "training_config.yaml").exists()

    def test_save_creates_directory(self, tmp_path):
        """
        Purpose:
            Test that save creates output directory if it doesn't exist.
            
        Workflow:
            1. Define non-existent path.
            2. Save.
            3. Verify path created and file inside.
            
        ToDo:
            - None
        """
        output_dir = tmp_path / "new_configs"
        assert not output_dir.exists()

        config = Config(model={"test": "value"})
        config.save(output_dir=str(output_dir))

        assert output_dir.exists()
        assert (output_dir / "model_config.yaml").exists()

    def test_save_overwrites_existing_files(self, tmp_path):
        """
        Purpose:
             Test that save overwrites existing configuration files.
             
        Workflow:
            1. Save version 1.
            2. Save version 2.
            3. Verify content matches version 2.
            
        ToDo:
            - None
        """
        config = Config(model={"version": 1})
        config.save(output_dir=str(tmp_path))

        # Modify and save again
        config.model["version"] = 2
        config.save(output_dir=str(tmp_path))

        with open(tmp_path / "model_config.yaml", 'r') as f:
            loaded = yaml.safe_load(f)
        assert loaded["version"] == 2

    def test_save_roundtrip(self, tmp_path):
        """
        Purpose:
            Test that saved configs can be loaded back correctly.
            
        Workflow:
            1. Create complex config.
            2. Save it.
            3. Load it back using from_files.
            4. Verify equality.
            
        ToDo:
            - None
        """
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
        """
        Purpose:
            Test that get and set work together correctly.
            
        Workflow:
            1. Set value.
            2. Get value.
            3. Modify value.
            4. Get new value.
            
        ToDo:
            - None
        """
        config = Config()

        # Set a value
        config.set("model.transformer.num_layers", 24)

        # Get it back
        assert config.get("model.transformer.num_layers") == 24

        # Modify it
        config.set("model.transformer.num_layers", 48)
        assert config.get("model.transformer.num_layers") == 48

    def test_load_modify_save(self, tmp_path):
        """
        Purpose:
            Test loading, modifying, and saving configuration.
            
        Workflow:
            1. Create YAML.
            2. Load Config.
            3. Modify via set().
            4. Save.
            5. Load again from new location.
            6. Verify modifications.
            
        ToDo:
            - None
        """
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
        """
        Purpose:
            Test multiple get/set operations in sequence.
            
        Workflow:
            1. Set multiple unrelated values.
            2. Get all of them.
            3. Modify a subset.
            4. Verify correct state.
            
        ToDo:
            - None
        """
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
