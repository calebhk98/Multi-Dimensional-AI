"""
Configuration module for loading and managing config files.
"""

import yaml
from pathlib import Path
from typing import Dict, Any
from dataclasses import dataclass, field


@dataclass
class Config:
    """
    Central configuration class that loads all config files.
    
    Attributes:
        model: Model architecture configuration
        training: Training configuration
        inference: Inference configuration
        evolution: Evolutionary training configuration
    """
    
    model: Dict[str, Any] = field(default_factory=dict)
    training: Dict[str, Any] = field(default_factory=dict)
    inference: Dict[str, Any] = field(default_factory=dict)
    evolution: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_files(
        cls,
        model_config_path: str = "configs/model_config.yaml",
        training_config_path: str = "configs/training_config.yaml",
        inference_config_path: str = "configs/inference_config.yaml",
        evolution_config_path: str = "configs/evolution_config.yaml",
    ) -> "Config":
        """
        Load configuration from YAML files.
        
        Args:
            model_config_path: Path to model configuration file
            training_config_path: Path to training configuration file
            inference_config_path: Path to inference configuration file
            evolution_config_path: Path to evolution configuration file
            
        Returns:
            Config object with loaded configurations
        """
        config = cls()
        
        if Path(model_config_path).exists():
            with open(model_config_path, 'r') as f:
                config.model = yaml.safe_load(f)
        
        if Path(training_config_path).exists():
            with open(training_config_path, 'r') as f:
                config.training = yaml.safe_load(f)
        
        if Path(inference_config_path).exists():
            with open(inference_config_path, 'r') as f:
                config.inference = yaml.safe_load(f)
        
        if Path(evolution_config_path).exists():
            with open(evolution_config_path, 'r') as f:
                config.evolution = yaml.safe_load(f)
        
        return config
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by dot-separated key.
        
        Examples:
            config.get('model.transformer.num_layers')
            config.get('training.optimizer.lr')
        
        Args:
            key: Dot-separated configuration key
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        parts = key.split('.')
        current = self.__dict__
        
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return default
        
        return current
    
    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value by dot-separated key.
        
        Args:
            key: Dot-separated configuration key
            value: Value to set
        """
        parts = key.split('.')
        current = self.__dict__
        
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        
        current[parts[-1]] = value
    
    def save(self, output_dir: str = "configs") -> None:
        """
        Save current configuration to YAML files.
        
        Args:
            output_dir: Directory to save configuration files
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if self.model:
            with open(output_path / "model_config.yaml", 'w') as f:
                yaml.dump(self.model, f, default_flow_style=False)
        
        if self.training:
            with open(output_path / "training_config.yaml", 'w') as f:
                yaml.dump(self.training, f, default_flow_style=False)
        
        if self.inference:
            with open(output_path / "inference_config.yaml", 'w') as f:
                yaml.dump(self.inference, f, default_flow_style=False)
        
        if self.evolution:
            with open(output_path / "evolution_config.yaml", 'w') as f:
                yaml.dump(self.evolution, f, default_flow_style=False)
