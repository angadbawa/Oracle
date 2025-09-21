"""Configuration management for Oracle project."""

import yaml
from pathlib import Path
from typing import Any, Dict, Optional
import logging

class Config:
    """Configuration manager for Oracle application."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration.
        
        Args:
            config_path: Path to configuration file
        """
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / "config" / "default.yaml"
        
        self.config_path = Path(config_path)
        self._config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except FileNotFoundError:
            logging.warning(f"Config file not found: {self.config_path}")
            return self._get_default_config()
        except yaml.YAMLError as e:
            logging.error(f"Error parsing config file: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration if file loading fails."""
        return {
            "models": {
                "sam": {
                    "model_type": "vit_h",
                    "checkpoint_path": "./weights/sam_vit_h_4b8939.pth",
                    "device": "cpu"
                },
                "stable_diffusion": {
                    "model_name": "stabilityai/stable-diffusion-2-inpainting",
                    "device": "cpu"
                }
            },
            "image": {
                "default_size": [512, 512],
                "supported_formats": ["jpg", "jpeg", "png", "bmp", "tiff"],
                "max_file_size_mb": 10
            },
            "ui": {
                "gradio": {
                    "title": "Oracle: SAM + Stable Diffusion Inpainting",
                    "server_port": 7860,
                    "server_name": "127.0.0.1"
                }
            }
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Args:
            key: Configuration key (e.g., 'models.sam.device')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self._config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value using dot notation.
        
        Args:
            key: Configuration key (e.g., 'models.sam.device')
            value: Value to set
        """
        keys = key.split('.')
        config = self._config
        
        # Navigate to parent of target key
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        # Set the value
        config[keys[-1]] = value
    
    def save(self, path: Optional[str] = None) -> None:
        """
        Save configuration to file.
        
        Args:
            path: Path to save configuration (uses original path if None)
        """
        save_path = Path(path) if path else self.config_path
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'w') as f:
            yaml.dump(self._config, f, default_flow_style=False, indent=2)
    
    def update(self, updates: Dict[str, Any]) -> None:
        """
        Update configuration with dictionary of values.
        
        Args:
            updates: Dictionary of configuration updates
        """
        def deep_update(base_dict, update_dict):
            for key, value in update_dict.items():
                if isinstance(value, dict) and key in base_dict and isinstance(base_dict[key], dict):
                    deep_update(base_dict[key], value)
                else:
                    base_dict[key] = value
        
        deep_update(self._config, updates)
    
    @property
    def sam_config(self) -> Dict[str, Any]:
        """Get SAM model configuration."""
        return self.get('models.sam', {})
    
    @property
    def diffusion_config(self) -> Dict[str, Any]:
        """Get Stable Diffusion configuration."""
        return self.get('models.stable_diffusion', {})
    
    @property
    def ui_config(self) -> Dict[str, Any]:
        """Get UI configuration."""
        return self.get('ui', {})
    
    @property
    def image_config(self) -> Dict[str, Any]:
        """Get image processing configuration."""
        return self.get('image', {})

# Global configuration instance
_config = None

def get_config(config_path: Optional[str] = None) -> Config:
    """Get global configuration instance."""
    global _config
    if _config is None:
        _config = Config(config_path)
    return _config

def reload_config(config_path: Optional[str] = None) -> Config:
    """Reload configuration from file."""
    global _config
    _config = Config(config_path)
    return _config
