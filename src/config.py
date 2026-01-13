"""
Configuration file for the poker solver.
Centralizes all magic numbers and configurable parameters.
"""
import json
import os
from typing import Any, Dict


class SolverConfig:
    """Manages solver configuration parameters."""
    
    # Default configuration values
    DEFAULTS: Dict[str, Any] = {
        # Game parameters
        "default_board": ["Ks", "Th", "7s", "4d", "2s"],
        "num_nodes": 4,
        "ante": 1.0,
        
        # Training parameters
        "default_iterations": 50000,
        "min_iterations": 1,
        "max_iterations": 1_000_000,
        "warmup_iterations": 1000,
        
        # Pot sizes by node and action
        "pot_sizes": {
            "0_1": 4.0,      # Root -> Bet, Call: 2.0 ante + 1.0 bet + 1.0 call
            "2_0": 2.0,      # Checked To -> Check: 2.0 ante
            "3_1": 6.0,      # Check Raise -> Call: 2.0 ante + 1.0 bet + 2.0 raise + 1.0 call
        },
        
        # API parameters
        "cors_origins": ["*"],
        "api_host": "0.0.0.0",
        "api_port": 8000,
        
        # Logging
        "log_level": "INFO",
        "log_dir": "logs",
    }
    
    def __init__(self, config_file: str = "config.json") -> None:
        """
        Initialize configuration from file or defaults.
        
        Args:
            config_file: Path to JSON config file (optional)
        """
        self.config = self.DEFAULTS.copy()
        
        # Load from JSON file if it exists
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    user_config = json.load(f)
                    self.config.update(user_config)
                    print(f"Configuration loaded from {config_file}")
            except Exception as e:
                print(f"Warning: Could not load config file {config_file}: {e}")
                print("Using default configuration")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value by key.
        
        Args:
            key: Configuration key (supports dot notation: "pot_sizes.0_1")
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        if '.' in key:
            keys = key.split('.')
            value = self.config
            for k in keys:
                if isinstance(value, dict):
                    value = value.get(k)
                else:
                    return default
            return value if value is not None else default
        return self.config.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """
        Set a configuration value.
        
        Args:
            key: Configuration key
            value: New value
        """
        self.config[key] = value
    
    def save(self, config_file: str = "config.json") -> None:
        """
        Save configuration to JSON file.
        
        Args:
            config_file: Path to save config
        """
        try:
            with open(config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
                print(f"Configuration saved to {config_file}")
        except Exception as e:
            print(f"Warning: Could not save config file: {e}")


# Global configuration instance
_config: SolverConfig | None = None


def get_config(config_file: str = "config.json") -> SolverConfig:
    """
    Get or create the global configuration instance.
    
    Args:
        config_file: Path to JSON config file
        
    Returns:
        SolverConfig instance
    """
    global _config
    if _config is None:
        _config = SolverConfig(config_file)
    return _config
