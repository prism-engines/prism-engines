"""
PRISM-CLIMATE Config - Configuration Management

This submodule will contain configuration management
for the climate module.

STATUS: SKELETON - Placeholder configuration only.

Planned configuration:
    - Data source API keys/credentials
    - Default parameters
    - Pipeline configurations
    - Output settings
"""

from typing import Dict, Any, Optional
from pathlib import Path

__all__ = [
    "ClimateConfig",
    "DEFAULT_CONFIG",
    "get_config",
]

# Default configuration (placeholder)
DEFAULT_CONFIG = {
    # Module settings
    "module": {
        "version": "0.1.0",
        "status": "skeleton",
        "active": False,
    },

    # Data source settings (placeholder - no real credentials)
    "sources": {
        "noaa": {
            "enabled": False,
            "api_key": None,
            "base_url": "https://www.ncdc.noaa.gov/cdo-web/api/v2/",
        },
        "nasa_giss": {
            "enabled": False,
            "base_url": "https://data.giss.nasa.gov/gistemp/",
        },
        "era5": {
            "enabled": False,
            "api_key": None,
            "api_url": "https://cds.climate.copernicus.eu/api/v2",
        },
    },

    # Indicator settings
    "indicators": {
        "default_baseline_start": "1951-01-01",
        "default_baseline_end": "1980-12-31",
        "output_decimals": 4,
    },

    # Transform settings
    "transforms": {
        "default_frequency": "M",
        "default_aggregation": "mean",
        "interpolation_method": "linear",
    },

    # Output settings
    "output": {
        "format": "parquet",
        "compression": "snappy",
        "output_dir": None,
    },

    # Cache settings
    "cache": {
        "enabled": False,
        "cache_dir": None,
        "ttl_days": 7,
    },
}


class ClimateConfig:
    """
    Configuration manager for climate module.

    STATUS: SKELETON - Basic config structure only.
    """

    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        """
        Initialize configuration.

        Args:
            config_dict: Configuration dictionary (uses defaults if None)
        """
        self._config = DEFAULT_CONFIG.copy()
        if config_dict:
            self._deep_update(self._config, config_dict)

    def _deep_update(self, base: dict, updates: dict) -> None:
        """Recursively update nested dictionaries."""
        for key, value in updates.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_update(base[key], value)
            else:
                base[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by dot-notation key.

        Args:
            key: Configuration key (e.g., 'sources.noaa.enabled')
            default: Default value if key not found

        Returns:
            Configuration value
        """
        keys = key.split(".")
        value = self._config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value

    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value by dot-notation key.

        Args:
            key: Configuration key
            value: Value to set
        """
        keys = key.split(".")
        config = self._config
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        config[keys[-1]] = value

    def to_dict(self) -> Dict[str, Any]:
        """Return configuration as dictionary."""
        return self._config.copy()

    def is_active(self) -> bool:
        """Check if module is configured as active."""
        return self.get("module.active", False)


# Global configuration instance
_global_config: Optional[ClimateConfig] = None


def get_config() -> ClimateConfig:
    """
    Get the global configuration instance.

    Returns:
        ClimateConfig instance
    """
    global _global_config
    if _global_config is None:
        _global_config = ClimateConfig()
    return _global_config


def load_config_from_file(path: Path) -> ClimateConfig:
    """
    Load configuration from a YAML file.

    STATUS: NOT IMPLEMENTED - Placeholder only.

    Args:
        path: Path to configuration file

    Raises:
        NotImplementedError: Always, as this is a skeleton.
    """
    raise NotImplementedError(
        "Configuration file loading not implemented. "
        "This module is in skeleton state."
    )
