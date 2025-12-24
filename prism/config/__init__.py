"""
PRISM Configuration Module

Centralized configuration loading for all agent parameters.
Reads from YAML files in the config/ directory at repo root.

Usage:
    from prism.config import get_config, get_windows_config, get_stability_config

    # Load specific config
    windows = get_windows_config()
    min_quality = windows.get("min_quality_score", 0.35)

    # Load any config by name
    agreement = get_config("agreement")

    # Access with defaults (matches hardcoded behavior)
    from prism.config import ConfigLoader
    loader = ConfigLoader()
    threshold = loader.get("stability", "stability_threshold", default=0.6)
"""

from pathlib import Path
from typing import Any, Dict, Optional
import logging
import yaml

logger = logging.getLogger(__name__)

# Config directory at repo root
REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "config"


class ConfigLoader:
    """
    Centralized configuration loader for PRISM agents.

    Loads YAML configuration files from the config/ directory.
    Caches loaded configs for performance.

    Thread-safe for read operations.

    Usage:
        loader = ConfigLoader()

        # Get entire config dict
        windows_config = loader.load("windows")

        # Get specific value with default
        min_quality = loader.get("windows", "min_quality_score", default=0.35)

        # Check if config exists
        if loader.exists("custom"):
            custom = loader.load("custom")
    """

    _instance: Optional["ConfigLoader"] = None
    _cache: Dict[str, Dict[str, Any]] = {}

    def __new__(cls) -> "ConfigLoader":
        """Singleton pattern - one loader instance per process."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._cache = {}
        return cls._instance

    @property
    def config_dir(self) -> Path:
        """Return the config directory path."""
        return CONFIG_DIR

    def exists(self, name: str) -> bool:
        """Check if a config file exists."""
        path = self.config_dir / f"{name}.yaml"
        return path.exists()

    def load(self, name: str, reload: bool = False) -> Dict[str, Any]:
        """
        Load a configuration file by name.

        Args:
            name: Config file name without extension (e.g., "windows", "stability")
            reload: Force reload from disk even if cached

        Returns:
            Configuration dictionary

        Raises:
            FileNotFoundError: If config file doesn't exist
        """
        if not reload and name in self._cache:
            return self._cache[name]

        path = self.config_dir / f"{name}.yaml"

        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        try:
            with open(path) as f:
                config = yaml.safe_load(f) or {}

            self._cache[name] = config
            logger.debug(f"Loaded config: {name} ({len(config)} keys)")
            return config

        except yaml.YAMLError as e:
            logger.error(f"Failed to parse config {name}: {e}")
            raise ValueError(f"Invalid YAML in {path}: {e}")

    def get(
        self,
        config_name: str,
        key: str,
        default: Any = None,
        required: bool = False
    ) -> Any:
        """
        Get a specific value from a config file.

        Args:
            config_name: Config file name (e.g., "windows")
            key: Key to retrieve
            default: Default value if key not found
            required: If True, raise KeyError when key not found

        Returns:
            Configuration value or default
        """
        try:
            config = self.load(config_name)
        except FileNotFoundError:
            if required:
                raise
            logger.warning(f"Config {config_name} not found, using default for {key}")
            return default

        if key in config:
            return config[key]

        if required:
            raise KeyError(f"Required key '{key}' not found in config '{config_name}'")

        return default

    def get_nested(
        self,
        config_name: str,
        *keys: str,
        default: Any = None
    ) -> Any:
        """
        Get a nested value from a config file.

        Args:
            config_name: Config file name
            *keys: Path of keys to traverse
            default: Default value if path not found

        Returns:
            Nested configuration value or default

        Example:
            loader.get_nested("validation", "reason_penalties", "ZERO_VARIANCE", "penalty")
        """
        try:
            config = self.load(config_name)
        except FileNotFoundError:
            return default

        value = config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default

        return value

    def clear_cache(self, name: Optional[str] = None) -> None:
        """
        Clear cached configurations.

        Args:
            name: Specific config to clear, or None to clear all
        """
        if name is None:
            self._cache.clear()
            logger.debug("Cleared all config cache")
        elif name in self._cache:
            del self._cache[name]
            logger.debug(f"Cleared config cache: {name}")

    def list_configs(self) -> list:
        """List all available config files."""
        if not self.config_dir.exists():
            return []
        return [p.stem for p in self.config_dir.glob("*.yaml")]


# Module-level convenience functions

_loader: Optional[ConfigLoader] = None


def _get_loader() -> ConfigLoader:
    """Get the global ConfigLoader instance."""
    global _loader
    if _loader is None:
        _loader = ConfigLoader()
    return _loader


def get_config(name: str, reload: bool = False) -> Dict[str, Any]:
    """
    Load a configuration by name.

    Args:
        name: Config name (e.g., "windows", "stability")
        reload: Force reload from disk

    Returns:
        Configuration dictionary
    """
    return _get_loader().load(name, reload=reload)


def get_windows_config() -> Dict[str, Any]:
    """Load WindowSuitabilityPolicy configuration."""
    return get_config("windows")


def get_data_quality_config() -> Dict[str, Any]:
    """Load DataQualityAuditAgent configuration."""
    return get_config("data_quality")


def get_persistence_config() -> Dict[str, Any]:
    """Load PersistencePolicy configuration."""
    return get_config("persistence")


def get_agreement_config() -> Dict[str, Any]:
    """Load AgreementPolicy configuration."""
    return get_config("agreement")


def get_validation_config() -> Dict[str, Any]:
    """Load ValidationPolicy configuration."""
    return get_config("validation")


def get_stability_config() -> Dict[str, Any]:
    """Load EngineStabilityAuditAgent configuration."""
    return get_config("stability")


def get_geometry_config() -> Dict[str, Any]:
    """Load MultiViewGeometryAgent configuration."""
    return get_config("geometry")
