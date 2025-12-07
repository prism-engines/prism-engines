"""
PRISM Settings Manager
=======================

Manages settings, defaults, and presets.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
import yaml

logger = logging.getLogger(__name__)


class SettingsManager:
    """
    Manages PRISM configuration.

    Configuration hierarchy (lowest to highest priority):
    1. Built-in defaults
    2. config/defaults.yaml
    3. config/settings.yaml
    4. Preset (if selected)
    5. Runtime overrides
    """

    BUILTIN_DEFAULTS = {
        "analysis": {
            "default_panel": "market",
            "default_workflow": "lens_validation",
            "default_lens_set": "quick",
            "comparison_periods": [2008, 2020, 2022],
        },
        "output": {
            "directory": "output",
            "format": "html",
            "include_json": True,
            "include_csv": False,
            "open_browser": False,
        },
        "engine": {
            "timeout_seconds": 300,
            "max_retries": 3,
            "parallel": False,
        },
        "logging": {
            "level": "INFO",
            "file": None,
        },
    }

    def __init__(self, config_dir: Optional[Path] = None):
        """
        Initialize settings manager.

        Args:
            config_dir: Configuration directory
        """
        self.config_dir = config_dir or Path("config")
        self.presets_dir = self.config_dir / "presets"

        # Settings layers
        self._defaults: Dict[str, Any] = dict(self.BUILTIN_DEFAULTS)
        self._settings: Dict[str, Any] = {}
        self._preset: Dict[str, Any] = {}
        self._overrides: Dict[str, Any] = {}

        # Load configuration files
        self._load_configs()

    def _load_configs(self) -> None:
        """Load configuration files."""
        # Load defaults
        defaults_file = self.config_dir / "defaults.yaml"
        if defaults_file.exists():
            try:
                with open(defaults_file) as f:
                    file_defaults = yaml.safe_load(f) or {}
                self._merge_dict(self._defaults, file_defaults)
                logger.info(f"Loaded defaults from {defaults_file}")
            except Exception as e:
                logger.warning(f"Failed to load defaults: {e}")

        # Load settings
        settings_file = self.config_dir / "settings.yaml"
        if settings_file.exists():
            try:
                with open(settings_file) as f:
                    self._settings = yaml.safe_load(f) or {}
                logger.info(f"Loaded settings from {settings_file}")
            except Exception as e:
                logger.warning(f"Failed to load settings: {e}")

    def _merge_dict(self, base: Dict, overlay: Dict) -> None:
        """Recursively merge overlay into base."""
        for key, value in overlay.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._merge_dict(base[key], value)
            else:
                base[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a setting value.

        Args:
            key: Dot-notation key (e.g., "analysis.default_panel")
            default: Default value if not found

        Returns:
            Setting value
        """
        # Check layers in priority order
        for layer in [self._overrides, self._preset, self._settings, self._defaults]:
            value = self._get_nested(layer, key)
            if value is not None:
                return value

        return default

    def _get_nested(self, d: Dict, key: str) -> Any:
        """Get nested dictionary value using dot notation."""
        keys = key.split(".")
        value = d
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return None
        return value

    def set(self, key: str, value: Any, persist: bool = False) -> None:
        """
        Set a setting value.

        Args:
            key: Dot-notation key
            value: Value to set
            persist: Whether to save to settings.yaml
        """
        self._set_nested(self._overrides, key, value)

        if persist:
            self._set_nested(self._settings, key, value)
            self._save_settings()

    def _set_nested(self, d: Dict, key: str, value: Any) -> None:
        """Set nested dictionary value using dot notation."""
        keys = key.split(".")
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        d[keys[-1]] = value

    def _save_settings(self) -> None:
        """Save settings to file."""
        self.config_dir.mkdir(parents=True, exist_ok=True)
        settings_file = self.config_dir / "settings.yaml"

        try:
            with open(settings_file, 'w') as f:
                yaml.dump(self._settings, f, default_flow_style=False)
            logger.info(f"Saved settings to {settings_file}")
        except Exception as e:
            logger.error(f"Failed to save settings: {e}")

    def load_preset(self, name: str) -> bool:
        """
        Load a preset configuration.

        Args:
            name: Preset name

        Returns:
            True if preset loaded successfully
        """
        preset_file = self.presets_dir / f"{name}.yaml"

        if not preset_file.exists():
            logger.warning(f"Preset not found: {name}")
            return False

        try:
            with open(preset_file) as f:
                self._preset = yaml.safe_load(f) or {}
            logger.info(f"Loaded preset: {name}")
            return True
        except Exception as e:
            logger.error(f"Failed to load preset {name}: {e}")
            return False

    def list_presets(self) -> List[str]:
        """List available presets."""
        if not self.presets_dir.exists():
            return []

        return [
            f.stem for f in self.presets_dir.glob("*.yaml")
            if not f.name.startswith("_")
        ]

    def create_preset(self, name: str, settings: Dict[str, Any]) -> bool:
        """
        Create a new preset.

        Args:
            name: Preset name
            settings: Preset settings

        Returns:
            True if created successfully
        """
        self.presets_dir.mkdir(parents=True, exist_ok=True)
        preset_file = self.presets_dir / f"{name}.yaml"

        try:
            with open(preset_file, 'w') as f:
                yaml.dump(settings, f, default_flow_style=False)
            logger.info(f"Created preset: {name}")
            return True
        except Exception as e:
            logger.error(f"Failed to create preset {name}: {e}")
            return False

    def get_all(self) -> Dict[str, Any]:
        """Get all effective settings."""
        result = dict(self._defaults)
        self._merge_dict(result, self._settings)
        self._merge_dict(result, self._preset)
        self._merge_dict(result, self._overrides)
        return result

    def reset(self) -> None:
        """Reset to defaults."""
        self._settings = {}
        self._preset = {}
        self._overrides = {}
