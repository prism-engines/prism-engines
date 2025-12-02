"""
Registry Helper - Utilities for loading and accessing PRISM registry data

This module provides functions to:
- Load system and metric registries
- Get panel paths from registry
- Access column metadata
- Filter series by type (economic, market, etc.)
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import pandas as pd

logger = logging.getLogger(__name__)


def _get_project_root() -> Path:
    """Get the project root directory."""
    # Start from this file's location (utils/registry.py)
    current = Path(__file__).resolve().parent

    # Go up to project root
    root = current.parent

    # Verify we found the right location
    if (root / 'config.py').exists():
        return root

    # Fallback to cwd
    return Path.cwd()


def load_system_registry(project_root: Optional[Path] = None) -> Dict[str, Any]:
    """
    Load the system registry configuration.

    Args:
        project_root: Optional project root path. Auto-detected if not provided.

    Returns:
        Dictionary containing system registry configuration.

    Raises:
        FileNotFoundError: If system_registry.json is not found.
    """
    root = Path(project_root) if project_root else _get_project_root()
    registry_path = root / 'data' / 'registry' / 'system_registry.json'

    if not registry_path.exists():
        raise FileNotFoundError(f"System registry not found at: {registry_path}")

    with open(registry_path, 'r') as f:
        registry = json.load(f)

    logger.debug(f"Loaded system registry from {registry_path}")
    return registry


def load_metric_registry(project_root: Optional[Path] = None) -> List[Dict[str, Any]]:
    """
    Load the metric registry configuration.

    Args:
        project_root: Optional project root path. Auto-detected if not provided.

    Returns:
        List of metric definitions.

    Raises:
        FileNotFoundError: If metric_registry.json is not found.
    """
    root = Path(project_root) if project_root else _get_project_root()
    registry_path = root / 'data' / 'registry' / 'metric_registry.json'

    if not registry_path.exists():
        raise FileNotFoundError(f"Metric registry not found at: {registry_path}")

    with open(registry_path, 'r') as f:
        registry = json.load(f)

    logger.debug(f"Loaded metric registry from {registry_path}")
    return registry


def get_panel_path(
    panel_type: str = 'master',
    project_root: Optional[Path] = None,
    absolute: bool = True
) -> Path:
    """
    Get the path to a panel file from the system registry.

    Args:
        panel_type: Type of panel ('master' or 'cleaned')
        project_root: Optional project root path
        absolute: If True, return absolute path

    Returns:
        Path to the panel file

    Raises:
        ValueError: If panel_type is not recognized
    """
    registry = load_system_registry(project_root)
    root = Path(project_root) if project_root else _get_project_root()

    if panel_type == 'master':
        rel_path = registry.get('panel_master_path', 'data/panels/master_panel.csv')
    elif panel_type == 'cleaned':
        rel_path = registry.get('panel_cleaned_path', 'data/cleaned/master_panel_cleaned.csv')
    else:
        raise ValueError(f"Unknown panel type: {panel_type}. Use 'master' or 'cleaned'.")

    panel_path = root / rel_path

    if absolute:
        return panel_path.resolve()
    return panel_path


def load_panel(
    panel_type: str = 'master',
    project_root: Optional[Path] = None,
    parse_dates: bool = True
) -> pd.DataFrame:
    """
    Load a panel DataFrame from the path specified in the system registry.

    Args:
        panel_type: Type of panel ('master' or 'cleaned')
        project_root: Optional project root path
        parse_dates: Whether to parse the first column as dates

    Returns:
        Panel DataFrame

    Raises:
        FileNotFoundError: If panel file is not found
    """
    panel_path = get_panel_path(panel_type, project_root)

    if not panel_path.exists():
        raise FileNotFoundError(f"Panel file not found at: {panel_path}")

    # Load CSV with date parsing
    if parse_dates:
        df = pd.read_csv(panel_path, index_col=0, parse_dates=True)
    else:
        df = pd.read_csv(panel_path, index_col=0)

    logger.info(f"Loaded panel from {panel_path}: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def get_series_by_type(
    series_type: str,
    project_root: Optional[Path] = None
) -> List[str]:
    """
    Get column names for a specific series type (economic, market, etc.).

    Args:
        series_type: Type of series ('economic' or 'market')
        project_root: Optional project root path

    Returns:
        List of column names for the specified series type

    Raises:
        ValueError: If series_type is not recognized
    """
    registry = load_system_registry(project_root)

    series_types = registry.get('column_metadata', {}).get('series_types', {})

    if series_type not in series_types:
        available = list(series_types.keys())
        raise ValueError(f"Unknown series type: {series_type}. Available: {available}")

    return series_types[series_type].get('columns', [])


def get_economic_series(project_root: Optional[Path] = None) -> List[str]:
    """Get list of economic series column names."""
    return get_series_by_type('economic', project_root)


def get_market_series(project_root: Optional[Path] = None) -> List[str]:
    """Get list of market series column names."""
    return get_series_by_type('market', project_root)


def get_metric_by_key(
    key: str,
    project_root: Optional[Path] = None
) -> Optional[Dict[str, Any]]:
    """
    Get a metric definition by its key.

    Args:
        key: The metric key (e.g., 'dgs10', 'spy')
        project_root: Optional project root path

    Returns:
        Metric definition dict or None if not found
    """
    metrics = load_metric_registry(project_root)

    for metric in metrics:
        if metric.get('key') == key:
            return metric

    return None


def get_metrics_by_source(
    source: str,
    project_root: Optional[Path] = None
) -> List[Dict[str, Any]]:
    """
    Get all metrics from a specific source.

    Args:
        source: Data source ('fred' or 'yahoo')
        project_root: Optional project root path

    Returns:
        List of metric definitions from the specified source
    """
    metrics = load_metric_registry(project_root)
    return [m for m in metrics if m.get('source') == source]


def get_engine_config(
    key: Optional[str] = None,
    project_root: Optional[Path] = None
) -> Union[Dict[str, Any], Any]:
    """
    Get engine configuration from the system registry.

    Args:
        key: Optional specific config key to retrieve
        project_root: Optional project root path

    Returns:
        Engine config dict or specific value if key provided
    """
    registry = load_system_registry(project_root)
    engine_config = registry.get('engine_config', {})

    if key is not None:
        return engine_config.get(key)

    return engine_config


class RegistryManager:
    """
    Class-based registry manager for convenient access to registry data.

    Usage:
        registry = RegistryManager()
        panel = registry.load_panel()
        economic_cols = registry.get_economic_series()
    """

    def __init__(self, project_root: Optional[Path] = None):
        """
        Initialize the registry manager.

        Args:
            project_root: Optional project root path
        """
        self.project_root = Path(project_root) if project_root else _get_project_root()
        self._system_registry: Optional[Dict] = None
        self._metric_registry: Optional[List] = None

    @property
    def system_registry(self) -> Dict[str, Any]:
        """Lazy-load and cache system registry."""
        if self._system_registry is None:
            self._system_registry = load_system_registry(self.project_root)
        return self._system_registry

    @property
    def metric_registry(self) -> List[Dict[str, Any]]:
        """Lazy-load and cache metric registry."""
        if self._metric_registry is None:
            self._metric_registry = load_metric_registry(self.project_root)
        return self._metric_registry

    def get_panel_path(self, panel_type: str = 'master') -> Path:
        """Get path to panel file."""
        return get_panel_path(panel_type, self.project_root)

    def load_panel(self, panel_type: str = 'master', parse_dates: bool = True) -> pd.DataFrame:
        """Load panel DataFrame."""
        return load_panel(panel_type, self.project_root, parse_dates)

    def get_economic_series(self) -> List[str]:
        """Get economic series column names."""
        return get_economic_series(self.project_root)

    def get_market_series(self) -> List[str]:
        """Get market series column names."""
        return get_market_series(self.project_root)

    def get_engine_config(self, key: Optional[str] = None) -> Union[Dict, Any]:
        """Get engine configuration."""
        return get_engine_config(key, self.project_root)

    def filter_panel_by_type(
        self,
        df: pd.DataFrame,
        series_type: str
    ) -> pd.DataFrame:
        """
        Filter a panel DataFrame to only include columns of a specific type.

        Args:
            df: Panel DataFrame
            series_type: Type of series ('economic' or 'market')

        Returns:
            Filtered DataFrame
        """
        columns = get_series_by_type(series_type, self.project_root)
        available_columns = [c for c in columns if c in df.columns]
        return df[available_columns]
