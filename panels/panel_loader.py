"""
PRISM Panel Loader
===================

Auto-discovers and loads analysis panels from the registry.
Wraps the existing panel building code with a unified interface.

Usage:
    from panels.panel_loader import PanelLoader

    loader = PanelLoader()

    # List available panels
    panels = loader.list_panels()

    # Load a panel
    market = loader.load_panel("market")

    # Get panel data
    df = market.get_data()
    indicators = market.get_indicators()
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
import yaml

logger = logging.getLogger(__name__)

# Project root
PROJECT_ROOT = Path(__file__).parent.parent
REGISTRY_PATH = PROJECT_ROOT / "panels" / "registry.yaml"


class PanelLoadError(Exception):
    """Raised when a panel cannot be loaded."""
    pass


class Panel:
    """
    Represents a loaded analysis panel.

    A panel is a collection of indicators for a specific domain
    (market, economy, climate, etc.) with associated data.
    """

    def __init__(
        self,
        panel_key: str,
        config: Dict[str, Any],
        data=None
    ):
        """
        Initialize a panel.

        Args:
            panel_key: Panel identifier
            config: Panel configuration from registry
            data: Optional preloaded DataFrame
        """
        self.panel_key = panel_key
        self.PANEL_NAME = panel_key  # For compatibility with Phase 3 executor
        self.config = config
        self._data = data
        self._indicators: Optional[List[str]] = None

    @property
    def name(self) -> str:
        """Panel display name."""
        return self.config.get("name", self.panel_key)

    @property
    def description(self) -> str:
        """Panel description."""
        return self.config.get("description", "")

    @property
    def icon(self) -> str:
        """Panel icon identifier."""
        return self.config.get("icon", "chart_increasing")

    @property
    def resolution(self) -> str:
        """Default data resolution."""
        return self.config.get("default_resolution", "daily")

    def get_data(self):
        """
        Get panel data as DataFrame.

        Returns:
            pandas DataFrame with panel data
        """
        if self._data is not None:
            return self._data

        # Try to load data using existing panel module
        try:
            from panel.build_panel import build_panel
            self._data = build_panel()
            return self._data
        except ImportError as e:
            logger.warning(f"Could not import build_panel: {e}")
        except Exception as e:
            logger.warning(f"Could not build panel: {e}")

        return None

    def get_indicators(self) -> List[str]:
        """
        Get list of indicator names in this panel.

        Returns:
            List of indicator column names
        """
        if self._indicators is not None:
            return self._indicators

        # Try to get from data
        data = self.get_data()
        if data is not None:
            self._indicators = list(data.columns)
            return self._indicators

        # Return placeholder based on panel type
        indicator_map = {
            "market": [
                "SPY", "VIX", "HYG", "LQD", "TLT", "GLD",
                "USO", "DXY", "MOVE", "SKEW", "PUT_CALL"
            ],
            "economy": [
                "GDP", "UNRATE", "CPI", "PPI", "PAYEMS",
                "INDPRO", "RSALES", "HOUST", "PERMIT"
            ],
            "climate": [
                "TEMP_ANOMALY", "CO2", "SEA_LEVEL", "ICE_EXTENT"
            ],
            "custom": []
        }

        self._indicators = indicator_map.get(self.panel_key, [])
        return self._indicators

    def get_indicator_count(self) -> int:
        """Get number of indicators."""
        return len(self.get_indicators())

    def __repr__(self) -> str:
        return f"Panel({self.panel_key}, indicators={self.get_indicator_count()})"


class PanelLoader:
    """
    Loads and manages analysis panels from the registry.

    The loader provides:
    - Auto-discovery of panels from registry.yaml
    - Panel instantiation with data loading
    - Caching of loaded panels
    """

    def __init__(self, registry_path: Optional[Path] = None):
        """
        Initialize the panel loader.

        Args:
            registry_path: Path to registry.yaml (default: panels/registry.yaml)
        """
        self.registry_path = registry_path or REGISTRY_PATH
        self._registry: Optional[Dict] = None
        self._panel_cache: Dict[str, Panel] = {}

    @property
    def registry(self) -> Dict:
        """Lazy-load the registry."""
        if self._registry is None:
            self._registry = self._load_registry()
        return self._registry

    def _load_registry(self) -> Dict:
        """Load the panel registry from YAML."""
        if not self.registry_path.exists():
            # Return default registry if file doesn't exist
            logger.warning(f"Panel registry not found: {self.registry_path}, using defaults")
            return {
                "panels": {
                    "market": {
                        "name": "Market Indicators",
                        "description": "Financial market data",
                        "icon": "chart_increasing",
                    },
                    "economy": {
                        "name": "Economic Indicators",
                        "description": "Macroeconomic data",
                        "icon": "bank",
                    },
                    "climate": {
                        "name": "Climate Indicators",
                        "description": "Environmental data",
                        "icon": "globe",
                    },
                    "custom": {
                        "name": "Custom Panel",
                        "description": "User-defined data",
                        "icon": "gear",
                    },
                }
            }

        with open(self.registry_path, 'r') as f:
            registry = yaml.safe_load(f)

        logger.info(f"Loaded panel registry: {len(registry.get('panels', {}))} panels")
        return registry

    def reload_registry(self) -> None:
        """Force reload of the registry."""
        self._registry = None
        self._panel_cache.clear()
        logger.info("Panel registry reloaded")

    # -------------------------------------------------------------------------
    # DISCOVERY METHODS
    # -------------------------------------------------------------------------

    def list_panels(self) -> List[str]:
        """
        List all available panel names.

        Returns:
            List of panel keys
        """
        return list(self.registry.get("panels", {}).keys())

    def get_panel_info(self, panel_key: str) -> Optional[Dict]:
        """
        Get full configuration for a panel.

        Args:
            panel_key: Panel identifier

        Returns:
            Panel configuration dict or None
        """
        return self.registry.get("panels", {}).get(panel_key)

    def get_panels_by_category(self, category: str) -> List[str]:
        """
        Get all panels in a category.

        Args:
            category: Category name (financial, macroeconomic, etc.)

        Returns:
            List of panel keys in that category
        """
        categories = self.registry.get("categories", {})
        cat_info = categories.get(category, {})
        return cat_info.get("panels", [])

    def get_categories(self) -> Dict[str, Dict]:
        """
        Get all panel categories with descriptions.

        Returns:
            Dictionary of category info
        """
        return self.registry.get("categories", {})

    def get_defaults(self) -> Dict[str, Any]:
        """Get default configuration."""
        return self.registry.get("defaults", {})

    # -------------------------------------------------------------------------
    # LOADING METHODS
    # -------------------------------------------------------------------------

    def load_panel(
        self,
        panel_key: str,
        load_data: bool = False,
        **kwargs
    ) -> Panel:
        """
        Load a panel by key.

        Args:
            panel_key: Panel identifier
            load_data: Whether to load data immediately
            **kwargs: Additional arguments

        Returns:
            Panel instance
        """
        cache_key = panel_key

        if cache_key in self._panel_cache:
            return self._panel_cache[cache_key]

        config = self.get_panel_info(panel_key)
        if not config:
            raise PanelLoadError(f"Unknown panel: {panel_key}")

        panel = Panel(panel_key, config)

        if load_data:
            panel.get_data()  # Trigger data loading

        self._panel_cache[cache_key] = panel
        logger.info(f"Loaded panel: {panel_key}")
        return panel

    def load_all_panels(self) -> Dict[str, Panel]:
        """
        Load all available panels.

        Returns:
            Dictionary mapping panel_key -> Panel instance
        """
        panels = {}
        for key in self.list_panels():
            try:
                panels[key] = self.load_panel(key)
            except PanelLoadError as e:
                logger.warning(f"Could not load panel {key}: {e}")
        return panels

    # -------------------------------------------------------------------------
    # UTILITY METHODS
    # -------------------------------------------------------------------------

    def get_panel_summary(self) -> str:
        """Get a formatted summary of all panels."""
        lines = [
            "=" * 60,
            "PRISM PANEL REGISTRY",
            "=" * 60,
            "",
        ]

        for key in self.list_panels():
            info = self.get_panel_info(key)
            if info:
                icon = self._get_emoji(info.get("icon", "chart"))
                lines.append(f"{icon} {info.get('name', key)} [{key}]")
                lines.append(f"   {info.get('description', '')}")
                lines.append("")

        lines.extend([
            "-" * 60,
            f"Total panels: {len(self.list_panels())}",
            "=" * 60,
        ])

        return "\n".join(lines)

    def _get_emoji(self, icon_name: str) -> str:
        """Convert icon name to emoji."""
        icon_map = {
            "chart_increasing": "📈",
            "bank": "🏛️",
            "globe": "🌍",
            "gear": "⚙️",
            "chart": "📊",
        }
        return icon_map.get(icon_name, "📊")

    def __repr__(self) -> str:
        return f"PanelLoader(panels={len(self.list_panels())})"


# =============================================================================
# MODULE-LEVEL CONVENIENCE FUNCTIONS
# =============================================================================

_default_loader: Optional[PanelLoader] = None


def get_loader() -> PanelLoader:
    """Get the default panel loader singleton."""
    global _default_loader
    if _default_loader is None:
        _default_loader = PanelLoader()
    return _default_loader


def list_panels() -> List[str]:
    """List all available panels."""
    return get_loader().list_panels()


def load_panel(panel_key: str, **kwargs) -> Panel:
    """Load a panel by key."""
    return get_loader().load_panel(panel_key, **kwargs)


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="PRISM Panel Loader")
    parser.add_argument("--list", action="store_true", help="List all panels")
    parser.add_argument("--info", type=str, help="Show info for specific panel")

    args = parser.parse_args()

    loader = PanelLoader()

    if args.list:
        print(loader.get_panel_summary())

    elif args.info:
        info = loader.get_panel_info(args.info)
        if info:
            print(f"\nPanel: {args.info}")
            print("-" * 40)
            for key, value in info.items():
                print(f"  {key}: {value}")
        else:
            print(f"Panel not found: {args.info}")

    else:
        print(loader.get_panel_summary())
