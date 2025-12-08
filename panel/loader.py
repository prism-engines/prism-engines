"""
Panel Loader Compatibility Wrapper
==================================

Provides a PanelLoader class that wraps the runtime loader
to maintain backward compatibility with the WorkflowExecutor.

The old static panel system (panels/) has been replaced with
the runtime loader (panel/runtime_loader.py). This wrapper
adapts the new system to the old interface.
"""

from typing import List, Dict, Any, Optional
import logging

import pandas as pd

logger = logging.getLogger(__name__)


class PanelResult:
    """
    Wrapper around a panel DataFrame to provide the old Panel interface.

    The old Panel class had methods like get_indicators(), PANEL_NAME, etc.
    This wrapper provides the same interface using the new DataFrame-based approach.
    """

    def __init__(self, df: pd.DataFrame, panel_name: str, indicators: List[str]):
        """
        Initialize panel result.

        Args:
            df: Wide-format DataFrame (date index, indicator columns)
            panel_name: Name of the panel
            indicators: List of indicator names
        """
        self._df = df
        self._panel_name = panel_name
        self._indicators = indicators

    @property
    def PANEL_NAME(self) -> str:
        """Panel name (backward compatibility)."""
        return self._panel_name

    @property
    def data(self) -> pd.DataFrame:
        """Access the underlying DataFrame."""
        return self._df

    def get_indicators(self) -> List[str]:
        """Get list of indicator names in this panel."""
        return self._indicators

    def get_data(self) -> pd.DataFrame:
        """Get the panel data as a DataFrame."""
        return self._df

    def __len__(self) -> int:
        """Number of indicators."""
        return len(self._indicators)


class PanelLoader:
    """
    Panel loader compatible with the old interface.

    Maps panel keys to indicator lists and loads them via runtime_loader.
    """

    # Panel definitions - maps panel keys to indicator configurations
    # These are the commonly used panels in PRISM
    PANEL_CONFIGS = {
        "market": {
            "description": "Market indicators (S&P 500, VIX, etc.)",
            "indicators": ["spy", "vix", "tnx", "dxy"],
            "tags": ["market", "equity", "volatility"],
        },
        "economy": {
            "description": "Economic indicators (GDP, unemployment, etc.)",
            "indicators": ["gdp", "unrate", "cpi", "indpro", "payems"],
            "tags": ["economic", "macro"],
        },
        "rates": {
            "description": "Interest rate indicators",
            "indicators": ["dgs2", "dgs10", "dgs30", "fedfunds", "t10y2y"],
            "tags": ["rates", "treasury", "yield"],
        },
        "credit": {
            "description": "Credit spread indicators",
            "indicators": ["bamlh0a0hym2", "bamlc0a0cm", "ted_spread"],
            "tags": ["credit", "spreads"],
        },
        "calibration": {
            "description": "Calibration panel (from registry)",
            "indicators": [],  # Loaded dynamically
            "tags": ["calibration"],
            "dynamic": True,
        },
    }

    def __init__(self):
        """Initialize the panel loader."""
        self._cache = {}

    def list_panels(self) -> List[str]:
        """
        List available panel keys.

        Returns:
            List of panel key names
        """
        return list(self.PANEL_CONFIGS.keys())

    def get_panel_info(self, panel_key: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata about a panel.

        Args:
            panel_key: Panel identifier

        Returns:
            Dict with panel info or None if not found
        """
        config = self.PANEL_CONFIGS.get(panel_key)
        if not config:
            return None

        return {
            "key": panel_key,
            "description": config.get("description", ""),
            "indicators": config.get("indicators", []),
            "tags": config.get("tags", []),
            "dynamic": config.get("dynamic", False),
        }

    def load_panel(
        self,
        panel_key: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> PanelResult:
        """
        Load a panel by key.

        Args:
            panel_key: Panel identifier (market, economy, etc.)
            start_date: Optional start date filter
            end_date: Optional end date filter

        Returns:
            PanelResult with the loaded data
        """
        from panel.runtime_loader import (
            load_panel,
            load_calibrated_panel,
            list_available_indicators,
        )

        config = self.PANEL_CONFIGS.get(panel_key)

        if not config:
            # Try loading as a custom panel with all available indicators
            logger.warning(f"Unknown panel '{panel_key}', loading all indicators")
            indicators = list_available_indicators()
            if not indicators:
                raise ValueError(f"No indicators available for panel: {panel_key}")
        elif config.get("dynamic"):
            # Dynamic panel - load from registry
            if panel_key == "calibration":
                df = load_calibrated_panel(
                    start_date=start_date,
                    end_date=end_date,
                )
                indicators = list(df.columns)
                return PanelResult(df, panel_key, indicators)
            else:
                indicators = list_available_indicators()
        else:
            indicators = config.get("indicators", [])

        # Load the panel
        df = load_panel(
            indicator_names=indicators,
            start_date=start_date,
            end_date=end_date,
        )

        # Filter to only indicators that actually loaded
        loaded_indicators = [c for c in indicators if c in df.columns]

        return PanelResult(df, panel_key, loaded_indicators)

    def get_available_indicators(self) -> List[str]:
        """
        Get all available indicators from the database.

        Returns:
            List of indicator names
        """
        from panel.runtime_loader import list_available_indicators
        return list_available_indicators()
