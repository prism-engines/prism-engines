"""
PRISM Panels Package
=====================

Auto-discoverable analysis panels loaded from registry.yaml.

Panels represent domains of analysis (market, economy, climate, etc.)
with associated indicators and data.

Usage:
    from panels import PanelLoader, list_panels, load_panel

    # List all panels
    panels = list_panels()

    # Load a specific panel
    market = load_panel("market")

    # Get panel data
    indicators = market.get_indicators()
"""

from .panel_loader import (
    PanelLoader,
    PanelLoadError,
    Panel,
    get_loader,
    list_panels,
    load_panel,
)

__all__ = [
    "PanelLoader",
    "PanelLoadError",
    "Panel",
    "get_loader",
    "list_panels",
    "load_panel",
]
