"""
PRISM Panel Module
==================

Domain-agnostic panel loading from the unified indicator_values table.

Panels are defined at runtime by the UI as simple lists of indicator names.
There are NO file-based panel definitions.

Usage:
    from panel import load_panel

    # Load indicators selected by UI at runtime
    selected = ["sp500", "vix", "t10y2y"]
    panel = load_panel(selected)

Legacy panel builders have been moved to legacy/panel_builders/
and are deprecated.
"""

from panel.runtime_loader import (
    load_panel,
    list_available_indicators,
    get_indicator_info,
)
from panel.validators import validate_panel

__all__ = [
    "load_panel",
    "list_available_indicators",
    "get_indicator_info",
    "validate_panel",
]
