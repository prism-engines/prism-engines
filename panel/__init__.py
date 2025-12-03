"""
PRISM Panel Module
==================

Panel building layer that reads from the database and registries
to produce a unified, engine-ready panel.

Usage:
    from panel.build_panel import build_panel

    # Build the master panel
    df = build_panel()
"""

from panel.build_panel import build_panel
from panel.validators import validate_panel
from panel.transforms_market import align_market_series, compute_returns
from panel.transforms_econ import align_econ_series, forward_fill_to_daily

__all__ = [
    "build_panel",
    "validate_panel",
    "align_market_series",
    "compute_returns",
    "align_econ_series",
    "forward_fill_to_daily",
]
