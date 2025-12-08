"""
engine/orchestration

Temporal analysis orchestration for PRISM Engine.

This module provides:
- TemporalProfile: Named preset configurations
- TemporalConfig: Full configuration for temporal runs
- TemporalResult: Canonical result container
- Profile-based API for running temporal analysis
"""

from .temporal_analysis import (
    TemporalProfile,
    TemporalConfig,
    TemporalResult,
    TEMPORAL_PROFILES,
    build_temporal_config,
    run_temporal_analysis_from_panel,
    run_temporal_analysis_from_db,
    load_temporal_panel,
    quick_start_from_db,
)

__all__ = [
    "TemporalProfile",
    "TemporalConfig",
    "TemporalResult",
    "TEMPORAL_PROFILES",
    "build_temporal_config",
    "run_temporal_analysis_from_panel",
    "run_temporal_analysis_from_db",
    "load_temporal_panel",
    "quick_start_from_db",
]
