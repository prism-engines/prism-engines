"""
engine/orchestration/temporal_analysis.py

Unified temporal analysis orchestration for PRISM.

This module:
- Defines temporal "profiles" (chromebook / standard / powerful)
- Encodes resolution (step size, window lengths) in a single config object
- Provides a clean API for:
    * run_temporal_analysis_from_panel()
    * run_temporal_analysis_from_db()
    * quick_start_from_db()

Usage:
    from engine.orchestration.temporal_analysis import (
        build_temporal_config,
        run_temporal_analysis_from_db,
        TEMPORAL_PROFILES,
    )

    # Build config from profile with optional overrides
    cfg = build_temporal_config(
        profile="standard",
        start_date="2010-01-01",
        systems=["finance"],
    )

    # Run analysis
    result = run_temporal_analysis_from_db(db_path=None, config=cfg)

    # Access scores
    print(result.scores.head())
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Literal, Tuple, Any
from datetime import datetime

import numpy as np
import pandas as pd


Frequency = Literal["D", "W", "M"]


@dataclass
class TemporalProfile:
    """Defines a named temporal profile for convenience presets."""

    name: str
    description: str
    step_months: float                  # e.g. 0.5, 1.0, 2.0
    window_months: Sequence[int]        # e.g. [3, 6, 12, 24]
    lenses: Sequence[str]               # engine lens IDs to run
    min_history_years: int = 10
    frequency: Frequency = "M"          # resample frequency for outputs


@dataclass
class TemporalConfig:
    """
    Full configuration for a temporal run.

    This can be constructed either from a TemporalProfile + overrides,
    or directly by callers.
    """

    profile_name: str
    start_date: Optional[pd.Timestamp] = None
    end_date: Optional[pd.Timestamp] = None

    step_months: float = 1.0
    window_months: Sequence[int] = field(default_factory=lambda: (3, 6, 12, 24))
    lenses: Sequence[str] = field(default_factory=lambda: ("coherence", "pca", "volatility"))
    min_history_years: int = 10
    frequency: Frequency = "M"

    systems: Optional[Sequence[str]] = None     # e.g. ["finance"], ["climate"], ["finance", "climate"]
    indicators: Optional[Sequence[str]] = None  # specific indicator IDs (optional)


@dataclass
class TemporalResult:
    """
    Canonical result for temporal runs.

    Contains:
    - config: The TemporalConfig used for this run
    - scores: DataFrame with time index and per-lens scores
    - metadata: Run timings, engine version, etc.
    """

    config: TemporalConfig
    scores: pd.DataFrame          # index: date, columns: lens_* scores
    metadata: Dict[str, object]   # e.g., run timings, engine version, etc.


# ---------------------------------------------------------------------
# Profile registry
# ---------------------------------------------------------------------

TEMPORAL_PROFILES: Dict[str, TemporalProfile] = {
    "chromebook": TemporalProfile(
        name="chromebook",
        description="Light profile for low-CPU machines (3 lenses, ~2-month steps).",
        step_months=2.0,
        window_months=(6, 12, 24),
        lenses=("magnitude", "pca", "influence"),
        min_history_years=10,
        frequency="M",
    ),
    "standard": TemporalProfile(
        name="standard",
        description="Balanced profile (4 lenses, ~1-month steps).",
        step_months=1.0,
        window_months=(3, 6, 12, 24),
        lenses=("magnitude", "pca", "influence", "clustering"),
        min_history_years=15,
        frequency="M",
    ),
    "powerful": TemporalProfile(
        name="powerful",
        description="High-resolution profile (5 lenses, 0.5-month steps).",
        step_months=0.5,
        window_months=(3, 6, 12, 24, 36),
        lenses=("magnitude", "pca", "influence", "clustering", "decomposition"),
        min_history_years=20,
        frequency="M",
    ),
}


def build_temporal_config(
    profile: str = "standard",
    start_date: Optional[str | pd.Timestamp] = None,
    end_date: Optional[str | pd.Timestamp] = None,
    step_months: Optional[float] = None,
    window_months: Optional[Sequence[int]] = None,
    lenses: Optional[Sequence[str]] = None,
    systems: Optional[Sequence[str]] = None,
    indicators: Optional[Sequence[str]] = None,
    frequency: Optional[Frequency] = None,
) -> TemporalConfig:
    """
    Construct a TemporalConfig from a profile and overrides.

    This is the main entry point for the CLI runner & dashboard.

    Args:
        profile: Profile name ('chromebook', 'standard', 'powerful')
        start_date: Optional start date (YYYY-MM-DD or Timestamp)
        end_date: Optional end date (YYYY-MM-DD or Timestamp)
        step_months: Override step size in months
        window_months: Override window sizes
        lenses: Override lenses to use
        systems: Filter by system IDs (e.g., ['finance', 'climate'])
        indicators: Filter by specific indicator IDs
        frequency: Override output frequency

    Returns:
        TemporalConfig ready for use with run_* functions

    Raises:
        ValueError: If profile name is unknown
    """
    if profile not in TEMPORAL_PROFILES:
        raise ValueError(f"Unknown temporal profile: {profile!r}")

    base = TEMPORAL_PROFILES[profile]

    def _to_ts(val: Optional[str | pd.Timestamp]) -> Optional[pd.Timestamp]:
        if val is None:
            return None
        if isinstance(val, pd.Timestamp):
            return val
        return pd.Timestamp(val)

    cfg = TemporalConfig(
        profile_name=base.name,
        start_date=_to_ts(start_date),
        end_date=_to_ts(end_date),
        step_months=step_months if step_months is not None else base.step_months,
        window_months=window_months if window_months is not None else tuple(base.window_months),
        lenses=lenses if lenses is not None else tuple(base.lenses),
        min_history_years=base.min_history_years,
        frequency=frequency if frequency is not None else base.frequency,
        systems=tuple(systems) if systems is not None else None,
        indicators=tuple(indicators) if indicators is not None else None,
    )
    return cfg


# ---------------------------------------------------------------------
# Orchestration API (wired to existing engine code)
# ---------------------------------------------------------------------


def _get_temporal_engine(panel: pd.DataFrame, lenses: Sequence[str]):
    """
    Get the existing TemporalEngine from engine_core.orchestration.

    Falls back to a minimal implementation if the import fails.
    """
    try:
        from engine_core.orchestration.temporal_analysis import TemporalEngine
        return TemporalEngine(panel, lenses=list(lenses))
    except ImportError:
        # Minimal fallback implementation
        return _MinimalTemporalEngine(panel, lenses=list(lenses))


class _MinimalTemporalEngine:
    """
    Minimal temporal engine for when engine_core is not available.
    Uses basic rolling window analysis without lens dependencies.
    """

    def __init__(self, panel: pd.DataFrame, lenses: List[str] = None):
        self.panel = panel.copy()
        self.lenses = lenses or ['magnitude', 'pca', 'influence']

    def run_rolling_analysis(
        self,
        window_days: int = 252,
        step_days: int = 21,
        min_window_pct: float = 0.8,
        progress_callback=None
    ) -> Dict[str, Any]:
        """Run basic rolling analysis."""
        panel = self.panel.copy()

        if not isinstance(panel.index, pd.DatetimeIndex):
            panel.index = pd.to_datetime(panel.index)

        panel = panel.sort_index()

        total_rows = len(panel)
        min_window_size = int(window_days * min_window_pct)

        if total_rows < min_window_size:
            raise ValueError(f"Not enough data. Need {min_window_size} rows, have {total_rows}")

        window_ends = list(range(window_days, total_rows, step_days))
        if window_ends and window_ends[-1] != total_rows - 1:
            window_ends.append(total_rows - 1)

        timestamps = []
        all_rankings = {col: [] for col in panel.columns}
        all_importance = {col: [] for col in panel.columns}

        for i, end_idx in enumerate(window_ends):
            start_idx = max(0, end_idx - window_days)
            window_data = panel.iloc[start_idx:end_idx].dropna()

            if len(window_data) < min_window_size:
                continue

            timestamps.append(panel.index[end_idx])

            # Basic importance: variance-based ranking
            variance = window_data.var()
            normalized = (variance - variance.min()) / (variance.max() - variance.min() + 1e-10)
            ranks = normalized.rank(ascending=False)

            for col in panel.columns:
                all_rankings[col].append(ranks.get(col, np.nan))
                all_importance[col].append(normalized.get(col, np.nan))

            if progress_callback:
                progress_callback(i + 1, len(window_ends))

        return {
            'timestamps': timestamps,
            'rankings': all_rankings,
            'importance': all_importance,
            'metadata': {
                'window_days': window_days,
                'step_days': step_days,
                'n_windows': len(timestamps),
                'date_range': (timestamps[0], timestamps[-1]) if timestamps else None,
                'indicators': list(panel.columns),
                'lenses_used': self.lenses,
            }
        }


def run_temporal_analysis_from_panel(
    panel: pd.DataFrame,
    config: TemporalConfig,
) -> TemporalResult:
    """
    Run temporal analysis on a pre-built panel (wide DataFrame).

    This function:
    - Reuses the existing temporal engine logic
    - Expects panel to be date-indexed and wide-form (columns = indicators)
    - Returns per-lens scores in a DataFrame with DatetimeIndex

    Args:
        panel: Wide-format DataFrame (index=date, columns=indicators)
        config: TemporalConfig with run parameters

    Returns:
        TemporalResult with scores DataFrame and metadata

    Raises:
        ValueError: If panel is empty
    """
    if panel.empty:
        raise ValueError("Panel is empty; cannot run temporal analysis.")

    start_time = datetime.now()

    # Apply date filters
    filtered_panel = panel.copy()
    if config.start_date is not None:
        filtered_panel = filtered_panel[filtered_panel.index >= config.start_date]
    if config.end_date is not None:
        filtered_panel = filtered_panel[filtered_panel.index <= config.end_date]

    if filtered_panel.empty:
        raise ValueError("Panel is empty after applying date filters.")

    # Convert step/window to trading days
    step_days = int(config.step_months * 21)  # ~21 trading days per month

    # Use primary window (first in list) for rolling analysis
    primary_window_months = config.window_months[0] if config.window_months else 12
    window_days = int(primary_window_months * 21)

    # Get temporal engine
    temporal = _get_temporal_engine(filtered_panel, config.lenses)

    # Run rolling analysis
    results = temporal.run_rolling_analysis(
        window_days=window_days,
        step_days=step_days,
    )

    elapsed = (datetime.now() - start_time).total_seconds()

    # Build scores DataFrame from results
    timestamps = results.get('timestamps', [])
    rankings = results.get('rankings', {})
    importance = results.get('importance', {})

    # Create score columns per lens (using rankings as proxy for scores)
    score_data = {}

    # Add consensus rank as primary score
    if rankings:
        for indicator, ranks in rankings.items():
            score_data[f"{indicator}_rank"] = ranks

    # Create combined lens scores
    for lens in config.lenses:
        # Use importance as lens score proxy
        lens_scores = []
        for i in range(len(timestamps)):
            indicator_scores = []
            for indicator in filtered_panel.columns:
                if indicator in importance and i < len(importance[indicator]):
                    indicator_scores.append(importance[indicator][i])
            lens_scores.append(np.nanmean(indicator_scores) if indicator_scores else np.nan)
        score_data[f"{lens}_score"] = lens_scores

    scores = pd.DataFrame(score_data, index=pd.DatetimeIndex(timestamps))

    metadata: Dict[str, object] = {
        "n_dates": len(timestamps),
        "n_indicators": filtered_panel.shape[1],
        "n_windows": results.get('metadata', {}).get('n_windows', len(timestamps)),
        "profile": config.profile_name,
        "lenses_used": list(config.lenses),
        "step_months": config.step_months,
        "window_months": list(config.window_months),
        "elapsed_seconds": elapsed,
    }

    if timestamps:
        metadata["date_range"] = (timestamps[0], timestamps[-1])

    return TemporalResult(config=config, scores=scores, metadata=metadata)


def load_temporal_panel(
    db_path: Optional[str] = None,
    systems: Optional[Sequence[str]] = None,
    indicators: Optional[Sequence[str]] = None,
    min_history_years: int = 10,
) -> pd.DataFrame:
    """
    Load a panel from the database for temporal analysis.

    Args:
        db_path: Database path (uses default if None)
        systems: Filter by system IDs
        indicators: Filter by specific indicator IDs
        min_history_years: Minimum years of history required

    Returns:
        Wide-format DataFrame ready for temporal analysis
    """
    try:
        from panel.runtime_loader import load_panel, list_available_indicators
        from data.sql.db_connector import list_indicators
    except ImportError:
        raise ImportError(
            "Cannot import panel.runtime_loader or data.sql.db_connector. "
            "Ensure PRISM engine is properly installed."
        )

    # Get indicator list
    if indicators:
        indicator_list = list(indicators)
    else:
        # Get all available indicators, optionally filtered by system
        try:
            if systems:
                indicator_list = []
                for system in systems:
                    indicator_list.extend(list_indicators(system=system))
            else:
                indicator_list = list_available_indicators()
        except Exception:
            indicator_list = list_available_indicators()

    if not indicator_list:
        raise ValueError("No indicators available in database")

    # Calculate start date based on min_history_years
    min_start_date = pd.Timestamp.now() - pd.DateOffset(years=min_history_years)
    start_date_str = min_start_date.strftime('%Y-%m-%d')

    # Load panel
    panel = load_panel(
        indicator_names=indicator_list,
        start_date=start_date_str,
        skip_hvd_check=True,
    )

    return panel


def run_temporal_analysis_from_db(
    db_path: Optional[str],
    config: TemporalConfig,
) -> TemporalResult:
    """
    Load data from the indicator_values table and run temporal analysis.

    This function:
    - Uses panel.runtime_loader to build a panel matching config
    - Delegates to run_temporal_analysis_from_panel()

    Args:
        db_path: Database path (uses default if None)
        config: TemporalConfig with run parameters

    Returns:
        TemporalResult with scores DataFrame and metadata
    """
    panel = load_temporal_panel(
        db_path=db_path,
        systems=config.systems,
        indicators=config.indicators,
        min_history_years=config.min_history_years,
    )

    return run_temporal_analysis_from_panel(panel, config)


def quick_start_from_db(
    profile: str = "standard",
    db_path: Optional[str] = None,
) -> Tuple[TemporalResult, Dict[str, object]]:
    """
    Convenience helper: build config from profile, run from DB, and return
    (result, summary_dict).

    Args:
        profile: Profile name ('chromebook', 'standard', 'powerful')
        db_path: Database path (uses default if None)

    Returns:
        Tuple of (TemporalResult, summary dict)
    """
    cfg = build_temporal_config(profile=profile)
    result = run_temporal_analysis_from_db(db_path=db_path, config=cfg)

    summary = {
        "profile": cfg.profile_name,
        "start": result.scores.index.min() if not result.scores.empty else None,
        "end": result.scores.index.max() if not result.scores.empty else None,
        "n_dates": result.scores.shape[0],
        "n_lenses": len(cfg.lenses),
    }
    return result, summary


# ---------------------------------------------------------------------
# Module exports
# ---------------------------------------------------------------------

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
