"""
engine/orchestration/temporal_analysis.py

Unified temporal analysis orchestration for PRISM.

This module:
- Defines resolution presets (weekly / monthly / quarterly)
- Encodes resolution (frequency, window, stride) in a single config object
- Provides aggregation rules for different data types
- Provides a clean API for:
    * run_temporal_analysis_from_panel()
    * run_temporal_analysis_from_db()
    * quick_start_from_db()

Usage:
    from engine.orchestration.temporal_analysis import (
        build_temporal_config,
        run_temporal_analysis_from_db,
        RESOLUTION_PRESETS,
    )

    # Build config from resolution with optional overrides
    cfg = build_temporal_config(
        resolution="monthly",
        start_date="2010-01-01",
        systems=["finance"],
    )

    # Run analysis
    result = run_temporal_analysis_from_db(db_path=None, config=cfg)

    # Access scores
    print(result.scores.head())

Resolution Presets:
- weekly: W-FRI frequency, 52-week window, 2Y lookback (tactical monitoring)
- monthly: M frequency, 60-month window, 10Y lookback (DEFAULT - cycle positioning)
- quarterly: Q frequency, 40-quarter window, 30Y lookback (structural analysis)

Aggregation Rules:
- price/index/ratio/level: last (point-in-time)
- yield/rate/volatility: mean (average conditions)
- volume/flow/change: sum (total activity)
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Literal, Tuple, Any
from datetime import datetime

import numpy as np
import pandas as pd

from engine.utils.parallel import run_parallel, should_use_parallel, get_optimal_workers


Frequency = Literal["D", "W", "W-FRI", "M", "Q"]
Resolution = Literal["weekly", "monthly", "quarterly"]


# =============================================================================
# RESOLUTION PRESETS (replaces device profiles)
# =============================================================================

RESOLUTION_PRESETS: Dict[str, Dict[str, Any]] = {
    'weekly': {
        'frequency': 'W-FRI',
        'window_periods': 52,       # 1 year of weeks
        'stride_divisor': 4,        # 13-week stride (quarterly)
        'lookback_default': '2Y',
        'description': 'Tactical monitoring - recent regime shifts, short-term analysis',
    },
    'monthly': {
        'frequency': 'M',
        'window_periods': 60,       # 5 years of months
        'stride_divisor': 4,        # 15-month stride
        'lookback_default': '10Y',
        'description': 'Strategic analysis - cycle positioning, portfolio tilting (DEFAULT)',
    },
    'quarterly': {
        'frequency': 'Q',
        'window_periods': 40,       # 10 years of quarters
        'stride_divisor': 4,        # 10-quarter stride
        'lookback_default': '30Y',
        'description': 'Structural analysis - secular trends, long-term patterns',
    },
}


# =============================================================================
# AGGREGATION RULES
# =============================================================================

AGGREGATION_RULES: Dict[str, str] = {
    'price': 'last',
    'yield': 'mean',
    'rate': 'mean',
    'ratio': 'last',
    'volume': 'sum',
    'flow': 'sum',
    'volatility': 'mean',
    'index': 'last',
    'level': 'last',
    'change': 'sum',
    'default': 'last',
}


@dataclass
class TemporalProfile:
    """
    Defines a named temporal profile for convenience presets.

    DEPRECATED: Use RESOLUTION_PRESETS instead. This class is kept for
    backward compatibility with existing code using device profiles.
    """

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

    This can be constructed either from:
    - A resolution preset (weekly/monthly/quarterly) - PREFERRED
    - A legacy TemporalProfile + overrides - DEPRECATED
    - Direct construction by callers

    The resolution-based approach determines temporal parameters based on
    the analytical question, not hardware assumptions.
    """

    profile_name: str
    start_date: Optional[pd.Timestamp] = None
    end_date: Optional[pd.Timestamp] = None

    # Resolution-based configuration (NEW)
    resolution: Resolution = "monthly"
    frequency: Frequency = "M"
    window_periods: int = 60            # Number of periods in analysis window
    stride_periods: int = 15            # Number of periods between windows
    lookback: str = "10Y"               # Lookback period string

    # Legacy configuration (DEPRECATED - kept for backward compatibility)
    step_months: float = 1.0
    window_months: Sequence[int] = field(default_factory=lambda: (3, 6, 12, 24))

    # Lens configuration
    lenses: Sequence[str] = field(default_factory=lambda: ("magnitude", "pca", "influence", "clustering"))
    min_history_years: int = 10

    systems: Optional[Sequence[str]] = None     # e.g. ["finance"], ["climate"], ["finance", "climate"]
    indicators: Optional[Sequence[str]] = None  # specific indicator IDs (optional)

    # Parallelization settings
    use_parallel: bool = True                   # Enable parallel window processing
    max_workers: Optional[int] = None           # Max parallel workers (None = auto)

    @classmethod
    def from_resolution(
        cls,
        resolution: str = 'monthly',
        lookback: Optional[str] = None,
        window_periods: Optional[int] = None,
        stride_periods: Optional[int] = None,
        start_date: Optional[pd.Timestamp] = None,
        end_date: Optional[pd.Timestamp] = None,
        lenses: Optional[Sequence[str]] = None,
        systems: Optional[Sequence[str]] = None,
        indicators: Optional[Sequence[str]] = None,
        use_parallel: bool = True,
        max_workers: Optional[int] = None,
    ) -> "TemporalConfig":
        """
        Create a TemporalConfig from a resolution preset.

        This is the PREFERRED way to create configurations.

        Args:
            resolution: 'weekly', 'monthly', or 'quarterly'
            lookback: Override lookback period (e.g., '5Y', '10Y')
            window_periods: Override window size in periods
            stride_periods: Override stride size in periods
            start_date: Explicit start date (overrides lookback)
            end_date: Explicit end date
            lenses: Override lenses to use
            systems: Filter by system IDs
            indicators: Filter by specific indicator IDs
            use_parallel: Enable parallel processing
            max_workers: Max parallel workers

        Returns:
            TemporalConfig configured for the specified resolution
        """
        preset = RESOLUTION_PRESETS.get(resolution, RESOLUTION_PRESETS['monthly'])

        # Calculate stride from window if not specified
        final_window = window_periods or preset['window_periods']
        final_stride = stride_periods or (final_window // preset['stride_divisor'])

        # Default lenses - all available for analysis-driven approach
        default_lenses = (
            'magnitude', 'pca', 'influence', 'clustering',
            'decomposition', 'granger', 'dmd', 'mutual_info',
            'network', 'regime_switching', 'anomaly',
            'transfer_entropy', 'tda', 'wavelet'
        )

        return cls(
            profile_name=resolution,
            resolution=resolution,
            frequency=preset['frequency'],
            window_periods=final_window,
            stride_periods=final_stride,
            lookback=lookback or preset['lookback_default'],
            start_date=start_date,
            end_date=end_date,
            lenses=lenses or default_lenses,
            systems=systems,
            indicators=indicators,
            use_parallel=use_parallel,
            max_workers=max_workers,
            # Legacy fields for backward compatibility
            step_months=1.0,
            window_months=(3, 6, 12, 24),
            min_history_years=10,
        )


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
    resolution: Optional[str] = None,
    profile: Optional[str] = None,
    start_date: Optional[str | pd.Timestamp] = None,
    end_date: Optional[str | pd.Timestamp] = None,
    lookback: Optional[str] = None,
    window: Optional[int] = None,
    stride: Optional[int] = None,
    step_months: Optional[float] = None,
    window_months: Optional[Sequence[int]] = None,
    lenses: Optional[Sequence[str]] = None,
    systems: Optional[Sequence[str]] = None,
    indicators: Optional[Sequence[str]] = None,
    frequency: Optional[Frequency] = None,
    use_parallel: bool = True,
    max_workers: Optional[int] = None,
) -> TemporalConfig:
    """
    Construct a TemporalConfig from a resolution preset or legacy profile.

    This is the main entry point for the CLI runner & dashboard.

    PREFERRED: Use resolution parameter ('weekly', 'monthly', 'quarterly')
    DEPRECATED: Use profile parameter ('chromebook', 'standard', 'powerful')

    Args:
        resolution: Resolution preset ('weekly', 'monthly', 'quarterly')
        profile: DEPRECATED - Legacy profile name ('chromebook', 'standard', 'powerful')
        start_date: Optional start date (YYYY-MM-DD or Timestamp)
        end_date: Optional end date (YYYY-MM-DD or Timestamp)
        lookback: Lookback period (e.g., '5Y', '10Y') - used if start_date not set
        window: Window size in periods (overrides preset)
        stride: Stride size in periods (overrides preset)
        step_months: DEPRECATED - Override step size in months
        window_months: DEPRECATED - Override window sizes
        lenses: Override lenses to use
        systems: Filter by system IDs (e.g., ['finance', 'climate'])
        indicators: Filter by specific indicator IDs
        frequency: Override output frequency
        use_parallel: Enable parallel window processing (default: True)
        max_workers: Maximum parallel workers (None = auto-detect)

    Returns:
        TemporalConfig ready for use with run_* functions

    Raises:
        ValueError: If neither resolution nor profile is valid
    """

    def _to_ts(val: Optional[str | pd.Timestamp]) -> Optional[pd.Timestamp]:
        if val is None:
            return None
        if isinstance(val, pd.Timestamp):
            return val
        return pd.Timestamp(val)

    # NEW: Resolution-based configuration (preferred)
    if resolution is not None:
        if resolution not in RESOLUTION_PRESETS:
            raise ValueError(
                f"Unknown resolution: {resolution!r}. "
                f"Valid options: {list(RESOLUTION_PRESETS.keys())}"
            )

        return TemporalConfig.from_resolution(
            resolution=resolution,
            lookback=lookback,
            window_periods=window,
            stride_periods=stride,
            start_date=_to_ts(start_date),
            end_date=_to_ts(end_date),
            lenses=tuple(lenses) if lenses else None,
            systems=tuple(systems) if systems else None,
            indicators=tuple(indicators) if indicators else None,
            use_parallel=use_parallel,
            max_workers=max_workers,
        )

    # LEGACY: Profile-based configuration (deprecated)
    if profile is not None:
        # Map legacy device profiles to resolutions
        PROFILE_TO_RESOLUTION = {
            'chromebook': 'weekly',
            'standard': 'monthly',
            'powerful': 'monthly',
        }

        if profile in PROFILE_TO_RESOLUTION:
            warnings.warn(
                f"Device profile '{profile}' is deprecated. "
                f"Use resolution='{PROFILE_TO_RESOLUTION[profile]}' instead.",
                DeprecationWarning,
                stacklevel=2
            )

        if profile not in TEMPORAL_PROFILES:
            raise ValueError(f"Unknown temporal profile: {profile!r}")

        base = TEMPORAL_PROFILES[profile]

        cfg = TemporalConfig(
            profile_name=base.name,
            resolution='monthly',  # Default for legacy
            start_date=_to_ts(start_date),
            end_date=_to_ts(end_date),
            step_months=step_months if step_months is not None else base.step_months,
            window_months=window_months if window_months is not None else tuple(base.window_months),
            lenses=tuple(lenses) if lenses is not None else tuple(base.lenses),
            min_history_years=base.min_history_years,
            frequency=frequency if frequency is not None else base.frequency,
            systems=tuple(systems) if systems is not None else None,
            indicators=tuple(indicators) if indicators is not None else None,
            use_parallel=use_parallel,
            max_workers=max_workers,
        )
        return cfg

    # Default: Use monthly resolution
    return TemporalConfig.from_resolution(
        resolution='monthly',
        lookback=lookback,
        window_periods=window,
        stride_periods=stride,
        start_date=_to_ts(start_date),
        end_date=_to_ts(end_date),
        lenses=tuple(lenses) if lenses else None,
        systems=tuple(systems) if systems else None,
        indicators=tuple(indicators) if indicators else None,
        use_parallel=use_parallel,
        max_workers=max_workers,
    )


# ---------------------------------------------------------------------
# Orchestration API (wired to existing engine code)
# ---------------------------------------------------------------------


def _get_temporal_engine(
    panel: pd.DataFrame,
    lenses: Sequence[str],
    use_parallel: bool = True,
    max_workers: Optional[int] = None,
):
    """
    Get the existing TemporalEngine from engine_core.orchestration.

    Falls back to a minimal implementation if the import fails.

    Args:
        panel: Data panel for analysis.
        lenses: List of lens IDs to use.
        use_parallel: Enable parallel window processing.
        max_workers: Maximum parallel workers (None = auto).

    Returns:
        TemporalEngine instance.
    """
    try:
        from engine_core.orchestration.temporal_analysis import TemporalEngine
        return TemporalEngine(panel, lenses=list(lenses))
    except ImportError:
        # Minimal fallback implementation with parallel support
        return _MinimalTemporalEngine(
            panel,
            lenses=list(lenses),
            use_parallel=use_parallel,
            max_workers=max_workers,
        )


class _MinimalTemporalEngine:
    """
    Minimal temporal engine for when engine_core is not available.
    Uses basic rolling window analysis without lens dependencies.
    Supports parallel processing for improved performance.
    """

    def __init__(
        self,
        panel: pd.DataFrame,
        lenses: List[str] = None,
        use_parallel: bool = True,
        max_workers: Optional[int] = None,
    ):
        self.panel = panel.copy()
        self.lenses = lenses or ['magnitude', 'pca', 'influence']
        self.use_parallel = use_parallel
        self.max_workers = max_workers

    def run_rolling_analysis(
        self,
        window_days: int = 252,
        step_days: int = 21,
        min_window_pct: float = 0.8,
        progress_callback=None
    ) -> Dict[str, Any]:
        """
        Run basic rolling analysis.

        Uses parallel processing when:
        - use_parallel is True
        - System has sufficient resources
        - Number of windows exceeds threshold

        Args:
            window_days: Size of rolling window in days.
            step_days: Step size between windows in days.
            min_window_pct: Minimum fraction of window that must have data.
            progress_callback: Optional callback(completed, total) for progress.

        Returns:
            Dict with timestamps, rankings, importance, and metadata.
        """
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

        # Decide whether to use parallel execution
        n_windows = len(window_ends)
        use_parallel_exec = (
            self.use_parallel
            and should_use_parallel(n_windows, self.max_workers)
        )

        if use_parallel_exec:
            return self._run_parallel_windows(
                panel, window_ends, window_days, min_window_size, progress_callback
            )
        else:
            return self._run_serial_windows(
                panel, window_ends, window_days, min_window_size, progress_callback
            )

    def _run_serial_windows(
        self,
        panel: pd.DataFrame,
        window_ends: List[int],
        window_days: int,
        min_window_size: int,
        progress_callback=None
    ) -> Dict[str, Any]:
        """Run window analysis serially."""
        timestamps = []
        all_rankings = {col: [] for col in panel.columns}
        all_importance = {col: [] for col in panel.columns}

        for i, end_idx in enumerate(window_ends):
            result = _process_single_window(
                panel, end_idx, window_days, min_window_size
            )

            if result is not None:
                timestamps.append(result['timestamp'])
                for col in panel.columns:
                    all_rankings[col].append(result['ranks'].get(col, np.nan))
                    all_importance[col].append(result['importance'].get(col, np.nan))

            if progress_callback:
                progress_callback(i + 1, len(window_ends))

        return self._build_result(
            timestamps, all_rankings, all_importance, panel,
            window_days, len(window_ends) // max(1, len(timestamps) or 1) * len(timestamps)
        )

    def _run_parallel_windows(
        self,
        panel: pd.DataFrame,
        window_ends: List[int],
        window_days: int,
        min_window_size: int,
        progress_callback=None
    ) -> Dict[str, Any]:
        """
        Run window analysis in parallel.

        Note: Progress callback is not called during parallel execution
        to avoid multiprocessing complications.
        """
        # Prepare window specs for parallel processing
        window_specs = [
            (end_idx, window_days, min_window_size)
            for end_idx in window_ends
        ]

        # Create a picklable processor function using panel data
        # We need to use a module-level function for multiprocessing
        processor = _WindowProcessor(panel)

        # Run in parallel
        results = run_parallel(
            processor,
            window_specs,
            max_workers=self.max_workers,
            force_serial=False
        )

        # Aggregate results (maintaining order)
        timestamps = []
        all_rankings = {col: [] for col in panel.columns}
        all_importance = {col: [] for col in panel.columns}

        for result in results:
            if result is not None:
                timestamps.append(result['timestamp'])
                for col in panel.columns:
                    all_rankings[col].append(result['ranks'].get(col, np.nan))
                    all_importance[col].append(result['importance'].get(col, np.nan))

        # Call progress callback with final count
        if progress_callback:
            progress_callback(len(window_ends), len(window_ends))

        return self._build_result(
            timestamps, all_rankings, all_importance, panel,
            window_days, len(window_ends)
        )

    def _build_result(
        self,
        timestamps: List,
        all_rankings: Dict,
        all_importance: Dict,
        panel: pd.DataFrame,
        window_days: int,
        step_days: int,
    ) -> Dict[str, Any]:
        """Build the result dictionary."""
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
                'parallel_enabled': self.use_parallel,
                'workers': get_optimal_workers(self.max_workers) if self.use_parallel else 1,
            }
        }


class _WindowProcessor:
    """
    Callable class for processing windows in parallel.

    This is a class instead of a function to allow pickling
    with the panel data.
    """

    def __init__(self, panel: pd.DataFrame):
        self.panel = panel

    def __call__(self, window_spec: Tuple[int, int, int]) -> Optional[Dict[str, Any]]:
        end_idx, window_days, min_window_size = window_spec
        return _process_single_window(
            self.panel, end_idx, window_days, min_window_size
        )


def _process_single_window(
    panel: pd.DataFrame,
    end_idx: int,
    window_days: int,
    min_window_size: int,
) -> Optional[Dict[str, Any]]:
    """
    Process a single rolling window.

    This is a module-level function to support multiprocessing.

    Args:
        panel: The data panel.
        end_idx: End index of the window.
        window_days: Window size in days.
        min_window_size: Minimum required data points.

    Returns:
        Dict with timestamp, ranks, and importance, or None if insufficient data.
    """
    start_idx = max(0, end_idx - window_days)
    window_data = panel.iloc[start_idx:end_idx].dropna()

    if len(window_data) < min_window_size:
        return None

    timestamp = panel.index[end_idx]

    # Basic importance: variance-based ranking
    variance = window_data.var()
    normalized = (variance - variance.min()) / (variance.max() - variance.min() + 1e-10)
    ranks = normalized.rank(ascending=False)

    return {
        'timestamp': timestamp,
        'ranks': ranks.to_dict(),
        'importance': normalized.to_dict(),
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

    # Get temporal engine with parallel configuration
    temporal = _get_temporal_engine(
        filtered_panel,
        config.lenses,
        use_parallel=config.use_parallel,
        max_workers=config.max_workers,
    )

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

    # Extract parallel metadata from results
    result_metadata = results.get('metadata', {})

    metadata: Dict[str, object] = {
        "n_dates": len(timestamps),
        "n_indicators": filtered_panel.shape[1],
        "n_windows": result_metadata.get('n_windows', len(timestamps)),
        "profile": config.profile_name,
        "lenses_used": list(config.lenses),
        "step_months": config.step_months,
        "window_months": list(config.window_months),
        "elapsed_seconds": elapsed,
        "parallel_enabled": result_metadata.get('parallel_enabled', config.use_parallel),
        "workers": result_metadata.get('workers', 1),
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
# Aggregation Functions
# ---------------------------------------------------------------------

def get_indicator_data_type(indicator_name: str, registry: Optional[Dict] = None) -> str:
    """
    Look up data type from registry, with heuristic fallback.

    Args:
        indicator_name: Name of the indicator
        registry: Optional registry dict with 'data_type' fields

    Returns:
        Data type string (e.g., 'price', 'yield', 'volatility', 'default')
    """
    registry = registry or {}

    if indicator_name in registry:
        return registry[indicator_name].get('data_type', 'default')

    # Heuristic fallbacks based on indicator name patterns
    name_lower = indicator_name.lower()

    if any(x in name_lower for x in ['spy', 'qqq', 'etf', 'close', 'price', '_px', '_adj']):
        return 'price'
    if any(x in name_lower for x in ['dgs', 'yield', 'rate', 't10y', 't2y', 'tb3', 'fedfunds']):
        return 'yield'
    if any(x in name_lower for x in ['volume', 'vol_', '_vol', 'turnover']):
        return 'volume'
    if any(x in name_lower for x in ['vix', 'volatility', 'realized_vol', 'impl_vol']):
        return 'volatility'
    if any(x in name_lower for x in ['ratio', 'ma_', '_ratio', 'spread']):
        return 'ratio'
    if any(x in name_lower for x in ['cpi', 'ppi', 'pce', 'gdp', 'index']):
        return 'index'
    if any(x in name_lower for x in ['flow', 'netflow', 'inflow', 'outflow']):
        return 'flow'

    return 'default'


def aggregate_panel_to_frequency(
    panel: pd.DataFrame,
    target_frequency: str,
    registry: Optional[Dict] = None,
) -> pd.DataFrame:
    """
    Aggregate panel to target frequency using appropriate methods per column.

    This function applies the correct aggregation method based on data type:
    - price/index/ratio/level: last (point-in-time snapshot)
    - yield/rate/volatility: mean (average conditions)
    - volume/flow/change: sum (total activity)

    Args:
        panel: DataFrame with DatetimeIndex
        target_frequency: pandas frequency string ('W-FRI', 'M', 'Q')
        registry: Optional indicator registry for data type lookup

    Returns:
        Aggregated DataFrame at target frequency
    """
    registry = registry or {}

    if not isinstance(panel.index, pd.DatetimeIndex):
        panel = panel.copy()
        panel.index = pd.to_datetime(panel.index)

    aggregated = {}
    for col in panel.columns:
        data_type = get_indicator_data_type(col, registry)
        method = AGGREGATION_RULES.get(data_type, 'last')

        try:
            aggregated[col] = panel[col].resample(target_frequency).agg(method)
        except Exception:
            # Fallback to last if aggregation fails
            aggregated[col] = panel[col].resample(target_frequency).last()

    result = pd.DataFrame(aggregated)
    result = result.dropna(how='all')  # Remove empty rows
    return result


def detect_native_frequency(series: pd.Series) -> str:
    """
    Detect the native frequency of a series.

    Args:
        series: Pandas Series with DatetimeIndex

    Returns:
        Frequency string: 'daily', 'weekly', 'monthly', 'quarterly', 'annual', or 'unknown'
    """
    if len(series) < 2:
        return 'unknown'

    # Calculate median gap between observations
    gaps = series.dropna().index.to_series().diff().dropna()
    if len(gaps) == 0:
        return 'unknown'

    median_gap = gaps.median()

    if median_gap <= pd.Timedelta(days=1):
        return 'daily'
    elif median_gap <= pd.Timedelta(days=7):
        return 'weekly'
    elif median_gap <= pd.Timedelta(days=32):
        return 'monthly'
    elif median_gap <= pd.Timedelta(days=95):
        return 'quarterly'
    else:
        return 'annual'


def parse_lookback(lookback: str, end_date: Optional[pd.Timestamp] = None) -> pd.Timestamp:
    """
    Parse a lookback string into a start timestamp.

    Args:
        lookback: Lookback string (e.g., '5Y', '10Y', '2Y', '6M')
        end_date: End date to calculate from (default: now)

    Returns:
        Start timestamp
    """
    end_date = end_date or pd.Timestamp.now()

    lookback = lookback.upper().strip()

    if lookback.endswith('Y'):
        years = int(lookback[:-1])
        return end_date - pd.DateOffset(years=years)
    elif lookback.endswith('M'):
        months = int(lookback[:-1])
        return end_date - pd.DateOffset(months=months)
    elif lookback.endswith('W'):
        weeks = int(lookback[:-1])
        return end_date - pd.DateOffset(weeks=weeks)
    elif lookback.endswith('D'):
        days = int(lookback[:-1])
        return end_date - pd.DateOffset(days=days)
    else:
        # Try to parse as a date string
        return pd.Timestamp(lookback)


# ---------------------------------------------------------------------
# Module exports
# ---------------------------------------------------------------------

__all__ = [
    # Resolution presets (NEW - preferred)
    "RESOLUTION_PRESETS",
    "AGGREGATION_RULES",
    "Resolution",
    # Configuration classes
    "TemporalProfile",
    "TemporalConfig",
    "TemporalResult",
    # Legacy profiles (DEPRECATED)
    "TEMPORAL_PROFILES",
    # Builder function
    "build_temporal_config",
    # Analysis functions
    "run_temporal_analysis_from_panel",
    "run_temporal_analysis_from_db",
    "load_temporal_panel",
    "quick_start_from_db",
    # Aggregation functions (NEW)
    "get_indicator_data_type",
    "aggregate_panel_to_frequency",
    "detect_native_frequency",
    "parse_lookback",
]
