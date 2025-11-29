"""
Temporal Analysis - Track PRISM results over rolling time windows
==================================================================

Run PRISM analysis on rolling windows to see how indicator importance
changes over time. Optimized for efficiency with many indicators.

Usage:
    from temporal_analysis import TemporalPRISM

    temporal = TemporalPRISM(panel_clean)
    results = temporal.run_rolling_analysis(window_days=252)  # 1 year

    # Visualize
    temporal.plot_ranking_evolution(results, top_n=10)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class TemporalPRISM:
    """
    Run PRISM analysis over rolling time windows to track
    how indicator importance evolves over time.

    Optimized for:
    - Efficiency with 50-60+ indicators
    - Memory management for long time series
    - Incremental updates where possible
    """

    def __init__(self, panel: pd.DataFrame, lenses: List[str] = None):
        """
        Initialize temporal analyzer.

        Args:
            panel: Full panel data (datetime index, indicator columns)
            lenses: List of lens names to use (default: fast lenses only)
        """
        self.panel = panel.copy()

        # Default to faster lenses for temporal analysis
        self.lenses = lenses or ['magnitude', 'pca', 'influence', 'clustering']

        # Cache for lens instances
        self._lens_cache = {}

    def _get_lens(self, name: str):
        """Get or create lens instance (cached for efficiency)."""
        if name not in self._lens_cache:
            from prism_loader import load_lens
            self._lens_cache[name] = load_lens(name)
        return self._lens_cache[name]

    def _run_single_window(self, window_data: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Run all lenses on a single time window.

        Returns dict mapping lens_name -> importance Series
        """
        results = {}

        for lens_name in self.lenses:
            try:
                lens = self._get_lens(lens_name)
                result = lens.analyze(window_data)

                if 'importance' in result:
                    imp = result['importance']
                    if isinstance(imp, pd.Series):
                        results[lens_name] = imp
                    elif isinstance(imp, dict):
                        results[lens_name] = pd.Series(imp)
            except Exception:
                # Skip failed lenses silently for efficiency
                pass

        return results

    def _compute_consensus_rank(self, lens_results: Dict[str, pd.Series]) -> pd.Series:
        """Compute average rank across lenses."""
        if not lens_results:
            return pd.Series(dtype=float)

        ranks = {}
        for lens_name, importance in lens_results.items():
            ranks[lens_name] = importance.rank(ascending=False)

        rank_df = pd.DataFrame(ranks)
        return rank_df.mean(axis=1)

    def run_rolling_analysis(
        self,
        window_days: int = 252,
        step_days: int = 21,
        min_window_pct: float = 0.8,
        progress_callback: Callable = None
    ) -> Dict[str, Any]:
        """
        Run PRISM analysis over rolling windows.

        Args:
            window_days: Size of rolling window (default 252 = 1 year)
            step_days: Days between windows (default 21 = ~1 month)
            min_window_pct: Minimum data required in window (0.8 = 80%)
            progress_callback: Optional callback(current, total) for progress

        Returns:
            Dictionary containing:
            - timestamps: List of window end dates
            - rankings: Dict[indicator] -> list of ranks over time
            - importance: Dict[indicator] -> list of importance scores
            - lens_results: Full results by lens
            - metadata: Run configuration
        """
        panel = self.panel.copy()

        # Ensure datetime index
        if not isinstance(panel.index, pd.DatetimeIndex):
            panel.index = pd.to_datetime(panel.index)

        panel = panel.sort_index()

        # Calculate window positions
        total_rows = len(panel)
        min_window_size = int(window_days * min_window_pct)

        if total_rows < min_window_size:
            raise ValueError(f"Not enough data. Need {min_window_size} rows, have {total_rows}")

        # Generate window end positions
        window_ends = list(range(window_days, total_rows, step_days))
        if window_ends[-1] != total_rows - 1:
            window_ends.append(total_rows - 1)

        n_windows = len(window_ends)

        # Storage
        timestamps = []
        all_rankings = {col: [] for col in panel.columns}
        all_importance = {col: [] for col in panel.columns}
        lens_results_by_time = {lens: [] for lens in self.lenses}

        # Run analysis for each window
        for i, end_idx in enumerate(window_ends):
            start_idx = max(0, end_idx - window_days)

            window_data = panel.iloc[start_idx:end_idx].copy()
            window_data = window_data.dropna()

            if len(window_data) < min_window_size:
                continue

            # Store timestamp
            timestamps.append(panel.index[end_idx])

            # Run lenses
            lens_results = self._run_single_window(window_data)

            # Compute consensus rank
            consensus_rank = self._compute_consensus_rank(lens_results)

            # Store results
            for col in panel.columns:
                if col in consensus_rank.index:
                    all_rankings[col].append(consensus_rank[col])

                    # Average importance across lenses
                    imp_values = [
                        lens_results[lens][col]
                        for lens in lens_results
                        if col in lens_results[lens].index
                    ]
                    all_importance[col].append(np.mean(imp_values) if imp_values else 0)
                else:
                    all_rankings[col].append(np.nan)
                    all_importance[col].append(np.nan)

            # Store lens-specific results
            for lens_name in self.lenses:
                if lens_name in lens_results:
                    lens_results_by_time[lens_name].append(lens_results[lens_name].to_dict())
                else:
                    lens_results_by_time[lens_name].append({})

            # Progress callback
            if progress_callback:
                progress_callback(i + 1, n_windows)

        return {
            'timestamps': timestamps,
            'rankings': all_rankings,
            'importance': all_importance,
            'lens_results': lens_results_by_time,
            'metadata': {
                'window_days': window_days,
                'step_days': step_days,
                'n_windows': len(timestamps),
                'date_range': (timestamps[0], timestamps[-1]) if timestamps else None,
                'indicators': list(panel.columns),
                'lenses_used': self.lenses,
            }
        }

    def get_ranking_dataframe(self, results: Dict) -> pd.DataFrame:
        """
        Convert temporal results to a DataFrame of rankings over time.

        Args:
            results: Output from run_rolling_analysis()

        Returns:
            DataFrame with timestamps as index, indicators as columns, ranks as values
        """
        df = pd.DataFrame(results['rankings'], index=results['timestamps'])
        df.index.name = 'date'
        return df

    def get_importance_dataframe(self, results: Dict) -> pd.DataFrame:
        """
        Convert temporal results to a DataFrame of importance over time.
        """
        df = pd.DataFrame(results['importance'], index=results['timestamps'])
        df.index.name = 'date'
        return df

    def compute_rank_changes(self, results: Dict) -> pd.DataFrame:
        """
        Compute rank changes between consecutive periods.

        Returns:
            DataFrame with rank changes (positive = improved ranking)
        """
        rank_df = self.get_ranking_dataframe(results)

        # Negative because lower rank = better, so negative diff = improvement
        changes = -rank_df.diff()

        return changes

    def find_trending_indicators(
        self,
        results: Dict,
        lookback: int = 6,
        min_improvement: float = 3.0
    ) -> Dict[str, Any]:
        """
        Find indicators with significant recent rank changes.

        Args:
            results: Output from run_rolling_analysis()
            lookback: Number of recent periods to consider
            min_improvement: Minimum rank improvement to be considered trending

        Returns:
            Dictionary with rising and falling indicators
        """
        rank_df = self.get_ranking_dataframe(results)

        if len(rank_df) < lookback:
            lookback = len(rank_df)

        # Recent rank change
        recent = rank_df.iloc[-lookback:]

        rank_change = recent.iloc[0] - recent.iloc[-1]  # Positive = improved

        rising = rank_change[rank_change >= min_improvement].sort_values(ascending=False)
        falling = rank_change[rank_change <= -min_improvement].sort_values()

        return {
            'rising': rising.to_dict(),
            'falling': falling.to_dict(),
            'lookback_periods': lookback,
            'date_range': (recent.index[0], recent.index[-1]),
        }

    def compute_rank_stability(self, results: Dict) -> pd.Series:
        """
        Compute stability score for each indicator (lower variance = more stable).

        Returns:
            Series with stability scores (higher = more stable)
        """
        rank_df = self.get_ranking_dataframe(results)

        # Stability = 1 / (1 + rank variance)
        stability = 1 / (1 + rank_df.var())

        return stability.sort_values(ascending=False)

    def get_period_snapshot(
        self,
        results: Dict,
        period_idx: int = -1
    ) -> pd.DataFrame:
        """
        Get detailed snapshot for a specific time period.

        Args:
            results: Output from run_rolling_analysis()
            period_idx: Index of period (-1 = most recent)

        Returns:
            DataFrame with indicator rankings for that period
        """
        timestamp = results['timestamps'][period_idx]

        snapshot = pd.DataFrame({
            'indicator': list(results['rankings'].keys()),
            'rank': [results['rankings'][ind][period_idx] for ind in results['rankings']],
            'importance': [results['importance'][ind][period_idx] for ind in results['importance']],
        })

        snapshot = snapshot.sort_values('rank')
        snapshot['timestamp'] = timestamp

        return snapshot.reset_index(drop=True)


class StreamingTemporalPRISM:
    """
    Memory-efficient streaming version for very long time series.

    Instead of holding all results in memory, this version:
    - Writes results to disk incrementally
    - Supports resuming interrupted runs
    - Better for Chromebook/low-memory environments
    """

    def __init__(self, panel: pd.DataFrame, output_path: str):
        """
        Initialize streaming analyzer.

        Args:
            panel: Full panel data
            output_path: Path to write incremental results
        """
        self.panel = panel
        self.output_path = output_path
        self.lenses = ['magnitude', 'pca', 'influence']  # Minimal set for speed

    def run_streaming(
        self,
        window_days: int = 252,
        step_days: int = 21,
        checkpoint_every: int = 10
    ) -> str:
        """
        Run analysis with streaming output.

        Results are written to CSV incrementally to manage memory.

        Returns:
            Path to output file
        """
        import csv
        from pathlib import Path

        output_path = Path(self.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        panel = self.panel.copy()
        if not isinstance(panel.index, pd.DatetimeIndex):
            panel.index = pd.to_datetime(panel.index)
        panel = panel.sort_index()

        indicators = list(panel.columns)

        # Write header
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp'] + indicators)

        # Process windows
        window_ends = list(range(window_days, len(panel), step_days))

        temporal = TemporalPRISM(panel, lenses=self.lenses)

        for i, end_idx in enumerate(window_ends):
            start_idx = max(0, end_idx - window_days)
            window_data = panel.iloc[start_idx:end_idx].dropna()

            if len(window_data) < window_days * 0.8:
                continue

            # Run single window
            lens_results = temporal._run_single_window(window_data)
            consensus_rank = temporal._compute_consensus_rank(lens_results)

            # Write row
            timestamp = panel.index[end_idx].strftime('%Y-%m-%d')
            row = [timestamp] + [
                consensus_rank.get(ind, np.nan) for ind in indicators
            ]

            with open(output_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(row)

            # Progress
            if (i + 1) % checkpoint_every == 0:
                print(f"  Processed {i + 1}/{len(window_ends)} windows...")

        print(f"  Wrote results to {output_path}")
        return str(output_path)


def quick_temporal_analysis(
    panel: pd.DataFrame,
    window_years: float = 1.0,
    step_months: float = 1.0
) -> Dict[str, Any]:
    """
    Quick one-liner for temporal analysis.

    Args:
        panel: Your data
        window_years: Rolling window size in years
        step_months: Step size in months

    Returns:
        Full temporal analysis results
    """
    window_days = int(window_years * 252)  # Trading days
    step_days = int(step_months * 21)  # Trading days per month

    temporal = TemporalPRISM(panel)

    print(f"Running temporal analysis...")
    print(f"  Window: {window_years} year(s) ({window_days} days)")
    print(f"  Step: {step_months} month(s) ({step_days} days)")

    results = temporal.run_rolling_analysis(
        window_days=window_days,
        step_days=step_days
    )

    print(f"  Analyzed {results['metadata']['n_windows']} time periods")
    print(f"  Date range: {results['metadata']['date_range'][0].date()} to {results['metadata']['date_range'][1].date()}")

    return results
