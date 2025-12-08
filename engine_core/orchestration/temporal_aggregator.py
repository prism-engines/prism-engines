#!/usr/bin/env python3
"""
Temporal Aggregator - Aggregate Granular Results
=======================================================

Aggregates 1-year (or any increment) temporal results into meaningful
regime periods or fixed time buckets for presentation and analysis.

Also provides frequency aggregation utilities for converting data between
different time resolutions (daily -> weekly -> monthly -> quarterly).

Usage (command line):
    # After running 1-year analysis
    python temporal_aggregator.py --group regime

    # By fixed periods
    python temporal_aggregator.py --group decade
    python temporal_aggregator.py --group 5year

    # Custom database path
    python temporal_aggregator.py --group regime --db path/to/temporal.db

Usage (Python):
    from temporal_aggregator import TemporalAggregator
    agg = TemporalAggregator(db_path='output/temporal/temporal.db')
    regime_summary = agg.aggregate_by_regime()
    decade_avg = agg.aggregate_by_period(period=10)

    # NEW: Frequency aggregation
    from temporal_aggregator import aggregate_panel_to_frequency
    monthly_panel = aggregate_panel_to_frequency(daily_panel, 'M')

Output (in output/temporal/summary/):
    - regime_summary.csv     : Average ranks by regime period
    - decade_averages.csv    : Average ranks by decade
    - 5year_averages.csv     : Average ranks by 5-year period
    - plots/
        - regime_heatmap.png     : Heatmap with regimes as columns
        - regime_stability.png   : Correlation stability across regimes
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# AGGREGATION RULES FOR FREQUENCY CONVERSION
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


# =============================================================================
# FREQUENCY AGGREGATION FUNCTIONS
# =============================================================================

def get_indicator_data_type(indicator_name: str, registry: Optional[Dict] = None) -> str:
    """
    Look up data type from registry, default to 'default' with heuristic fallbacks.

    Args:
        indicator_name: Name of the indicator
        registry: Optional indicator registry with 'data_type' fields

    Returns:
        Data type string for aggregation lookup
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

    IMPORTANT: This only aggregates DOWN to slower frequencies.
    Never interpolate UP to faster frequencies (e.g., monthly -> daily).

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


def detect_panel_frequencies(panel: pd.DataFrame) -> Dict[str, str]:
    """
    Detect the native frequency of each column in a panel.

    Args:
        panel: DataFrame with DatetimeIndex

    Returns:
        Dict mapping column name to detected frequency
    """
    frequencies = {}
    for col in panel.columns:
        frequencies[col] = detect_native_frequency(panel[col])
    return frequencies


def get_slowest_frequency(frequencies: Dict[str, str]) -> str:
    """
    Get the slowest (lowest) frequency from a set of frequencies.

    Args:
        frequencies: Dict mapping indicator names to frequency strings

    Returns:
        The slowest frequency found
    """
    freq_order = ['annual', 'quarterly', 'monthly', 'weekly', 'daily', 'unknown']

    slowest_idx = len(freq_order) - 1
    for freq in frequencies.values():
        if freq in freq_order:
            idx = freq_order.index(freq)
            if idx < slowest_idx:
                slowest_idx = idx

    return freq_order[slowest_idx]


def determine_analysis_frequency(
    panel: pd.DataFrame,
    target_resolution: str = 'monthly',
) -> str:
    """
    Determine the appropriate analysis frequency for a panel.

    The analysis frequency should be the SLOWER of:
    - The target resolution (weekly/monthly/quarterly)
    - The slowest native frequency in the data

    This ensures we never upsample data.

    Args:
        panel: DataFrame with DatetimeIndex
        target_resolution: Desired resolution ('weekly', 'monthly', 'quarterly')

    Returns:
        pandas frequency string ('W-FRI', 'M', 'Q')
    """
    resolution_to_freq = {
        'weekly': 'W-FRI',
        'monthly': 'M',
        'quarterly': 'Q',
    }

    resolution_to_native = {
        'weekly': 'weekly',
        'monthly': 'monthly',
        'quarterly': 'quarterly',
    }

    # Detect panel frequencies
    native_frequencies = detect_panel_frequencies(panel)
    slowest_native = get_slowest_frequency(native_frequencies)

    # Map to comparable strings
    target_native = resolution_to_native.get(target_resolution, 'monthly')

    # Frequency ordering (slower to faster)
    freq_order = ['annual', 'quarterly', 'monthly', 'weekly', 'daily']

    target_idx = freq_order.index(target_native) if target_native in freq_order else 2
    slowest_idx = freq_order.index(slowest_native) if slowest_native in freq_order else 2

    # Use the slower of the two
    if slowest_idx < target_idx:
        # Data is slower than target, use data's native frequency
        native_to_pandas = {
            'annual': 'A',
            'quarterly': 'Q',
            'monthly': 'M',
            'weekly': 'W-FRI',
            'daily': 'D',
        }
        return native_to_pandas.get(slowest_native, 'M')

    # Target is slower or equal, use target frequency
    return resolution_to_freq.get(target_resolution, 'M')


# =============================================================================
# REGIME DEFINITIONS (editable)
# =============================================================================

REGIMES = {
    'inflation_era': (1970, 1982),
    'great_moderation': (1982, 2000),
    'bubble_era': (2000, 2008),
    'qe_era': (2008, 2015),
    'normalization': (2015, 2020),
    'post_covid': (2020, 2025),
}

# Human-readable regime labels for display
REGIME_LABELS = {
    'inflation_era': 'Inflation Era (1970-1982)',
    'great_moderation': 'Great Moderation (1982-2000)',
    'bubble_era': 'Bubble Era (2000-2008)',
    'qe_era': 'QE Era (2008-2015)',
    'normalization': 'Normalization (2015-2020)',
    'post_covid': 'Post-COVID (2020-2025)',
}


# =============================================================================
# TEMPORAL AGGREGATOR CLASS
# =============================================================================

class TemporalAggregator:
    """
    Aggregate granular temporal results into regime-based or
    fixed-period summaries.
    """

    def __init__(
        self,
        db_path: str = None,
        output_dir: str = None,
        regimes: Dict[str, Tuple[int, int]] = None,
    ):
        """
        Initialize aggregator.

        Args:
            db_path: Path to temporal.db
            output_dir: Output directory for aggregated results
            regimes: Custom regime definitions (default: REGIMES constant)
        """
        self.project_root = PROJECT_ROOT
        self.db_path = Path(db_path) if db_path else self.project_root / "output" / "temporal" / "temporal.db"
        self.output_dir = Path(output_dir) if output_dir else self.project_root / "output" / "temporal" / "summary"
        self.regimes = regimes or REGIMES

        # Storage
        self.windows_df: Optional[pd.DataFrame] = None
        self.consensus_df: Optional[pd.DataFrame] = None

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load data from SQLite database."""
        from utils.db_manager import TemporalDB

        if not self.db_path.exists():
            raise FileNotFoundError(f"Database not found: {self.db_path}")

        print(f"Loading data from {self.db_path}...")

        db = TemporalDB(str(self.db_path))

        # Get all windows
        self.windows_df = db.query_all_windows()
        print(f"  Found {len(self.windows_df)} windows")

        # Get full consensus data via SQL
        import sqlite3
        with sqlite3.connect(self.db_path) as conn:
            self.consensus_df = pd.read_sql_query("""
                SELECT
                    w.start_year,
                    w.end_year,
                    w.start_year || '-' || w.end_year as window,
                    i.name as indicator,
                    i.category,
                    c.consensus_rank,
                    c.consensus_score,
                    c.n_lenses
                FROM consensus c
                JOIN windows w ON c.window_id = w.id
                JOIN indicators i ON c.indicator_id = i.id
                ORDER BY w.start_year, c.consensus_rank
            """, conn)

        print(f"  Loaded {len(self.consensus_df)} consensus rankings")

        return self.windows_df, self.consensus_df

    def get_window_year(self, window: str) -> int:
        """Extract start year from window label like '2005-2010'."""
        return int(window.split('-')[0])

    def assign_regime(self, year: int) -> Optional[str]:
        """Assign a year to a regime period."""
        for regime_name, (start, end) in self.regimes.items():
            if start <= year < end:
                return regime_name
        return None

    def assign_period(self, year: int, period: int) -> str:
        """
        Assign a year to a fixed period bucket.

        Args:
            year: Year to assign
            period: Period size (e.g., 5 for 5-year, 10 for decade)

        Returns:
            Period label like '2000-2005' or '2000-2010'
        """
        start = (year // period) * period
        end = start + period
        return f"{start}-{end}"

    def aggregate_by_regime(self, verbose: bool = True) -> pd.DataFrame:
        """
        Aggregate consensus rankings by regime period.

        Returns:
            DataFrame with average ranks per indicator per regime
        """
        if self.consensus_df is None:
            self.load_data()

        df = self.consensus_df.copy()

        # Assign regime to each window
        df['regime'] = df['start_year'].apply(self.assign_regime)

        # Filter out windows that don't fall into any regime
        df = df[df['regime'].notna()]

        if verbose:
            regimes_found = df['regime'].unique()
            print(f"\nAggregating by regime ({len(regimes_found)} regimes found):")
            for r in regimes_found:
                count = len(df[df['regime'] == r]['window'].unique())
                print(f"  {r}: {count} windows")

        # Group by indicator and regime, compute mean rank
        agg = df.groupby(['indicator', 'regime', 'category']).agg({
            'consensus_rank': 'mean',
            'consensus_score': 'mean',
            'n_lenses': 'mean',
        }).reset_index()

        agg.columns = ['indicator', 'regime', 'category', 'avg_rank', 'avg_score', 'avg_lenses']

        # Pivot to get regime columns
        pivot = agg.pivot(
            index='indicator',
            columns='regime',
            values='avg_rank'
        )

        # Add overall average
        pivot['overall_avg'] = pivot.mean(axis=1)
        pivot = pivot.sort_values('overall_avg')

        # Reorder columns to match regime chronology
        regime_order = sorted(self.regimes.keys(), key=lambda r: self.regimes[r][0])
        ordered_cols = [r for r in regime_order if r in pivot.columns] + ['overall_avg']
        pivot = pivot[ordered_cols]

        if verbose:
            print(f"\nTop 10 indicators by overall average rank:")
            print(pivot.head(10)[['overall_avg']].round(1).to_string())

        return pivot

    def aggregate_by_period(
        self,
        period: int = 10,
        verbose: bool = True
    ) -> pd.DataFrame:
        """
        Aggregate consensus rankings by fixed time period.

        Args:
            period: Period size in years (default: 10 for decade)
            verbose: Print progress

        Returns:
            DataFrame with average ranks per indicator per period
        """
        if self.consensus_df is None:
            self.load_data()

        df = self.consensus_df.copy()

        # Assign period to each window
        df['period'] = df['start_year'].apply(lambda y: self.assign_period(y, period))

        if verbose:
            periods_found = sorted(df['period'].unique())
            print(f"\nAggregating by {period}-year periods ({len(periods_found)} periods found):")
            for p in periods_found:
                count = len(df[df['period'] == p]['window'].unique())
                print(f"  {p}: {count} windows")

        # Group by indicator and period, compute mean rank
        agg = df.groupby(['indicator', 'period', 'category']).agg({
            'consensus_rank': 'mean',
            'consensus_score': 'mean',
            'n_lenses': 'mean',
        }).reset_index()

        agg.columns = ['indicator', 'period', 'category', 'avg_rank', 'avg_score', 'avg_lenses']

        # Pivot to get period columns
        pivot = agg.pivot(
            index='indicator',
            columns='period',
            values='avg_rank'
        )

        # Add overall average
        pivot['overall_avg'] = pivot.mean(axis=1)
        pivot = pivot.sort_values('overall_avg')

        # Sort columns chronologically
        period_cols = sorted([c for c in pivot.columns if c != 'overall_avg'])
        pivot = pivot[period_cols + ['overall_avg']]

        if verbose:
            print(f"\nTop 10 indicators by overall average rank:")
            print(pivot.head(10)[['overall_avg']].round(1).to_string())

        return pivot

    def compute_regime_stability(self) -> pd.DataFrame:
        """
        Compute Spearman correlation between adjacent regimes.

        Returns:
            DataFrame with correlations between regime transitions
        """
        from scipy.stats import spearmanr

        if self.consensus_df is None:
            self.load_data()

        # Get regime pivot
        regime_pivot = self.aggregate_by_regime(verbose=False)

        # Get regime order
        regime_order = sorted(self.regimes.keys(), key=lambda r: self.regimes[r][0])
        regime_cols = [r for r in regime_order if r in regime_pivot.columns]

        stability = []

        for i in range(len(regime_cols) - 1):
            r1, r2 = regime_cols[i], regime_cols[i + 1]

            # Get common indicators with valid values
            mask = regime_pivot[[r1, r2]].notna().all(axis=1)
            ranks1 = regime_pivot.loc[mask, r1]
            ranks2 = regime_pivot.loc[mask, r2]

            if len(ranks1) >= 5:
                corr, p_value = spearmanr(ranks1, ranks2)

                stability.append({
                    'regime_from': r1,
                    'regime_to': r2,
                    'transition_label': f"{REGIME_LABELS.get(r1, r1)} -> {REGIME_LABELS.get(r2, r2)}",
                    'spearman_corr': corr,
                    'p_value': p_value,
                    'n_indicators': len(ranks1),
                })

        return pd.DataFrame(stability)

    def generate_heatmap(
        self,
        pivot_df: pd.DataFrame,
        output_path: Path,
        title: str = "Indicator Rankings",
        top_n: int = 20,
    ) -> Optional[Path]:
        """
        Generate a heatmap visualization of rankings.

        Args:
            pivot_df: Pivoted DataFrame (indicators × regimes/periods)
            output_path: Path to save PNG
            title: Plot title
            top_n: Number of top indicators to show

        Returns:
            Path to saved plot or None if matplotlib unavailable
        """
        try:
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError:
            print("Warning: matplotlib/seaborn not available, skipping heatmap")
            return None

        # Select top N indicators
        if 'overall_avg' in pivot_df.columns:
            plot_df = pivot_df.head(top_n).drop(columns=['overall_avg'], errors='ignore')
        else:
            plot_df = pivot_df.head(top_n)

        # Create figure
        fig_height = max(8, len(plot_df) * 0.4)
        fig_width = max(10, len(plot_df.columns) * 1.2)
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))

        # Generate heatmap (lower rank = better = darker color)
        sns.heatmap(
            plot_df,
            annot=True,
            fmt='.0f',
            cmap='RdYlGn_r',  # Red=high rank (bad), Green=low rank (good)
            ax=ax,
            cbar_kws={'label': 'Consensus Rank'},
            linewidths=0.5,
        )

        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Period', fontsize=12)
        ax.set_ylabel('Indicator', fontsize=12)

        plt.tight_layout()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

        return output_path

    def generate_stability_plot(
        self,
        stability_df: pd.DataFrame,
        output_path: Path,
    ) -> Optional[Path]:
        """
        Generate a plot of regime stability (correlation over time).

        Args:
            stability_df: DataFrame from compute_regime_stability()
            output_path: Path to save PNG

        Returns:
            Path to saved plot or None if matplotlib unavailable
        """
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except ImportError:
            print("Warning: matplotlib not available, skipping stability plot")
            return None

        if stability_df.empty:
            print("Warning: No stability data to plot")
            return None

        fig, ax = plt.subplots(figsize=(10, 6))

        x = range(len(stability_df))
        y = stability_df['spearman_corr']
        labels = [f"{row['regime_from']}\n->\n{row['regime_to']}"
                  for _, row in stability_df.iterrows()]

        bars = ax.bar(x, y, color=['green' if v > 0.7 else 'orange' if v > 0.5 else 'red' for v in y])

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=0, ha='center', fontsize=9)
        ax.set_ylabel('Spearman Correlation', fontsize=12)
        ax.set_xlabel('Regime Transition', fontsize=12)
        ax.set_title('Ranking Stability Across Regime Transitions', fontsize=14, fontweight='bold')
        ax.axhline(y=0.7, color='gray', linestyle='--', alpha=0.5, label='High stability (0.7)')
        ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5, label='Moderate stability (0.5)')
        ax.legend(loc='lower right')
        ax.set_ylim(0, 1)

        # Add correlation values on bars
        for bar, corr in zip(bars, y):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{corr:.2f}', ha='center', va='bottom', fontsize=10)

        plt.tight_layout()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

        return output_path

    def run(
        self,
        group_by: str = 'regime',
        generate_plots: bool = True,
        verbose: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        Run full aggregation pipeline.

        Args:
            group_by: 'regime', 'decade', '5year', or custom integer
            generate_plots: Generate heatmap and stability plots
            verbose: Print progress

        Returns:
            Dict with aggregated DataFrames
        """
        start_time = datetime.now()

        if verbose:
            print("=" * 60)
            print("TEMPORAL AGGREGATOR")
            print("=" * 60)
            print(f"Database: {self.db_path}")
            print(f"Group by: {group_by}")
            print()

        # Load data
        self.load_data()

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        plots_dir = self.output_dir / "plots"
        plots_dir.mkdir(exist_ok=True)

        results = {}

        # Aggregate based on group_by parameter
        if group_by == 'regime':
            summary = self.aggregate_by_regime(verbose=verbose)
            output_file = self.output_dir / "regime_summary.csv"
            summary.to_csv(output_file)
            results['regime_summary'] = summary
            if verbose:
                print(f"\nSaved: {output_file}")

            # Stability analysis
            stability = self.compute_regime_stability()
            stability_file = self.output_dir / "regime_stability.csv"
            stability.to_csv(stability_file, index=False)
            results['regime_stability'] = stability
            if verbose:
                print(f"Saved: {stability_file}")

            if generate_plots:
                # Heatmap
                heatmap_path = plots_dir / "regime_heatmap.png"
                if self.generate_heatmap(summary, heatmap_path, "Rankings by Regime"):
                    if verbose:
                        print(f"Saved: {heatmap_path}")

                # Stability plot
                stability_plot_path = plots_dir / "regime_stability.png"
                if self.generate_stability_plot(stability, stability_plot_path):
                    if verbose:
                        print(f"Saved: {stability_plot_path}")

        elif group_by == 'decade':
            summary = self.aggregate_by_period(period=10, verbose=verbose)
            output_file = self.output_dir / "decade_averages.csv"
            summary.to_csv(output_file)
            results['decade_averages'] = summary
            if verbose:
                print(f"\nSaved: {output_file}")

            if generate_plots:
                heatmap_path = plots_dir / "decade_heatmap.png"
                if self.generate_heatmap(summary, heatmap_path, "Rankings by Decade"):
                    if verbose:
                        print(f"Saved: {heatmap_path}")

        elif group_by == '5year':
            summary = self.aggregate_by_period(period=5, verbose=verbose)
            output_file = self.output_dir / "5year_averages.csv"
            summary.to_csv(output_file)
            results['5year_averages'] = summary
            if verbose:
                print(f"\nSaved: {output_file}")

            if generate_plots:
                heatmap_path = plots_dir / "5year_heatmap.png"
                if self.generate_heatmap(summary, heatmap_path, "Rankings by 5-Year Period"):
                    if verbose:
                        print(f"Saved: {heatmap_path}")

        else:
            # Try to parse as integer period
            try:
                period = int(group_by)
                summary = self.aggregate_by_period(period=period, verbose=verbose)
                output_file = self.output_dir / f"{period}year_averages.csv"
                summary.to_csv(output_file)
                results[f'{period}year_averages'] = summary
                if verbose:
                    print(f"\nSaved: {output_file}")
            except ValueError:
                raise ValueError(f"Unknown group_by value: {group_by}. Use 'regime', 'decade', '5year', or an integer.")

        if verbose:
            elapsed = datetime.now() - start_time
            print()
            print("=" * 60)
            print(f"COMPLETE - {elapsed}")
            print("=" * 60)

        return results


# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Temporal Aggregator - Aggregate granular temporal results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # After running 1-year analysis
  python temporal_aggregator.py --group regime

  # By fixed periods
  python temporal_aggregator.py --group decade
  python temporal_aggregator.py --group 5year

  # Custom period
  python temporal_aggregator.py --group 3

  # Custom database path
  python temporal_aggregator.py --group regime --db path/to/temporal.db

Output:
  Saved to output/temporal/summary/
        """
    )

    parser.add_argument(
        '--group', '-g',
        type=str,
        default='regime',
        help='Grouping method: regime, decade, 5year, or integer for custom period (default: regime)'
    )

    parser.add_argument(
        '--db',
        type=str,
        default=None,
        help='Path to temporal.db (default: output/temporal/temporal.db)'
    )

    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Output directory (default: output/temporal/summary/)'
    )

    parser.add_argument(
        '--no-plots',
        action='store_true',
        help='Skip generating plots'
    )

    parser.add_argument(
        '--list-regimes',
        action='store_true',
        help='List default regime definitions and exit'
    )

    args = parser.parse_args()

    # List regimes option
    if args.list_regimes:
        print("Default regime definitions:")
        for name, (start, end) in sorted(REGIMES.items(), key=lambda x: x[1][0]):
            print(f"  {name}: {start}-{end}")
        return

    # Create aggregator
    aggregator = TemporalAggregator(
        db_path=args.db,
        output_dir=args.output,
    )

    # Run
    aggregator.run(
        group_by=args.group,
        generate_plots=not args.no_plots,
    )


if __name__ == "__main__":
    main()
