"""
PRISM Rolling Window Regime Evolution

Track how cluster structure evolves over time.

Static correlation = noise averaged over regimes
Dynamic geometry = signal revealing regime changes

What this shows:
- Alignment periods: 1 big cluster, everything coupled
- Divergence periods: Multiple clusters emerge, structure appears
- Crisis periods: Rapid cluster reconfiguration

DuckDB does the heavy lifting:
- Rolling correlations computed in SQL
- Window functions handle the time stepping
- Python only touches small result sets per window

Usage:
    python prism_rolling_regime.py --db prism.db \
        --indicators SPY TLT GLD VIX IEF HYG \
        --start 2007-01-01 --end 2024-12-31 \
        --window 252 --step 21 \
        --output regime_evolution.json

Author: Jason (PRISM Project)
Date: December 2024
"""

import numpy as np
import pandas as pd
import duckdb
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from collections import defaultdict
import json
import logging
import argparse

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class WindowSnapshot:
    """Cluster structure at one point in time."""
    window_id: int
    window_start: datetime
    window_end: datetime
    window_center: datetime
    
    # Cluster structure
    n_clusters: int
    clusters: List[List[str]]
    singletons: List[str]
    
    # Confidence
    confidence: float
    method_agreement: float
    
    # System metrics
    mean_correlation: float
    correlation_dispersion: float  # Std of correlations
    effective_dimension: int
    
    # Regime classification
    regime: str  # 'aligned', 'divergent', 'transitioning', 'crisis'
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'window_id': self.window_id,
            'window_start': self.window_start.isoformat(),
            'window_end': self.window_end.isoformat(),
            'window_center': self.window_center.isoformat(),
            'n_clusters': self.n_clusters,
            'clusters': self.clusters,
            'singletons': self.singletons,
            'confidence': self.confidence,
            'method_agreement': self.method_agreement,
            'mean_correlation': self.mean_correlation,
            'correlation_dispersion': self.correlation_dispersion,
            'effective_dimension': self.effective_dimension,
            'regime': self.regime,
        }


@dataclass
class ClusterTransition:
    """A change in cluster structure between windows."""
    from_window: int
    to_window: int
    timestamp: datetime
    
    # What changed
    clusters_before: int
    clusters_after: int
    
    # Type of change
    transition_type: str  # 'merge', 'split', 'reconfigure', 'stable'
    
    # Which indicators moved
    movers: List[str]
    
    # Magnitude
    magnitude: float  # 0-1, how much changed


@dataclass 
class RegimeEvolution:
    """Complete evolution of cluster structure over time."""
    
    # Metadata
    n_windows: int
    n_indicators: int
    start_date: datetime
    end_date: datetime
    window_days: int
    step_days: int
    
    # Time series of snapshots
    snapshots: List[WindowSnapshot]
    
    # Transitions
    transitions: List[ClusterTransition]
    
    # Summary statistics
    regime_counts: Dict[str, int]
    mean_clusters: float
    cluster_volatility: float  # How much does n_clusters jump around?
    
    # Notable periods
    alignment_periods: List[Tuple[datetime, datetime]]
    divergence_periods: List[Tuple[datetime, datetime]]
    crisis_periods: List[Tuple[datetime, datetime]]
    
    def summary(self) -> str:
        lines = [
            "=" * 70,
            "REGIME EVOLUTION SUMMARY",
            "=" * 70,
            "",
            f"Period: {self.start_date.date()} to {self.end_date.date()}",
            f"Windows: {self.n_windows} ({self.window_days}d window, {self.step_days}d step)",
            f"Indicators: {self.n_indicators}",
            "",
            f"Mean clusters: {self.mean_clusters:.1f}",
            f"Cluster volatility: {self.cluster_volatility:.2f}",
            "",
            "REGIME DISTRIBUTION:",
        ]
        
        for regime, count in sorted(self.regime_counts.items()):
            pct = 100 * count / self.n_windows
            lines.append(f"  {regime}: {count} windows ({pct:.1f}%)")
        
        lines.append("")
        lines.append(f"TRANSITIONS: {len(self.transitions)}")
        
        transition_types = defaultdict(int)
        for t in self.transitions:
            transition_types[t.transition_type] += 1
        
        for ttype, count in sorted(transition_types.items()):
            lines.append(f"  {ttype}: {count}")
        
        if self.crisis_periods:
            lines.append("")
            lines.append("CRISIS PERIODS:")
            for start, end in self.crisis_periods[:5]:
                lines.append(f"  {start.date()} to {end.date()}")
        
        return '\n'.join(lines)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'n_windows': self.n_windows,
            'n_indicators': self.n_indicators,
            'start_date': self.start_date.isoformat(),
            'end_date': self.end_date.isoformat(),
            'window_days': self.window_days,
            'step_days': self.step_days,
            'snapshots': [s.to_dict() for s in self.snapshots],
            'transitions': [
                {
                    'from_window': t.from_window,
                    'to_window': t.to_window,
                    'timestamp': t.timestamp.isoformat(),
                    'clusters_before': t.clusters_before,
                    'clusters_after': t.clusters_after,
                    'transition_type': t.transition_type,
                    'movers': t.movers,
                    'magnitude': t.magnitude,
                }
                for t in self.transitions
            ],
            'regime_counts': self.regime_counts,
            'mean_clusters': self.mean_clusters,
            'cluster_volatility': self.cluster_volatility,
        }


# =============================================================================
# DUCKDB QUERY BUILDER
# =============================================================================

class RollingCorrelationEngine:
    """
    Compute rolling correlations using DuckDB.
    
    DuckDB does the work. Python gets small results.
    """
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._con = None
    
    @property
    def con(self):
        if self._con is None:
            self._con = duckdb.connect(self.db_path, read_only=True)
        return self._con
    
    def close(self):
        if self._con:
            self._con.close()
            self._con = None
    
    def get_available_indicators(self) -> List[str]:
        """Get list of available indicators."""
        query = """
        SELECT DISTINCT indicator_id 
        FROM series_data 
        ORDER BY indicator_id
        """
        try:
            result = self.con.execute(query).fetchdf()
            return result['indicator_id'].tolist()
        except:
            return []
    
    def get_date_range(self, indicators: List[str] = None) -> Tuple[datetime, datetime]:
        """Get available date range."""
        query = """
        SELECT MIN(obs_date) as min_date, MAX(obs_date) as max_date
        FROM series_data
        """
        if indicators:
            placeholders = ','.join(['?' for _ in indicators])
            query += f" WHERE indicator_id IN ({placeholders})"
            result = self.con.execute(query, indicators).fetchdf()
        else:
            result = self.con.execute(query).fetchdf()
        
        return result['min_date'].iloc[0], result['max_date'].iloc[0]
    
    def compute_window_correlation(
        self,
        indicators: List[str],
        window_start: datetime,
        window_end: datetime
    ) -> pd.DataFrame:
        """
        Compute correlation matrix for a single window.
        
        Uses DuckDB to pivot and compute efficiently.
        """
        n = len(indicators)
        placeholders = ','.join(['?' for _ in indicators])
        
        # Step 1: Get returns in this window
        # Using log returns: ln(price_t / price_{t-1})
        query = f"""
        WITH windowed_data AS (
            SELECT 
                obs_date,
                indicator_id,
                value,
                LAG(value) OVER (PARTITION BY indicator_id ORDER BY obs_date) as prev_value
            FROM series_data
            WHERE indicator_id IN ({placeholders})
              AND obs_date >= ?
              AND obs_date <= ?
        ),
        returns AS (
            SELECT 
                obs_date,
                indicator_id,
                LN(value / NULLIF(prev_value, 0)) as ret
            FROM windowed_data
            WHERE prev_value IS NOT NULL
              AND prev_value > 0
              AND value > 0
        )
        SELECT 
            obs_date,
            indicator_id,
            ret
        FROM returns
        WHERE ret IS NOT NULL
          AND ABS(ret) < 1  -- Filter outliers (100% moves)
        ORDER BY obs_date, indicator_id
        """
        
        params = indicators + [window_start, window_end]
        
        try:
            df = self.con.execute(query, params).fetchdf()
        except Exception as e:
            logger.warning(f"Query failed: {e}")
            return pd.DataFrame()
        
        if df.empty:
            return pd.DataFrame()
        
        # Pivot to wide format
        wide = df.pivot(index='obs_date', columns='indicator_id', values='ret')
        
        # Compute correlation
        corr = wide.corr()
        
        return corr
    
    def compute_all_window_correlations(
        self,
        indicators: List[str],
        start_date: datetime,
        end_date: datetime,
        window_days: int = 252,
        step_days: int = 21
    ) -> List[Tuple[datetime, datetime, pd.DataFrame]]:
        """
        Compute correlations for all rolling windows.
        
        Returns list of (window_start, window_end, correlation_matrix)
        """
        windows = []
        
        current_end = start_date + timedelta(days=window_days)
        
        while current_end <= end_date:
            current_start = current_end - timedelta(days=window_days)
            
            corr = self.compute_window_correlation(
                indicators, current_start, current_end
            )
            
            if not corr.empty and len(corr) >= 2:
                windows.append((current_start, current_end, corr))
            
            current_end += timedelta(days=step_days)
        
        return windows
    
    def compute_correlation_stats(
        self,
        corr_matrix: pd.DataFrame
    ) -> Dict[str, float]:
        """Compute summary statistics from correlation matrix."""
        # Get upper triangle (excluding diagonal)
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        corr_values = corr_matrix.values[mask]
        
        if len(corr_values) == 0:
            return {
                'mean': 0,
                'std': 0,
                'min': 0,
                'max': 0,
                'effective_dim': len(corr_matrix),
            }
        
        # Effective dimension from eigenvalues
        try:
            eigenvalues = np.linalg.eigvalsh(corr_matrix.values)
            eigenvalues = np.sort(eigenvalues)[::-1]
            eigenvalues = np.maximum(eigenvalues, 0)
            total = np.sum(eigenvalues)
            if total > 0:
                cumsum = np.cumsum(eigenvalues / total)
                effective_dim = np.searchsorted(cumsum, 0.95) + 1
            else:
                effective_dim = len(corr_matrix)
        except:
            effective_dim = len(corr_matrix)
        
        return {
            'mean': np.mean(corr_values),
            'std': np.std(corr_values),
            'min': np.min(corr_values),
            'max': np.max(corr_values),
            'effective_dim': effective_dim,
        }


# =============================================================================
# REGIME EVOLUTION TRACKER
# =============================================================================

class RegimeEvolutionTracker:
    """
    Track how cluster structure evolves over rolling windows.
    """
    
    def __init__(self, db_path: str):
        self.correlation_engine = RollingCorrelationEngine(db_path)
        self._cluster_detector = None
    
    @property
    def cluster_detector(self):
        if self._cluster_detector is None:
            from prism_emergent_clusters import EmergentClusterDetector
            self._cluster_detector = EmergentClusterDetector()
        return self._cluster_detector
    
    def compute_evolution(
        self,
        indicators: List[str],
        start_date: datetime,
        end_date: datetime,
        window_days: int = 252,
        step_days: int = 21,
        methods: List[str] = None
    ) -> RegimeEvolution:
        """
        Compute full regime evolution over time.
        
        Args:
            indicators: List of indicator IDs
            start_date: Analysis start
            end_date: Analysis end
            window_days: Rolling window size
            step_days: Step between windows
            methods: Cluster detection methods to use
        
        Returns:
            RegimeEvolution with full timeline
        """
        logger.info(f"Computing regime evolution for {len(indicators)} indicators")
        logger.info(f"Period: {start_date.date()} to {end_date.date()}")
        logger.info(f"Window: {window_days}d, Step: {step_days}d")
        
        # Compute all window correlations
        windows = self.correlation_engine.compute_all_window_correlations(
            indicators, start_date, end_date, window_days, step_days
        )
        
        logger.info(f"Computed {len(windows)} rolling windows")
        
        if len(windows) == 0:
            return self._empty_evolution(indicators, start_date, end_date, window_days, step_days)
        
        # Process each window
        snapshots = []
        
        for i, (win_start, win_end, corr_matrix) in enumerate(windows):
            if i % 10 == 0:
                logger.info(f"Processing window {i+1}/{len(windows)}")
            
            snapshot = self._process_window(
                window_id=i,
                window_start=win_start,
                window_end=win_end,
                corr_matrix=corr_matrix,
                methods=methods
            )
            
            snapshots.append(snapshot)
        
        # Detect transitions
        transitions = self._detect_transitions(snapshots)
        
        # Compute summary statistics
        n_clusters_series = [s.n_clusters for s in snapshots]
        regime_counts = defaultdict(int)
        for s in snapshots:
            regime_counts[s.regime] += 1
        
        # Find notable periods
        alignment_periods = self._find_periods(snapshots, 'aligned')
        divergence_periods = self._find_periods(snapshots, 'divergent')
        crisis_periods = self._find_periods(snapshots, 'crisis')
        
        return RegimeEvolution(
            n_windows=len(snapshots),
            n_indicators=len(indicators),
            start_date=start_date,
            end_date=end_date,
            window_days=window_days,
            step_days=step_days,
            snapshots=snapshots,
            transitions=transitions,
            regime_counts=dict(regime_counts),
            mean_clusters=np.mean(n_clusters_series),
            cluster_volatility=np.std(n_clusters_series),
            alignment_periods=alignment_periods,
            divergence_periods=divergence_periods,
            crisis_periods=crisis_periods,
        )
    
    def _process_window(
        self,
        window_id: int,
        window_start: datetime,
        window_end: datetime,
        corr_matrix: pd.DataFrame,
        methods: List[str] = None
    ) -> WindowSnapshot:
        """Process a single window."""
        
        indicator_ids = corr_matrix.index.tolist()
        
        # Correlation stats
        corr_stats = self.correlation_engine.compute_correlation_stats(corr_matrix)
        
        # Run cluster detection
        try:
            result = self.cluster_detector.detect_clusters(
                corr_matrix.values,
                indicator_ids,
                methods=methods
            )
            
            clusters = [c.members for c in result.clusters]
            singletons = result.singletons
            n_clusters = result.n_clusters
            confidence = result.confidence
            agreement = result.method_agreement
            
        except Exception as e:
            logger.warning(f"Cluster detection failed for window {window_id}: {e}")
            clusters = [indicator_ids]
            singletons = []
            n_clusters = 1
            confidence = 0
            agreement = 0
        
        # Classify regime
        regime = self._classify_regime(
            n_clusters=n_clusters,
            n_indicators=len(indicator_ids),
            mean_correlation=corr_stats['mean'],
            correlation_dispersion=corr_stats['std'],
            confidence=confidence
        )
        
        window_center = window_start + (window_end - window_start) / 2
        
        return WindowSnapshot(
            window_id=window_id,
            window_start=window_start,
            window_end=window_end,
            window_center=window_center,
            n_clusters=n_clusters,
            clusters=clusters,
            singletons=singletons,
            confidence=confidence,
            method_agreement=agreement,
            mean_correlation=corr_stats['mean'],
            correlation_dispersion=corr_stats['std'],
            effective_dimension=corr_stats['effective_dim'],
            regime=regime,
        )
    
    def _classify_regime(
        self,
        n_clusters: int,
        n_indicators: int,
        mean_correlation: float,
        correlation_dispersion: float,
        confidence: float
    ) -> str:
        """
        Classify the current regime.
        
        aligned:      High correlation, few clusters, everything moving together
        divergent:    Low correlation, many clusters, structure visible
        transitioning: Medium correlation, cluster structure unstable
        crisis:       Very high correlation dispersion, rapid changes
        """
        # Ratio of clusters to indicators
        cluster_ratio = n_clusters / max(n_indicators, 1)
        
        # Crisis: high dispersion (some things very correlated, others not)
        if correlation_dispersion > 0.4:
            return 'crisis'
        
        # Aligned: high correlation, few clusters
        if mean_correlation > 0.6 and cluster_ratio < 0.3:
            return 'aligned'
        
        # Divergent: low correlation or many clusters
        if mean_correlation < 0.3 or cluster_ratio > 0.5:
            return 'divergent'
        
        # Transitioning: in between
        return 'transitioning'
    
    def _detect_transitions(
        self,
        snapshots: List[WindowSnapshot]
    ) -> List[ClusterTransition]:
        """Detect transitions between consecutive windows."""
        
        transitions = []
        
        for i in range(1, len(snapshots)):
            prev = snapshots[i - 1]
            curr = snapshots[i]
            
            # Compare cluster structures
            prev_clusters = [set(c) for c in prev.clusters]
            curr_clusters = [set(c) for c in curr.clusters]
            
            # Find movers (indicators that changed clusters)
            prev_assignments = {}
            for j, cluster in enumerate(prev_clusters):
                for ind in cluster:
                    prev_assignments[ind] = j
            
            curr_assignments = {}
            for j, cluster in enumerate(curr_clusters):
                for ind in cluster:
                    curr_assignments[ind] = j
            
            movers = []
            all_indicators = set(prev_assignments.keys()) | set(curr_assignments.keys())
            for ind in all_indicators:
                prev_cluster = prev_assignments.get(ind, -1)
                curr_cluster = curr_assignments.get(ind, -1)
                if prev_cluster != curr_cluster:
                    movers.append(ind)
            
            # Classify transition
            if len(movers) == 0:
                transition_type = 'stable'
            elif curr.n_clusters > prev.n_clusters:
                transition_type = 'split'
            elif curr.n_clusters < prev.n_clusters:
                transition_type = 'merge'
            else:
                transition_type = 'reconfigure'
            
            # Magnitude: fraction of indicators that moved
            magnitude = len(movers) / len(all_indicators) if all_indicators else 0
            
            transitions.append(ClusterTransition(
                from_window=prev.window_id,
                to_window=curr.window_id,
                timestamp=curr.window_center,
                clusters_before=prev.n_clusters,
                clusters_after=curr.n_clusters,
                transition_type=transition_type,
                movers=movers,
                magnitude=magnitude,
            ))
        
        return transitions
    
    def _find_periods(
        self,
        snapshots: List[WindowSnapshot],
        regime: str
    ) -> List[Tuple[datetime, datetime]]:
        """Find contiguous periods of a given regime."""
        
        periods = []
        in_period = False
        period_start = None
        
        for snapshot in snapshots:
            if snapshot.regime == regime:
                if not in_period:
                    in_period = True
                    period_start = snapshot.window_start
            else:
                if in_period:
                    in_period = False
                    periods.append((period_start, snapshot.window_start))
        
        # Handle period extending to end
        if in_period:
            periods.append((period_start, snapshots[-1].window_end))
        
        return periods
    
    def _empty_evolution(
        self,
        indicators: List[str],
        start_date: datetime,
        end_date: datetime,
        window_days: int,
        step_days: int
    ) -> RegimeEvolution:
        """Return empty evolution for no data case."""
        return RegimeEvolution(
            n_windows=0,
            n_indicators=len(indicators),
            start_date=start_date,
            end_date=end_date,
            window_days=window_days,
            step_days=step_days,
            snapshots=[],
            transitions=[],
            regime_counts={},
            mean_clusters=0,
            cluster_volatility=0,
            alignment_periods=[],
            divergence_periods=[],
            crisis_periods=[],
        )
    
    def close(self):
        self.correlation_engine.close()


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_regime_evolution(evolution: RegimeEvolution, output_path: str = None):
    """Plot regime evolution over time."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    
    if not evolution.snapshots:
        print("No data to plot")
        return
    
    dates = [s.window_center for s in evolution.snapshots]
    n_clusters = [s.n_clusters for s in evolution.snapshots]
    mean_corr = [s.mean_correlation for s in evolution.snapshots]
    dispersion = [s.correlation_dispersion for s in evolution.snapshots]
    
    # Regime colors
    regime_colors = {
        'aligned': 'green',
        'divergent': 'blue',
        'transitioning': 'orange',
        'crisis': 'red',
    }
    colors = [regime_colors.get(s.regime, 'gray') for s in evolution.snapshots]
    
    fig, axes = plt.subplots(4, 1, figsize=(16, 12), sharex=True)
    
    # 1. Number of clusters
    axes[0].scatter(dates, n_clusters, c=colors, s=30, alpha=0.7)
    axes[0].plot(dates, n_clusters, 'k-', alpha=0.3)
    axes[0].set_ylabel('N Clusters')
    axes[0].set_title('Regime Evolution Over Time')
    axes[0].axhline(y=evolution.mean_clusters, color='gray', linestyle='--', alpha=0.5)
    
    # 2. Mean correlation
    axes[1].plot(dates, mean_corr, 'b-', linewidth=1.5)
    axes[1].fill_between(dates, mean_corr, alpha=0.3)
    axes[1].set_ylabel('Mean Correlation')
    axes[1].axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    
    # 3. Correlation dispersion
    axes[2].plot(dates, dispersion, 'purple', linewidth=1.5)
    axes[2].fill_between(dates, dispersion, alpha=0.3, color='purple')
    axes[2].set_ylabel('Corr Dispersion')
    axes[2].axhline(y=0.4, color='red', linestyle='--', alpha=0.5, label='Crisis threshold')
    
    # 4. Regime timeline
    regime_map = {'aligned': 0, 'transitioning': 1, 'divergent': 2, 'crisis': 3}
    regime_values = [regime_map.get(s.regime, 1) for s in evolution.snapshots]
    axes[3].scatter(dates, regime_values, c=colors, s=30)
    axes[3].set_ylabel('Regime')
    axes[3].set_yticks([0, 1, 2, 3])
    axes[3].set_yticklabels(['Aligned', 'Transitioning', 'Divergent', 'Crisis'])
    axes[3].set_xlabel('Date')
    
    # Legend
    patches = [mpatches.Patch(color=c, label=r) for r, c in regime_colors.items()]
    axes[0].legend(handles=patches, loc='upper right')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved plot to {output_path}")
    else:
        plt.show()
    
    plt.close()


def export_evolution_csv(evolution: RegimeEvolution, output_path: str):
    """Export evolution as CSV for external analysis."""
    rows = []
    for s in evolution.snapshots:
        rows.append({
            'date': s.window_center.date(),
            'window_start': s.window_start.date(),
            'window_end': s.window_end.date(),
            'n_clusters': s.n_clusters,
            'confidence': s.confidence,
            'mean_correlation': s.mean_correlation,
            'correlation_dispersion': s.correlation_dispersion,
            'effective_dimension': s.effective_dimension,
            'regime': s.regime,
            'cluster_sizes': ','.join(str(len(c)) for c in s.clusters),
        })
    
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    logger.info(f"Saved CSV to {output_path}")
    return df


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='PRISM Rolling Window Regime Evolution',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic run
    python prism_rolling_regime.py --db prism.db \\
        --indicators SPY TLT GLD VIX IEF HYG \\
        --start 2007-01-01 --end 2024-12-31 \\
        --window 252 --step 21

    # With output files
    python prism_rolling_regime.py --db prism.db \\
        --indicators SPY TLT GLD VIX IEF HYG \\
        --start 2007-01-01 --end 2024-12-31 \\
        --output regime_evolution.json \\
        --plot regime_plot.png \\
        --csv regime_data.csv
"""
    )
    
    parser.add_argument('--db', type=str, required=True, help='Path to DuckDB database')
    parser.add_argument('--indicators', type=str, nargs='+', help='Indicator IDs')
    parser.add_argument('--start', type=str, required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, required=True, help='End date (YYYY-MM-DD)')
    parser.add_argument('--window', type=int, default=252, help='Window size in days (default: 252)')
    parser.add_argument('--step', type=int, default=21, help='Step size in days (default: 21)')
    parser.add_argument('--output', '-o', type=str, help='Output JSON path')
    parser.add_argument('--plot', type=str, help='Plot output path')
    parser.add_argument('--csv', type=str, help='CSV output path')
    parser.add_argument('--quiet', '-q', action='store_true', help='Minimal output')
    
    args = parser.parse_args()
    
    # Parse dates
    start_date = datetime.strptime(args.start, '%Y-%m-%d')
    end_date = datetime.strptime(args.end, '%Y-%m-%d')
    
    # Get indicators
    tracker = RegimeEvolutionTracker(args.db)
    
    if args.indicators:
        indicators = args.indicators
    else:
        indicators = tracker.correlation_engine.get_available_indicators()
        if not args.quiet:
            logger.info(f"Using all available indicators: {len(indicators)}")
    
    try:
        # Compute evolution
        evolution = tracker.compute_evolution(
            indicators=indicators,
            start_date=start_date,
            end_date=end_date,
            window_days=args.window,
            step_days=args.step,
        )
        
        # Print summary
        if not args.quiet:
            print()
            print(evolution.summary())
        
        # Save outputs
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(evolution.to_dict(), f, indent=2, default=str)
            logger.info(f"Saved JSON to {args.output}")
        
        if args.csv:
            export_evolution_csv(evolution, args.csv)
        
        if args.plot:
            plot_regime_evolution(evolution, args.plot)
        
    finally:
        tracker.close()


if __name__ == '__main__':
    main()