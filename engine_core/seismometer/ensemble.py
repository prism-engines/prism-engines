"""
Seismometer - Ensemble instability detection

Main interface combining all detectors into a unified stability index.
"""

from typing import Dict, Any, Optional, List, Tuple
import pandas as pd
import numpy as np
import logging

from .clustering import ClusteringDriftDetector
from .autoencoder import ReconstructionErrorDetector
from .correlation_graph import CorrelationGraphDetector
from .utils import get_default_features

logger = logging.getLogger(__name__)


class Seismometer:
    """
    Main seismometer class combining multiple instability detectors.

    Stability Index interpretation (0-1):
    - 0.90-1.00: Stable - Normal market behavior
    - 0.70-0.90: Elevated - Increased noise, heightened attention
    - 0.50-0.70: Pre-instability - Structural stress emerging
    - 0.30-0.50: Divergence - Active structural breakdown
    - 0.00-0.30: High-risk - Imminent or ongoing instability

    Note: The stability index is 1 - instability_score, so higher = more stable.
    """

    ALERT_LEVELS = {
        (0.90, 1.01): 'stable',
        (0.70, 0.90): 'elevated',
        (0.50, 0.70): 'pre_instability',
        (0.30, 0.50): 'divergence',
        (0.00, 0.30): 'high_risk',
    }

    DEFAULT_WEIGHTS = {
        'clustering_drift': 0.35,
        'reconstruction_error': 0.35,
        'correlation_graph': 0.30,
    }

    def __init__(
        self,
        conn=None,
        weights: Optional[Dict[str, float]] = None,
        n_clusters: int = 5,
        encoding_dim: int = 8,
        correlation_window: int = 60
    ):
        """
        Initialize seismometer.

        Args:
            conn: Database connection (optional, for loading from indicator_values)
            weights: Detector weights (must sum to 1.0)
            n_clusters: Number of clusters for clustering detector
            encoding_dim: Encoding dimension for autoencoder
            correlation_window: Window for correlation detector
        """
        self.conn = conn
        self.weights = weights or self.DEFAULT_WEIGHTS.copy()

        # Validate weights sum to 1
        weight_sum = sum(self.weights.values())
        if abs(weight_sum - 1.0) > 0.01:
            logger.warning(f"Weights sum to {weight_sum}, normalizing to 1.0")
            for k in self.weights:
                self.weights[k] /= weight_sum

        # Initialize detectors
        self.detectors = {
            'clustering_drift': ClusteringDriftDetector(n_clusters=n_clusters),
            'reconstruction_error': ReconstructionErrorDetector(encoding_dim=encoding_dim),
            'correlation_graph': CorrelationGraphDetector(window_days=correlation_window),
        }

        self._features: Optional[pd.DataFrame] = None
        self._scores: Optional[pd.DataFrame] = None
        self._baseline_end: Optional[str] = None
        self.is_fitted = False

    def load_features(
        self,
        start: str = '2010-01-01',
        end: Optional[str] = None,
        feature_names: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Load features from indicator_values table.

        Args:
            start: Start date for data
            end: End date (default: latest available)
            feature_names: List of feature names (default: use defaults)

        Returns:
            DataFrame with features
        """
        if self.conn is None:
            raise ValueError("No database connection provided")

        features = feature_names or get_default_features()

        # Build query
        feature_list = "', '".join(features)
        query = f"""
            SELECT date, indicator_name, value
            FROM indicator_values
            WHERE indicator_name IN ('{feature_list}')
              AND date >= '{start}'
        """
        if end:
            query += f" AND date <= '{end}'"

        query += " ORDER BY date"

        # Execute query
        df = pd.read_sql(query, self.conn)

        if df.empty:
            raise ValueError("No data returned from query")

        # Pivot to wide format
        df_wide = df.pivot(
            index='date',
            columns='indicator_name',
            values='value'
        )
        df_wide.index = pd.to_datetime(df_wide.index)
        df_wide = df_wide.sort_index()

        # Forward fill and drop rows with remaining NaN
        df_wide = df_wide.ffill().dropna()

        self._features = df_wide
        logger.info(
            f"Loaded {len(df_wide)} observations, "
            f"{len(df_wide.columns)} features"
        )

        return df_wide

    def fit(
        self,
        features: Optional[pd.DataFrame] = None,
        start: str = '2010-01-01',
        end: str = '2019-12-31'
    ) -> 'Seismometer':
        """
        Fit all detectors on baseline period.

        Recommend using pre-COVID period (end='2019-12-31') for baseline.

        Args:
            features: Feature DataFrame (optional if using load_features)
            start: Start date for baseline
            end: End date for baseline

        Returns:
            self for method chaining
        """
        # Load features if needed
        if features is not None:
            self._features = features.copy()
        elif self._features is None:
            self.load_features(start=start)

        # Ensure datetime index
        if not isinstance(self._features.index, pd.DatetimeIndex):
            self._features.index = pd.to_datetime(self._features.index)

        # Get baseline period
        baseline = self._features.loc[start:end]
        self._baseline_end = end

        if len(baseline) < 100:
            logger.warning(
                f"Small baseline period: {len(baseline)} observations. "
                f"Consider using more data."
            )

        logger.info(f"Fitting detectors on baseline period: {start} to {end}")

        # Fit each detector
        for name, detector in self.detectors.items():
            try:
                detector.fit(baseline)
                logger.info(f"  {name}: fitted successfully")
            except Exception as e:
                logger.error(f"  {name}: fitting failed - {e}")
                raise

        self.is_fitted = True
        return self

    def score(self, features: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Score all observations with all detectors.

        Args:
            features: Feature DataFrame (optional if already loaded)

        Returns:
            DataFrame with detector scores and stability_index
        """
        if not self.is_fitted:
            raise ValueError("Seismometer not fitted - call fit() first")

        if features is not None:
            self._features = features.copy()

        if self._features is None:
            raise ValueError("No features available - provide features or call load_features()")

        # Ensure datetime index
        if not isinstance(self._features.index, pd.DatetimeIndex):
            self._features.index = pd.to_datetime(self._features.index)

        # Score with each detector
        scores_dict = {}
        for name, detector in self.detectors.items():
            try:
                detector_scores = detector.score(self._features)
                scores_dict[name] = detector_scores
            except Exception as e:
                logger.error(f"Error scoring with {name}: {e}")
                scores_dict[name] = pd.Series(
                    np.nan, index=self._features.index, name=name
                )

        # Combine into DataFrame
        scores_df = pd.DataFrame(scores_dict)

        # Compute weighted instability score
        instability = pd.Series(0.0, index=scores_df.index)
        for name, weight in self.weights.items():
            if name in scores_df.columns:
                instability += weight * scores_df[name].fillna(0)

        # Stability index = 1 - instability (higher = more stable)
        scores_df['instability_score'] = instability
        scores_df['stability_index'] = 1.0 - instability

        # Clip stability to [0, 1]
        scores_df['stability_index'] = scores_df['stability_index'].clip(0, 1)

        self._scores = scores_df
        return scores_df

    def get_stability_index(self, date: str) -> Dict[str, Any]:
        """
        Get stability assessment for a specific date.

        Args:
            date: Date to assess (YYYY-MM-DD format)

        Returns:
            Dict with date, stability_index, alert_level, components
        """
        if self._scores is None:
            self.score()

        date = pd.to_datetime(date)

        if date not in self._scores.index:
            # Find nearest date
            available_dates = self._scores.index
            idx = available_dates.get_indexer([date], method='nearest')[0]
            date = available_dates[idx]
            logger.warning(f"Exact date not found, using nearest: {date}")

        row = self._scores.loc[date]
        stability = row['stability_index']

        # Determine alert level
        alert_level = 'unknown'
        for (low, high), level in self.ALERT_LEVELS.items():
            if low <= stability < high:
                alert_level = level
                break

        # Get component scores
        components = {
            name: row.get(name, np.nan)
            for name in self.detectors.keys()
        }

        return {
            'date': str(date.date()),
            'stability_index': float(stability),
            'alert_level': alert_level,
            'components': components,
        }

    def get_current_status(self) -> Dict[str, Any]:
        """
        Get most recent stability assessment.

        Returns:
            Dict with current stability status
        """
        if self._scores is None:
            self.score()

        latest_date = self._scores.index[-1]
        return self.get_stability_index(str(latest_date.date()))

    def get_alert_history(
        self,
        start: Optional[str] = None,
        end: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Get history of alert levels.

        Args:
            start: Start date filter
            end: End date filter

        Returns:
            DataFrame with date, stability_index, alert_level
        """
        if self._scores is None:
            self.score()

        scores = self._scores.copy()

        if start:
            scores = scores.loc[start:]
        if end:
            scores = scores.loc[:end]

        # Compute alert levels
        def get_alert_level(stability):
            for (low, high), level in self.ALERT_LEVELS.items():
                if low <= stability < high:
                    return level
            return 'unknown'

        result = pd.DataFrame({
            'stability_index': scores['stability_index'],
            'alert_level': scores['stability_index'].apply(get_alert_level)
        })

        return result

    def plot_stability_history(
        self,
        start: Optional[str] = None,
        end: Optional[str] = None,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (14, 8)
    ):
        """
        Plot stability index with alert bands.

        Args:
            start: Start date for plot
            end: End date for plot
            save_path: Path to save figure (optional)
            figsize: Figure size (width, height)

        Returns:
            matplotlib figure
        """
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        if self._scores is None:
            self.score()

        scores = self._scores.copy()

        if start:
            scores = scores.loc[start:]
        if end:
            scores = scores.loc[:end]

        fig, ax = plt.subplots(figsize=figsize)

        # Alert band colors
        band_colors = {
            'stable': '#2ecc71',        # Green
            'elevated': '#f1c40f',       # Yellow
            'pre_instability': '#e67e22', # Orange
            'divergence': '#e74c3c',      # Red
            'high_risk': '#8e44ad',       # Dark purple
        }

        # Plot alert bands
        ax.axhspan(0.90, 1.00, alpha=0.3, color=band_colors['stable'])
        ax.axhspan(0.70, 0.90, alpha=0.3, color=band_colors['elevated'])
        ax.axhspan(0.50, 0.70, alpha=0.3, color=band_colors['pre_instability'])
        ax.axhspan(0.30, 0.50, alpha=0.3, color=band_colors['divergence'])
        ax.axhspan(0.00, 0.30, alpha=0.3, color=band_colors['high_risk'])

        # Plot stability index
        ax.plot(
            scores.index,
            scores['stability_index'],
            'k-',
            linewidth=1.5,
            label='Stability Index'
        )

        # Add threshold lines
        for thresh in [0.30, 0.50, 0.70, 0.90]:
            ax.axhline(y=thresh, color='gray', linestyle='--', alpha=0.5)

        # Mark baseline end if available
        if self._baseline_end:
            baseline_date = pd.to_datetime(self._baseline_end)
            if baseline_date in scores.index or baseline_date < scores.index[-1]:
                ax.axvline(
                    x=baseline_date,
                    color='blue',
                    linestyle=':',
                    alpha=0.7,
                    label=f'Baseline end ({self._baseline_end})'
                )

        # Legend
        patches = [
            mpatches.Patch(color=band_colors['stable'], alpha=0.3, label='Stable (0.90+)'),
            mpatches.Patch(color=band_colors['elevated'], alpha=0.3, label='Elevated (0.70-0.90)'),
            mpatches.Patch(color=band_colors['pre_instability'], alpha=0.3, label='Pre-instability (0.50-0.70)'),
            mpatches.Patch(color=band_colors['divergence'], alpha=0.3, label='Divergence (0.30-0.50)'),
            mpatches.Patch(color=band_colors['high_risk'], alpha=0.3, label='High-risk (<0.30)'),
        ]
        ax.legend(handles=patches, loc='lower left', fontsize=9)

        ax.set_xlabel('Date')
        ax.set_ylabel('Stability Index')
        ax.set_title('Seismometer - Stability Index History')
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved plot to {save_path}")

        return fig

    def get_detector_contributions(self, date: str) -> pd.DataFrame:
        """
        Get contribution of each detector to instability score.

        Args:
            date: Date to analyze

        Returns:
            DataFrame with detector names, scores, weights, contributions
        """
        status = self.get_stability_index(date)

        contributions = []
        for name, score in status['components'].items():
            weight = self.weights.get(name, 0)
            contribution = score * weight

            contributions.append({
                'detector': name,
                'score': score,
                'weight': weight,
                'contribution': contribution,
            })

        df = pd.DataFrame(contributions)
        df = df.sort_values('contribution', ascending=False)
        df['contribution_pct'] = df['contribution'] / df['contribution'].sum() * 100

        return df

    def summary(self) -> str:
        """
        Get text summary of current status.

        Returns:
            Formatted summary string
        """
        status = self.get_current_status()

        lines = [
            "=" * 50,
            "SEISMOMETER STATUS",
            "=" * 50,
            f"Date: {status['date']}",
            f"Stability Index: {status['stability_index']:.3f}",
            f"Alert Level: {status['alert_level'].upper()}",
            "",
            "Component Scores (higher = more unstable):",
        ]

        for name, score in status['components'].items():
            weight = self.weights.get(name, 0)
            lines.append(f"  {name}: {score:.3f} (weight: {weight:.2f})")

        lines.append("=" * 50)

        return "\n".join(lines)

    def __repr__(self) -> str:
        fitted_str = "fitted" if self.is_fitted else "not fitted"
        return f"Seismometer({fitted_str}, detectors={list(self.detectors.keys())})"
