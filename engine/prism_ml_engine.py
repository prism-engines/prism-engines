"""
PRISM ML Engine - Machine Learning Based Analysis

This engine focuses on machine learning approaches for indicator analysis.
It uses registry-driven configuration to load panel data and interpret columns.

Usage:
    from engine import PrismMLEngine

    engine = PrismMLEngine()
    results = engine.analyze()
    print(results['feature_importance'])
"""

import logging
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

import sys
_pkg_root = Path(__file__).parent.parent
if str(_pkg_root) not in sys.path:
    sys.path.insert(0, str(_pkg_root))

from utils.registry import RegistryManager, load_panel, get_engine_config

logger = logging.getLogger(__name__)


class PrismMLEngine:
    """
    Engine for machine learning based analysis.

    This engine:
    - Loads panel data from registry-specified paths
    - Provides ML-based feature importance and clustering
    - Implements dimensionality reduction and pattern recognition
    """

    name = "ml"
    description = "Machine learning analysis engine"

    def __init__(
        self,
        project_root: Optional[Path] = None,
        config: Optional[Dict] = None
    ):
        """
        Initialize the ML engine.

        Args:
            project_root: Optional project root path
            config: Optional configuration overrides
        """
        self.registry = RegistryManager(project_root)
        self.config = config or {}

        # Get default config from registry
        self._default_config = self.registry.get_engine_config() or {}
        self._panel: Optional[pd.DataFrame] = None
        self._last_results: Optional[Dict] = None

    @property
    def panel(self) -> pd.DataFrame:
        """Lazy-load the panel data."""
        if self._panel is None:
            self._panel = self._load_panel()
        return self._panel

    def _load_panel(self) -> pd.DataFrame:
        """
        Load full panel data.

        Returns:
            Full DataFrame with all available series
        """
        # Load from registry-specified path
        full_panel = self.registry.load_panel(panel_type='master')

        logger.info(f"Loaded {len(full_panel.columns)} series from panel for ML analysis")
        return full_panel

    def reload_panel(self) -> pd.DataFrame:
        """Force reload of panel data."""
        self._panel = None
        return self.panel

    def analyze(
        self,
        target: Optional[str] = None,
        lookback_years: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run ML-based analysis.

        Args:
            target: Optional target variable for supervised learning
            lookback_years: Number of years to analyze
            **kwargs: Additional analysis parameters

        Returns:
            Dictionary with ML analysis results
        """
        lookback = lookback_years or self._default_config.get('default_lookback_years', 5)

        df = self.panel.copy()

        # Filter by lookback period
        if lookback and hasattr(df.index, 'year'):
            cutoff = datetime.now().year - lookback
            df = df[df.index.year >= cutoff]

        # Drop columns with too many NaNs
        nan_threshold = self._default_config.get('nan_threshold', 0.5)
        valid_cols = df.columns[df.isna().mean() < nan_threshold]
        df = df[valid_cols]

        # Forward fill then drop remaining NaNs
        df = df.ffill().dropna()

        results = {
            'timestamp': datetime.now().isoformat(),
            'engine': self.name,
            'n_features': len(df.columns),
            'n_observations': len(df),
            'date_range': {
                'start': str(df.index.min()) if len(df) > 0 else None,
                'end': str(df.index.max()) if len(df) > 0 else None,
            },
        }

        # PCA-based feature importance
        results['pca_analysis'] = self._run_pca(df)

        # Clustering analysis
        results['clustering'] = self._run_clustering(df)

        # Feature correlation analysis
        results['correlation_groups'] = self._find_correlation_groups(df)

        # Rolling pattern detection
        results['patterns'] = self._detect_patterns(df)

        # Top indicators based on ML analysis
        results['top_indicators'] = self._rank_features(df, results)

        # Supervised analysis if target provided
        if target and target in df.columns:
            results['supervised_analysis'] = self._run_supervised(df, target)

        self._last_results = results
        return results

    def _run_pca(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Run PCA analysis for dimensionality reduction and feature importance."""
        # Standardize data
        X = (df - df.mean()) / df.std()
        X = X.fillna(0)

        # SVD-based PCA
        try:
            U, S, Vt = np.linalg.svd(X.values, full_matrices=False)
        except np.linalg.LinAlgError:
            return {'error': 'SVD did not converge'}

        # Explained variance
        explained_var = (S ** 2) / (len(X) - 1)
        explained_var_ratio = explained_var / explained_var.sum()
        cumulative_var = np.cumsum(explained_var_ratio)

        # Number of components for different thresholds
        n_for_80 = int(np.searchsorted(cumulative_var, 0.8) + 1)
        n_for_90 = int(np.searchsorted(cumulative_var, 0.9) + 1)
        n_for_95 = int(np.searchsorted(cumulative_var, 0.95) + 1)

        # Feature loadings on first 3 components
        n_components = min(3, len(S))
        loadings = pd.DataFrame(
            Vt[:n_components].T * S[:n_components],
            index=df.columns,
            columns=[f'PC{i+1}' for i in range(n_components)]
        )

        # Feature importance based on loadings
        feature_importance = np.abs(loadings).sum(axis=1)
        feature_importance = feature_importance / feature_importance.sum()

        return {
            'n_components_80pct': n_for_80,
            'n_components_90pct': n_for_90,
            'n_components_95pct': n_for_95,
            'explained_variance_ratio': explained_var_ratio[:5].tolist(),
            'cumulative_variance': cumulative_var[:5].tolist(),
            'feature_importance': feature_importance.sort_values(ascending=False).head(10).to_dict(),
            'top_pc1_features': loadings['PC1'].abs().sort_values(ascending=False).head(5).to_dict(),
        }

    def _run_clustering(self, df: pd.DataFrame, n_clusters: int = 5) -> Dict[str, Any]:
        """Run k-means-style clustering on features."""
        # Standardize
        X = (df - df.mean()) / df.std()
        X = X.fillna(0)

        # Use correlation as distance metric
        corr = X.corr()
        dist = 1 - np.abs(corr.values)

        # Simple k-means on correlation structure
        np.random.seed(42)
        n_features = len(df.columns)
        n_clusters = min(n_clusters, n_features // 2)

        # Initialize cluster centers
        centers_idx = np.random.choice(n_features, n_clusters, replace=False)

        labels = np.zeros(n_features, dtype=int)

        # Iterate
        for _ in range(10):
            # Assign to nearest center
            for i in range(n_features):
                distances = [dist[i, c] for c in centers_idx]
                labels[i] = np.argmin(distances)

            # Update centers
            for k in range(n_clusters):
                cluster_members = np.where(labels == k)[0]
                if len(cluster_members) > 0:
                    avg_dist = [dist[m, cluster_members].mean() for m in cluster_members]
                    centers_idx[k] = cluster_members[np.argmin(avg_dist)]

        # Build cluster info
        clusters = {}
        for k in range(n_clusters):
            members = df.columns[labels == k].tolist()
            if members:
                clusters[f'cluster_{k}'] = {
                    'members': members,
                    'n_members': len(members),
                    'center': df.columns[centers_idx[k]],
                }

        return {
            'n_clusters': n_clusters,
            'clusters': clusters,
            'silhouette_approx': self._approx_silhouette(dist, labels),
        }

    def _approx_silhouette(self, dist: np.ndarray, labels: np.ndarray) -> float:
        """Approximate silhouette score."""
        n = len(labels)
        if n < 2:
            return 0.0

        silhouettes = []
        for i in range(n):
            # Intra-cluster distance
            same_cluster = np.where(labels == labels[i])[0]
            if len(same_cluster) > 1:
                a = dist[i, same_cluster[same_cluster != i]].mean()
            else:
                a = 0

            # Nearest cluster distance
            b_vals = []
            for k in np.unique(labels):
                if k != labels[i]:
                    other_cluster = np.where(labels == k)[0]
                    if len(other_cluster) > 0:
                        b_vals.append(dist[i, other_cluster].mean())

            b = min(b_vals) if b_vals else 0

            if max(a, b) > 0:
                silhouettes.append((b - a) / max(a, b))
            else:
                silhouettes.append(0)

        return float(np.mean(silhouettes))

    def _find_correlation_groups(self, df: pd.DataFrame, threshold: float = 0.7) -> Dict[str, Any]:
        """Find groups of highly correlated features."""
        corr = df.corr()

        # Find pairs above threshold
        high_corr_pairs = []
        for i, col1 in enumerate(corr.columns):
            for j, col2 in enumerate(corr.columns):
                if i < j:
                    corr_val = corr.loc[col1, col2]
                    if abs(corr_val) > threshold:
                        high_corr_pairs.append({
                            'feature1': col1,
                            'feature2': col2,
                            'correlation': float(corr_val)
                        })

        # Group connected features
        groups = []
        used = set()

        for pair in sorted(high_corr_pairs, key=lambda x: abs(x['correlation']), reverse=True):
            f1, f2 = pair['feature1'], pair['feature2']

            # Find existing groups
            group_idx = None
            for i, group in enumerate(groups):
                if f1 in group or f2 in group:
                    group_idx = i
                    break

            if group_idx is not None:
                groups[group_idx].add(f1)
                groups[group_idx].add(f2)
            else:
                groups.append({f1, f2})

            used.add(f1)
            used.add(f2)

        return {
            'n_groups': len(groups),
            'groups': [list(g) for g in groups],
            'high_correlation_pairs': high_corr_pairs[:10],
        }

    def _detect_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect patterns like trends and seasonality."""
        patterns = {}

        for col in df.columns:
            series = df[col].dropna()
            if len(series) < 24:  # Need at least 2 years of monthly data
                continue

            # Trend detection via linear regression
            x = np.arange(len(series))
            y = series.values

            # Simple linear regression
            x_mean = x.mean()
            y_mean = y.mean()
            slope = np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean) ** 2)

            # R-squared
            y_pred = slope * (x - x_mean) + y_mean
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - y_mean) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

            # Trend strength
            if r_squared > 0.5:
                trend = 'strong_' + ('up' if slope > 0 else 'down')
            elif r_squared > 0.2:
                trend = 'weak_' + ('up' if slope > 0 else 'down')
            else:
                trend = 'no_trend'

            # Simple seasonality check (autocorrelation at lag 12)
            if len(series) >= 24:
                autocorr_12 = series.autocorr(lag=12)
                has_seasonality = abs(autocorr_12) > 0.3 if not pd.isna(autocorr_12) else False
            else:
                has_seasonality = False

            patterns[col] = {
                'trend': trend,
                'trend_r_squared': float(r_squared),
                'slope': float(slope),
                'has_seasonality': has_seasonality,
            }

        return patterns

    def _rank_features(self, df: pd.DataFrame, results: Dict) -> List[Dict[str, Any]]:
        """Rank features based on ML analysis."""
        rankings = []

        pca_importance = results.get('pca_analysis', {}).get('feature_importance', {})
        patterns = results.get('patterns', {})

        for col in df.columns:
            # PCA importance
            pca_score = pca_importance.get(col, 0)

            # Pattern score (trend strength)
            pattern_info = patterns.get(col, {})
            trend_score = pattern_info.get('trend_r_squared', 0)

            # Combine scores
            combined_score = 0.6 * pca_score + 0.4 * trend_score

            rankings.append({
                'indicator': col,
                'score': float(combined_score),
                'pca_importance': float(pca_score),
                'trend_r_squared': float(trend_score),
                'trend': pattern_info.get('trend', 'unknown'),
            })

        rankings.sort(key=lambda x: x['score'], reverse=True)

        for i, r in enumerate(rankings):
            r['rank'] = i + 1

        return rankings[:10]

    def _run_supervised(self, df: pd.DataFrame, target: str) -> Dict[str, Any]:
        """Run supervised learning analysis with a target variable."""
        if target not in df.columns:
            return {'error': f'Target {target} not found'}

        y = df[target]
        X = df.drop(columns=[target])

        # Standardize
        X_scaled = (X - X.mean()) / X.std()
        X_scaled = X_scaled.fillna(0)

        # Calculate correlation with target as simple feature importance
        correlations = {}
        for col in X.columns:
            corr = X[col].corr(y)
            if not pd.isna(corr):
                correlations[col] = float(corr)

        # Sort by absolute correlation
        sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)

        return {
            'target': target,
            'n_features': len(X.columns),
            'feature_target_correlations': dict(sorted_corr[:10]),
            'most_predictive': sorted_corr[0][0] if sorted_corr else None,
            'most_predictive_correlation': sorted_corr[0][1] if sorted_corr else 0,
        }

    def predict_importance(
        self,
        forward_periods: int = 12,
        target: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Predict which indicators will be most important for future returns.

        Args:
            forward_periods: Number of periods to look ahead
            target: Target variable (default: first market series)

        Returns:
            Prediction results
        """
        df = self.panel.copy()

        # Default target to SPY if available
        if target is None:
            market_cols = self.registry.get_market_series()
            target = market_cols[0] if market_cols and market_cols[0] in df.columns else df.columns[0]

        if target not in df.columns:
            return {'error': f'Target {target} not found'}

        # Create forward returns
        forward_returns = df[target].pct_change(forward_periods).shift(-forward_periods)

        # Create lagged features
        features = df.drop(columns=[target]).shift(forward_periods)

        # Combine and drop NaNs
        combined = pd.concat([forward_returns, features], axis=1).dropna()

        if len(combined) < 50:
            return {'error': 'Insufficient data for prediction analysis'}

        y = combined.iloc[:, 0]
        X = combined.iloc[:, 1:]

        # Calculate predictive power via correlation
        predictive_power = {}
        for col in X.columns:
            corr = X[col].corr(y)
            if not pd.isna(corr):
                predictive_power[col] = float(corr)

        sorted_power = sorted(predictive_power.items(), key=lambda x: abs(x[1]), reverse=True)

        return {
            'target': target,
            'forward_periods': forward_periods,
            'n_observations': len(combined),
            'predictive_features': dict(sorted_power[:10]),
            'best_predictor': sorted_power[0][0] if sorted_power else None,
            'best_predictor_correlation': sorted_power[0][1] if sorted_power else 0,
        }

    def __repr__(self) -> str:
        return f"PrismMLEngine(features={len(self.panel.columns) if self._panel is not None else 'not loaded'})"
