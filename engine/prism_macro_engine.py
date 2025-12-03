"""
PRISM Macro Engine - Macroeconomic Data Analysis

This engine focuses on analyzing macroeconomic indicators from FRED and other
economic data sources. It uses registry-driven configuration to load panel data
and interpret columns.

Usage:
    from engine import PrismMacroEngine

    engine = PrismMacroEngine()
    results = engine.analyze()
    print(results['top_indicators'])
"""

import logging
from typing import Dict, Any, Optional, List
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

import sys
_pkg_root = Path(__file__).parent.parent
if str(_pkg_root) not in sys.path:
    sys.path.insert(0, str(_pkg_root))

from utils.registry import RegistryManager, load_panel, get_economic_series, get_engine_config

logger = logging.getLogger(__name__)


class PrismMacroEngine:
    """
    Engine for analyzing macroeconomic indicators.

    This engine:
    - Loads panel data from registry-specified paths
    - Focuses on economic series (FRED data)
    - Provides macro-specific analysis methods
    """

    name = "macro"
    description = "Macroeconomic indicator analysis engine"

    # Economic indicator categories
    CATEGORIES = {
        'rates': ['dgs10', 'dgs2', 'dgs3mo'],
        'spreads': ['t10y2y', 't10y3m'],
        'inflation': ['cpiaucsl', 'cpilfesl', 'ppiaco'],
        'labor': ['unrate', 'payems'],
        'activity': ['indpro', 'houst', 'permit'],
        'money': ['m2sl', 'walcl'],
        'financial_conditions': ['anfci', 'nfci'],
    }

    def __init__(
        self,
        project_root: Optional[Path] = None,
        config: Optional[Dict] = None
    ):
        """
        Initialize the macro engine.

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
        Load and filter panel data for economic series only.

        Returns:
            DataFrame with economic series
        """
        # Load from registry-specified path
        full_panel = self.registry.load_panel(panel_type='master')

        # Get economic series columns from registry
        economic_cols = self.registry.get_economic_series()

        # Filter to available columns
        available_cols = [c for c in economic_cols if c in full_panel.columns]

        if not available_cols:
            logger.warning("No economic columns found in panel. Using all columns.")
            return full_panel

        logger.info(f"Loaded {len(available_cols)} economic series from panel")
        return full_panel[available_cols]

    def reload_panel(self) -> pd.DataFrame:
        """Force reload of panel data."""
        self._panel = None
        return self.panel

    def analyze(
        self,
        lookback_years: Optional[int] = None,
        metrics: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run macroeconomic analysis.

        Args:
            lookback_years: Number of years to analyze (default from config)
            metrics: List of metrics to analyze (default: all available)
            **kwargs: Additional analysis parameters

        Returns:
            Dictionary with analysis results
        """
        lookback = lookback_years or self._default_config.get('default_lookback_years', 5)

        # Get data
        df = self.panel.copy()

        # Filter by lookback period
        if lookback and hasattr(df.index, 'year'):
            cutoff = datetime.now().year - lookback
            df = df[df.index.year >= cutoff]

        # Filter to specific metrics if provided
        if metrics:
            available = [m for m in metrics if m in df.columns]
            df = df[available]

        # Drop rows with too many NaNs
        nan_threshold = self._default_config.get('nan_threshold', 0.5)
        df = df.dropna(thresh=int(len(df.columns) * (1 - nan_threshold)))

        # Run analysis
        results = {
            'timestamp': datetime.now().isoformat(),
            'engine': self.name,
            'n_series': len(df.columns),
            'n_observations': len(df),
            'date_range': {
                'start': str(df.index.min()) if len(df) > 0 else None,
                'end': str(df.index.max()) if len(df) > 0 else None,
            },
            'series_analyzed': list(df.columns),
        }

        # Calculate basic statistics
        results['statistics'] = self._compute_statistics(df)

        # Calculate correlations
        results['correlations'] = self._compute_correlations(df)

        # Calculate momentum indicators
        results['momentum'] = self._compute_momentum(df)

        # Identify top indicators
        results['top_indicators'] = self._rank_indicators(df)

        # Categorize by economic theme
        results['category_analysis'] = self._analyze_categories(df)

        self._last_results = results
        return results

    def _compute_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Compute descriptive statistics."""
        stats = {}

        for col in df.columns:
            series = df[col].dropna()
            if len(series) == 0:
                continue

            stats[col] = {
                'mean': float(series.mean()),
                'std': float(series.std()),
                'min': float(series.min()),
                'max': float(series.max()),
                'current': float(series.iloc[-1]) if len(series) > 0 else None,
                'pct_change_1y': float(series.pct_change(12).iloc[-1]) if len(series) > 12 else None,
            }

        return stats

    def _compute_correlations(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Compute correlation matrix and identify key relationships."""
        corr_matrix = df.corr()

        # Find highest correlations
        high_corr = []
        for i, col1 in enumerate(corr_matrix.columns):
            for j, col2 in enumerate(corr_matrix.columns):
                if i < j:
                    corr_val = corr_matrix.loc[col1, col2]
                    if abs(corr_val) > 0.7:
                        high_corr.append({
                            'series1': col1,
                            'series2': col2,
                            'correlation': float(corr_val)
                        })

        return {
            'high_correlations': sorted(high_corr, key=lambda x: abs(x['correlation']), reverse=True)[:10],
            'avg_correlation': float(corr_matrix.values[np.triu_indices_from(corr_matrix.values, 1)].mean()),
        }

    def _compute_momentum(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Compute momentum indicators."""
        momentum = {}

        for col in df.columns:
            series = df[col].dropna()
            if len(series) < 13:
                continue

            # Calculate z-score of recent changes
            changes = series.pct_change(12)
            if changes.std() > 0:
                z_score = (changes.iloc[-1] - changes.mean()) / changes.std()
            else:
                z_score = 0

            momentum[col] = {
                'pct_change_12m': float(changes.iloc[-1]) if not pd.isna(changes.iloc[-1]) else 0,
                'z_score': float(z_score),
                'trend': 'up' if changes.iloc[-1] > 0 else 'down',
            }

        return momentum

    def _rank_indicators(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Rank indicators by activity/importance."""
        rankings = []

        for col in df.columns:
            series = df[col].dropna()
            if len(series) < 12:
                continue

            # Score based on: volatility, recent change magnitude, data quality
            volatility = series.std() / series.mean() if series.mean() != 0 else 0
            recent_change = abs(series.pct_change(12).iloc[-1]) if len(series) > 12 else 0
            data_quality = len(series) / len(df)

            # Composite score
            score = (0.4 * abs(volatility) + 0.4 * recent_change + 0.2 * data_quality)

            rankings.append({
                'indicator': col,
                'score': float(score),
                'volatility': float(volatility),
                'recent_change': float(recent_change) if not pd.isna(recent_change) else 0,
            })

        # Sort by score
        rankings.sort(key=lambda x: x['score'], reverse=True)

        # Add rank
        for i, r in enumerate(rankings):
            r['rank'] = i + 1

        return rankings[:10]

    def _analyze_categories(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze indicators by economic category."""
        category_analysis = {}

        for category, indicators in self.CATEGORIES.items():
            available = [i for i in indicators if i in df.columns]
            if not available:
                continue

            cat_df = df[available].dropna()
            if len(cat_df) == 0:
                continue

            # Calculate category-level metrics
            category_analysis[category] = {
                'indicators': available,
                'n_indicators': len(available),
                'avg_correlation': float(cat_df.corr().values.mean()),
                'combined_volatility': float(cat_df.std().mean()),
            }

        return category_analysis

    def get_indicator_details(self, indicator: str) -> Dict[str, Any]:
        """
        Get detailed analysis for a specific indicator.

        Args:
            indicator: Indicator name

        Returns:
            Dictionary with indicator details
        """
        if indicator not in self.panel.columns:
            return {'error': f'Indicator {indicator} not found in panel'}

        series = self.panel[indicator].dropna()

        if len(series) == 0:
            return {'error': f'No data for indicator {indicator}'}

        return {
            'indicator': indicator,
            'n_observations': len(series),
            'date_range': {
                'start': str(series.index.min()),
                'end': str(series.index.max()),
            },
            'current_value': float(series.iloc[-1]),
            'statistics': {
                'mean': float(series.mean()),
                'std': float(series.std()),
                'min': float(series.min()),
                'max': float(series.max()),
                'median': float(series.median()),
            },
            'percentile_current': float((series <= series.iloc[-1]).mean() * 100),
        }

    def __repr__(self) -> str:
        return f"PrismMacroEngine(series={len(self.panel.columns) if self._panel is not None else 'not loaded'})"
