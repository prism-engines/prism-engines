"""
PRISM Market Engine - Market Data Analysis

This engine focuses on analyzing market data from Yahoo Finance and other
market data sources. It uses registry-driven configuration to load panel data
and interpret columns.

Usage:
    from engine import PrismMarketEngine

    engine = PrismMarketEngine()
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

from utils.registry import RegistryManager, load_panel, get_market_series, get_engine_config

logger = logging.getLogger(__name__)


class PrismMarketEngine:
    """
    Engine for analyzing market data.

    This engine:
    - Loads panel data from registry-specified paths
    - Focuses on market series (Yahoo Finance data)
    - Provides market-specific analysis methods (returns, volatility, risk)
    """

    name = "market"
    description = "Market data analysis engine"

    # Market asset categories
    CATEGORIES = {
        'equity': ['spy_spy', 'qqq_qqq', 'iwm_iwm'],
        'commodity': ['gld_gld', 'slv_slv', 'uso_uso'],
        'fixed_income': ['bnd_bnd', 'tlt_tlt', 'shy_shy', 'ief_ief', 'tip_tip'],
        'credit': ['lqd_lqd', 'hyg_hyg'],
        'defensive': ['xlu_xlu'],
    }

    # Risk-free rate proxy
    RISK_FREE_PROXY = 'shy_shy'

    def __init__(
        self,
        project_root: Optional[Path] = None,
        config: Optional[Dict] = None
    ):
        """
        Initialize the market engine.

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
        Load and filter panel data for market series only.

        Returns:
            DataFrame with market series
        """
        # Load from registry-specified path
        full_panel = self.registry.load_panel(panel_type='master')

        # Get market series columns from registry
        market_cols = self.registry.get_market_series()

        # Filter to available columns
        available_cols = [c for c in market_cols if c in full_panel.columns]

        if not available_cols:
            logger.warning("No market columns found in panel. Using all columns.")
            return full_panel

        logger.info(f"Loaded {len(available_cols)} market series from panel")
        return full_panel[available_cols]

    def reload_panel(self) -> pd.DataFrame:
        """Force reload of panel data."""
        self._panel = None
        return self.panel

    def analyze(
        self,
        lookback_years: Optional[int] = None,
        tickers: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run market analysis.

        Args:
            lookback_years: Number of years to analyze (default from config)
            tickers: List of tickers to analyze (default: all available)
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

        # Filter to specific tickers if provided
        if tickers:
            available = [t for t in tickers if t in df.columns]
            df = df[available]

        # Drop rows with too many NaNs
        nan_threshold = self._default_config.get('nan_threshold', 0.5)
        df = df.dropna(thresh=int(len(df.columns) * (1 - nan_threshold)))

        # Calculate returns
        returns = df.pct_change().dropna()

        # Run analysis
        results = {
            'timestamp': datetime.now().isoformat(),
            'engine': self.name,
            'n_assets': len(df.columns),
            'n_observations': len(df),
            'date_range': {
                'start': str(df.index.min()) if len(df) > 0 else None,
                'end': str(df.index.max()) if len(df) > 0 else None,
            },
            'assets_analyzed': list(df.columns),
        }

        # Calculate return statistics
        results['return_statistics'] = self._compute_return_stats(returns)

        # Calculate risk metrics
        results['risk_metrics'] = self._compute_risk_metrics(returns)

        # Calculate correlations
        results['correlations'] = self._compute_correlations(returns)

        # Rank assets
        results['top_indicators'] = self._rank_assets(returns)

        # Category analysis
        results['category_analysis'] = self._analyze_categories(returns)

        self._last_results = results
        return results

    def _compute_return_stats(self, returns: pd.DataFrame) -> Dict[str, Any]:
        """Compute return statistics."""
        stats = {}

        for col in returns.columns:
            series = returns[col].dropna()
            if len(series) == 0:
                continue

            annualized_return = series.mean() * 252
            annualized_vol = series.std() * np.sqrt(252)

            stats[col] = {
                'annualized_return': float(annualized_return),
                'annualized_volatility': float(annualized_vol),
                'sharpe_ratio': float(annualized_return / annualized_vol) if annualized_vol > 0 else 0,
                'max_daily_gain': float(series.max()),
                'max_daily_loss': float(series.min()),
                'positive_days_pct': float((series > 0).mean() * 100),
            }

        return stats

    def _compute_risk_metrics(self, returns: pd.DataFrame) -> Dict[str, Any]:
        """Compute risk metrics."""
        risk = {}

        for col in returns.columns:
            series = returns[col].dropna()
            if len(series) == 0:
                continue

            # VaR and CVaR (Expected Shortfall)
            var_95 = float(series.quantile(0.05))
            cvar_95 = float(series[series <= var_95].mean()) if (series <= var_95).any() else var_95

            # Maximum drawdown
            cumulative = (1 + series).cumprod()
            running_max = cumulative.cummax()
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = float(drawdown.min())

            # Skewness and Kurtosis
            skew = float(series.skew())
            kurt = float(series.kurtosis())

            risk[col] = {
                'var_95': var_95,
                'cvar_95': cvar_95,
                'max_drawdown': max_drawdown,
                'skewness': skew,
                'kurtosis': kurt,
                'downside_volatility': float(series[series < 0].std() * np.sqrt(252)) if (series < 0).any() else 0,
            }

        return risk

    def _compute_correlations(self, returns: pd.DataFrame) -> Dict[str, Any]:
        """Compute correlation analysis."""
        corr_matrix = returns.corr()

        # Find diversification opportunities (low correlations)
        low_corr = []
        for i, col1 in enumerate(corr_matrix.columns):
            for j, col2 in enumerate(corr_matrix.columns):
                if i < j:
                    corr_val = corr_matrix.loc[col1, col2]
                    if abs(corr_val) < 0.3:
                        low_corr.append({
                            'asset1': col1,
                            'asset2': col2,
                            'correlation': float(corr_val)
                        })

        return {
            'low_correlations': sorted(low_corr, key=lambda x: abs(x['correlation']))[:10],
            'avg_correlation': float(corr_matrix.values[np.triu_indices_from(corr_matrix.values, 1)].mean()),
            'max_correlation': float(corr_matrix.values[np.triu_indices_from(corr_matrix.values, 1)].max()),
        }

    def _rank_assets(self, returns: pd.DataFrame) -> List[Dict[str, Any]]:
        """Rank assets by risk-adjusted returns."""
        rankings = []

        for col in returns.columns:
            series = returns[col].dropna()
            if len(series) < 20:
                continue

            annualized_return = series.mean() * 252
            annualized_vol = series.std() * np.sqrt(252)
            sharpe = annualized_return / annualized_vol if annualized_vol > 0 else 0

            # Sortino ratio (downside risk adjusted)
            downside_returns = series[series < 0]
            downside_vol = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else annualized_vol
            sortino = annualized_return / downside_vol if downside_vol > 0 else 0

            rankings.append({
                'indicator': col,
                'score': float(sharpe),  # Use Sharpe as primary score
                'sharpe_ratio': float(sharpe),
                'sortino_ratio': float(sortino),
                'annualized_return': float(annualized_return),
                'annualized_volatility': float(annualized_vol),
            })

        # Sort by Sharpe ratio
        rankings.sort(key=lambda x: x['score'], reverse=True)

        # Add rank
        for i, r in enumerate(rankings):
            r['rank'] = i + 1

        return rankings[:10]

    def _analyze_categories(self, returns: pd.DataFrame) -> Dict[str, Any]:
        """Analyze assets by category."""
        category_analysis = {}

        for category, assets in self.CATEGORIES.items():
            available = [a for a in assets if a in returns.columns]
            if not available:
                continue

            cat_returns = returns[available].dropna()
            if len(cat_returns) == 0:
                continue

            # Calculate category-level metrics
            avg_return = cat_returns.mean().mean() * 252
            avg_vol = cat_returns.std().mean() * np.sqrt(252)

            category_analysis[category] = {
                'assets': available,
                'n_assets': len(available),
                'avg_annualized_return': float(avg_return),
                'avg_annualized_volatility': float(avg_vol),
                'avg_sharpe': float(avg_return / avg_vol) if avg_vol > 0 else 0,
                'intra_category_correlation': float(cat_returns.corr().values.mean()),
            }

        return category_analysis

    def get_asset_details(self, ticker: str) -> Dict[str, Any]:
        """
        Get detailed analysis for a specific asset.

        Args:
            ticker: Asset ticker

        Returns:
            Dictionary with asset details
        """
        if ticker not in self.panel.columns:
            return {'error': f'Asset {ticker} not found in panel'}

        prices = self.panel[ticker].dropna()
        if len(prices) == 0:
            return {'error': f'No data for asset {ticker}'}

        returns = prices.pct_change().dropna()

        return {
            'ticker': ticker,
            'n_observations': len(prices),
            'date_range': {
                'start': str(prices.index.min()),
                'end': str(prices.index.max()),
            },
            'current_price': float(prices.iloc[-1]),
            'price_statistics': {
                'high': float(prices.max()),
                'low': float(prices.min()),
                'mean': float(prices.mean()),
            },
            'return_statistics': {
                'annualized_return': float(returns.mean() * 252),
                'annualized_volatility': float(returns.std() * np.sqrt(252)),
                'sharpe_ratio': float(returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0,
            },
            'ytd_return': float((prices.iloc[-1] / prices.iloc[0] - 1) * 100),
        }

    def __repr__(self) -> str:
        return f"PrismMarketEngine(assets={len(self.panel.columns) if self._panel is not None else 'not loaded'})"
