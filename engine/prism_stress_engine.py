"""
PRISM Stress Engine - Stress Testing and Scenario Analysis

This engine focuses on stress testing and scenario analysis across both
economic and market data. It uses registry-driven configuration to load
panel data and interpret columns.

Usage:
    from engine import PrismStressEngine

    engine = PrismStressEngine()
    results = engine.analyze()
    print(results['stress_scenarios'])
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


class PrismStressEngine:
    """
    Engine for stress testing and scenario analysis.

    This engine:
    - Loads panel data from registry-specified paths
    - Analyzes both economic and market data
    - Provides stress testing and scenario analysis methods
    """

    name = "stress"
    description = "Stress testing and scenario analysis engine"

    # Historical stress events for reference
    HISTORICAL_EVENTS = {
        'gfc_2008': {
            'name': 'Global Financial Crisis',
            'start': '2008-09-01',
            'end': '2009-03-31',
        },
        'covid_2020': {
            'name': 'COVID-19 Crash',
            'start': '2020-02-19',
            'end': '2020-03-23',
        },
        'dot_com_2000': {
            'name': 'Dot-Com Bubble',
            'start': '2000-03-10',
            'end': '2002-10-09',
        },
        'rate_hike_2022': {
            'name': '2022 Rate Hike Cycle',
            'start': '2022-01-01',
            'end': '2022-10-31',
        },
    }

    def __init__(
        self,
        project_root: Optional[Path] = None,
        config: Optional[Dict] = None
    ):
        """
        Initialize the stress engine.

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
        Load full panel data (both economic and market).

        Returns:
            Full DataFrame with all available series
        """
        # Load from registry-specified path
        full_panel = self.registry.load_panel(panel_type='master')

        logger.info(f"Loaded {len(full_panel.columns)} series from panel for stress testing")
        return full_panel

    def reload_panel(self) -> pd.DataFrame:
        """Force reload of panel data."""
        self._panel = None
        return self.panel

    def analyze(
        self,
        scenarios: Optional[List[str]] = None,
        custom_scenarios: Optional[Dict[str, Dict]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run stress analysis.

        Args:
            scenarios: List of historical scenario names to analyze
            custom_scenarios: Custom scenario definitions
            **kwargs: Additional analysis parameters

        Returns:
            Dictionary with stress analysis results
        """
        df = self.panel.copy()

        # Use all historical scenarios if none specified
        if scenarios is None:
            scenarios = list(self.HISTORICAL_EVENTS.keys())

        results = {
            'timestamp': datetime.now().isoformat(),
            'engine': self.name,
            'n_series': len(df.columns),
            'n_observations': len(df),
            'date_range': {
                'start': str(df.index.min()) if len(df) > 0 else None,
                'end': str(df.index.max()) if len(df) > 0 else None,
            },
        }

        # Analyze historical stress scenarios
        results['historical_scenarios'] = {}
        for scenario_key in scenarios:
            if scenario_key in self.HISTORICAL_EVENTS:
                scenario = self.HISTORICAL_EVENTS[scenario_key]
                results['historical_scenarios'][scenario_key] = self._analyze_scenario(
                    df, scenario['start'], scenario['end'], scenario['name']
                )

        # Analyze custom scenarios
        if custom_scenarios:
            results['custom_scenarios'] = {}
            for name, params in custom_scenarios.items():
                results['custom_scenarios'][name] = self._analyze_scenario(
                    df, params.get('start'), params.get('end'), name
                )

        # Calculate tail risk metrics
        results['tail_risk'] = self._compute_tail_risk(df)

        # Calculate regime indicators
        results['regime_indicators'] = self._identify_regime_indicators(df)

        # Calculate correlation breakdown
        results['correlation_breakdown'] = self._analyze_correlation_breakdown(df)

        # Top stress indicators
        results['top_indicators'] = self._rank_stress_indicators(df)

        self._last_results = results
        return results

    def _analyze_scenario(
        self,
        df: pd.DataFrame,
        start: str,
        end: str,
        name: str
    ) -> Dict[str, Any]:
        """Analyze performance during a specific stress scenario."""
        try:
            scenario_df = df.loc[start:end]
        except (KeyError, TypeError):
            return {'error': f'Could not filter data for period {start} to {end}'}

        if len(scenario_df) == 0:
            return {'error': 'No data available for this period'}

        scenario_results = {
            'name': name,
            'period': {'start': start, 'end': end},
            'n_observations': len(scenario_df),
            'series_performance': {},
        }

        for col in scenario_df.columns:
            series = scenario_df[col].dropna()
            if len(series) < 2:
                continue

            # Calculate performance metrics
            start_val = series.iloc[0]
            end_val = series.iloc[-1]
            min_val = series.min()
            max_val = series.max()

            total_change = (end_val - start_val) / start_val if start_val != 0 else 0
            max_drawdown = (min_val - max_val) / max_val if max_val != 0 else 0

            scenario_results['series_performance'][col] = {
                'total_change_pct': float(total_change * 100),
                'max_drawdown_pct': float(max_drawdown * 100),
                'start_value': float(start_val),
                'end_value': float(end_val),
                'min_value': float(min_val),
                'max_value': float(max_val),
            }

        # Summary statistics
        changes = [v['total_change_pct'] for v in scenario_results['series_performance'].values()]
        if changes:
            scenario_results['summary'] = {
                'avg_change_pct': float(np.mean(changes)),
                'worst_performer': min(scenario_results['series_performance'].items(),
                                      key=lambda x: x[1]['total_change_pct'])[0],
                'best_performer': max(scenario_results['series_performance'].items(),
                                     key=lambda x: x[1]['total_change_pct'])[0],
            }

        return scenario_results

    def _compute_tail_risk(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Compute tail risk metrics across all series."""
        returns = df.pct_change().dropna()

        tail_risk = {}
        for col in returns.columns:
            series = returns[col].dropna()
            if len(series) < 100:
                continue

            # Calculate VaR at different confidence levels
            var_99 = float(series.quantile(0.01))
            var_95 = float(series.quantile(0.05))

            # Expected Shortfall (CVaR)
            cvar_99 = float(series[series <= var_99].mean()) if (series <= var_99).any() else var_99
            cvar_95 = float(series[series <= var_95].mean()) if (series <= var_95).any() else var_95

            # Tail ratio (ratio of extreme positive to negative returns)
            extreme_neg = series[series <= series.quantile(0.05)]
            extreme_pos = series[series >= series.quantile(0.95)]
            tail_ratio = float(abs(extreme_pos.mean() / extreme_neg.mean())) if extreme_neg.mean() != 0 else 1

            tail_risk[col] = {
                'var_95': var_95,
                'var_99': var_99,
                'cvar_95': cvar_95,
                'cvar_99': cvar_99,
                'tail_ratio': tail_ratio,
                'kurtosis': float(series.kurtosis()),
            }

        return tail_risk

    def _identify_regime_indicators(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Identify indicators that signal regime changes."""
        returns = df.pct_change().dropna()

        if len(returns) < 252:
            return {'error': 'Insufficient data for regime analysis'}

        regime_indicators = {}

        for col in returns.columns:
            series = returns[col].dropna()
            if len(series) < 252:
                continue

            # Calculate rolling volatility
            rolling_vol = series.rolling(window=21).std() * np.sqrt(252)

            # Calculate volatility regime changes
            vol_mean = rolling_vol.mean()
            vol_std = rolling_vol.std()

            # Count high volatility periods
            high_vol_threshold = vol_mean + 2 * vol_std
            high_vol_periods = (rolling_vol > high_vol_threshold).sum()

            # Volatility of volatility
            vol_of_vol = rolling_vol.std() / rolling_vol.mean() if rolling_vol.mean() > 0 else 0

            regime_indicators[col] = {
                'avg_volatility': float(vol_mean),
                'vol_of_vol': float(vol_of_vol),
                'high_vol_periods': int(high_vol_periods),
                'high_vol_pct': float(high_vol_periods / len(rolling_vol) * 100),
                'current_vol_percentile': float((rolling_vol <= rolling_vol.iloc[-1]).mean() * 100) if len(rolling_vol) > 0 else 0,
            }

        return regime_indicators

    def _analyze_correlation_breakdown(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze how correlations change during stress periods."""
        returns = df.pct_change().dropna()

        if len(returns) < 252:
            return {'error': 'Insufficient data for correlation analysis'}

        # Calculate normal period correlations (middle 80% of returns)
        normal_mask = (returns > returns.quantile(0.1)).all(axis=1) & \
                      (returns < returns.quantile(0.9)).all(axis=1)
        normal_corr = returns[normal_mask].corr() if normal_mask.sum() > 30 else returns.corr()

        # Calculate stress period correlations (bottom 10% of any series)
        stress_mask = (returns < returns.quantile(0.1)).any(axis=1)
        stress_corr = returns[stress_mask].corr() if stress_mask.sum() > 30 else returns.corr()

        # Calculate correlation changes
        corr_change = stress_corr - normal_corr

        # Find biggest correlation increases during stress
        biggest_increases = []
        for i, col1 in enumerate(corr_change.columns):
            for j, col2 in enumerate(corr_change.columns):
                if i < j:
                    change = corr_change.loc[col1, col2]
                    if not pd.isna(change):
                        biggest_increases.append({
                            'pair': f"{col1} / {col2}",
                            'normal_corr': float(normal_corr.loc[col1, col2]),
                            'stress_corr': float(stress_corr.loc[col1, col2]),
                            'change': float(change),
                        })

        biggest_increases.sort(key=lambda x: x['change'], reverse=True)

        return {
            'avg_normal_correlation': float(normal_corr.values[np.triu_indices_from(normal_corr.values, 1)].mean()),
            'avg_stress_correlation': float(stress_corr.values[np.triu_indices_from(stress_corr.values, 1)].mean()),
            'correlation_increase_pairs': biggest_increases[:10],
        }

    def _rank_stress_indicators(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Rank indicators by their stress sensitivity."""
        returns = df.pct_change().dropna()

        rankings = []

        for col in returns.columns:
            series = returns[col].dropna()
            if len(series) < 100:
                continue

            # Calculate stress sensitivity score
            var_95 = abs(series.quantile(0.05))
            kurtosis = abs(series.kurtosis())
            skewness = abs(series.skew())

            # Combine into stress score (higher = more sensitive to stress)
            stress_score = var_95 * 10 + kurtosis * 0.1 + skewness * 0.5

            rankings.append({
                'indicator': col,
                'score': float(stress_score),
                'var_95': float(var_95),
                'kurtosis': float(kurtosis),
                'skewness': float(skewness),
            })

        rankings.sort(key=lambda x: x['score'], reverse=True)

        for i, r in enumerate(rankings):
            r['rank'] = i + 1

        return rankings[:10]

    def run_custom_stress_test(
        self,
        shocks: Dict[str, float],
        propagation_matrix: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """
        Run a custom stress test with specified shocks.

        Args:
            shocks: Dictionary of {indicator: shock_magnitude}
            propagation_matrix: Optional correlation/propagation matrix

        Returns:
            Stress test results
        """
        df = self.panel.copy()

        if propagation_matrix is None:
            # Use historical correlation as propagation mechanism
            returns = df.pct_change().dropna()
            propagation_matrix = returns.corr()

        results = {
            'direct_shocks': shocks,
            'propagated_shocks': {},
            'total_impact': {},
        }

        # Calculate propagated shocks
        for indicator, shock in shocks.items():
            if indicator in propagation_matrix.columns:
                for other in propagation_matrix.columns:
                    if other != indicator:
                        corr = propagation_matrix.loc[indicator, other]
                        propagated = shock * corr
                        if other not in results['propagated_shocks']:
                            results['propagated_shocks'][other] = 0
                        results['propagated_shocks'][other] += propagated

        # Calculate total impact
        for indicator in df.columns:
            direct = shocks.get(indicator, 0)
            propagated = results['propagated_shocks'].get(indicator, 0)
            results['total_impact'][indicator] = {
                'direct': float(direct),
                'propagated': float(propagated),
                'total': float(direct + propagated),
            }

        return results

    def __repr__(self) -> str:
        return f"PrismStressEngine(series={len(self.panel.columns) if self._panel is not None else 'not loaded'})"
