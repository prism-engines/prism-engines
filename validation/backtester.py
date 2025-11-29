"""
Backtester - Historical validation of indicator predictions
"""

from typing import Dict, Any, List, Optional, Callable
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class Backtester:
    """
    Backtest indicator rankings against historical outcomes.

    Tests if highly-ranked indicators actually had predictive value
    for future market movements or regime changes.
    """

    def __init__(self, target_col: Optional[str] = None):
        """
        Initialize backtester.

        Args:
            target_col: Target column for prediction (e.g., 'spy' for S&P 500)
        """
        self.target_col = target_col
        self.results: List[Dict] = []

    def rolling_backtest(
        self,
        df: pd.DataFrame,
        lens_class: type,
        train_window: int = 252,
        test_window: int = 21,
        step_size: int = 21,
        top_n: int = 5,
        **lens_kwargs
    ) -> Dict[str, Any]:
        """
        Perform rolling window backtest.

        Args:
            df: Input DataFrame with date column
            lens_class: Lens class to test
            train_window: Training window size (days)
            test_window: Test/prediction window size (days)
            step_size: Step size between windows (days)
            top_n: Number of top indicators to track
            **lens_kwargs: Arguments passed to lens

        Returns:
            Dictionary with backtest results
        """
        if 'date' not in df.columns:
            raise ValueError("DataFrame must have 'date' column")

        df = df.sort_values('date').reset_index(drop=True)
        value_cols = [c for c in df.columns if c != 'date']

        if self.target_col and self.target_col not in value_cols:
            raise ValueError(f"Target column '{self.target_col}' not found")

        target = self.target_col or value_cols[0]

        predictions = []

        # Rolling windows
        for start in range(0, len(df) - train_window - test_window, step_size):
            train_end = start + train_window
            test_end = train_end + test_window

            train_df = df.iloc[start:train_end]
            test_df = df.iloc[train_end:test_end]

            try:
                # Get rankings from training data
                lens = lens_class()
                ranking = lens.rank_indicators(train_df, **lens_kwargs)
                top_indicators = ranking.head(top_n)['indicator'].tolist()

                # Measure performance in test window
                # Use correlation with target as performance metric
                test_performance = {}
                target_return = (test_df[target].iloc[-1] / test_df[target].iloc[0]) - 1

                for ind in top_indicators:
                    if ind in test_df.columns:
                        ind_return = (test_df[ind].iloc[-1] / test_df[ind].iloc[0]) - 1
                        # Correlation of daily returns
                        if len(test_df) > 2:
                            corr = test_df[target].pct_change().corr(
                                test_df[ind].pct_change()
                            )
                        else:
                            corr = np.nan
                        test_performance[ind] = {
                            'return': ind_return,
                            'correlation': corr
                        }

                predictions.append({
                    'train_start': train_df['date'].iloc[0],
                    'train_end': train_df['date'].iloc[-1],
                    'test_start': test_df['date'].iloc[0],
                    'test_end': test_df['date'].iloc[-1],
                    'top_indicators': top_indicators,
                    'target_return': target_return,
                    'test_performance': test_performance
                })

            except Exception as e:
                logger.warning(f"Error in window {start}: {e}")
                continue

        # Aggregate results
        if not predictions:
            return {'error': 'No valid predictions generated'}

        # Track how often each indicator was in top N
        indicator_frequency = defaultdict(int)
        indicator_performance = defaultdict(list)

        for pred in predictions:
            for ind in pred['top_indicators']:
                indicator_frequency[ind] += 1
                if ind in pred['test_performance']:
                    perf = pred['test_performance'][ind]
                    if not np.isnan(perf.get('correlation', np.nan)):
                        indicator_performance[ind].append(perf['correlation'])

        # Compute summary statistics
        summary = {
            'lens': lens_class.__name__,
            'n_windows': len(predictions),
            'train_window': train_window,
            'test_window': test_window,
            'target': target,
            'indicator_frequency': dict(indicator_frequency),
            'indicator_mean_correlation': {
                ind: np.mean(corrs)
                for ind, corrs in indicator_performance.items()
                if corrs
            },
            'predictions': predictions
        }

        return summary

    def regime_prediction_backtest(
        self,
        df: pd.DataFrame,
        lens_class: type,
        regime_lens_class: type,
        lookahead: int = 5,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Test if indicator rankings predict regime changes.

        Args:
            df: Input DataFrame
            lens_class: Lens for indicator ranking
            regime_lens_class: Lens for regime detection
            lookahead: Days to look ahead for regime change

        Returns:
            Dictionary with regime prediction results
        """
        # First, detect regimes on full data
        regime_lens = regime_lens_class()
        regime_result = regime_lens.analyze(df)
        regime_labels = regime_result.get('regime_labels', [])

        if not regime_labels:
            return {'error': 'Could not detect regimes'}

        # Find regime change points
        change_points = []
        for i in range(1, len(regime_labels)):
            if regime_labels[i] != regime_labels[i-1]:
                change_points.append(i)

        # For each change point, look at indicator rankings before
        pre_change_rankings = []

        for cp in change_points:
            if cp < 60:  # Need enough history
                continue

            # Get rankings from window before change
            window_df = df.iloc[cp-60:cp]

            try:
                lens = lens_class()
                ranking = lens.rank_indicators(window_df)
                top_5 = ranking.head(5)['indicator'].tolist()

                pre_change_rankings.append({
                    'change_point': cp,
                    'date': df.iloc[cp]['date'] if 'date' in df.columns else cp,
                    'from_regime': regime_labels[cp-1],
                    'to_regime': regime_labels[cp],
                    'top_indicators_before': top_5
                })
            except Exception:
                continue

        # Analyze which indicators most frequently appear before changes
        pre_change_frequency = defaultdict(int)
        for pred in pre_change_rankings:
            for ind in pred['top_indicators_before']:
                pre_change_frequency[ind] += 1

        return {
            'lens': lens_class.__name__,
            'n_regime_changes': len(change_points),
            'n_analyzed': len(pre_change_rankings),
            'pre_change_indicator_frequency': dict(pre_change_frequency),
            'change_details': pre_change_rankings
        }

    def compute_information_coefficient(
        self,
        df: pd.DataFrame,
        lens_class: type,
        forward_period: int = 21
    ) -> Dict[str, float]:
        """
        Compute Information Coefficient (IC) for indicator rankings.

        IC measures correlation between predicted rankings and actual
        forward returns.

        Args:
            df: Input DataFrame
            lens_class: Lens class
            forward_period: Forward return period (days)

        Returns:
            Dictionary with IC statistics
        """
        value_cols = [c for c in df.columns if c != 'date']

        # Compute forward returns for each indicator
        forward_returns = {}
        for col in value_cols:
            forward_returns[col] = df[col].pct_change(forward_period).shift(-forward_period)

        ics = []

        # Rolling IC calculation
        window = 60
        for start in range(0, len(df) - window - forward_period, 21):
            window_df = df.iloc[start:start+window]

            try:
                lens = lens_class()
                ranking = lens.rank_indicators(window_df)

                # Get scores and forward returns at end of window
                end_idx = start + window

                scores = []
                returns = []

                for _, row in ranking.iterrows():
                    ind = row['indicator']
                    if ind in forward_returns:
                        fwd_ret = forward_returns[ind].iloc[end_idx]
                        if not np.isnan(fwd_ret):
                            scores.append(row['score'])
                            returns.append(fwd_ret)

                if len(scores) >= 5:
                    ic = np.corrcoef(scores, returns)[0, 1]
                    if not np.isnan(ic):
                        ics.append(ic)

            except Exception:
                continue

        if not ics:
            return {'error': 'Could not compute IC'}

        return {
            'lens': lens_class.__name__,
            'mean_ic': np.mean(ics),
            'std_ic': np.std(ics),
            'ic_ir': np.mean(ics) / (np.std(ics) + 1e-10),  # IC Information Ratio
            'pct_positive_ic': np.mean([ic > 0 for ic in ics]),
            'n_periods': len(ics)
        }


# Helper for defaultdict
from collections import defaultdict
