"""
PRISM Traditional Methods Baseline

Implements standard quantitative methods for comparison against PRISM.
These are the methods practitioners actually use - PRISM must beat them.

Categories:
1. Regime Detection (thresholds, Markov switching)
2. Correlation Analysis (rolling correlation)
3. Trend Detection (moving averages)
4. Volatility Regimes (VIX-based)
5. Lead/Lag Analysis (simple cross-correlation)
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import date

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


# =============================================================================
# Regime Detection - Traditional
# =============================================================================

class ThresholdRegimes:
    """
    Simple threshold-based regime detection.
    
    This is what most practitioners actually use:
    - VIX > 25 = "risk-off"
    - Yield curve < 0 = "recession warning"
    - etc.
    """
    
    def __init__(self):
        self.default_thresholds = {
            "VIX": {"low": 15, "high": 25},
            "T10Y2Y": {"low": 0, "high": 1.0},
        }
    
    def detect_regimes(
        self,
        series: pd.Series,
        thresholds: Dict[str, float],
    ) -> pd.Series:
        """
        Simple threshold regime detection.
        
        Args:
            series: Time series data
            thresholds: {"low": x, "high": y} for 3-regime split
        
        Returns:
            Series with regime labels (0=low, 1=normal, 2=high)
        """
        low = thresholds.get("low", series.quantile(0.25))
        high = thresholds.get("high", series.quantile(0.75))
        
        regimes = pd.Series(index=series.index, data=1)
        regimes[series < low] = 0
        regimes[series > high] = 2
        
        return regimes
    
    def vix_regime(self, vix: pd.Series) -> pd.Series:
        """
        Standard VIX-based risk regime.
        
        Returns:
            0 = Complacent (VIX < 15)
            1 = Normal (15-25)
            2 = Fear (VIX > 25)
            3 = Panic (VIX > 35)
        """
        regimes = pd.Series(index=vix.index, data=1)
        regimes[vix < 15] = 0
        regimes[vix > 25] = 2
        regimes[vix > 35] = 3
        return regimes
    
    def yield_curve_regime(self, spread: pd.Series) -> pd.Series:
        """
        Yield curve regime (10Y-2Y spread).
        
        Returns:
            0 = Inverted (< 0) - recession warning
            1 = Flat (0 - 0.5)
            2 = Normal (0.5 - 1.5)
            3 = Steep (> 1.5)
        """
        regimes = pd.Series(index=spread.index, data=2)
        regimes[spread < 0] = 0
        regimes[(spread >= 0) & (spread < 0.5)] = 1
        regimes[spread > 1.5] = 3
        return regimes


class MarkovSwitchingBaseline:
    """
    Hamilton-style Markov Regime Switching.
    
    The academic gold standard for regime detection since 1989.
    Uses statsmodels if available, otherwise simple approximation.
    """
    
    def __init__(self):
        self._has_statsmodels = False
        try:
            from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
            self._has_statsmodels = True
        except ImportError:
            logger.warning("statsmodels not available. Using simplified Markov switching.")
    
    def fit(
        self,
        returns: pd.Series,
        n_regimes: int = 2,
    ) -> Dict[str, Any]:
        """
        Fit Markov switching model.
        
        Args:
            returns: Return series
            n_regimes: Number of regimes (typically 2: bull/bear)
        
        Returns:
            Dict with regime probabilities and parameters
        """
        if self._has_statsmodels:
            return self._fit_statsmodels(returns, n_regimes)
        else:
            return self._fit_simple(returns, n_regimes)
    
    def _fit_statsmodels(
        self,
        returns: pd.Series,
        n_regimes: int
    ) -> Dict[str, Any]:
        """Fit using statsmodels."""
        from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
        
        returns_clean = returns.dropna()
        
        try:
            model = MarkovRegression(
                returns_clean,
                k_regimes=n_regimes,
                switching_variance=True,
            )
            
            result = model.fit(disp=False)
            smoothed_probs = result.smoothed_marginal_probabilities
            regimes = smoothed_probs.idxmax(axis=1)
            
            return {
                "regimes": regimes,
                "probabilities": smoothed_probs,
                "transition_matrix": result.regime_transition,
                "log_likelihood": result.llf,
                "aic": result.aic,
                "bic": result.bic,
                "method": "statsmodels",
            }
        except Exception as e:
            logger.warning(f"Statsmodels fitting failed: {e}. Falling back to simple method.")
            return self._fit_simple(returns, n_regimes)
    
    def _fit_simple(
        self,
        returns: pd.Series,
        n_regimes: int
    ) -> Dict[str, Any]:
        """
        Simple approximation using rolling volatility regimes.
        Not a true Markov model, but captures similar intuition.
        """
        vol = returns.rolling(20).std() * np.sqrt(252)
        thresholds = [vol.quantile(i / n_regimes) for i in range(1, n_regimes)]
        
        regimes = pd.Series(index=returns.index, data=0)
        for i, thresh in enumerate(thresholds):
            regimes[vol > thresh] = i + 1
        
        return {
            "regimes": regimes,
            "probabilities": None,
            "transition_matrix": self._estimate_transitions(regimes, n_regimes),
            "method": "simplified_volatility",
        }
    
    def _estimate_transitions(
        self,
        regimes: pd.Series,
        n_regimes: int
    ) -> np.ndarray:
        """Estimate transition matrix from regime sequence."""
        trans = np.zeros((n_regimes, n_regimes))
        regimes_clean = regimes.dropna().astype(int)
        
        for i in range(len(regimes_clean) - 1):
            from_state = regimes_clean.iloc[i]
            to_state = regimes_clean.iloc[i + 1]
            if 0 <= from_state < n_regimes and 0 <= to_state < n_regimes:
                trans[from_state, to_state] += 1
        
        row_sums = trans.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        return trans / row_sums


# =============================================================================
# Correlation Analysis - Traditional
# =============================================================================

class RollingCorrelation:
    """
    Standard rolling correlation analysis.
    The baseline that everyone uses.
    """
    
    def compute(
        self,
        df: pd.DataFrame,
        window: int = 60,
    ) -> Dict[str, pd.DataFrame]:
        """
        Compute rolling correlation matrix.
        
        Args:
            df: DataFrame with indicators as columns
            window: Rolling window size
        
        Returns:
            Dict with correlation time series
        """
        correlations = {}
        columns = df.columns.tolist()
        
        for i, col1 in enumerate(columns):
            for col2 in columns[i+1:]:
                key = f"{col1}__{col2}"
                correlations[key] = df[col1].rolling(window).corr(df[col2])
        
        corr_df = pd.DataFrame(correlations)
        
        avg_corr = corr_df.mean(axis=1)
        max_corr = corr_df.max(axis=1)
        min_corr = corr_df.min(axis=1)
        
        return {
            "pairwise": corr_df,
            "average": avg_corr,
            "max": max_corr,
            "min": min_corr,
            "dispersion": max_corr - min_corr,
        }
    
    def detect_correlation_breakdown(
        self,
        corr_series: pd.Series,
        threshold: float = 0.8,
        lookback: int = 252,
    ) -> pd.Series:
        """
        Detect when correlations spike (crisis indicator).
        Traditional view: "correlations go to 1 in a crisis"
        """
        rolling_pct = corr_series.rolling(lookback).apply(
            lambda x: stats.percentileofscore(x, x.iloc[-1]) / 100
        )
        
        breakdown = rolling_pct > threshold
        return breakdown


# =============================================================================
# Trend Detection - Traditional
# =============================================================================

class MovingAverageTrend:
    """
    Classic moving average trend detection.
    The most widely used technical indicator.
    """
    
    def golden_death_cross(
        self,
        prices: pd.Series,
        fast: int = 50,
        slow: int = 200,
    ) -> pd.DataFrame:
        """
        50/200 day moving average crossover.
        
        Returns:
            DataFrame with signals
        """
        ma_fast = prices.rolling(fast).mean()
        ma_slow = prices.rolling(slow).mean()
        
        signal = pd.Series(index=prices.index, data=0)
        signal[ma_fast > ma_slow] = 1
        signal[ma_fast < ma_slow] = -1
        
        crosses = signal.diff()
        golden_cross = crosses == 2
        death_cross = crosses == -2
        
        return pd.DataFrame({
            "ma_fast": ma_fast,
            "ma_slow": ma_slow,
            "signal": signal,
            "golden_cross": golden_cross,
            "death_cross": death_cross,
        })
    
    def trend_strength(
        self,
        prices: pd.Series,
        window: int = 20,
    ) -> pd.Series:
        """
        Simple trend strength: distance from moving average.
        """
        ma = prices.rolling(window).mean()
        return (prices - ma) / ma


# =============================================================================
# Lead/Lag Analysis - Traditional
# =============================================================================

class SimpleCrossCorrelation:
    """
    Basic lead/lag analysis using cross-correlation.
    """
    
    def find_optimal_lag(
        self,
        x: pd.Series,
        y: pd.Series,
        max_lag: int = 20,
    ) -> Dict[str, Any]:
        """
        Find optimal lag between two series.
        Positive lag means x leads y.
        """
        correlations = []
        
        for lag in range(-max_lag, max_lag + 1):
            if lag < 0:
                corr = x.iloc[-lag:].corr(y.iloc[:lag])
            elif lag > 0:
                corr = x.iloc[:-lag].corr(y.iloc[lag:])
            else:
                corr = x.corr(y)
            
            correlations.append({"lag": lag, "correlation": corr})
        
        df = pd.DataFrame(correlations)
        optimal_idx = df["correlation"].abs().idxmax()
        
        return {
            "optimal_lag": df.loc[optimal_idx, "lag"],
            "optimal_correlation": df.loc[optimal_idx, "correlation"],
            "all_lags": df,
        }


# =============================================================================
# Combined Traditional Analysis
# =============================================================================

class TraditionalAnalysis:
    """
    Combined traditional analysis suite.
    Runs all standard methods for comparison against PRISM.
    """
    
    def __init__(self):
        self.threshold_regimes = ThresholdRegimes()
        self.markov = MarkovSwitchingBaseline()
        self.correlation = RollingCorrelation()
        self.trend = MovingAverageTrend()
        self.leadlag = SimpleCrossCorrelation()
    
    def full_analysis(
        self,
        df: pd.DataFrame,
        vix: Optional[pd.Series] = None,
        yield_curve: Optional[pd.Series] = None,
        reference_price: Optional[pd.Series] = None,
    ) -> Dict[str, Any]:
        """
        Run complete traditional analysis.
        
        Args:
            df: DataFrame with indicators
            vix: VIX series (optional)
            yield_curve: 10Y-2Y spread (optional)
            reference_price: Price series for trend (optional)
        
        Returns:
            Dict with all traditional analysis results
        """
        results = {}
        
        if vix is not None:
            results["vix_regime"] = self.threshold_regimes.vix_regime(vix)
        
        if yield_curve is not None:
            results["yc_regime"] = self.threshold_regimes.yield_curve_regime(yield_curve)
        
        results["rolling_correlation"] = self.correlation.compute(df)
        
        if reference_price is not None:
            results["trend"] = self.trend.golden_death_cross(reference_price)
        
        if reference_price is not None:
            returns = reference_price.pct_change().dropna()
            results["markov"] = self.markov.fit(returns)
        
        return results
    
    def regime_summary(
        self,
        results: Dict[str, Any],
        as_of_date: Optional[date] = None,
    ) -> Dict[str, str]:
        """
        Generate human-readable regime summary.
        This is what a traditional dashboard would show.
        """
        summary = {}
        
        if "vix_regime" in results:
            regime = results["vix_regime"]
            if as_of_date:
                regime = regime.loc[:str(as_of_date)]
            current = regime.iloc[-1]
            labels = {0: "Complacent", 1: "Normal", 2: "Fear", 3: "Panic"}
            summary["VIX Regime"] = labels.get(current, "Unknown")
        
        if "yc_regime" in results:
            regime = results["yc_regime"]
            if as_of_date:
                regime = regime.loc[:str(as_of_date)]
            current = regime.iloc[-1]
            labels = {0: "Inverted", 1: "Flat", 2: "Normal", 3: "Steep"}
            summary["Yield Curve"] = labels.get(current, "Unknown")
        
        if "trend" in results:
            trend = results["trend"]
            if as_of_date:
                trend = trend.loc[:str(as_of_date)]
            current = trend["signal"].iloc[-1]
            summary["Trend"] = "Bullish" if current > 0 else "Bearish"
        
        if "rolling_correlation" in results:
            avg_corr = results["rolling_correlation"]["average"]
            if as_of_date:
                avg_corr = avg_corr.loc[:str(as_of_date)]
            current = avg_corr.iloc[-1]
            summary["Avg Correlation"] = f"{current:.2f}"
        
        return summary
