"""
Seismometer Utilities - Rolling stats and helper functions
"""

from typing import Optional, List
import pandas as pd
import numpy as np


def compute_rolling_zscore(
    series: pd.Series,
    window: int = 60,
    min_periods: Optional[int] = None
) -> pd.Series:
    """
    Compute rolling z-score for a series.

    Args:
        series: Input time series
        window: Rolling window size
        min_periods: Minimum periods required (default: window // 2)

    Returns:
        Series of rolling z-scores
    """
    if min_periods is None:
        min_periods = window // 2

    rolling_mean = series.rolling(window=window, min_periods=min_periods).mean()
    rolling_std = series.rolling(window=window, min_periods=min_periods).std()

    # Avoid division by zero
    rolling_std = rolling_std.replace(0, np.nan)

    zscore = (series - rolling_mean) / rolling_std

    return zscore


def compute_regime_duration(
    stability: pd.Series,
    threshold: float = 0.5
) -> pd.Series:
    """
    Compute duration (in days) of current stability regime.

    Counts consecutive days above or below threshold.

    Args:
        stability: Series of stability scores (0-1)
        threshold: Threshold dividing stable/unstable regimes

    Returns:
        Series of regime durations (positive = stable, negative = unstable)
    """
    # Create regime indicator: 1 = stable (above threshold), -1 = unstable
    regime = (stability >= threshold).astype(int) * 2 - 1

    # Find regime changes
    regime_change = regime.diff().fillna(0) != 0

    # Create regime groups
    regime_groups = regime_change.cumsum()

    # Count duration within each group
    duration = regime.groupby(regime_groups).cumcount() + 1

    # Apply sign based on regime
    signed_duration = duration * regime

    return signed_duration


def detect_divergence_acceleration(
    scores: pd.Series,
    window: int = 20,
    threshold: float = 0.1
) -> pd.Series:
    """
    Detect acceleration of instability (second-order signal).

    Measures whether instability is accelerating, indicating
    imminent regime break.

    Args:
        scores: Instability scores (higher = more unstable)
        window: Window for computing acceleration
        threshold: Minimum acceleration to flag

    Returns:
        Series of acceleration values (positive = accelerating instability)
    """
    # First derivative: rate of change
    velocity = scores.diff()

    # Second derivative: acceleration
    acceleration = velocity.diff()

    # Smooth with rolling mean
    smooth_acceleration = acceleration.rolling(window=window, min_periods=5).mean()

    return smooth_acceleration


def compute_exponential_moving_score(
    scores: pd.Series,
    span: int = 20
) -> pd.Series:
    """
    Compute exponentially weighted moving average of scores.

    Useful for smoothing noisy instability signals.

    Args:
        scores: Raw instability scores
        span: EMA span (higher = smoother)

    Returns:
        EMA-smoothed scores
    """
    return scores.ewm(span=span, adjust=False).mean()


def find_regime_transitions(
    stability: pd.Series,
    high_threshold: float = 0.7,
    low_threshold: float = 0.5
) -> pd.DataFrame:
    """
    Identify regime transition points.

    Detects when stability crosses from stable to unstable territory.

    Args:
        stability: Series of stability scores
        high_threshold: Upper threshold (stable → elevated)
        low_threshold: Lower threshold (elevated → pre-instability)

    Returns:
        DataFrame with transition dates and types
    """
    transitions = []

    prev_level = 'stable'

    for date, value in stability.items():
        if pd.isna(value):
            continue

        # Determine current level
        if value >= high_threshold:
            current_level = 'stable'
        elif value >= low_threshold:
            current_level = 'elevated'
        elif value >= 0.3:
            current_level = 'pre_instability'
        else:
            current_level = 'high_risk'

        # Check for transition
        if current_level != prev_level:
            transitions.append({
                'date': date,
                'from_level': prev_level,
                'to_level': current_level,
                'stability_value': value,
            })
            prev_level = current_level

    return pd.DataFrame(transitions)


def compute_stability_percentiles(
    stability: pd.Series,
    reference_period: Optional[pd.Series] = None
) -> pd.DataFrame:
    """
    Compute percentile ranks for stability scores.

    Args:
        stability: Series of stability scores
        reference_period: Optional reference period for percentile calculation

    Returns:
        DataFrame with stability values and percentile ranks
    """
    if reference_period is None:
        reference_period = stability

    # Compute percentile for each value
    percentiles = stability.apply(
        lambda x: (reference_period < x).sum() / len(reference_period) * 100
        if not pd.isna(x) else np.nan
    )

    return pd.DataFrame({
        'stability': stability,
        'percentile': percentiles
    })


def interpolate_gaps(
    series: pd.Series,
    max_gap: int = 5,
    method: str = 'linear'
) -> pd.Series:
    """
    Interpolate small gaps in a time series.

    Args:
        series: Input series with potential gaps
        max_gap: Maximum gap size to interpolate
        method: Interpolation method ('linear', 'ffill', 'bfill')

    Returns:
        Series with gaps filled
    """
    # Identify gap sizes
    is_nan = series.isna()
    gap_groups = (~is_nan).cumsum()
    gap_sizes = is_nan.groupby(gap_groups).transform('sum')

    # Only interpolate gaps within max_gap
    interpolate_mask = is_nan & (gap_sizes <= max_gap)

    result = series.copy()

    if method == 'linear':
        result = result.interpolate(method='linear')
    elif method == 'ffill':
        result = result.ffill()
    elif method == 'bfill':
        result = result.bfill()

    # Restore large gaps as NaN
    large_gap_mask = is_nan & ~interpolate_mask
    result[large_gap_mask] = np.nan

    return result


def resample_to_frequency(
    df: pd.DataFrame,
    freq: str = 'W',
    agg_method: str = 'last'
) -> pd.DataFrame:
    """
    Resample DataFrame to different frequency.

    Args:
        df: Input DataFrame with DatetimeIndex
        freq: Target frequency ('D', 'W', 'M', 'Q')
        agg_method: Aggregation method ('last', 'mean', 'median')

    Returns:
        Resampled DataFrame
    """
    if agg_method == 'last':
        return df.resample(freq).last()
    elif agg_method == 'mean':
        return df.resample(freq).mean()
    elif agg_method == 'median':
        return df.resample(freq).median()
    else:
        raise ValueError(f"Unknown aggregation method: {agg_method}")


def get_default_features() -> List[str]:
    """
    Get list of default features for seismometer.

    Returns:
        List of default feature names from indicator_values table
    """
    return [
        # Spreads
        't10y2y',           # 10Y-2Y Treasury spread
        't10y3m',           # 10Y-3M Treasury spread
        'credit_spread',    # Credit spread (BAA-AAA or similar)
        'real_yield_10y',   # Real yield 10Y

        # Technicals
        'spy_mom_12m',      # S&P 500 12-month momentum
        'spy_vol_20d',      # S&P 500 20-day volatility
        'spy_rsi_14',       # S&P 500 14-day RSI

        # Sector metrics
        'sector_breadth',   # Breadth of sector performance
        'sector_mom_diff',  # Sector momentum dispersion

        # Macro
        'liquidity_ratio',  # Liquidity measure
    ]
