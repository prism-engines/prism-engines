"""
Synthetic Data Scenarios for PRISM Engine Benchmarks

This module provides generators for creating small synthetic DataFrame panels
with known statistical properties. These are used for correctness testing
of lenses and the temporal engine.

Scenarios:
- Sine wave: Single frequency, clean cycle
- Trend + noise: Upward drift with Gaussian noise
- Step change: Regime shift
- White noise: No structure (random baseline)
- Correlated pairs: For coherence/influence tests
"""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import pandas as pd


@dataclass
class SyntheticScenario:
    """Container for a synthetic test scenario."""

    name: str
    description: str
    data: pd.DataFrame
    expected_properties: dict = field(default_factory=dict)

    @property
    def n_rows(self) -> int:
        return len(self.data)

    @property
    def n_columns(self) -> int:
        # Exclude 'date' column from count
        return len([c for c in self.data.columns if c != "date"])

    @property
    def columns(self) -> list[str]:
        return [c for c in self.data.columns if c != "date"]


def _create_date_index(n_periods: int, freq: str = "D", start: str = "2020-01-01") -> pd.DatetimeIndex:
    """Create a datetime index for synthetic data."""
    return pd.date_range(start=start, periods=n_periods, freq=freq)


def _make_dataframe_with_date(data: dict, n_periods: int, start: str = "2020-01-01") -> pd.DataFrame:
    """Create a DataFrame with a 'date' column from data dictionary.

    The lenses expect a 'date' column, not a DatetimeIndex.
    """
    dates = pd.date_range(start=start, periods=n_periods, freq="D")
    df = pd.DataFrame(data)
    df.insert(0, "date", dates)
    return df


def generate_sine_wave(
    n_periods: int = 252,
    n_columns: int = 5,
    frequency: float = 0.02,
    amplitude: float = 1.0,
    phase_shift: float = 0.0,
    noise_std: float = 0.0,
    seed: Optional[int] = None,
) -> SyntheticScenario:
    """
    Generate a panel with sinusoidal signals.

    Each column has a sine wave with optional phase offset.
    Useful for testing periodicity detection and spectral lenses.

    Parameters
    ----------
    n_periods : int
        Number of time periods (rows).
    n_columns : int
        Number of indicators (columns).
    frequency : float
        Sine wave frequency (cycles per period).
    amplitude : float
        Wave amplitude.
    phase_shift : float
        Phase shift between consecutive columns.
    noise_std : float
        Standard deviation of additive Gaussian noise.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    SyntheticScenario
        Container with data and expected properties.
    """
    if seed is not None:
        np.random.seed(seed)

    t = np.arange(n_periods)

    data = {}
    for i in range(n_columns):
        phase = phase_shift * i
        signal = amplitude * np.sin(2 * np.pi * frequency * t + phase)
        if noise_std > 0:
            signal += np.random.normal(0, noise_std, n_periods)
        data[f"sine_{i+1}"] = signal

    df = _make_dataframe_with_date(data, n_periods)

    return SyntheticScenario(
        name="sine_wave",
        description=f"Sinusoidal signals with frequency={frequency}, amplitude={amplitude}",
        data=df,
        expected_properties={
            "has_periodicity": True,
            "dominant_frequency": frequency,
            "amplitude": amplitude,
            "cross_correlation": "high" if phase_shift < 0.5 else "moderate",
        }
    )


def generate_trend_noise(
    n_periods: int = 252,
    n_columns: int = 5,
    trend_slope: float = 0.01,
    noise_std: float = 0.1,
    trend_variation: float = 0.0,
    seed: Optional[int] = None,
) -> SyntheticScenario:
    """
    Generate a panel with upward trend plus Gaussian noise.

    Each column has a linear trend with additive noise.
    Useful for testing trend detection and decomposition lenses.

    Parameters
    ----------
    n_periods : int
        Number of time periods (rows).
    n_columns : int
        Number of indicators (columns).
    trend_slope : float
        Slope of the linear trend per period.
    noise_std : float
        Standard deviation of additive Gaussian noise.
    trend_variation : float
        Variation in slope between columns (0 = same slope for all).
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    SyntheticScenario
        Container with data and expected properties.
    """
    if seed is not None:
        np.random.seed(seed)

    t = np.arange(n_periods)

    data = {}
    for i in range(n_columns):
        slope = trend_slope + (np.random.uniform(-1, 1) * trend_variation if trend_variation > 0 else 0)
        trend = slope * t
        noise = np.random.normal(0, noise_std, n_periods)
        data[f"trend_{i+1}"] = trend + noise

    df = _make_dataframe_with_date(data, n_periods)

    # Calculate signal-to-noise ratio
    signal_variance = (trend_slope * n_periods / 2) ** 2 / 3  # Variance of linear trend
    snr = signal_variance / (noise_std ** 2) if noise_std > 0 else float('inf')

    return SyntheticScenario(
        name="trend_noise",
        description=f"Linear trend (slope={trend_slope}) with Gaussian noise (std={noise_std})",
        data=df,
        expected_properties={
            "has_trend": True,
            "trend_direction": "positive" if trend_slope > 0 else "negative",
            "signal_to_noise_ratio": snr,
            "trend_detectable": snr > 1.0,
        }
    )


def generate_step_change(
    n_periods: int = 252,
    n_columns: int = 5,
    change_point: Optional[int] = None,
    level_before: float = 0.0,
    level_after: float = 1.0,
    noise_std: float = 0.1,
    transition_width: int = 0,
    seed: Optional[int] = None,
) -> SyntheticScenario:
    """
    Generate a panel with a regime shift (step change).

    Each column transitions from one level to another at a change point.
    Useful for testing regime detection and volatility lenses.

    Parameters
    ----------
    n_periods : int
        Number of time periods (rows).
    n_columns : int
        Number of indicators (columns).
    change_point : int, optional
        Index where regime shift occurs (default: middle).
    level_before : float
        Mean level before the change point.
    level_after : float
        Mean level after the change point.
    noise_std : float
        Standard deviation of additive Gaussian noise.
    transition_width : int
        Width of transition zone (0 = instant step).
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    SyntheticScenario
        Container with data and expected properties.
    """
    if seed is not None:
        np.random.seed(seed)

    if change_point is None:
        change_point = n_periods // 2

    dates = pd.date_range(start="2020-01-01", periods=n_periods, freq="D")

    data = {}
    for i in range(n_columns):
        signal = np.zeros(n_periods)

        if transition_width == 0:
            # Instant step change
            signal[:change_point] = level_before
            signal[change_point:] = level_after
        else:
            # Smooth transition
            for t in range(n_periods):
                if t < change_point - transition_width // 2:
                    signal[t] = level_before
                elif t > change_point + transition_width // 2:
                    signal[t] = level_after
                else:
                    # Sigmoid transition
                    x = (t - change_point) / (transition_width / 4)
                    signal[t] = level_before + (level_after - level_before) / (1 + np.exp(-x))

        noise = np.random.normal(0, noise_std, n_periods)
        data[f"step_{i+1}"] = signal + noise

    df = _make_dataframe_with_date(data, n_periods)

    step_magnitude = abs(level_after - level_before)

    return SyntheticScenario(
        name="step_change",
        description=f"Regime shift at period {change_point}: {level_before} -> {level_after}",
        data=df,
        expected_properties={
            "has_regime_change": True,
            "change_point_index": change_point,
            "change_point_date": str(dates[change_point].date()),
            "step_magnitude": step_magnitude,
            "step_detectable": step_magnitude > 2 * noise_std,
        }
    )


def generate_white_noise(
    n_periods: int = 252,
    n_columns: int = 5,
    mean: float = 0.0,
    std: float = 1.0,
    seed: Optional[int] = None,
) -> SyntheticScenario:
    """
    Generate a panel of pure white noise (no structure).

    Each column is independent Gaussian white noise.
    Useful as a null/baseline case - lenses should show no structure.

    Parameters
    ----------
    n_periods : int
        Number of time periods (rows).
    n_columns : int
        Number of indicators (columns).
    mean : float
        Mean of the noise distribution.
    std : float
        Standard deviation of the noise.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    SyntheticScenario
        Container with data and expected properties.
    """
    if seed is not None:
        np.random.seed(seed)

    data = {}
    for i in range(n_columns):
        data[f"noise_{i+1}"] = np.random.normal(mean, std, n_periods)

    df = _make_dataframe_with_date(data, n_periods)

    return SyntheticScenario(
        name="white_noise",
        description=f"Pure white noise (mean={mean}, std={std})",
        data=df,
        expected_properties={
            "has_structure": False,
            "has_trend": False,
            "has_periodicity": False,
            "cross_correlation": "none",
            "expected_hurst": 0.5,  # Random walk behavior
        }
    )


def generate_correlated_pairs(
    n_periods: int = 252,
    n_pairs: int = 3,
    correlation: float = 0.8,
    noise_std: float = 0.1,
    include_uncorrelated: bool = True,
    seed: Optional[int] = None,
) -> SyntheticScenario:
    """
    Generate a panel with correlated and uncorrelated indicator pairs.

    Creates pairs of indicators with specified correlation, plus optional
    uncorrelated indicators. Useful for testing coherence and influence lenses.

    Parameters
    ----------
    n_periods : int
        Number of time periods (rows).
    n_pairs : int
        Number of correlated pairs to generate.
    correlation : float
        Target correlation between pairs (0 to 1).
    noise_std : float
        Additional noise standard deviation.
    include_uncorrelated : bool
        Whether to include uncorrelated control indicators.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    SyntheticScenario
        Container with data and expected properties.
    """
    if seed is not None:
        np.random.seed(seed)

    data = {}

    # Generate correlated pairs using Cholesky decomposition
    for pair_idx in range(n_pairs):
        # Generate base signal
        base = np.random.normal(0, 1, n_periods)

        # Generate correlated signal
        independent = np.random.normal(0, 1, n_periods)
        correlated = correlation * base + np.sqrt(1 - correlation**2) * independent

        # Add noise
        if noise_std > 0:
            base += np.random.normal(0, noise_std, n_periods)
            correlated += np.random.normal(0, noise_std, n_periods)

        data[f"pair{pair_idx+1}_a"] = base
        data[f"pair{pair_idx+1}_b"] = correlated

    # Add uncorrelated control indicators
    if include_uncorrelated:
        for i in range(2):
            data[f"uncorrelated_{i+1}"] = np.random.normal(0, 1, n_periods)

    df = _make_dataframe_with_date(data, n_periods)

    return SyntheticScenario(
        name="correlated_pairs",
        description=f"{n_pairs} correlated pairs (r={correlation}) + uncorrelated controls",
        data=df,
        expected_properties={
            "n_correlated_pairs": n_pairs,
            "target_correlation": correlation,
            "has_uncorrelated": include_uncorrelated,
            "influence_detectable": correlation > 0.5,
        }
    )


def generate_mean_reverting(
    n_periods: int = 252,
    n_columns: int = 5,
    mean_level: float = 0.0,
    reversion_speed: float = 0.1,
    volatility: float = 0.1,
    seed: Optional[int] = None,
) -> SyntheticScenario:
    """
    Generate mean-reverting (Ornstein-Uhlenbeck) processes.

    Useful for testing stationarity detection and regime lenses.

    Parameters
    ----------
    n_periods : int
        Number of time periods (rows).
    n_columns : int
        Number of indicators (columns).
    mean_level : float
        Long-term mean level.
    reversion_speed : float
        Speed of mean reversion (0 to 1).
    volatility : float
        Process volatility.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    SyntheticScenario
        Container with data and expected properties.
    """
    if seed is not None:
        np.random.seed(seed)

    data = {}

    for i in range(n_columns):
        # Ornstein-Uhlenbeck process
        x = np.zeros(n_periods)
        x[0] = mean_level + np.random.normal(0, volatility)

        for t in range(1, n_periods):
            dx = reversion_speed * (mean_level - x[t-1]) + volatility * np.random.normal()
            x[t] = x[t-1] + dx

        data[f"mr_{i+1}"] = x

    df = _make_dataframe_with_date(data, n_periods)

    # Expected long-term variance: volatility^2 / (2 * reversion_speed)
    expected_variance = volatility**2 / (2 * reversion_speed) if reversion_speed > 0 else float('inf')

    return SyntheticScenario(
        name="mean_reverting",
        description=f"Mean-reverting process (mean={mean_level}, speed={reversion_speed})",
        data=df,
        expected_properties={
            "is_stationary": True,
            "mean_level": mean_level,
            "reversion_speed": reversion_speed,
            "expected_variance": expected_variance,
        }
    )


def generate_random_walk(
    n_periods: int = 252,
    n_columns: int = 5,
    drift: float = 0.0,
    volatility: float = 0.01,
    seed: Optional[int] = None,
) -> SyntheticScenario:
    """
    Generate random walk (Brownian motion) processes.

    Simulates price-like behavior. Useful for testing trend/momentum lenses.

    Parameters
    ----------
    n_periods : int
        Number of time periods (rows).
    n_columns : int
        Number of indicators (columns).
    drift : float
        Drift per period (expected return).
    volatility : float
        Step volatility (standard deviation of returns).
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    SyntheticScenario
        Container with data and expected properties.
    """
    if seed is not None:
        np.random.seed(seed)

    data = {}

    for i in range(n_columns):
        returns = np.random.normal(drift, volatility, n_periods)
        prices = 100 * np.exp(np.cumsum(returns))  # Log-normal prices starting at 100
        data[f"rw_{i+1}"] = prices

    df = _make_dataframe_with_date(data, n_periods)

    return SyntheticScenario(
        name="random_walk",
        description=f"Random walk (drift={drift}, vol={volatility})",
        data=df,
        expected_properties={
            "is_stationary": False,
            "drift": drift,
            "volatility": volatility,
            "expected_hurst": 0.5,
        }
    )


def get_all_scenarios(
    n_periods: int = 252,
    seed: int = 42,
) -> list[SyntheticScenario]:
    """
    Generate all standard synthetic scenarios.

    Parameters
    ----------
    n_periods : int
        Number of time periods for each scenario.
    seed : int
        Base random seed (incremented for each scenario).

    Returns
    -------
    list[SyntheticScenario]
        List of all scenarios for comprehensive testing.
    """
    scenarios = [
        generate_sine_wave(n_periods=n_periods, seed=seed),
        generate_trend_noise(n_periods=n_periods, seed=seed + 1),
        generate_step_change(n_periods=n_periods, seed=seed + 2),
        generate_white_noise(n_periods=n_periods, seed=seed + 3),
        generate_correlated_pairs(n_periods=n_periods, seed=seed + 4),
        generate_mean_reverting(n_periods=n_periods, seed=seed + 5),
        generate_random_walk(n_periods=n_periods, seed=seed + 6),
    ]
    return scenarios
