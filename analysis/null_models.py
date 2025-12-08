"""
Null Models Module
==================

Generate null model data for hypothesis testing.
Different null models encode different assumptions about "chance".

Models:
- White noise: IID Gaussian
- Random walk: I(1) process
- AR(1): Matched autocorrelation
- ARMA: Matched ACF/PACF
- GARCH: Volatility clustering without signal
- Regime switching: Random regime changes
- Coupled null: Two series with known coupling

Use these to test: "Is my finding different from what we'd expect under [null model]?"
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field


@dataclass
class NullModelSpec:
    """Specification for a null model."""
    name: str
    description: str
    parameters: Dict[str, Any]
    preserves: List[str]  # What properties are preserved
    destroys: List[str]   # What properties are destroyed


@dataclass
class NullModelResult:
    """Result from null model generation/testing."""
    model_name: str
    parameters: Dict[str, Any]
    sample_data: Optional[np.ndarray] = None
    n_samples: int = 0


class NullModelGenerator:
    """
    Generate data from various null models.

    Use to create baseline expectations for statistical tests.
    """

    def __init__(self, random_seed: Optional[int] = None):
        self.rng = np.random.RandomState(random_seed)

    def white_noise(self,
                    n: int,
                    mean: float = 0.0,
                    std: float = 1.0) -> np.ndarray:
        """
        IID Gaussian white noise.

        The simplest null: no temporal structure at all.

        Preserves: Nothing
        Destroys: All temporal dependencies
        """
        return self.rng.randn(n) * std + mean

    def random_walk(self,
                    n: int,
                    drift: float = 0.0,
                    sigma: float = 1.0,
                    initial: float = 0.0) -> np.ndarray:
        """
        Random walk (unit root process).

        x_t = x_{t-1} + drift + sigma * epsilon

        This is the null hypothesis for many financial series.
        "Markets are efficient" implies returns are random walk.

        Preserves: I(1) nature
        Destroys: Mean reversion, predictability
        """
        innovations = self.rng.randn(n) * sigma + drift
        walk = np.cumsum(innovations) + initial
        return walk

    def ar1(self,
            n: int,
            phi: float = 0.9,
            sigma: float = 1.0,
            mean: float = 0.0) -> np.ndarray:
        """
        AR(1) process with specified autocorrelation.

        x_t = mean + phi * (x_{t-1} - mean) + sigma * epsilon

        Use when you want to control for simple autocorrelation.
        The observed series may have high autocorrelation that explains results.

        Args:
            n: Length of series
            phi: AR coefficient (-1 < phi < 1 for stationarity)
            sigma: Innovation standard deviation
            mean: Long-run mean
        """
        x = np.zeros(n)
        x[0] = mean + self.rng.randn() * sigma / np.sqrt(1 - phi**2)

        for t in range(1, n):
            x[t] = mean + phi * (x[t-1] - mean) + self.rng.randn() * sigma

        return x

    def ar1_matched(self,
                    reference: np.ndarray) -> np.ndarray:
        """
        AR(1) matched to reference series.

        Estimates phi from reference and generates matching process.
        """
        reference = np.asarray(reference)
        reference = reference[~np.isnan(reference)]

        n = len(reference)
        mean = np.mean(reference)
        centered = reference - mean

        # Estimate phi via autocorrelation
        phi = np.sum(centered[1:] * centered[:-1]) / np.sum(centered[:-1]**2)
        phi = np.clip(phi, -0.999, 0.999)  # Ensure stationarity

        # Estimate innovation variance
        residuals = centered[1:] - phi * centered[:-1]
        sigma = np.std(residuals)

        return self.ar1(n, phi=phi, sigma=sigma, mean=mean)

    def arma(self,
             n: int,
             ar_coeffs: List[float] = [0.5],
             ma_coeffs: List[float] = [0.3],
             sigma: float = 1.0) -> np.ndarray:
        """
        ARMA(p, q) process.

        x_t = sum(phi_i * x_{t-i}) + epsilon_t + sum(theta_j * epsilon_{t-j})

        Use for more complex autocorrelation structures.
        """
        p = len(ar_coeffs)
        q = len(ma_coeffs)

        x = np.zeros(n)
        eps = self.rng.randn(n) * sigma

        for t in range(max(p, q), n):
            ar_term = sum(ar_coeffs[i] * x[t-i-1] for i in range(p) if t-i-1 >= 0)
            ma_term = sum(ma_coeffs[j] * eps[t-j-1] for j in range(q) if t-j-1 >= 0)
            x[t] = ar_term + eps[t] + ma_term

        return x

    def garch(self,
              n: int,
              omega: float = 0.1,
              alpha: float = 0.1,
              beta: float = 0.8) -> np.ndarray:
        """
        GARCH(1,1) process.

        sigma_t^2 = omega + alpha * x_{t-1}^2 + beta * sigma_{t-1}^2
        x_t = sigma_t * epsilon_t

        Produces volatility clustering without any signal.
        Use to test if apparent patterns are just volatility dynamics.
        """
        x = np.zeros(n)
        sigma2 = np.zeros(n)

        # Initial variance
        sigma2[0] = omega / (1 - alpha - beta) if (alpha + beta) < 1 else 1.0

        for t in range(1, n):
            sigma2[t] = omega + alpha * x[t-1]**2 + beta * sigma2[t-1]
            x[t] = np.sqrt(sigma2[t]) * self.rng.randn()

        return x

    def regime_switching(self,
                         n: int,
                         means: List[float] = [0, 2, -1],
                         stds: List[float] = [1, 1.5, 0.5],
                         transition_prob: float = 0.02) -> Tuple[np.ndarray, np.ndarray]:
        """
        Random regime switching model.

        Switches between regimes with fixed probability.
        No dependence on state or data.

        Use to test if detected regimes are better than random.

        Returns:
            Tuple of (data, regime_labels)
        """
        n_regimes = len(means)
        x = np.zeros(n)
        regimes = np.zeros(n, dtype=int)

        current_regime = 0

        for t in range(n):
            # Random transition
            if self.rng.random() < transition_prob:
                current_regime = self.rng.randint(0, n_regimes)

            regimes[t] = current_regime
            x[t] = self.rng.randn() * stds[current_regime] + means[current_regime]

        return x, regimes

    def coupled_null(self,
                     n: int,
                     coupling: float = 0.0,
                     lag: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Two series with specified coupling strength.

        y_t = coupling * x_{t-lag} + sqrt(1 - coupling^2) * noise

        Use coupling=0 to test for spurious correlations.
        Use specific coupling to test detection sensitivity.

        Returns:
            Tuple of (series1, series2)
        """
        x = self.rng.randn(n)
        noise = self.rng.randn(n)

        # Create y with specified coupling
        coupling_factor = np.clip(coupling, -1, 1)
        noise_factor = np.sqrt(1 - coupling_factor**2)

        y = np.zeros(n)
        for t in range(n):
            if t >= lag:
                y[t] = coupling_factor * x[t - lag] + noise_factor * noise[t]
            else:
                y[t] = noise[t]

        return x, y

    def matched_spectrum(self, reference: np.ndarray) -> np.ndarray:
        """
        Generate series with same power spectrum as reference but random phases.

        Preserves frequency content but destroys phase relationships.
        """
        reference = np.asarray(reference)
        n = len(reference)

        # Get spectrum
        fft = np.fft.fft(reference)
        amplitudes = np.abs(fft)

        # Random phases
        phases = self.rng.uniform(0, 2 * np.pi, n)
        phases[0] = 0  # DC component
        if n % 2 == 0:
            phases[n//2] = 0  # Nyquist

        # Ensure conjugate symmetry for real output
        for i in range(1, n // 2):
            phases[n - i] = -phases[i]

        # Reconstruct
        new_fft = amplitudes * np.exp(1j * phases)
        return np.real(np.fft.ifft(new_fft))

    def matched_distribution(self,
                             reference: np.ndarray,
                             preserve_autocorr: bool = False) -> np.ndarray:
        """
        Generate series with same distribution as reference.

        If preserve_autocorr=False, just shuffle.
        If preserve_autocorr=True, use rank-based matching.
        """
        reference = np.asarray(reference)
        n = len(reference)

        if not preserve_autocorr:
            result = reference.copy()
            self.rng.shuffle(result)
            return result

        # Generate AR(1) with matched autocorrelation
        base = self.ar1_matched(reference)

        # Transform to match distribution
        sorted_ref = np.sort(reference)
        ranks = np.argsort(np.argsort(base))

        return sorted_ref[ranks]

    def generate(self,
                 model: str,
                 n: int,
                 **kwargs) -> Union[np.ndarray, Tuple[np.ndarray, ...]]:
        """
        Generate null model data.

        Args:
            model: Model name
            n: Series length
            **kwargs: Model-specific parameters

        Returns:
            Generated data (array or tuple of arrays)
        """
        models = {
            'white_noise': self.white_noise,
            'random_walk': self.random_walk,
            'ar1': self.ar1,
            'arma': self.arma,
            'garch': self.garch,
            'regime_switching': self.regime_switching,
            'coupled_null': self.coupled_null,
        }

        if model not in models:
            raise ValueError(f"Unknown model: {model}. Available: {list(models.keys())}")

        return models[model](n, **kwargs)


class NullModelTest:
    """
    Test observed statistic against null model distribution.
    """

    def __init__(self,
                 observed_data: np.ndarray,
                 statistic: Callable[[np.ndarray], float],
                 null_model: str,
                 n_simulations: int = 1000,
                 alpha: float = 0.05,
                 null_params: Optional[Dict] = None,
                 random_seed: Optional[int] = None):
        """
        Initialize null model test.

        Args:
            observed_data: Observed time series
            statistic: Test statistic function
            null_model: Name of null model
            n_simulations: Number of null simulations
            alpha: Significance level
            null_params: Parameters for null model (auto-estimated if None)
            random_seed: For reproducibility
        """
        self.observed_data = np.asarray(observed_data)
        self.statistic = statistic
        self.null_model = null_model
        self.n_simulations = n_simulations
        self.alpha = alpha
        self.null_params = null_params or {}

        self.generator = NullModelGenerator(random_seed)
        self.n = len(self.observed_data)

    def run(self) -> Dict[str, Any]:
        """
        Run null model test.

        Returns:
            Dictionary with test results
        """
        # Observed statistic
        observed_stat = self.statistic(self.observed_data)

        # Generate null distribution
        null_stats = np.zeros(self.n_simulations)

        for i in range(self.n_simulations):
            null_data = self.generator.generate(
                self.null_model,
                self.n,
                **self.null_params
            )
            if isinstance(null_data, tuple):
                null_data = null_data[0]  # Use first series if tuple

            null_stats[i] = self.statistic(null_data)

        # Compute statistics
        null_mean = np.mean(null_stats)
        null_std = np.std(null_stats)
        z_score = (observed_stat - null_mean) / null_std if null_std > 0 else 0

        # Two-sided p-value
        p_value = np.mean(np.abs(null_stats - null_mean) >= np.abs(observed_stat - null_mean))
        p_value = max(p_value, 1 / (self.n_simulations + 1))

        return {
            'observed_statistic': round(observed_stat, 6),
            'null_model': self.null_model,
            'null_mean': round(null_mean, 6),
            'null_std': round(null_std, 6),
            'z_score': round(z_score, 4),
            'p_value': round(p_value, 6),
            'significant': p_value < self.alpha,
            'n_simulations': self.n_simulations,
            'percentile': round(np.mean(null_stats <= observed_stat) * 100, 1)
        }


def get_null_model_spec(model: str) -> NullModelSpec:
    """Get specification for a null model."""
    specs = {
        'white_noise': NullModelSpec(
            name='White Noise',
            description='IID Gaussian noise with no temporal structure',
            parameters={'mean': 0, 'std': 1},
            preserves=[],
            destroys=['autocorrelation', 'trends', 'patterns']
        ),
        'random_walk': NullModelSpec(
            name='Random Walk',
            description='Unit root process - efficient market null',
            parameters={'drift': 0, 'sigma': 1},
            preserves=['I(1) property', 'unpredictability'],
            destroys=['mean reversion', 'predictability']
        ),
        'ar1': NullModelSpec(
            name='AR(1)',
            description='First-order autoregressive process',
            parameters={'phi': 'estimated', 'sigma': 'estimated'},
            preserves=['first-order autocorrelation'],
            destroys=['higher-order dependencies', 'nonlinearity']
        ),
        'garch': NullModelSpec(
            name='GARCH(1,1)',
            description='Volatility clustering model',
            parameters={'omega': 0.1, 'alpha': 0.1, 'beta': 0.8},
            preserves=['volatility dynamics'],
            destroys=['level predictability', 'directional signals']
        ),
        'regime_switching': NullModelSpec(
            name='Random Regime Switching',
            description='Random transitions between states',
            parameters={'transition_prob': 0.02, 'n_regimes': 3},
            preserves=['regime structure'],
            destroys=['state dependence', 'predictable transitions']
        ),
        'coupled_null': NullModelSpec(
            name='Coupled Null',
            description='Two series with specified coupling',
            parameters={'coupling': 0, 'lag': 0},
            preserves=['specified coupling strength'],
            destroys=['other relationships']
        )
    }

    if model not in specs:
        raise ValueError(f"Unknown model: {model}")

    return specs[model]


if __name__ == "__main__":
    print("=" * 60)
    print("Null Models Module - Test")
    print("=" * 60)

    np.random.seed(42)

    generator = NullModelGenerator(random_seed=42)
    n = 500

    print("\nGenerating null models:")
    print("-" * 60)

    # White noise
    wn = generator.white_noise(n)
    print(f"White noise: mean={np.mean(wn):.4f}, std={np.std(wn):.4f}")

    # Random walk
    rw = generator.random_walk(n)
    print(f"Random walk: start={rw[0]:.2f}, end={rw[-1]:.2f}")

    # AR(1)
    ar = generator.ar1(n, phi=0.9)
    autocorr = np.corrcoef(ar[1:], ar[:-1])[0, 1]
    print(f"AR(1) phi=0.9: estimated autocorr={autocorr:.3f}")

    # GARCH
    garch = generator.garch(n)
    print(f"GARCH: kurtosis={np.mean((garch - np.mean(garch))**4) / np.std(garch)**4:.2f}")

    # Regime switching
    rs, regimes = generator.regime_switching(n)
    regime_counts = [np.sum(regimes == i) for i in range(3)]
    print(f"Regime switching: regime counts={regime_counts}")

    # Coupled null
    x, y = generator.coupled_null(n, coupling=0.5)
    actual_corr = np.corrcoef(x, y)[0, 1]
    print(f"Coupled null (coupling=0.5): actual corr={actual_corr:.3f}")

    print("\n" + "=" * 60)
    print("Null Model Test:")
    print("-" * 60)

    # Create data with structure
    structured_data = np.zeros(n)
    for i in range(1, n):
        structured_data[i] = 0.8 * structured_data[i-1] + np.random.randn()

    # Test against white noise null
    def autocorr_stat(x):
        return np.abs(np.corrcoef(x[1:], x[:-1])[0, 1])

    test = NullModelTest(
        structured_data,
        autocorr_stat,
        null_model='white_noise',
        n_simulations=500
    )
    result = test.run()

    print(f"Observed autocorr: {result['observed_statistic']:.4f}")
    print(f"Null mean: {result['null_mean']:.4f}")
    print(f"P-value: {result['p_value']:.4f}")
    print(f"Significant: {result['significant']}")

    print("\n" + "=" * 60)
    print("Test completed!")
