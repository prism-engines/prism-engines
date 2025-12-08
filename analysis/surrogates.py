"""
Surrogate Data Framework
========================

Generate surrogate data to test if observed patterns are statistically significant.
Critical for validating that findings aren't artifacts.

Methods:
- Random shuffle: Destroys all temporal structure
- Phase randomization: Preserves spectrum, destroys phase coupling
- AAFT: Amplitude Adjusted Fourier Transform (preserves distribution + spectrum)
- IAAFT: Iterative AAFT (better preservation)
- Block bootstrap: Preserves local structure
- Twin surrogates: For deterministic systems

Usage:
    test = SurrogateTest(data, statistic_func, method='iaaft', n_surrogates=1000)
    result = test.run()
    print(f"p-value: {result.p_value}")
"""

import numpy as np
import pandas as pd
from typing import Callable, Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum


class SurrogateMethod(Enum):
    """Available surrogate generation methods."""
    RANDOM_SHUFFLE = 'random_shuffle'
    PHASE_RANDOMIZATION = 'phase_randomization'
    AAFT = 'aaft'
    IAAFT = 'iaaft'
    BLOCK_BOOTSTRAP = 'block_bootstrap'
    TWIN = 'twin'


@dataclass
class SurrogateTestResult:
    """Result from surrogate data testing."""
    observed_statistic: float
    surrogate_statistics: np.ndarray
    surrogate_mean: float
    surrogate_std: float
    z_score: float
    p_value: float
    significant: bool
    method: str
    n_surrogates: int
    alpha: float
    alternative: str  # 'two-sided', 'greater', 'less'

    def __repr__(self):
        sig_str = "SIGNIFICANT" if self.significant else "not significant"
        return (f"SurrogateTestResult(observed={self.observed_statistic:.4f}, "
                f"null={self.surrogate_mean:.4f}±{self.surrogate_std:.4f}, "
                f"p={self.p_value:.4f}, {sig_str})")

    def to_dict(self) -> Dict[str, Any]:
        return {
            'observed': round(self.observed_statistic, 6),
            'surrogate_mean': round(self.surrogate_mean, 6),
            'surrogate_std': round(self.surrogate_std, 6),
            'z_score': round(self.z_score, 4),
            'p_value': round(self.p_value, 6),
            'significant': self.significant,
            'method': self.method,
            'n_surrogates': self.n_surrogates
        }


class SurrogateGenerator:
    """
    Generate surrogate data using various methods.
    """

    def __init__(self, random_seed: Optional[int] = None):
        self.rng = np.random.RandomState(random_seed)

    def random_shuffle(self, data: np.ndarray) -> np.ndarray:
        """
        Random permutation of data.

        Destroys all temporal structure (autocorrelation, trends, etc.)
        Use to test if ANY temporal pattern matters.
        """
        surrogate = data.copy()
        self.rng.shuffle(surrogate)
        return surrogate

    def phase_randomization(self, data: np.ndarray) -> np.ndarray:
        """
        Randomize phases while preserving power spectrum.

        Preserves: Autocorrelation structure, power spectrum
        Destroys: Phase relationships, nonlinear dependencies

        Use to test for nonlinear dynamics or phase coupling.
        """
        n = len(data)
        fft = np.fft.fft(data)

        # Random phases
        phases = self.rng.uniform(0, 2 * np.pi, n // 2)

        # Construct symmetric phases for real output
        if n % 2 == 0:
            new_phases = np.concatenate([[0], phases, [0], -phases[::-1]])
        else:
            new_phases = np.concatenate([[0], phases, -phases[::-1]])

        # Apply random phases
        fft_randomized = np.abs(fft) * np.exp(1j * new_phases[:len(fft)])

        return np.real(np.fft.ifft(fft_randomized))

    def aaft(self, data: np.ndarray) -> np.ndarray:
        """
        Amplitude Adjusted Fourier Transform.

        Preserves: Distribution AND power spectrum (approximately)
        Destroys: Phase relationships

        Better than phase randomization for non-Gaussian data.
        """
        n = len(data)
        sorted_data = np.sort(data)

        # Phase randomize Gaussian data
        gaussian = self.rng.randn(n)
        gaussian_pr = self.phase_randomization(gaussian)

        # Rank-order transform: replace Gaussian values with original data values
        ranks = np.argsort(np.argsort(gaussian_pr))
        surrogate = sorted_data[ranks]

        return surrogate

    def iaaft(self, data: np.ndarray, max_iter: int = 100, tol: float = 1e-6) -> np.ndarray:
        """
        Iterative Amplitude Adjusted Fourier Transform.

        Iteratively refines AAFT to better preserve both distribution and spectrum.
        Gold standard for surrogate testing in nonlinear time series analysis.
        """
        n = len(data)
        sorted_data = np.sort(data)
        target_spectrum = np.abs(np.fft.fft(data))

        # Initialize with AAFT
        surrogate = self.aaft(data)

        for iteration in range(max_iter):
            # Step 1: Match spectrum
            fft_surrogate = np.fft.fft(surrogate)
            phases = np.angle(fft_surrogate)
            fft_adjusted = target_spectrum * np.exp(1j * phases)
            surrogate_new = np.real(np.fft.ifft(fft_adjusted))

            # Step 2: Match distribution
            ranks = np.argsort(np.argsort(surrogate_new))
            surrogate_new = sorted_data[ranks]

            # Check convergence
            if np.max(np.abs(surrogate_new - surrogate)) < tol:
                break

            surrogate = surrogate_new

        return surrogate

    def block_bootstrap(self, data: np.ndarray, block_size: Optional[int] = None) -> np.ndarray:
        """
        Block bootstrap resampling.

        Preserves: Local structure within blocks
        Destroys: Long-range dependencies

        Use to test if local patterns drive results.
        """
        n = len(data)
        if block_size is None:
            block_size = max(1, int(n ** (1/3)))

        n_blocks = int(np.ceil(n / block_size))
        block_starts = self.rng.randint(0, n - block_size + 1, size=n_blocks)

        surrogate = []
        for start in block_starts:
            surrogate.extend(data[start:start + block_size])

        return np.array(surrogate[:n])

    def twin_surrogate(self,
                       data: np.ndarray,
                       embedding_dim: int = 3,
                       delay: int = 1,
                       threshold: float = 0.1) -> np.ndarray:
        """
        Twin surrogate method for testing deterministic dynamics.

        Preserves local state-space structure by finding "twins" (nearly identical states)
        and randomly choosing which trajectory to follow.
        """
        n = len(data)
        m = embedding_dim
        tau = delay

        # Create embedding
        embedded = np.zeros((n - (m - 1) * tau, m))
        for i in range(m):
            embedded[:, i] = data[i * tau:n - (m - 1 - i) * tau]

        # Find twins (nearby points in state space)
        surrogate = data.copy()
        max_dist = threshold * np.std(data)

        for i in range(len(embedded) - 1):
            # Find points within threshold distance
            distances = np.sqrt(np.sum((embedded - embedded[i]) ** 2, axis=1))
            twins = np.where((distances < max_dist) & (distances > 0))[0]

            if len(twins) > 0:
                # Randomly choose a twin's successor
                twin_idx = self.rng.choice(twins)
                if twin_idx + (m - 1) * tau + 1 < n:
                    surrogate[i + (m - 1) * tau + 1] = data[twin_idx + (m - 1) * tau + 1]

        return surrogate

    def generate(self,
                 data: np.ndarray,
                 method: str = 'iaaft',
                 **kwargs) -> np.ndarray:
        """
        Generate surrogate using specified method.

        Args:
            data: Original time series
            method: Surrogate method name
            **kwargs: Method-specific parameters

        Returns:
            Surrogate time series
        """
        methods = {
            'random_shuffle': self.random_shuffle,
            'phase_randomization': self.phase_randomization,
            'aaft': self.aaft,
            'iaaft': self.iaaft,
            'block_bootstrap': self.block_bootstrap,
            'twin': self.twin_surrogate,
        }

        if method not in methods:
            raise ValueError(f"Unknown method: {method}. Available: {list(methods.keys())}")

        return methods[method](data, **kwargs)


class SurrogateTest:
    """
    Statistical hypothesis testing using surrogate data.

    Tests whether an observed statistic could arise by chance
    under a null hypothesis specified by the surrogate method.
    """

    def __init__(self,
                 data: Union[np.ndarray, pd.Series],
                 statistic: Callable[[np.ndarray], float],
                 method: str = 'iaaft',
                 n_surrogates: int = 1000,
                 alpha: float = 0.05,
                 alternative: str = 'two-sided',
                 random_seed: Optional[int] = None):
        """
        Initialize surrogate test.

        Args:
            data: Time series to test
            statistic: Function that computes test statistic from data
            method: Surrogate generation method
            n_surrogates: Number of surrogates to generate
            alpha: Significance level
            alternative: 'two-sided', 'greater', or 'less'
            random_seed: For reproducibility
        """
        if isinstance(data, pd.Series):
            data = data.values
        self.data = np.asarray(data).flatten()
        self.data = self.data[~np.isnan(self.data)]

        self.statistic = statistic
        self.method = method
        self.n_surrogates = n_surrogates
        self.alpha = alpha
        self.alternative = alternative

        self.generator = SurrogateGenerator(random_seed)

    def run(self) -> SurrogateTestResult:
        """
        Run the surrogate test.

        Returns:
            SurrogateTestResult with p-value and significance
        """
        # Compute observed statistic
        observed = self.statistic(self.data)

        # Generate surrogates and compute statistics
        surrogate_stats = np.zeros(self.n_surrogates)

        for i in range(self.n_surrogates):
            surrogate = self.generator.generate(self.data, method=self.method)
            surrogate_stats[i] = self.statistic(surrogate)

        # Compute null distribution statistics
        null_mean = np.mean(surrogate_stats)
        null_std = np.std(surrogate_stats, ddof=1)

        # Z-score
        z_score = (observed - null_mean) / null_std if null_std > 0 else 0

        # P-value
        if self.alternative == 'two-sided':
            p_value = np.mean(np.abs(surrogate_stats - null_mean) >= np.abs(observed - null_mean))
        elif self.alternative == 'greater':
            p_value = np.mean(surrogate_stats >= observed)
        else:  # 'less'
            p_value = np.mean(surrogate_stats <= observed)

        # Ensure p-value has minimum resolution
        p_value = max(p_value, 1 / (self.n_surrogates + 1))

        significant = p_value < self.alpha

        return SurrogateTestResult(
            observed_statistic=observed,
            surrogate_statistics=surrogate_stats,
            surrogate_mean=null_mean,
            surrogate_std=null_std,
            z_score=z_score,
            p_value=p_value,
            significant=significant,
            method=self.method,
            n_surrogates=self.n_surrogates,
            alpha=self.alpha,
            alternative=self.alternative
        )


def test_cross_domain_coherence(data1: np.ndarray,
                                data2: np.ndarray,
                                coherence_func: Callable,
                                n_surrogates: int = 1000,
                                method: str = 'phase_randomization',
                                random_seed: Optional[int] = None) -> SurrogateTestResult:
    """
    Test if cross-domain coherence is significant.

    Null hypothesis: The two series are independent.
    Surrogates preserve within-series structure but destroy cross-series relationship.

    Args:
        data1, data2: Two time series
        coherence_func: Function computing coherence between two series
        n_surrogates: Number of surrogates
        method: Surrogate method for data2
        random_seed: For reproducibility

    Returns:
        SurrogateTestResult
    """
    data1 = np.asarray(data1).flatten()
    data2 = np.asarray(data2).flatten()

    # Remove NaN
    mask = ~(np.isnan(data1) | np.isnan(data2))
    data1, data2 = data1[mask], data2[mask]

    generator = SurrogateGenerator(random_seed)

    # Observed coherence
    observed = coherence_func(data1, data2)

    # Surrogate coherences
    surrogate_stats = np.zeros(n_surrogates)
    for i in range(n_surrogates):
        # Surrogate one series while keeping the other fixed
        surrogate2 = generator.generate(data2, method=method)
        surrogate_stats[i] = coherence_func(data1, surrogate2)

    # Statistics
    null_mean = np.mean(surrogate_stats)
    null_std = np.std(surrogate_stats, ddof=1)
    z_score = (observed - null_mean) / null_std if null_std > 0 else 0

    # One-sided test (coherence should be higher than chance)
    p_value = np.mean(surrogate_stats >= observed)
    p_value = max(p_value, 1 / (n_surrogates + 1))

    return SurrogateTestResult(
        observed_statistic=observed,
        surrogate_statistics=surrogate_stats,
        surrogate_mean=null_mean,
        surrogate_std=null_std,
        z_score=z_score,
        p_value=p_value,
        significant=p_value < 0.05,
        method=f'cross_domain_{method}',
        n_surrogates=n_surrogates,
        alpha=0.05,
        alternative='greater'
    )


def batch_surrogate_test(data: pd.DataFrame,
                         statistic: Callable,
                         method: str = 'iaaft',
                         n_surrogates: int = 500) -> Dict[str, SurrogateTestResult]:
    """
    Run surrogate tests on all columns of a DataFrame.

    Args:
        data: DataFrame with time series as columns
        statistic: Function computing statistic from 1D array
        method: Surrogate method
        n_surrogates: Number of surrogates per test

    Returns:
        Dictionary mapping column names to test results
    """
    results = {}
    for col in data.columns:
        test = SurrogateTest(
            data[col].values,
            statistic,
            method=method,
            n_surrogates=n_surrogates
        )
        results[col] = test.run()
    return results


if __name__ == "__main__":
    print("=" * 60)
    print("Surrogate Data Framework - Test")
    print("=" * 60)

    np.random.seed(42)
    n = 500

    # Create test data with nonlinear structure
    t = np.linspace(0, 10 * np.pi, n)
    nonlinear_data = np.sin(t) + 0.5 * np.sin(2 * t + np.sin(t)) + np.random.randn(n) * 0.2

    # Test statistic: sample entropy (simplified)
    def autocorr_sum(x):
        """Sum of squared autocorrelations (tests for temporal structure)."""
        n = len(x)
        x = x - np.mean(x)
        result = 0
        for lag in range(1, min(20, n)):
            acf = np.sum(x[lag:] * x[:-lag]) / (n - lag)
            result += acf ** 2
        return result

    print("\nTest 1: Testing nonlinear data with different surrogate methods")
    print("-" * 60)

    generator = SurrogateGenerator(random_seed=42)

    methods = ['random_shuffle', 'phase_randomization', 'iaaft']
    for method in methods:
        test = SurrogateTest(
            nonlinear_data,
            autocorr_sum,
            method=method,
            n_surrogates=100
        )
        result = test.run()
        print(f"\n{method}:")
        print(f"  Observed: {result.observed_statistic:.4f}")
        print(f"  Null: {result.surrogate_mean:.4f} ± {result.surrogate_std:.4f}")
        print(f"  Z-score: {result.z_score:.2f}")
        print(f"  P-value: {result.p_value:.4f}")
        print(f"  Significant: {result.significant}")

    print("\n" + "=" * 60)
    print("Test 2: Cross-domain coherence test")
    print("-" * 60)

    # Create two related series
    common = np.cumsum(np.random.randn(n))
    series1 = common + np.random.randn(n) * 0.5
    series2 = 0.8 * common + np.random.randn(n) * 0.5

    def simple_coherence(x, y):
        """Simple coherence measure (correlation of absolute values)."""
        return np.abs(np.corrcoef(x, y)[0, 1])

    result = test_cross_domain_coherence(
        series1, series2,
        simple_coherence,
        n_surrogates=100
    )
    print(f"Observed coherence: {result.observed_statistic:.4f}")
    print(f"Null coherence: {result.surrogate_mean:.4f} ± {result.surrogate_std:.4f}")
    print(f"P-value: {result.p_value:.4f}")
    print(f"Significant: {result.significant}")

    print("\n" + "=" * 60)
    print("Test completed!")
