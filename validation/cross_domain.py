"""
Cross-Domain Coherence Testing
==============================

Test framework for validating cross-domain coherence hypotheses.

This module provides tools to:
1. Compare coherence patterns across different domain pairs
2. Test if financial-climate coherence generalizes to other domains
3. Validate the universality claims of the PRISM framework
4. Generate comprehensive cross-domain analysis reports

Key tests:
- Climate-Economics coherence
- Epidemiology-Economics coherence
- Geophysics-Markets coherence
- Social sentiment-Markets coherence
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime
import logging
from enum import Enum

logger = logging.getLogger(__name__)


class Domain(Enum):
    """Domain categories for cross-domain analysis."""
    CLIMATE = 'climate'
    ECONOMICS = 'economics'
    EPIDEMIOLOGY = 'epidemiology'
    GEOPHYSICS = 'geophysics'
    SOCIAL = 'social'
    FINANCE = 'finance'


@dataclass
class DomainSeries:
    """A time series from a specific domain."""
    name: str
    domain: Domain
    data: pd.DataFrame  # Must have 'date' and 'value' columns
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CoherenceResult:
    """Result of coherence analysis between two series."""
    series1: str
    series2: str
    domain1: Domain
    domain2: Domain
    coherence: float  # Overall coherence (0-1)
    frequency_coherence: np.ndarray  # Coherence by frequency
    frequencies: np.ndarray
    phase_lag: float  # Phase relationship
    p_value: float  # Significance
    n_observations: int
    method: str


@dataclass
class CrossDomainReport:
    """Complete cross-domain analysis report."""
    domain_pairs: List[Tuple[Domain, Domain]]
    coherence_results: List[CoherenceResult]
    significant_pairs: List[Tuple[str, str, float]]
    null_hypothesis_rejected: Dict[Tuple[Domain, Domain], bool]
    summary_stats: Dict[str, float]
    conclusions: List[str]
    caveats: List[str]


class CrossDomainAnalyzer:
    """
    Analyze coherence patterns across multiple domains.

    Tests the hypothesis that cross-domain coherence patterns
    are universal (not specific to finance-climate).
    """

    def __init__(self,
                 significance_level: float = 0.05,
                 min_overlap: int = 52,  # At least 1 year of weekly data
                 coherence_method: str = 'wavelet'):
        """
        Initialize analyzer.

        Args:
            significance_level: Threshold for significance
            min_overlap: Minimum overlapping observations required
            coherence_method: Method for computing coherence
        """
        self.significance_level = significance_level
        self.min_overlap = min_overlap
        self.coherence_method = coherence_method
        self.series: Dict[str, DomainSeries] = {}

    def add_series(self,
                   name: str,
                   domain: Domain,
                   data: pd.DataFrame,
                   source: str = "unknown",
                   metadata: Dict[str, Any] = None) -> None:
        """
        Add a time series to the analyzer.

        Args:
            name: Series name
            domain: Domain category
            data: DataFrame with 'date' and 'value' columns
            source: Data source name
            metadata: Additional metadata
        """
        if 'date' not in data.columns or 'value' not in data.columns:
            raise ValueError("Data must have 'date' and 'value' columns")

        series = DomainSeries(
            name=name,
            domain=domain,
            data=data.copy(),
            source=source,
            metadata=metadata or {}
        )
        self.series[name] = series

    def _align_series(self,
                      series1: DomainSeries,
                      series2: DomainSeries) -> Tuple[np.ndarray, np.ndarray]:
        """
        Align two time series to common dates.

        Args:
            series1: First series
            series2: Second series

        Returns:
            Tuple of aligned value arrays
        """
        df1 = series1.data.copy()
        df2 = series2.data.copy()

        df1['date'] = pd.to_datetime(df1['date'])
        df2['date'] = pd.to_datetime(df2['date'])

        # Merge on date
        merged = df1.merge(df2, on='date', suffixes=('_1', '_2'))

        if len(merged) < self.min_overlap:
            logger.warning(
                f"Insufficient overlap ({len(merged)}) between "
                f"{series1.name} and {series2.name}"
            )
            return None, None

        return merged['value_1'].values, merged['value_2'].values

    def _compute_coherence_simple(self,
                                   x: np.ndarray,
                                   y: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Compute coherence using cross-spectral analysis.

        Args:
            x: First time series
            y: Second time series

        Returns:
            Tuple of (mean coherence, coherence by frequency, frequencies)
        """
        n = len(x)

        # Standardize
        x = (x - np.mean(x)) / (np.std(x) + 1e-10)
        y = (y - np.mean(y)) / (np.std(y) + 1e-10)

        # FFT
        fx = np.fft.rfft(x)
        fy = np.fft.rfft(y)

        # Cross-spectral density
        pxy = fx * np.conj(fy)

        # Auto-spectral densities
        pxx = np.abs(fx) ** 2
        pyy = np.abs(fy) ** 2

        # Coherence (squared coherency)
        coherence = np.abs(pxy) ** 2 / (pxx * pyy + 1e-10)

        # Frequencies
        frequencies = np.fft.rfftfreq(n)

        # Mean coherence (excluding DC and Nyquist)
        mean_coh = np.mean(coherence[1:-1])

        return mean_coh, coherence, frequencies

    def _compute_phase_lag(self,
                           x: np.ndarray,
                           y: np.ndarray) -> float:
        """
        Compute average phase lag between series.

        Args:
            x: First time series
            y: Second time series

        Returns:
            Phase lag in radians
        """
        fx = np.fft.rfft(x)
        fy = np.fft.rfft(y)

        # Cross-spectrum phase
        pxy = fx * np.conj(fy)
        phases = np.angle(pxy)

        # Weighted average by magnitude
        weights = np.abs(pxy)
        weights = weights / (np.sum(weights) + 1e-10)

        avg_phase = np.sum(phases * weights)

        return avg_phase

    def _surrogate_test(self,
                        x: np.ndarray,
                        y: np.ndarray,
                        observed_coh: float,
                        n_surrogates: int = 199) -> float:
        """
        Test significance using phase-randomized surrogates.

        Args:
            x: First time series
            y: Second time series
            observed_coh: Observed coherence value
            n_surrogates: Number of surrogates

        Returns:
            P-value
        """
        n = len(x)
        surrogate_cohs = []

        for _ in range(n_surrogates):
            # Phase-randomize one series
            fx = np.fft.rfft(x)
            phases = np.angle(fx)
            magnitudes = np.abs(fx)

            # Random phases (preserving conjugate symmetry)
            random_phases = np.random.uniform(-np.pi, np.pi, len(phases))
            random_phases[0] = 0  # DC component
            if n % 2 == 0:
                random_phases[-1] = 0  # Nyquist

            # Create surrogate
            fx_surr = magnitudes * np.exp(1j * random_phases)
            x_surr = np.fft.irfft(fx_surr, n=n)

            # Compute coherence
            coh, _, _ = self._compute_coherence_simple(x_surr, y)
            surrogate_cohs.append(coh)

        # P-value: proportion of surrogates with coherence >= observed
        p_value = (np.sum(np.array(surrogate_cohs) >= observed_coh) + 1) / (n_surrogates + 1)

        return p_value

    def compute_coherence(self,
                          name1: str,
                          name2: str,
                          test_significance: bool = True) -> Optional[CoherenceResult]:
        """
        Compute coherence between two named series.

        Args:
            name1: First series name
            name2: Second series name
            test_significance: Whether to run significance test

        Returns:
            CoherenceResult or None if insufficient data
        """
        if name1 not in self.series or name2 not in self.series:
            logger.error(f"Series not found: {name1} or {name2}")
            return None

        series1 = self.series[name1]
        series2 = self.series[name2]

        x, y = self._align_series(series1, series2)

        if x is None:
            return None

        # Compute coherence
        mean_coh, freq_coh, frequencies = self._compute_coherence_simple(x, y)

        # Phase lag
        phase_lag = self._compute_phase_lag(x, y)

        # Significance test
        if test_significance:
            p_value = self._surrogate_test(x, y, mean_coh)
        else:
            p_value = np.nan

        return CoherenceResult(
            series1=name1,
            series2=name2,
            domain1=series1.domain,
            domain2=series2.domain,
            coherence=mean_coh,
            frequency_coherence=freq_coh,
            frequencies=frequencies,
            phase_lag=phase_lag,
            p_value=p_value,
            n_observations=len(x),
            method=self.coherence_method
        )

    def analyze_all_pairs(self,
                          domains: List[Domain] = None,
                          test_significance: bool = True) -> List[CoherenceResult]:
        """
        Analyze coherence for all pairs of series.

        Args:
            domains: List of domains to include (default: all)
            test_significance: Whether to run significance tests

        Returns:
            List of CoherenceResult objects
        """
        results = []
        series_list = list(self.series.keys())

        if domains:
            series_list = [
                name for name, s in self.series.items()
                if s.domain in domains
            ]

        for i, name1 in enumerate(series_list):
            for name2 in series_list[i+1:]:
                result = self.compute_coherence(
                    name1, name2,
                    test_significance=test_significance
                )
                if result:
                    results.append(result)

        return results

    def analyze_cross_domain(self,
                             domain1: Domain,
                             domain2: Domain,
                             test_significance: bool = True) -> List[CoherenceResult]:
        """
        Analyze coherence between all series from two domains.

        Args:
            domain1: First domain
            domain2: Second domain
            test_significance: Whether to run significance tests

        Returns:
            List of CoherenceResult objects
        """
        results = []

        series1 = [
            name for name, s in self.series.items()
            if s.domain == domain1
        ]
        series2 = [
            name for name, s in self.series.items()
            if s.domain == domain2
        ]

        for name1 in series1:
            for name2 in series2:
                result = self.compute_coherence(
                    name1, name2,
                    test_significance=test_significance
                )
                if result:
                    results.append(result)

        return results

    def generate_report(self,
                        results: List[CoherenceResult] = None) -> CrossDomainReport:
        """
        Generate comprehensive cross-domain analysis report.

        Args:
            results: Pre-computed results (or compute all pairs)

        Returns:
            CrossDomainReport
        """
        if results is None:
            results = self.analyze_all_pairs()

        # Domain pairs analyzed
        domain_pairs = list(set(
            (r.domain1, r.domain2) for r in results
        ))

        # Significant pairs
        significant_pairs = [
            (r.series1, r.series2, r.coherence)
            for r in results
            if r.p_value < self.significance_level
        ]

        # Null hypothesis rejected by domain pair
        null_rejected = {}
        for d1, d2 in domain_pairs:
            pair_results = [
                r for r in results
                if (r.domain1 == d1 and r.domain2 == d2) or
                   (r.domain1 == d2 and r.domain2 == d1)
            ]
            # Reject if at least one significant pair
            null_rejected[(d1, d2)] = any(
                r.p_value < self.significance_level
                for r in pair_results
            )

        # Summary statistics
        coherences = [r.coherence for r in results]
        p_values = [r.p_value for r in results if not np.isnan(r.p_value)]

        summary_stats = {
            'n_pairs': len(results),
            'n_significant': len(significant_pairs),
            'mean_coherence': np.mean(coherences) if coherences else 0,
            'max_coherence': np.max(coherences) if coherences else 0,
            'min_p_value': np.min(p_values) if p_values else 1,
            'fraction_significant': len(significant_pairs) / max(len(results), 1)
        }

        # Conclusions
        conclusions = []

        if summary_stats['fraction_significant'] > 0.5:
            conclusions.append(
                "Strong evidence for cross-domain coherence: "
                f"{summary_stats['n_significant']}/{summary_stats['n_pairs']} "
                "pairs show significant coherence."
            )
        elif summary_stats['fraction_significant'] > 0.1:
            conclusions.append(
                "Moderate evidence for cross-domain coherence: "
                "some domain pairs show significant relationships."
            )
        else:
            conclusions.append(
                "Limited evidence for cross-domain coherence: "
                "most pairs do not show significant relationships."
            )

        # Check domain-specific patterns
        for (d1, d2), rejected in null_rejected.items():
            if rejected:
                conclusions.append(
                    f"{d1.value.capitalize()}-{d2.value.capitalize()} "
                    "coherence detected."
                )

        # Caveats
        caveats = [
            "Correlation does not imply causation.",
            "Results may be sensitive to time period and frequency range.",
            "Spurious coherence is possible for non-stationary series.",
            "Multiple testing may inflate false positive rate.",
            "Synthetic data used for some indicators may bias results.",
        ]

        return CrossDomainReport(
            domain_pairs=domain_pairs,
            coherence_results=results,
            significant_pairs=significant_pairs,
            null_hypothesis_rejected=null_rejected,
            summary_stats=summary_stats,
            conclusions=conclusions,
            caveats=caveats
        )


def print_cross_domain_report(report: CrossDomainReport):
    """Pretty-print cross-domain report."""
    print("=" * 70)
    print("CROSS-DOMAIN COHERENCE ANALYSIS")
    print("=" * 70)
    print()

    print("SUMMARY STATISTICS:")
    print("-" * 50)
    for key, value in report.summary_stats.items():
        print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")

    print()
    print("DOMAIN PAIRS ANALYZED:")
    print("-" * 50)
    for d1, d2 in report.domain_pairs:
        rejected = report.null_hypothesis_rejected.get((d1, d2), False)
        status = "SIGNIFICANT" if rejected else "not significant"
        print(f"  {d1.value} <-> {d2.value}: {status}")

    print()
    print("TOP SIGNIFICANT PAIRS:")
    print("-" * 50)
    sorted_pairs = sorted(report.significant_pairs, key=lambda x: -x[2])
    for s1, s2, coh in sorted_pairs[:10]:
        print(f"  {s1} <-> {s2}: coherence={coh:.4f}")

    print()
    print("CONCLUSIONS:")
    print("-" * 50)
    for conclusion in report.conclusions:
        print(f"  * {conclusion}")

    print()
    print("CAVEATS:")
    print("-" * 50)
    for caveat in report.caveats:
        print(f"  ! {caveat}")

    print("=" * 70)


if __name__ == "__main__":
    print("=" * 70)
    print("Cross-Domain Coherence Analyzer - Demo")
    print("=" * 70)

    np.random.seed(42)

    # Create analyzer
    analyzer = CrossDomainAnalyzer(significance_level=0.05)

    # Generate synthetic cross-domain data
    n = 520  # 10 years weekly
    dates = pd.date_range(start='2014-01-01', periods=n, freq='W')
    t = np.arange(n)

    # Common hidden driver (represents shared external factor)
    driver = np.sin(2 * np.pi * t / 156)  # 3-year cycle

    # Climate: ENSO-like
    climate_data = pd.DataFrame({
        'date': dates,
        'value': 0.6 * driver + 0.4 * np.random.randn(n)
    })
    analyzer.add_series('enso', Domain.CLIMATE, climate_data, 'synthetic')

    # Finance: SPY returns influenced by driver
    finance_data = pd.DataFrame({
        'date': dates,
        'value': 0.3 * driver + 0.7 * np.random.randn(n)
    })
    analyzer.add_series('spy_returns', Domain.FINANCE, finance_data, 'synthetic')

    # Epidemiology: Flu cases with seasonal + driver
    seasonal = np.sin(2 * np.pi * t / 52)
    epi_data = pd.DataFrame({
        'date': dates,
        'value': seasonal + 0.4 * driver + 0.3 * np.random.randn(n)
    })
    analyzer.add_series('flu_cases', Domain.EPIDEMIOLOGY, epi_data, 'synthetic')

    # Geophysics: Independent (null control)
    geo_data = pd.DataFrame({
        'date': dates,
        'value': np.random.randn(n)
    })
    analyzer.add_series('earthquakes', Domain.GEOPHYSICS, geo_data, 'synthetic')

    # Social: Google Trends influenced by finance
    social_data = pd.DataFrame({
        'date': dates,
        'value': 0.5 * finance_data['value'].values + 0.5 * np.random.randn(n)
    })
    analyzer.add_series('recession_searches', Domain.SOCIAL, social_data, 'synthetic')

    print("\nAnalyzing all pairs...")
    results = analyzer.analyze_all_pairs(test_significance=True)

    print(f"\nAnalyzed {len(results)} pairs")

    # Generate report
    report = analyzer.generate_report(results)
    print_cross_domain_report(report)

    print("\nTest completed!")
