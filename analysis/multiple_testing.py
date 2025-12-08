"""
Multiple Testing Corrections Module
====================================

When running many statistical tests, false positives accumulate.
This module provides corrections to control error rates.

Methods:
- Bonferroni: Controls Family-Wise Error Rate (FWER) - conservative
- Holm: Step-down FWER control - less conservative
- Benjamini-Hochberg: Controls False Discovery Rate (FDR)
- Benjamini-Yekutieli: FDR under arbitrary dependence

For PRISM:
- Running 14 lenses on 10 indicators = 140+ tests
- Without correction, expect ~7 false positives at alpha=0.05
- FDR correction essential for credible findings
"""

import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class CorrectionResult:
    """Result from multiple testing correction."""
    original_pvalues: List[float]
    adjusted_pvalues: List[float]
    significant: List[bool]
    method: str
    alpha: float
    n_tests: int
    n_significant: int
    n_rejected_before: int  # Without correction

    def summary(self) -> str:
        return (f"Multiple Testing Correction ({self.method})\n"
                f"  Tests: {self.n_tests}\n"
                f"  Significant before correction: {self.n_rejected_before}\n"
                f"  Significant after correction: {self.n_significant}\n"
                f"  False positives prevented: {self.n_rejected_before - self.n_significant}")


def bonferroni(pvalues: List[float],
               alpha: float = 0.05) -> CorrectionResult:
    """
    Bonferroni correction.

    Most conservative. Controls FWER (probability of ANY false positive).
    Adjusted p-value = p * n

    Args:
        pvalues: List of p-values from individual tests
        alpha: Significance level

    Returns:
        CorrectionResult with adjusted p-values
    """
    pvalues = np.asarray(pvalues)
    n = len(pvalues)

    # Adjust p-values
    adjusted = np.minimum(pvalues * n, 1.0)

    # Determine significance
    significant = adjusted < alpha

    return CorrectionResult(
        original_pvalues=pvalues.tolist(),
        adjusted_pvalues=adjusted.tolist(),
        significant=significant.tolist(),
        method='bonferroni',
        alpha=alpha,
        n_tests=n,
        n_significant=int(np.sum(significant)),
        n_rejected_before=int(np.sum(pvalues < alpha))
    )


def holm(pvalues: List[float],
         alpha: float = 0.05) -> CorrectionResult:
    """
    Holm-Bonferroni step-down procedure.

    Less conservative than Bonferroni while still controlling FWER.
    Tests p-values from smallest to largest, stops at first non-rejection.

    Args:
        pvalues: List of p-values
        alpha: Significance level

    Returns:
        CorrectionResult
    """
    pvalues = np.asarray(pvalues)
    n = len(pvalues)

    # Sort p-values and track original indices
    sorted_indices = np.argsort(pvalues)
    sorted_pvalues = pvalues[sorted_indices]

    # Compute adjusted p-values
    adjusted_sorted = np.zeros(n)
    for i in range(n):
        adjusted_sorted[i] = sorted_pvalues[i] * (n - i)

    # Enforce monotonicity (adjusted p-values should increase)
    for i in range(1, n):
        adjusted_sorted[i] = max(adjusted_sorted[i], adjusted_sorted[i-1])

    # Cap at 1.0
    adjusted_sorted = np.minimum(adjusted_sorted, 1.0)

    # Unsort to original order
    adjusted = np.zeros(n)
    adjusted[sorted_indices] = adjusted_sorted

    significant = adjusted < alpha

    return CorrectionResult(
        original_pvalues=pvalues.tolist(),
        adjusted_pvalues=adjusted.tolist(),
        significant=significant.tolist(),
        method='holm',
        alpha=alpha,
        n_tests=n,
        n_significant=int(np.sum(significant)),
        n_rejected_before=int(np.sum(pvalues < alpha))
    )


def benjamini_hochberg(pvalues: List[float],
                       alpha: float = 0.05) -> CorrectionResult:
    """
    Benjamini-Hochberg procedure.

    Controls False Discovery Rate (FDR) - expected proportion of
    false positives among rejections. Less conservative than FWER methods.

    Recommended for exploratory analysis with many tests.

    Args:
        pvalues: List of p-values
        alpha: Target FDR level

    Returns:
        CorrectionResult
    """
    pvalues = np.asarray(pvalues)
    n = len(pvalues)

    # Sort p-values
    sorted_indices = np.argsort(pvalues)
    sorted_pvalues = pvalues[sorted_indices]

    # Compute adjusted p-values (q-values)
    # q_i = min(p_i * n/i, q_{i+1}) working backwards
    adjusted_sorted = np.zeros(n)

    for i in range(n - 1, -1, -1):
        rank = i + 1
        adjusted_sorted[i] = sorted_pvalues[i] * n / rank
        if i < n - 1:
            adjusted_sorted[i] = min(adjusted_sorted[i], adjusted_sorted[i + 1])

    # Cap at 1.0
    adjusted_sorted = np.minimum(adjusted_sorted, 1.0)

    # Unsort
    adjusted = np.zeros(n)
    adjusted[sorted_indices] = adjusted_sorted

    significant = adjusted < alpha

    return CorrectionResult(
        original_pvalues=pvalues.tolist(),
        adjusted_pvalues=adjusted.tolist(),
        significant=significant.tolist(),
        method='benjamini_hochberg',
        alpha=alpha,
        n_tests=n,
        n_significant=int(np.sum(significant)),
        n_rejected_before=int(np.sum(pvalues < alpha))
    )


def benjamini_yekutieli(pvalues: List[float],
                        alpha: float = 0.05) -> CorrectionResult:
    """
    Benjamini-Yekutieli procedure.

    Controls FDR under arbitrary dependence between tests.
    More conservative than BH but valid when tests are correlated.

    For PRISM, tests ARE correlated (same data, related indicators),
    so BY may be more appropriate than BH.

    Args:
        pvalues: List of p-values
        alpha: Target FDR level

    Returns:
        CorrectionResult
    """
    pvalues = np.asarray(pvalues)
    n = len(pvalues)

    # BY correction factor: sum(1/i) for i = 1 to n
    c_n = np.sum(1.0 / np.arange(1, n + 1))

    # Sort p-values
    sorted_indices = np.argsort(pvalues)
    sorted_pvalues = pvalues[sorted_indices]

    # Compute adjusted p-values with BY factor
    adjusted_sorted = np.zeros(n)

    for i in range(n - 1, -1, -1):
        rank = i + 1
        adjusted_sorted[i] = sorted_pvalues[i] * n * c_n / rank
        if i < n - 1:
            adjusted_sorted[i] = min(adjusted_sorted[i], adjusted_sorted[i + 1])

    # Cap at 1.0
    adjusted_sorted = np.minimum(adjusted_sorted, 1.0)

    # Unsort
    adjusted = np.zeros(n)
    adjusted[sorted_indices] = adjusted_sorted

    significant = adjusted < alpha

    return CorrectionResult(
        original_pvalues=pvalues.tolist(),
        adjusted_pvalues=adjusted.tolist(),
        significant=significant.tolist(),
        method='benjamini_yekutieli',
        alpha=alpha,
        n_tests=n,
        n_significant=int(np.sum(significant)),
        n_rejected_before=int(np.sum(pvalues < alpha))
    )


def storey_qvalue(pvalues: List[float],
                  lambda_val: float = 0.5) -> CorrectionResult:
    """
    Storey's q-value method.

    Estimates the proportion of true null hypotheses (pi_0)
    for more powerful FDR control.

    Args:
        pvalues: List of p-values
        lambda_val: Tuning parameter for pi_0 estimation

    Returns:
        CorrectionResult
    """
    pvalues = np.asarray(pvalues)
    n = len(pvalues)

    # Estimate pi_0 (proportion of true nulls)
    pi0 = np.mean(pvalues > lambda_val) / (1 - lambda_val)
    pi0 = min(pi0, 1.0)

    # Sort p-values
    sorted_indices = np.argsort(pvalues)
    sorted_pvalues = pvalues[sorted_indices]

    # Compute q-values
    qvalues_sorted = np.zeros(n)

    for i in range(n - 1, -1, -1):
        rank = i + 1
        qvalues_sorted[i] = pi0 * sorted_pvalues[i] * n / rank
        if i < n - 1:
            qvalues_sorted[i] = min(qvalues_sorted[i], qvalues_sorted[i + 1])

    qvalues_sorted = np.minimum(qvalues_sorted, 1.0)

    # Unsort
    qvalues = np.zeros(n)
    qvalues[sorted_indices] = qvalues_sorted

    significant = qvalues < 0.05

    result = CorrectionResult(
        original_pvalues=pvalues.tolist(),
        adjusted_pvalues=qvalues.tolist(),
        significant=significant.tolist(),
        method='storey_qvalue',
        alpha=0.05,
        n_tests=n,
        n_significant=int(np.sum(significant)),
        n_rejected_before=int(np.sum(pvalues < 0.05))
    )

    return result


# =============================================================================
# Unified interface
# =============================================================================

def apply_correction(pvalues: List[float],
                     method: str = 'benjamini_hochberg',
                     alpha: float = 0.05) -> CorrectionResult:
    """
    Apply multiple testing correction.

    Args:
        pvalues: List of p-values
        method: Correction method
        alpha: Significance level

    Returns:
        CorrectionResult
    """
    methods = {
        'bonferroni': bonferroni,
        'holm': holm,
        'benjamini_hochberg': benjamini_hochberg,
        'bh': benjamini_hochberg,  # Alias
        'benjamini_yekutieli': benjamini_yekutieli,
        'by': benjamini_yekutieli,  # Alias
        'storey': storey_qvalue,
        'qvalue': storey_qvalue,  # Alias
        'none': lambda p, a: CorrectionResult(
            original_pvalues=list(p),
            adjusted_pvalues=list(p),
            significant=[x < a for x in p],
            method='none',
            alpha=a,
            n_tests=len(p),
            n_significant=sum(x < a for x in p),
            n_rejected_before=sum(x < a for x in p)
        )
    }

    if method not in methods:
        raise ValueError(f"Unknown method: {method}. Available: {list(methods.keys())}")

    return methods[method](pvalues, alpha)


def recommend_correction(n_tests: int,
                         test_dependence: str = 'unknown') -> str:
    """
    Recommend appropriate correction method.

    Args:
        n_tests: Number of tests being performed
        test_dependence: 'independent', 'positive', 'arbitrary', 'unknown'

    Returns:
        Recommended method name
    """
    if n_tests <= 5:
        return 'bonferroni'  # Few tests, be conservative

    if test_dependence == 'independent':
        return 'benjamini_hochberg'

    if test_dependence == 'positive':
        # Positive dependence (PRDS) - BH still valid
        return 'benjamini_hochberg'

    if test_dependence in ['arbitrary', 'unknown']:
        # Use BY for safety
        return 'benjamini_yekutieli'

    return 'benjamini_hochberg'


# =============================================================================
# PRISM-specific helpers
# =============================================================================

def correct_lens_results(lens_pvalues: Dict[str, Dict[str, float]],
                         method: str = 'benjamini_yekutieli',
                         alpha: float = 0.05) -> Dict[str, Any]:
    """
    Apply correction to results from multiple lenses and indicators.

    Args:
        lens_pvalues: Nested dict {lens_name: {indicator: p_value}}
        method: Correction method
        alpha: Significance level

    Returns:
        Dictionary with corrected results and summary
    """
    # Flatten all p-values
    all_tests = []
    for lens, indicators in lens_pvalues.items():
        for indicator, pvalue in indicators.items():
            all_tests.append({
                'lens': lens,
                'indicator': indicator,
                'pvalue': pvalue
            })

    # Extract p-values
    pvalues = [t['pvalue'] for t in all_tests]

    # Apply correction
    result = apply_correction(pvalues, method=method, alpha=alpha)

    # Rebuild structure with adjusted p-values
    corrected = {}
    significant_findings = []

    for i, test in enumerate(all_tests):
        lens = test['lens']
        indicator = test['indicator']

        if lens not in corrected:
            corrected[lens] = {}

        corrected[lens][indicator] = {
            'original_pvalue': test['pvalue'],
            'adjusted_pvalue': result.adjusted_pvalues[i],
            'significant': result.significant[i]
        }

        if result.significant[i]:
            significant_findings.append({
                'lens': lens,
                'indicator': indicator,
                'adjusted_pvalue': result.adjusted_pvalues[i]
            })

    return {
        'corrected_results': corrected,
        'significant_findings': significant_findings,
        'summary': {
            'method': method,
            'alpha': alpha,
            'n_tests': result.n_tests,
            'n_significant_before': result.n_rejected_before,
            'n_significant_after': result.n_significant,
            'false_positives_prevented': result.n_rejected_before - result.n_significant
        }
    }


# =============================================================================
# Visualization helpers
# =============================================================================

def pvalue_histogram_data(pvalues: List[float],
                          n_bins: int = 20) -> Dict[str, Any]:
    """
    Generate histogram data for p-value distribution.

    Under null hypothesis, p-values should be uniform.
    Spike near 0 indicates true effects.

    Args:
        pvalues: List of p-values
        n_bins: Number of histogram bins

    Returns:
        Dict with histogram data and uniformity test
    """
    pvalues = np.asarray(pvalues)

    # Histogram
    counts, bin_edges = np.histogram(pvalues, bins=n_bins, range=(0, 1))

    # Expected under uniform (null)
    expected = len(pvalues) / n_bins

    # Chi-square test for uniformity
    chi2 = np.sum((counts - expected) ** 2 / expected)

    try:
        from scipy import stats
        p_uniform = 1 - stats.chi2.cdf(chi2, df=n_bins - 1)
    except ImportError:
        p_uniform = None

    # Estimate proportion of true nulls (pi_0)
    # Using fraction above 0.5
    pi0_estimate = 2 * np.mean(pvalues > 0.5)
    pi0_estimate = min(pi0_estimate, 1.0)

    return {
        'bin_edges': bin_edges.tolist(),
        'counts': counts.tolist(),
        'expected_uniform': expected,
        'chi2_statistic': round(chi2, 2),
        'p_uniform': round(p_uniform, 4) if p_uniform else None,
        'is_uniform': p_uniform > 0.05 if p_uniform else None,
        'pi0_estimate': round(pi0_estimate, 3),
        'interpretation': (
            "Uniform distribution suggests no true effects" if (p_uniform and p_uniform > 0.05)
            else "Non-uniform distribution suggests some true effects"
        )
    }


if __name__ == "__main__":
    print("=" * 60)
    print("Multiple Testing Corrections Module - Test")
    print("=" * 60)

    np.random.seed(42)

    # Simulate p-values: 80% true nulls, 20% true effects
    n_tests = 100
    n_true_effects = 20

    # True nulls: uniform p-values
    null_pvalues = np.random.uniform(0, 1, n_tests - n_true_effects)

    # True effects: p-values concentrated near 0
    effect_pvalues = np.random.beta(1, 10, n_true_effects)

    pvalues = np.concatenate([null_pvalues, effect_pvalues])
    np.random.shuffle(pvalues)

    print(f"\nSimulated {n_tests} tests:")
    print(f"   True effects: {n_true_effects}")
    print(f"   True nulls: {n_tests - n_true_effects}")
    print(f"   P-values < 0.05 (uncorrected): {np.sum(pvalues < 0.05)}")

    print("\n" + "-" * 60)
    print("CORRECTION METHODS:")
    print("-" * 60)

    methods = ['none', 'bonferroni', 'holm', 'benjamini_hochberg', 'benjamini_yekutieli']

    results = {}
    for method in methods:
        result = apply_correction(pvalues.tolist(), method=method)
        results[method] = result
        print(f"\n{method.upper()}:")
        print(f"   Significant: {result.n_significant}")

    # Verification
    print("\n" + "-" * 60)
    print("VERIFICATION:")

    if results['bonferroni'].n_significant <= results['benjamini_hochberg'].n_significant:
        print("OK: Bonferroni more conservative than BH")

    if results['benjamini_yekutieli'].n_significant <= results['benjamini_hochberg'].n_significant:
        print("OK: BY more conservative than BH")

    print("\nTest completed!")
