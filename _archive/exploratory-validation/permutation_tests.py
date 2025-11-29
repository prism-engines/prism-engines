"""
Permutation Tests for VCF Statistical Significance
===================================================

Tests whether lens results are statistically significant by comparing
real results to shuffled/randomized data where no real structure exists.

Key insight: If shuffling destroys the signal, the original finding was real.

Usage:
    from permutation_tests import PermutationTester
    
    tester = PermutationTester(n_permutations=100)
    result = tester.test_lens(lens, real_data)
    
    if result.p_value < 0.05:
        print("Significant!")
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Callable, Any
from dataclasses import dataclass
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing


@dataclass
class PermutationResult:
    """Result of a permutation test."""
    lens_name: str
    metric_name: str
    observed_value: float
    null_distribution: np.ndarray
    p_value: float
    significant: bool
    effect_size: float  # How many std devs from null mean
    
    def __repr__(self):
        sig = "***" if self.p_value < 0.001 else "**" if self.p_value < 0.01 else "*" if self.p_value < 0.05 else ""
        return f"PermutationResult({self.metric_name}: p={self.p_value:.4f}{sig}, effect={self.effect_size:.2f}σ)"


class PermutationTester:
    """
    Tests statistical significance of lens results using permutation tests.
    """
    
    def __init__(
        self, 
        n_permutations: int = 100,
        alpha: float = 0.05,
        random_seed: int = None
    ):
        """
        Args:
            n_permutations: Number of permutations (100-1000 typical)
            alpha: Significance level
            random_seed: For reproducibility
        """
        self.n_permutations = n_permutations
        self.alpha = alpha
        self.random_seed = random_seed
        if random_seed is not None:
            np.random.seed(random_seed)
    
    # =========================================================================
    # SHUFFLING STRATEGIES
    # =========================================================================
    
    def shuffle_temporal(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Shuffle each column independently, destroying temporal structure.
        Preserves marginal distributions but breaks cross-correlations and autocorrelation.
        """
        shuffled = data.copy()
        for col in shuffled.columns:
            shuffled[col] = np.random.permutation(shuffled[col].values)
        return shuffled
    
    def shuffle_rows(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Shuffle rows (time points) together.
        Preserves cross-sectional correlations but breaks temporal patterns.
        """
        indices = np.random.permutation(len(data))
        return data.iloc[indices].reset_index(drop=True)
    
    def shuffle_phase(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Phase randomization - preserves power spectrum but randomizes phases.
        Breaks temporal structure while preserving frequency content.
        """
        shuffled = pd.DataFrame(index=data.index, columns=data.columns)
        
        for col in data.columns:
            x = data[col].values
            n = len(x)
            
            # FFT
            fft = np.fft.rfft(x)
            
            # Random phases (preserve DC and Nyquist)
            phases = np.random.uniform(0, 2*np.pi, len(fft))
            phases[0] = 0  # DC component
            if n % 2 == 0:
                phases[-1] = 0  # Nyquist
            
            # Apply random phases
            fft_shuffled = np.abs(fft) * np.exp(1j * phases)
            
            # Inverse FFT
            shuffled[col] = np.fft.irfft(fft_shuffled, n=n)
        
        return shuffled
    
    def shuffle_block(self, data: pd.DataFrame, block_size: int = 20) -> pd.DataFrame:
        """
        Block bootstrap shuffle - preserves short-range dependencies.
        Good for testing long-range patterns while keeping local structure.
        """
        n = len(data)
        n_blocks = n // block_size
        
        # Create block indices
        block_indices = np.arange(n_blocks)
        np.random.shuffle(block_indices)
        
        # Reconstruct with shuffled blocks
        new_indices = []
        for block_idx in block_indices:
            start = block_idx * block_size
            new_indices.extend(range(start, min(start + block_size, n)))
        
        # Handle remainder
        remainder = n - len(new_indices)
        if remainder > 0:
            new_indices.extend(range(n - remainder, n))
        
        return data.iloc[new_indices].reset_index(drop=True)
    
    # =========================================================================
    # METRIC EXTRACTORS
    # =========================================================================
    
    def extract_metric(self, result: Dict, metric_type: str) -> float:
        """Extract a scalar metric from lens result."""
        
        if metric_type == 'max_importance':
            if 'importance' in result:
                imp = result['importance']
                if isinstance(imp, pd.Series):
                    return imp.max()
                elif isinstance(imp, dict):
                    return max(imp.values())
                return float(np.max(imp))
            return 0.0
        
        elif metric_type == 'importance_spread':
            if 'importance' in result:
                imp = result['importance']
                if isinstance(imp, pd.Series):
                    return imp.std()
                elif isinstance(imp, dict):
                    return np.std(list(imp.values()))
                return float(np.std(imp))
            return 0.0
        
        elif metric_type == 'regime_separation':
            if 'regime_separation' in result:
                sep = result['regime_separation']
                if isinstance(sep, pd.Series):
                    return sep.mean()
                elif isinstance(sep, dict):
                    return np.mean(list(sep.values()))
                return float(np.mean(sep))
            return 0.0
        
        elif metric_type == 'total_persistence':
            if 'total_persistence' in result:
                pers = result['total_persistence']
                if isinstance(pers, pd.Series):
                    return pers.sum()
                elif isinstance(pers, dict):
                    return sum(pers.values())
                return float(pers)
            return 0.0
        
        elif metric_type == 'max_transfer_entropy':
            if 'te_matrix' in result:
                return float(np.max(result['te_matrix']))
            return 0.0
        
        elif metric_type == 'network_density':
            if 'mst_edges' in result:
                return len(result['mst_edges'])
            return 0.0
        
        elif metric_type == 'anomaly_rate':
            if 'anomaly_rate' in result:
                rate = result['anomaly_rate']
                if isinstance(rate, dict):
                    return np.mean(list(rate.values()))
                return float(rate)
            return 0.0
        
        elif metric_type == 'explained_variance':
            if 'explained_variance' in result:
                return float(result['explained_variance'])
            elif 'explained_variance_ratio' in result:
                ev = result['explained_variance_ratio']
                return ev[0] if isinstance(ev, (list, np.ndarray)) else float(ev)
            return 0.0
        
        else:
            # Try to find any scalar in result
            for key, val in result.items():
                if isinstance(val, (int, float)) and not np.isnan(val):
                    return float(val)
            return 0.0
    
    def get_default_metrics(self, lens_name: str) -> List[str]:
        """Get appropriate metrics for a lens type."""
        lens_lower = lens_name.lower()
        
        if 'regime' in lens_lower or 'switch' in lens_lower or 'hmm' in lens_lower:
            return ['regime_separation', 'importance_spread']
        elif 'wavelet' in lens_lower or 'spectral' in lens_lower:
            return ['max_importance', 'importance_spread']
        elif 'transfer' in lens_lower or 'entropy' in lens_lower:
            return ['max_transfer_entropy', 'importance_spread']
        elif 'network' in lens_lower or 'graph' in lens_lower:
            return ['network_density', 'importance_spread']
        elif 'anomaly' in lens_lower:
            return ['anomaly_rate', 'importance_spread']
        elif 'topo' in lens_lower or 'tda' in lens_lower:
            return ['total_persistence', 'importance_spread']
        elif 'pca' in lens_lower:
            return ['explained_variance', 'importance_spread']
        else:
            return ['max_importance', 'importance_spread']
    
    # =========================================================================
    # MAIN TESTING
    # =========================================================================
    
    def test_lens(
        self,
        lens,
        data: pd.DataFrame,
        shuffle_method: str = 'temporal',
        metrics: List[str] = None
    ) -> List[PermutationResult]:
        """
        Test a lens for statistical significance.
        
        Args:
            lens: Lens instance with analyze() method
            data: Real data
            shuffle_method: 'temporal', 'rows', 'phase', or 'block'
            metrics: Metrics to test (auto-detected if None)
            
        Returns:
            List of PermutationResult for each metric
        """
        lens_name = type(lens).__name__
        
        if metrics is None:
            metrics = self.get_default_metrics(lens_name)
        
        # Get shuffle function
        shufflers = {
            'temporal': self.shuffle_temporal,
            'rows': self.shuffle_rows,
            'phase': self.shuffle_phase,
            'block': self.shuffle_block,
        }
        shuffle_fn = shufflers.get(shuffle_method, self.shuffle_temporal)
        
        # Run on real data
        try:
            real_result = lens.analyze(data)
        except Exception as e:
            warnings.warn(f"Lens failed on real data: {e}")
            return []
        
        # Extract observed values
        observed = {m: self.extract_metric(real_result, m) for m in metrics}
        
        # Run permutations
        null_distributions = {m: [] for m in metrics}
        
        for i in range(self.n_permutations):
            shuffled_data = shuffle_fn(data)
            
            try:
                shuffled_result = lens.analyze(shuffled_data)
                for m in metrics:
                    null_distributions[m].append(self.extract_metric(shuffled_result, m))
            except Exception:
                # Skip failed permutations
                continue
        
        # Build results
        results = []
        for metric in metrics:
            null = np.array(null_distributions[metric])
            
            if len(null) < 10:
                warnings.warn(f"Too few valid permutations for {metric}")
                continue
            
            obs = observed[metric]
            
            # Two-tailed p-value
            p_value = np.mean(np.abs(null - np.mean(null)) >= np.abs(obs - np.mean(null)))
            
            # Effect size (Cohen's d style)
            null_std = np.std(null)
            effect_size = (obs - np.mean(null)) / null_std if null_std > 0 else 0
            
            results.append(PermutationResult(
                lens_name=lens_name,
                metric_name=metric,
                observed_value=obs,
                null_distribution=null,
                p_value=p_value,
                significant=p_value < self.alpha,
                effect_size=effect_size
            ))
        
        return results
    
    def test_multiple_lenses(
        self,
        lenses: List,
        data: pd.DataFrame,
        shuffle_method: str = 'temporal'
    ) -> Dict[str, List[PermutationResult]]:
        """Test multiple lenses and return organized results."""
        results = {}
        
        for lens in lenses:
            lens_name = type(lens).__name__
            print(f"Testing {lens_name}...", end=" ", flush=True)
            
            lens_results = self.test_lens(lens, data, shuffle_method)
            results[lens_name] = lens_results
            
            n_sig = sum(1 for r in lens_results if r.significant)
            print(f"{n_sig}/{len(lens_results)} significant")
        
        return results
    
    # =========================================================================
    # REPORTING
    # =========================================================================
    
    def print_report(self, results: Dict[str, List[PermutationResult]]):
        """Print formatted permutation test report."""
        print("\n" + "="*70)
        print("PERMUTATION TEST RESULTS")
        print("="*70)
        print(f"Permutations: {self.n_permutations}, α = {self.alpha}\n")
        
        all_results = []
        for lens_name, lens_results in results.items():
            print(f"{lens_name}")
            print("-" * 40)
            
            for r in lens_results:
                sig_stars = "***" if r.p_value < 0.001 else "**" if r.p_value < 0.01 else "*" if r.p_value < 0.05 else ""
                sig_str = "SIGNIFICANT" if r.significant else "not significant"
                
                print(f"  {r.metric_name}:")
                print(f"    Observed: {r.observed_value:.4f}")
                print(f"    Null mean: {np.mean(r.null_distribution):.4f} ± {np.std(r.null_distribution):.4f}")
                print(f"    p-value: {r.p_value:.4f} {sig_stars} ({sig_str})")
                print(f"    Effect size: {r.effect_size:.2f}σ")
            
            all_results.extend(lens_results)
            print()
        
        # Summary
        total = len(all_results)
        significant = sum(1 for r in all_results if r.significant)
        print("="*70)
        print(f"SUMMARY: {significant}/{total} metrics significant at α={self.alpha}")
        
        if significant > 0:
            print("\nSignificant findings:")
            for r in sorted(all_results, key=lambda x: x.p_value):
                if r.significant:
                    print(f"  • {r.lens_name}.{r.metric_name}: p={r.p_value:.4f}, effect={r.effect_size:.1f}σ")
        
        print("="*70)


class ComparativePermutationTest:
    """
    Compare lens results across different conditions (e.g., time periods).
    Tests whether differences between conditions are significant.
    """
    
    def __init__(self, n_permutations: int = 100, random_seed: int = None):
        self.n_permutations = n_permutations
        if random_seed is not None:
            np.random.seed(random_seed)
    
    def test_difference(
        self,
        lens,
        data_a: pd.DataFrame,
        data_b: pd.DataFrame,
        metric: str = 'max_importance'
    ) -> PermutationResult:
        """
        Test whether lens results differ significantly between two datasets.
        
        Uses permutation test on the difference of metrics.
        """
        tester = PermutationTester(n_permutations=1)
        
        # Get observed values
        result_a = lens.analyze(data_a)
        result_b = lens.analyze(data_b)
        
        obs_a = tester.extract_metric(result_a, metric)
        obs_b = tester.extract_metric(result_b, metric)
        observed_diff = obs_a - obs_b
        
        # Combine data for permutation
        combined = pd.concat([data_a, data_b], axis=0, ignore_index=True)
        n_a = len(data_a)
        
        # Permutation test
        null_diffs = []
        for _ in range(self.n_permutations):
            # Random split
            indices = np.random.permutation(len(combined))
            perm_a = combined.iloc[indices[:n_a]]
            perm_b = combined.iloc[indices[n_a:]]
            
            try:
                res_a = lens.analyze(perm_a)
                res_b = lens.analyze(perm_b)
                
                val_a = tester.extract_metric(res_a, metric)
                val_b = tester.extract_metric(res_b, metric)
                null_diffs.append(val_a - val_b)
            except:
                continue
        
        null_diffs = np.array(null_diffs)
        
        # P-value
        p_value = np.mean(np.abs(null_diffs) >= np.abs(observed_diff))
        
        # Effect size
        effect_size = observed_diff / np.std(null_diffs) if np.std(null_diffs) > 0 else 0
        
        return PermutationResult(
            lens_name=type(lens).__name__,
            metric_name=f"{metric}_difference",
            observed_value=observed_diff,
            null_distribution=null_diffs,
            p_value=p_value,
            significant=p_value < 0.05,
            effect_size=effect_size
        )


if __name__ == '__main__':
    print("Permutation Tester - Example")
    print("="*50)
    
    # Generate example data
    np.random.seed(42)
    n = 500
    dates = pd.date_range('2000-01-01', periods=n, freq='D')
    
    # Data with structure (should be significant)
    structured_data = pd.DataFrame({
        'A': np.cumsum(np.random.randn(n)) * 0.1,
        'B': np.cumsum(np.random.randn(n)) * 0.1,
        'C': np.sin(np.arange(n) * 0.1) + 0.2 * np.random.randn(n),
    }, index=dates)
    
    # Add correlation
    structured_data['D'] = 0.7 * structured_data['A'] + 0.3 * np.random.randn(n)
    
    print("Generated structured test data")
    print(f"Shape: {structured_data.shape}")
    print("\nTo test a lens:")
    print("  tester = PermutationTester(n_permutations=100)")
    print("  results = tester.test_lens(my_lens, data)")
    print("  tester.print_report({'MyLens': results})")
