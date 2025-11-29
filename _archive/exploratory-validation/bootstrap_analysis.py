"""
Bootstrap Analysis for VCF Confidence Intervals
================================================

Provides confidence intervals on lens outputs through resampling.
Key question: How stable are the importance rankings?

Usage:
    from bootstrap_analysis import BootstrapAnalyzer
    
    analyzer = BootstrapAnalyzer(n_bootstrap=1000)
    ci = analyzer.importance_confidence_intervals(lens, data)
    # ci['indicator_0'] = (lower_rank, median_rank, upper_rank)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
import warnings


@dataclass
class BootstrapCI:
    """Bootstrap confidence interval result."""
    indicator: str
    point_estimate: float
    ci_lower: float
    ci_upper: float
    ci_level: float
    standard_error: float
    
    @property
    def ci_width(self) -> float:
        return self.ci_upper - self.ci_lower
    
    def __repr__(self):
        return f"{self.indicator}: {self.point_estimate:.3f} [{self.ci_lower:.3f}, {self.ci_upper:.3f}]"


@dataclass
class RankStability:
    """Stability of indicator rankings."""
    indicator: str
    median_rank: float
    rank_ci_lower: int
    rank_ci_upper: int
    rank_std: float
    probability_top_n: Dict[int, float]  # P(rank <= n)
    
    def __repr__(self):
        return f"{self.indicator}: median rank {self.median_rank:.1f} [{self.rank_ci_lower}, {self.rank_ci_upper}]"


class BootstrapAnalyzer:
    """
    Bootstrap analysis for confidence intervals on VCF lens results.
    """
    
    def __init__(
        self,
        n_bootstrap: int = 1000,
        ci_level: float = 0.95,
        random_seed: int = None
    ):
        """
        Args:
            n_bootstrap: Number of bootstrap resamples
            ci_level: Confidence level (0.95 = 95% CI)
            random_seed: For reproducibility
        """
        self.n_bootstrap = n_bootstrap
        self.ci_level = ci_level
        self.random_seed = random_seed
        if random_seed is not None:
            np.random.seed(random_seed)
    
    # =========================================================================
    # RESAMPLING STRATEGIES
    # =========================================================================
    
    def resample_iid(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Standard IID bootstrap - sample rows with replacement.
        Assumes independence (not great for time series).
        """
        indices = np.random.choice(len(data), size=len(data), replace=True)
        return data.iloc[indices].reset_index(drop=True)
    
    def resample_block(
        self, 
        data: pd.DataFrame, 
        block_size: int = None
    ) -> pd.DataFrame:
        """
        Block bootstrap - preserves temporal dependence.
        Resamples blocks of consecutive observations.
        """
        n = len(data)
        
        if block_size is None:
            # Rule of thumb: n^(1/3)
            block_size = max(1, int(n ** (1/3)))
        
        n_blocks = n // block_size
        
        # Sample block starting points with replacement
        block_starts = np.random.choice(n - block_size + 1, size=n_blocks + 1, replace=True)
        
        # Build resampled data
        indices = []
        for start in block_starts:
            indices.extend(range(start, start + block_size))
            if len(indices) >= n:
                break
        
        indices = indices[:n]
        return data.iloc[indices].reset_index(drop=True)
    
    def resample_stationary(self, data: pd.DataFrame, mean_block_size: int = 20) -> pd.DataFrame:
        """
        Stationary bootstrap - random block lengths (geometric distribution).
        Better preserves stationarity properties.
        """
        n = len(data)
        p = 1.0 / mean_block_size  # Geometric parameter
        
        indices = []
        current_pos = np.random.randint(0, n)
        
        while len(indices) < n:
            # Add current position
            indices.append(current_pos)
            
            # Decide whether to continue block or start new one
            if np.random.random() < p:
                # Start new block at random position
                current_pos = np.random.randint(0, n)
            else:
                # Continue block
                current_pos = (current_pos + 1) % n
        
        return data.iloc[indices].reset_index(drop=True)
    
    # =========================================================================
    # IMPORTANCE CONFIDENCE INTERVALS
    # =========================================================================
    
    def importance_confidence_intervals(
        self,
        lens,
        data: pd.DataFrame,
        resample_method: str = 'block'
    ) -> Dict[str, BootstrapCI]:
        """
        Get confidence intervals for importance scores.
        
        Args:
            lens: Lens instance with analyze() method
            data: Input data
            resample_method: 'iid', 'block', or 'stationary'
            
        Returns:
            Dict mapping indicator name to BootstrapCI
        """
        # Choose resampler
        resamplers = {
            'iid': self.resample_iid,
            'block': self.resample_block,
            'stationary': self.resample_stationary,
        }
        resample_fn = resamplers.get(resample_method, self.resample_block)
        
        # Get point estimate
        try:
            point_result = lens.analyze(data)
            point_importance = self._extract_importance(point_result)
        except Exception as e:
            warnings.warn(f"Lens failed on original data: {e}")
            return {}
        
        indicators = list(point_importance.keys())
        
        # Bootstrap
        bootstrap_importance = {ind: [] for ind in indicators}
        
        for b in range(self.n_bootstrap):
            resampled = resample_fn(data)
            
            try:
                result = lens.analyze(resampled)
                importance = self._extract_importance(result)
                
                for ind in indicators:
                    if ind in importance:
                        bootstrap_importance[ind].append(importance[ind])
            except Exception:
                continue
        
        # Build confidence intervals
        alpha = 1 - self.ci_level
        ci_results = {}
        
        for ind in indicators:
            samples = np.array(bootstrap_importance[ind])
            
            if len(samples) < 10:
                warnings.warn(f"Too few bootstrap samples for {ind}")
                continue
            
            ci_results[ind] = BootstrapCI(
                indicator=ind,
                point_estimate=point_importance[ind],
                ci_lower=np.percentile(samples, 100 * alpha / 2),
                ci_upper=np.percentile(samples, 100 * (1 - alpha / 2)),
                ci_level=self.ci_level,
                standard_error=np.std(samples)
            )
        
        return ci_results
    
    def _extract_importance(self, result: Dict) -> Dict[str, float]:
        """Extract importance scores as dict."""
        if 'importance' not in result:
            return {}
        
        imp = result['importance']
        
        if isinstance(imp, pd.Series):
            return imp.to_dict()
        elif isinstance(imp, dict):
            return imp
        elif isinstance(imp, np.ndarray):
            return {f'ind_{i}': v for i, v in enumerate(imp)}
        else:
            return {}
    
    # =========================================================================
    # RANK STABILITY
    # =========================================================================
    
    def rank_stability(
        self,
        lens,
        data: pd.DataFrame,
        resample_method: str = 'block'
    ) -> Dict[str, RankStability]:
        """
        Assess stability of indicator rankings.
        
        Key outputs:
        - How often does each indicator rank in top N?
        - What's the confidence interval on ranks?
        """
        # Choose resampler
        resamplers = {
            'iid': self.resample_iid,
            'block': self.resample_block,
            'stationary': self.resample_stationary,
        }
        resample_fn = resamplers.get(resample_method, self.resample_block)
        
        # Get point estimate
        try:
            point_result = lens.analyze(data)
            point_importance = self._extract_importance(point_result)
        except Exception as e:
            warnings.warn(f"Lens failed: {e}")
            return {}
        
        indicators = list(point_importance.keys())
        n_indicators = len(indicators)
        
        # Bootstrap ranks
        bootstrap_ranks = {ind: [] for ind in indicators}
        
        for b in range(self.n_bootstrap):
            resampled = resample_fn(data)
            
            try:
                result = lens.analyze(resampled)
                importance = self._extract_importance(result)
                
                # Compute ranks (1 = highest importance)
                sorted_inds = sorted(importance.keys(), key=lambda x: importance[x], reverse=True)
                ranks = {ind: i + 1 for i, ind in enumerate(sorted_inds)}
                
                for ind in indicators:
                    if ind in ranks:
                        bootstrap_ranks[ind].append(ranks[ind])
            except Exception:
                continue
        
        # Build stability results
        alpha = 1 - self.ci_level
        results = {}
        
        for ind in indicators:
            ranks = np.array(bootstrap_ranks[ind])
            
            if len(ranks) < 10:
                continue
            
            # Probability of being in top N
            prob_top_n = {}
            for n in [1, 3, 5, 10]:
                if n <= n_indicators:
                    prob_top_n[n] = np.mean(ranks <= n)
            
            results[ind] = RankStability(
                indicator=ind,
                median_rank=np.median(ranks),
                rank_ci_lower=int(np.percentile(ranks, 100 * alpha / 2)),
                rank_ci_upper=int(np.percentile(ranks, 100 * (1 - alpha / 2))),
                rank_std=np.std(ranks),
                probability_top_n=prob_top_n
            )
        
        return results
    
    # =========================================================================
    # CROSS-LENS AGREEMENT STABILITY
    # =========================================================================
    
    def agreement_stability(
        self,
        lenses: List,
        data: pd.DataFrame,
        n_top: int = 5,
        resample_method: str = 'block'
    ) -> Dict[str, Any]:
        """
        Assess stability of cross-lens agreement.
        
        Questions answered:
        - How stable is the agreement between lenses?
        - Which lens pairs have robust agreement?
        """
        resamplers = {
            'iid': self.resample_iid,
            'block': self.resample_block,
            'stationary': self.resample_stationary,
        }
        resample_fn = resamplers.get(resample_method, self.resample_block)
        
        lens_names = [type(l).__name__ for l in lenses]
        n_lenses = len(lenses)
        
        # Store pairwise agreement scores
        agreement_samples = {
            (i, j): [] for i in range(n_lenses) for j in range(i+1, n_lenses)
        }
        
        # Bootstrap
        for b in range(self.n_bootstrap):
            resampled = resample_fn(data)
            
            # Run all lenses
            top_indicators = []
            for lens in lenses:
                try:
                    result = lens.analyze(resampled)
                    importance = self._extract_importance(result)
                    sorted_inds = sorted(importance.keys(), key=lambda x: importance[x], reverse=True)
                    top_indicators.append(set(sorted_inds[:n_top]))
                except:
                    top_indicators.append(set())
            
            # Pairwise Jaccard similarity
            for i in range(n_lenses):
                for j in range(i+1, n_lenses):
                    if top_indicators[i] and top_indicators[j]:
                        intersection = len(top_indicators[i] & top_indicators[j])
                        union = len(top_indicators[i] | top_indicators[j])
                        jaccard = intersection / union if union > 0 else 0
                        agreement_samples[(i, j)].append(jaccard)
        
        # Build results
        results = {
            'pairwise_agreement': {},
            'lens_names': lens_names,
        }
        
        alpha = 1 - self.ci_level
        
        for (i, j), samples in agreement_samples.items():
            if len(samples) >= 10:
                samples = np.array(samples)
                results['pairwise_agreement'][(lens_names[i], lens_names[j])] = {
                    'mean': np.mean(samples),
                    'ci_lower': np.percentile(samples, 100 * alpha / 2),
                    'ci_upper': np.percentile(samples, 100 * (1 - alpha / 2)),
                    'std': np.std(samples),
                }
        
        return results
    
    # =========================================================================
    # REPORTING
    # =========================================================================
    
    def print_importance_report(self, ci_results: Dict[str, BootstrapCI]):
        """Print confidence interval report for importance scores."""
        print("\n" + "="*70)
        print("IMPORTANCE CONFIDENCE INTERVALS")
        print(f"({self.ci_level*100:.0f}% CI, {self.n_bootstrap} bootstrap samples)")
        print("="*70 + "\n")
        
        # Sort by point estimate
        sorted_results = sorted(ci_results.values(), key=lambda x: x.point_estimate, reverse=True)
        
        print(f"{'Indicator':<20} {'Estimate':>10} {'CI Lower':>10} {'CI Upper':>10} {'CI Width':>10}")
        print("-" * 62)
        
        for ci in sorted_results:
            print(f"{ci.indicator:<20} {ci.point_estimate:>10.4f} {ci.ci_lower:>10.4f} {ci.ci_upper:>10.4f} {ci.ci_width:>10.4f}")
        
        print("\nIndicators with tightest CIs (most reliable):")
        by_width = sorted(sorted_results, key=lambda x: x.ci_width)
        for ci in by_width[:3]:
            print(f"  • {ci.indicator}: width = {ci.ci_width:.4f}")
        
        print("\nIndicators with widest CIs (least reliable):")
        for ci in by_width[-3:]:
            print(f"  • {ci.indicator}: width = {ci.ci_width:.4f}")
    
    def print_rank_report(self, stability_results: Dict[str, RankStability]):
        """Print rank stability report."""
        print("\n" + "="*70)
        print("RANK STABILITY ANALYSIS")
        print(f"({self.ci_level*100:.0f}% CI, {self.n_bootstrap} bootstrap samples)")
        print("="*70 + "\n")
        
        # Sort by median rank
        sorted_results = sorted(stability_results.values(), key=lambda x: x.median_rank)
        
        print(f"{'Indicator':<20} {'Median Rank':>12} {'CI':>15} {'P(Top 3)':>10} {'P(Top 5)':>10}")
        print("-" * 70)
        
        for rs in sorted_results:
            ci_str = f"[{rs.rank_ci_lower}, {rs.rank_ci_upper}]"
            p_top3 = rs.probability_top_n.get(3, 0)
            p_top5 = rs.probability_top_n.get(5, 0)
            print(f"{rs.indicator:<20} {rs.median_rank:>12.1f} {ci_str:>15} {p_top3:>10.2f} {p_top5:>10.2f}")
        
        print("\nMost stable rankings (smallest CI width):")
        by_width = sorted(sorted_results, key=lambda x: x.rank_ci_upper - x.rank_ci_lower)
        for rs in by_width[:3]:
            width = rs.rank_ci_upper - rs.rank_ci_lower
            print(f"  • {rs.indicator}: CI width = {width}")
        
        print("\nMost consistent top performers (highest P(Top 3)):")
        by_top3 = sorted(sorted_results, key=lambda x: x.probability_top_n.get(3, 0), reverse=True)
        for rs in by_top3[:3]:
            print(f"  • {rs.indicator}: P(Top 3) = {rs.probability_top_n.get(3, 0):.2f}")


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def quick_bootstrap(
    lens, 
    data: pd.DataFrame, 
    n_bootstrap: int = 500
) -> Tuple[Dict[str, BootstrapCI], Dict[str, RankStability]]:
    """
    Quick bootstrap analysis returning both CIs and rank stability.
    
    Returns:
        (importance_ci, rank_stability)
    """
    analyzer = BootstrapAnalyzer(n_bootstrap=n_bootstrap)
    ci = analyzer.importance_confidence_intervals(lens, data)
    ranks = analyzer.rank_stability(lens, data)
    return ci, ranks


if __name__ == '__main__':
    print("Bootstrap Analyzer - Example")
    print("="*50)
    
    # Generate example data
    np.random.seed(42)
    n = 500
    dates = pd.date_range('2000-01-01', periods=n, freq='D')
    
    data = pd.DataFrame({
        'A': np.cumsum(np.random.randn(n)) * 0.1,
        'B': np.cumsum(np.random.randn(n)) * 0.05,
        'C': np.random.randn(n) * 0.5,
        'D': np.sin(np.arange(n) * 0.1) + 0.1 * np.random.randn(n),
    }, index=dates)
    
    print("Generated test data")
    print(f"Shape: {data.shape}")
    print("\nTo analyze a lens:")
    print("  analyzer = BootstrapAnalyzer(n_bootstrap=500)")
    print("  ci = analyzer.importance_confidence_intervals(lens, data)")
    print("  ranks = analyzer.rank_stability(lens, data)")
    print("  analyzer.print_importance_report(ci)")
    print("  analyzer.print_rank_report(ranks)")
