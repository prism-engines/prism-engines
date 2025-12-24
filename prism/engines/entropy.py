"""
PRISM Entropy Engine

Measures complexity and predictability of time series.

Measures:
- Shannon entropy (uncertainty)
- Permutation entropy (complexity)
- Sample entropy (regularity)
- Approximate entropy

Phase: Unbound
Normalization: Varies by method
"""

import logging
from typing import Dict, Any, Optional
from datetime import date
from itertools import permutations
from collections import Counter
from math import factorial

import numpy as np
import pandas as pd

from .base import BaseEngine


logger = logging.getLogger(__name__)


class EntropyEngine(BaseEngine):
    """
    Entropy engine for complexity analysis.
    
    Multiple entropy measures for different aspects of complexity.
    
    Outputs:
        - derived.geometry_fingerprints: Entropy values per indicator
    """
    
    name = "entropy"
    phase = "derived"
    default_normalization = None
    
    def run(
        self,
        df: pd.DataFrame,
        run_id: str,
        methods: Optional[list] = None,
        embedding_dim: int = 3,
        **params
    ) -> Dict[str, Any]:
        """
        Run entropy analysis.
        
        Args:
            df: Indicator data
            run_id: Unique run identifier
            methods: List of methods ['shannon', 'permutation', 'sample']
            embedding_dim: Embedding dimension for permutation/sample entropy
        
        Returns:
            Dict with summary metrics
        """
        if methods is None:
            methods = ["shannon", "permutation", "sample"]
        
        df_clean = df
        indicators = list(df_clean.columns)
        
        window_start = df_clean.index.min().date()
        window_end = df_clean.index.max().date()
        
        results = []
        
        for indicator in indicators:
            series = df_clean[indicator].values
            
            indicator_results = {
                "indicator_id": indicator,
            }
            
            if "shannon" in methods:
                # Discretize for Shannon entropy
                discrete = pd.qcut(series, q=8, labels=False, duplicates="drop")
                indicator_results["shannon"] = self._shannon_entropy(discrete)
            
            if "permutation" in methods:
                indicator_results["permutation"] = self._permutation_entropy(
                    series, embedding_dim
                )
            
            if "sample" in methods:
                indicator_results["sample"] = self._sample_entropy(
                    series, embedding_dim
                )
            
            results.append(indicator_results)
        
        # Store as geometry fingerprints
        self._store_entropy(results, window_start, window_end, run_id, methods)
        
        # Summary metrics
        df_results = pd.DataFrame(results)
        
        metrics = {
            "n_indicators": len(indicators),
            "methods": methods,
            "embedding_dim": embedding_dim,
        }
        
        for method in methods:
            if method in df_results.columns:
                values = df_results[method].dropna()
                metrics[f"avg_{method}_entropy"] = float(values.mean())
                metrics[f"std_{method}_entropy"] = float(values.std())
        
        logger.info(f"Entropy analysis complete: {len(results)} indicators")
        
        return metrics
    
    def _shannon_entropy(self, discrete: np.ndarray) -> float:
        """
        Shannon entropy: H = -sum(p * log2(p))
        
        Higher = more uncertain/complex
        """
        counts = Counter(discrete)
        n = len(discrete)
        probs = [count / n for count in counts.values()]
        
        entropy = -sum(p * np.log2(p) for p in probs if p > 0)
        return entropy
    
    def _permutation_entropy(
        self,
        series: np.ndarray,
        dim: int,
        tau: int = 1
    ) -> float:
        """
        Permutation entropy based on ordinal patterns.
        
        Robust to noise, captures complexity of dynamics.
        """
        n = len(series)
        if n < dim * tau:
            return np.nan
        
        # Generate ordinal patterns
        patterns = []
        for i in range(n - (dim - 1) * tau):
            window = series[i:i + dim * tau:tau]
            # Get rank order
            pattern = tuple(np.argsort(window))
            patterns.append(pattern)
        
        # Count pattern frequencies
        counts = Counter(patterns)
        n_patterns = len(patterns)
        probs = [count / n_patterns for count in counts.values()]
        
        # Shannon entropy of patterns
        entropy = -sum(p * np.log2(p) for p in probs if p > 0)
        
        # Normalize by maximum possible entropy
        max_entropy = np.log2(factorial(dim))
        normalized = entropy / max_entropy if max_entropy > 0 else 0
        
        return normalized
    
    def _sample_entropy(
        self,
        series: np.ndarray,
        dim: int,
        r: Optional[float] = None
    ) -> float:
        """
        Sample entropy: SampEn(m, r, N)
        
        Measures regularity. Lower = more regular/predictable.
        """
        n = len(series)
        if r is None:
            r = 0.2 * np.std(series)
        
        if r == 0 or n < dim + 1:
            return np.nan
        
        def _count_matches(templates, r):
            n_templates = len(templates)
            matches = 0
            for i in range(n_templates):
                for j in range(i + 1, n_templates):
                    if np.max(np.abs(templates[i] - templates[j])) < r:
                        matches += 1
            return matches
        
        # Templates of length m
        templates_m = np.array([
            series[i:i + dim] for i in range(n - dim)
        ])
        
        # Templates of length m+1
        templates_m1 = np.array([
            series[i:i + dim + 1] for i in range(n - dim - 1)
        ])
        
        # Count matches
        B = _count_matches(templates_m, r)
        A = _count_matches(templates_m1, r)
        
        if B == 0:
            return np.nan
        
        # Sample entropy
        return -np.log(A / B) if A > 0 else np.nan
    
    def _store_entropy(
        self,
        results: list,
        window_start: date,
        window_end: date,
        run_id: str,
        methods: list
    ):
        """Store entropy values as geometry fingerprints."""
        records = []
        
        for r in results:
            for method in methods:
                if method in r and not np.isnan(r[method]):
                    records.append({
                        "indicator_id": r["indicator_id"],
                        "window_start": window_start,
                        "window_end": window_end,
                        "dimension": f"entropy_{method}",
                        "value": float(r[method]),
                        "run_id": run_id,
                    })
        
        if records:
            df = pd.DataFrame(records)
            self.store_results("geometry_fingerprints", df, run_id)
