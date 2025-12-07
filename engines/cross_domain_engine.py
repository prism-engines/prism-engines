"""
Cross-Domain Analysis Engine
============================

Discovers rhythmic coherence across any combination of domains.
This is the core PRISM engine that reveals how social, biological,
economic, and climate systems share mathematical patterns.

Key insight: The math doesn't care about domain labels. Rhythms are rhythms.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import warnings

# Import our analysis engines (if available)
try:
    from plugins.engines.hurst_engine import HurstEngine
    HURST_AVAILABLE = True
except ImportError:
    HurstEngine = None
    HURST_AVAILABLE = False

try:
    from plugins.engines.spectral_coherence_engine import SpectralCoherenceEngine
    COHERENCE_AVAILABLE = True
except ImportError:
    SpectralCoherenceEngine = None
    COHERENCE_AVAILABLE = False


class AnalysisMode(Enum):
    ML = "ml"
    META = "meta"
    HYBRID = "hybrid"


@dataclass
class CrossDomainResult:
    """Results from cross-domain analysis."""
    coherence_matrix: Dict[str, Dict[str, float]]
    hurst_values: Dict[str, float]
    shared_rhythms: List[Dict[str, Any]]
    domain_summary: Dict[str, Any]
    cross_domain_insights: List[str]
    raw_results: Dict[str, Any]


class CrossDomainEngine:
    """
    Unified cross-domain rhythm analysis engine.

    Combines spectral coherence and Hurst analysis to discover
    shared patterns across any combination of indicator domains.
    """

    def __init__(self, mode: AnalysisMode = AnalysisMode.META):
        """
        Initialize cross-domain engine.

        Args:
            mode: Analysis approach (ML, META, or HYBRID)
        """
        self.mode = mode
        self.hurst_engine = HurstEngine() if HURST_AVAILABLE else None
        self.coherence_engine = SpectralCoherenceEngine() if COHERENCE_AVAILABLE else None

    def analyze(self,
                data: pd.DataFrame,
                indicator_domains: Dict[str, str],
                **kwargs) -> CrossDomainResult:
        """
        Run cross-domain rhythm analysis.

        Args:
            data: DataFrame with indicators as columns (already normalized)
            indicator_domains: Mapping of column name -> domain name
            **kwargs: Additional engine parameters

        Returns:
            CrossDomainResult with all findings
        """
        results = {
            'hurst': None,
            'coherence': None,
            'mode': self.mode.value
        }

        # Run Hurst analysis (persistence/memory)
        if self.hurst_engine:
            try:
                results['hurst'] = self.hurst_engine.analyze(data, method='both')
            except Exception as e:
                results['hurst'] = {'status': 'error', 'error': str(e)}
        else:
            # Fallback: compute simple Hurst approximation
            results['hurst'] = self._compute_simple_hurst(data)

        # Run Spectral Coherence (frequency relationships)
        if self.coherence_engine:
            try:
                results['coherence'] = self.coherence_engine.analyze(data)
            except Exception as e:
                results['coherence'] = {'status': 'error', 'error': str(e)}
        else:
            # Fallback: compute correlation-based coherence approximation
            results['coherence'] = self._compute_simple_coherence(data)

        # Build coherence matrix
        coherence_matrix = self._build_coherence_matrix(results['coherence'], list(data.columns))

        # Extract Hurst values
        hurst_values = self._extract_hurst_values(results['hurst'])

        # Find shared rhythms (high coherence across domains)
        shared_rhythms = self._find_shared_rhythms(
            results['coherence'],
            indicator_domains
        )

        # Generate domain summary
        domain_summary = self._summarize_by_domain(
            list(data.columns),
            indicator_domains,
            coherence_matrix,
            hurst_values
        )

        # Generate insights
        insights = self._generate_insights(
            shared_rhythms,
            hurst_values,
            indicator_domains
        )

        return CrossDomainResult(
            coherence_matrix=coherence_matrix,
            hurst_values=hurst_values,
            shared_rhythms=shared_rhythms,
            domain_summary=domain_summary,
            cross_domain_insights=insights,
            raw_results=results
        )

    def _compute_simple_hurst(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Compute simple Hurst exponent approximation using R/S method."""
        hurst_values = {}

        for col in data.columns:
            series = data[col].dropna().values
            if len(series) < 20:
                hurst_values[col] = {'status': 'insufficient_data', 'hurst': 0.5}
                continue

            try:
                # Simple R/S Hurst estimation
                n = len(series)
                max_k = min(n // 4, 100)

                if max_k < 8:
                    hurst_values[col] = {'status': 'insufficient_data', 'hurst': 0.5}
                    continue

                rs_values = []
                ns = []

                for k in range(8, max_k + 1):
                    num_chunks = n // k
                    if num_chunks < 1:
                        continue

                    rs_sum = 0
                    for i in range(num_chunks):
                        chunk = series[i * k:(i + 1) * k]
                        mean = np.mean(chunk)
                        cumdev = np.cumsum(chunk - mean)
                        R = np.max(cumdev) - np.min(cumdev)
                        S = np.std(chunk, ddof=1)
                        if S > 0:
                            rs_sum += R / S

                    if num_chunks > 0:
                        rs_values.append(rs_sum / num_chunks)
                        ns.append(k)

                if len(ns) > 2:
                    # Linear regression in log-log space
                    log_n = np.log(ns)
                    log_rs = np.log(rs_values)
                    hurst = np.polyfit(log_n, log_rs, 1)[0]
                    hurst = max(0.0, min(1.0, hurst))  # Clamp to [0, 1]
                    hurst_values[col] = {'status': 'success', 'hurst': round(hurst, 4)}
                else:
                    hurst_values[col] = {'status': 'insufficient_data', 'hurst': 0.5}

            except Exception as e:
                hurst_values[col] = {'status': 'error', 'hurst': 0.5, 'error': str(e)}

        return {
            'status': 'completed',
            'hurst_values': hurst_values,
            'method': 'simple_rs'
        }

    def _compute_simple_coherence(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Compute correlation-based coherence approximation."""
        columns = list(data.columns)
        n_cols = len(columns)

        pairwise_coherence = {}
        top_pairs_by_band = {'all_bands': []}

        # Compute correlation matrix as coherence proxy
        corr_matrix = data.corr().fillna(0)

        for i in range(n_cols):
            for j in range(i + 1, n_cols):
                col1, col2 = columns[i], columns[j]
                pair_key = f"{col1}|{col2}"

                # Use absolute correlation as coherence proxy
                coh = abs(corr_matrix.loc[col1, col2])

                pairwise_coherence[pair_key] = {
                    'mean_coherence': round(coh, 4),
                    'phase': 0 if corr_matrix.loc[col1, col2] >= 0 else np.pi
                }

                top_pairs_by_band['all_bands'].append({
                    'pair': pair_key,
                    'coherence': round(coh, 4),
                    'phase': 0 if corr_matrix.loc[col1, col2] >= 0 else np.pi
                })

        # Sort by coherence
        top_pairs_by_band['all_bands'].sort(key=lambda x: x['coherence'], reverse=True)

        return {
            'status': 'completed',
            'pairwise_coherence': pairwise_coherence,
            'top_pairs_by_band': top_pairs_by_band,
            'method': 'correlation_proxy'
        }

    def _build_coherence_matrix(self,
                                 coherence_results: Optional[Dict],
                                 columns: List[str]) -> Dict[str, Dict[str, float]]:
        """Build pairwise coherence matrix."""
        matrix = {col: {} for col in columns}

        if not coherence_results or coherence_results.get('status') != 'completed':
            return matrix

        for pair_key, pair_data in coherence_results.get('pairwise_coherence', {}).items():
            parts = pair_key.split('|')
            if len(parts) != 2:
                continue
            col1, col2 = parts
            coh = pair_data.get('mean_coherence', 0)

            if col1 in matrix:
                matrix[col1][col2] = round(coh, 4)
            if col2 in matrix:
                matrix[col2][col1] = round(coh, 4)

        # Self-coherence = 1
        for col in columns:
            matrix[col][col] = 1.0

        return matrix

    def _extract_hurst_values(self, hurst_results: Optional[Dict]) -> Dict[str, float]:
        """Extract Hurst exponents."""
        if not hurst_results or hurst_results.get('status') != 'completed':
            return {}

        values = {}
        for indicator, data in hurst_results.get('hurst_values', {}).items():
            if data.get('status') == 'success':
                values[indicator] = data.get('hurst', 0.5)

        return values

    def _find_shared_rhythms(self,
                              coherence_results: Optional[Dict],
                              indicator_domains: Dict[str, str]) -> List[Dict[str, Any]]:
        """
        Find high-coherence pairs that span different domains.

        This is the key cross-domain discovery function.
        """
        shared = []

        if not coherence_results or coherence_results.get('status') != 'completed':
            return shared

        # Look at top pairs by band
        for band, pairs in coherence_results.get('top_pairs_by_band', {}).items():
            for pair_data in pairs[:10]:  # Top 10 per band
                pair_key = pair_data['pair']
                parts = pair_key.split('|')
                if len(parts) != 2:
                    continue
                col1, col2 = parts

                domain1 = indicator_domains.get(col1, 'unknown')
                domain2 = indicator_domains.get(col2, 'unknown')

                # Cross-domain pair?
                if domain1 != domain2:
                    shared.append({
                        'indicator_1': col1,
                        'indicator_2': col2,
                        'domain_1': domain1,
                        'domain_2': domain2,
                        'coherence': pair_data['coherence'],
                        'phase': pair_data.get('phase', 0),
                        'frequency_band': band,
                        'is_cross_domain': True
                    })

        # Sort by coherence
        shared.sort(key=lambda x: x['coherence'], reverse=True)

        return shared

    def _summarize_by_domain(self,
                              columns: List[str],
                              indicator_domains: Dict[str, str],
                              coherence_matrix: Dict,
                              hurst_values: Dict) -> Dict[str, Any]:
        """Summarize results by domain."""
        domains = set(indicator_domains.values())
        summary = {}

        for domain in domains:
            domain_indicators = [c for c in columns if indicator_domains.get(c) == domain]

            # Average Hurst for domain
            domain_hurst = [hurst_values.get(ind, 0.5) for ind in domain_indicators]

            # Average within-domain coherence
            within_coherence = []
            for i, ind1 in enumerate(domain_indicators):
                for ind2 in domain_indicators[i+1:]:
                    if ind2 in coherence_matrix.get(ind1, {}):
                        within_coherence.append(coherence_matrix[ind1][ind2])

            # Average cross-domain coherence
            cross_coherence = []
            for ind1 in domain_indicators:
                for ind2, d2 in indicator_domains.items():
                    if d2 != domain and ind2 in coherence_matrix.get(ind1, {}):
                        cross_coherence.append(coherence_matrix[ind1][ind2])

            avg_hurst = float(np.mean(domain_hurst)) if domain_hurst else 0.5

            summary[domain] = {
                'indicator_count': len(domain_indicators),
                'indicators': domain_indicators,
                'avg_hurst': round(avg_hurst, 4),
                'hurst_interpretation': self._interpret_hurst(avg_hurst),
                'avg_within_coherence': round(float(np.mean(within_coherence)), 4) if within_coherence else 0,
                'avg_cross_coherence': round(float(np.mean(cross_coherence)), 4) if cross_coherence else 0
            }

        return summary

    def _generate_insights(self,
                           shared_rhythms: List[Dict],
                           hurst_values: Dict[str, float],
                           indicator_domains: Dict[str, str]) -> List[str]:
        """Generate human-readable cross-domain insights."""
        insights = []

        # Top cross-domain coherence
        if shared_rhythms:
            top = shared_rhythms[0]
            insights.append(
                f"Strongest cross-domain rhythm: {top['indicator_1']} ({top['domain_1']}) "
                f"and {top['indicator_2']} ({top['domain_2']}) show {top['coherence']:.2f} "
                f"coherence in the {top['frequency_band']} band."
            )

        # Domain character from Hurst
        domains = set(indicator_domains.values())
        for domain in domains:
            domain_inds = [k for k, v in indicator_domains.items() if v == domain]
            domain_hurst = [hurst_values.get(ind, 0.5) for ind in domain_inds if ind in hurst_values]

            if domain_hurst:
                avg_h = np.mean(domain_hurst)
                if avg_h > 0.6:
                    insights.append(f"{domain.title()} indicators show trending behavior (H={avg_h:.2f})")
                elif avg_h < 0.4:
                    insights.append(f"{domain.title()} indicators show mean-reverting behavior (H={avg_h:.2f})")

        # Cross-domain sync
        high_cross = [r for r in shared_rhythms if r['coherence'] > 0.5]
        if len(high_cross) >= 3:
            domain_pairs = set((r['domain_1'], r['domain_2']) for r in high_cross[:5])
            insights.append(
                f"Strong cross-domain synchronization detected between: "
                f"{', '.join(f'{d1}-{d2}' for d1, d2 in domain_pairs)}"
            )

        return insights

    def _interpret_hurst(self, h: float) -> str:
        """Interpret Hurst value."""
        if h > 0.65:
            return "strongly trending"
        elif h > 0.55:
            return "trending"
        elif h > 0.45:
            return "random walk"
        elif h > 0.35:
            return "mean reverting"
        else:
            return "strongly mean reverting"


def run_cross_domain_analysis(
    data: pd.DataFrame,
    indicator_domains: Dict[str, str],
    mode: str = "meta"
) -> Dict[str, Any]:
    """
    Convenience function for cross-domain analysis.

    Args:
        data: DataFrame with indicators as columns
        indicator_domains: Column -> domain mapping
        mode: 'ml', 'meta', or 'hybrid'

    Returns:
        Analysis results dictionary
    """
    engine = CrossDomainEngine(mode=AnalysisMode(mode))
    result = engine.analyze(data, indicator_domains)

    return {
        'status': 'completed',
        'coherence_matrix': result.coherence_matrix,
        'hurst_values': result.hurst_values,
        'shared_rhythms': result.shared_rhythms,
        'domain_summary': result.domain_summary,
        'insights': result.cross_domain_insights
    }


if __name__ == "__main__":
    print("=" * 60)
    print("Cross-Domain Engine Demo")
    print("=" * 60)

    # Create synthetic multi-domain data
    np.random.seed(42)
    n = 512
    t = np.arange(n)

    # Shared rhythm (annual-ish cycle)
    shared_cycle = np.sin(2 * np.pi * t / 365)

    data = pd.DataFrame({
        # Economic
        'spy_returns': shared_cycle * 0.5 + np.random.randn(n) * 0.3,
        'yield_10y': shared_cycle * 0.3 + np.random.randn(n) * 0.2,

        # Climate
        'temp_anomaly': shared_cycle * 0.8 + np.random.randn(n) * 0.2,
        'enso': np.sin(2 * np.pi * t / 365 + np.pi/4) + np.random.randn(n) * 0.3,

        # Biological
        'flu_activity': shared_cycle * 0.7 + np.random.randn(n) * 0.3,

        # Social
        'sentiment': shared_cycle * 0.4 + np.random.randn(n) * 0.4,
    })

    indicator_domains = {
        'spy_returns': 'economic',
        'yield_10y': 'economic',
        'temp_anomaly': 'climate',
        'enso': 'climate',
        'flu_activity': 'biological',
        'sentiment': 'social'
    }

    print("\nRunning cross-domain analysis...")
    result = run_cross_domain_analysis(data, indicator_domains, mode='meta')

    print(f"\nStatus: {result['status']}")

    print("\nDomain Summary:")
    for domain, summary in result['domain_summary'].items():
        print(f"  {domain}: {summary['indicator_count']} indicators, "
              f"H={summary['avg_hurst']:.2f} ({summary['hurst_interpretation']})")

    print("\nTop Cross-Domain Rhythms:")
    for rhythm in result['shared_rhythms'][:5]:
        print(f"  {rhythm['indicator_1']} <-> {rhythm['indicator_2']}: "
              f"{rhythm['coherence']:.3f} ({rhythm['frequency_band']})")

    print("\nInsights:")
    for insight in result['insights']:
        print(f"  - {insight}")
