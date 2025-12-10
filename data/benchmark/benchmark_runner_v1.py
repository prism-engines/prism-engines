"""
PRISM Benchmark Runner v1
=========================
Validates 6 lenses against synthetic data with known ground truth.

Benchmarks:
  - clear_leader: Granger causality
  - two_regimes: Regime detection
  - clusters: Clustering
  - periodic: Wavelet analysis
  - anomalies: Anomaly detection
  - pure_noise: Control (no false positives)

Run: python benchmark_runner_v1.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from typing import Dict, List, Any

# Import lenses
try:
    from engine_core.lenses.granger_lens import GrangerLens
    from engine_core.lenses.regime_switching_lens import RegimeSwitchingLens
    from engine_core.lenses.clustering_lens import ClusteringLens
    from engine_core.lenses.wavelet_lens import WaveletLens
    from engine_core.lenses.anomaly_lens import AnomalyLens
    LENSES_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Could not import lenses: {e}")
    LENSES_AVAILABLE = False


# =============================================================================
# LENS RUNNERS
# =============================================================================

def run_granger_lens(df: pd.DataFrame) -> Dict[str, Any]:
    """Run Granger causality lens."""
    lens = GrangerLens()
    result = lens.analyze(df)
    
    print(f"    [DEBUG] Granger keys: {list(result.keys())}")
    
    leaders = list(result.get('top_leaders', []))  # Convert to list for slicing
    
    # Build causality network - handle both dict and list formats
    causality = {}
    pairwise = result.get('pairwise_results', {})
    
    if isinstance(pairwise, dict):
        # Dict format: {'A': {'B': {'significant': True}, ...}, ...}
        for source, targets in pairwise.items():
            if isinstance(targets, dict):
                causality[source] = [t for t, v in targets.items() 
                                    if isinstance(v, dict) and v.get('significant', False)]
        a_causes = list(pairwise.get('A', {}).keys()) if isinstance(pairwise.get('A'), dict) else []
    elif isinstance(pairwise, list):
        # List format: [{'source': 'A', 'target': 'B', 'significant': True}, ...]
        for item in pairwise:
            if isinstance(item, dict) and item.get('significant', False):
                source = item.get('source', item.get('cause', ''))
                target = item.get('target', item.get('effect', ''))
                if source and target:
                    if source not in causality:
                        causality[source] = []
                    causality[source].append(target)
        a_causes = causality.get('A', [])
    else:
        a_causes = []
    
    print(f"    [DEBUG] Leaders: {leaders[:3]}, A causes: {a_causes}")
    
    return {
        'leaders': leaders,
        'causality_network': causality,
    }


def run_regime_lens(df: pd.DataFrame) -> Dict[str, Any]:
    """Run regime switching lens."""
    lens = RegimeSwitchingLens()
    result = lens.analyze(df)
    
    print(f"    [DEBUG] Regime keys: {list(result.keys())}")
    
    # Find regime breakpoints from regime_labels (where regime changes)
    labels = result.get('regime_labels', [])
    breakpoints = []
    
    if isinstance(labels, (list, np.ndarray)) and len(labels) > 1:
        labels = list(labels)
        for i in range(1, len(labels)):
            if labels[i] != labels[i-1]:
                breakpoints.append(i)
    
    # Fallback: try durations if no labels
    if not breakpoints:
        durations = result.get('regime_durations', [])
        if isinstance(durations, dict):
            # Dict format: {regime_id: duration, ...}
            durations = list(durations.values())
        durations = list(durations) if durations else []
        cumsum = 0
        for d in durations[:-1]:
            if isinstance(d, (int, float)) and d > 0:
                cumsum += int(d)
                breakpoints.append(cumsum)
    
    print(f"    [DEBUG] Breakpoints: {breakpoints[:5] if breakpoints else []}, n_regimes: {result.get('n_regimes')}")
    
    return {
        'n_regimes': result.get('n_regimes', 0),
        'breakpoints': breakpoints,
    }


def run_clustering_lens(df: pd.DataFrame) -> Dict[str, Any]:
    """Run clustering lens."""
    lens = ClusteringLens()
    result = lens.analyze(df)
    
    print(f"    [DEBUG] Clustering keys: {list(result.keys())}")
    
    clusters = result.get('clusters', {})
    print(f"    [DEBUG] Raw clusters: {clusters}")
    
    # Parse clusters into list of lists
    cluster_lists = []
    if isinstance(clusters, dict):
        for cluster_id, members in clusters.items():
            if isinstance(members, list):
                cluster_lists.append(members)
    
    print(f"    [DEBUG] Parsed clusters: {cluster_lists}")
    
    return {
        'n_clusters': result.get('n_clusters', 0),
        'clusters': cluster_lists,
    }


def run_wavelet_lens(df: pd.DataFrame) -> Dict[str, Any]:
    """Run wavelet lens."""
    lens = WaveletLens()
    result = lens.analyze(df)
    
    print(f"    [DEBUG] Wavelet keys: {list(result.keys())}")
    
    # Get significant periods (multiple cycles) or fall back to dominant_periods
    periods = result.get('significant_periods', [])
    
    if isinstance(periods, list):
        print(f"    [DEBUG] Significant periods: {periods}")
    else:
        dominant = result.get('dominant_periods', {})
        print(f"    [DEBUG] Dominant periods: {dominant}")
        periods = list(set(dominant.values())) if isinstance(dominant, dict) else []
    
    return {'dominant_periods': sorted(periods) if periods else []}


def run_anomaly_lens(df: pd.DataFrame) -> Dict[str, Any]:
    """Run Anomaly lens."""
    lens = AnomalyLens()
    result = lens.analyze(df)
    
    print(f"    [DEBUG] Anomaly keys: {list(result.keys())}")
    
    contributions = result.get('indicator_contribution', {})
    print(f"    [DEBUG] Contributions: {contributions}")
    
    rankings = sorted(contributions.keys(), key=lambda x: contributions.get(x, 0), reverse=True)
    print(f"    [DEBUG] Rankings: {rankings}")
    
    return {
        'contributions': contributions,
        'rankings': rankings,
    }


# =============================================================================
# VALIDATORS
# =============================================================================

def check_clear_leader(output: Dict) -> tuple:
    """Validate: A causes B, C, D."""
    leaders = output.get('leaders', [])
    causality = output.get('causality_network', {})
    
    a_in_top_3 = 'A' in leaders[:3] if leaders else False
    a_causes = set(causality.get('A', []))
    expected_followers = {'B', 'C', 'D'}
    
    passed = a_in_top_3 and len(a_causes & expected_followers) >= 2
    
    return passed, f"Top={leaders[0] if leaders else 'N/A'}, A→{a_causes}"


def check_two_regimes(output: Dict) -> tuple:
    """Validate: Breakpoint near day 500."""
    breakpoints = output.get('breakpoints', [])
    
    # Check if any breakpoint is within 50 days of 500
    near_500 = any(abs(bp - 500) < 50 for bp in breakpoints)
    
    passed = near_500 and len(breakpoints) >= 1
    
    return passed, f"Breakpoints: {breakpoints}"


def check_clusters(output: Dict) -> tuple:
    """Validate: 2-3 distinct clusters found."""
    clusters = output.get('clusters', [])
    n_clusters = len(clusters)
    
    passed = n_clusters >= 2
    
    return passed, f"Found {n_clusters}: {clusters}"


def check_periodic(output: Dict) -> tuple:
    """Validate: Detects ~16, ~32, ~64, ~128 day periods (wavelet scales)."""
    periods = output.get('dominant_periods', [])
    expected = [16, 32, 64, 128]
    tolerance = 10
    
    found = []
    for exp in expected:
        for p in periods:
            if abs(p - exp) <= tolerance:
                found.append(exp)
                break
    
    passed = len(found) >= 2
    
    return passed, f"Periods: {periods}, matched: {found}"


def check_anomalies(output: Dict) -> tuple:
    """Validate: B, C, E rank in top anomaly contributors."""
    rankings = output.get('rankings', [])
    
    expected = {'B', 'C', 'E'}
    top_5 = set(rankings[:5]) if len(rankings) >= 5 else set(rankings)
    
    found = expected & top_5
    passed = len(found) >= 2
    
    return passed, f"Top 5: {rankings[:5]}"


def check_pure_noise(output_granger: Dict, output_regime: Dict) -> tuple:
    """Validate: No strong patterns found (control test)."""
    # Count total "findings"
    findings = 0
    
    # Granger: count significant causalities
    causality = output_granger.get('causality_network', {})
    for source, targets in causality.items():
        findings += len(targets)
    
    # Regime: count breakpoints
    breakpoints = output_regime.get('breakpoints', [])
    findings += len(breakpoints)
    
    # Should have few findings in pure noise
    passed = findings < 20
    confidence = 1.0 - (findings / 30)
    
    return passed, f"Confidence={confidence:.2f}, findings={findings}"


# =============================================================================
# MAIN RUNNER
# =============================================================================

class BenchmarkRunner:
    def __init__(self, benchmark_dir: str = '.'):
        self.benchmark_dir = Path(benchmark_dir)
        self.results = {}
    
    def run_all(self):
        print("=" * 60)
        print("PRISM BENCHMARK VALIDATION")
        print("=" * 60)
        
        if not LENSES_AVAILABLE:
            print("❌ Lenses not available. Check imports.")
            return
        
        benchmarks = [
            ('benchmark_clear_leader.csv', self._test_clear_leader),
            ('benchmark_two_regimes.csv', self._test_two_regimes),
            ('benchmark_clusters.csv', self._test_clusters),
            ('benchmark_periodic.csv', self._test_periodic),
            ('benchmark_anomalies.csv', self._test_anomalies),
            ('benchmark_pure_noise.csv', self._test_pure_noise),
        ]
        
        for filename, test_func in benchmarks:
            print(f"\n▶ {filename}")
            filepath = self.benchmark_dir / filename
            
            if not filepath.exists():
                print(f"  ⊘ SKIP: File not found")
                self.results[filename.replace('benchmark_', '').replace('.csv', '')] = None
                continue
            
            try:
                df = pd.read_csv(filepath)
                print(f"  Loaded: {df.shape}, cols: {list(df.columns)}")
                
                passed, message = test_func(df)
                status = "✓ PASS" if passed else "✗ FAIL"
                print(f"  {status}: {message}")
                
                self.results[filename.replace('benchmark_', '').replace('.csv', '')] = passed
                
            except Exception as e:
                print(f"  ✗ ERROR: {e}")
                import traceback
                traceback.print_exc()
                self.results[filename.replace('benchmark_', '').replace('.csv', '')] = False
        
        # Summary
        print("\n" + "=" * 60)
        for name, passed in self.results.items():
            if passed is None:
                print(f"  ⊘ {name}")
            elif passed:
                print(f"  ✓ {name}")
            else:
                print(f"  ✗ {name}")
        
        passed_count = sum(1 for v in self.results.values() if v is True)
        total = sum(1 for v in self.results.values() if v is not None)
        
        if passed_count == total and total > 0:
            print(f"\n🎉 ALL PASSED!")
        else:
            print(f"\nResult: {passed_count}/{total}")
    
    def _test_clear_leader(self, df: pd.DataFrame) -> tuple:
        output = run_granger_lens(df)
        return check_clear_leader(output)
    
    def _test_two_regimes(self, df: pd.DataFrame) -> tuple:
        output = run_regime_lens(df)
        return check_two_regimes(output)
    
    def _test_clusters(self, df: pd.DataFrame) -> tuple:
        output = run_clustering_lens(df)
        return check_clusters(output)
    
    def _test_periodic(self, df: pd.DataFrame) -> tuple:
        output = run_wavelet_lens(df)
        return check_periodic(output)
    
    def _test_anomalies(self, df: pd.DataFrame) -> tuple:
        output = run_anomaly_lens(df)
        return check_anomalies(output)
    
    def _test_pure_noise(self, df: pd.DataFrame) -> tuple:
        output_granger = run_granger_lens(df)
        output_regime = run_regime_lens(df)
        return check_pure_noise(output_granger, output_regime)


if __name__ == '__main__':
    # Generate benchmarks if they don't exist
    if not Path('benchmark_clear_leader.csv').exists():
        print("Generating benchmark data first...\n")
        from benchmark_generator_v1 import BenchmarkGenerator
        BenchmarkGenerator().generate_all()
        print()
    
    runner = BenchmarkRunner()
    runner.run_all()
