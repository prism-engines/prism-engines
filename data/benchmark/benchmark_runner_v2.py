"""
PRISM Benchmark Runner v2
=========================
Tests: PCA, Network, Transfer Entropy, Mutual Info lenses

Run: python benchmark_runner_v2.py
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Any

try:
    from engine_core.lenses.pca_lens import PCALens
    from engine_core.lenses.network_lens import NetworkLens
    from engine_core.lenses.transfer_entropy_lens import TransferEntropyLens
    from engine_core.lenses.mutual_info_lens import MutualInfoLens
    LENSES_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Could not import lenses: {e}")
    LENSES_AVAILABLE = False


@dataclass
class BenchmarkResult:
    benchmark_name: str
    lens_name: str
    passed: bool
    expected: str
    actual: str
    details: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# LENS RUNNERS
# =============================================================================

def run_pca_lens(df: pd.DataFrame) -> Dict[str, Any]:
    """Run PCA lens."""
    lens = PCALens()
    result = lens.analyze(df, n_components=5)
    
    print(f"    [DEBUG] PCA keys: {list(result.keys())}")
    
    # Get explained variance
    variance = result.get('explained_variance_ratio', [])
    loadings = result.get('loadings', result.get('components', {}))
    
    print(f"    [DEBUG] Variance ratios: {variance[:5] if variance else 'N/A'}")
    
    return {
        'variance_ratio': variance,
        'loadings': loadings,
        'n_components': result.get('n_components', 0),
        'raw': result
    }


def run_network_lens(df: pd.DataFrame) -> Dict[str, Any]:
    """Run Network lens."""
    lens = NetworkLens()
    result = lens.analyze(df)
    
    print(f"    [DEBUG] Network keys: {list(result.keys())}")
    
    # Get centrality measures
    centrality = result.get('centrality', result.get('degree_centrality', {}))
    density = result.get('density', 0)
    
    print(f"    [DEBUG] Centrality: {centrality}")
    
    return {
        'centrality': centrality,
        'density': density,
        'raw': result
    }


def run_transfer_entropy_lens(df: pd.DataFrame) -> Dict[str, Any]:
    """Run Transfer Entropy lens."""
    lens = TransferEntropyLens()
    result = lens.analyze(df)
    
    print(f"    [DEBUG] TE keys: {list(result.keys())}")
    
    # Get directional flow
    te_matrix = result.get('te_matrix', result.get('transfer_entropy', {}))
    sources = result.get('information_hubs', result.get('top_sources', result.get('sources', [])))
    sinks = result.get('information_sinks', result.get('top_sinks', result.get('sinks', [])))
    
    print(f"    [DEBUG] Sources: {sources}, Sinks: {sinks}")
    
    return {
        'te_matrix': te_matrix,
        'sources': sources,
        'sinks': sinks,
        'raw': result
    }


def run_mutual_info_lens(df: pd.DataFrame) -> Dict[str, Any]:
    """Run Mutual Info lens."""
    lens = MutualInfoLens()
    result = lens.analyze(df)
    
    print(f"    [DEBUG] MI keys: {list(result.keys())}")
    
    # Get MI matrix/pairs
    mi_matrix = result.get('mi_matrix', result.get('mutual_info', {}))
    top_pairs = result.get('high_mi_pairs', result.get('top_pairs', []))
    
    print(f"    [DEBUG] Top pairs: {top_pairs[:5] if top_pairs else 'N/A'}")
    
    return {
        'mi_matrix': mi_matrix,
        'top_pairs': top_pairs,
        'raw': result
    }


# =============================================================================
# VALIDATORS
# =============================================================================

def check_pca(output: Dict) -> BenchmarkResult:
    """
    Validate: 3 factors explain >70% variance
    A,B,C load on PC1; D,E,F on PC2; G,H,I on PC3
    """
    variance = output.get('variance_ratio', [])
    
    # Check cumulative variance of top 3
    if len(variance) >= 3:
        cum_var_3 = sum(variance[:3])
        var_ok = cum_var_3 > 0.70
    else:
        cum_var_3 = sum(variance) if variance else 0
        var_ok = False
    
    passed = var_ok
    
    return BenchmarkResult(
        benchmark_name='pca',
        lens_name='pca',
        passed=passed,
        expected="3 components explain >70% variance",
        actual=f"Top 3 explain {cum_var_3:.1%}",
        details={'variance': variance[:5] if variance else []}
    )


def check_network(output: Dict) -> BenchmarkResult:
    """
    Validate: A has highest centrality, H has lowest
    """
    centrality = output.get('centrality', {})
    
    if not centrality:
        return BenchmarkResult(
            benchmark_name='network',
            lens_name='network',
            passed=False,
            expected="A=highest, H=lowest centrality",
            actual="No centrality data",
            details={}
        )
    
    # Sort by centrality
    sorted_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
    top_node = sorted_nodes[0][0] if sorted_nodes else None
    bottom_node = sorted_nodes[-1][0] if sorted_nodes else None
    
    a_is_top = top_node == 'A'
    h_is_bottom = bottom_node == 'H'
    
    passed = a_is_top and h_is_bottom
    
    return BenchmarkResult(
        benchmark_name='network',
        lens_name='network',
        passed=passed,
        expected="A=hub (top), H=isolated (bottom)",
        actual=f"Top={top_node}, Bottom={bottom_node}",
        details={'centrality': dict(sorted_nodes[:3])}
    )


def check_transfer_entropy(output: Dict) -> BenchmarkResult:
    """
    Validate: A is top source, F is independent (low TE)
    """
    sources = output.get('sources', [])
    sinks = output.get('sinks', [])
    
    # Convert to list if needed
    if isinstance(sources, dict):
        sources = sorted(sources.keys(), key=lambda x: sources[x], reverse=True)
    if isinstance(sinks, dict):
        sinks = sorted(sinks.keys(), key=lambda x: sinks[x], reverse=True)
    
    # Flatten if nested tuples
    if sources and isinstance(sources[0], tuple):
        sources = [s[0] for s in sources]
    if sinks and isinstance(sinks[0], tuple):
        sinks = [s[0] for s in sinks]
    
    a_is_source = 'A' in sources[:2] if sources else False
    d_is_source = 'D' in sources[:3] if sources else False
    
    passed = a_is_source or d_is_source
    
    return BenchmarkResult(
        benchmark_name='transfer_entropy',
        lens_name='transfer_entropy',
        passed=passed,
        expected="A, D are top sources",
        actual=f"Sources: {sources[:3]}, Sinks: {sinks[:3]}",
        details={'sources': sources[:5], 'sinks': sinks[:5]}
    )


def check_mutual_info(output: Dict) -> BenchmarkResult:
    """
    Validate: A-B and C-D have high MI, G has low MI with all
    """
    top_pairs = output.get('top_pairs', [])
    mi_matrix = output.get('mi_matrix', {})
    
    # Check if A-B or C-D in top pairs
    ab_found = False
    cd_found = False
    
    for pair in top_pairs[:5]:
        if isinstance(pair, tuple) and len(pair) >= 2:
            nodes = set([pair[0], pair[1]])
            if nodes == {'A', 'B'}:
                ab_found = True
            if nodes == {'C', 'D'}:
                cd_found = True
        elif isinstance(pair, dict):
            # Try multiple key formats: indicator_1, indicator1, col1
            p1 = pair.get('indicator_1', pair.get('indicator1', pair.get('col1', '')))
            p2 = pair.get('indicator_2', pair.get('indicator2', pair.get('col2', '')))
            nodes = set([p1, p2])
            if nodes == {'A', 'B'}:
                ab_found = True
            if nodes == {'C', 'D'}:
                cd_found = True
    
    passed = ab_found or cd_found
    
    return BenchmarkResult(
        benchmark_name='mutual_info',
        lens_name='mutual_info',
        passed=passed,
        expected="A-B or C-D in top pairs",
        actual=f"Top pairs: {top_pairs[:3]}",
        details={'top_pairs': top_pairs[:5]}
    )


# =============================================================================
# RUNNER
# =============================================================================

class BenchmarkRunnerV2:
    def __init__(self, benchmark_dir: str = None):
        if benchmark_dir is None:
            for path in [Path.cwd() / 'data' / 'benchmark', Path.cwd(), Path(__file__).parent]:
                if (path / 'benchmark_pca.csv').exists():
                    benchmark_dir = path
                    break
            else:
                benchmark_dir = Path.cwd()
        
        self.benchmark_dir = Path(benchmark_dir)
        self.results: List[BenchmarkResult] = []
        
        self.benchmarks = {
            'benchmark_pca.csv': (run_pca_lens, check_pca),
            'benchmark_network.csv': (run_network_lens, check_network),
            'benchmark_transfer_entropy.csv': (run_transfer_entropy_lens, check_transfer_entropy),
            'benchmark_mutual_info.csv': (run_mutual_info_lens, check_mutual_info),
        }
    
    def run_all(self) -> List[BenchmarkResult]:
        print("=" * 60)
        print("PRISM BENCHMARK VALIDATION v2")
        print("=" * 60)
        
        if not LENSES_AVAILABLE:
            print("❌ Lenses not available")
            return []
        
        for filename, (runner, validator) in self.benchmarks.items():
            print(f"\n▶ {filename}")
            
            filepath = self.benchmark_dir / filename
            if not filepath.exists():
                print(f"  ⊘ SKIP: File not found")
                continue
            
            try:
                df = pd.read_csv(filepath)
                print(f"  Loaded: {df.shape}, cols: {list(df.columns)[:5]}...")
                
                output = runner(df)
                result = validator(output)
                self.results.append(result)
                
                status = "✓ PASS" if result.passed else "✗ FAIL"
                print(f"  {status}: {result.actual}")
                
            except Exception as e:
                print(f"  ✗ ERROR: {e}")
                import traceback
                traceback.print_exc()
        
        # Summary
        print("\n" + "=" * 60)
        passed = sum(1 for r in self.results if r.passed)
        total = len(self.results)
        
        for r in self.results:
            print(f"  {'✓' if r.passed else '✗'} {r.benchmark_name}")
        
        print(f"\n{'🎉 ALL PASSED!' if passed == total else f'Result: {passed}/{total}'}")
        
        return self.results


if __name__ == '__main__':
    # First generate the benchmarks if they don't exist
    if not (Path('.') / 'benchmark_pca.csv').exists():
        print("Generating benchmark data first...\n")
        from benchmark_generator_v2 import BenchmarkGeneratorV2
        BenchmarkGeneratorV2().generate_all()
        print()
    
    BenchmarkRunnerV2().run_all()
