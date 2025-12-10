"""
PRISM Full Lens Suite Runner
=============================
Runs all lenses against economic data and generates a comprehensive report.

Usage:
    python prism_suite.py path/to/data.csv
    python prism_suite.py  # Uses default data location

Place in: prism-engine/
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
from typing import Dict, Any, List
import json

# Add project root to path
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import lenses
try:
    from engine_core.lenses.granger_lens import GrangerLens
    from engine_core.lenses.regime_switching_lens import RegimeSwitchingLens
    from engine_core.lenses.clustering_lens import ClusteringLens
    from engine_core.lenses.wavelet_lens import WaveletLens
    from engine_core.lenses.anomaly_lens import AnomalyLens
    from engine_core.lenses.pca_lens import PCALens
    from engine_core.lenses.magnitude_lens import MagnitudeLens
    from engine_core.lenses.mutual_info_lens import MutualInfoLens
    from engine_core.lenses.network_lens import NetworkLens
    from engine_core.lenses.transfer_entropy_lens import TransferEntropyLens
    LENSES_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Could not import some lenses: {e}")
    LENSES_AVAILABLE = False


# =============================================================================
# LENS CONFIGURATIONS
# =============================================================================

LENS_CONFIGS = {
    'magnitude': {
        'class': MagnitudeLens,
        'params': {},
        'description': 'Basic magnitude/volatility analysis',
    },
    'pca': {
        'class': PCALens,
        'params': {'n_components': 5},
        'description': 'Principal Component Analysis',
    },
    'granger': {
        'class': GrangerLens,
        'params': {'max_lag': 5, 'significance': 0.05},
        'description': 'Granger causality testing',
    },
    'regime': {
        'class': RegimeSwitchingLens,
        'params': {},
        'description': 'Regime detection and switching',
    },
    'clustering': {
        'class': ClusteringLens,
        'params': {},
        'description': 'Hierarchical clustering of indicators',
    },
    'wavelet': {
        'class': WaveletLens,
        'params': {'wavelet': 'db4'},
        'description': 'Multi-scale wavelet analysis',
    },
    'anomaly': {
        'class': AnomalyLens,
        'params': {},
        'description': 'Anomaly detection',
    },
    'mutual_info': {
        'class': MutualInfoLens,
        'params': {},
        'description': 'Mutual information analysis',
    },
    'network': {
        'class': NetworkLens,
        'params': {},
        'description': 'Network/graph analysis',
    },
    'transfer_entropy': {
        'class': TransferEntropyLens,
        'params': {},
        'description': 'Transfer entropy (information flow)',
    },
}

# Lenses to skip (add 'temporal' or others here)
SKIP_LENSES = ['temporal']


# =============================================================================
# MAIN RUNNER
# =============================================================================

class PRISMSuiteRunner:
    """Runs the full PRISM lens suite on economic data."""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.results: Dict[str, Any] = {}
        self.rankings: Dict[str, List] = {}
        self.run_time = None
        
    def load_data(self, filepath: str) -> pd.DataFrame:
        """Load economic data from CSV."""
        df = pd.read_csv(filepath)
        
        if self.verbose:
            print(f"Loaded: {filepath}")
            print(f"  Shape: {df.shape}")
            print(f"  Date range: {df['date'].iloc[0]} to {df['date'].iloc[-1]}")
            print(f"  Columns: {[c for c in df.columns if c != 'date'][:10]}...")
        
        return df
    
    def run_lens(self, name: str, df: pd.DataFrame) -> Dict[str, Any]:
        """Run a single lens and capture results."""
        if name in SKIP_LENSES:
            return {'skipped': True, 'reason': 'In skip list'}
        
        config = LENS_CONFIGS.get(name)
        if not config:
            return {'error': f'Unknown lens: {name}'}
        
        try:
            lens = config['class']()
            result = lens.analyze(df, **config['params'])
            
            # Get rankings if available
            try:
                rankings = lens.rank_indicators(df)
                if isinstance(rankings, pd.DataFrame):
                    rankings = rankings.to_dict('records')
                elif isinstance(rankings, pd.Series):
                    rankings = rankings.to_dict()
            except:
                rankings = None
            
            return {
                'success': True,
                'result': result,
                'rankings': rankings,
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def run_all(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Run all lenses and collect results."""
        start_time = datetime.now()
        
        print("\n" + "=" * 60)
        print("PRISM FULL LENS SUITE")
        print("=" * 60)
        print(f"Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Data: {df.shape[0]} rows × {df.shape[1]} columns")
        print("=" * 60)
        
        for name, config in LENS_CONFIGS.items():
            if name in SKIP_LENSES:
                print(f"\n⊘ {name}: SKIPPED")
                continue
                
            print(f"\n▶ Running: {name}")
            print(f"  {config['description']}")
            
            result = self.run_lens(name, df)
            self.results[name] = result
            
            if result.get('success'):
                print(f"  ✓ Complete")
                # Show key findings
                self._print_highlights(name, result)
            elif result.get('error'):
                print(f"  ✗ Error: {result['error']}")
        
        end_time = datetime.now()
        self.run_time = (end_time - start_time).total_seconds()
        
        print("\n" + "=" * 60)
        print(f"COMPLETE - Total time: {self.run_time:.1f}s")
        print("=" * 60)
        
        return self.results
    
    def _print_highlights(self, name: str, result: Dict):
        """Print key highlights from lens results."""
        r = result.get('result', {})
        
        if name == 'granger':
            n_sig = r.get('n_significant', 0)
            leaders = r.get('top_leaders', [])[:3]
            print(f"    Significant pairs: {n_sig}")
            print(f"    Top leaders: {leaders}")
            
        elif name == 'regime':
            n_regimes = r.get('n_regimes', 0)
            current = r.get('current_regime_character', 'unknown')
            print(f"    Regimes detected: {n_regimes}")
            print(f"    Current regime: {current}")
            
        elif name == 'clustering':
            n_clusters = r.get('n_clusters', 0)
            clusters = r.get('clusters', {})
            print(f"    Clusters: {n_clusters}")
            for cid, members in list(clusters.items())[:3]:
                print(f"      {cid}: {members[:5]}...")
                
        elif name == 'wavelet':
            periods = r.get('significant_periods', [])
            print(f"    Significant periods: {periods}")
            
        elif name == 'anomaly':
            n_anomalies = r.get('n_anomalies', 0)
            rate = r.get('anomaly_rate', 0)
            recent = r.get('recent_anomalies', 0)
            print(f"    Total anomalies: {n_anomalies} ({rate:.1%})")
            print(f"    Recent (30d): {recent}")
            
        elif name == 'pca':
            n_components = r.get('n_components', 0)
            variance = r.get('explained_variance_ratio', [])
            if variance:
                cum_var = sum(variance[:3]) if len(variance) >= 3 else sum(variance)
                print(f"    Components: {n_components}")
                print(f"    Top 3 explain: {cum_var:.1%}")
                
        elif name == 'network':
            n_edges = r.get('n_edges', 0)
            density = r.get('density', 0)
            print(f"    Edges: {n_edges}, Density: {density:.2f}")
    
    def get_consensus_rankings(self) -> pd.DataFrame:
        """Aggregate rankings across all lenses."""
        all_rankings = {}
        
        for name, result in self.results.items():
            if not result.get('success'):
                continue
                
            rankings = result.get('rankings')
            if rankings is None:
                continue
            
            # Normalize rankings to {indicator: score} format
            if isinstance(rankings, list):
                for item in rankings:
                    if isinstance(item, dict) and 'indicator' in item:
                        ind = item['indicator']
                        score = item.get('score', item.get('rank', 0))
                        if ind not in all_rankings:
                            all_rankings[ind] = {}
                        all_rankings[ind][name] = score
            elif isinstance(rankings, dict):
                for ind, score in rankings.items():
                    if ind not in all_rankings:
                        all_rankings[ind] = {}
                    all_rankings[ind][name] = score
        
        if not all_rankings:
            return pd.DataFrame()
        
        df = pd.DataFrame(all_rankings).T
        df['mean_score'] = df.mean(axis=1)
        df['n_lenses'] = df.notna().sum(axis=1) - 1  # Exclude mean_score
        df = df.sort_values('mean_score', ascending=False)
        
        return df
    
    def generate_report(self, output_path: str = None) -> str:
        """Generate a text report of findings."""
        lines = []
        lines.append("=" * 70)
        lines.append("PRISM LENS SUITE REPORT")
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"Runtime: {self.run_time:.1f}s")
        lines.append("=" * 70)
        
        # Summary
        success = sum(1 for r in self.results.values() if r.get('success'))
        total = len(self.results)
        lines.append(f"\nLenses run: {success}/{total}")
        
        # Key findings per lens
        for name, result in self.results.items():
            lines.append(f"\n--- {name.upper()} ---")
            if result.get('skipped'):
                lines.append("  SKIPPED")
            elif result.get('error'):
                lines.append(f"  ERROR: {result['error']}")
            else:
                r = result.get('result', {})
                for key, value in list(r.items())[:5]:
                    if not key.startswith('_'):
                        lines.append(f"  {key}: {str(value)[:80]}")
        
        # Consensus rankings
        consensus = self.get_consensus_rankings()
        if not consensus.empty:
            lines.append("\n--- CONSENSUS RANKINGS (Top 10) ---")
            for ind in consensus.head(10).index:
                score = consensus.loc[ind, 'mean_score']
                n = int(consensus.loc[ind, 'n_lenses'])
                lines.append(f"  {ind}: {score:.3f} (from {n} lenses)")
        
        report = "\n".join(lines)
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report)
            print(f"\nReport saved to: {output_path}")
        
        return report
    
    def save_results(self, output_path: str):
        """Save raw results to JSON."""
        # Convert non-serializable objects
        def clean(obj):
            if isinstance(obj, pd.DataFrame):
                return obj.to_dict('records')
            elif isinstance(obj, pd.Series):
                return obj.to_dict()
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.int64, np.float64)):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: clean(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [clean(v) for v in obj]
            return obj
        
        cleaned = clean(self.results)
        
        with open(output_path, 'w') as f:
            json.dump(cleaned, f, indent=2, default=str)
        
        print(f"Results saved to: {output_path}")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Run PRISM lens suite on economic data')
    parser.add_argument('data', nargs='?', default=None, help='Path to CSV data file')
    parser.add_argument('--output', '-o', default='prism_results', help='Output prefix for reports')
    parser.add_argument('--quiet', '-q', action='store_true', help='Minimal output')
    
    args = parser.parse_args()
    
    if not LENSES_AVAILABLE:
        print("❌ Lenses not available. Check imports.")
        sys.exit(1)
    
    # Find data file
    if args.data:
        data_path = args.data
    else:
        # Try default locations
        candidates = [
            'data/economic_data.csv',
            'data/panel_data.csv',
            'data/prism_data.csv',
        ]
        data_path = None
        for c in candidates:
            if Path(c).exists():
                data_path = c
                break
        
        if not data_path:
            print("❌ No data file specified and no default found.")
            print("Usage: python prism_suite.py path/to/data.csv")
            sys.exit(1)
    
    # Run suite
    runner = PRISMSuiteRunner(verbose=not args.quiet)
    df = runner.load_data(data_path)
    runner.run_all(df)
    
    # Generate outputs
    report = runner.generate_report(f"{args.output}_report.txt")
    runner.save_results(f"{args.output}_raw.json")
    
    print("\n" + report)


if __name__ == '__main__':
    main()
