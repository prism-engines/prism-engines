"""
PRISM Engine - Main Entry Point
================================

This is your front door. Run this to execute the full pipeline.

Usage:
    # From command line:
    python main.py
    
    # Or in Colab/Python:
    from main import run_prism
    results = run_prism()
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')


def run_prism(
    data_path: str = None,
    output_dir: str = None,
    lenses: list = None,
    verbose: bool = True,
    use_sql: bool = False
):
    """
    Run the full PRISM analysis pipeline.
    
    Args:
        data_path: Path to data CSV (default: data/raw/master_panel.csv)
        output_dir: Where to save results (default: 06_output/latest/)
        lenses: List of lens names to run (default: all)
        verbose: Print progress
        use_sql: Load data from SQL database instead of CSV
        
    Returns:
        dict with all results
    """
    
    if verbose:
        print("="*60)
        print("PRISM ENGINE")
        print("="*60)
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
    
    # =========================================================================
    # STEP 1: Load Data
    # =========================================================================
    if verbose:
        print("[1/5] Loading data...")
    
    panel = None
    
    # Try SQL first if requested
    if use_sql:
        try:
            from data.sql import SQLDataManager
            import sqlite3
            db = SQLDataManager()
            panel = db.load_cleaned_panel()
            if panel.empty:
                # Try loading from panel storage
                panel = db.load_panel('master_panel')
            if not panel.empty and verbose:
                print(f"  Loaded from SQL database")
        except ImportError as e:
            if verbose:
                print(f"  SQL module not available: {e}")
            panel = None
        except sqlite3.Error as e:
            if verbose:
                print(f"  Database error: {e}")
            panel = None
        except Exception as e:
            if verbose:
                print(f"  Could not load from SQL: {e}")
            panel = None
    
    # Fall back to CSV if SQL didn't work
    if panel is None or panel.empty:
        if data_path is None:
            data_path = PROJECT_ROOT / "data" / "raw" / "master_panel.csv"
        
        data_path = Path(data_path)
        
        if not data_path.exists():
            # Try to find any CSV in data/raw
            raw_dir = PROJECT_ROOT / "data" / "raw"
            csvs = list(raw_dir.glob("*.csv"))
            if csvs:
                if verbose:
                    print(f"  master_panel.csv not found, using: {csvs[0].name}")
                data_path = csvs[0]
            else:
                raise FileNotFoundError(f"No data found in {raw_dir}")
        
        # Load the data
        panel = pd.read_csv(data_path, index_col=0, parse_dates=True)
    
    if verbose:
        print(f"  Loaded: {panel.shape[0]} rows, {panel.shape[1]} columns")
        print(f"  Date range: {panel.index[0]} to {panel.index[-1]}")
        print(f"  Columns: {list(panel.columns[:5])}..." if len(panel.columns) > 5 else f"  Columns: {list(panel.columns)}")
    
    # =========================================================================
    # STEP 2: Clean Data (handle NaNs)
    # =========================================================================
    if verbose:
        print("\n[2/5] Cleaning data...")
    
    nan_before = panel.isna().sum().sum()
    
    # Simple cleaning: forward fill then backward fill
    panel_clean = panel.ffill().bfill()
    
    # Drop any columns that are still all NaN
    panel_clean = panel_clean.dropna(axis=1, how='all')
    
    # Drop any rows that still have NaN
    panel_clean = panel_clean.dropna()
    
    nan_after = panel_clean.isna().sum().sum()
    
    if verbose:
        print(f"  NaNs before: {nan_before}, after: {nan_after}")
        print(f"  Clean shape: {panel_clean.shape[0]} rows, {panel_clean.shape[1]} columns")
    
    # =========================================================================
    # STEP 3: Initialize Lenses
    # =========================================================================
    if verbose:
        print("\n[3/5] Initializing lenses...")
    
    # Import lenses
    from importlib import import_module
    
    available_lenses = {
        'magnitude': 'magnitude_lens.MagnitudeLens',
        'pca': 'pca_lens.PCALens',
        'granger': 'granger_lens.GrangerLens',
        'dmd': 'dmd_lens.DMDLens',
        'influence': 'influence_lens.InfluenceLens',
        'mutual_info': 'mutual_info_lens.MutualInfoLens',
        'clustering': 'clustering_lens.ClusteringLens',
        'decomposition': 'decomposition_lens.DecompositionLens',
        'wavelet': 'wavelet_lens.WaveletLens',
        'network': 'network_lens.NetworkLens',
        'regime_switching': 'regime_switching_lens.RegimeSwitchingLens',
        'anomaly': 'anomaly_lens.AnomalyLens',
        'transfer_entropy': 'transfer_entropy_lens.TransferEntropyLens',
        'tda': 'tda_lens.TDALens',
    }
    
    if lenses is None:
        lenses = list(available_lenses.keys())
    
    lens_instances = {}
    for name in lenses:
        if name not in available_lenses:
            if verbose:
                print(f"  ⚠ Unknown lens: {name}, skipping")
            continue
        
        try:
            module_name, class_name = available_lenses[name].rsplit('.', 1)
            module = import_module(f"05_engine.lenses.{module_name}")
            lens_class = getattr(module, class_name)
            lens_instances[name] = lens_class()
            if verbose:
                print(f"  ✓ {name}")
        except Exception as e:
            if verbose:
                print(f"  ✗ {name}: {e}")
    
    if verbose:
        print(f"  Loaded {len(lens_instances)}/{len(lenses)} lenses")
    
    # =========================================================================
    # STEP 4: Run Analysis
    # =========================================================================
    if verbose:
        print("\n[4/5] Running analysis...")
    
    results = {
        'lens_results': {},
        'importance_rankings': {},
        'errors': {},
    }
    
    for name, lens in lens_instances.items():
        try:
            if verbose:
                print(f"  Running {name}...", end=" ", flush=True)
            
            # Run the lens
            lens_result = lens.analyze(panel_clean)
            results['lens_results'][name] = lens_result
            
            # Extract importance rankings
            if 'importance' in lens_result:
                imp = lens_result['importance']
                if isinstance(imp, pd.Series):
                    sorted_imp = imp.sort_values(ascending=False)
                    results['importance_rankings'][name] = sorted_imp.to_dict()
                elif isinstance(imp, dict):
                    sorted_imp = dict(sorted(imp.items(), key=lambda x: x[1], reverse=True))
                    results['importance_rankings'][name] = sorted_imp
            
            if verbose:
                print("✓")
                
        except Exception as e:
            results['errors'][name] = str(e)
            if verbose:
                print(f"✗ ({e})")
    
    # =========================================================================
    # STEP 5: Compute Consensus
    # =========================================================================
    if verbose:
        print("\n[5/5] Computing consensus...")
    
    if results['importance_rankings']:
        # Get all indicators
        all_indicators = set()
        for rankings in results['importance_rankings'].values():
            all_indicators.update(rankings.keys())
        
        # Compute average rank for each indicator
        indicator_ranks = {ind: [] for ind in all_indicators}
        
        for lens_name, rankings in results['importance_rankings'].items():
            sorted_inds = list(rankings.keys())
            for rank, ind in enumerate(sorted_inds, 1):
                indicator_ranks[ind].append(rank)
        
        # Average rank (lower = more important)
        consensus = {}
        for ind, ranks in indicator_ranks.items():
            if ranks:
                consensus[ind] = {
                    'avg_rank': np.mean(ranks),
                    'std_rank': np.std(ranks),
                    'n_lenses': len(ranks),
                }
        
        # Sort by average rank
        consensus_sorted = dict(sorted(consensus.items(), key=lambda x: x[1]['avg_rank']))
        results['consensus'] = consensus_sorted
        
        if verbose:
            print("\n  TOP 10 CONSENSUS INDICATORS:")
            print("  " + "-"*40)
            for i, (ind, stats) in enumerate(list(consensus_sorted.items())[:10], 1):
                print(f"  {i:2}. {ind:<20} avg_rank={stats['avg_rank']:.1f} (±{stats['std_rank']:.1f})")
    
    # =========================================================================
    # STEP 6: Save Results
    # =========================================================================
    if output_dir is None:
        output_dir = PROJECT_ROOT / "06_output" / "latest"
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save consensus
    consensus_df = pd.DataFrame(results.get('consensus', {})).T
    if not consensus_df.empty:
        consensus_df.to_csv(output_dir / "consensus_indicators.csv")
    
    # Save run metadata
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'data_path': str(data_path),
        'data_shape': list(panel_clean.shape),
        'lenses_run': list(results['lens_results'].keys()),
        'lenses_failed': list(results['errors'].keys()),
        'n_indicators': len(all_indicators) if results['importance_rankings'] else 0,
    }
    
    with open(output_dir / "run_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    if verbose:
        print(f"\n  Results saved to: {output_dir}")
        print("\n" + "="*60)
        print("COMPLETE")
        print("="*60)
    
    return results


# =============================================================================
# QUICK ACCESS FUNCTIONS
# =============================================================================

def load_data(path: str = None) -> pd.DataFrame:
    """Quick load data."""
    if path is None:
        path = PROJECT_ROOT / "data" / "raw" / "master_panel.csv"
    return pd.read_csv(path, index_col=0, parse_dates=True)


def list_lenses() -> list:
    """List available lenses."""
    return [
        'magnitude', 'pca', 'granger', 'dmd', 'influence', 
        'mutual_info', 'clustering', 'decomposition',
        'wavelet', 'network', 'regime_switching', 'anomaly',
        'transfer_entropy', 'tda'
    ]


def quick_test(n_lenses: int = 3) -> dict:
    """Run a quick test with just a few lenses."""
    return run_prism(lenses=list_lenses()[:n_lenses], verbose=True)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    results = run_prism()
