#!/usr/bin/env python3
"""
PRISM Full Engine Runner

Runs all 20 engines on indicator data and saves results.
No interpretation, just metrics.

Usage:
    python scripts/run_all_engines.py
    python scripts/run_all_engines.py --indicators SPY,XLF,XLK,GLD
    python scripts/run_all_engines.py --start 2020-01-01
    python scripts/run_all_engines.py --output results/baseline.json
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from prism.db.connection import get_db_path, get_connection
from prism.engines import list_engines, get_engine


def load_data(db_path: str, indicators: list, start_date: str = None) -> pd.DataFrame:
    """Load and pivot indicator data."""
    conn = get_connection(Path(db_path) if db_path else None)
    
    ind_list = ", ".join(f"'{i}'" for i in indicators)
    
    query = f"""
        SELECT date, indicator_id, value
        FROM clean.indicators
        WHERE indicator_id IN ({ind_list})
    """
    
    if start_date:
        query += f" AND date >= '{start_date}'"
    
    query += " ORDER BY date"
    
    df = conn.execute(query).fetchdf()
    df_wide = df.pivot(index='date', columns='indicator_id', values='value')
    df_wide = df_wide.dropna()
    
    return df_wide


def prepare_data_variants(df: pd.DataFrame) -> dict:
    """Prepare different normalizations."""
    return {
        "raw": df,
        "zscore": (df - df.mean()) / df.std(),
        "returns": df.pct_change().dropna(),
    }


# Engine -> preferred data type
ENGINE_DATA_MAP = {
    "pca": "zscore",
    "cross_correlation": "zscore",
    "rolling_beta": "returns",
    "clustering": "zscore",
    "hurst": "raw",
    "granger": "returns",
    "mutual_information": "zscore",
    "transfer_entropy": "zscore",
    "wavelet": "raw",
    "entropy": "zscore",
    "spectral": "returns",
    "garch": "returns",
    "hmm": "zscore",
    "dmd": "zscore",
    "dtw": "zscore",
    "copula": "raw",
    "rqa": "zscore",
    "cointegration": "raw",
    "lyapunov": "zscore",
    "distance": "zscore",
}


def run_all_engines(df: pd.DataFrame, verbose: bool = True) -> dict:
    """Run all 20 engines and collect metrics."""
    data_variants = prepare_data_variants(df)
    
    results = {
        "meta": {
            "timestamp": datetime.now().isoformat(),
            "n_indicators": len(df.columns),
            "n_samples": len(df),
            "indicators": list(df.columns),
            "date_range": {
                "start": str(df.index.min().date()),
                "end": str(df.index.max().date()),
            }
        },
        "engines": {}
    }
    
    errors = []
    
    for engine_name in list_engines().keys():
        data_type = ENGINE_DATA_MAP.get(engine_name, "zscore")
        data = data_variants[data_type]
        
        try:
            engine = get_engine(engine_name)
            engine.store_results = lambda *args: None  # Skip DB storage
            
            metrics = engine.run(data, run_id="full_run")
            
            # Keep only numeric metrics (JSON serializable)
            numeric_metrics = {}
            for k, v in metrics.items():
                if isinstance(v, (int, float)) and not pd.isna(v):
                    numeric_metrics[k] = round(v, 6) if isinstance(v, float) else v
                elif isinstance(v, bool):
                    numeric_metrics[k] = v
            
            results["engines"][engine_name] = {
                "status": "success",
                "data_type": data_type,
                "metrics": numeric_metrics
            }
            
            if verbose:
                print(f"✓ {engine_name}: {len(numeric_metrics)} metrics")
                
        except Exception as e:
            results["engines"][engine_name] = {
                "status": "error",
                "error": str(e)
            }
            errors.append((engine_name, str(e)))
            
            if verbose:
                print(f"✗ {engine_name}: {e}")
    
    results["meta"]["engines_succeeded"] = len(results["engines"]) - len(errors)
    results["meta"]["engines_failed"] = len(errors)
    
    return results


def print_summary(results: dict):
    """Print formatted summary."""
    meta = results["meta"]
    
    print()
    print("=" * 60)
    print("PRISM FINGERPRINT")
    print("=" * 60)
    print(f"Timestamp:  {meta['timestamp']}")
    print(f"Indicators: {meta['n_indicators']} ({', '.join(meta['indicators'])})")
    print(f"Samples:    {meta['n_samples']}")
    print(f"Period:     {meta['date_range']['start']} to {meta['date_range']['end']}")
    print(f"Engines:    {meta['engines_succeeded']}/{meta['engines_succeeded'] + meta['engines_failed']} succeeded")
    print()
    
    print("-" * 60)
    print("KEY METRICS")
    print("-" * 60)
    
    engines = results["engines"]
    
    # Structure
    if "pca" in engines and engines["pca"]["status"] == "success":
        m = engines["pca"]["metrics"]
        print(f"PCA:         PC1={m.get('variance_pc1', 0):.1%}, eff_dim={m.get('effective_dimensionality', 0):.2f}")
    
    if "clustering" in engines and engines["clustering"]["status"] == "success":
        m = engines["clustering"]["metrics"]
        print(f"Clustering:  n={m.get('n_clusters', 0)}, silhouette={m.get('silhouette_score', 0):.3f}")
    
    if "hmm" in engines and engines["hmm"]["status"] == "success":
        m = engines["hmm"]["metrics"]
        print(f"HMM:         states={m.get('n_states', 0)}")
    
    # Persistence
    if "hurst" in engines and engines["hurst"]["status"] == "success":
        m = engines["hurst"]["metrics"]
        print(f"Hurst:       avg={m.get('avg_hurst', 0):.3f}")
    
    if "lyapunov" in engines and engines["lyapunov"]["status"] == "success":
        m = engines["lyapunov"]["metrics"]
        print(f"Lyapunov:    avg={m.get('avg_lyapunov', 0):.4f}, chaotic={m.get('chaotic_fraction', 0):.0%}")
    
    # Dependence
    if "cross_correlation" in engines and engines["cross_correlation"]["status"] == "success":
        m = engines["cross_correlation"]["metrics"]
        print(f"Correlation: avg={m.get('avg_abs_correlation', 0):.3f}")
    
    if "copula" in engines and engines["copula"]["status"] == "success":
        m = engines["copula"]["metrics"]
        print(f"Tails:       lower={m.get('avg_lower_tail', 0):.3f}, upper={m.get('avg_upper_tail', 0):.3f}")
    
    if "cointegration" in engines and engines["cointegration"]["status"] == "success":
        m = engines["cointegration"]["metrics"]
        print(f"Cointegr:    {m.get('n_cointegrated', 0)}/{m.get('n_pairs', 0)} pairs ({m.get('cointegration_rate', 0):.0%})")
    
    # Complexity
    if "rqa" in engines and engines["rqa"]["status"] == "success":
        m = engines["rqa"]["metrics"]
        print(f"RQA:         det={m.get('avg_determinism', 0):.3f}, lam={m.get('avg_laminarity', 0):.3f}")
    
    print()
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Run all PRISM engines")
    parser.add_argument("--db", default=str(get_db_path()), help="Database path")
    parser.add_argument("--indicators", default="SPY,XLU,XLF,XLK,XLE,AGG,GLD,VIXCLS",
                        help="Comma-separated indicator list")
    parser.add_argument("--start", default="2005-01-01", help="Start date")
    parser.add_argument("--output", "-o", type=Path, help="Output JSON file")
    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress engine output")
    
    args = parser.parse_args()
    
    indicators = [i.strip() for i in args.indicators.split(",")]
    
    print(f"Loading data: {len(indicators)} indicators from {args.start}...")
    df = load_data(args.db, indicators, args.start)
    print(f"Loaded: {len(df)} rows, {len(df.columns)} indicators")
    print()
    
    print("Running engines...")
    results = run_all_engines(df, verbose=not args.quiet)
    
    print_summary(results)
    
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()
