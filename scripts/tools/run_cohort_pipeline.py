#!/usr/bin/env python3
"""
PRISM Cohort Pipeline

Discovers structural cohorts, then runs engines per cohort.

Usage:
    python run_cohort_pipeline.py
    python run_cohort_pipeline.py --threshold 0.5
    python run_cohort_pipeline.py --n-clusters 4
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from prism.db.open import open_prism_db
from prism.agents.agent_cohort_discovery import CohortDiscoveryAgent
from prism.engines import list_engines, get_engine


# =============================================================================
# Engine Selection by Cohort Type
# =============================================================================

# Singleton cohorts (1 indicator) — univariate engines only
UNIVARIATE_ENGINES = [
    "entropy",
    "hurst", 
    "spectral",
    # "garch",             # Commented out - needs arch library 
]

# Multi-indicator cohorts — full engine suite
MULTIVARIATE_ENGINES = [
    "pca",
    "cross_correlation",  # was "correlation"
    "entropy",
    "wavelet",            # was "wavelets"
    "mutual_information",
    "clustering",
]

# Minimum requirements
MIN_ROWS = 50           # Minimum observations after dropna
MIN_OVERLAP = 0.25      # Minimum cohort overlap ratio


# =============================================================================
# Data Loading
# =============================================================================

def load_clean_data() -> pd.DataFrame:
    """Load clean data from database and pivot to wide format."""
    conn = open_prism_db()

    df = conn.execute("""
        SELECT date, indicator_id, value
        FROM data.indicators
        ORDER BY date
    """).fetchdf()

    conn.close()

    if df.empty:
        raise ValueError("No data in data.indicators")

    # Pivot to wide format
    df_wide = df.pivot(index='date', columns='indicator_id', values='value')
    df_wide.index = pd.to_datetime(df_wide.index)

    return df_wide


# =============================================================================
# Engine Runner
# =============================================================================

def run_engine_safe(engine_name: str, df: pd.DataFrame, run_id: str) -> dict:
    """Run a single engine with error handling."""
    try:
        engine = get_engine(engine_name)
        engine.store_results = lambda *args: None  # Skip DB storage for now
        
        # Normalize if needed
        if engine_name in ["pca", "correlation", "clustering", "mutual_information"]:
            df_input = (df - df.mean()) / df.std()
        elif engine_name in ["garch", "spectral"]:
            df_input = df.pct_change().dropna()
        else:
            df_input = df
        
        result = engine.run(df_input, run_id=run_id)
        return {"status": "success", "metrics": result}
        
    except Exception as e:
        return {"status": "error", "error": str(e)}


def run_engines_for_cohort(
    cohort_id: str,
    cohort_df: pd.DataFrame,
    is_singleton: bool,
    run_id: str,
) -> dict:
    """Run appropriate engines for a cohort."""
    
    engines = UNIVARIATE_ENGINES if is_singleton else MULTIVARIATE_ENGINES
    
    results = {}
    for engine_name in engines:
        result = run_engine_safe(engine_name, cohort_df, run_id)
        results[engine_name] = result
    
    return results


# =============================================================================
# Main Pipeline
# =============================================================================

def run_cohort_pipeline(
    df_wide: pd.DataFrame,
    threshold: float = None,
    n_clusters: int = None,
    verbose: bool = True,
) -> dict:
    """
    Main cohort pipeline.
    
    1. Discover cohorts
    2. Filter viable cohorts
    3. Run engines per cohort
    4. Return results
    """
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # -------------------------------------------------------------------------
    # Step 1: Discover Cohorts
    # -------------------------------------------------------------------------
    if verbose:
        print("=" * 60)
        print("COHORT DISCOVERY")
        print("=" * 60)
    
    agent = CohortDiscoveryAgent(threshold=threshold, n_clusters=n_clusters)
    cohort_result = agent.run(df_wide)
    
    if verbose:
        print(agent.explain())
    
    # -------------------------------------------------------------------------
    # Step 2: Run Engines Per Cohort
    # -------------------------------------------------------------------------
    if verbose:
        print("=" * 60)
        print("ENGINE EXECUTION")
        print("=" * 60)
    
    pipeline_results = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "cohorts": {},
        "skipped": [],
        "summary": {},
    }
    
    for cohort_id, profile in cohort_result.cohorts.items():
        
        if verbose:
            print(f"\n{cohort_id}: {profile.indicators}")
            print(f"  Cadence: {profile.inferred_cadence}, Overlap: {profile.overlap_ratio:.0%}")
        
        # Check viability
        if profile.overlap_ratio < MIN_OVERLAP:
            reason = f"overlap {profile.overlap_ratio:.0%} < {MIN_OVERLAP:.0%}"
            pipeline_results["skipped"].append({"cohort": cohort_id, "reason": reason})
            if verbose:
                print(f"  SKIP: {reason}")
            continue
        
        # Get dense cohort data
        cohort_df = df_wide[profile.indicators].dropna()
        n_rows = len(cohort_df)
        
        if n_rows < MIN_ROWS:
            reason = f"only {n_rows} rows after dropna (min: {MIN_ROWS})"
            pipeline_results["skipped"].append({"cohort": cohort_id, "reason": reason})
            if verbose:
                print(f"  SKIP: {reason}")
            continue
        
        # Determine engine type
        is_singleton = profile.n_indicators == 1
        engine_type = "univariate" if is_singleton else "multivariate"
        
        if verbose:
            print(f"  Running {engine_type} engines on {n_rows} rows...")
        
        # Run engines
        engine_results = run_engines_for_cohort(
            cohort_id=cohort_id,
            cohort_df=cohort_df,
            is_singleton=is_singleton,
            run_id=run_id,
        )
        
        # Store results
        pipeline_results["cohorts"][cohort_id] = {
            "profile": {
                "indicators": profile.indicators,
                "cadence": profile.inferred_cadence,
                "missing_raw": profile.mean_missing_rate,
                "missing_adj": profile.mean_missing_adjusted,
                "overlap": profile.overlap_ratio,
                "confidence": profile.confidence,
            },
            "data": {
                "n_rows": n_rows,
                "n_indicators": profile.n_indicators,
            },
            "engines": engine_results,
        }
        
        # Report
        if verbose:
            succeeded = sum(1 for r in engine_results.values() if r["status"] == "success")
            failed = sum(1 for r in engine_results.values() if r["status"] == "error")
            print(f"  Results: {succeeded} succeeded, {failed} failed")
    
    # -------------------------------------------------------------------------
    # Step 3: Summary
    # -------------------------------------------------------------------------
    pipeline_results["summary"] = {
        "total_cohorts": cohort_result.n_cohorts,
        "cohorts_run": len(pipeline_results["cohorts"]),
        "cohorts_skipped": len(pipeline_results["skipped"]),
        "total_indicators": cohort_result.n_indicators,
    }
    
    if verbose:
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"Cohorts discovered: {pipeline_results['summary']['total_cohorts']}")
        print(f"Cohorts run:        {pipeline_results['summary']['cohorts_run']}")
        print(f"Cohorts skipped:    {pipeline_results['summary']['cohorts_skipped']}")
        
        if pipeline_results["skipped"]:
            print("\nSkipped cohorts:")
            for skip in pipeline_results["skipped"]:
                print(f"  {skip['cohort']}: {skip['reason']}")
    
    return pipeline_results


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="PRISM Cohort Pipeline")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Clustering distance threshold (default: auto)"
    )
    parser.add_argument(
        "--n-clusters",
        type=int,
        default=None,
        help="Force number of clusters"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress verbose output"
    )

    args = parser.parse_args()

    # Load data
    try:
        df_wide = load_clean_data()
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)
    
    # Run pipeline
    results = run_cohort_pipeline(
        df_wide=df_wide,
        threshold=args.threshold,
        n_clusters=args.n_clusters,
        verbose=not args.quiet,
    )
    
    return results


if __name__ == "__main__":
    main()