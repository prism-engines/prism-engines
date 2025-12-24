#!/usr/bin/env python3
"""
PRISM Phase 2: Certified Engine Runner

Runs engines ONLY on certified (indicator, window, engine) combinations.

Rules:
1. Query meta.engine_eligibility for eligible/conditional pairs
2. For each certified (indicator, window): run only allowed_engines
3. No execution without eligibility record - fail loud
4. Skip under-resolved windows entirely

Usage:
    python scripts/run_phase2_engines.py --run-id <run_id>
    python scripts/run_phase2_engines.py --run-id <run_id> --engine pca
    python scripts/run_phase2_engines.py --run-id <run_id> --dry-run

Cross-validated by: Claude, GPT-4
Date: December 2024
"""

import sys
import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Set, Optional
from dataclasses import dataclass

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from prism.db.connection import get_connection


@dataclass
class CertifiedRun:
    """A certified (indicator, window, engines) combination."""
    indicator_id: str
    window_years: float
    geometry: str
    confidence: float
    status: str
    allowed_engines: List[str]
    quality_score: float
    is_optimal: bool


def get_certified_runs(
    run_id: str,
    engine_filter: Optional[str] = None,
    optimal_only: bool = False,
    conn=None,
) -> List[CertifiedRun]:
    """
    Query database for certified (indicator, window) pairs.
    
    Args:
        run_id: Phase 1 run ID
        engine_filter: Only include runs where this engine is allowed
        optimal_only: Only include optimal windows
        
    Returns:
        List of CertifiedRun objects
    """
    if conn is None:
        conn = get_connection()
        should_close = True
    else:
        should_close = False
    
    try:
        # Query eligibility + geometry windows
        query = """
            SELECT 
                e.indicator_id,
                e.window_years,
                e.geometry,
                e.confidence,
                e.status,
                e.allowed_engines,
                g.quality_score,
                g.is_optimal
            FROM meta.engine_eligibility e
            JOIN meta.geometry_windows g 
                ON e.run_id = g.run_id 
                AND e.indicator_id = g.indicator_id 
                AND e.window_years = g.window_years
            WHERE e.run_id = ?
            AND e.status IN ('eligible', 'conditional')
        """
        
        if optimal_only:
            query += " AND g.is_optimal = TRUE"
        
        query += " ORDER BY e.indicator_id, e.window_years"
        
        df = conn.execute(query, [run_id]).fetchdf()
        
        if len(df) == 0:
            return []
        
        runs = []
        for _, row in df.iterrows():
            allowed = json.loads(row['allowed_engines'])
            
            # Apply engine filter
            if engine_filter and engine_filter not in allowed:
                continue
            
            runs.append(CertifiedRun(
                indicator_id=row['indicator_id'],
                window_years=row['window_years'],
                geometry=row['geometry'],
                confidence=row['confidence'],
                status=row['status'],
                allowed_engines=allowed,
                quality_score=row['quality_score'],
                is_optimal=row['is_optimal'],
            ))
        
        return runs
        
    finally:
        if should_close:
            conn.close()


def run_engine_on_window(
    engine_name: str,
    indicator_id: str,
    window_years: float,
    conn=None,
) -> Optional[Dict]:
    """
    Run a single engine on a specific (indicator, window).
    
    This is a stub - actual engine execution would go here.
    """
    # Load data for this window
    if conn is None:
        conn = get_connection()
        should_close = True
    else:
        should_close = False
    
    try:
        df = conn.execute("""
            SELECT date, value
            FROM clean.indicator_values
            WHERE indicator_id = ?
            ORDER BY date
        """, [indicator_id]).fetchdf()
        
        if len(df) == 0:
            return None
        
        # Get last N days for window
        window_days = int(window_years * 252)
        if len(df) < window_days:
            return None
        
        window_data = df.tail(window_days)['value'].dropna().values
        
        # Stub: Return basic stats as placeholder
        # Real implementation would import and run actual engine
        result = {
            'engine': engine_name,
            'indicator_id': indicator_id,
            'window_years': window_years,
            'n_samples': len(window_data),
            'mean': float(np.mean(window_data)),
            'std': float(np.std(window_data)),
            'computed_at': datetime.now().isoformat(),
        }
        
        return result
        
    finally:
        if should_close:
            conn.close()


def run_phase2(
    run_id: str,
    engine_filter: Optional[str] = None,
    optimal_only: bool = False,
    dry_run: bool = False,
    verbose: bool = False,
) -> Dict:
    """
    Run Phase 2: Execute certified engines.
    
    Args:
        run_id: Phase 1 run ID to use for eligibility
        engine_filter: Only run this specific engine
        optimal_only: Only run on optimal windows
        dry_run: Show what would run without executing
        
    Returns:
        Dict with execution results
    """
    print("=" * 70)
    print("PRISM PHASE 2: Certified Engine Execution")
    print("=" * 70)
    print(f"Run ID: {run_id}")
    print(f"Engine filter: {engine_filter or 'all'}")
    print(f"Optimal only: {optimal_only}")
    print(f"Dry run: {dry_run}")
    print()
    
    # Get certified runs
    conn = get_connection()
    certified = get_certified_runs(
        run_id, 
        engine_filter=engine_filter,
        optimal_only=optimal_only,
        conn=conn,
    )
    
    if not certified:
        print("No certified runs found!")
        print(f"Check that run_id '{run_id}' exists in meta.engine_eligibility")
        return {'error': 'No certified runs'}
    
    print(f"Certified (indicator, window) pairs: {len(certified)}")
    
    # Count engines to run
    engine_counts = {}
    for run in certified:
        for eng in run.allowed_engines:
            if engine_filter and eng != engine_filter:
                continue
            engine_counts[eng] = engine_counts.get(eng, 0) + 1
    
    print("\nEngines to run:")
    for eng, count in sorted(engine_counts.items(), key=lambda x: -x[1]):
        print(f"  {eng:20} × {count}")
    
    total_runs = sum(engine_counts.values())
    print(f"\nTotal engine runs: {total_runs}")
    
    if dry_run:
        print("\n[DRY RUN - no engines executed]")
        
        # Show sample of what would run
        print("\nSample certified runs:")
        for run in certified[:10]:
            marker = "★" if run.is_optimal else " "
            engines = run.allowed_engines[:3]
            print(f"  {marker} {run.indicator_id:12} @ {run.window_years:4.1f}y "
                  f"({run.geometry:20}) → {engines}")
        
        if len(certified) > 10:
            print(f"  ... and {len(certified) - 10} more")
        
        return {'dry_run': True, 'certified': len(certified), 'total_runs': total_runs}
    
    # Execute engines
    print("\n" + "-" * 70)
    print("EXECUTING ENGINES")
    print("-" * 70)
    
    results = []
    errors = []
    
    for i, run in enumerate(certified):
        if verbose:
            print(f"\n[{i+1}/{len(certified)}] {run.indicator_id} @ {run.window_years}y")
        
        for eng in run.allowed_engines:
            if engine_filter and eng != engine_filter:
                continue
            
            try:
                result = run_engine_on_window(
                    engine_name=eng,
                    indicator_id=run.indicator_id,
                    window_years=run.window_years,
                    conn=conn,
                )
                
                if result:
                    result['geometry'] = run.geometry
                    result['is_optimal'] = run.is_optimal
                    results.append(result)
                    
                    if verbose:
                        print(f"    ✓ {eng}")
                        
            except Exception as e:
                errors.append({
                    'indicator_id': run.indicator_id,
                    'window_years': run.window_years,
                    'engine': eng,
                    'error': str(e),
                })
                if verbose:
                    print(f"    ✗ {eng}: {e}")
    
    conn.close()
    
    # Summary
    print("\n" + "=" * 70)
    print("PHASE 2 COMPLETE")
    print("=" * 70)
    print(f"  Engine runs completed: {len(results)}")
    print(f"  Errors: {len(errors)}")
    
    return {
        'results': results,
        'errors': errors,
        'certified': len(certified),
        'completed': len(results),
    }


def main():
    parser = argparse.ArgumentParser(description="PRISM Phase 2 Engine Runner")
    parser.add_argument("--run-id", "-r", type=str, required=True,
                       help="Phase 1 run ID")
    parser.add_argument("--engine", "-e", type=str, default=None,
                       help="Only run specific engine")
    parser.add_argument("--optimal-only", action="store_true",
                       help="Only run on optimal windows")
    parser.add_argument("--dry-run", action="store_true",
                       help="Show what would run without executing")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Verbose output")
    args = parser.parse_args()
    
    run_phase2(
        run_id=args.run_id,
        engine_filter=args.engine,
        optimal_only=args.optimal_only,
        dry_run=args.dry_run,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
