#!/usr/bin/env python3
"""
PRISM Phase 1: Temporal Geometry Scan + Suitability Licensing

The complete pre-engine pipeline:

1. TEMPORAL GEOMETRY SCAN
   For each indicator, test windows [0.5y, 1y, 2y, 3y, 5y, 7y]
   Compute: geometry, confidence, stability, quality_score
   Mark optimal window per indicator

2. MATH SUITABILITY AGENT  
   For each (indicator, window) pair:
   Evaluate: eligible | conditional | ineligible | unknown
   Determine: allowed_engines for THIS window

3. PERSISTENCE
   Write to: meta.geometry_windows, meta.engine_eligibility
   Engine runner queries these tables - no execution without record

Usage:
    python scripts/run_phase1_scan.py
    python scripts/run_phase1_scan.py --indicators SPY,QQQ,DGS10
    python scripts/run_phase1_scan.py --all --persist

Output:
    meta.geometry_windows      - Geometry at each (indicator, window)
    meta.engine_eligibility    - Eligibility at each (indicator, window)

Cross-validated by: Claude, GPT-4
Date: December 2024
"""

import sys
import argparse
import uuid
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from prism.db.connection import get_connection


# =============================================================================
# TEMPORAL GEOMETRY SCANNER
# =============================================================================

class TemporalGeometryScanner:
    """
    Scans each indicator at multiple window sizes.
    
    Output: List of WindowGeometryResult per indicator.
    """
    
    WINDOWS = [0.5, 1.0, 2.0, 3.0, 5.0, 7.0]  # Years
    
    def __init__(self, step_days: int = 63, verbose: bool = False):
        self.step_days = step_days
        self.verbose = verbose
        
        # Lazy import agents
        self._base_agent = None
        self._mv_agent = None
    
    def _get_agents(self):
        """Lazy initialization of geometry agents."""
        if self._base_agent is None:
            from scripts.agent_geometry_signature import GeometrySignatureAgent
            from scripts.agent_multiview_geometry import MultiViewGeometryAgent
            
            self._base_agent = GeometrySignatureAgent(verbose=False)
            self._mv_agent = MultiViewGeometryAgent(
                base_agent=self._base_agent, 
                verbose=False
            )
        return self._base_agent, self._mv_agent
    
    def scan_indicator(
        self,
        indicator_id: str,
        df: pd.DataFrame,
        windows: List[float] = None,
    ) -> List[Dict]:
        """
        Scan single indicator at all window sizes.
        
        Args:
            indicator_id: Indicator to scan
            df: DataFrame with 'date' and 'value' columns
            windows: Optional list of window sizes (years)
            
        Returns:
            List of WindowGeometryResult dicts
        """
        from scripts.agent_multiview_geometry import ViewType
        
        windows = windows or self.WINDOWS
        _, mv_agent = self._get_agents()
        
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date').sort_index()
        
        # Determine views
        spread_indicators = {'T10Y2Y', 'T10Y3M', 'T10YFF', 'TEDRATE'}
        is_spread = indicator_id in spread_indicators
        
        results = []
        
        for window_years in windows:
            window_days = int(window_years * 252)
            
            if len(df) < window_days + self.step_days * 10:
                continue
            
            # Rolling analysis at this window
            geometries = []
            confidences = []
            disagreements = []
            
            dates = df.index.tolist()
            
            for end_idx in range(window_days, len(dates), self.step_days):
                start_idx = end_idx - window_days
                window_data = df.iloc[start_idx:end_idx]['value'].dropna()
                
                if len(window_data) < window_days * 0.8:
                    continue
                
                try:
                    if is_spread:
                        views = [ViewType.LEVEL, ViewType.DEVIATION]
                    else:
                        views = [ViewType.LEVEL, ViewType.RETURNS, ViewType.VOLATILITY]
                    
                    result = mv_agent.analyze(
                        window_data.values, 
                        indicator_id=indicator_id, 
                        views=views
                    )
                    
                    geometries.append(result.consensus_geometry)
                    confidences.append(result.consensus_confidence)
                    disagreements.append(result.disagreement_score)
                    
                except Exception:
                    continue
            
            if len(geometries) < 10:
                continue
            
            # Compute metrics
            geom_counts = pd.Series(geometries).value_counts()
            dominant_geometry = geom_counts.index[0]
            geometry_pct = geom_counts.iloc[0] / len(geometries)
            
            avg_confidence = np.mean(confidences)
            avg_disagreement = np.mean(disagreements)
            confidence_std = np.std(confidences)
            
            n_transitions = sum(
                1 for i in range(1, len(geometries))
                if geometries[i] != geometries[i-1]
            )
            stability = 1.0 - (n_transitions / max(1, len(geometries) - 1))
            
            # Quality score
            quality_score = (
                0.40 * avg_confidence +
                0.30 * (1.0 - avg_disagreement) +
                0.20 * stability +
                0.10 * geometry_pct
            )
            
            results.append({
                'indicator_id': indicator_id,
                'window_years': window_years,
                'window_days': window_days,
                'dominant_geometry': dominant_geometry,
                'geometry_pct': geometry_pct,
                'avg_confidence': avg_confidence,
                'avg_disagreement': avg_disagreement,
                'confidence_std': confidence_std,
                'n_observations': len(geometries),
                'n_transitions': n_transitions,
                'stability': stability,
                'quality_score': quality_score,
                'is_optimal': False,
            })
        
        # Mark optimal window
        if results:
            best_idx = max(range(len(results)), key=lambda i: results[i]['quality_score'])
            results[best_idx]['is_optimal'] = True
        
        if self.verbose and results:
            opt = [r for r in results if r['is_optimal']][0]
            print(f"  {indicator_id}: optimal={opt['window_years']}y, "
                  f"geom={opt['dominant_geometry']}, "
                  f"score={opt['quality_score']:.3f}")
        
        return results
    
    def scan_all(
        self,
        indicators: List[str],
        conn=None,
    ) -> Dict[str, List[Dict]]:
        """
        Scan all indicators.
        
        Returns:
            Dict mapping indicator_id -> list of window results
        """
        if conn is None:
            conn = get_connection()
            should_close = True
        else:
            should_close = False
        
        try:
            all_results = {}
            
            for ind_id in indicators:
                df = conn.execute("""
                    SELECT date, value
                    FROM clean.indicator_values
                    WHERE indicator_id = ?
                    ORDER BY date
                """, [ind_id]).fetchdf()
                
                if len(df) < 252:
                    continue
                
                results = self.scan_indicator(ind_id, df)
                if results:
                    all_results[ind_id] = results
            
            return all_results
            
        finally:
            if should_close:
                conn.close()


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_phase1(
    indicators: Optional[List[str]] = None,
    min_rows: int = 500,
    persist: bool = False,
    verbose: bool = False,
) -> Dict:
    """
    Run complete Phase 1 pipeline.
    
    Returns:
        Dict with 'geometry_windows' and 'eligibility' results
    """
    from prism.agents.math_suitability_v2 import (
        WindowSuitabilityAgent,
        WindowSuitabilityPolicy,
        WindowGeometryResult,
        summarize_eligibility,
    )
    
    conn = get_connection()
    
    # Get indicator list
    if indicators is None:
        counts = conn.execute("""
            SELECT indicator_id, COUNT(*) as n
            FROM clean.indicator_values
            GROUP BY indicator_id
            HAVING n >= ?
        """, [min_rows]).fetchdf()
        indicators = counts['indicator_id'].tolist()
    
    print("=" * 70)
    print("PRISM PHASE 1: Temporal Geometry Scan + Suitability")
    print("=" * 70)
    print(f"Indicators: {len(indicators)}")
    print(f"Windows: 0.5, 1, 2, 3, 5, 7 years")
    print()
    
    # =================================================================
    # STEP 1: TEMPORAL GEOMETRY SCAN
    # =================================================================
    print("-" * 70)
    print("STEP 1: Temporal Geometry Scan")
    print("-" * 70)
    
    scanner = TemporalGeometryScanner(verbose=verbose)
    geometry_results = scanner.scan_all(indicators, conn)
    
    total_windows = sum(len(v) for v in geometry_results.values())
    print(f"\nScanned: {len(geometry_results)} indicators, {total_windows} (indicator, window) pairs")
    
    # Show optimal window distribution
    optimal_windows = {}
    for ind_id, results in geometry_results.items():
        for r in results:
            if r['is_optimal']:
                w = r['window_years']
                optimal_windows[w] = optimal_windows.get(w, 0) + 1
    
    print("\nOptimal Window Distribution:")
    for w in sorted(optimal_windows.keys()):
        count = optimal_windows[w]
        pct = 100 * count / len(geometry_results)
        print(f"  {w:4.1f}y: {count:3} ({pct:5.1f}%)")
    
    # =================================================================
    # STEP 2: MATH SUITABILITY EVALUATION
    # =================================================================
    print("\n" + "-" * 70)
    print("STEP 2: Math Suitability Evaluation")
    print("-" * 70)
    
    # Convert to WindowGeometryResult objects
    all_window_results = {}
    for ind_id, results in geometry_results.items():
        all_window_results[ind_id] = [
            WindowGeometryResult(**r) for r in results
        ]
    
    # Evaluate eligibility
    agent = WindowSuitabilityAgent()
    eligibilities = agent.evaluate_all(all_window_results)
    
    summarize_eligibility(eligibilities)
    
    # =================================================================
    # STEP 3: PERSISTENCE
    # =================================================================
    if persist:
        run_id = f"phase1_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        
        print("\n" + "-" * 70)
        print(f"STEP 3: Persisting to Database (run_id: {run_id})")
        print("-" * 70)
        
        # Persist geometry windows
        for ind_id, results in geometry_results.items():
            for r in results:
                conn.execute("""
                    INSERT OR REPLACE INTO meta.geometry_windows
                    (run_id, indicator_id, window_years, window_days,
                     dominant_geometry, geometry_pct, avg_confidence,
                     avg_disagreement, confidence_std, n_observations,
                     n_transitions, stability, quality_score, is_optimal)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, [
                    run_id, r['indicator_id'], r['window_years'], r['window_days'],
                    r['dominant_geometry'], r['geometry_pct'], r['avg_confidence'],
                    r['avg_disagreement'], r['confidence_std'], r['n_observations'],
                    r['n_transitions'], r['stability'], r['quality_score'], r['is_optimal']
                ])
        
        print(f"  Wrote {total_windows} geometry_windows records")
        
        # Persist eligibility
        agent.persist(run_id, eligibilities, conn)
        
        elig_count = sum(len(w) for w in eligibilities.values())
        print(f"  Wrote {elig_count} engine_eligibility records")
    
    conn.close()
    
    # =================================================================
    # SUMMARY
    # =================================================================
    print("\n" + "=" * 70)
    print("PHASE 1 COMPLETE")
    print("=" * 70)
    
    # Count eligible combinations
    eligible_count = 0
    conditional_count = 0
    ineligible_count = 0
    
    for ind_id, windows in eligibilities.items():
        for window_years, elig in windows.items():
            if elig.status.value == "eligible":
                eligible_count += 1
            elif elig.status.value == "conditional":
                conditional_count += 1
            else:
                ineligible_count += 1
    
    print(f"  Indicators scanned: {len(geometry_results)}")
    print(f"  (Indicator, Window) pairs: {total_windows}")
    print(f"  Eligible: {eligible_count}")
    print(f"  Conditional: {conditional_count}")
    print(f"  Ineligible: {ineligible_count}")
    
    if persist:
        print(f"\n  Results persisted to database.")
        print(f"  Engine runner should query: meta.v_certified_runs")
    
    return {
        'geometry_windows': geometry_results,
        'eligibility': eligibilities,
    }


def main():
    parser = argparse.ArgumentParser(description="PRISM Phase 1 Pipeline")
    parser.add_argument("--indicators", "-i", type=str, default=None,
                       help="Comma-separated indicator list")
    parser.add_argument("--all", action="store_true",
                       help="Scan all indicators")
    parser.add_argument("--min-rows", type=int, default=500,
                       help="Minimum rows required")
    parser.add_argument("--persist", action="store_true",
                       help="Persist to database")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Verbose output")
    args = parser.parse_args()
    
    if args.indicators:
        indicators = [x.strip() for x in args.indicators.split(",")]
    elif args.all:
        indicators = None  # Will be determined from database
    else:
        print("Specify --indicators or --all")
        return
    
    run_phase1(
        indicators=indicators,
        min_rows=args.min_rows,
        persist=args.persist,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
