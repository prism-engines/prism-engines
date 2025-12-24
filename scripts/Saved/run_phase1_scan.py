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
   Write to: meta.geometry_windows, meta.math_suitability
   Engine runner queries these tables - no execution without record

Usage:
    python scripts/run_phase1_scan.py
    python scripts/run_phase1_scan.py --indicators SPY,QQQ,DGS10
    python scripts/run_phase1_scan.py --all --persist

Output:
    meta.geometry_windows      - Geometry at each (indicator, window)
    meta.math_suitability      - Suitability at each (indicator, window)

NON-NEGOTIABLE:
- Uses CANONICAL_DB_PATH only
- Fails fast if DB cannot be opened
- Never auto-creates tables - hard error if missing
- Never uses :memory:

Cross-validated by: Claude, GPT-4
Date: December 2024
"""

import sys
import argparse
import json
import uuid
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

import duckdb
import numpy as np
import pandas as pd

from prism.db.config import CANONICAL_DB_PATH


# =============================================================================
# DATABASE VALIDATION
# =============================================================================

REQUIRED_TABLES = [
    "meta.geometry_windows",
    "meta.math_suitability",
    "clean.indicator_values",
]


def validate_database() -> duckdb.DuckDBPyConnection:
    """
    Open canonical database with validation.

    Fails fast if:
    - Database file does not exist
    - Required Phase 1 tables are missing

    Returns:
        Validated DuckDB connection
    """
    if not CANONICAL_DB_PATH.exists():
        raise FileNotFoundError(
            f"Canonical database not found: {CANONICAL_DB_PATH}\n"
            "Run fetch pipeline first to create database."
        )

    conn = duckdb.connect(str(CANONICAL_DB_PATH))

    # Verify required tables exist
    missing = []
    for table in REQUIRED_TABLES:
        schema, name = table.split(".")
        result = conn.execute("""
            SELECT COUNT(*) FROM information_schema.tables
            WHERE table_schema = ? AND table_name = ?
        """, [schema, name]).fetchone()

        if result[0] == 0:
            missing.append(table)

    if missing:
        conn.close()
        raise RuntimeError(
            f"Required Phase 1 tables missing: {missing}\n"
            "Run schema migration to create tables:\n"
            "  python -c \"from prism.db.connection import DatabaseConnection; DatabaseConnection().connect()\""
        )

    print(f"[OK] Database validated: {CANONICAL_DB_PATH}")
    return conn


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
        conn: duckdb.DuckDBPyConnection,
    ) -> Dict[str, List[Dict]]:
        """
        Scan all indicators.

        Args:
            indicators: List of indicator IDs to scan
            conn: Validated DuckDB connection (required)

        Returns:
            Dict mapping indicator_id -> list of window results
        """
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

    Uses CANONICAL_DB_PATH only. Fails fast if DB or tables missing.

    Returns:
        Dict with 'geometry_windows' and 'eligibility' results
    """
    from prism.agents.math_suitability_v2 import (
        WindowSuitabilityAgent,
        WindowGeometryResult,
        summarize_eligibility,
    )

    # Validate database before proceeding
    conn = validate_database()

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
    print(f"Database: {CANONICAL_DB_PATH}")
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
        pct = 100 * count / len(geometry_results) if geometry_results else 0
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
        now = datetime.now()

        print("\n" + "-" * 70)
        print(f"STEP 3: Persisting to Database (run_id: {run_id})")
        print("-" * 70)

        # Persist geometry windows
        for ind_id, results in geometry_results.items():
            for r in results:
                # Calculate date range
                window_days = r['window_days']
                end_date = datetime.now().date()
                start_date = end_date - pd.Timedelta(days=window_days)

                conn.execute("""
                    INSERT OR REPLACE INTO meta.geometry_windows
                    (run_id, indicator_id, window_years, start_date, end_date,
                     geometry_type, confidence, disagreement, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, [
                    run_id, r['indicator_id'], r['window_years'],
                    start_date, end_date,
                    r['dominant_geometry'], r['avg_confidence'],
                    r['avg_disagreement'], now
                ])

        print(f"  Wrote {total_windows} geometry_windows records")

        # Persist math suitability
        suitability_count = 0
        for ind_id, windows in eligibilities.items():
            for window_years, elig in windows.items():
                conn.execute("""
                    INSERT OR REPLACE INTO meta.math_suitability
                    (run_id, indicator_id, window_years, status,
                     allowed_engines, conditional_engines, prohibited_engines,
                     rationale, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, [
                    run_id, ind_id, window_years, elig.status.value,
                    json.dumps(list(elig.allowed_engines)),
                    json.dumps(list(elig.conditional_engines)),
                    json.dumps(list(elig.prohibited_engines)),
                    elig.rationale, now
                ])
                suitability_count += 1

        print(f"  Wrote {suitability_count} math_suitability records")

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
        print(f"\n  Results persisted to: {CANONICAL_DB_PATH}")
        print(f"  Query: SELECT * FROM meta.geometry_windows WHERE run_id = '{run_id}'")
        print(f"  Query: SELECT * FROM meta.math_suitability WHERE run_id = '{run_id}'")

    return {
        'geometry_windows': geometry_results,
        'eligibility': eligibilities,
        'run_id': run_id if persist else None,
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
