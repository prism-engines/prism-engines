#!/usr/bin/env python3
"""
PRISM Geometry → Suitability Pipeline Runner

Runs the complete Phase 1 + Phase 1.5 pipeline:
1. Multi-view geometry analysis (measurement)
2. Math suitability evaluation (control plane)
3. Persist eligibility decisions to database

This is the complete "what math is allowed" pipeline.

Usage:
    python scripts/run_geometry_suitability.py
    python scripts/run_geometry_suitability.py --indicators SPY,GLD,DGS10
    python scripts/run_geometry_suitability.py --policy strict
"""

import sys
import argparse
import uuid
from pathlib import Path
from datetime import datetime

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from prism.db.connection import get_connection


def main():
    parser = argparse.ArgumentParser(description="Geometry + Suitability Pipeline")
    parser.add_argument("--indicators", type=str, help="Comma-separated indicator list")
    parser.add_argument("--min-rows", type=int, default=200, help="Minimum rows required")
    parser.add_argument("--policy", choices=["default", "strict", "permissive"], 
                       default="default", help="Suitability policy to use")
    parser.add_argument("--persist", action="store_true", help="Persist to database")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()
    
    # Import after path setup
    from scripts.agent_multiview_geometry import (
        MultiViewGeometryAgent,
        ViewType,
    )
    from scripts.agent_geometry_signature import GeometrySignatureAgent
    from prism.control.suitability_policy import (
        get_default_policy,
        get_strict_policy,
        get_permissive_policy,
    )
    from prism.agents.math_suitability import (
        MathSuitabilityAgent,
        MultiViewSignature,
        ViewGeometry,
    )
    
    # Select policy
    if args.policy == "strict":
        policy = get_strict_policy()
    elif args.policy == "permissive":
        policy = get_permissive_policy()
    else:
        policy = get_default_policy()
    
    print("=" * 70)
    print("PRISM Geometry → Suitability Pipeline")
    print(f"Policy: {policy.version}")
    print("=" * 70)
    
    # Load data
    conn = get_connection()
    df = conn.execute("""
        SELECT indicator_id, date, value
        FROM clean.indicator_values
        ORDER BY indicator_id, date
    """).fetchdf()
    
    # Determine indicators
    if args.indicators:
        indicator_list = [x.strip() for x in args.indicators.split(",")]
    else:
        counts = df.groupby('indicator_id').size()
        indicator_list = counts[counts >= args.min_rows].index.tolist()
    
    print(f"\nAnalyzing {len(indicator_list)} indicators...")
    
    # Initialize agents
    base_geom = GeometrySignatureAgent(verbose=False)
    mv_agent = MultiViewGeometryAgent(base_agent=base_geom, verbose=args.verbose)
    suit_agent = MathSuitabilityAgent(policy)
    
    # Process indicators
    spread_indicators = {'T10Y2Y', 'T10Y3M', 'T10YFF', 'TEDRATE'}
    
    signatures = []
    
    print("\n" + "-" * 70)
    print("PHASE 1: Multi-View Geometry Analysis")
    print("-" * 70)
    
    for ind_id in indicator_list:
        ind_data = df[df['indicator_id'] == ind_id]['value'].dropna().values
        
        if len(ind_data) < args.min_rows:
            continue
        
        series = ind_data[-1000:] if len(ind_data) > 1000 else ind_data
        
        try:
            # Choose views
            if ind_id in spread_indicators:
                views = [ViewType.LEVEL, ViewType.DEVIATION]
            else:
                views = [ViewType.LEVEL, ViewType.RETURNS, ViewType.VOLATILITY]
            
            result = mv_agent.analyze(series, indicator_id=ind_id, views=views)
            
            # Convert to MultiViewSignature for suitability agent
            view_geometries = {}
            for view_type, view_result in result.views.items():
                view_geometries[view_type.value] = ViewGeometry(
                    view=view_type.value,
                    geometry=view_result.dominant_geometry,
                    confidence=view_result.confidence,
                    is_hybrid=view_result.is_hybrid,
                    scores=view_result.scores,
                )
            
            signature = MultiViewSignature(
                indicator_id=ind_id,
                views=view_geometries,
                consensus_geometry=result.consensus_geometry,
                consensus_confidence=result.consensus_confidence,
                disagreement=result.disagreement_score,
                is_hybrid=any(v.is_hybrid for v in view_geometries.values()),
            )
            signatures.append(signature)
            
            if not args.verbose:
                print(f"  {ind_id:12} → {result.consensus_geometry:24} "
                      f"(conf={result.consensus_confidence:.2f}, disagree={result.disagreement_score:.2f})")
            
        except Exception as e:
            print(f"  {ind_id}: ERROR - {e}")
    
    print(f"\n  Processed: {len(signatures)} indicators")
    
    # Phase 1.5: Suitability evaluation
    print("\n" + "-" * 70)
    print("PHASE 1.5: Math Suitability Evaluation")
    print("-" * 70)
    
    decisions = suit_agent.evaluate_cohort(signatures)
    
    # Summary by status
    status_counts = {}
    for decision in decisions.values():
        status = decision.status.value
        status_counts[status] = status_counts.get(status, 0) + 1
    
    print("\nEligibility Distribution:")
    for status, count in sorted(status_counts.items()):
        pct = 100 * count / len(decisions)
        print(f"  {status:15} {count:3} ({pct:.0f}%)")
    
    # Show examples of each status
    print("\n" + "-" * 70)
    print("EXAMPLE DECISIONS BY STATUS")
    print("-" * 70)
    
    for status in ["eligible", "conditional", "ineligible", "unknown"]:
        examples = [d for d in decisions.values() if d.status.value == status][:3]
        if examples:
            print(f"\n{status.upper()}:")
            for d in examples:
                print(f"  {d.indicator_id}:")
                print(f"    geometry: {d.consensus_geometry} (conf={d.confidence:.2f})")
                print(f"    allowed: {d.allowed_engines[:5]}...")
                if d.conditional_engines:
                    print(f"    conditional: {[c.engine for c in d.conditional_engines]}")
                if d.rationale:
                    print(f"    → {d.rationale[0]}")
    
    # Persist if requested
    if args.persist:
        run_id = f"geom_suit_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        print(f"\n" + "-" * 70)
        print(f"PERSISTING to database (run_id: {run_id})")
        print("-" * 70)
        
        suit_agent.persist_cohort(run_id, decisions, conn)
        print(f"  Persisted {len(decisions)} eligibility records")
    
    conn.close()
    
    # Final summary
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print(f"  Indicators analyzed: {len(signatures)}")
    print(f"  Eligible: {status_counts.get('eligible', 0)}")
    print(f"  Conditional: {status_counts.get('conditional', 0)}")
    print(f"  Ineligible: {status_counts.get('ineligible', 0)}")
    print(f"  Unknown: {status_counts.get('unknown', 0)}")
    print(f"  Policy: {policy.version}")


if __name__ == "__main__":
    main()
