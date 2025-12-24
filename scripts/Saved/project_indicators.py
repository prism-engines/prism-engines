#!/usr/bin/env python3
"""
PRISM Indicator Projection

Project indicators into behavioral space within geometric states.
See how indicators MOVE when structure changes.

Usage:
    python scripts/project_indicators.py
    python scripts/project_indicators.py --states 4
    python scripts/project_indicators.py --indicator SPY
    python scripts/project_indicators.py -o results/projection.json
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from prism.db.connection import get_db_path, get_connection
from prism.structure import StructureExtractor
from prism.projection import IndicatorProjector


def load_data(db_path: str, indicators: list, start_date: str = None) -> pd.DataFrame:
    """Load indicator data."""
    conn = get_connection(Path(db_path) if db_path else None)
    ind_list = ", ".join(f"'{i}'" for i in indicators)
    
    query = f"""
        SELECT date, indicator_id, value
        FROM clean.indicators
        WHERE indicator_id IN ({ind_list})
    """
    if start_date:
        query += f" AND date >= '{start_date}'"
    
    df = conn.execute(query).fetchdf()
    df_wide = df.pivot(index='date', columns='indicator_id', values='value').dropna()
    return df_wide


def main():
    parser = argparse.ArgumentParser(description="Project indicators into behavioral space")
    parser.add_argument("--db", default=str(get_db_path()))
    parser.add_argument("--indicators", default="SPY,XLU,XLF,XLK,XLE,AGG,GLD,VIXCLS")
    parser.add_argument("--start", default="2005-01-01")
    parser.add_argument("--window", type=int, default=252)
    parser.add_argument("--step", type=int, default=21)
    parser.add_argument("--states", type=int, default=3)
    parser.add_argument("--indicator", "-i", help="Focus on specific indicator")
    parser.add_argument("--output", "-o", type=Path, help="Output JSON file")
    
    args = parser.parse_args()
    
    indicators = [i.strip() for i in args.indicators.split(",")]
    
    print("=" * 70)
    print("PRISM INDICATOR PROJECTION")
    print("=" * 70)
    
    # Load data
    print(f"\n1. Loading data...")
    df = load_data(args.db, indicators, args.start)
    print(f"   {len(df)} rows, {len(df.columns)} indicators")
    
    # Extract structure
    print(f"\n2. Extracting geometry...")
    extractor = StructureExtractor(df)
    extractor.extract_geometry(window=args.window, step=args.step, verbose=False)
    extractor.cluster_geometry(n_states=args.states)
    print(f"   {len(extractor.states)} states identified")
    
    # Project indicators
    print(f"\n3. Projecting indicators into behavioral space...")
    projector = IndicatorProjector(df, extractor)
    behavioral_space = projector.project_all_states(verbose=True)
    
    print(f"\n   Behavioral dimensions: {len(behavioral_space.dimensions)}")
    for dim in behavioral_space.dimensions:
        print(f"     - {dim}")
    
    # Compute movement
    print(f"\n4. Computing indicator movement...")
    movements = projector.compute_movement()
    print(f"   {len(movements)} transitions analyzed")
    
    # Print reports
    if args.indicator:
        # Focus on one indicator
        print(f"\n{'='*70}")
        print(f"INDICATOR: {args.indicator}")
        print(f"{'='*70}")
        
        comparison = projector.compare_indicator_across_states(args.indicator)
        if comparison.empty:
            print(f"No data for {args.indicator}")
        else:
            print(f"\nPosition across states:")
            print(comparison.round(4).to_string())
            
            # Movement for this indicator
            ind_movements = [m for m in movements if m.indicator_id == args.indicator]
            if ind_movements:
                print(f"\nMovement:")
                for m in ind_movements:
                    print(f"  State {m.from_state} → {m.to_state}: distance = {m.distance:.4f}")
                    for dim, change in sorted(m.dimension_changes.items()):
                        print(f"    {dim}: {change:+.4f}")
    else:
        # Full reports
        projector.print_position_report()
        projector.print_movement_report()
    
    # Movement matrix
    print(f"\n{'='*70}")
    print("MOVEMENT MATRIX")
    print("(Distance each indicator moves between states)")
    print(f"{'='*70}\n")
    
    movement_df = projector.get_movement_matrix()
    if not movement_df.empty:
        print(movement_df.round(4).to_string())
    
    # State centroids comparison
    print(f"\n{'='*70}")
    print("STATE CENTROIDS (Average indicator position)")
    print(f"{'='*70}\n")
    
    centroid_records = []
    for state_id, centroid in behavioral_space.centroids.items():
        record = {"state": state_id}
        record.update(centroid)
        centroid_records.append(record)
    
    centroid_df = pd.DataFrame(centroid_records).set_index("state").sort_index()
    print(centroid_df.round(4).to_string())
    
    # Save results
    if args.output:
        results = {
            "meta": {
                "timestamp": datetime.now().isoformat(),
                "indicators": indicators,
                "n_states": args.states,
                "behavioral_dimensions": behavioral_space.dimensions,
            },
            "positions": {},
            "movements": [],
            "centroids": {},
        }
        
        # Positions
        for (ind, state), pos in behavioral_space.positions.items():
            key = f"{ind}__state_{state}"
            results["positions"][key] = {
                "indicator": ind,
                "state": state,
                "n_samples": pos.n_samples,
                "coordinates": {k: round(v, 6) for k, v in pos.coordinates.items()},
            }
        
        # Movements
        for m in movements:
            results["movements"].append({
                "indicator": m.indicator_id,
                "from_state": m.from_state,
                "to_state": m.to_state,
                "distance": round(m.distance, 6),
                "dimension_changes": {k: round(v, 6) for k, v in m.dimension_changes.items()},
            })
        
        # Centroids
        for state_id, centroid in behavioral_space.centroids.items():
            results["centroids"][state_id] = {k: round(v, 6) for k, v in centroid.items()}
        
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output}")
    
    print("\n" + "=" * 70)
    print("Projection complete")
    print("=" * 70)


if __name__ == "__main__":
    main()
