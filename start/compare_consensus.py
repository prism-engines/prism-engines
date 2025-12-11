#!/usr/bin/env python3
"""
PRISM Consensus Comparison
==========================
Shows the evolution of consensus rankings through three stages:

1. UNWEIGHTED: Simple average across all lenses (structure lenses over-represented)
2. WEIGHTED: Geometry-corrected weights (balanced perspectives)

This demonstrates why lens geometry analysis matters:
- 4 structure lenses (clustering, network, pca, regime) correlate r > 0.8
- Unweighted consensus counts structure 4x
- Weighted consensus balances independent perspectives

Usage:
    python compare_consensus.py              # Compare latest run
    python compare_consensus.py --run 2      # Compare specific run
    python compare_consensus.py --export     # Export to CSV for presentation
"""

import sys
import argparse
from pathlib import Path
from typing import Dict, Optional
import pandas as pd
import numpy as np

# Setup paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name == 'start' else SCRIPT_DIR
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def get_latest_run_id() -> Optional[int]:
    """Get most recent analysis run ID."""
    from data.duckdb_connector import get_connection
    conn = get_connection()
    row = conn.execute(
        "SELECT id as run_id FROM analysis_runs ORDER BY run_time DESC LIMIT 1"
    ).fetchone()
    conn.close()
    return row[0] if row else None


def load_lens_rankings(run_id: int) -> pd.DataFrame:
    """Load per-lens rankings as wide matrix."""
    from data.duckdb_connector import get_connection
    conn = get_connection()
    
    df = pd.read_sql(
        """
        SELECT indicator, lens_name, score
        FROM indicator_rankings
        WHERE run_id = ?
        """,
        conn,
        params=(run_id,)
    )
    conn.close()
    
    if df.empty:
        return pd.DataFrame()
    
    # Pivot to wide format
    matrix = df.pivot(index='indicator', columns='lens_name', values='score')
    return matrix


def load_weights(run_id: int, method: str = 'combined') -> Dict[str, float]:
    """Load lens weights from database, with fallback to defaults."""
    import json
    from data.duckdb_connector import get_connection
    
    try:
        conn = get_connection()
        
        # Check if table exists
        table_exists = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='lens_weights'"
        ).fetchone()
        
        if not table_exists:
            conn.close()
            print("   ⚠️  lens_weights table not found - using defaults")
            return get_default_weights()
        
        row = conn.execute(
            "SELECT weights FROM lens_weights WHERE run_id = ? AND method = ?",
            (run_id, method)
        ).fetchone()
        conn.close()
        
        if row:
            weights = json.loads(row[0])
            # Normalize
            total = sum(weights.values())
            return {k: v/total for k, v in weights.items()}
        
        # Try any run with this method
        conn = get_connection()
        row = conn.execute(
            "SELECT weights FROM lens_weights WHERE method = ? ORDER BY id DESC LIMIT 1",
            (method,)
        ).fetchone()
        conn.close()
        
        if row:
            print(f"   ⚠️  No weights for run {run_id}, using latest {method} weights")
            weights = json.loads(row[0])
            total = sum(weights.values())
            return {k: v/total for k, v in weights.items()}
        
    except Exception as e:
        print(f"   ⚠️  Could not load weights: {e}")
    
    print("   Using default weights from geometry analysis")
    return get_default_weights()


def get_default_weights() -> Dict[str, float]:
    """
    Fallback weights based on lens geometry analysis.
    From lens_geometry.py combined weights on 2025-12-10.
    """
    weights = {
        'granger': 1.714,
        'mutual_info': 1.261,
        'anomaly': 1.132,
        'clustering': 1.098,
        'decomposition': 1.070,
        'magnitude': 0.943,
        'dmd': 0.892,
        'network': 0.873,
        'influence': 0.849,
        'regime': 0.824,
        'transfer_entropy': 0.819,
        'wavelet': 0.782,
        'pca': 0.743,
    }
    total = sum(weights.values())
    return {k: v/total for k, v in weights.items()}


def compute_unweighted_consensus(matrix: pd.DataFrame) -> pd.Series:
    """Simple average across all lenses."""
    # Normalize each lens to [0,1]
    normalized = matrix.copy()
    for col in normalized.columns:
        s = normalized[col]
        if s.notna().any() and s.max() != s.min():
            normalized[col] = (s - s.min()) / (s.max() - s.min())
    
    return normalized.mean(axis=1).sort_values(ascending=False)


def compute_weighted_consensus(matrix: pd.DataFrame, weights: Dict[str, float]) -> pd.Series:
    """Weighted average using geometry-derived weights."""
    # Normalize each lens to [0,1]
    normalized = matrix.copy()
    for col in normalized.columns:
        s = normalized[col]
        if s.notna().any() and s.max() != s.min():
            normalized[col] = (s - s.min()) / (s.max() - s.min())
    
    # Weighted sum
    weighted_sum = pd.Series(0.0, index=normalized.index)
    weight_sum = pd.Series(0.0, index=normalized.index)
    
    for lens in normalized.columns:
        if lens in weights:
            mask = normalized[lens].notna()
            weighted_sum[mask] += normalized.loc[mask, lens] * weights[lens]
            weight_sum[mask] += weights[lens]
    
    weight_sum = weight_sum.replace(0, np.nan)
    return (weighted_sum / weight_sum).sort_values(ascending=False)


def print_comparison(unweighted: pd.Series, weighted: pd.Series, 
                    weights: Dict[str, float], top_n: int = 20):
    """Print side-by-side comparison."""
    
    print("\n" + "=" * 80)
    print("📊 PRISM CONSENSUS COMPARISON")
    print("=" * 80)
    
    # Lens weights summary
    print("\n⚖️  LENS WEIGHTS (combined geometry)")
    print("-" * 40)
    sorted_weights = sorted(weights.items(), key=lambda x: x[1], reverse=True)
    high_weight = [f"{k} ({v:.0%})" for k, v in sorted_weights[:3]]
    low_weight = [f"{k} ({v:.0%})" for k, v in sorted_weights[-3:]]
    print(f"   Highest: {', '.join(high_weight)}")
    print(f"   Lowest:  {', '.join(low_weight)}")
    
    # The story
    print("\n📖 THE STORY")
    print("-" * 40)
    print("   UNWEIGHTED: 4 structure lenses (clustering, network, pca, regime)")
    print("               correlate r > 0.8 → structure is counted 4x")
    print("   WEIGHTED:   Each independent perspective gets equal voice")
    print("               Granger (causality) gets 13%, PCA (structure) gets 6%")
    
    # Side by side rankings
    print("\n" + "=" * 80)
    print(f"{'Rank':<6} {'UNWEIGHTED':<20} {'Score':<8} {'WEIGHTED':<20} {'Score':<8} {'Δ':<6}")
    print("-" * 80)
    
    for i in range(top_n):
        # Unweighted
        uw_indicator = unweighted.index[i]
        uw_score = unweighted.iloc[i]
        
        # Weighted  
        w_indicator = weighted.index[i]
        w_score = weighted.iloc[i]
        
        # Change for unweighted indicator
        if uw_indicator in weighted.index:
            new_rank = list(weighted.index).index(uw_indicator) + 1
            change = (i + 1) - new_rank
            if change > 0:
                change_str = f"↑{change}"
            elif change < 0:
                change_str = f"↓{abs(change)}"
            else:
                change_str = "="
        else:
            change_str = "?"
        
        # Highlight big movers
        marker = ""
        if uw_indicator != w_indicator:
            marker = "←"
        
        print(f"{i+1:<6} {uw_indicator:<20} {uw_score:<8.3f} {w_indicator:<20} {w_score:<8.3f} {change_str:<6} {marker}")
    
    # Key insights
    print("\n" + "=" * 80)
    print("🔑 KEY INSIGHTS")
    print("-" * 80)
    
    # Find biggest risers and fallers
    rank_changes = {}
    for i, indicator in enumerate(unweighted.index[:30]):
        if indicator in weighted.index:
            old_rank = i + 1
            new_rank = list(weighted.index).index(indicator) + 1
            rank_changes[indicator] = old_rank - new_rank
    
    risers = sorted(rank_changes.items(), key=lambda x: x[1], reverse=True)[:5]
    fallers = sorted(rank_changes.items(), key=lambda x: x[1])[:5]
    
    print("\n   📈 BIGGEST RISERS (favored by independent lenses):")
    for indicator, change in risers:
        if change > 0:
            print(f"      {indicator}: +{change} ranks")
    
    print("\n   📉 BIGGEST FALLERS (over-counted by structure):")
    for indicator, change in fallers:
        if change < 0:
            print(f"      {indicator}: {change} ranks")


def export_comparison(unweighted: pd.Series, weighted: pd.Series, 
                     output_path: Path):
    """Export comparison to CSV."""
    df = pd.DataFrame({
        'unweighted_score': unweighted,
        'unweighted_rank': range(1, len(unweighted) + 1),
    })
    
    # Add weighted info
    weighted_df = pd.DataFrame({
        'weighted_score': weighted,
        'weighted_rank': range(1, len(weighted) + 1),
    })
    
    df = df.join(weighted_df, how='outer')
    df['rank_change'] = df['unweighted_rank'] - df['weighted_rank']
    df = df.sort_values('weighted_rank')
    
    df.to_csv(output_path)
    print(f"\n📁 Exported to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='PRISM Consensus Comparison')
    parser.add_argument('--run', '-r', type=int, help='Analysis run ID')
    parser.add_argument('--top', '-n', type=int, default=20, help='Top N to show')
    parser.add_argument('--export', '-e', action='store_true', help='Export to CSV')
    
    args = parser.parse_args()
    
    # Get run ID
    run_id = args.run or get_latest_run_id()
    if not run_id:
        print("No analysis runs found. Run analyze.py first.")
        return 1
    
    print(f"Loading run {run_id}...")
    
    # Load data
    matrix = load_lens_rankings(run_id)
    if matrix.empty:
        print(f"No rankings found for run {run_id}")
        return 1
    
    weights = load_weights(run_id, 'combined')
    if not weights:
        print("Could not load or compute weights")
        return 1
    
    # Compute consensuses
    unweighted = compute_unweighted_consensus(matrix)
    weighted = compute_weighted_consensus(matrix, weights)
    
    # Display
    print_comparison(unweighted, weighted, weights, args.top)
    
    # Export if requested
    if args.export:
        output_path = SCRIPT_DIR / 'output' / f'consensus_comparison_run{run_id}.csv'
        output_path.parent.mkdir(parents=True, exist_ok=True)
        export_comparison(unweighted, weighted, output_path)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
