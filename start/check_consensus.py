#!/usr/bin/env python3
"""
Quick diagnostic: What does the consensus distribution look like?
"""
import pandas as pd
import numpy as np
from pathlib import Path

# === PRISM PATH CONFIG ===
import os
import sys
# Double-click support: ensure correct directory and path
if __name__ == "__main__":
    _script_dir = Path(__file__).parent.parent
    os.chdir(_script_dir)
    if str(_script_dir) not in sys.path:
        sys.path.insert(0, str(_script_dir))
from output_config import OUTPUT_DIR, DATA_DIR
# === END PATH CONFIG ===


output_dir = OUTPUT_DIR / "full_40y_analysis"

# Load the normalized scores
normalized = pd.read_csv(output_dir / "lens_normalized_40y.csv", index_col=0, parse_dates=True)

# Compute consensus at different thresholds
lens_cols = [c for c in normalized.columns if c not in ['date', 'window_size']]

print("=" * 60)
print("CONSENSUS DISTRIBUTION ANALYSIS")
print("=" * 60)

# Compute consensus (fraction of lenses > 0.6)
consensus = (normalized[lens_cols] > 0.6).sum(axis=1) / len(lens_cols)

print(f"\nConsensus Statistics:")
print(f"  Min:    {consensus.min():.3f}")
print(f"  25%:    {consensus.quantile(0.25):.3f}")
print(f"  50%:    {consensus.quantile(0.50):.3f}")
print(f"  75%:    {consensus.quantile(0.75):.3f}")
print(f"  90%:    {consensus.quantile(0.90):.3f}")
print(f"  95%:    {consensus.quantile(0.95):.3f}")
print(f"  99%:    {consensus.quantile(0.99):.3f}")
print(f"  Max:    {consensus.max():.3f}")

print(f"\n\nEvents detected at different thresholds:")
for thresh in [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]:
    n_events = (consensus > thresh).sum()
    pct = 100 * n_events / len(consensus)
    print(f"  Threshold {thresh:.0%}: {n_events:4} windows ({pct:.1f}%)")

print(f"\n\nTop 20 highest consensus dates:")
top_dates = consensus.nlargest(20)
for date, val in top_dates.items():
    print(f"  {date}: {val:.1%}")

# Also check what the raw (unnormalized) scores look like
print(f"\n\n" + "=" * 60)
print("LENS SCORE DISTRIBUTIONS (normalized)")
print("=" * 60)
for col in lens_cols:
    print(f"  {col:15}: min={normalized[col].min():.3f}, "
          f"median={normalized[col].median():.3f}, "
          f"max={normalized[col].max():.3f}, "
          f">0.6: {(normalized[col]>0.6).mean():.1%}")
