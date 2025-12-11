#!/usr/bin/env python3
"""
PRISM Methodology Evolution Demo
================================
Shows the three-stage evolution of PRISM's consensus methodology:

Stage 1: RAW - Equal-weight average, no per-lens normalization
         Problem: Volatile indicators (oil, VIX) dominate
         
Stage 2: NORMALIZED - Per-lens data normalization applied
         Problem: Structure lenses (4 of 13) still over-represented
         
Stage 3: WEIGHTED - Geometry-corrected lens weights
         Solution: Each independent perspective gets fair voice

This is a DEMONSTRATION script for presentations.
It uses hardcoded examples from actual PRISM runs to illustrate the methodology.

Usage:
    python methodology_demo.py
"""

def print_stage_header(stage_num: int, title: str, subtitle: str):
    """Print formatted stage header."""
    print("\n" + "=" * 70)
    print(f"STAGE {stage_num}: {title}")
    print(f"         {subtitle}")
    print("=" * 70)


def print_rankings(rankings: list, title: str = "Top 10 Indicators"):
    """Print a ranking list."""
    print(f"\n{title}:")
    print("-" * 40)
    for i, (indicator, score) in enumerate(rankings, 1):
        print(f"  {i:2}. {indicator:<20} {score:.3f}")


def main():
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║           PRISM: METHODOLOGY EVOLUTION DEMONSTRATION                 ║
║                                                                      ║
║  How we went from "oil looks important" to "labor markets lead"      ║
╚══════════════════════════════════════════════════════════════════════╝
""")

    # =========================================================================
    # STAGE 1: RAW (Before per-lens normalization)
    # =========================================================================
    print_stage_header(1, "RAW CONSENSUS", 
                       "Equal-weight average, no per-lens normalization")
    
    print("""
    APPROACH:
    • 13 lenses, each gets equal vote
    • No data preprocessing per lens
    • Simple average of raw scores
    
    PROBLEM:
    • Volatile indicators dominate (oil, VIX have large magnitude swings)
    • PCA sees variance → oil is "important"
    • Anomaly sees outliers → oil is "important"  
    • Granger sees nothing special → oil is average
    • But 10+ lenses vote oil high, Granger is outvoted
""")
    
    # These are representative values from early PRISM runs
    raw_rankings = [
        ("uso", 1.086),      # Oil ETF - dominated due to volatility
        ("spy", 0.709),      # S&P 500
        ("init_claims", 0.643),
        ("trade_balance", 0.630),
        ("qqq", 0.621),
        ("job_openings", 0.604),
        ("payrolls", 0.522),
        ("xlv", 0.496),
        ("xlp", 0.460),
        ("fed_balance", 0.460),
    ]
    print_rankings(raw_rankings, "Raw Consensus Rankings")
    
    print("""
    ⚠️  ISSUE: USO (oil) is #1, but is it really the most important 
              economic indicator, or just the most volatile?
""")

    # =========================================================================
    # STAGE 2: NORMALIZED (Per-lens data normalization)
    # =========================================================================
    print_stage_header(2, "NORMALIZED CONSENSUS",
                       "Per-lens data normalization, equal weights")
    
    print("""
    APPROACH:
    • Each lens gets appropriately preprocessed data:
      - Granger: differenced (requires stationarity)
      - PCA: z-scored (scale-sensitive)
      - Anomaly: robust scaling (outlier detection)
      - Wavelet: raw (needs actual levels for cycles)
    • Still equal weight per lens
    
    IMPROVEMENT:
    • Lenses now analyze data in their native format
    • Reduces false signal from preprocessing mismatch
    
    REMAINING PROBLEM:
    • 4 structure lenses (clustering, network, pca, regime) 
      correlate r > 0.8 with each other
    • Structure perspective is counted 4x
    • Causality (Granger) is counted 1x
""")
    
    normalized_rankings = [
        ("xlv", 0.657),      # Healthcare - structure lenses agree
        ("xli", 0.611),      # Industrials
        ("xlf", 0.595),      # Financials
        ("xlu", 0.590),      # Utilities
        ("xlb", 0.574),      # Materials
        ("fed_balance", 0.567),
        ("xlk", 0.566),
        ("qqq", 0.566),
        ("spy", 0.565),
        ("xlp", 0.563),
    ]
    print_rankings(normalized_rankings, "Normalized Consensus Rankings")
    
    print("""
    ⚠️  ISSUE: Sector ETFs dominate. Why? Because 4 structure lenses 
              all see the same correlation patterns → 4x voting power.
""")

    # =========================================================================
    # LENS GEOMETRY ANALYSIS
    # =========================================================================
    print("\n" + "=" * 70)
    print("LENS GEOMETRY ANALYSIS")
    print("         Discovering the redundancy structure")
    print("=" * 70)
    
    print("""
    We computed Spearman correlations between all lens rankings:
    
    REDUNDANT PAIRS (r > 0.8):
    ┌─────────────────────────────────────────────┐
    │  clustering ↔ network   : 0.854             │
    │  clustering ↔ regime    : 0.818             │
    │  network    ↔ pca       : 0.831             │
    │  network    ↔ regime    : 0.871             │
    │  pca        ↔ regime    : 0.831             │
    └─────────────────────────────────────────────┘
    
    ORTHOGONAL PAIRS (r < 0.2):
    ┌─────────────────────────────────────────────┐
    │  granger    ↔ everything : ~0.0             │
    │  magnitude  ↔ most       : ~0.0             │
    │  anomaly    ↔ structure  : negative!        │
    └─────────────────────────────────────────────┘
    
    CLUSTERS DISCOVERED:
    • Cluster 0: anomaly, decomposition, network, regime
    • Cluster 1: dmd, influence, magnitude, pca, transfer_entropy, wavelet  
    • Cluster 2: granger (ALONE - orthogonal to all!)
    • Cluster 3: clustering, mutual_info
    
    KEY INSIGHT: 
    13 lenses collapse to ~4 independent perspectives.
    Granger (predictive causality) is the ONLY truly independent lens.
""")

    # =========================================================================
    # STAGE 3: WEIGHTED (Geometry-corrected)
    # =========================================================================
    print_stage_header(3, "WEIGHTED CONSENSUS",
                       "Geometry-corrected lens weights")
    
    print("""
    APPROACH:
    • Weight each lens by its independence
    • Granger is alone in its cluster → 13% weight
    • PCA shares cluster with 5 others → 6% weight
    • Each PERSPECTIVE gets equal voice, not each METHOD
    
    LENS WEIGHTS (combined):
    ┌────────────────────────────────────────┐
    │  granger          1.714  (13.2%)  ██████ │
    │  mutual_info      1.261  ( 9.7%)  ████   │
    │  anomaly          1.132  ( 8.7%)  ████   │
    │  clustering       1.098  ( 8.4%)  ████   │
    │  decomposition    1.070  ( 8.2%)  ████   │
    │  magnitude        0.943  ( 7.3%)  ███    │
    │  ...                                     │
    │  pca              0.743  ( 5.7%)  ██     │
    └────────────────────────────────────────┘
""")
    
    weighted_rankings = [
        ("fed_balance", 0.661),  # Fed balance sheet - macro driver
        ("spy", 0.609),
        ("job_openings", 0.591),  # Labor market - Granger sees causality
        ("payrolls", 0.588),      # Labor market
        ("qqq", 0.569),
        ("init_claims", 0.566),   # Labor market
        ("retail_sales", 0.564),
        ("xlv", 0.562),
        ("trade_balance", 0.556),
        ("iwm", 0.556),
    ]
    print_rankings(weighted_rankings, "Weighted Consensus Rankings")
    
    print("""
    ✅ RESULT: Labor market indicators (job_openings, payrolls, init_claims)
              rise in importance. These have PREDICTIVE CAUSALITY that
              Granger uniquely detects.
""")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY: THE EVOLUTION")
    print("=" * 70)
    
    print("""
    ┌────────────────────────────────────────────────────────────────────┐
    │  Stage      │ Top Indicator │ Problem / Solution                   │
    ├─────────────┼───────────────┼──────────────────────────────────────┤
    │  1. RAW     │ USO (oil)     │ Volatility ≠ importance              │
    │  2. NORMAL  │ XLV (health)  │ Structure over-counted 4x            │
    │  3. WEIGHT  │ fed_balance   │ Each perspective gets fair voice     │
    └─────────────┴───────────────┴──────────────────────────────────────┘
    
    KEY TAKEAWAY:
    
    The "correct" answer isn't about which indicator wins.
    It's about having a PRINCIPLED methodology:
    
    1. Normalize data per-lens requirements (stationarity, scaling)
    2. Measure lens independence empirically (correlation matrix)
    3. Weight by unique information contribution
    
    This is reproducible, defensible, and mathematically grounded.
""")

    print("\n" + "=" * 70)
    print("For the math professor: lens_geometry.py computes all of this")
    print("from the actual data. Weights update dynamically per analysis run.")
    print("=" * 70 + "\n")


if __name__ == '__main__':
    main()
