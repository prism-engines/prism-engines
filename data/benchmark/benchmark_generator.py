"""
PRISM Benchmark Data Generator
==============================

Creates CSV files with KNOWN structure for validating PRISM.

Run this, then run PRISM on each file. 
If PRISM finds what we planted, the math works.

Usage:
    exec(open('benchmark_generator.py').read())
    # Creates 6 CSV files in current directory (or specify path)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

# Output directory - change if needed
OUTPUT_DIR = '.'  # Current directory, or set to your data/raw path

def set_output_dir(path):
    """Set where to save the benchmark files."""
    global OUTPUT_DIR
    OUTPUT_DIR = path
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")


# =============================================================================
# BENCHMARK 1: Clear Leader
# =============================================================================
def create_clear_leader():
    """
    GROUND TRUTH: Column 'LEADER' drives everything else.
    
    What PRISM should find:
    - LEADER ranked #1 by most lenses
    - High Granger causality FROM leader TO others
    - High transfer entropy FROM leader
    """
    np.random.seed(42)
    n = 1000
    dates = pd.date_range('2020-01-01', periods=n, freq='D')
    
    # The leader - random walk with momentum
    leader = np.cumsum(np.random.randn(n) * 0.02) + 100
    
    # Followers - leader + lag + noise
    follower1 = np.roll(leader, 3) + np.random.randn(n) * 0.5  # 3-day lag
    follower2 = np.roll(leader, 5) + np.random.randn(n) * 0.8  # 5-day lag
    follower3 = np.roll(leader, 7) + np.random.randn(n) * 1.0  # 7-day lag
    
    # Fix the roll artifacts at the beginning
    follower1[:10] = leader[:10] + np.random.randn(10) * 0.5
    follower2[:10] = leader[:10] + np.random.randn(10) * 0.8
    follower3[:10] = leader[:10] + np.random.randn(10) * 1.0
    
    # Noise columns - independent
    noise1 = np.cumsum(np.random.randn(n) * 0.01) + 50
    noise2 = np.cumsum(np.random.randn(n) * 0.01) + 50
    
    df = pd.DataFrame({
        'A': leader,      # THE LEADER
        'B': follower1,   # Follows A, lag 3
        'C': follower2,   # Follows A, lag 5
        'D': follower3,   # Follows A, lag 7
        'E': noise1,      # Noise
        'F': noise2,      # Noise
    }, index=dates)
    
    path = os.path.join(OUTPUT_DIR, 'benchmark_01_clear_leader.csv')
    df.to_csv(path)
    print(f"✓ Created: {path}")
    print("  TRUTH: A is the leader. B,C,D follow with lags 3,5,7. E,F are noise.")
    return df


# =============================================================================
# BENCHMARK 2: Two Regimes
# =============================================================================
def create_two_regimes():
    """
    GROUND TRUTH: Clear regime change at day 500.
    
    Regime 1 (days 1-500): Low volatility, positive drift
    Regime 2 (days 501-1000): High volatility, negative drift
    
    What PRISM should find:
    - Regime lens detects split around day 500
    - Different correlation structure in each regime
    - Column behavior differs across regimes
    """
    np.random.seed(123)
    n = 1000
    regime_change = 500
    dates = pd.date_range('2020-01-01', periods=n, freq='D')
    
    data = {}
    for col in ['A', 'B', 'C', 'D', 'E']:
        series = np.zeros(n)
        
        # Regime 1: calm, trending up
        drift1 = np.random.uniform(0.001, 0.003)
        vol1 = np.random.uniform(0.005, 0.01)
        series[:regime_change] = np.cumsum(np.random.randn(regime_change) * vol1 + drift1) + 100
        
        # Regime 2: volatile, trending down
        drift2 = np.random.uniform(-0.003, -0.001)
        vol2 = np.random.uniform(0.02, 0.04)  # 3-4x higher vol
        series[regime_change:] = series[regime_change-1] + np.cumsum(
            np.random.randn(n - regime_change) * vol2 + drift2
        )
        
        data[col] = series
    
    # Add one column that DOESN'T change regime (control)
    data['F'] = np.cumsum(np.random.randn(n) * 0.01) + 100  # Constant behavior
    
    df = pd.DataFrame(data, index=dates)
    
    path = os.path.join(OUTPUT_DIR, 'benchmark_02_two_regimes.csv')
    df.to_csv(path)
    print(f"✓ Created: {path}")
    print("  TRUTH: Regime change at day 500. A-E change behavior. F does not.")
    return df


# =============================================================================
# BENCHMARK 3: Clustered Groups
# =============================================================================
def create_clusters():
    """
    GROUND TRUTH: Three distinct clusters of correlated assets.
    
    Cluster 1: A, B, C (correlated ~0.9)
    Cluster 2: D, E, F (correlated ~0.9)
    Cluster 3: G, H (correlated ~0.9)
    Between clusters: low correlation (~0.1)
    
    What PRISM should find:
    - Clustering lens finds 3 groups
    - Network lens shows 3 communities
    - Within-cluster correlation >> between-cluster
    """
    np.random.seed(456)
    n = 1000
    dates = pd.date_range('2020-01-01', periods=n, freq='D')
    
    # Cluster 1 base signal
    base1 = np.cumsum(np.random.randn(n) * 0.02) + 100
    A = base1 + np.random.randn(n) * 0.3
    B = base1 + np.random.randn(n) * 0.3
    C = base1 + np.random.randn(n) * 0.3
    
    # Cluster 2 base signal (independent)
    base2 = np.cumsum(np.random.randn(n) * 0.02) + 50
    D = base2 + np.random.randn(n) * 0.3
    E = base2 + np.random.randn(n) * 0.3
    F = base2 + np.random.randn(n) * 0.3
    
    # Cluster 3 base signal (independent)
    base3 = np.cumsum(np.random.randn(n) * 0.02) + 75
    G = base3 + np.random.randn(n) * 0.3
    H = base3 + np.random.randn(n) * 0.3
    
    df = pd.DataFrame({
        'A': A, 'B': B, 'C': C,  # Cluster 1
        'D': D, 'E': E, 'F': F,  # Cluster 2
        'G': G, 'H': H,          # Cluster 3
    }, index=dates)
    
    path = os.path.join(OUTPUT_DIR, 'benchmark_03_clusters.csv')
    df.to_csv(path)
    print(f"✓ Created: {path}")
    print("  TRUTH: 3 clusters. {A,B,C}, {D,E,F}, {G,H}. High within, low between correlation.")
    return df


# =============================================================================
# BENCHMARK 4: Hidden Periodicity
# =============================================================================
def create_periodic():
    """
    GROUND TRUTH: Different periodicities in different columns.
    
    A: 20-day cycle
    B: 50-day cycle
    C: 100-day cycle
    D: No cycle (random walk)
    E: Mixed (20 + 50 day cycles)
    
    What PRISM should find:
    - Wavelet lens detects different dominant frequencies
    - Decomposition lens shows different seasonal strengths
    - A and E should be linked (share 20-day cycle)
    """
    np.random.seed(789)
    n = 1000
    dates = pd.date_range('2020-01-01', periods=n, freq='D')
    t = np.arange(n)
    
    # Pure cycles + trend + noise
    A = 10 * np.sin(2 * np.pi * t / 20) + t * 0.01 + np.random.randn(n) * 1
    B = 15 * np.sin(2 * np.pi * t / 50) + t * 0.01 + np.random.randn(n) * 1
    C = 20 * np.sin(2 * np.pi * t / 100) + t * 0.01 + np.random.randn(n) * 1
    D = np.cumsum(np.random.randn(n) * 0.5) + 100  # No cycle
    E = 8 * np.sin(2 * np.pi * t / 20) + 8 * np.sin(2 * np.pi * t / 50) + np.random.randn(n) * 1
    
    df = pd.DataFrame({
        'A': A + 100,  # Offset to positive
        'B': B + 100,
        'C': C + 100,
        'D': D,
        'E': E + 100,
    }, index=dates)
    
    path = os.path.join(OUTPUT_DIR, 'benchmark_04_periodic.csv')
    df.to_csv(path)
    print(f"✓ Created: {path}")
    print("  TRUTH: A=20day, B=50day, C=100day cycles. D=none. E=20+50day mixed.")
    return df


# =============================================================================
# BENCHMARK 5: Anomaly Injection
# =============================================================================
def create_anomalies():
    """
    GROUND TRUTH: Specific columns have injected anomalies.
    
    A: Clean (no anomalies)
    B: 5 point anomalies (spikes)
    C: 1 collective anomaly (days 400-420 go crazy)
    D: Clean
    E: 10 point anomalies
    
    What PRISM should find:
    - Anomaly lens ranks B, C, E highest
    - A, D should have lowest anomaly scores
    """
    np.random.seed(101)
    n = 1000
    dates = pd.date_range('2020-01-01', periods=n, freq='D')
    
    # Base random walks
    A = np.cumsum(np.random.randn(n) * 0.01) + 100  # Clean
    B = np.cumsum(np.random.randn(n) * 0.01) + 100  # Will add spikes
    C = np.cumsum(np.random.randn(n) * 0.01) + 100  # Will add collective anomaly
    D = np.cumsum(np.random.randn(n) * 0.01) + 100  # Clean
    E = np.cumsum(np.random.randn(n) * 0.01) + 100  # Will add spikes
    
    # Inject point anomalies in B (5 spikes)
    spike_idx_B = [100, 300, 500, 700, 900]
    for idx in spike_idx_B:
        B[idx] += np.random.choice([-1, 1]) * 5  # Big spike
    
    # Inject collective anomaly in C (days 400-420)
    C[400:420] += np.cumsum(np.random.randn(20) * 0.5)  # Sudden drift
    
    # Inject point anomalies in E (10 spikes)
    spike_idx_E = np.random.choice(range(50, 950), 10, replace=False)
    for idx in spike_idx_E:
        E[idx] += np.random.choice([-1, 1]) * 4
    
    df = pd.DataFrame({
        'A': A,
        'B': B,
        'C': C,
        'D': D,
        'E': E,
    }, index=dates)
    
    path = os.path.join(OUTPUT_DIR, 'benchmark_05_anomalies.csv')
    df.to_csv(path)
    print(f"✓ Created: {path}")
    print("  TRUTH: A,D clean. B has 5 spikes. C has collective anomaly days 400-420. E has 10 spikes.")
    return df


# =============================================================================
# BENCHMARK 6: Pure Noise (Control)
# =============================================================================
def create_pure_noise():
    """
    GROUND TRUTH: All columns are independent random walks.
    
    NO structure. NO leaders. NO regimes. NO clusters.
    
    What PRISM should find:
    - No consensus (high disagreement between lenses)
    - No clear leader
    - No regime changes
    - Low cross-correlation
    
    If PRISM finds strong patterns here, something is wrong.
    """
    np.random.seed(999)
    n = 1000
    dates = pd.date_range('2020-01-01', periods=n, freq='D')
    
    df = pd.DataFrame({
        'A': np.cumsum(np.random.randn(n) * 0.01) + 100,
        'B': np.cumsum(np.random.randn(n) * 0.01) + 100,
        'C': np.cumsum(np.random.randn(n) * 0.01) + 100,
        'D': np.cumsum(np.random.randn(n) * 0.01) + 100,
        'E': np.cumsum(np.random.randn(n) * 0.01) + 100,
        'F': np.cumsum(np.random.randn(n) * 0.01) + 100,
    }, index=dates)
    
    path = os.path.join(OUTPUT_DIR, 'benchmark_06_pure_noise.csv')
    df.to_csv(path)
    print(f"✓ Created: {path}")
    print("  TRUTH: Pure noise. No structure. PRISM should find NOTHING.")
    return df


# =============================================================================
# GENERATE ALL
# =============================================================================
def generate_all(output_dir=None):
    """Generate all benchmark datasets."""
    if output_dir:
        set_output_dir(output_dir)
    
    print("="*60)
    print("PRISM BENCHMARK DATA GENERATOR")
    print("="*60)
    print()
    
    create_clear_leader()
    print()
    create_two_regimes()
    print()
    create_clusters()
    print()
    create_periodic()
    print()
    create_anomalies()
    print()
    create_pure_noise()
    
    print()
    print("="*60)
    print("✓ All 6 benchmark files created!")
    print()
    print("VALIDATION CHECKLIST:")
    print("  □ 01_clear_leader: A should rank #1, Granger A→others")
    print("  □ 02_two_regimes: Regime split at day 500")
    print("  □ 03_clusters: 3 clusters detected")
    print("  □ 04_periodic: Wavelet finds 20/50/100 day cycles")
    print("  □ 05_anomalies: B,C,E rank high on anomaly lens")
    print("  □ 06_pure_noise: NO strong patterns (control)")
    print("="*60)


# =============================================================================
# ANSWER KEY (for after you run PRISM)
# =============================================================================
ANSWER_KEY = """
BENCHMARK ANSWER KEY
====================

01_clear_leader.csv
-------------------
- Column A is the LEADER
- B follows A with 3-day lag
- C follows A with 5-day lag  
- D follows A with 7-day lag
- E, F are independent noise
EXPECTED: A ranks #1 on Granger, Transfer Entropy, Influence

02_two_regimes.csv
------------------
- Regime 1: Days 1-500 (low vol, positive drift)
- Regime 2: Days 501-1000 (high vol, negative drift)
- Column F does NOT change (control)
EXPECTED: Regime lens detects split ~day 500. F differs from others.

03_clusters.csv
---------------
- Cluster 1: A, B, C (ρ ≈ 0.9 within)
- Cluster 2: D, E, F (ρ ≈ 0.9 within)
- Cluster 3: G, H (ρ ≈ 0.9 within)
- Between clusters: ρ ≈ 0.1
EXPECTED: Clustering/Network lens finds 3 groups

04_periodic.csv
---------------
- A: 20-day cycle
- B: 50-day cycle
- C: 100-day cycle
- D: No cycle (random walk)
- E: 20 + 50 day mixed
EXPECTED: Wavelet lens detects frequencies. A correlates with E.

05_anomalies.csv
----------------
- A: Clean
- B: 5 point anomalies (days 100,300,500,700,900)
- C: Collective anomaly (days 400-420)
- D: Clean
- E: 10 point anomalies (random locations)
EXPECTED: Anomaly lens ranks B,C,E high. A,D low.

06_pure_noise.csv
-----------------
- All columns: independent random walks
- NO structure whatsoever
EXPECTED: Low agreement between lenses. No clear rankings.
          If PRISM finds strong patterns, it's overfitting.
"""

def print_answer_key():
    print(ANSWER_KEY)


# Auto-run if executed
if __name__ != '__main__':
    print("Benchmark generator loaded!")
    print()
    print("Usage:")
    print("  generate_all()                    # Creates all 6 files in current dir")
    print("  generate_all('/path/to/data')     # Creates in specific directory")
    print("  print_answer_key()                # Show expected results")
