"""
PRISM Benchmark Data Generator v2
=================================

Creates CSV files with KNOWN structure for validating PRISM.

FIXES from v1:
- clear_leader: Leader signal now 10x stronger than follower noise (was inverted)
- two_regimes: Increased regime contrast for clearer detection
- pure_noise: Added structure_detected=False validation target

Run this, then run PRISM on each file.
If PRISM finds what we planted, the math works.

Usage:
    python benchmark_generator.py
    # Or:
    from benchmark_generator import generate_all
    generate_all('/path/to/output')
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

# Output directory - change if needed
OUTPUT_DIR = '.'

def set_output_dir(path):
    """Set where to save the benchmark files."""
    global OUTPUT_DIR
    OUTPUT_DIR = path
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")


# =============================================================================
# BENCHMARK 1: Clear Leader (FIXED)
# =============================================================================
def create_clear_leader():
    """
    GROUND TRUTH: Column 'A' drives everything else.
    
    What PRISM should find:
    - A ranked #1 by most lenses
    - High Granger causality FROM A TO others
    - High transfer entropy FROM A
    - B, C, D highly correlated with A (lagged)
    - E, F uncorrelated with A
    
    FIX: Leader signal is now 10x stronger than follower noise.
         Previously noise swamped the signal.
    """
    np.random.seed(42)
    n = 1000
    dates = pd.date_range('2020-01-01', periods=n, freq='D')
    
    # The leader - STRONG signal (vol = 0.5)
    leader = np.cumsum(np.random.randn(n) * 0.5) + 100
    
    # Followers - leader + lag + SMALL noise (10% of leader vol)
    # The lag creates detectable Granger causality
    follower1 = np.roll(leader, 3) + np.random.randn(n) * 0.05  # 3-day lag
    follower2 = np.roll(leader, 5) + np.random.randn(n) * 0.05  # 5-day lag
    follower3 = np.roll(leader, 7) + np.random.randn(n) * 0.05  # 7-day lag
    
    # Fix the roll artifacts at the beginning
    follower1[:10] = leader[:10] + np.random.randn(10) * 0.05
    follower2[:10] = leader[:10] + np.random.randn(10) * 0.05
    follower3[:10] = leader[:10] + np.random.randn(10) * 0.05
    
    # Noise columns - independent, LOW variance (should rank last)
    noise1 = np.cumsum(np.random.randn(n) * 0.05) + 50
    noise2 = np.cumsum(np.random.randn(n) * 0.05) + 50
    
    df = pd.DataFrame({
        'A': leader,      # THE LEADER - highest variance, drives B,C,D
        'B': follower1,   # Follows A, lag 3
        'C': follower2,   # Follows A, lag 5
        'D': follower3,   # Follows A, lag 7
        'E': noise1,      # Independent noise
        'F': noise2,      # Independent noise
    }, index=dates)
    
    path = os.path.join(OUTPUT_DIR, 'benchmark_01_clear_leader.csv')
    df.to_csv(path)
    print(f"✓ Created: {path}")
    print("  TRUTH: A is the leader (highest variance). B,C,D follow with lags 3,5,7.")
    print("  EXPECTED: A ranks #1 on PCA, Magnitude, Influence, Granger.")
    return df


# =============================================================================
# BENCHMARK 2: Two Regimes (IMPROVED)
# =============================================================================
def create_two_regimes():
    """
    GROUND TRUTH: Clear regime change at day 500.
    
    Regime 1 (days 1-500): Low volatility, positive drift, HIGH correlation
    Regime 2 (days 501-1000): High volatility, negative drift, LOW correlation
    
    What PRISM should find:
    - Regime lens detects split around day 500
    - Correlation structure changes dramatically
    - Volatility spike at regime boundary
    
    FIX: Increased vol contrast (10x) and added correlation structure change.
    """
    np.random.seed(123)
    n = 1000
    regime_change = 500
    dates = pd.date_range('2020-01-01', periods=n, freq='D')
    
    data = {}
    
    # Regime 1: All indicators follow a common factor (high correlation)
    common_factor_r1 = np.cumsum(np.random.randn(regime_change) * 0.01 + 0.002)
    
    # Regime 2: Independent movement (low correlation), high vol
    
    for i, col in enumerate(['A', 'B', 'C', 'D', 'E']):
        series = np.zeros(n)
        
        # Regime 1: Follow common factor + small idiosyncratic noise
        series[:regime_change] = (
            common_factor_r1 + 
            np.random.randn(regime_change) * 0.002 +  # Small noise
            100
        )
        
        # Regime 2: Independent, high volatility, negative drift
        series[regime_change:] = (
            series[regime_change-1] + 
            np.cumsum(np.random.randn(n - regime_change) * 0.05 - 0.005)  # 5x vol, neg drift
        )
        
        data[col] = series
    
    # Control column F: Constant behavior across both regimes
    data['F'] = np.cumsum(np.random.randn(n) * 0.01) + 100
    
    df = pd.DataFrame(data, index=dates)
    
    path = os.path.join(OUTPUT_DIR, 'benchmark_02_two_regimes.csv')
    df.to_csv(path)
    print(f"✓ Created: {path}")
    print("  TRUTH: Regime change at day 500.")
    print("  Regime 1: High correlation, low vol, positive drift")
    print("  Regime 2: Low correlation, high vol, negative drift")
    print("  F is control (no regime change)")
    return df


# =============================================================================
# BENCHMARK 3: Clustered Groups (IMPROVED)
# =============================================================================
def create_clusters():
    """
    GROUND TRUTH: Three distinct clusters of correlated assets.
    
    Cluster 1: A, B, C (correlated ~0.95)
    Cluster 2: D, E, F (correlated ~0.95)
    Cluster 3: G, H (correlated ~0.95)
    Between clusters: correlation ~0.0 (independent)
    
    What PRISM should find:
    - Clustering lens finds exactly 3 groups
    - Network lens shows 3 communities
    - PCA shows 3 dominant components
    
    FIX: Reduced within-cluster noise for cleaner separation.
    """
    np.random.seed(456)
    n = 1000
    dates = pd.date_range('2020-01-01', periods=n, freq='D')
    
    # Cluster 1 base signal
    base1 = np.cumsum(np.random.randn(n) * 0.3) + 100
    A = base1 + np.random.randn(n) * 0.03  # 10% noise
    B = base1 + np.random.randn(n) * 0.03
    C = base1 + np.random.randn(n) * 0.03
    
    # Cluster 2 base signal (completely independent)
    base2 = np.cumsum(np.random.randn(n) * 0.3) + 50
    D = base2 + np.random.randn(n) * 0.03
    E = base2 + np.random.randn(n) * 0.03
    F = base2 + np.random.randn(n) * 0.03
    
    # Cluster 3 base signal (completely independent)
    base3 = np.cumsum(np.random.randn(n) * 0.3) + 75
    G = base3 + np.random.randn(n) * 0.03
    H = base3 + np.random.randn(n) * 0.03
    
    df = pd.DataFrame({
        'A': A, 'B': B, 'C': C,  # Cluster 1
        'D': D, 'E': E, 'F': F,  # Cluster 2
        'G': G, 'H': H,          # Cluster 3
    }, index=dates)
    
    path = os.path.join(OUTPUT_DIR, 'benchmark_03_clusters.csv')
    df.to_csv(path)
    print(f"✓ Created: {path}")
    print("  TRUTH: 3 clusters: {A,B,C}, {D,E,F}, {G,H}")
    print("  Within-cluster correlation ~0.95, between ~0.0")
    return df


# =============================================================================
# BENCHMARK 4: Hidden Periodicity (IMPROVED)
# =============================================================================
def create_periodic():
    """
    GROUND TRUTH: Different periodicities in different columns.
    
    A: Strong 20-day cycle
    B: Strong 50-day cycle
    C: Strong 100-day cycle
    D: No cycle (pure random walk)
    E: Mixed (20 + 50 day cycles)
    
    What PRISM should find:
    - Wavelet lens detects different dominant frequencies
    - A and E linked (share 20-day cycle)
    - B and E linked (share 50-day cycle)
    - D shows no periodicity
    
    FIX: Increased cycle amplitude relative to noise.
    """
    np.random.seed(789)
    n = 1000
    dates = pd.date_range('2020-01-01', periods=n, freq='D')
    t = np.arange(n)
    
    # Strong cycles with minimal noise (SNR ~10:1)
    A = 20 * np.sin(2 * np.pi * t / 20) + np.random.randn(n) * 2 + 100
    B = 20 * np.sin(2 * np.pi * t / 50) + np.random.randn(n) * 2 + 100
    C = 20 * np.sin(2 * np.pi * t / 100) + np.random.randn(n) * 2 + 100
    D = np.cumsum(np.random.randn(n) * 0.5) + 100  # Pure random walk, no cycle
    E = 15 * np.sin(2 * np.pi * t / 20) + 15 * np.sin(2 * np.pi * t / 50) + np.random.randn(n) * 2 + 100
    
    df = pd.DataFrame({
        'A': A,
        'B': B,
        'C': C,
        'D': D,
        'E': E,
    }, index=dates)
    
    path = os.path.join(OUTPUT_DIR, 'benchmark_04_periodic.csv')
    df.to_csv(path)
    print(f"✓ Created: {path}")
    print("  TRUTH: A=20day, B=50day, C=100day cycles. D=none. E=20+50day mixed.")
    print("  EXPECTED: Wavelet lens detects frequencies. D ranks lowest on periodicity.")
    return df


# =============================================================================
# BENCHMARK 5: Anomaly Injection (IMPROVED)
# =============================================================================
def create_anomalies():
    """
    GROUND TRUTH: Specific columns have injected anomalies.
    
    A: Clean (no anomalies) - CONTROL
    B: 5 large point anomalies (10 std spikes)
    C: 1 collective anomaly (days 400-420 regime break)
    D: Clean (no anomalies) - CONTROL
    E: 10 large point anomalies (10 std spikes)
    
    What PRISM should find:
    - Anomaly lens ranks B, C, E highest
    - A, D should have lowest anomaly scores
    - C's collective anomaly is different from B, E's point anomalies
    
    FIX: Increased anomaly magnitude to 10 std (was ~5 std).
    """
    np.random.seed(101)
    n = 1000
    dates = pd.date_range('2020-01-01', periods=n, freq='D')
    
    # Base random walks with consistent volatility
    base_vol = 0.02
    A = np.cumsum(np.random.randn(n) * base_vol) + 100  # Clean
    B = np.cumsum(np.random.randn(n) * base_vol) + 100  # Will add spikes
    C = np.cumsum(np.random.randn(n) * base_vol) + 100  # Will add collective anomaly
    D = np.cumsum(np.random.randn(n) * base_vol) + 100  # Clean
    E = np.cumsum(np.random.randn(n) * base_vol) + 100  # Will add spikes
    
    # Calculate typical std for scaling anomalies
    typical_std = np.std(np.diff(A)) * np.sqrt(n)
    anomaly_size = typical_std * 0.5  # Large but not absurd
    
    # Inject point anomalies in B (5 spikes at known locations)
    spike_idx_B = [100, 300, 500, 700, 900]
    for idx in spike_idx_B:
        B[idx] += np.random.choice([-1, 1]) * anomaly_size
    
    # Inject collective anomaly in C (days 400-420 - sudden level shift and volatility)
    C[400:420] += np.linspace(0, anomaly_size, 20)  # Ramp up
    C[420:] += anomaly_size  # Permanent shift
    
    # Inject point anomalies in E (10 spikes at random locations)
    np.random.seed(102)  # Different seed for E's spike locations
    spike_idx_E = np.sort(np.random.choice(range(50, 950), 10, replace=False))
    for idx in spike_idx_E:
        E[idx] += np.random.choice([-1, 1]) * anomaly_size
    
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
    print("  TRUTH: A,D clean. B has 5 spikes. C has collective anomaly @ 400-420. E has 10 spikes.")
    print(f"  Spike locations B: {spike_idx_B}")
    print(f"  Spike locations E: {list(spike_idx_E)}")
    print("  EXPECTED: Anomaly lens ranks B,C,E high. A,D low.")
    return df


# =============================================================================
# BENCHMARK 6: Pure Noise (Control)
# =============================================================================
def create_pure_noise():
    """
    GROUND TRUTH: All columns are independent random walks.
    
    NO structure. NO leaders. NO regimes. NO clusters. NO periodicity.
    
    What PRISM should find:
    - structure_detected = FALSE
    - Low lens agreement (tau near 0)
    - No clear leader (rankings vary across lenses)
    - PCA: relatively flat explained variance (no dominant component)
    
    If PRISM finds strong patterns here, it's overfitting to noise.
    This is the most important validation benchmark.
    """
    np.random.seed(999)
    n = 1000
    dates = pd.date_range('2020-01-01', periods=n, freq='D')
    
    # All columns: identical, independent random walks
    # Same volatility, same starting point - no distinguishing features
    vol = 0.02
    df = pd.DataFrame({
        'A': np.cumsum(np.random.randn(n) * vol) + 100,
        'B': np.cumsum(np.random.randn(n) * vol) + 100,
        'C': np.cumsum(np.random.randn(n) * vol) + 100,
        'D': np.cumsum(np.random.randn(n) * vol) + 100,
        'E': np.cumsum(np.random.randn(n) * vol) + 100,
        'F': np.cumsum(np.random.randn(n) * vol) + 100,
    }, index=dates)
    
    path = os.path.join(OUTPUT_DIR, 'benchmark_06_pure_noise.csv')
    df.to_csv(path)
    print(f"✓ Created: {path}")
    print("  TRUTH: Pure independent noise. No structure whatsoever.")
    print("  EXPECTED: structure_detected=FALSE, lens_agreement~0, no clear rankings.")
    return df


# =============================================================================
# GENERATE ALL
# =============================================================================
def generate_all(output_dir=None):
    """Generate all benchmark datasets."""
    if output_dir:
        set_output_dir(output_dir)
    
    print("="*60)
    print("PRISM BENCHMARK DATA GENERATOR v2")
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
    print("  □ 01_clear_leader: A ranks #1, variance(A) >> variance(E,F)")
    print("  □ 02_two_regimes: Regime split detected at day 500")
    print("  □ 03_clusters: Exactly 3 clusters detected")
    print("  □ 04_periodic: Wavelet finds 20/50/100 day cycles")
    print("  □ 05_anomalies: B,C,E rank high on anomaly lens; A,D low")
    print("  □ 06_pure_noise: structure_detected=FALSE (critical!)")
    print("="*60)


# =============================================================================
# ANSWER KEY
# =============================================================================
ANSWER_KEY = """
BENCHMARK ANSWER KEY v2
=======================

01_clear_leader.csv
-------------------
- Column A is the LEADER (highest variance by far)
- B follows A with 3-day lag (correlation ~0.99)
- C follows A with 5-day lag (correlation ~0.99)
- D follows A with 7-day lag (correlation ~0.99)
- E, F are independent noise (low variance, uncorrelated)
EXPECTED: 
  - A ranks #1 on PCA, Magnitude, Influence, Granger, Transfer Entropy
  - Lens agreement should be HIGH (tau > 0.5)
  - variance(A) / variance(E) > 50

02_two_regimes.csv
------------------
- Regime 1: Days 1-500 (high correlation, low vol, positive drift)
- Regime 2: Days 501-1000 (low correlation, high vol, negative drift)
- Column F does NOT change (constant behavior = control)
EXPECTED:
  - Regime lens detects split ~day 500
  - F should behave differently from A-E in regime analysis

03_clusters.csv
---------------
- Cluster 1: A, B, C (within-cluster ρ ≈ 0.95)
- Cluster 2: D, E, F (within-cluster ρ ≈ 0.95)
- Cluster 3: G, H (within-cluster ρ ≈ 0.95)
- Between clusters: ρ ≈ 0.0
EXPECTED:
  - Clustering lens finds exactly 3 groups
  - Network lens shows 3 disconnected communities
  - PCA: 3 components explain ~95% variance

04_periodic.csv
---------------
- A: 20-day cycle (strong)
- B: 50-day cycle (strong)
- C: 100-day cycle (strong)
- D: No cycle (pure random walk)
- E: 20 + 50 day mixed cycles
EXPECTED:
  - Wavelet lens detects correct frequencies
  - D ranks lowest on any periodicity measure
  - A correlates with E (shared 20-day component)

05_anomalies.csv
----------------
- A: Clean (control)
- B: 5 point anomalies at days 100, 300, 500, 700, 900
- C: Collective anomaly at days 400-420 (level shift)
- D: Clean (control)
- E: 10 point anomalies (random locations)
EXPECTED:
  - Anomaly lens: B, C, E rank high
  - Anomaly lens: A, D rank low (clean controls)

06_pure_noise.csv
-----------------
- All columns: identical independent random walks
- NO structure whatsoever
EXPECTED:
  - structure_detected = FALSE (CRITICAL!)
  - lens_agreement ≈ 0 (random disagreement)
  - No consistent #1 ranking across lenses
  - If PRISM finds patterns here, it's a FALSE POSITIVE
"""

def print_answer_key():
    print(ANSWER_KEY)


# Auto-run if executed directly
if __name__ == '__main__':
    generate_all()
