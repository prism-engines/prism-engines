#!/usr/bin/env python3
"""
PRISM Climate Analysis Runner
=============================

Runs PRISM regime detection on climate data.
Detects climate regime shifts using the same lenses as market analysis.

Usage:
    python run_climate_analysis.py

Output:
    ~/prism_data/climate_output/
"""

import pandas as pd
import numpy as np
import sqlite3
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from scipy import stats

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


DB_PATH = DATA_DIR / "climate.db"
PANEL_PATH = DATA_DIR / "climate_panel.csv"
CLIMATE_# OUTPUT_DIR imported from output_config / "climate"


# ============================================================================
# LENS FUNCTIONS
# ============================================================================

def compute_pca_importance(data):
    """PCA-based importance."""
    try:
        scaler = StandardScaler()
        scaled = scaler.fit_transform(data)
        pca = PCA(n_components=min(3, len(data.columns)))
        pca.fit(scaled)
        loadings = np.abs(pca.components_).mean(axis=0)
        return pd.Series(loadings, index=data.columns)
    except:
        return pd.Series(1.0, index=data.columns)


def compute_volatility(data):
    """Volatility (standard deviation of changes)."""
    changes = data.pct_change().dropna()
    return changes.std()


def compute_trend_strength(data):
    """Trend strength via linear regression R²."""
    results = {}
    x = np.arange(len(data)).reshape(-1, 1)
    
    for col in data.columns:
        y = data[col].values
        if len(y) > 10:
            slope, intercept, r, p, se = stats.linregress(x.flatten(), y)
            results[col] = abs(r)  # R² indicates trend strength
        else:
            results[col] = 0
    
    return pd.Series(results)


def compute_anomaly_score(data):
    """Isolation Forest anomaly detection."""
    try:
        iso = IsolationForest(n_estimators=50, random_state=42)
        scores = []
        for col in data.columns:
            iso.fit(data[[col]])
            scores.append(-iso.score_samples(data[[col]]).mean())
        return pd.Series(scores, index=data.columns)
    except:
        return pd.Series(1.0, index=data.columns)


def compute_all_lenses(data):
    """Compute all lens scores."""
    results = {
        'pca': compute_pca_importance(data),
        'volatility': compute_volatility(data),
        'trend': compute_trend_strength(data),
        'anomaly': compute_anomaly_score(data),
    }
    
    rankings = pd.DataFrame(results)
    
    # Rank each (higher = more important)
    for col in rankings.columns:
        rankings[f'{col}_rank'] = rankings[col].rank(ascending=False)
    
    rank_cols = [c for c in rankings.columns if '_rank' in c]
    rankings['consensus_rank'] = rankings[rank_cols].mean(axis=1)
    
    return rankings


# ============================================================================
# REGIME DETECTION
# ============================================================================

def detect_climate_regimes(panel, n_regimes=3):
    """Detect climate regimes using clustering."""
    
    # Normalize data
    scaler = StandardScaler()
    scaled = scaler.fit_transform(panel)
    
    # Cluster
    kmeans = KMeans(n_clusters=n_regimes, random_state=42, n_init=10)
    labels = kmeans.fit_predict(scaled)
    
    # Characterize regimes by temperature anomaly
    if 'temp_anomaly' in panel.columns:
        regime_temps = panel.groupby(labels)['temp_anomaly'].mean()
        regime_map = {i: rank for rank, i in enumerate(regime_temps.argsort())}
        regime_names = {0: 'Cool', 1: 'Normal', 2: 'Warm'}
    else:
        regime_map = {i: i for i in range(n_regimes)}
        regime_names = {i: f'Regime {i}' for i in range(n_regimes)}
    
    regimes = pd.Series(labels, index=panel.index)
    regimes = regimes.map(regime_map).map(regime_names)
    
    return regimes


def compute_regime_stability(panel, window=60):
    """Compute rolling regime stability (correlation of rankings over time)."""
    
    stabilities = []
    dates = []
    
    for i in range(window, len(panel), window // 2):
        window1 = panel.iloc[i-window:i-window//2]
        window2 = panel.iloc[i-window//2:i]
        
        if len(window1) < 10 or len(window2) < 10:
            continue
        
        rank1 = compute_all_lenses(window1)['consensus_rank']
        rank2 = compute_all_lenses(window2)['consensus_rank']
        
        # Spearman correlation of rankings
        corr, _ = stats.spearmanr(rank1, rank2)
        
        stabilities.append(corr)
        dates.append(panel.index[i])
    
    return pd.Series(stabilities, index=dates)


# ============================================================================
# MAIN ANALYSIS
# ============================================================================

def run_analysis():
    """Run full climate analysis."""
    print("=" * 60)
    print("🌍 PRISM CLIMATE REGIME ANALYSIS")
    print("=" * 60)
    
    # Load panel
    if not PANEL_PATH.exists():
        print(f"❌ Panel not found: {PANEL_PATH}")
        print("   Run: python build_climate_panel.py")
        return
    
    panel = pd.read_csv(PANEL_PATH, parse_dates=['date'], index_col='date')
    print(f"   Loaded panel: {panel.shape[0]} rows × {panel.shape[1]} columns")
    print(f"   Date range: {panel.index.min().date()} to {panel.index.max().date()}")
    print(f"   Indicators: {list(panel.columns)}")
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Current indicator importance
    print("\n" + "-" * 40)
    print("📊 INDICATOR IMPORTANCE (Last 10 Years)")
    print("-" * 40)
    
    recent = panel.iloc[-120:]  # Last 10 years (monthly)
    rankings = compute_all_lenses(recent)
    rankings = rankings.sort_values('consensus_rank')
    
    print(rankings[['pca', 'volatility', 'trend', 'anomaly', 'consensus_rank']].round(2).to_string())
    
    rankings.to_csv(OUTPUT_DIR / 'indicator_rankings.csv')
    print(f"\n   ✅ Saved: indicator_rankings.csv")
    
    # 2. Regime detection
    print("\n" + "-" * 40)
    print("🎯 CLIMATE REGIMES")
    print("-" * 40)
    
    regimes = detect_climate_regimes(panel)
    current_regime = regimes.iloc[-1]
    print(f"   Current regime: {current_regime}")
    
    regime_dist = regimes.value_counts(normalize=True)
    for regime, pct in regime_dist.items():
        print(f"   {regime}: {pct:.1%}")
    
    regimes.to_frame('regime').to_csv(OUTPUT_DIR / 'climate_regimes.csv')
    print(f"\n   ✅ Saved: climate_regimes.csv")
    
    # 3. Regime stability over time
    print("\n" + "-" * 40)
    print("📈 REGIME STABILITY OVER TIME")
    print("-" * 40)
    
    stability = compute_regime_stability(panel, window=60)
    
    if len(stability) > 0:
        print(f"   Mean stability: {stability.mean():.2f}")
        print(f"   Min stability: {stability.min():.2f} ({stability.idxmin().date()})")
        
        # Find regime breaks (stability < 0.3)
        breaks = stability[stability < 0.3]
        if len(breaks) > 0:
            print(f"\n   ⚠️  Major regime breaks (stability < 0.3):")
            for date, val in breaks.items():
                print(f"      {date.date()}: {val:.2f}")
        
        stability.to_frame('stability').to_csv(OUTPUT_DIR / 'regime_stability.csv')
        print(f"\n   ✅ Saved: regime_stability.csv")
    
    # 4. Visualizations
    print("\n" + "-" * 40)
    print("📈 CREATING VISUALIZATIONS")
    print("-" * 40)
    
    # Temperature + CO2 over time
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    
    if 'temp_anomaly' in panel.columns:
        axes[0].plot(panel.index, panel['temp_anomaly'], 'r-', alpha=0.7)
        axes[0].axhline(0, color='black', linestyle='--', alpha=0.3)
        axes[0].set_ylabel('Temp Anomaly (°C)')
        axes[0].set_title('Global Temperature Anomaly')
    
    if 'co2' in panel.columns:
        axes[1].plot(panel.index, panel['co2'], 'b-', alpha=0.7)
        axes[1].set_ylabel('CO2 (ppm)')
        axes[1].set_title('Atmospheric CO2')
    
    # Regime timeline
    regime_colors = {'Cool': 'blue', 'Normal': 'gray', 'Warm': 'red'}
    for regime, color in regime_colors.items():
        mask = regimes == regime
        if mask.any():
            axes[2].scatter(regimes.index[mask], [1]*mask.sum(), 
                           c=color, label=regime, alpha=0.5, s=10)
    axes[2].legend()
    axes[2].set_yticks([])
    axes[2].set_title('Climate Regime')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'climate_overview.png', dpi=150)
    plt.close()
    print("   ✅ climate_overview.png")
    
    # Stability over time
    if len(stability) > 0:
        fig, ax = plt.subplots(figsize=(14, 4))
        ax.plot(stability.index, stability.values, 'b-', alpha=0.7)
        ax.axhline(0.3, color='red', linestyle='--', label='Regime break threshold')
        ax.fill_between(stability.index, 0, stability.values, 
                        where=stability.values < 0.3, alpha=0.3, color='red')
        ax.set_ylabel('Stability (rank correlation)')
        ax.set_title('Climate Regime Stability Over Time')
        ax.legend()
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / 'stability_timeline.png', dpi=150)
        plt.close()
        print("   ✅ stability_timeline.png")
    
    # Summary
    print("\n" + "=" * 60)
    print("✅ ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"   Output directory: {OUTPUT_DIR}")
    print(f"   Current regime: {current_regime}")
    
    if 'temp_anomaly' in panel.columns:
        recent_temp = panel['temp_anomaly'].iloc[-12:].mean()
        print(f"   Recent temp anomaly (12mo avg): {recent_temp:.2f}°C")
    
    if 'co2' in panel.columns:
        recent_co2 = panel['co2'].iloc[-1]
        print(f"   Latest CO2: {recent_co2:.1f} ppm")


if __name__ == "__main__":
    run_analysis()
