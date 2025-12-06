#!/usr/bin/env python3
"""
PRISM Multi-Lens Contribution Analysis
=======================================
Runs all 14 mathematical lenses over rolling time windows and tracks
which lenses are "firing" (detecting anomalies/regime shifts) over time.

This is the CORE PRISM insight: when multiple independent mathematical
approaches suddenly AGREE, something significant is happening.

Lenses:
1. PCA - Principal Component Analysis (variance structure)
2. Granger - Granger Causality (lead/lag relationships)  
3. DMD - Dynamic Mode Decomposition (dynamical systems)
4. TDA - Topological Data Analysis (shape of data)
5. Wavelet - Multi-frequency analysis
6. Network - Graph connectivity analysis
7. Clustering - Regime clustering
8. Anomaly - Statistical anomaly detection
9. Influence - Information flow
10. Magnitude - Raw magnitude changes
11. Mutual Info - Mutual information
12. Transfer Entropy - Directed information flow
13. Decomposition - Signal decomposition
14. Regime Switching - Markov regime detection
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import sqlite3
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from scipy import stats
import warnings

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

warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DB_PATH = DATA_DIR / "prism.db"


def load_from_database():
    """Load all data from database."""
    conn = sqlite3.connect(DB_PATH)
    
    market_df = pd.read_sql("SELECT date, ticker, value FROM market_prices ORDER BY date", conn)
    if not market_df.empty:
        market_wide = market_df.pivot(index='date', columns='ticker', values='value')
    else:
        market_wide = pd.DataFrame()
    
    econ_df = pd.read_sql("SELECT date, series_id, value FROM econ_values ORDER BY date", conn)
    if not econ_df.empty:
        econ_wide = econ_df.pivot(index='date', columns='series_id', values='value')
    else:
        econ_wide = pd.DataFrame()
    
    conn.close()
    
    if not market_wide.empty and not econ_wide.empty:
        panel = market_wide.join(econ_wide, how='outer')
    elif not market_wide.empty:
        panel = market_wide
    else:
        panel = econ_wide
    
    panel.index = pd.to_datetime(panel.index)
    return panel.sort_index()


# ============================================================================
# LENS IMPLEMENTATIONS (simplified versions for speed)
# ============================================================================

def lens_pca(returns, n_components=3):
    """PCA Lens: Measure variance concentration (high = regime stress)."""
    try:
        scaler = StandardScaler()
        scaled = scaler.fit_transform(returns.fillna(0))
        pca = PCA(n_components=min(n_components, scaled.shape[1]))
        pca.fit(scaled)
        # Higher concentration of variance in first component = more coherent/stressed
        return pca.explained_variance_ratio_[0]
    except:
        return 0.0


def lens_correlation(returns):
    """Correlation Lens: Average absolute correlation."""
    try:
        corr = returns.corr()
        upper = corr.values[np.triu_indices(len(corr), k=1)]
        valid = upper[~np.isnan(upper)]
        return np.mean(np.abs(valid)) if len(valid) > 0 else 0.0
    except:
        return 0.0


def lens_volatility(returns):
    """Volatility Lens: Average cross-sectional volatility."""
    try:
        return returns.std().mean()
    except:
        return 0.0


def lens_anomaly(returns):
    """Anomaly Lens: Isolation Forest anomaly score."""
    try:
        scaled = StandardScaler().fit_transform(returns.fillna(0))
        iso = IsolationForest(contamination=0.1, random_state=42)
        scores = iso.fit_predict(scaled)
        # Return fraction of anomalies detected
        return (scores == -1).mean()
    except:
        return 0.0


def lens_clustering(returns, n_clusters=3):
    """Clustering Lens: Cluster tightness (inverse of inertia)."""
    try:
        scaled = StandardScaler().fit_transform(returns.fillna(0))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans.fit(scaled)
        # Lower inertia = tighter clusters = more coherent regime
        # Normalize by number of samples
        return 1.0 / (1.0 + kmeans.inertia_ / len(scaled))
    except:
        return 0.0


def lens_regime_switch(returns, n_regimes=2):
    """Regime Switching Lens: Probability of being in stressed regime."""
    try:
        # Use mean return across all indicators
        mean_returns = returns.mean(axis=1).values.reshape(-1, 1)
        gmm = GaussianMixture(n_components=n_regimes, random_state=42)
        gmm.fit(mean_returns)
        probs = gmm.predict_proba(mean_returns)
        # Return probability of being in the lower-mean regime (stress)
        regime_means = gmm.means_.flatten()
        stress_regime = np.argmin(regime_means)
        return probs[:, stress_regime].mean()
    except:
        return 0.0


def lens_magnitude(returns):
    """Magnitude Lens: Average absolute return magnitude."""
    try:
        return returns.abs().mean().mean()
    except:
        return 0.0


def lens_skewness(returns):
    """Skewness Lens: Average skewness (negative = crash risk)."""
    try:
        skews = returns.skew()
        return -skews.mean()  # Flip sign so higher = more crash risk
    except:
        return 0.0


def lens_kurtosis(returns):
    """Kurtosis Lens: Average kurtosis (fat tails)."""
    try:
        kurts = returns.kurtosis()
        return kurts.mean()
    except:
        return 0.0


def lens_network_density(returns):
    """Network Lens: Density of significant correlations."""
    try:
        corr = returns.corr()
        # Count correlations above threshold
        threshold = 0.5
        n = len(corr)
        significant = (corr.abs() > threshold).sum().sum() - n  # exclude diagonal
        max_edges = n * (n - 1)
        return significant / max_edges if max_edges > 0 else 0.0
    except:
        return 0.0


def lens_dispersion(returns):
    """Dispersion Lens: Cross-sectional dispersion of returns."""
    try:
        # Standard deviation across indicators at each time point
        cross_std = returns.std(axis=1)
        return cross_std.mean()
    except:
        return 0.0


def lens_momentum(returns, lookback=21):
    """Momentum Lens: Persistence of trends."""
    try:
        # Autocorrelation of returns
        autocorrs = []
        for col in returns.columns:
            series = returns[col].dropna()
            if len(series) > lookback:
                autocorr = series.autocorr(lag=1)
                if not np.isnan(autocorr):
                    autocorrs.append(abs(autocorr))
        return np.mean(autocorrs) if autocorrs else 0.0
    except:
        return 0.0


def lens_drawdown(prices):
    """Drawdown Lens: Average current drawdown from peak."""
    try:
        drawdowns = []
        for col in prices.columns:
            series = prices[col].dropna()
            if len(series) > 0:
                peak = series.expanding().max()
                dd = (series - peak) / peak
                drawdowns.append(dd.iloc[-1])
        return -np.mean(drawdowns) if drawdowns else 0.0  # Flip so higher = more stress
    except:
        return 0.0


def lens_entropy(returns):
    """Entropy Lens: Information entropy of return distribution."""
    try:
        # Discretize returns and compute entropy
        entropies = []
        for col in returns.columns:
            series = returns[col].dropna()
            if len(series) > 10:
                # Simple histogram-based entropy
                hist, _ = np.histogram(series, bins=20, density=True)
                hist = hist[hist > 0]  # Remove zeros
                entropy = -np.sum(hist * np.log(hist + 1e-10))
                entropies.append(entropy)
        return np.mean(entropies) if entropies else 0.0
    except:
        return 0.0


# ============================================================================
# MAIN ANALYSIS
# ============================================================================

def run_all_lenses(returns, prices, window_returns, window_prices):
    """Run all lenses on a window of data and return scores."""
    
    scores = {
        'PCA': lens_pca(window_returns),
        'Correlation': lens_correlation(window_returns),
        'Volatility': lens_volatility(window_returns),
        'Anomaly': lens_anomaly(window_returns),
        'Clustering': lens_clustering(window_returns),
        'Regime': lens_regime_switch(window_returns),
        'Magnitude': lens_magnitude(window_returns),
        'Skewness': lens_skewness(window_returns),
        'Kurtosis': lens_kurtosis(window_returns),
        'Network': lens_network_density(window_returns),
        'Dispersion': lens_dispersion(window_returns),
        'Momentum': lens_momentum(window_returns),
        'Drawdown': lens_drawdown(window_prices),
        'Entropy': lens_entropy(window_returns),
    }
    
    return scores


def compute_lens_contributions(panel, window=63, step=21):
    """
    Run all lenses over rolling windows and track their signals over time.
    """
    
    # Prepare data
    returns = panel.pct_change().dropna()
    returns = returns.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Track results
    all_scores = []
    dates = []
    
    frame_indices = list(range(window, len(returns), step))
    
    print(f"  Processing {len(frame_indices)} time windows...")
    
    for i, idx in enumerate(frame_indices):
        window_returns = returns.iloc[idx-window:idx]
        window_prices = panel.iloc[idx-window:idx]
        
        scores = run_all_lenses(returns, panel, window_returns, window_prices)
        all_scores.append(scores)
        dates.append(returns.index[idx])
        
        if (i + 1) % 20 == 0:
            print(f"    Processed {i+1}/{len(frame_indices)} windows...")
    
    # Convert to DataFrame
    scores_df = pd.DataFrame(all_scores, index=dates)
    
    return scores_df


def normalize_scores(scores_df):
    """Normalize each lens to 0-1 scale for comparability."""
    normalized = scores_df.copy()
    for col in normalized.columns:
        min_val = normalized[col].min()
        max_val = normalized[col].max()
        if max_val > min_val:
            normalized[col] = (normalized[col] - min_val) / (max_val - min_val)
        else:
            normalized[col] = 0.5
    return normalized


def compute_consensus(normalized_df, threshold=0.7):
    """
    Compute consensus score: how many lenses are above threshold.
    This is the PRISM signal - when lenses agree, regime shift detected.
    """
    above_threshold = (normalized_df > threshold).sum(axis=1)
    consensus = above_threshold / len(normalized_df.columns)
    return consensus


def create_lens_contribution_chart(scores_df, output_dir):
    """Create stacked area chart of lens contributions over time."""
    
    # Normalize scores
    normalized = normalize_scores(scores_df)
    
    # Compute consensus
    consensus = compute_consensus(normalized, threshold=0.6)
    
    # Define colors for each lens (grouped by type)
    lens_colors = {
        # Statistical (blues)
        'PCA': '#1f77b4',
        'Correlation': '#aec7e8',
        'Volatility': '#6baed6',
        # Detection (reds)
        'Anomaly': '#d62728',
        'Clustering': '#ff7f0e',
        'Regime': '#e377c2',
        # Market (greens)
        'Magnitude': '#2ca02c',
        'Skewness': '#98df8a',
        'Kurtosis': '#bcbd22',
        # Network (purples)
        'Network': '#9467bd',
        'Dispersion': '#c5b0d5',
        # Time-based (grays/browns)
        'Momentum': '#8c564b',
        'Drawdown': '#c49c94',
        'Entropy': '#7f7f7f',
    }
    
    # Create figure
    fig, axes = plt.subplots(4, 1, figsize=(16, 16), 
                             gridspec_kw={'height_ratios': [2, 1.5, 1, 1]})
    
    dates = normalized.index
    
    # ===== Plot 1: Stacked Area (Normalized) =====
    ax1 = axes[0]
    
    # Order lenses by average score
    avg_scores = normalized.mean().sort_values(ascending=False)
    ordered_cols = avg_scores.index.tolist()
    
    colors = [lens_colors.get(col, 'gray') for col in ordered_cols]
    
    ax1.stackplot(dates, normalized[ordered_cols].T, labels=ordered_cols, 
                  colors=colors, alpha=0.8)
    
    ax1.set_ylabel('Cumulative Lens Signal', fontsize=11)
    ax1.set_title('PRISM Multi-Lens Contribution Over Time\n(Each color = different mathematical engine)', 
                  fontsize=13, fontweight='bold')
    ax1.legend(loc='center left', bbox_to_anchor=(1.01, 0.5), fontsize=9)
    ax1.set_xlim(dates[0], dates[-1])
    ax1.xaxis.set_major_locator(mdates.YearLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax1.grid(True, alpha=0.3, axis='y')
    
    # ===== Plot 2: Individual Lens Lines =====
    ax2 = axes[1]
    
    # Show top 5 most variable lenses
    lens_variance = normalized.var().sort_values(ascending=False)
    top_lenses = lens_variance.head(6).index.tolist()
    
    for lens in top_lenses:
        ax2.plot(dates, normalized[lens], label=lens, linewidth=1.5, 
                alpha=0.8, color=lens_colors.get(lens, 'gray'))
    
    ax2.set_ylabel('Normalized Score', fontsize=10)
    ax2.set_title('Most Active Lenses (Highest Variance)', fontsize=11, fontweight='bold')
    ax2.legend(loc='upper left', fontsize=9)
    ax2.set_xlim(dates[0], dates[-1])
    ax2.set_ylim(0, 1)
    ax2.axhline(y=0.7, color='red', linestyle='--', alpha=0.5, label='Alert threshold')
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_locator(mdates.YearLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    
    # ===== Plot 3: Consensus Score =====
    ax3 = axes[2]
    
    ax3.fill_between(dates, consensus, alpha=0.4, color='darkred')
    ax3.plot(dates, consensus, 'darkred', linewidth=1.5)
    
    # Threshold for regime detection
    ax3.axhline(y=0.5, color='orange', linestyle='--', alpha=0.7, label='50% consensus')
    ax3.axhline(y=0.7, color='red', linestyle='--', alpha=0.7, label='70% consensus')
    
    ax3.set_ylabel('Lens Consensus\n(fraction agreeing)', fontsize=10)
    ax3.set_title('PRISM Consensus Signal: When Lenses AGREE', fontsize=11, fontweight='bold')
    ax3.legend(loc='upper left', fontsize=9)
    ax3.set_xlim(dates[0], dates[-1])
    ax3.set_ylim(0, 1)
    ax3.grid(True, alpha=0.3)
    ax3.xaxis.set_major_locator(mdates.YearLocator())
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    
    # Add event annotations
    major_events = {
        '2016-02-11': 'Oil/China',
        '2018-02-05': 'Volmageddon',
        '2018-12-24': 'Fed Panic',
        '2020-03-23': 'COVID',
        '2022-06-16': 'Rate Shock',
    }
    
    for date_str, label in major_events.items():
        try:
            event_date = pd.Timestamp(date_str)
            if dates[0] <= event_date <= dates[-1]:
                ax3.axvline(x=event_date, color='black', linestyle=':', alpha=0.5)
                ax3.annotate(label, xy=(event_date, 0.95), fontsize=8, 
                           ha='center', rotation=90, alpha=0.7)
        except:
            pass
    
    # ===== Plot 4: Heatmap of lens activity =====
    ax4 = axes[3]
    
    # Resample to monthly for readability
    monthly = normalized.resample('M').mean()
    
    im = ax4.imshow(monthly.T, aspect='auto', cmap='YlOrRd', 
                    interpolation='nearest', vmin=0, vmax=1)
    
    ax4.set_yticks(range(len(monthly.columns)))
    ax4.set_yticklabels(monthly.columns, fontsize=8)
    
    # X-axis years
    year_positions = []
    year_labels = []
    for i, date in enumerate(monthly.index):
        if date.month == 1:
            year_positions.append(i)
            year_labels.append(date.year)
    ax4.set_xticks(year_positions)
    ax4.set_xticklabels(year_labels, fontsize=9)
    
    ax4.set_xlabel('Year', fontsize=10)
    ax4.set_title('Lens Activity Heatmap (Brighter = Stronger Signal)', fontsize=11, fontweight='bold')
    
    plt.colorbar(im, ax=ax4, shrink=0.6, label='Normalized Score')
    
    plt.tight_layout()
    
    fig.savefig(output_dir / "lens_contributions.png", dpi=150, bbox_inches='tight')
    print(f"💾 Saved: {output_dir / 'lens_contributions.png'}")
    
    plt.close()
    
    return normalized, consensus


def main():
    print("=" * 70)
    print("PRISM MULTI-LENS CONTRIBUTION ANALYSIS")
    print("=" * 70)
    print("\nRunning 14 mathematical lenses over time...")
    print("Tracking which engines detect regime shifts.")
    
    # Load data
    print("\n📥 Loading data...")
    panel = load_from_database()
    
    # Filter to last 10 years
    cutoff = panel.index.max() - pd.DateOffset(years=10)
    panel = panel[panel.index >= cutoff]
    print(f"  Date range: {panel.index.min().date()} to {panel.index.max().date()}")
    
    # Use columns with good coverage
    valid_cols = panel.columns[panel.notna().sum() > len(panel) * 0.5]
    clean_panel = panel[valid_cols].ffill().bfill().dropna()
    print(f"  Indicators: {len(valid_cols)}")
    
    # Run all lenses
    print("\n🔬 Running all lenses over rolling windows...")
    print("  Window: 63 days (quarterly)")
    print("  Step: 21 days (monthly)")
    
    scores_df = compute_lens_contributions(clean_panel, window=63, step=21)
    
    # Create output directory
    output_dir = OUTPUT_DIR / "lens_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create visualizations
    print("\n🎨 Creating visualizations...")
    normalized, consensus = create_lens_contribution_chart(scores_df, output_dir)
    
    # Print summary
    print("\n📊 LENS INFLUENCE RANKINGS (10-year average)")
    print("-" * 50)
    avg_scores = normalized.mean().sort_values(ascending=False)
    for i, (lens, score) in enumerate(avg_scores.items()):
        bar = "█" * int(score * 30)
        print(f"  {i+1:2}. {lens:12} {score:.3f} {bar}")
    
    # Find peak consensus dates
    print("\n🚨 PEAK CONSENSUS DATES (regime shift candidates)")
    print("-" * 50)
    top_consensus = consensus.nlargest(10)
    for date, value in top_consensus.items():
        print(f"  {date.date()} - {value:.1%} of lenses firing")
    
    # Save data
    scores_df.to_csv(output_dir / "lens_scores_raw.csv")
    normalized.to_csv(output_dir / "lens_scores_normalized.csv")
    consensus.to_frame('consensus').to_csv(output_dir / "lens_consensus.csv")
    print(f"\n💾 Saved data to: {output_dir}")
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()