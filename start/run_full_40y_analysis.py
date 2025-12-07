#!/usr/bin/env python3
"""
================================================================================
PRISM ENGINE: COMPREHENSIVE 40-YEAR MULTI-LENS REGIME DETECTION
================================================================================

This is the FULL PRISM analysis - designed to run for extended periods (30-60 min)
on a capable machine. It runs 14+ mathematical lenses over 40 years of data with
multiple temporal resolutions.

WHAT THIS DOES:
1. Loads 40 years of market data (1985-2025)
2. Runs 14 mathematical "lenses" (analytical engines) over rolling windows
3. Tracks when lenses AGREE (consensus = regime shift signal)
4. Adds ML-enhanced detection (Hidden Markov Models, Isolation Forest, LSTM-ready)
5. Generates comprehensive visualizations and reports

ESTIMATED RUNTIME:
- ~30-60 minutes on modern hardware
- ~2500+ time windows analyzed
- 14 lenses × 2500 windows = 35,000+ calculations

LENSES INCLUDED:
Statistical:
  1. PCA - Principal Component Analysis (variance concentration)
  2. Correlation - Average market coherence
  3. Volatility - Cross-sectional volatility
  4. Skewness - Crash risk (negative tail)
  5. Kurtosis - Fat tails / extreme events
  6. Entropy - Information disorder

Detection:
  7. Anomaly - Isolation Forest outlier detection
  8. Clustering - K-means regime tightness
  9. Regime - Gaussian Mixture Model state probability
  10. HMM - Hidden Markov Model state transitions (ML-enhanced)

Market:
  11. Magnitude - Return magnitude
  12. Drawdown - Distance from peak
  13. Momentum - Trend persistence (autocorrelation)
  14. Network - Correlation network density

Advanced (ML-Enhanced):
  15. Spillover - Cross-market information flow
  16. Covariance Regime - Realized covariance matrix analysis

OUTPUT FILES:
- lens_scores_40y.csv: Raw scores for all lenses over time
- lens_normalized_40y.csv: Normalized 0-1 scores
- consensus_signal_40y.csv: The PRISM consensus signal
- regime_shifts_detected.csv: Dates where consensus > threshold
- lens_contributions_40y.png: Stacked area visualization
- consensus_timeline_40y.png: Master regime detection chart
- lens_heatmap_40y.png: Lens × Time heatmap

USAGE:
    python run_full_40y_analysis.py

Author: PRISM Engine
Date: December 2025
================================================================================
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import sqlite3
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import LinearSegmentedColormap
from datetime import datetime
import time
import warnings
warnings.filterwarnings('ignore')

# ML imports
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.covariance import LedoitWolf
from scipy import stats
from scipy.ndimage import gaussian_filter1d

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


# Optional: Hidden Markov Model (install with: pip install hmmlearn)
try:
    from hmmlearn.hmm import GaussianHMM
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False
    print("⚠️  hmmlearn not installed. HMM lens will be skipped.")
    print("   Install with: pip install hmmlearn")

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    'years_back': 40,           # How many years to analyze
    'window_size': 63,          # Rolling window (63 = quarterly)
    'step_size': 5,             # Days between calculations (5 = weekly resolution)
    'consensus_threshold': 0.5, # Fraction of lenses needed for regime shift signal
    'n_regimes': 3,             # Number of regimes for GMM/HMM
    'anomaly_contamination': 0.05,  # Fraction of data considered anomalous
}


# ============================================================================
# DATA LOADING
# ============================================================================

def load_from_database():
    """Load all data from unified indicator_values table."""
    print("\n📥 LOADING DATA FROM DATABASE...")

    from data.sql.db_connector import load_all_indicators_wide

    # Use unified loader (falls back to legacy tables if needed)
    panel = load_all_indicators_wide()

    if panel.empty:
        print("   ⚠️ No data found in database")
        return pd.DataFrame()

    print(f"   Combined panel: {len(panel)} days, {len(panel.columns)} indicators")
    print(f"   Date range: {panel.index.min().date()} to {panel.index.max().date()}")

    return panel


def prepare_data(panel, years_back=40):
    """Prepare data for analysis - filter dates and clean."""
    print(f"\n🔧 PREPARING DATA (last {years_back} years)...")
    
    # Filter to requested time period
    cutoff = panel.index.max() - pd.DateOffset(years=years_back)
    panel = panel[panel.index >= cutoff]
    print(f"   Filtered date range: {panel.index.min().date()} to {panel.index.max().date()}")
    
    # Keep columns with at least 30% coverage
    min_coverage = 0.30
    valid_cols = panel.columns[panel.notna().sum() > len(panel) * min_coverage]
    panel = panel[valid_cols]
    print(f"   Indicators with >{min_coverage*100:.0f}% coverage: {len(valid_cols)}")
    
    # Forward fill, backward fill, then drop remaining NaN
    panel = panel.ffill().bfill()
    panel = panel.dropna(axis=1)  # Drop any columns still with NaN
    print(f"   Final indicators: {len(panel.columns)}")
    
    # Compute returns
    returns = panel.pct_change().dropna()
    returns = returns.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    print(f"   Return observations: {len(returns)}")
    
    return panel, returns


# ============================================================================
# LENS IMPLEMENTATIONS
# ============================================================================

class PRISMLenses:
    """
    Collection of analytical lenses for regime detection.
    Each lens looks at the market from a different mathematical perspective.
    """
    
    @staticmethod
    def pca_lens(returns, n_components=3):
        """
        PCA LENS: Measures variance concentration.
        
        High score = variance concentrated in few factors = coherent/stressed market
        Low score = variance distributed = normal diversified market
        """
        try:
            if len(returns) < 10 or returns.shape[1] < 3:
                return 0.0
            scaler = StandardScaler()
            scaled = scaler.fit_transform(returns.fillna(0))
            n_comp = min(n_components, scaled.shape[1], scaled.shape[0])
            pca = PCA(n_components=n_comp)
            pca.fit(scaled)
            # Return variance explained by first component
            return pca.explained_variance_ratio_[0]
        except:
            return 0.0
    
    @staticmethod
    def correlation_lens(returns):
        """
        CORRELATION LENS: Measures market coherence.
        
        High score = all assets moving together = crisis/stress
        Low score = assets moving independently = normal market
        """
        try:
            corr = returns.corr()
            upper = corr.values[np.triu_indices(len(corr), k=1)]
            valid = upper[~np.isnan(upper)]
            return np.mean(np.abs(valid)) if len(valid) > 0 else 0.0
        except:
            return 0.0
    
    @staticmethod
    def volatility_lens(returns):
        """
        VOLATILITY LENS: Average cross-sectional volatility.
        
        High score = high volatility = stressed market
        Low score = calm market
        """
        try:
            return returns.std().mean()
        except:
            return 0.0
    
    @staticmethod
    def skewness_lens(returns):
        """
        SKEWNESS LENS: Measures crash risk.
        
        High score (originally negative skew) = left tail risk = crash potential
        Low score = balanced distribution
        """
        try:
            skews = returns.skew()
            # Flip sign so higher = more crash risk
            return -skews.mean()
        except:
            return 0.0
    
    @staticmethod
    def kurtosis_lens(returns):
        """
        KURTOSIS LENS: Measures fat tails / extreme events.
        
        High score = fat tails = more extreme events likely
        Low score = thin tails = normal distribution
        """
        try:
            kurts = returns.kurtosis()
            # Normalize to positive range
            return max(0, kurts.mean())
        except:
            return 0.0
    
    @staticmethod
    def entropy_lens(returns, bins=20):
        """
        ENTROPY LENS: Information entropy of return distribution.
        
        High score = disordered/unpredictable market
        Low score = ordered/predictable market
        """
        try:
            entropies = []
            for col in returns.columns:
                series = returns[col].dropna()
                if len(series) > bins:
                    hist, _ = np.histogram(series, bins=bins, density=True)
                    hist = hist[hist > 0]
                    entropy = -np.sum(hist * np.log(hist + 1e-10))
                    entropies.append(entropy)
            return np.mean(entropies) if entropies else 0.0
        except:
            return 0.0
    
    @staticmethod
    def anomaly_lens(returns, contamination=0.05):
        """
        ANOMALY LENS: Isolation Forest outlier detection.
        
        High score = many anomalies detected = unusual market behavior
        Low score = normal behavior
        """
        try:
            if len(returns) < 20:
                return 0.0
            scaled = StandardScaler().fit_transform(returns.fillna(0))
            iso = IsolationForest(contamination=contamination, random_state=42, n_jobs=-1)
            scores = iso.fit_predict(scaled)
            return (scores == -1).mean()
        except:
            return 0.0
    
    @staticmethod
    def clustering_lens(returns, n_clusters=3):
        """
        CLUSTERING LENS: K-means cluster tightness.
        
        High score = tight clusters = clear regime
        Low score = loose clusters = ambiguous state
        """
        try:
            if len(returns) < n_clusters * 5:
                return 0.0
            scaled = StandardScaler().fit_transform(returns.fillna(0))
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            kmeans.fit(scaled)
            # Inverse of inertia, normalized
            return 1.0 / (1.0 + kmeans.inertia_ / len(scaled))
        except:
            return 0.0
    
    @staticmethod
    def regime_gmm_lens(returns, n_regimes=3):
        """
        REGIME LENS (GMM): Gaussian Mixture Model probability of stress regime.
        
        High score = likely in stress/crisis regime
        Low score = likely in normal/growth regime
        """
        try:
            if len(returns) < n_regimes * 10:
                return 0.0
            # Use mean return across indicators as the observable
            mean_returns = returns.mean(axis=1).values.reshape(-1, 1)
            gmm = GaussianMixture(n_components=n_regimes, random_state=42, n_init=3)
            gmm.fit(mean_returns)
            probs = gmm.predict_proba(mean_returns)
            # Find the stress regime (lowest mean)
            regime_means = gmm.means_.flatten()
            stress_regime = np.argmin(regime_means)
            return probs[:, stress_regime].mean()
        except:
            return 0.0
    
    @staticmethod
    def hmm_lens(returns, n_states=2):
        """
        HMM LENS: Hidden Markov Model state detection.
        
        Uses transition probabilities to detect regime changes.
        High score = transitioning between states = regime shift
        """
        if not HMM_AVAILABLE:
            return 0.0
        try:
            if len(returns) < n_states * 20:
                return 0.0
            # Use mean return as observable
            obs = returns.mean(axis=1).values.reshape(-1, 1)
            model = GaussianHMM(n_components=n_states, covariance_type='full', 
                              n_iter=100, random_state=42)
            model.fit(obs)
            
            # Get state sequence
            states = model.predict(obs)
            
            # Count state transitions (regime changes)
            transitions = np.sum(states[1:] != states[:-1])
            transition_rate = transitions / len(states)
            
            return transition_rate
        except:
            return 0.0
    
    @staticmethod
    def magnitude_lens(returns):
        """
        MAGNITUDE LENS: Average absolute return size.
        
        High score = large moves = volatile/stressed market
        Low score = small moves = calm market
        """
        try:
            return returns.abs().mean().mean()
        except:
            return 0.0
    
    @staticmethod
    def drawdown_lens(prices):
        """
        DRAWDOWN LENS: Average drawdown from peak.
        
        High score = deep drawdowns = stressed market
        Low score = near peaks = healthy market
        """
        try:
            drawdowns = []
            for col in prices.columns:
                series = prices[col].dropna()
                if len(series) > 0:
                    peak = series.expanding().max()
                    dd = (series - peak) / peak
                    current_dd = dd.iloc[-1]
                    if not np.isnan(current_dd):
                        drawdowns.append(current_dd)
            # Flip sign so higher = more stress
            return -np.mean(drawdowns) if drawdowns else 0.0
        except:
            return 0.0
    
    @staticmethod
    def momentum_lens(returns, lag=5):
        """
        MOMENTUM LENS: Autocorrelation of returns (trend persistence).
        
        High score = strong trends persisting
        Low score = mean-reverting market
        """
        try:
            autocorrs = []
            for col in returns.columns:
                series = returns[col].dropna()
                if len(series) > lag + 10:
                    autocorr = series.autocorr(lag=lag)
                    if not np.isnan(autocorr):
                        autocorrs.append(abs(autocorr))
            return np.mean(autocorrs) if autocorrs else 0.0
        except:
            return 0.0
    
    @staticmethod
    def network_lens(returns, threshold=0.5):
        """
        NETWORK LENS: Density of significant correlations.
        
        High score = many significant connections = crisis network
        Low score = sparse connections = normal market
        """
        try:
            corr = returns.corr()
            n = len(corr)
            if n < 2:
                return 0.0
            # Count edges above threshold
            significant = (corr.abs() > threshold).sum().sum() - n  # exclude diagonal
            max_edges = n * (n - 1)
            return significant / max_edges if max_edges > 0 else 0.0
        except:
            return 0.0
    
    @staticmethod
    def dispersion_lens(returns):
        """
        DISPERSION LENS: Cross-sectional return dispersion.
        
        High score = high dispersion = differentiated market
        Low score = low dispersion = uniform market
        """
        try:
            cross_std = returns.std(axis=1)
            return cross_std.mean()
        except:
            return 0.0
    
    @staticmethod
    def covariance_regime_lens(returns):
        """
        COVARIANCE REGIME LENS: Realized covariance stability.
        
        Uses Ledoit-Wolf shrinkage to estimate covariance.
        High score = unstable covariance = regime transition
        """
        try:
            if len(returns) < 20 or returns.shape[1] < 3:
                return 0.0
            
            # Split window in half
            mid = len(returns) // 2
            first_half = returns.iloc[:mid]
            second_half = returns.iloc[mid:]
            
            # Estimate covariance for each half
            lw1 = LedoitWolf().fit(first_half.fillna(0))
            lw2 = LedoitWolf().fit(second_half.fillna(0))
            
            # Frobenius norm of difference
            cov_diff = np.linalg.norm(lw1.covariance_ - lw2.covariance_, 'fro')
            
            # Normalize by average covariance norm
            avg_norm = (np.linalg.norm(lw1.covariance_, 'fro') + 
                       np.linalg.norm(lw2.covariance_, 'fro')) / 2
            
            return cov_diff / avg_norm if avg_norm > 0 else 0.0
        except:
            return 0.0


# ============================================================================
# MAIN ANALYSIS ENGINE
# ============================================================================

def run_all_lenses_on_window(returns, prices, config):
    """Run all lenses on a single window of data."""
    
    lenses = PRISMLenses()
    
    scores = {
        'PCA': lenses.pca_lens(returns),
        'Correlation': lenses.correlation_lens(returns),
        'Volatility': lenses.volatility_lens(returns),
        'Skewness': lenses.skewness_lens(returns),
        'Kurtosis': lenses.kurtosis_lens(returns),
        'Entropy': lenses.entropy_lens(returns),
        'Anomaly': lenses.anomaly_lens(returns, config['anomaly_contamination']),
        'Clustering': lenses.clustering_lens(returns, config['n_regimes']),
        'Regime_GMM': lenses.regime_gmm_lens(returns, config['n_regimes']),
        'HMM': lenses.hmm_lens(returns),
        'Magnitude': lenses.magnitude_lens(returns),
        'Drawdown': lenses.drawdown_lens(prices),
        'Momentum': lenses.momentum_lens(returns),
        'Network': lenses.network_lens(returns),
        'Dispersion': lenses.dispersion_lens(returns),
        'Covariance': lenses.covariance_regime_lens(returns),
    }
    
    return scores


def compute_all_lens_scores(panel, returns, config):
    """
    Main computation loop - runs all lenses over all time windows.
    This is the computationally intensive part.
    """
    window = config['window_size']
    step = config['step_size']
    
    # Calculate number of windows
    n_windows = (len(returns) - window) // step + 1
    
    print(f"\n🔬 RUNNING LENS ANALYSIS...")
    print(f"   Window size: {window} days")
    print(f"   Step size: {step} days")
    print(f"   Total windows to process: {n_windows}")
    print(f"   Lenses per window: 16")
    print(f"   Total calculations: {n_windows * 16:,}")
    print()
    
    all_scores = []
    dates = []
    
    start_time = time.time()
    
    for i, idx in enumerate(range(window, len(returns), step)):
        # Extract window data
        window_returns = returns.iloc[idx-window:idx]
        window_prices = panel.iloc[idx-window:idx]
        
        # Run all lenses
        scores = run_all_lenses_on_window(window_returns, window_prices, config)
        
        all_scores.append(scores)
        dates.append(returns.index[idx])
        
        # Progress reporting
        if (i + 1) % 100 == 0 or i == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            remaining = (n_windows - i - 1) / rate if rate > 0 else 0
            
            print(f"   Progress: {i+1:5}/{n_windows} windows "
                  f"({100*(i+1)/n_windows:5.1f}%) | "
                  f"Elapsed: {elapsed/60:5.1f}min | "
                  f"Remaining: {remaining/60:5.1f}min")
    
    total_time = time.time() - start_time
    print(f"\n   ✅ Completed in {total_time/60:.1f} minutes")
    
    # Convert to DataFrame
    scores_df = pd.DataFrame(all_scores, index=dates)
    
    return scores_df


def normalize_scores(scores_df):
    """Normalize each lens to 0-1 scale."""
    normalized = scores_df.copy()
    for col in normalized.columns:
        min_val = normalized[col].min()
        max_val = normalized[col].max()
        if max_val > min_val:
            normalized[col] = (normalized[col] - min_val) / (max_val - min_val)
        else:
            normalized[col] = 0.5
    return normalized


def compute_consensus(normalized_df, threshold=0.6):
    """
    Compute PRISM consensus signal.
    
    This is the KEY OUTPUT: when multiple independent lenses agree
    that something unusual is happening, we have a regime shift signal.
    """
    above_threshold = (normalized_df > threshold).sum(axis=1)
    consensus = above_threshold / len(normalized_df.columns)
    return consensus


def detect_regime_shifts(consensus, threshold=0.5):
    """Identify specific dates where regime shifts are detected."""
    shift_dates = consensus[consensus > threshold]
    
    # Cluster nearby dates into events
    events = []
    if len(shift_dates) > 0:
        current_event = {'start': shift_dates.index[0], 
                        'end': shift_dates.index[0],
                        'peak_consensus': shift_dates.iloc[0],
                        'peak_date': shift_dates.index[0]}
        
        for date, value in shift_dates.iloc[1:].items():
            if (date - current_event['end']).days <= 30:
                current_event['end'] = date
                if value > current_event['peak_consensus']:
                    current_event['peak_consensus'] = value
                    current_event['peak_date'] = date
            else:
                events.append(current_event)
                current_event = {'start': date, 'end': date,
                               'peak_consensus': value, 'peak_date': date}
        
        events.append(current_event)
    
    return events


# ============================================================================
# VISUALIZATION
# ============================================================================

def create_comprehensive_visualizations(scores_df, normalized_df, consensus, events, output_dir):
    """Create all visualization outputs."""
    
    print("\n🎨 GENERATING VISUALIZATIONS...")
    
    dates = normalized_df.index
    
    # Define lens colors by category
    lens_colors = {
        # Statistical (blues)
        'PCA': '#1f77b4', 'Correlation': '#aec7e8', 'Volatility': '#6baed6',
        # Risk (reds/oranges)
        'Skewness': '#d62728', 'Kurtosis': '#ff7f0e', 'Entropy': '#ffbb78',
        # Detection (greens)
        'Anomaly': '#2ca02c', 'Clustering': '#98df8a', 
        'Regime_GMM': '#17becf', 'HMM': '#9edae5',
        # Market (purples)
        'Magnitude': '#9467bd', 'Drawdown': '#c5b0d5',
        'Momentum': '#8c564b', 'Network': '#c49c94',
        # Advanced (grays)
        'Dispersion': '#7f7f7f', 'Covariance': '#bcbd22',
    }
    
    # =========================================================================
    # FIGURE 1: Master Consensus Timeline
    # =========================================================================
    print("   Creating master consensus timeline...")
    
    fig, axes = plt.subplots(3, 1, figsize=(20, 14), 
                             gridspec_kw={'height_ratios': [2, 1, 1]})
    
    # Plot 1: Consensus signal with event markers
    ax1 = axes[0]
    
    # Smooth the consensus for visualization
    consensus_smooth = pd.Series(
        gaussian_filter1d(consensus.values, sigma=3),
        index=consensus.index
    )
    
    ax1.fill_between(dates, consensus_smooth, alpha=0.3, color='darkred')
    ax1.plot(dates, consensus_smooth, 'darkred', linewidth=1.5, label='Consensus')
    
    # Thresholds
    ax1.axhline(y=0.5, color='orange', linestyle='--', alpha=0.7, linewidth=1)
    ax1.axhline(y=0.7, color='red', linestyle='--', alpha=0.7, linewidth=1)
    
    # Mark detected events
    for event in events:
        ax1.axvspan(event['start'], event['end'], alpha=0.2, color='red')
        ax1.annotate(f"{event['peak_consensus']:.0%}",
                    xy=(event['peak_date'], event['peak_consensus']),
                    xytext=(0, 10), textcoords='offset points',
                    fontsize=8, ha='center', fontweight='bold')
    
    # Known historical events
    historical_events = {
        '1987-10-19': 'Black Monday',
        '1990-08-02': 'Gulf War',
        '1998-08-17': 'LTCM/Russia',
        '2000-03-10': 'Dot-com Peak',
        '2001-09-11': '9/11',
        '2008-09-15': 'Lehman',
        '2010-05-06': 'Flash Crash',
        '2011-08-05': 'US Downgrade',
        '2015-08-24': 'China Deval',
        '2018-02-05': 'Volmageddon',
        '2020-03-16': 'COVID',
        '2022-06-13': 'Rate Shock',
        '2025-04-02': 'Tariff Crash',
    }
    
    for date_str, label in historical_events.items():
        try:
            event_date = pd.Timestamp(date_str)
            if dates[0] <= event_date <= dates[-1]:
                ax1.axvline(x=event_date, color='black', linestyle=':', alpha=0.3)
                ax1.annotate(label, xy=(event_date, 0.95), fontsize=7,
                           rotation=90, ha='right', va='top', alpha=0.6)
        except:
            pass
    
    ax1.set_ylabel('Lens Consensus', fontsize=12)
    ax1.set_title('PRISM ENGINE: 40-Year Regime Detection\n'
                  'Consensus = Fraction of Lenses Detecting Anomaly',
                  fontsize=14, fontweight='bold')
    ax1.set_ylim(0, 1)
    ax1.set_xlim(dates[0], dates[-1])
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Top contributing lenses
    ax2 = axes[1]
    
    # Find most active lenses (highest variance)
    lens_variance = normalized_df.var().sort_values(ascending=False)
    top_lenses = lens_variance.head(6).index.tolist()
    
    for lens in top_lenses:
        color = lens_colors.get(lens, 'gray')
        ax2.plot(dates, normalized_df[lens], label=lens, linewidth=1, 
                alpha=0.8, color=color)
    
    ax2.set_ylabel('Normalized Score', fontsize=10)
    ax2.set_title('Most Active Lenses (Highest Variance)', fontsize=11)
    ax2.legend(loc='upper left', ncol=3, fontsize=8)
    ax2.set_ylim(0, 1)
    ax2.set_xlim(dates[0], dates[-1])
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Stacked contribution
    ax3 = axes[2]
    
    # Normalize to sum to 1
    contrib_norm = normalized_df.div(normalized_df.sum(axis=1), axis=0)
    colors = [lens_colors.get(col, 'gray') for col in contrib_norm.columns]
    
    ax3.stackplot(dates, contrib_norm.T, labels=contrib_norm.columns,
                  colors=colors, alpha=0.8)
    ax3.set_ylabel('Contribution Share', fontsize=10)
    ax3.set_xlabel('Date', fontsize=11)
    ax3.set_title('Lens Contribution Breakdown', fontsize=11)
    ax3.legend(loc='center left', bbox_to_anchor=(1.01, 0.5), fontsize=7)
    ax3.set_ylim(0, 1)
    ax3.set_xlim(dates[0], dates[-1])
    
    # Format x-axes
    for ax in axes:
        ax.xaxis.set_major_locator(mdates.YearLocator(5))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    
    plt.tight_layout()
    fig.savefig(output_dir / "consensus_timeline_40y.png", dpi=150, bbox_inches='tight')
    print(f"   💾 Saved: consensus_timeline_40y.png")
    plt.close()
    
    # =========================================================================
    # FIGURE 2: Lens Heatmap
    # =========================================================================
    print("   Creating lens heatmap...")
    
    fig, ax = plt.subplots(figsize=(20, 10))
    
    # Resample to monthly for readability
    monthly = normalized_df.resample('M').mean()
    
    # Custom colormap: blue (low) -> white -> red (high)
    cmap = LinearSegmentedColormap.from_list('prism', ['#2166ac', 'white', '#b2182b'])
    
    im = ax.imshow(monthly.T, aspect='auto', cmap=cmap, 
                   interpolation='nearest', vmin=0, vmax=1)
    
    ax.set_yticks(range(len(monthly.columns)))
    ax.set_yticklabels(monthly.columns, fontsize=9)
    
    # X-axis: years
    year_positions = []
    year_labels = []
    for i, date in enumerate(monthly.index):
        if date.month == 1 and date.year % 5 == 0:
            year_positions.append(i)
            year_labels.append(date.year)
    ax.set_xticks(year_positions)
    ax.set_xticklabels(year_labels, fontsize=10)
    
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Lens', fontsize=12)
    ax.set_title('PRISM Lens Activity Heatmap (40 Years)\n'
                 'Red = High Signal / Blue = Low Signal', 
                 fontsize=14, fontweight='bold')
    
    plt.colorbar(im, ax=ax, shrink=0.6, label='Normalized Score')
    
    plt.tight_layout()
    fig.savefig(output_dir / "lens_heatmap_40y.png", dpi=150, bbox_inches='tight')
    print(f"   💾 Saved: lens_heatmap_40y.png")
    plt.close()
    
    # =========================================================================
    # FIGURE 3: Detailed Event Analysis
    # =========================================================================
    print("   Creating event detail charts...")
    
    if len(events) > 0:
        n_events = min(len(events), 12)
        fig, axes = plt.subplots(3, 4, figsize=(20, 12))
        axes = axes.flatten()
        
        # Sort events by peak consensus
        sorted_events = sorted(events, key=lambda x: x['peak_consensus'], reverse=True)
        
        for i, event in enumerate(sorted_events[:n_events]):
            ax = axes[i]
            
            # Get data around event
            start = event['start'] - pd.DateOffset(days=30)
            end = event['end'] + pd.DateOffset(days=30)
            
            event_data = normalized_df[(normalized_df.index >= start) & 
                                       (normalized_df.index <= end)]
            
            if len(event_data) > 0:
                for col in event_data.columns:
                    ax.plot(event_data.index, event_data[col], 
                           alpha=0.5, linewidth=0.8)
                
                # Highlight event period
                ax.axvspan(event['start'], event['end'], alpha=0.2, color='red')
                
                ax.set_title(f"{event['peak_date'].strftime('%Y-%m-%d')}\n"
                            f"Consensus: {event['peak_consensus']:.0%}",
                            fontsize=10)
                ax.set_ylim(0, 1)
                ax.tick_params(labelsize=7)
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
        
        # Hide unused subplots
        for i in range(n_events, len(axes)):
            axes[i].set_visible(False)
        
        fig.suptitle('Top 12 Detected Regime Shifts - Lens Detail', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        fig.savefig(output_dir / "event_details_40y.png", dpi=150, bbox_inches='tight')
        print(f"   💾 Saved: event_details_40y.png")
        plt.close()
    
    print("   ✅ Visualizations complete")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    
    print("=" * 80)
    print("PRISM ENGINE: COMPREHENSIVE 40-YEAR MULTI-LENS ANALYSIS")
    print("=" * 80)
    print(f"\nStarted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nConfiguration:")
    for key, value in CONFIG.items():
        print(f"   {key}: {value}")
    
    # Load and prepare data
    panel = load_from_database()
    panel, returns = prepare_data(panel, years_back=CONFIG['years_back'])
    
    # Run all lenses
    scores_df = compute_all_lens_scores(panel, returns, CONFIG)
    
    # Normalize scores
    print("\n📊 NORMALIZING SCORES...")
    normalized_df = normalize_scores(scores_df)
    
    # Compute consensus
    print("\n🎯 COMPUTING CONSENSUS SIGNAL...")
    consensus = compute_consensus(normalized_df, threshold=0.6)
    
    # Detect regime shifts
    print("\n🚨 DETECTING REGIME SHIFTS...")
    events = detect_regime_shifts(consensus, threshold=CONFIG['consensus_threshold'])
    print(f"   Detected {len(events)} regime shift events")
    
    # Create output directory
    output_dir = OUTPUT_DIR / "full_40y_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save data files
    print("\n💾 SAVING DATA FILES...")
    scores_df.to_csv(output_dir / "lens_scores_40y.csv")
    normalized_df.to_csv(output_dir / "lens_normalized_40y.csv")
    consensus.to_frame('consensus').to_csv(output_dir / "consensus_signal_40y.csv")
    print(f"   Saved to: {output_dir}")
    
    # Save detected events
    if events:
        events_df = pd.DataFrame(events)
        events_df.to_csv(output_dir / "regime_shifts_detected.csv", index=False)
        print(f"   Saved {len(events)} detected events")
    
    # Create visualizations
    create_comprehensive_visualizations(scores_df, normalized_df, consensus, 
                                        events, output_dir)
    
    # Print summary report
    print("\n" + "=" * 80)
    print("ANALYSIS SUMMARY")
    print("=" * 80)
    
    print(f"\n📅 Date Range: {scores_df.index.min().date()} to {scores_df.index.max().date()}")
    print(f"   Total windows analyzed: {len(scores_df)}")
    print(f"   Total lens calculations: {len(scores_df) * len(scores_df.columns):,}")
    
    print(f"\n📊 LENS INFLUENCE RANKINGS (average normalized score):")
    print("-" * 50)
    avg_scores = normalized_df.mean().sort_values(ascending=False)
    for i, (lens, score) in enumerate(avg_scores.items()):
        bar = "█" * int(score * 30)
        print(f"   {i+1:2}. {lens:15} {score:.3f} {bar}")
    
    print(f"\n🚨 TOP REGIME SHIFT EVENTS DETECTED:")
    print("-" * 50)
    sorted_events = sorted(events, key=lambda x: x['peak_consensus'], reverse=True)
    for i, event in enumerate(sorted_events[:15]):
        duration = (event['end'] - event['start']).days + 1
        print(f"   {i+1:2}. {event['peak_date'].strftime('%Y-%m-%d')} | "
              f"Consensus: {event['peak_consensus']:.0%} | "
              f"Duration: {duration} days")
    
    print(f"\n📁 Output files saved to: {output_dir}")
    print(f"\n✅ Analysis completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)


if __name__ == "__main__":
    main()
