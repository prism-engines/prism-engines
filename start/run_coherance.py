#!/usr/bin/env python3
"""
PRISM Correlation Coherence Over Time
======================================
Measures how "aligned" the market is at each point in time.

Key metric: Average pairwise correlation in a rolling window
- High coherence = everything moving together (crisis/regime shift)
- Low coherence = normal diversified market behavior

This is the core PRISM insight: when different lenses suddenly AGREE,
something significant is happening.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import sqlite3
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

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


PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DB_PATH = DATA_DIR / "prism.db"


def load_from_database():
    """Load all data from database and merge into a single panel."""
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


def compute_rolling_coherence(returns_df, window=63):
    """
    Compute rolling average correlation (coherence) over time.
    
    Args:
        returns_df: DataFrame of returns
        window: Rolling window size (63 = ~1 quarter of trading days)
    
    Returns:
        Series of coherence values indexed by date
    """
    coherence_values = []
    dates = []
    
    for i in range(window, len(returns_df)):
        window_data = returns_df.iloc[i-window:i]
        
        # Compute correlation matrix for this window
        corr_matrix = window_data.corr()
        
        # Extract upper triangle (excluding diagonal)
        upper_tri = corr_matrix.values[np.triu_indices(len(corr_matrix), k=1)]
        
        # Remove NaN values
        valid_corrs = upper_tri[~np.isnan(upper_tri)]
        
        if len(valid_corrs) > 0:
            # Average absolute correlation (coherence)
            avg_corr = np.mean(np.abs(valid_corrs))
            coherence_values.append(avg_corr)
            dates.append(returns_df.index[i])
    
    return pd.Series(coherence_values, index=dates, name='coherence')


def compute_correlation_dispersion(returns_df, window=63):
    """
    Compute standard deviation of correlations (how spread out they are).
    
    Low dispersion + high coherence = crisis (everything correlated)
    High dispersion + low coherence = normal (diverse behavior)
    """
    dispersion_values = []
    dates = []
    
    for i in range(window, len(returns_df)):
        window_data = returns_df.iloc[i-window:i]
        corr_matrix = window_data.corr()
        upper_tri = corr_matrix.values[np.triu_indices(len(corr_matrix), k=1)]
        valid_corrs = upper_tri[~np.isnan(upper_tri)]
        
        if len(valid_corrs) > 0:
            dispersion_values.append(np.std(valid_corrs))
            dates.append(returns_df.index[i])
    
    return pd.Series(dispersion_values, index=dates, name='dispersion')


def identify_regime_breaks(coherence, threshold_std=1.5):
    """
    Identify dates where coherence spikes above threshold.
    """
    mean_coh = coherence.mean()
    std_coh = coherence.std()
    threshold = mean_coh + threshold_std * std_coh
    
    high_coherence_dates = coherence[coherence > threshold]
    return high_coherence_dates, threshold


def main():
    print("=" * 70)
    print("PRISM CORRELATION COHERENCE ANALYSIS")
    print("=" * 70)
    print("\nMeasuring how 'aligned' the market is over time...")
    print("High coherence = everything moving together (potential regime shift)")
    
    # Load data
    print("\n📥 Loading data...")
    panel = load_from_database()
    
    # Filter to last 10 years (better indicator coverage)
    cutoff = panel.index.max() - pd.DateOffset(years=10)
    panel = panel[panel.index >= cutoff]
    print(f"  Date range: {panel.index.min().date()} to {panel.index.max().date()}")
    
    # Use only columns with reasonable coverage (50%)
    valid_cols = panel.columns[panel.notna().sum() > len(panel) * 0.5]
    clean_panel = panel[valid_cols].ffill().bfill()
    
    # Drop rows where we still have NaN
    clean_panel = clean_panel.dropna()
    print(f"  Indicators: {len(valid_cols)}")
    
    # Compute returns
    returns = clean_panel.pct_change().dropna()
    returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
    print(f"  Return observations: {len(returns)}")
    
    # Compute rolling coherence (quarterly window = 63 trading days)
    print("\n🔬 Computing rolling coherence (63-day window)...")
    coherence = compute_rolling_coherence(returns, window=63)
    dispersion = compute_correlation_dispersion(returns, window=63)
    
    # Identify regime breaks
    high_coh_dates, threshold = identify_regime_breaks(coherence, threshold_std=1.5)
    
    print(f"\n📊 COHERENCE STATISTICS")
    print("-" * 50)
    print(f"  Mean coherence: {coherence.mean():.3f}")
    print(f"  Std coherence:  {coherence.std():.3f}")
    print(f"  Max coherence:  {coherence.max():.3f} on {coherence.idxmax().date()}")
    print(f"  Min coherence:  {coherence.min():.3f} on {coherence.idxmin().date()}")
    print(f"  Regime break threshold: {threshold:.3f}")
    
    # Show top coherence spikes
    print(f"\n🚨 HIGH COHERENCE PERIODS (potential regime shifts)")
    print("-" * 50)
    
    # Group consecutive high-coherence days into events
    if len(high_coh_dates) > 0:
        sorted_spikes = coherence.nlargest(20)
        
        # Cluster nearby dates
        events = []
        current_event = None
        
        for date in sorted_spikes.index.sort_values():
            if current_event is None:
                current_event = {'start': date, 'end': date, 'peak': coherence[date], 'peak_date': date}
            elif (date - current_event['end']).days <= 30:
                current_event['end'] = date
                if coherence[date] > current_event['peak']:
                    current_event['peak'] = coherence[date]
                    current_event['peak_date'] = date
            else:
                events.append(current_event)
                current_event = {'start': date, 'end': date, 'peak': coherence[date], 'peak_date': date}
        
        if current_event:
            events.append(current_event)
        
        # Sort by peak coherence
        events.sort(key=lambda x: x['peak'], reverse=True)
        
        for i, event in enumerate(events[:10]):
            duration = (event['end'] - event['start']).days + 1
            print(f"  {i+1}. {event['peak_date'].date()} - Coherence: {event['peak']:.3f} ({duration} days)")
    
    # Create visualization
    print("\n🎨 Generating coherence timeline...")
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    
    # Plot 1: Coherence over time
    ax1 = axes[0]
    ax1.plot(coherence.index, coherence.values, 'b-', linewidth=0.8, alpha=0.8)
    ax1.axhline(y=threshold, color='r', linestyle='--', alpha=0.7, label=f'Threshold ({threshold:.2f})')
    ax1.axhline(y=coherence.mean(), color='g', linestyle=':', alpha=0.7, label=f'Mean ({coherence.mean():.2f})')
    ax1.fill_between(coherence.index, coherence.values, threshold, 
                     where=coherence.values > threshold, alpha=0.3, color='red')
    ax1.set_ylabel('Coherence\n(Avg |Correlation|)', fontsize=10)
    ax1.set_title('PRISM Correlation Coherence Over Time\nHigher = Market Moving Together', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Add event annotations
    major_events = {
        '2008-09-15': 'Lehman',
        '2008-10-15': 'Crisis Peak',
        '2011-08-08': 'US Downgrade',
        '2015-08-24': 'China Deval',
        '2018-02-05': 'Volmageddon',
        '2018-12-24': 'Fed Panic',
        '2020-03-16': 'COVID Crash',
        '2022-06-13': 'Rate Shock',
    }
    
    for date_str, label in major_events.items():
        try:
            event_date = pd.Timestamp(date_str)
            if event_date in coherence.index or (coherence.index.min() < event_date < coherence.index.max()):
                # Find closest date
                closest = coherence.index[coherence.index.get_indexer([event_date], method='nearest')[0]]
                ax1.annotate(label, xy=(closest, coherence[closest]), 
                           xytext=(0, 15), textcoords='offset points',
                           fontsize=7, ha='center', alpha=0.7,
                           arrowprops=dict(arrowstyle='->', alpha=0.5))
        except:
            pass
    
    # Plot 2: Dispersion over time
    ax2 = axes[1]
    ax2.plot(dispersion.index, dispersion.values, 'purple', linewidth=0.8, alpha=0.8)
    ax2.set_ylabel('Dispersion\n(Std of Correlations)', fontsize=10)
    ax2.set_title('Correlation Dispersion (Low = Everyone Agrees)', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Coherence / Dispersion ratio (regime signal)
    ax3 = axes[2]
    ratio = coherence / dispersion
    ratio = ratio.replace([np.inf, -np.inf], np.nan).dropna()
    ax3.plot(ratio.index, ratio.values, 'darkgreen', linewidth=0.8, alpha=0.8)
    ax3.axhline(y=ratio.mean(), color='g', linestyle=':', alpha=0.7)
    ax3.set_ylabel('Coherence/Dispersion\nRatio', fontsize=10)
    ax3.set_title('Regime Signal (High = Strong Consensus)', fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlabel('Date')
    
    # Format x-axis
    for ax in axes:
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    
    plt.tight_layout()
    
    # Save
    output_dir = OUTPUT_DIR / "coherence"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig.savefig(output_dir / "coherence_timeline.png", dpi=150, bbox_inches='tight')
    print(f"💾 Saved to {output_dir / 'coherence_timeline.png'}")
    
    # Save data
    coherence_df = pd.DataFrame({
        'coherence': coherence,
        'dispersion': dispersion
    })
    coherence_df.to_csv(output_dir / "coherence_data.csv")
    print(f"💾 Saved data to {output_dir / 'coherence_data.csv'}")
    
    plt.close()
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    
    return coherence, dispersion


if __name__ == "__main__":
    main()