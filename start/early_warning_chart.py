#!/usr/bin/env python3
"""
PRISM Early Warning Chart
=========================

Stacked bar chart showing leading indicators (inverted ranks = higher = more warning)
overlaid with SP500 price line.

When the bars grow tall, trouble is brewing.

Usage:
    python early_warning_chart.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
import sqlite3

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


# Paths
DB_PATH = DATA_DIR / "prism.db"
# OUTPUT_DIR imported from output_config

def load_multiresolution():
    """Load multiresolution data."""
    csv_path = OUTPUT_DIR / "multiresolution.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path, parse_dates=['date'])
    return None

def load_spy_prices():
    """Load SP500 prices from unified indicator_values table."""
    from data.sql.db_connector import load_indicator

    # Use unified loader (falls back to legacy tables if needed)
    df = load_indicator('SP500')

    if df.empty:
        return pd.DataFrame(columns=['date', 'value'])

    df['date'] = pd.to_datetime(df['date'])
    return df[['date', 'value']]

def create_early_warning_chart(window_size=63):
    """Create the early warning visualization."""
    
    print("Loading data...")
    
    # Load multiresolution rankings
    mr = load_multiresolution()
    if mr is None:
        print("Error: multiresolution.csv not found")
        return
    
    # Filter to specified window size
    mr = mr[mr['window_size'] == window_size].copy()
    
    # Pivot to get ranks by date
    pivot = mr.pivot(index='date', columns='indicator', values='consensus_rank')
    pivot.index = pd.to_datetime(pivot.index)
    pivot = pivot.sort_index()
    
    # Key early warning indicators
    warning_indicators = ['T10Y2Y', 'BAMLH0A0HYM2', 'VIXCLS', 'iwm_us']
    
    # Check which exist
    available = [ind for ind in warning_indicators if ind in pivot.columns]
    print(f"Available indicators: {available}")
    
    # Invert ranks: rank 1 = most important = highest bar
    # Max rank is ~33, so inverted = 34 - rank
    max_rank = 34
    inverted = pd.DataFrame(index=pivot.index)
    
    for ind in available:
        inverted[ind] = max_rank - pivot[ind]
    
    # Normalize each to 0-100 scale for stacking
    for col in inverted.columns:
        inverted[col] = (inverted[col] / inverted[col].max()) * 25  # Each gets max 25%
    
    # Load SP500 for overlay
    spy = load_spy_prices()
    spy = spy.set_index('date')
    
    # Align dates
    common_start = max(inverted.index.min(), spy.index.min())
    common_end = min(inverted.index.max(), spy.index.max())
    
    inverted = inverted.loc[common_start:common_end]
    spy = spy.loc[common_start:common_end]
    
    # Resample to weekly for cleaner bars
    inverted_weekly = inverted.resample('W').mean()
    spy_weekly = spy.resample('W').last()
    
    # Create figure
    fig, ax1 = plt.subplots(figsize=(16, 8))
    
    # Color scheme
    colors = {
        'T10Y2Y': '#FF6B6B',      # Red - Yield Curve
        'BAMLH0A0HYM2': '#4ECDC4', # Teal - Credit Spreads
        'VIXCLS': '#FFE66D',       # Yellow - VIX
        'iwm_us': '#95E1D3',       # Light green - Small Caps
    }
    
    # Plot stacked bars
    bottom = np.zeros(len(inverted_weekly))
    bar_width = 5  # days
    
    for ind in available:
        ax1.bar(inverted_weekly.index, inverted_weekly[ind], 
                bottom=bottom, width=bar_width, 
                label=ind, color=colors.get(ind, 'gray'), alpha=0.7)
        bottom += inverted_weekly[ind].values
    
    ax1.set_ylabel('Warning Level (higher = more important)', fontsize=12)
    ax1.set_ylim(0, 100)
    
    # Add SP500 on secondary axis
    ax2 = ax1.twinx()
    ax2.plot(spy_weekly.index, spy_weekly['value'], 
             color='black', linewidth=2, label='SP500', alpha=0.8)
    ax2.set_ylabel('SP500', fontsize=12)
    
    # Mark major events
    events = [
        ('2020-02-20', 'COVID\nCrash', 'red'),
        ('2022-01-03', '2022\nBear', 'red'),
        ('2025-06-27', 'June 2025\nBreak', 'orange'),
    ]
    
    for date_str, label, color in events:
        try:
            event_date = pd.Timestamp(date_str)
            if common_start <= event_date <= common_end:
                ax1.axvline(event_date, color=color, linestyle='--', alpha=0.7, linewidth=1.5)
                ax1.text(event_date, 95, label, ha='center', fontsize=9, color=color)
        except:
            pass
    
    # Formatting
    ax1.set_xlabel('Date', fontsize=12)
    ax1.set_title('PRISM Early Warning System\n(Taller bars = indicators rising in importance = trouble brewing)', 
                  fontsize=14, fontweight='bold')
    
    # Legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10)
    
    # Date formatting
    ax1.xaxis.set_major_locator(mdates.YearLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax1.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    
    plt.tight_layout()
    
    # Save
    output_path = OUTPUT_DIR / 'early_warning_chart.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved: {output_path}")
    
    # Also create a zoomed version for recent period
    create_zoomed_chart(inverted, spy, available, colors)
    
    return output_path

def create_zoomed_chart(inverted, spy, available, colors):
    """Create zoomed chart for 2024-2025."""
    
    # Filter to recent period
    start_date = '2024-01-01'
    inverted_recent = inverted.loc[start_date:]
    spy_recent = spy.loc[start_date:]
    
    if len(inverted_recent) == 0:
        return
    
    # Weekly resample
    inverted_weekly = inverted_recent.resample('W').mean()
    spy_weekly = spy_recent.resample('W').last()
    
    fig, ax1 = plt.subplots(figsize=(14, 7))
    
    # Stacked bars
    bottom = np.zeros(len(inverted_weekly))
    
    for ind in available:
        ax1.bar(inverted_weekly.index, inverted_weekly[ind], 
                bottom=bottom, width=5, 
                label=ind, color=colors.get(ind, 'gray'), alpha=0.7)
        bottom += inverted_weekly[ind].values
    
    ax1.set_ylabel('Warning Level', fontsize=12)
    ax1.set_ylim(0, 100)
    
    # SP500 overlay
    ax2 = ax1.twinx()
    ax2.plot(spy_weekly.index, spy_weekly['value'], 
             color='black', linewidth=2.5, label='SP500')
    ax2.set_ylabel('SP500', fontsize=12)
    
    # Mark June 2025 break
    ax1.axvline(pd.Timestamp('2025-06-27'), color='orange', linestyle='--', 
                linewidth=2, label='June 2025 Break')
    
    ax1.set_title('PRISM Early Warning - 2024-2025 Detail\n(Watch for bars growing before SP500 drops)', 
                  fontsize=13, fontweight='bold')
    
    # Legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    ax1.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    
    plt.tight_layout()
    
    output_path = OUTPUT_DIR / 'early_warning_chart_2024_2025.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved: {output_path}")


def create_line_version():
    """Alternative: Line chart version showing rank evolution."""
    
    print("\nCreating line version...")
    
    mr = load_multiresolution()
    if mr is None:
        return
    
    mr = mr[mr['window_size'] == 63].copy()
    pivot = mr.pivot(index='date', columns='indicator', values='consensus_rank')
    pivot.index = pd.to_datetime(pivot.index)
    
    # Invert (lower rank = more important, so invert for visual)
    warning_indicators = ['T10Y2Y', 'BAMLH0A0HYM2', 'VIXCLS', 'iwm_us']
    available = [ind for ind in warning_indicators if ind in pivot.columns]
    
    spy = load_spy_prices().set_index('date')
    
    # Align
    common_start = max(pivot.index.min(), spy.index.min())
    common_end = min(pivot.index.max(), spy.index.max())
    
    pivot = pivot.loc[common_start:common_end]
    spy = spy.loc[common_start:common_end]
    
    # Smooth with rolling average
    pivot_smooth = pivot[available].rolling(10).mean()
    
    fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True, 
                              gridspec_kw={'height_ratios': [2, 1]})
    
    # Top: SP500
    axes[0].plot(spy.index, spy['value'], color='black', linewidth=1.5)
    axes[0].set_ylabel('SP500', fontsize=12)
    axes[0].set_title('SP500 Price', fontsize=12)
    axes[0].grid(True, alpha=0.3)
    
    # Bottom: Warning indicators (inverted - lower line = more warning)
    colors = ['#FF6B6B', '#4ECDC4', '#FFE66D', '#95E1D3']
    
    for ind, color in zip(available, colors):
        # Invert: rank 1 should be at top (most warning)
        axes[1].plot(pivot_smooth.index, 35 - pivot_smooth[ind], 
                     label=ind, color=color, linewidth=1.5, alpha=0.8)
    
    axes[1].set_ylabel('Warning Level\n(higher = more important)', fontsize=11)
    axes[1].set_xlabel('Date', fontsize=12)
    axes[1].legend(loc='upper left')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0, 35)
    
    # Add threshold line
    axes[1].axhline(22, color='red', linestyle='--', alpha=0.5, label='Danger Zone')
    axes[1].text(pivot.index[10], 23, 'Danger Zone (rank < 12)', fontsize=9, color='red')
    
    # Mark events
    events = [
        ('2020-02-20', 'COVID'),
        ('2022-01-03', '2022 Bear'),
        ('2025-06-27', 'June Break'),
    ]
    
    for date_str, label in events:
        try:
            event_date = pd.Timestamp(date_str)
            if common_start <= event_date <= common_end:
                for ax in axes:
                    ax.axvline(event_date, color='red', linestyle='--', alpha=0.5)
                axes[0].text(event_date, axes[0].get_ylim()[1] * 0.95, label, 
                           ha='center', fontsize=9, color='red')
        except:
            pass
    
    plt.tight_layout()
    
    output_path = OUTPUT_DIR / 'early_warning_lines.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved: {output_path}")


if __name__ == "__main__":
    print("="*60)
    print("📊 PRISM EARLY WARNING CHART GENERATOR")
    print("="*60)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    create_early_warning_chart()
    create_line_version()
    
    print("\n✅ Done! Check output folder.")
