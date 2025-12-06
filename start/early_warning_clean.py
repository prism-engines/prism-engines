#!/usr/bin/env python3
"""
PRISM Early Warning Chart - Clean Version
==========================================

Multiple cleaner visualization options:
1. Traffic Light Gauge - simple red/yellow/green
2. Multi-panel with SP500 on top, indicators below
3. Composite "Danger Score" single line

Usage:
    python early_warning_clean.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
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
    """Load SP500 prices from database."""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("""
        SELECT date, value 
        FROM econ_values 
        WHERE series_id = 'SP500'
        ORDER BY date
    """, conn)
    conn.close()
    df['date'] = pd.to_datetime(df['date'])
    return df


def create_multi_panel_chart():
    """
    Clean multi-panel design:
    - Top: SP500 with shaded danger zones
    - Bottom panels: Each indicator as its own clean line
    """
    
    print("Creating multi-panel chart...")
    
    mr = load_multiresolution()
    if mr is None:
        print("Error: multiresolution.csv not found")
        return
    
    mr = mr[mr['window_size'] == 63].copy()
    pivot = mr.pivot(index='date', columns='indicator', values='consensus_rank')
    pivot.index = pd.to_datetime(pivot.index)
    pivot = pivot.sort_index()
    
    # Load SP500
    spy = load_spy_prices().set_index('date')
    
    # Align dates
    common_start = max(pivot.index.min(), spy.index.min())
    common_end = min(pivot.index.max(), spy.index.max())
    pivot = pivot.loc[common_start:common_end]
    spy = spy.loc[common_start:common_end]
    
    # Smooth data
    pivot_smooth = pivot.rolling(5).mean()
    
    # Focus on 2024-2025
    start = '2023-06-01'
    pivot_smooth = pivot_smooth.loc[start:]
    spy = spy.loc[start:]
    
    # Indicators (in order of lead time)
    indicators = [
        ('T10Y2Y', 'Yield Curve (10Y-2Y)', '#E74C3C'),      # Red
        ('BAMLH0A0HYM2', 'High Yield Credit Spread', '#3498DB'),  # Blue
        ('VIXCLS', 'VIX', '#F39C12'),                        # Orange
        ('iwm_us', 'Small Caps (IWM)', '#27AE60'),           # Green
    ]
    
    available = [(code, name, color) for code, name, color in indicators 
                 if code in pivot_smooth.columns]
    
    # Create figure
    fig, axes = plt.subplots(len(available) + 1, 1, figsize=(14, 12), 
                              sharex=True, 
                              gridspec_kw={'height_ratios': [2] + [1]*len(available)})
    
    # Top panel: SP500
    ax_spy = axes[0]
    ax_spy.plot(spy.index, spy['value'], color='black', linewidth=2)
    ax_spy.fill_between(spy.index, spy['value'].min(), spy['value'], alpha=0.1, color='blue')
    ax_spy.set_ylabel('SP500', fontsize=11, fontweight='bold')
    ax_spy.set_title('PRISM Early Warning System\n(Lower rank = Higher importance = More warning)', 
                     fontsize=14, fontweight='bold')
    ax_spy.grid(True, alpha=0.3)
    
    # Mark events
    events = [('2025-06-27', 'June Break')]
    for date_str, label in events:
        try:
            event_date = pd.Timestamp(date_str)
            ax_spy.axvline(event_date, color='red', linestyle='--', linewidth=2, alpha=0.7)
            ax_spy.text(event_date, ax_spy.get_ylim()[1], f' {label}', 
                       fontsize=10, color='red', va='top')
        except:
            pass
    
    # Bottom panels: Each indicator
    danger_threshold = 12  # Rank below this = danger
    
    for i, (code, name, color) in enumerate(available):
        ax = axes[i + 1]
        
        data = pivot_smooth[code]
        
        # Plot rank (inverted y-axis so lower rank = higher on chart)
        ax.plot(data.index, data, color=color, linewidth=2)
        
        # Fill danger zone (when rank < threshold)
        ax.fill_between(data.index, danger_threshold, data, 
                        where=(data < danger_threshold),
                        alpha=0.3, color='red', label='Danger Zone')
        
        # Threshold line
        ax.axhline(danger_threshold, color='red', linestyle=':', alpha=0.5)
        
        # Invert y-axis so lower rank (more important) is at TOP
        ax.invert_yaxis()
        ax.set_ylim(30, 5)  # Rank 30 at bottom, rank 5 at top
        
        ax.set_ylabel(name, fontsize=9, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add current value annotation
        current_rank = data.iloc[-1]
        ax.annotate(f'{current_rank:.0f}', 
                   xy=(data.index[-1], current_rank),
                   xytext=(10, 0), textcoords='offset points',
                   fontsize=10, fontweight='bold', color=color)
    
    axes[-1].set_xlabel('Date', fontsize=11)
    axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    
    plt.tight_layout()
    
    output_path = OUTPUT_DIR / 'early_warning_panels.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved: {output_path}")


def create_composite_danger_score():
    """
    Single "Danger Score" combining all indicators.
    Higher score = more danger.
    """
    
    print("Creating composite danger score chart...")
    
    mr = load_multiresolution()
    if mr is None:
        return
    
    mr = mr[mr['window_size'] == 63].copy()
    pivot = mr.pivot(index='date', columns='indicator', values='consensus_rank')
    pivot.index = pd.to_datetime(pivot.index)
    pivot = pivot.sort_index()
    
    spy = load_spy_prices().set_index('date')
    
    # Align
    common_start = max(pivot.index.min(), spy.index.min())
    common_end = min(pivot.index.max(), spy.index.max())
    pivot = pivot.loc[common_start:common_end]
    spy = spy.loc[common_start:common_end]
    
    # Calculate composite danger score
    # Invert ranks and weight by lead time importance
    indicators = {
        'T10Y2Y': 1.5,        # Highest weight - longest lead
        'BAMLH0A0HYM2': 1.3,  # Credit
        'VIXCLS': 1.0,        # VIX
        'iwm_us': 0.8,        # Small caps - shortest lead
    }
    
    max_rank = 33
    danger_score = pd.Series(0.0, index=pivot.index)
    
    for ind, weight in indicators.items():
        if ind in pivot.columns:
            # Invert: low rank = high danger
            inverted = (max_rank - pivot[ind]) / max_rank * 100
            danger_score += inverted * weight
    
    # Normalize to 0-100
    total_weight = sum(indicators.values())
    danger_score = danger_score / total_weight
    
    # Smooth
    danger_smooth = danger_score.rolling(10).mean()
    
    # Create figure
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                              gridspec_kw={'height_ratios': [1.5, 1]})
    
    # Top: SP500
    ax1 = axes[0]
    ax1.plot(spy.index, spy['value'], color='black', linewidth=1.5)
    ax1.set_ylabel('SP500', fontsize=12, fontweight='bold')
    ax1.set_title('PRISM Danger Score vs SP500\n(Higher score = More warning signs)', 
                  fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Bottom: Danger score with gradient fill
    ax2 = axes[1]
    
    # Create color gradient based on score
    for i in range(len(danger_smooth) - 1):
        score = danger_smooth.iloc[i]
        if pd.isna(score):
            continue
            
        # Color based on score
        if score < 50:
            color = '#27AE60'  # Green
        elif score < 65:
            color = '#F39C12'  # Yellow
        elif score < 75:
            color = '#E67E22'  # Orange
        else:
            color = '#E74C3C'  # Red
        
        ax2.fill_between([danger_smooth.index[i], danger_smooth.index[i+1]], 
                        0, [score, danger_smooth.iloc[i+1]],
                        color=color, alpha=0.7)
    
    ax2.plot(danger_smooth.index, danger_smooth, color='black', linewidth=1)
    
    # Threshold lines
    ax2.axhline(50, color='green', linestyle='--', alpha=0.5, linewidth=1)
    ax2.axhline(65, color='orange', linestyle='--', alpha=0.5, linewidth=1)
    ax2.axhline(75, color='red', linestyle='--', alpha=0.5, linewidth=1)
    
    # Labels
    ax2.text(danger_smooth.index[50], 48, 'Safe', fontsize=9, color='green')
    ax2.text(danger_smooth.index[50], 63, 'Caution', fontsize=9, color='orange')
    ax2.text(danger_smooth.index[50], 73, 'Warning', fontsize=9, color='red')
    ax2.text(danger_smooth.index[50], 83, 'DANGER', fontsize=9, color='darkred', fontweight='bold')
    
    ax2.set_ylabel('Danger Score', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Date', fontsize=11)
    ax2.set_ylim(30, 90)
    ax2.grid(True, alpha=0.3)
    
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
                    ax.axvline(event_date, color='red', linestyle='--', alpha=0.6, linewidth=1.5)
                ax1.text(event_date, ax1.get_ylim()[1] * 0.98, f' {label}', 
                        fontsize=9, color='red', va='top')
        except:
            pass
    
    # Current score annotation
    current = danger_smooth.iloc[-1]
    ax2.annotate(f'Current: {current:.0f}', 
                xy=(danger_smooth.index[-1], current),
                xytext=(-60, 20), textcoords='offset points',
                fontsize=11, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='black'))
    
    plt.tight_layout()
    
    output_path = OUTPUT_DIR / 'danger_score.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved: {output_path}")


def create_traffic_light_gauge():
    """
    Simple traffic light gauge showing current status.
    """
    
    print("Creating traffic light gauge...")
    
    mr = load_multiresolution()
    if mr is None:
        return
    
    mr = mr[mr['window_size'] == 63].copy()
    
    # Get most recent readings
    latest_date = mr['date'].max()
    latest = mr[mr['date'] == latest_date].set_index('indicator')['consensus_rank']
    
    indicators = {
        'T10Y2Y': ('Yield Curve', 1.5),
        'BAMLH0A0HYM2': ('Credit Spreads', 1.3),
        'VIXCLS': ('VIX', 1.0),
        'iwm_us': ('Small Caps', 0.8),
    }
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    y_positions = list(range(len(indicators)))
    
    for i, (code, (name, weight)) in enumerate(indicators.items()):
        if code not in latest.index:
            continue
            
        rank = latest[code]
        
        # Determine color
        if rank < 10:
            color = '#E74C3C'  # Red - danger
            status = 'DANGER'
        elif rank < 15:
            color = '#F39C12'  # Yellow - caution
            status = 'CAUTION'
        else:
            color = '#27AE60'  # Green - safe
            status = 'SAFE'
        
        # Draw bar
        bar_length = (33 - rank) / 33 * 100  # Invert so higher = more warning
        ax.barh(i, bar_length, color=color, alpha=0.7, height=0.6)
        
        # Labels
        ax.text(-5, i, f'{name}', ha='right', va='center', fontsize=11, fontweight='bold')
        ax.text(bar_length + 2, i, f'Rank {rank:.0f} ({status})', 
                ha='left', va='center', fontsize=10, color=color, fontweight='bold')
    
    ax.set_xlim(-40, 110)
    ax.set_ylim(-0.5, len(indicators) - 0.5)
    ax.set_xlabel('Warning Level (higher = more important = more danger)', fontsize=11)
    ax.set_title(f'PRISM Early Warning Status\n{latest_date}', fontsize=14, fontweight='bold')
    
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    # Add threshold lines
    safe_thresh = (33 - 15) / 33 * 100
    danger_thresh = (33 - 10) / 33 * 100
    
    ax.axvline(safe_thresh, color='orange', linestyle='--', alpha=0.5)
    ax.axvline(danger_thresh, color='red', linestyle='--', alpha=0.5)
    
    ax.text(safe_thresh, len(indicators) - 0.3, 'Caution\nThreshold', 
            ha='center', fontsize=8, color='orange')
    ax.text(danger_thresh, len(indicators) - 0.3, 'Danger\nThreshold', 
            ha='center', fontsize=8, color='red')
    
    plt.tight_layout()
    
    output_path = OUTPUT_DIR / 'warning_gauge.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved: {output_path}")


if __name__ == "__main__":
    print("=" * 60)
    print("📊 PRISM EARLY WARNING - CLEAN CHARTS")
    print("=" * 60)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    create_multi_panel_chart()
    create_composite_danger_score()
    create_traffic_light_gauge()
    
    print("\n✅ Done! Check output folder.")
