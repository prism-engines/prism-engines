#!/usr/bin/env python3
"""
PRISM Indicator Contribution Analysis
======================================
Shows how much each indicator contributes to overall market coherence.
Visualized as stacked area chart over time.

The "weight" of each indicator = its average absolute correlation with all others
Higher weight = more connected to the rest of the market
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


def compute_indicator_contributions(returns_df, window=63, step=21):
    """
    For each time window, compute each indicator's "connectedness" score.
    
    Connectedness = average absolute correlation with all other indicators
    This shows which indicators are driving the overall coherence.
    """
    
    # Select key indicators
    all_cols = returns_df.columns.tolist()
    
    priority_order = [
        'xlk_us', 'xlf_us', 'xle_us', 'xlv_us', 'xli_us', 
        'xly_us', 'xlp_us', 'xlu_us', 'xlb_us', 'iwm_us', 'gld_us',
        'VIXCLS', 'DGS10', 'DGS2', 'T10Y2Y', 'BAMLH0A0HYM2',
        'SP500', 'NASDAQCOM', 'DCOILWTICO', 'DTWEXBGS', 'DFF'
    ]
    
    selected_cols = [c for c in priority_order if c in all_cols]
    returns_subset = returns_df[selected_cols]
    
    # Generate frame indices
    frame_indices = list(range(window, len(returns_subset), step))
    
    # Compute contributions over time
    contributions = {col: [] for col in selected_cols}
    coherence_values = []
    dates = []
    
    print(f"  Computing contributions for {len(selected_cols)} indicators...")
    print(f"  Processing {len(frame_indices)} time windows...")
    
    for i in frame_indices:
        window_data = returns_subset.iloc[i-window:i]
        corr = window_data.corr()
        
        # For each indicator, compute its average |correlation| with others
        for col in selected_cols:
            if col in corr.columns:
                # Get correlations with all other indicators
                col_corrs = corr[col].drop(col)  # exclude self-correlation
                avg_abs_corr = col_corrs.abs().mean()
                contributions[col].append(avg_abs_corr if not np.isnan(avg_abs_corr) else 0)
            else:
                contributions[col].append(0)
        
        # Overall coherence
        upper_tri = corr.values[np.triu_indices(len(corr), k=1)]
        valid_corrs = upper_tri[~np.isnan(upper_tri)]
        coherence = np.mean(np.abs(valid_corrs)) if len(valid_corrs) > 0 else 0
        coherence_values.append(coherence)
        dates.append(returns_subset.index[i])
    
    # Convert to DataFrame
    contrib_df = pd.DataFrame(contributions, index=dates)
    
    return contrib_df, coherence_values, dates


def create_contribution_chart(contrib_df, coherence_values, dates, output_dir):
    """Create stacked area chart of indicator contributions."""
    
    # Group indicators by type for coloring
    groups = {
        'Sectors': ['xlk_us', 'xlf_us', 'xle_us', 'xlv_us', 'xli_us', 
                   'xly_us', 'xlp_us', 'xlu_us', 'xlb_us', 'iwm_us'],
        'Safe Haven': ['gld_us', 'VIXCLS'],
        'Rates': ['DGS10', 'DGS2', 'T10Y2Y', 'DFF'],
        'Credit': ['BAMLH0A0HYM2'],
        'Indices': ['SP500', 'NASDAQCOM'],
        'Commodities': ['DCOILWTICO'],
        'FX': ['DTWEXBGS']
    }
    
    # Assign colors
    group_colors = {
        'Sectors': plt.cm.Reds(np.linspace(0.3, 0.9, 10)),
        'Safe Haven': ['gold', 'purple'],
        'Rates': plt.cm.Blues(np.linspace(0.4, 0.8, 4)),
        'Credit': ['orange'],
        'Indices': ['darkgreen', 'limegreen'],
        'Commodities': ['brown'],
        'FX': ['teal']
    }
    
    # Create color mapping
    colors = {}
    for group, cols in groups.items():
        group_cols = group_colors[group]
        for i, col in enumerate(cols):
            if col in contrib_df.columns:
                if isinstance(group_cols, np.ndarray):
                    colors[col] = group_cols[i % len(group_cols)]
                else:
                    colors[col] = group_cols[i % len(group_cols)]
    
    # Sort columns by average contribution (most influential first)
    avg_contrib = contrib_df.mean().sort_values(ascending=False)
    sorted_cols = avg_contrib.index.tolist()
    
    # Create figure with multiple subplots
    fig, axes = plt.subplots(3, 1, figsize=(16, 14), 
                             gridspec_kw={'height_ratios': [2, 1.5, 1]})
    
    # ===== Plot 1: Stacked Area Chart =====
    ax1 = axes[0]
    
    # Normalize contributions so they sum to 1 at each time point
    contrib_normalized = contrib_df[sorted_cols].div(contrib_df[sorted_cols].sum(axis=1), axis=0)
    
    # Stack the areas
    color_list = [colors.get(col, 'gray') for col in sorted_cols]
    
    ax1.stackplot(dates, contrib_normalized.T, labels=sorted_cols, colors=color_list, alpha=0.8)
    
    ax1.set_ylabel('Contribution Share', fontsize=11)
    ax1.set_title('Indicator Contribution to Market Coherence Over Time\n(Stacked by relative influence)', 
                  fontsize=13, fontweight='bold')
    ax1.set_xlim(dates[0], dates[-1])
    ax1.set_ylim(0, 1)
    
    # Legend outside
    ax1.legend(loc='center left', bbox_to_anchor=(1.01, 0.5), fontsize=8, ncol=1)
    
    ax1.xaxis.set_major_locator(mdates.YearLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax1.grid(True, alpha=0.3, axis='y')
    
    # ===== Plot 2: Raw Contributions (not normalized) =====
    ax2 = axes[1]
    
    # Show top 5 most influential indicators as lines
    top_5 = sorted_cols[:5]
    
    for col in top_5:
        ax2.plot(dates, contrib_df[col], label=col, linewidth=1.5, alpha=0.8)
    
    ax2.set_ylabel('Connectedness Score\n(Avg |correlation| with others)', fontsize=10)
    ax2.set_title('Top 5 Most Connected Indicators', fontsize=11, fontweight='bold')
    ax2.legend(loc='upper left', fontsize=9)
    ax2.set_xlim(dates[0], dates[-1])
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_locator(mdates.YearLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    
    # ===== Plot 3: Overall Coherence =====
    ax3 = axes[2]
    
    ax3.fill_between(dates, coherence_values, alpha=0.4, color='navy')
    ax3.plot(dates, coherence_values, 'navy', linewidth=1)
    
    # Threshold line
    mean_coh = np.mean(coherence_values)
    std_coh = np.std(coherence_values)
    threshold = mean_coh + 1.5 * std_coh
    
    ax3.axhline(y=threshold, color='red', linestyle='--', alpha=0.7, label=f'Alert threshold ({threshold:.2f})')
    ax3.axhline(y=mean_coh, color='green', linestyle=':', alpha=0.7, label=f'Mean ({mean_coh:.2f})')
    
    ax3.set_ylabel('Overall Coherence', fontsize=10)
    ax3.set_xlabel('Date', fontsize=11)
    ax3.set_title('Total Market Coherence', fontsize=11, fontweight='bold')
    ax3.legend(loc='upper left', fontsize=9)
    ax3.set_xlim(dates[0], dates[-1])
    ax3.grid(True, alpha=0.3)
    ax3.xaxis.set_major_locator(mdates.YearLocator())
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    
    # Add event annotations
    major_events = {
        '2016-02-11': 'Oil/China\nPanic',
        '2018-02-05': 'Vol-\nmageddon',
        '2018-12-24': 'Fed\nPanic',
        '2020-03-23': 'COVID\nCrash',
        '2022-06-16': 'Rate\nShock',
    }
    
    for date_str, label in major_events.items():
        try:
            event_date = pd.Timestamp(date_str)
            if dates[0] <= event_date <= dates[-1]:
                ax3.annotate(label, xy=(event_date, threshold * 0.95), 
                           fontsize=7, ha='center', alpha=0.8,
                           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
        except:
            pass
    
    plt.tight_layout()
    
    # Save
    fig.savefig(output_dir / "indicator_contributions.png", dpi=150, bbox_inches='tight')
    print(f"💾 Saved: {output_dir / 'indicator_contributions.png'}")
    
    plt.close()
    
    return avg_contrib


def create_heatmap_over_time(contrib_df, output_dir):
    """Create heatmap showing each indicator's contribution over time."""
    
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Transpose so indicators are rows, time is columns
    # Resample to monthly for readability
    monthly = contrib_df.resample('M').mean()
    
    # Sort by average contribution
    sorted_cols = monthly.mean().sort_values(ascending=False).index
    data = monthly[sorted_cols].T
    
    # Create heatmap
    im = ax.imshow(data.values, aspect='auto', cmap='YlOrRd', interpolation='nearest')
    
    # Labels
    ax.set_yticks(range(len(sorted_cols)))
    ax.set_yticklabels([c.replace('_us', '').upper() if '_us' in c else c for c in sorted_cols], fontsize=9)
    
    # X-axis: show years
    n_months = len(data.columns)
    year_positions = []
    year_labels = []
    for i, date in enumerate(data.columns):
        if date.month == 1:  # January
            year_positions.append(i)
            year_labels.append(date.year)
    
    ax.set_xticks(year_positions)
    ax.set_xticklabels(year_labels, fontsize=10)
    
    ax.set_xlabel('Year', fontsize=11)
    ax.set_ylabel('Indicator', fontsize=11)
    ax.set_title('Indicator Connectedness Over Time\n(Brighter = More correlated with market)', 
                fontsize=13, fontweight='bold')
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Connectedness Score', fontsize=10)
    
    plt.tight_layout()
    
    fig.savefig(output_dir / "contribution_heatmap.png", dpi=150, bbox_inches='tight')
    print(f"💾 Saved: {output_dir / 'contribution_heatmap.png'}")
    
    plt.close()


def main():
    print("=" * 70)
    print("PRISM INDICATOR CONTRIBUTION ANALYSIS")
    print("=" * 70)
    print("\nAnalyzing which indicators drive market coherence over time...")
    
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
    
    # Compute returns
    returns = clean_panel.pct_change().dropna()
    returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
    print(f"  Return observations: {len(returns)}")
    
    # Compute contributions
    print("\n🔬 Computing indicator contributions...")
    contrib_df, coherence_values, dates = compute_indicator_contributions(
        returns, window=63, step=21
    )
    
    # Create output directory
    output_dir = OUTPUT_DIR / "contributions"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create visualizations
    print("\n🎨 Creating visualizations...")
    
    avg_contrib = create_contribution_chart(contrib_df, coherence_values, dates, output_dir)
    create_heatmap_over_time(contrib_df, output_dir)
    
    # Print rankings
    print("\n📊 INDICATOR INFLUENCE RANKINGS (10-year average)")
    print("-" * 50)
    for i, (indicator, score) in enumerate(avg_contrib.head(10).items()):
        label = indicator.replace('_us', '').upper() if '_us' in indicator else indicator
        bar = "█" * int(score * 50)
        print(f"  {i+1:2}. {label:12} {score:.3f} {bar}")
    
    # Save data
    contrib_df.to_csv(output_dir / "contributions_data.csv")
    print(f"\n💾 Saved data: {output_dir / 'contributions_data.csv'}")
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"\nOutputs in: {output_dir}")


if __name__ == "__main__":
    main()