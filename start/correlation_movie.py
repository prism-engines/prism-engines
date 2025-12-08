#!/usr/bin/env python3
"""
PRISM Correlation Heatmap Animation
====================================
Creates a movie showing how correlations evolve over time.
Watch the market "breathe" - expanding and contracting correlations.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import sqlite3
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import TwoSlopeNorm
import seaborn as sns
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
    """Load all data from database using unified indicator_values table."""
    from data.sql.db_connector import load_all_indicators_wide

    panel = load_all_indicators_wide()

    if panel.empty:
        import warnings
        warnings.warn(
            "indicator_values table is empty. Data may need migration.",
            DeprecationWarning
        )
        return pd.DataFrame()

    panel.index = pd.to_datetime(panel.index)
    return panel.sort_index()


def create_correlation_movie(returns_df, window=63, step=21, output_path=None):
    """
    Create animated heatmap of rolling correlations.
    
    Args:
        returns_df: DataFrame of returns
        window: Rolling window size (63 = ~3 months)
        step: Days between frames (21 = ~1 month)
        output_path: Where to save the movie
    """
    
    # Select key indicators for readability (max ~20)
    all_cols = returns_df.columns.tolist()
    
    # Prioritize: sectors, then key economic indicators
    priority_order = [
        # Sectors
        'xlk_us', 'xlf_us', 'xle_us', 'xlv_us', 'xli_us', 
        'xly_us', 'xlp_us', 'xlu_us', 'xlb_us', 'iwm_us', 'gld_us',
        # Key economic
        'VIXCLS', 'DGS10', 'DGS2', 'T10Y2Y', 'BAMLH0A0HYM2',
        'SP500', 'NASDAQCOM', 'DCOILWTICO', 'DTWEXBGS', 'DFF'
    ]
    
    selected_cols = [c for c in priority_order if c in all_cols]
    
    # Add any remaining columns up to 20
    for c in all_cols:
        if c not in selected_cols and len(selected_cols) < 20:
            selected_cols.append(c)
    
    returns_subset = returns_df[selected_cols]
    
    # Generate frame indices
    frame_indices = list(range(window, len(returns_subset), step))
    
    print(f"  Creating {len(frame_indices)} frames...")
    print(f"  Using {len(selected_cols)} indicators")
    
    # Pre-compute all correlation matrices
    print("  Pre-computing correlation matrices...")
    corr_matrices = []
    coherence_values = []
    dates = []
    
    for i in frame_indices:
        window_data = returns_subset.iloc[i-window:i]
        corr = window_data.corr()
        corr_matrices.append(corr)
        
        # Compute coherence for this frame
        upper_tri = corr.values[np.triu_indices(len(corr), k=1)]
        valid_corrs = upper_tri[~np.isnan(upper_tri)]
        coherence = np.mean(np.abs(valid_corrs)) if len(valid_corrs) > 0 else 0
        coherence_values.append(coherence)
        dates.append(returns_subset.index[i])
    
    # Create figure with two subplots: heatmap and coherence timeline
    fig = plt.figure(figsize=(14, 10))
    
    # Heatmap subplot (larger)
    ax1 = fig.add_axes([0.1, 0.25, 0.75, 0.65])
    
    # Coherence timeline subplot (smaller, below)
    ax2 = fig.add_axes([0.1, 0.08, 0.75, 0.12])
    
    # Colorbar axes
    cbar_ax = fig.add_axes([0.88, 0.25, 0.02, 0.65])
    
    # Plot coherence timeline (static background)
    coherence_series = pd.Series(coherence_values, index=dates)
    ax2.fill_between(dates, coherence_values, alpha=0.3, color='blue')
    ax2.plot(dates, coherence_values, 'b-', linewidth=0.5, alpha=0.5)
    ax2.set_xlim(dates[0], dates[-1])
    ax2.set_ylim(0, max(coherence_values) * 1.1)
    ax2.set_ylabel('Coherence', fontsize=8)
    ax2.set_xlabel('Date', fontsize=8)
    ax2.tick_params(labelsize=7)
    
    # Vertical line for current position (will be updated)
    vline = ax2.axvline(x=dates[0], color='red', linewidth=2, alpha=0.8)
    
    # Initialize heatmap
    initial_corr = corr_matrices[0]
    
    # Create short labels
    short_labels = []
    for col in selected_cols:
        if col.endswith('_us'):
            short_labels.append(col.replace('_us', '').upper())
        elif col.startswith('econ_'):
            short_labels.append(col.replace('econ_', ''))
        else:
            short_labels.append(col[:8])
    
    # Initial heatmap
    mask = np.triu(np.ones_like(initial_corr, dtype=bool), k=1)
    
    heatmap = sns.heatmap(
        initial_corr,
        mask=mask,
        annot=False,
        cmap='RdBu_r',
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        ax=ax1,
        cbar_ax=cbar_ax,
        xticklabels=short_labels,
        yticklabels=short_labels
    )
    
    ax1.tick_params(labelsize=7)
    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
    plt.setp(ax1.get_yticklabels(), rotation=0)
    
    title = ax1.set_title(
        f'Correlation Matrix - {dates[0].strftime("%Y-%m-%d")}\nCoherence: {coherence_values[0]:.3f}',
        fontsize=12, fontweight='bold'
    )
    
    def update(frame_num):
        """Update function for animation."""
        ax1.clear()
        
        corr = corr_matrices[frame_num]
        date = dates[frame_num]
        coherence = coherence_values[frame_num]
        
        # Determine color intensity based on coherence
        # Higher coherence = more intense colors
        
        sns.heatmap(
            corr,
            mask=mask,
            annot=False,
            cmap='RdBu_r',
            center=0,
            vmin=-1,
            vmax=1,
            square=True,
            linewidths=0.5,
            ax=ax1,
            cbar=False,
            xticklabels=short_labels,
            yticklabels=short_labels
        )
        
        ax1.tick_params(labelsize=7)
        plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
        plt.setp(ax1.get_yticklabels(), rotation=0)
        
        # Color-code the title based on coherence
        if coherence > 0.45:
            title_color = 'red'
            status = '🚨 HIGH COHERENCE'
        elif coherence > 0.38:
            title_color = 'orange'
            status = '⚠️ ELEVATED'
        else:
            title_color = 'black'
            status = ''
        
        ax1.set_title(
            f'Correlation Matrix - {date.strftime("%Y-%m-%d")} {status}\nCoherence: {coherence:.3f}',
            fontsize=12, fontweight='bold', color=title_color
        )
        
        # Update vertical line position
        vline.set_xdata([date, date])
        
        return [ax1, vline]
    
    print("  Rendering animation...")
    
    # Create animation
    anim = animation.FuncAnimation(
        fig, update, frames=len(frame_indices),
        interval=150,  # milliseconds between frames
        blit=False
    )
    
    # Save as MP4
    if output_path:
        print(f"  Saving to {output_path}...")
        try:
            writer = animation.FFMpegWriter(fps=7, bitrate=2000)
            anim.save(output_path, writer=writer)
            print(f"  ✅ Saved MP4: {output_path}")
        except Exception as e:
            print(f"  ⚠️ FFmpeg not available, trying GIF...")
            gif_path = str(output_path).replace('.mp4', '.gif')
            try:
                anim.save(gif_path, writer='pillow', fps=5)
                print(f"  ✅ Saved GIF: {gif_path}")
            except Exception as e2:
                print(f"  ❌ Could not save animation: {e2}")
                # Save individual frames instead
                print("  📸 Saving key frames instead...")
                save_key_frames(corr_matrices, dates, coherence_values, 
                              selected_cols, short_labels, output_path.parent)
    
    plt.close()
    return coherence_series


def save_key_frames(corr_matrices, dates, coherence_values, cols, labels, output_dir):
    """Save key frames as individual images."""
    
    # Find interesting frames: max coherence, min coherence, and regular intervals
    coherence_arr = np.array(coherence_values)
    
    key_indices = [
        0,  # First
        len(dates) - 1,  # Last
        np.argmax(coherence_arr),  # Max coherence
        np.argmin(coherence_arr),  # Min coherence
    ]
    
    # Add quarterly samples
    quarterly = list(range(0, len(dates), len(dates) // 8))
    key_indices.extend(quarterly)
    
    key_indices = sorted(set(key_indices))
    
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(exist_ok=True)
    
    for idx in key_indices:
        fig, ax = plt.subplots(figsize=(12, 10))
        
        corr = corr_matrices[idx]
        date = dates[idx]
        coherence = coherence_values[idx]
        
        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
        
        sns.heatmap(
            corr,
            mask=mask,
            annot=True,
            fmt='.2f',
            annot_kws={'size': 6},
            cmap='RdBu_r',
            center=0,
            vmin=-1,
            vmax=1,
            square=True,
            linewidths=0.5,
            ax=ax,
            xticklabels=labels,
            yticklabels=labels
        )
        
        ax.tick_params(labelsize=8)
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        status = '🚨 HIGH' if coherence > 0.45 else '⚠️ ELEVATED' if coherence > 0.38 else ''
        ax.set_title(
            f'Correlation Matrix - {date.strftime("%Y-%m-%d")} {status}\nCoherence: {coherence:.3f}',
            fontsize=12, fontweight='bold'
        )
        
        plt.tight_layout()
        fig.savefig(frames_dir / f"corr_{date.strftime('%Y%m%d')}.png", dpi=100)
        plt.close()
        print(f"    Saved frame: {date.strftime('%Y-%m-%d')} (coherence: {coherence:.3f})")


def main():
    print("=" * 70)
    print("PRISM CORRELATION HEATMAP ANIMATION")
    print("=" * 70)
    print("\nCreating movie of correlation evolution over time...")
    
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
    
    # Create output directory
    output_dir = OUTPUT_DIR / "correlation_movie"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create animation
    print("\n🎬 Creating correlation movie...")
    print("  Window: 63 days (quarterly)")
    print("  Step: 21 days (monthly frames)")
    
    coherence = create_correlation_movie(
        returns,
        window=63,
        step=21,
        output_path=output_dir / "correlation_evolution.mp4"
    )
    
    print("\n" + "=" * 70)
    print("ANIMATION COMPLETE")
    print("=" * 70)
    print(f"\nOutput: {output_dir}")


if __name__ == "__main__":
    main()