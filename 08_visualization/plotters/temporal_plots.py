"""
Temporal Visualization - Plot how indicator rankings change over time
=====================================================================

Visualizations for temporal PRISM analysis:
- Ranking evolution line charts
- Bump charts (rank trajectories)
- Heatmaps of rank changes
- Animation/recording support

Usage:
    from temporal_plots import (
        plot_ranking_evolution,
        plot_bump_chart,
        plot_rank_heatmap,
        create_animation_frames
    )

    # After running temporal analysis
    plot_ranking_evolution(results, top_n=10)
    plot_bump_chart(results, indicators=['SPY', 'VIX', 'DXY'])
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path


def plot_ranking_evolution(
    results: Dict,
    top_n: int = 10,
    figsize: Tuple[int, int] = (14, 8),
    save_path: Optional[str] = None,
    title: str = "Indicator Ranking Evolution Over Time"
) -> None:
    """
    Plot line chart showing how top indicator rankings change over time.

    Args:
        results: Output from TemporalPRISM.run_rolling_analysis()
        top_n: Number of top indicators to show
        figsize: Figure size
        save_path: Optional path to save figure
        title: Chart title
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
    except ImportError:
        print("matplotlib required for plotting")
        return

    # Build ranking DataFrame
    rank_df = pd.DataFrame(results['rankings'], index=results['timestamps'])

    # Get indicators that were in top N at any point
    ever_top_n = rank_df.min().nsmallest(top_n).index.tolist()

    fig, ax = plt.subplots(figsize=figsize)

    # Color palette
    colors = plt.cm.tab10(np.linspace(0, 1, len(ever_top_n)))

    for i, indicator in enumerate(ever_top_n):
        ax.plot(
            rank_df.index,
            rank_df[indicator],
            label=indicator,
            color=colors[i],
            linewidth=2,
            alpha=0.8
        )

    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Rank (lower = more important)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')

    # Invert y-axis (rank 1 at top)
    ax.invert_yaxis()

    # Format x-axis dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45, ha='right')

    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
        plt.close()
    else:
        plt.show()


def plot_bump_chart(
    results: Dict,
    indicators: Optional[List[str]] = None,
    top_n: int = 15,
    figsize: Tuple[int, int] = (16, 10),
    save_path: Optional[str] = None
) -> None:
    """
    Create a bump chart showing rank trajectories.

    Bump charts clearly show when indicators overtake each other.

    Args:
        results: Output from TemporalPRISM.run_rolling_analysis()
        indicators: Specific indicators to show (if None, uses top_n)
        top_n: Number of top indicators if indicators not specified
        figsize: Figure size
        save_path: Optional path to save figure
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib required for plotting")
        return

    rank_df = pd.DataFrame(results['rankings'], index=results['timestamps'])

    # Select indicators
    if indicators is None:
        indicators = rank_df.min().nsmallest(top_n).index.tolist()

    # Filter to selected indicators
    rank_df = rank_df[indicators]

    fig, ax = plt.subplots(figsize=figsize)

    # Color palette
    n_colors = len(indicators)
    colors = plt.cm.Set2(np.linspace(0, 1, min(n_colors, 8)))
    if n_colors > 8:
        colors = plt.cm.tab20(np.linspace(0, 1, n_colors))

    # Plot each indicator
    x_positions = np.arange(len(rank_df))

    for i, indicator in enumerate(indicators):
        ranks = rank_df[indicator].values

        # Plot line
        ax.plot(x_positions, ranks, color=colors[i % len(colors)],
                linewidth=2.5, alpha=0.7)

        # Add dots at each point
        ax.scatter(x_positions, ranks, color=colors[i % len(colors)],
                   s=50, zorder=5)

        # Label at the end
        ax.annotate(
            indicator,
            xy=(x_positions[-1], ranks[-1]),
            xytext=(5, 0),
            textcoords='offset points',
            fontsize=9,
            fontweight='bold',
            color=colors[i % len(colors)],
            va='center'
        )

    ax.set_ylabel('Rank', fontsize=12)
    ax.set_title('Indicator Rank Trajectories (Bump Chart)', fontsize=14, fontweight='bold')

    # Invert y-axis
    ax.invert_yaxis()

    # X-axis labels (show every nth date)
    n_labels = min(10, len(rank_df))
    label_positions = np.linspace(0, len(rank_df) - 1, n_labels, dtype=int)
    ax.set_xticks(label_positions)
    ax.set_xticklabels(
        [rank_df.index[i].strftime('%Y-%m') for i in label_positions],
        rotation=45, ha='right'
    )

    ax.grid(True, alpha=0.3, axis='y')

    # Extend x-axis for labels
    ax.set_xlim(-0.5, len(rank_df) + 2)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
        plt.close()
    else:
        plt.show()


def plot_rank_heatmap(
    results: Dict,
    top_n: int = 20,
    figsize: Tuple[int, int] = (16, 10),
    save_path: Optional[str] = None,
    cmap: str = 'RdYlGn_r'
) -> None:
    """
    Create heatmap showing ranks over time.

    Color intensity shows rank (dark = high rank/important).

    Args:
        results: Output from TemporalPRISM.run_rolling_analysis()
        top_n: Number of indicators to show
        figsize: Figure size
        save_path: Optional path to save figure
        cmap: Colormap (default reversed RdYlGn - green=good rank)
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib required for plotting")
        return

    rank_df = pd.DataFrame(results['rankings'], index=results['timestamps'])

    # Select top indicators (by average rank)
    avg_rank = rank_df.mean().sort_values()
    top_indicators = avg_rank.head(top_n).index.tolist()

    # Filter and transpose for heatmap
    heatmap_data = rank_df[top_indicators].T

    fig, ax = plt.subplots(figsize=figsize)

    # Create heatmap
    im = ax.imshow(heatmap_data.values, aspect='auto', cmap=cmap)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, label='Rank (lower = more important)')

    # Labels
    ax.set_yticks(range(len(top_indicators)))
    ax.set_yticklabels(top_indicators, fontsize=9)

    # X-axis: show subset of dates
    n_labels = min(12, len(heatmap_data.columns))
    label_positions = np.linspace(0, len(heatmap_data.columns) - 1, n_labels, dtype=int)
    ax.set_xticks(label_positions)
    ax.set_xticklabels(
        [heatmap_data.columns[i].strftime('%Y-%m') for i in label_positions],
        rotation=45, ha='right'
    )

    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Indicator', fontsize=12)
    ax.set_title('Indicator Rankings Over Time', fontsize=14, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
        plt.close()
    else:
        plt.show()


def plot_rank_changes(
    results: Dict,
    lookback: int = 6,
    top_n: int = 15,
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[str] = None
) -> None:
    """
    Plot recent rank changes as a bar chart.

    Shows which indicators are rising/falling in importance.

    Args:
        results: Output from TemporalPRISM.run_rolling_analysis()
        lookback: Number of periods to look back
        top_n: Number of indicators to show (top movers)
        figsize: Figure size
        save_path: Optional path to save figure
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib required for plotting")
        return

    rank_df = pd.DataFrame(results['rankings'], index=results['timestamps'])

    if len(rank_df) < lookback:
        lookback = len(rank_df)

    recent = rank_df.iloc[-lookback:]

    # Rank change: positive = improved (was lower rank, now higher rank)
    rank_change = recent.iloc[0] - recent.iloc[-1]

    # Get top movers
    top_risers = rank_change.nlargest(top_n // 2)
    top_fallers = rank_change.nsmallest(top_n // 2)
    top_movers = pd.concat([top_risers, top_fallers]).sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=figsize)

    colors = ['green' if x > 0 else 'red' for x in top_movers.values]

    bars = ax.barh(range(len(top_movers)), top_movers.values, color=colors, alpha=0.7)

    ax.set_yticks(range(len(top_movers)))
    ax.set_yticklabels(top_movers.index)

    ax.axvline(x=0, color='black', linewidth=0.5)
    ax.set_xlabel('Rank Change (positive = improving)', fontsize=12)
    ax.set_title(
        f'Indicator Rank Changes (Last {lookback} Periods)\n'
        f'{recent.index[0].strftime("%Y-%m")} to {recent.index[-1].strftime("%Y-%m")}',
        fontsize=14, fontweight='bold'
    )

    # Add value labels
    for bar, val in zip(bars, top_movers.values):
        x_pos = bar.get_width() + (0.2 if val >= 0 else -0.5)
        ax.text(x_pos, bar.get_y() + bar.get_height()/2, f'{val:+.1f}',
                va='center', fontsize=9)

    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
        plt.close()
    else:
        plt.show()


def create_animation_frames(
    results: Dict,
    output_dir: str,
    top_n: int = 15,
    figsize: Tuple[int, int] = (12, 8)
) -> List[str]:
    """
    Create individual frames for an animation showing rankings over time.

    Each frame shows the ranking at that time period as a bar chart.
    Use ffmpeg or similar to combine into video.

    Args:
        results: Output from TemporalPRISM.run_rolling_analysis()
        output_dir: Directory to save frames
        top_n: Number of top indicators per frame
        figsize: Figure size

    Returns:
        List of saved frame paths
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib required for plotting")
        return []

    from pathlib import Path
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamps = results['timestamps']
    n_frames = len(timestamps)
    frame_paths = []

    # Get consistent set of top indicators
    rank_df = pd.DataFrame(results['rankings'], index=timestamps)
    ever_top = rank_df.min().nsmallest(top_n).index.tolist()

    for i, timestamp in enumerate(timestamps):
        fig, ax = plt.subplots(figsize=figsize)

        # Get ranks for this timestamp
        current_ranks = rank_df.loc[timestamp, ever_top].sort_values()

        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(current_ranks)))

        ax.barh(range(len(current_ranks)), current_ranks.values, color=colors)
        ax.set_yticks(range(len(current_ranks)))
        ax.set_yticklabels(current_ranks.index)

        ax.set_xlabel('Rank')
        ax.set_title(f'Indicator Rankings: {timestamp.strftime("%Y-%m-%d")}',
                     fontsize=14, fontweight='bold')

        ax.invert_xaxis()  # Lower rank on right (better)

        plt.tight_layout()

        frame_path = output_path / f'frame_{i:04d}.png'
        plt.savefig(frame_path, dpi=100)
        plt.close()

        frame_paths.append(str(frame_path))

        if (i + 1) % 10 == 0:
            print(f"  Created frame {i + 1}/{n_frames}")

    print(f"Created {len(frame_paths)} frames in {output_dir}")
    print("To create video: ffmpeg -framerate 4 -i frame_%04d.png -c:v libx264 rankings.mp4")

    return frame_paths


def plot_stability_analysis(
    results: Dict,
    top_n: int = 20,
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[str] = None
) -> None:
    """
    Plot indicator stability (rank consistency) vs average rank.

    Helps identify reliable vs volatile indicators.

    Args:
        results: Output from TemporalPRISM.run_rolling_analysis()
        top_n: Number of indicators to show
        figsize: Figure size
        save_path: Optional path to save figure
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib required for plotting")
        return

    rank_df = pd.DataFrame(results['rankings'], index=results['timestamps'])

    avg_rank = rank_df.mean()
    rank_std = rank_df.std()
    stability = 1 / (1 + rank_std)

    # Select top indicators by average rank
    top_indicators = avg_rank.nsmallest(top_n).index.tolist()

    fig, ax = plt.subplots(figsize=figsize)

    x = avg_rank[top_indicators]
    y = stability[top_indicators]

    # Color by stability
    scatter = ax.scatter(x, y, c=y, cmap='RdYlGn', s=100, alpha=0.7)

    # Add labels
    for i, ind in enumerate(top_indicators):
        ax.annotate(ind, (x[ind], y[ind]), fontsize=8,
                    xytext=(5, 5), textcoords='offset points')

    ax.set_xlabel('Average Rank (lower = more important)', fontsize=12)
    ax.set_ylabel('Stability Score (higher = more consistent)', fontsize=12)
    ax.set_title('Indicator Importance vs Stability', fontsize=14, fontweight='bold')

    plt.colorbar(scatter, label='Stability')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
        plt.close()
    else:
        plt.show()


def generate_temporal_report(
    results: Dict,
    output_dir: str,
    include_animation: bool = False
) -> Dict[str, str]:
    """
    Generate a complete set of temporal visualizations.

    Creates all plot types and saves to output directory.

    Args:
        results: Output from TemporalPRISM.run_rolling_analysis()
        output_dir: Directory to save all visualizations
        include_animation: Whether to create animation frames

    Returns:
        Dictionary mapping plot type -> file path
    """
    from pathlib import Path
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    saved_files = {}

    print("Generating temporal visualizations...")

    # 1. Ranking evolution
    print("  1/5 Ranking evolution...")
    path = output_path / 'ranking_evolution.png'
    plot_ranking_evolution(results, save_path=str(path))
    saved_files['ranking_evolution'] = str(path)

    # 2. Bump chart
    print("  2/5 Bump chart...")
    path = output_path / 'bump_chart.png'
    plot_bump_chart(results, save_path=str(path))
    saved_files['bump_chart'] = str(path)

    # 3. Heatmap
    print("  3/5 Rank heatmap...")
    path = output_path / 'rank_heatmap.png'
    plot_rank_heatmap(results, save_path=str(path))
    saved_files['rank_heatmap'] = str(path)

    # 4. Rank changes
    print("  4/5 Rank changes...")
    path = output_path / 'rank_changes.png'
    plot_rank_changes(results, save_path=str(path))
    saved_files['rank_changes'] = str(path)

    # 5. Stability analysis
    print("  5/5 Stability analysis...")
    path = output_path / 'stability_analysis.png'
    plot_stability_analysis(results, save_path=str(path))
    saved_files['stability_analysis'] = str(path)

    # Optional: Animation frames
    if include_animation:
        print("  Creating animation frames...")
        frames_dir = output_path / 'animation_frames'
        create_animation_frames(results, str(frames_dir))
        saved_files['animation_frames'] = str(frames_dir)

    print(f"\nAll visualizations saved to: {output_dir}")

    return saved_files
