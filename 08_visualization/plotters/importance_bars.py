"""
Importance Bar Chart - Visualize indicator rankings
"""

from typing import Optional
from pathlib import Path
import pandas as pd


def plot_importance_bars(
    ranking_df: pd.DataFrame,
    top_n: int = 20,
    save_path: Optional[Path] = None,
    figsize: tuple = (12, 8)
) -> None:
    """
    Plot horizontal bar chart of indicator importance.

    Args:
        ranking_df: DataFrame with 'indicator' and 'score' columns
        top_n: Number of top indicators to show
        save_path: Path to save figure (optional)
        figsize: Figure size
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib required for plotting")
        return

    # Get top N
    data = ranking_df.head(top_n).copy()
    data = data.sort_values("score", ascending=True)  # For horizontal bars

    fig, ax = plt.subplots(figsize=figsize)

    # Create horizontal bars
    bars = ax.barh(data["indicator"], data["score"], color="steelblue")

    # Add value labels
    for bar, score in zip(bars, data["score"]):
        ax.text(
            bar.get_width() + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{score:.3f}",
            va="center",
            fontsize=9
        )

    ax.set_xlabel("Consensus Score")
    ax.set_title(f"Top {top_n} Indicators by Consensus Ranking")
    ax.set_xlim(0, max(data["score"]) * 1.15)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
