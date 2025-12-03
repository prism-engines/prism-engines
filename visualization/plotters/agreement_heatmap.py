"""
Agreement Heatmap - Visualize lens agreement matrix
"""

from typing import Optional
from pathlib import Path
import pandas as pd


def plot_agreement_heatmap(
    agreement_matrix: pd.DataFrame,
    save_path: Optional[Path] = None,
    figsize: tuple = (10, 8)
) -> None:
    """
    Plot heatmap of lens agreement.

    Args:
        agreement_matrix: DataFrame with lens agreement scores
        save_path: Path to save figure (optional)
        figsize: Figure size
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("matplotlib and seaborn required for plotting")
        return

    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        agreement_matrix,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        vmin=0,
        vmax=1,
        ax=ax,
        cbar_kws={"label": "Agreement (fraction of top 10 overlap)"}
    )

    ax.set_title("Lens Agreement Matrix")
    ax.set_xlabel("Lens")
    ax.set_ylabel("Lens")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
