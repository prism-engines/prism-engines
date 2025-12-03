#!/usr/bin/env python3
"""
Temporal Visualizer - Visualize ranking changes over time windows
=======================================================================

Reads output from temporal_runner.py and generates visualizations.

Usage (command line):
    python temporal_visualizer.py --increment 5
    python temporal_visualizer.py --input output/temporal/rank_evolution_5yr.csv
    python temporal_visualizer.py --increment 8 --top 20

Usage (Python):
    from temporal_visualizer import TemporalVisualizer
    viz = TemporalVisualizer(increment=5)
    viz.generate_all()

Output (in output/temporal/plots/):
    - rank_trajectory_{increment}yr.png  : Line plot of top 15 indicator ranks
    - rank_heatmap_{increment}yr.png     : Heatmap of ranks over time
    - rank_stability_{increment}yr.png   : Scatter plot (avg rank vs std rank)
    - regime_stability_{increment}yr.png : Spearman correlation between adjacent windows
"""

import sys
import os
from pathlib import Path
import argparse
from typing import List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class TemporalVisualizer:
    """Visualize temporal results."""

    def __init__(
        self,
        increment: int = 5,
        input_dir: str = None,
        output_dir: str = None,
        top_n: int = 15,
    ):
        """
        Initialize visualizer.

        Args:
            increment: Year increment used in analysis (to find correct files)
            input_dir: Directory containing temporal results
            output_dir: Directory for saving plots
            top_n: Number of top indicators to highlight
        """
        self.increment = increment
        self.top_n = top_n

        # Paths
        self.project_root = PROJECT_ROOT
        self.input_dir = Path(input_dir) if input_dir else self.project_root / "output" / "temporal"
        self.output_dir = Path(output_dir) if output_dir else self.input_dir / "plots"

        # Data
        self.rank_evolution: Optional[pd.DataFrame] = None
        self.long_format: Optional[pd.DataFrame] = None
        self.regime_stability: Optional[pd.DataFrame] = None

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load the temporal results files."""
        # Rank evolution file
        evolution_path = self.input_dir / f"rank_evolution_{self.increment}yr.csv"
        if not evolution_path.exists():
            raise FileNotFoundError(f"File not found: {evolution_path}")

        self.rank_evolution = pd.read_csv(evolution_path, index_col=0)
        print(f"Loaded: {evolution_path}")
        print(f"  Shape: {self.rank_evolution.shape}")

        # Regime stability file
        stability_path = self.input_dir / f"regime_stability_{self.increment}yr.csv"
        if stability_path.exists():
            self.regime_stability = pd.read_csv(stability_path)
            print(f"Loaded: {stability_path}")
            print(f"  Shape: {self.regime_stability.shape}")

        # Long format file
        long_path = self.input_dir / f"temporal_results_{self.increment}yr.csv"
        if long_path.exists():
            self.long_format = pd.read_csv(long_path)
            print(f"Loaded: {long_path}")

        return self.rank_evolution, self.long_format

    def get_top_indicators(self, n: int = None) -> List[str]:
        """Get top N indicators by average rank."""
        n = n or self.top_n

        if 'avg_rank' in self.rank_evolution.columns:
            return self.rank_evolution.head(n).index.tolist()
        else:
            # Compute average rank
            window_cols = [c for c in self.rank_evolution.columns if '-' in str(c)]
            avg_rank = self.rank_evolution[window_cols].mean(axis=1)
            return avg_rank.nsmallest(n).index.tolist()

    def plot_rank_trajectory(
        self,
        indicators: List[str] = None,
        figsize: Tuple[int, int] = (14, 8),
        save: bool = True
    ) -> None:
        """
        Plot rank trajectory line chart for top indicators.

        Shows how each indicator's rank changes across time windows.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib required: pip install matplotlib")
            return

        indicators = indicators or self.get_top_indicators()

        # Get window columns (exclude avg_rank if present)
        window_cols = [c for c in self.rank_evolution.columns if '-' in str(c)]

        fig, ax = plt.subplots(figsize=figsize)

        # Color palette
        colors = plt.cm.tab20(np.linspace(0, 1, len(indicators)))

        for i, indicator in enumerate(indicators):
            if indicator in self.rank_evolution.index:
                ranks = self.rank_evolution.loc[indicator, window_cols]
                ax.plot(
                    range(len(window_cols)),
                    ranks.values,
                    label=indicator,
                    color=colors[i],
                    linewidth=2,
                    marker='o',
                    markersize=6,
                    alpha=0.8
                )

        ax.set_xlabel('Time Window', fontsize=12)
        ax.set_ylabel('Consensus Rank (lower = more important)', fontsize=12)
        ax.set_title(f'Indicator Rank Trajectories ({self.increment}-Year Windows)', fontsize=14, fontweight='bold')

        # Invert y-axis (rank 1 at top)
        ax.invert_yaxis()

        # X-axis labels
        ax.set_xticks(range(len(window_cols)))
        ax.set_xticklabels(window_cols, rotation=45, ha='right')

        # Legend outside plot
        ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=9)

        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        if save:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            save_path = self.output_dir / f"rank_trajectory_{self.increment}yr.png"
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")
            plt.close()
        else:
            plt.show()

    def plot_rank_heatmap(
        self,
        n_indicators: int = None,
        figsize: Tuple[int, int] = (14, 10),
        save: bool = True
    ) -> None:
        """
        Plot heatmap of ranks over time windows.

        Color intensity shows rank (darker = higher rank/more important).
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib required: pip install matplotlib")
            return

        n_indicators = n_indicators or self.top_n * 2  # Show more in heatmap

        # Get window columns
        window_cols = [c for c in self.rank_evolution.columns if '-' in str(c)]

        # Get top indicators
        top_indicators = self.get_top_indicators(n_indicators)

        # Extract heatmap data
        heatmap_data = self.rank_evolution.loc[top_indicators, window_cols]

        fig, ax = plt.subplots(figsize=figsize)

        # Create heatmap (reversed colormap: low rank = dark/important)
        im = ax.imshow(heatmap_data.values, aspect='auto', cmap='RdYlGn_r')

        # Colorbar
        cbar = plt.colorbar(im, ax=ax, label='Rank (lower = more important)')

        # Y-axis (indicators)
        ax.set_yticks(range(len(top_indicators)))
        ax.set_yticklabels(top_indicators, fontsize=9)

        # X-axis (windows)
        ax.set_xticks(range(len(window_cols)))
        ax.set_xticklabels(window_cols, rotation=45, ha='right', fontsize=10)

        ax.set_xlabel('Time Window', fontsize=12)
        ax.set_ylabel('Indicator', fontsize=12)
        ax.set_title(f'Indicator Rankings Over Time ({self.increment}-Year Windows)', fontsize=14, fontweight='bold')

        plt.tight_layout()

        if save:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            save_path = self.output_dir / f"rank_heatmap_{self.increment}yr.png"
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")
            plt.close()
        else:
            plt.show()

    def plot_rank_stability(
        self,
        n_indicators: int = None,
        figsize: Tuple[int, int] = (12, 8),
        save: bool = True
    ) -> None:
        """
        Plot rank stability scatter (average rank vs standard deviation).

        Helps identify:
        - Consistently important indicators (low avg rank, low std)
        - Volatile indicators (high std)
        - Consistently unimportant indicators (high avg rank, low std)
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib required: pip install matplotlib")
            return

        n_indicators = n_indicators or self.top_n * 2

        # Get window columns
        window_cols = [c for c in self.rank_evolution.columns if '-' in str(c)]

        # Compute stats
        avg_rank = self.rank_evolution[window_cols].mean(axis=1)
        std_rank = self.rank_evolution[window_cols].std(axis=1)

        # Get top indicators to label
        top_indicators = avg_rank.nsmallest(n_indicators).index.tolist()

        fig, ax = plt.subplots(figsize=figsize)

        # Plot all points
        ax.scatter(
            avg_rank.values,
            std_rank.values,
            alpha=0.4,
            c='gray',
            s=50
        )

        # Highlight and label top indicators
        for indicator in top_indicators:
            x = avg_rank[indicator]
            y = std_rank[indicator]

            ax.scatter([x], [y], c='steelblue', s=100, zorder=5)
            ax.annotate(
                indicator,
                (x, y),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=8,
                alpha=0.8
            )

        ax.set_xlabel('Average Rank (lower = more important)', fontsize=12)
        ax.set_ylabel('Rank Standard Deviation (lower = more stable)', fontsize=12)
        ax.set_title(f'Indicator Stability Analysis ({self.increment}-Year Windows)', fontsize=14, fontweight='bold')

        # Add quadrant lines at median
        ax.axhline(y=std_rank.median(), color='red', linestyle='--', alpha=0.3)
        ax.axvline(x=avg_rank.median(), color='red', linestyle='--', alpha=0.3)

        # Quadrant labels
        ax.text(0.02, 0.98, 'Stable & Important', transform=ax.transAxes,
                fontsize=10, va='top', color='green', alpha=0.7)
        ax.text(0.98, 0.98, 'Stable & Less Important', transform=ax.transAxes,
                fontsize=10, va='top', ha='right', color='gray', alpha=0.7)
        ax.text(0.02, 0.02, 'Volatile & Important', transform=ax.transAxes,
                fontsize=10, color='orange', alpha=0.7)
        ax.text(0.98, 0.02, 'Volatile & Less Important', transform=ax.transAxes,
                fontsize=10, ha='right', color='gray', alpha=0.7)

        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        if save:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            save_path = self.output_dir / f"rank_stability_{self.increment}yr.png"
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")
            plt.close()
        else:
            plt.show()

    def plot_regime_stability(
        self,
        figsize: Tuple[int, int] = (12, 6),
        save: bool = True
    ) -> None:
        """
        Plot regime stability over time (Spearman correlation between adjacent windows).

        Dips in the line indicate potential regime changes in market structure.
        This provides a single defensible metric per transition point.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib required: pip install matplotlib")
            return

        if self.regime_stability is None or self.regime_stability.empty:
            print("No regime stability data available")
            return

        fig, ax = plt.subplots(figsize=figsize)

        # Extract data
        transitions = self.regime_stability['transition_year'].values
        correlations = self.regime_stability['spearman_corr'].values

        # Plot line
        ax.plot(transitions, correlations, 'b-o', linewidth=2, markersize=8, label='Spearman ρ')

        # Add horizontal reference lines
        ax.axhline(y=1.0, color='green', linestyle='--', alpha=0.3, label='Perfect stability')
        ax.axhline(y=0.8, color='orange', linestyle='--', alpha=0.3, label='High stability (0.8)')
        ax.axhline(y=0.6, color='red', linestyle='--', alpha=0.3, label='Moderate stability (0.6)')

        # Highlight potential regime changes (low correlation)
        for i, (year, corr) in enumerate(zip(transitions, correlations)):
            if corr < 0.7:
                ax.annotate(
                    f'{corr:.2f}',
                    (year, corr),
                    xytext=(0, -15),
                    textcoords='offset points',
                    fontsize=9,
                    ha='center',
                    color='red',
                    fontweight='bold'
                )
            else:
                ax.annotate(
                    f'{corr:.2f}',
                    (year, corr),
                    xytext=(0, 10),
                    textcoords='offset points',
                    fontsize=9,
                    ha='center',
                    alpha=0.7
                )

        ax.set_xlabel('Transition Year', fontsize=12)
        ax.set_ylabel('Spearman Correlation (ρ)', fontsize=12)
        ax.set_title(
            f'Regime Stability: Ranking Correlation Between Adjacent Windows\n'
            f'({self.increment}-Year Windows)',
            fontsize=14, fontweight='bold'
        )

        ax.set_ylim(0, 1.05)
        ax.set_xticks(transitions)
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            save_path = self.output_dir / f"regime_stability_{self.increment}yr.png"
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")
            plt.close()
        else:
            plt.show()

    def generate_all(self, save: bool = True) -> None:
        """Generate all visualizations."""
        print("\n" + "=" * 60)
        print("TEMPORAL VISUALIZER")
        print("=" * 60)

        # Load data
        self.load_data()
        print()

        # Generate plots
        print("Generating plots...")

        print("\n1. Rank trajectory (line plot)...")
        self.plot_rank_trajectory(save=save)

        print("\n2. Rank heatmap...")
        self.plot_rank_heatmap(save=save)

        print("\n3. Rank stability (scatter)...")
        self.plot_rank_stability(save=save)

        print("\n4. Regime stability (Spearman correlation)...")
        self.plot_regime_stability(save=save)

        print("\n" + "=" * 60)
        print(f"All plots saved to: {self.output_dir}")
        print("=" * 60)

    def print_summary(self) -> None:
        """Print summary statistics."""
        if self.rank_evolution is None:
            self.load_data()

        window_cols = [c for c in self.rank_evolution.columns if '-' in str(c)]

        avg_rank = self.rank_evolution[window_cols].mean(axis=1)
        std_rank = self.rank_evolution[window_cols].std(axis=1)

        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)

        print(f"\nTime Windows: {window_cols}")
        print(f"Total Indicators: {len(self.rank_evolution)}")

        print("\nTOP 15 BY AVERAGE RANK:")
        print("-" * 40)
        for i, (indicator, rank) in enumerate(avg_rank.nsmallest(15).items(), 1):
            std = std_rank[indicator]
            print(f"  {i:2}. {indicator:<25} avg={rank:.1f}  std={std:.1f}")

        # Most volatile
        print("\nMOST VOLATILE (high std in rankings):")
        print("-" * 40)
        for i, (indicator, std) in enumerate(std_rank.nlargest(10).items(), 1):
            avg = avg_rank[indicator]
            print(f"  {i:2}. {indicator:<25} std={std:.1f}  avg={avg:.1f}")

        # Most stable (low std)
        print("\nMOST STABLE (low std in rankings):")
        print("-" * 40)
        for i, (indicator, std) in enumerate(std_rank.nsmallest(10).items(), 1):
            avg = avg_rank[indicator]
            print(f"  {i:2}. {indicator:<25} std={std:.1f}  avg={avg:.1f}")


# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Temporal Visualizer - Generate plots from temporal analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python temporal_visualizer.py --increment 5
  python temporal_visualizer.py --increment 8 --top 20
  python temporal_visualizer.py --input /path/to/results --output /path/to/plots
  python temporal_visualizer.py --increment 5 --summary
        """
    )

    parser.add_argument(
        '--increment', '-i',
        type=int,
        default=5,
        help='Year increment used in analysis (default: 5)'
    )

    parser.add_argument(
        '--input',
        type=str,
        default=None,
        help='Input directory containing temporal results'
    )

    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Output directory for plots'
    )

    parser.add_argument(
        '--top', '-t',
        type=int,
        default=15,
        help='Number of top indicators to highlight (default: 15)'
    )

    parser.add_argument(
        '--summary', '-s',
        action='store_true',
        help='Print summary statistics only (no plots)'
    )

    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Display plots instead of saving'
    )

    args = parser.parse_args()

    # Create visualizer
    viz = TemporalVisualizer(
        increment=args.increment,
        input_dir=args.input,
        output_dir=args.output,
        top_n=args.top,
    )

    if args.summary:
        viz.print_summary()
    else:
        viz.generate_all(save=not args.no_save)


if __name__ == "__main__":
    main()
