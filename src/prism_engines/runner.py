"""
PRISM Engine Runner - Main analysis API.
"""

import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union
from pathlib import Path

from .engines import (
    EngineResult,
    get_engine,
    list_engines,
    DEFAULT_ENGINES,
)
from .data import load_csv, validate_data


@dataclass
class Results:
    """
    Container for PRISM analysis results.

    Attributes:
        engine_results: Results from each engine
        data_info: Information about the input data
    """
    engine_results: Dict[str, EngineResult] = field(default_factory=dict)
    data_info: Dict[str, Any] = field(default_factory=dict)

    def __getitem__(self, engine_name: str) -> EngineResult:
        """Get result for a specific engine."""
        return self.engine_results[engine_name]

    def report(self) -> str:
        """Generate a text report of all results."""
        lines = []
        lines.append("=" * 60)
        lines.append("PRISM ENGINES ANALYSIS REPORT")
        lines.append("=" * 60)

        # Data info
        if self.data_info:
            lines.append(f"\nData: {self.data_info.get('shape', 'unknown')} "
                        f"({self.data_info.get('n_columns', '?')} series)")
            if self.data_info.get('date_range'):
                lines.append(f"Range: {self.data_info['date_range'][0]} to "
                            f"{self.data_info['date_range'][1]}")

        # Engine results
        for name, result in self.engine_results.items():
            lines.append(f"\n--- {name.upper()} ---")

            if not result.success:
                lines.append(f"  Error: {result.error}")
                continue

            for key, value in result.metrics.items():
                if key.endswith("_matrix"):
                    lines.append(f"  {key}: [{len(value)}x{len(value)} matrix]")
                elif isinstance(value, dict) and len(value) > 5:
                    lines.append(f"  {key}: {{{len(value)} items}}")
                elif isinstance(value, float):
                    lines.append(f"  {key}: {value:.4f}")
                elif isinstance(value, list) and all(isinstance(v, float) for v in value):
                    formatted = [f"{v:.3f}" for v in value[:5]]
                    lines.append(f"  {key}: [{', '.join(formatted)}]")
                else:
                    lines.append(f"  {key}: {value}")

        lines.append("\n" + "=" * 60)
        return "\n".join(lines)

    def print_report(self):
        """Print the analysis report."""
        print(self.report())

    def plot(self, figsize: tuple = (12, 8)):
        """
        Generate visualization of results.

        Returns:
            matplotlib Figure
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib required for plotting: pip install matplotlib")

        n_engines = len(self.engine_results)
        fig, axes = plt.subplots(1, n_engines, figsize=figsize)

        if n_engines == 1:
            axes = [axes]

        for ax, (name, result) in zip(axes, self.engine_results.items()):
            ax.set_title(name.upper())

            if not result.success:
                ax.text(0.5, 0.5, f"Error: {result.error}",
                       ha='center', va='center', transform=ax.transAxes)
                continue

            # Engine-specific plots
            if name == "correlation":
                matrix = result.metrics.get("correlation_matrix")
                if matrix:
                    im = ax.imshow(matrix, cmap='RdBu_r', vmin=-1, vmax=1)
                    ax.set_xlabel("Series")
                    ax.set_ylabel("Series")
                    plt.colorbar(im, ax=ax, label="Correlation")

            elif name == "pca":
                var_ratio = result.metrics.get("explained_variance_ratio", [])
                if var_ratio:
                    ax.bar(range(1, len(var_ratio) + 1), var_ratio)
                    ax.set_xlabel("Principal Component")
                    ax.set_ylabel("Variance Explained")
                    ax.set_xticks(range(1, len(var_ratio) + 1))

            elif name == "hurst":
                h_vals = result.metrics.get("hurst_exponents", {})
                valid_h = {k: v for k, v in h_vals.items() if v is not None}
                if valid_h:
                    ax.barh(list(valid_h.keys()), list(valid_h.values()))
                    ax.axvline(0.5, color='red', linestyle='--', label='Random Walk')
                    ax.set_xlabel("Hurst Exponent")
                    ax.set_xlim(0, 1)
                    ax.legend()

            else:
                ax.text(0.5, 0.5, f"No plot for {name}",
                       ha='center', va='center', transform=ax.transAxes)

        plt.tight_layout()
        return fig

    def save(self, output_dir: Union[str, Path], name: str = "prism_report"):
        """
        Save report and plot to files.

        Args:
            output_dir: Directory to save to
            name: Base name for output files
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save report
        report_path = output_dir / f"{name}.txt"
        with open(report_path, "w") as f:
            f.write(self.report())
        print(f"Report saved: {report_path}")

        # Save plot
        try:
            fig = self.plot()
            plot_path = output_dir / f"{name}.png"
            fig.savefig(plot_path, dpi=150, bbox_inches='tight')
            print(f"Plot saved: {plot_path}")
        except ImportError:
            print("Skipping plot (matplotlib not installed)")


def run_engines(
    data: Union[str, Path, pd.DataFrame],
    engines: Optional[List[str]] = None,
) -> Results:
    """
    Run PRISM engines on data.

    Args:
        data: Path to CSV file or DataFrame
        engines: List of engines to run (default: all)

    Returns:
        Results object with analysis results

    Example:
        >>> results = run_engines("prices.csv")
        >>> results.print_report()
        >>> results.plot()
    """
    # Load data if path
    if isinstance(data, (str, Path)):
        df = load_csv(data)
    else:
        df = data

    # Validate
    validation = validate_data(df)
    if not validation["valid"]:
        raise ValueError(f"Invalid data: {validation['issues']}")

    # Select engines
    if engines is None:
        engines = DEFAULT_ENGINES
    else:
        available = list_engines()
        for e in engines:
            if e not in available:
                raise ValueError(f"Unknown engine: {e}. Available: {available}")

    # Run engines
    results = Results(
        data_info={
            "shape": df.shape,
            "n_columns": len(df.columns),
            "columns": list(df.columns),
            "date_range": (
                str(df.index.min()),
                str(df.index.max()),
            ) if hasattr(df.index, 'min') else None,
        }
    )

    for engine_name in engines:
        engine = get_engine(engine_name)
        result = engine.run(df)
        results.engine_results[engine_name] = result

    return results
