#!/usr/bin/env python3
"""
PRISM BENCHMARK REPORT GENERATOR
--------------------------------
Creates a full academic-style Markdown report + plots for all synthetic benchmarks.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import textwrap
import yaml

BASE = Path(__file__).resolve().parents[1]
CONFIG = yaml.safe_load(open(BASE / "config.yaml"))
OUTPUT = Path(CONFIG["output"]["dir"]) / "benchmarks"
OUTPUT.mkdir(parents=True, exist_ok=True)

BENCHMARK_DIR = BASE / "data" / "benchmark"


def load_benchmarks():
    """Load all benchmark CSV files from the benchmark directory."""
    benchmarks = {}
    for f in sorted(BENCHMARK_DIR.glob("*.csv")):
        df = pd.read_csv(f, index_col=0, parse_dates=True)
        df.attrs["path"] = str(f)
        benchmarks[f.stem] = df
    return benchmarks


def plot_series(name, df):
    """Generate a time series plot for all columns in the benchmark."""
    fig, ax = plt.subplots(figsize=(12, 4))

    for col in df.columns:
        ax.plot(df.index, df[col], label=col, alpha=0.8)

    ax.set_title(f"{name}: Synthetic Benchmark Series")
    ax.set_xlabel("Date")
    ax.set_ylabel("Value")
    ax.legend(loc="upper right", ncol=2)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUTPUT / f"{name}_series.png", dpi=180)
    plt.close(fig)


def compute_deviation(df):
    """
    Compute expected value and RMSE deviation for each series.
    Expected value = mean of all series at each time point.
    Deviation = RMSE from expected.
    """
    series_cols = df.columns.tolist()
    expected = df[series_cols].mean(axis=1).values
    dev = {}
    for col in series_cols:
        diff = df[col].values - expected
        dev[col] = np.sqrt(np.mean(diff**2))
    return expected, dev


def compute_statistics(df):
    """Compute summary statistics for each series."""
    stats = {}
    for col in df.columns:
        stats[col] = {
            "mean": df[col].mean(),
            "std": df[col].std(),
            "min": df[col].min(),
            "max": df[col].max(),
            "range": df[col].max() - df[col].min(),
        }
    return stats


def generate_markdown(benchmarks):
    """Generate the full Markdown report."""
    md = ["# PRISM Benchmark Suite Report", ""]
    md.append("Generated automatically by PRISM Benchmark Report Engine.")
    md.append("")
    md.append("---")
    md.append("")

    for name, df in benchmarks.items():
        md.append(f"## Benchmark: {name}")
        md.append("")
        md.append(f"**Source:** `{df.attrs.get('path', 'unknown')}`")
        md.append("")
        md.append(f"**Time Range:** {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")
        md.append("")
        md.append(f"**Number of Observations:** {len(df)}")
        md.append("")

        # Compute deviation
        expected, dev = compute_deviation(df)
        md.append("### Deviation Summary (RMSE from Expected)")
        md.append("")
        md.append("| Series | RMSE |")
        md.append("|--------|------|")
        for k, v in sorted(dev.items()):
            md.append(f"| {k} | {v:.4f} |")
        md.append("")

        # Summary statistics
        stats = compute_statistics(df)
        md.append("### Summary Statistics")
        md.append("")
        md.append("| Series | Mean | Std Dev | Min | Max | Range |")
        md.append("|--------|------|---------|-----|-----|-------|")
        for col, s in sorted(stats.items()):
            md.append(f"| {col} | {s['mean']:.2f} | {s['std']:.2f} | {s['min']:.2f} | {s['max']:.2f} | {s['range']:.2f} |")
        md.append("")

        # Insert plot reference
        md.append(f"![{name} Plot](benchmarks/{name}_series.png)")
        md.append("")

        # Add analytic notes
        md.append("### Analytic Notes")
        md.append("")
        md.append(textwrap.dedent("""\
        - The expected synthetic series is computed as the average of all included signals.
        - RMSE is used to quantify divergence from the underlying synthetic archetype.
        - These benchmarks are designed to test PRISM's sensitivity to noise, regime shifts,
          and nonlinear structure.
        """))
        md.append("")
        md.append("---")
        md.append("")

    return "\n".join(md)


def main():
    print("[PRISM] Loading benchmark files...")
    benchmarks = load_benchmarks()
    print(f"[PRISM] Found {len(benchmarks)} benchmark files")

    # Create plots
    print("[PRISM] Generating plots...")
    for name, df in benchmarks.items():
        plot_series(name, df)
        print(f"  - {name}_series.png")

    # Create Markdown report
    print("[PRISM] Generating Markdown report...")
    md = generate_markdown(benchmarks)
    md_path = OUTPUT / "PRISM_BENCHMARK_REPORT.md"
    md_path.write_text(md)
    print(f"[PRISM] Benchmark Report saved → {md_path}")

    print("\n[PRISM] All plots saved to:")
    for p in sorted(OUTPUT.glob("*.png")):
        print(f"  - {p.name}")

    print("\n[PRISM] Benchmark report generation complete!")


if __name__ == "__main__":
    main()
