# examples/observe_demo.py
"""
Demonstration of the PRISM observation pipeline.

This shows how to:
1. Create an Observer with multiple lenses
2. Observe indicators
3. Inspect the resulting geometry
4. Export artifacts
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Import PRISM components
from prism.engine.observer import Observer
from prism.engine.registry import get_lens, list_lenses
from prism.io.writer import ArtifactWriter


def create_sample_data():
    """Create synthetic data for demonstration."""
    dates = pd.date_range("2020-01-01", periods=500, freq="D")

    # SP500-like: upward trend with volatility
    sp500 = pd.DataFrame({
        "date": dates,
        "value": 3000 + np.cumsum(np.random.randn(500) * 20) + np.arange(500) * 0.5,
    })

    # Treasury yield: mean-reverting
    tnx = pd.DataFrame({
        "date": dates,
        "value": 2.0 + np.cumsum(np.random.randn(500) * 0.02) * 0.1,
    })

    # Dollar index: range-bound
    dxy = pd.DataFrame({
        "date": dates,
        "value": 100 + np.sin(np.arange(500) / 50) * 5 + np.random.randn(500) * 2,
    })

    # Aggregate bonds: inverse correlation to rates
    agg = pd.DataFrame({
        "date": dates,
        "value": 110 - tnx["value"] * 2 + np.random.randn(500) * 0.5,
    })

    return {
        "SP500": sp500,
        "TNX": tnx,
        "DXY": dxy,
        "AGG": agg,
    }


def main():
    print("PRISM Observation Demo")
    print("=" * 50)

    # Show available lenses
    print(f"\nAvailable lenses: {list_lenses()}")

    # Create observer with all standard lenses
    observer = Observer([
        get_lens("geometry"),
        get_lens("entropy"),
        get_lens("curvature"),
        get_lens("gap"),
        get_lens("changepoint"),
        get_lens("sparsity"),
        get_lens("compare"),
    ])

    print(f"Observer configured with: {observer.lens_names}")

    # Create sample data
    data = create_sample_data()
    print(f"\nIndicators: {list(data.keys())}")
    print(f"Data points per indicator: {len(data['SP500'])}")

    # Observe!
    print("\nRunning observation...")
    snapshot = observer.observe(
        data,
        manifest={
            "run_name": "demo_run",
            "notes": "Example observation for testing",
        }
    )

    # Inspect results
    print(f"\n--- Snapshot Summary ---")
    print(f"Hash: {snapshot.snapshot_hash}")
    print(f"Window: {snapshot.window_start} to {snapshot.window_end}")
    print(f"Indicators: {snapshot.indicator_names}")
    print(f"Lenses: {snapshot.lens_names}")

    print(f"\n--- System Metrics ---")
    for k, v in snapshot.system_metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    # Drill into one indicator
    print(f"\n--- SP500 Geometry ---")
    sp500_geo = snapshot.get_indicator("SP500")
    for lens, metrics in sp500_geo.metrics_by_lens().items():
        print(f"  [{lens}]")
        for k, v in metrics.items():
            print(f"    {k}: {v:.4f}" if isinstance(v, float) else f"    {k}: {v}")

    # Get a metric across all indicators
    print(f"\n--- Standard Deviation by Indicator ---")
    std_matrix = snapshot.get_metric_matrix("std")
    for ind, lens_vals in std_matrix.items():
        vals = [f"{l}={v:.2f}" for l, v in lens_vals.items()]
        print(f"  {ind}: {', '.join(vals)}")

    # Export
    output_dir = Path("./output/demo")
    writer = ArtifactWriter(output_dir)
    writer.write(snapshot, format="csv")
    print(f"\n--- Exported to {output_dir} ---")
    print(f"  observations.csv")
    print(f"  manifest.json")

    print("\nDone.")


if __name__ == "__main__":
    main()
