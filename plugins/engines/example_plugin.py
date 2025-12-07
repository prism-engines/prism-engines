"""
Example Engine Plugin
======================

Demonstrates how to create a PRISM engine plugin.

To create your own:
1. Copy this file to plugins/engines/your_engine.py
2. Rename the class and update the attributes
3. Implement analyze() and optionally rank_indicators()
4. The engine will be auto-discovered!
"""

import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from typing import Dict, Any, List
import numpy as np

try:
    from core.plugin_base import EnginePlugin
except ImportError:
    # Fallback for standalone testing
    class EnginePlugin:
        name = ""
        version = "1.0.0"
        description = ""
        def __init__(self, settings=None): pass


class ExamplePlugin(EnginePlugin):
    """
    Example analysis engine demonstrating plugin structure.

    This engine performs simple statistical analysis on input data.
    """

    # Required: Unique identifier
    name = "example_plugin"

    # Metadata
    version = "1.0.0"
    description = "Example plugin demonstrating engine structure"
    author = "PRISM Team"
    category = "engine"
    tags = ["example", "demo", "statistics"]

    # Engine configuration
    engine_type = "lens"
    supports_ranking = True
    supports_analysis = True

    def get_settings_schema(self) -> Dict[str, Any]:
        """Define configurable settings."""
        return {
            "type": "object",
            "properties": {
                "threshold": {
                    "type": "number",
                    "default": 0.5,
                    "description": "Analysis threshold"
                },
                "method": {
                    "type": "string",
                    "enum": ["mean", "median", "std"],
                    "default": "mean"
                }
            }
        }

    def analyze(self, data: Any, **kwargs) -> Dict[str, Any]:
        """
        Perform analysis on input data.

        Args:
            data: Input data (DataFrame or array-like)
            **kwargs: Additional parameters

        Returns:
            Analysis results
        """
        threshold = self.settings.get("threshold", 0.5)
        method = self.settings.get("method", "mean")

        # Handle different input types
        if hasattr(data, 'values'):
            values = data.values.flatten()
        elif hasattr(data, '__iter__'):
            values = np.array(list(data)).flatten()
        else:
            values = np.array([data])

        # Remove NaN values
        values = values[~np.isnan(values.astype(float))]

        if len(values) == 0:
            return {
                "status": "no_data",
                "message": "No valid data points"
            }

        # Calculate statistics
        results = {
            "status": "completed",
            "plugin": self.name,
            "version": self.version,
            "statistics": {
                "mean": float(np.mean(values)),
                "median": float(np.median(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "count": len(values),
            },
            "threshold": threshold,
            "method": method,
        }

        # Add method-specific result
        if method == "mean":
            results["primary_metric"] = results["statistics"]["mean"]
        elif method == "median":
            results["primary_metric"] = results["statistics"]["median"]
        else:
            results["primary_metric"] = results["statistics"]["std"]

        return results

    def rank_indicators(self, data: Any, **kwargs) -> List[Dict[str, Any]]:
        """
        Rank indicators by variability.

        Args:
            data: DataFrame with indicators as columns

        Returns:
            Ranked list of indicators
        """
        rankings = []

        if not hasattr(data, 'columns'):
            return rankings

        for col in data.columns:
            try:
                values = data[col].dropna().values
                if len(values) > 0:
                    rankings.append({
                        "indicator": col,
                        "score": float(np.std(values)),
                        "metric": "volatility",
                    })
            except Exception:
                continue

        # Sort by score descending
        rankings.sort(key=lambda x: x["score"], reverse=True)

        # Add ranks
        for i, item in enumerate(rankings):
            item["rank"] = i + 1

        return rankings

    def validate(self) -> bool:
        """Validate plugin configuration."""
        return True


# For standalone testing
if __name__ == "__main__":
    import pandas as pd

    # Create test data
    np.random.seed(42)
    test_data = pd.DataFrame({
        "A": np.random.randn(100),
        "B": np.random.randn(100) * 2,
        "C": np.random.randn(100) + 5,
    })

    # Test the plugin
    engine = ExamplePlugin()

    print(f"Plugin: {engine.name} v{engine.version}")
    print(f"Description: {engine.description}")
    print()

    # Test analyze
    results = engine.analyze(test_data)
    print("Analysis Results:")
    print(f"  Mean: {results['statistics']['mean']:.4f}")
    print(f"  Std:  {results['statistics']['std']:.4f}")
    print()

    # Test ranking
    rankings = engine.rank_indicators(test_data)
    print("Indicator Rankings:")
    for r in rankings:
        print(f"  {r['rank']}. {r['indicator']}: {r['score']:.4f}")
