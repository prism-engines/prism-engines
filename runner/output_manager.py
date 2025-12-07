"""
PRISM Output Manager
=====================

Handles result formatting and file output.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
import pandas as pd

logger = logging.getLogger(__name__)


class OutputManager:
    """Manages analysis output formatting and file saving."""

    def __init__(self, output_dir: Optional[Path] = None):
        """
        Initialize output manager.

        Args:
            output_dir: Base output directory (default: ./output)
        """
        self.base_dir = output_dir or Path("output")
        self.current_run_dir: Optional[Path] = None

    def create_run_directory(self, prefix: str = "run") -> Path:
        """Create a timestamped directory for this run."""
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        run_dir = self.base_dir / f"{prefix}_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)
        self.current_run_dir = run_dir
        logger.info(f"Created output directory: {run_dir}")
        return run_dir

    def get_run_directory(self) -> Path:
        """Get current run directory, creating if needed."""
        if self.current_run_dir is None:
            return self.create_run_directory()
        return self.current_run_dir

    def save_json(self, data: Dict[str, Any], filename: str) -> Path:
        """Save data as JSON file."""
        run_dir = self.get_run_directory()
        filepath = run_dir / filename

        # Convert non-serializable objects
        def convert(obj):
            if isinstance(obj, (pd.Timestamp, datetime)):
                return obj.isoformat()
            if isinstance(obj, pd.DataFrame):
                return obj.to_dict()
            if isinstance(obj, pd.Series):
                return obj.to_dict()
            if hasattr(obj, '__dict__'):
                return str(obj)
            return obj

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=convert)

        logger.info(f"Saved JSON: {filepath}")
        return filepath

    def save_csv(self, df: pd.DataFrame, filename: str) -> Path:
        """Save DataFrame as CSV file."""
        run_dir = self.get_run_directory()
        filepath = run_dir / filename
        df.to_csv(filepath)
        logger.info(f"Saved CSV: {filepath}")
        return filepath

    def save_text(self, content: str, filename: str) -> Path:
        """Save text content to file."""
        run_dir = self.get_run_directory()
        filepath = run_dir / filename
        with open(filepath, 'w') as f:
            f.write(content)
        logger.info(f"Saved text: {filepath}")
        return filepath

    def format_rankings(self, rankings: Dict[str, List], top_n: int = 10) -> str:
        """Format indicator rankings for display."""
        lines = []
        lines.append("\n" + "=" * 60)
        lines.append("INDICATOR RANKINGS")
        lines.append("=" * 60)

        for lens_name, ranking in rankings.items():
            lines.append(f"\n{lens_name}:")
            lines.append("-" * 40)
            for i, indicator in enumerate(ranking[:top_n], 1):
                if isinstance(indicator, dict):
                    name = indicator.get('indicator', indicator.get('name', str(indicator)))
                    score = indicator.get('score', indicator.get('importance', ''))
                    if score:
                        lines.append(f"  {i:2}. {name}: {score:.4f}")
                    else:
                        lines.append(f"  {i:2}. {name}")
                else:
                    lines.append(f"  {i:2}. {indicator}")

        return "\n".join(lines)

    def format_summary(self, results: Dict[str, Any]) -> str:
        """Format analysis summary for display."""
        lines = []
        lines.append("\n" + "=" * 60)
        lines.append("ANALYSIS SUMMARY")
        lines.append("=" * 60)

        if "panel" in results:
            lines.append(f"Panel: {results['panel']}")
        if "workflow" in results:
            lines.append(f"Workflow: {results['workflow']}")
        if "duration" in results:
            lines.append(f"Duration: {results['duration']:.2f}s")
        if "timestamp" in results:
            lines.append(f"Timestamp: {results['timestamp']}")
        if "indicators_analyzed" in results:
            lines.append(f"Indicators: {results['indicators_analyzed']}")
        if "lenses_run" in results:
            lenses = results['lenses_run']
            if isinstance(lenses, list):
                lines.append(f"Lenses: {len(lenses)}")
            else:
                lines.append(f"Lenses: {lenses}")

        lines.append("=" * 60)
        return "\n".join(lines)

    def format_regime_comparison(self, comparison: Dict[str, Any]) -> str:
        """Format regime comparison results."""
        lines = []
        lines.append("\n" + "=" * 60)
        lines.append("REGIME COMPARISON RESULTS")
        lines.append("=" * 60)

        if "current_period" in comparison:
            lines.append(f"\nCurrent Period: {comparison['current_period']}")

        if "similarities" in comparison:
            lines.append("\nSimilarity to Historical Regimes:")
            lines.append("-" * 40)
            for period, score in comparison["similarities"].items():
                # Handle both float scores and other types
                if isinstance(score, (int, float)):
                    bar_len = int(score * 20)
                    bar = "#" * bar_len + "-" * (20 - bar_len)
                    lines.append(f"  {period:20} [{bar}] {score:.1%}")
                else:
                    lines.append(f"  {period:20} {score}")

        if "closest_match" in comparison:
            lines.append(f"\nClosest Match: {comparison['closest_match']}")

        return "\n".join(lines)

    def print_banner(self, title: str) -> None:
        """Print a formatted banner."""
        width = 60
        print("\n" + "+" + "=" * (width - 2) + "+")
        print("|" + title.center(width - 2) + "|")
        print("+" + "=" * (width - 2) + "+")
