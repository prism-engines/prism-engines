"""
PRISM Engine - Stage 06: Analysis Output

Stores analysis results from the engine.

Structure:
    financial/
        latest/         - Most recent run
        archive/        - Historical runs (timestamped)

    climate/
        latest/
        archive/

    checkpoints/
        output_diff.html  - What changed from last run
"""

from pathlib import Path
import json
from datetime import datetime
from typing import Dict, Any, Optional

OUTPUT_ROOT = Path(__file__).parent


def save_output(
    results: Dict[str, Any],
    domain: str = "financial",
    archive: bool = True
) -> Path:
    """
    Save analysis output.

    Args:
        results: Analysis results dictionary
        domain: 'financial' or 'climate'
        archive: Whether to also save to archive

    Returns:
        Path to saved output
    """
    domain_dir = OUTPUT_ROOT / domain
    latest_dir = domain_dir / "latest"
    latest_dir.mkdir(parents=True, exist_ok=True)

    # Save to latest
    output_path = latest_dir / "run_summary.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Archive
    if archive:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        archive_dir = domain_dir / "archive" / timestamp
        archive_dir.mkdir(parents=True, exist_ok=True)

        archive_path = archive_dir / "run_summary.json"
        with open(archive_path, "w") as f:
            json.dump(results, f, indent=2, default=str)

    return output_path


def load_latest(domain: str = "financial") -> Optional[Dict]:
    """Load most recent output."""
    path = OUTPUT_ROOT / domain / "latest" / "run_summary.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None
