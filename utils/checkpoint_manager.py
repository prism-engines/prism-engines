"""
Checkpoint Manager - Save and load pipeline state
"""

from pathlib import Path
from typing import Any, Optional, Dict, List
from datetime import datetime
import json
import pandas as pd
import logging

logger = logging.getLogger(__name__)

# Get project root (utils -> project root)
_SCRIPT_DIR = Path(__file__).parent.resolve()
_PROJECT_ROOT = _SCRIPT_DIR.parent


class CheckpointManager:
    """
    Manage checkpoints throughout the PRISM pipeline.

    Provides:
    - Unified save/load interface
    - Automatic format detection
    - Checkpoint inspection and diffing
    """

    def __init__(self, base_path: Path = None):
        # Default to project root if not specified
        self.base_path = Path(base_path) if base_path else _PROJECT_ROOT

    def save(
        self,
        stage: str,
        name: str,
        data: Any,
        format: str = "auto"
    ) -> Path:
        """
        Save checkpoint data.

        Args:
            stage: Pipeline stage (e.g., "fetch", "engine_core")
            name: Checkpoint name
            data: Data to save
            format: 'auto', 'json', 'parquet', 'csv'

        Returns:
            Path to saved file
        """
        checkpoint_dir = self.base_path / stage / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Auto-detect format
        if format == "auto":
            if isinstance(data, pd.DataFrame):
                format = "parquet"
            elif isinstance(data, dict):
                format = "json"
            else:
                format = "json"

        # Save
        if format == "parquet":
            path = checkpoint_dir / f"{name}.parquet"
            data.to_parquet(path)
        elif format == "csv":
            path = checkpoint_dir / f"{name}.csv"
            data.to_csv(path, index=False)
        else:
            path = checkpoint_dir / f"{name}.json"
            with open(path, "w") as f:
                json.dump(data, f, indent=2, default=str)

        logger.info(f"Checkpoint saved: {path}")
        return path

    def load(self, stage: str, name: str) -> Any:
        """
        Load checkpoint data.

        Args:
            stage: Pipeline stage
            name: Checkpoint name

        Returns:
            Loaded data
        """
        checkpoint_dir = self.base_path / stage / "checkpoints"

        # Try different extensions
        for ext, loader in [
            (".parquet", pd.read_parquet),
            (".csv", pd.read_csv),
            (".json", lambda p: json.load(open(p)))
        ]:
            path = checkpoint_dir / f"{name}{ext}"
            if path.exists():
                return loader(path)

        raise FileNotFoundError(f"Checkpoint not found: {stage}/{name}")

    def list_checkpoints(self, stage: str) -> List[str]:
        """List all checkpoints for a stage."""
        checkpoint_dir = self.base_path / stage / "checkpoints"
        if not checkpoint_dir.exists():
            return []

        return [f.stem for f in checkpoint_dir.glob("*") if f.is_file()]

    def inspect(self, stage: str) -> None:
        """Print summary of all checkpoints at a stage."""
        checkpoint_dir = self.base_path / stage / "checkpoints"
        if not checkpoint_dir.exists():
            print(f"No checkpoints at {stage}")
            return

        print(f"\nCheckpoints at {stage}:")
        print("-" * 40)

        for f in sorted(checkpoint_dir.glob("*")):
            if f.is_file():
                size = f.stat().st_size
                modified = datetime.fromtimestamp(f.stat().st_mtime)
                print(f"  {f.name:<30} {size:>10} bytes  {modified}")
