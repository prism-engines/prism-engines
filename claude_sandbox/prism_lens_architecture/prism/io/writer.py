# prism/io/writer.py
"""
Artifact writer for geometry outputs.

Writes:
- observations.csv (or .parquet): flat observation data
- manifest.json: provenance and system metrics

No interpretation. Just faithful serialization.
"""

import json
from pathlib import Path
from typing import Union
import pandas as pd

from prism.schema.observation import SystemSnapshot


class ArtifactWriter:
    """
    Writes geometry artifacts to disk.

    Usage:
        writer = ArtifactWriter("./output/2024-01-15")
        writer.write(snapshot, format="csv")
    """

    def __init__(self, output_dir: Union[str, Path]):
        """
        Initialize writer with output directory.

        Args:
            output_dir: Directory to write artifacts to (created if needed)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write(
        self,
        snapshot: SystemSnapshot,
        format: str = "csv",
        prefix: str = "",
    ) -> Path:
        """
        Write snapshot to disk.

        Args:
            snapshot: SystemSnapshot to write
            format: Output format - "csv" or "parquet"
            prefix: Optional prefix for filenames

        Returns:
            Path to output directory
        """
        # Build filename prefix
        file_prefix = f"{prefix}_" if prefix else ""

        # Write observations
        rows = snapshot.to_flat_rows()
        df = pd.DataFrame(rows)

        if format == "parquet":
            obs_path = self.output_dir / f"{file_prefix}observations.parquet"
            df.to_parquet(obs_path, index=False)
        else:
            obs_path = self.output_dir / f"{file_prefix}observations.csv"
            df.to_csv(obs_path, index=False)

        # Write manifest
        manifest = snapshot.to_manifest()
        manifest_path = self.output_dir / f"{file_prefix}manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2, default=str)

        return self.output_dir

    def write_incremental(
        self,
        snapshot: SystemSnapshot,
        format: str = "csv",
    ) -> Path:
        """
        Write snapshot with timestamp-based filename.

        Useful for accumulating snapshots over time.

        Returns:
            Path to output directory
        """
        # Use snapshot hash as prefix for uniqueness
        prefix = f"{snapshot.window_end}_{snapshot.snapshot_hash}"
        return self.write(snapshot, format=format, prefix=prefix)


class ArtifactReader:
    """
    Reads geometry artifacts from disk.

    Usage:
        reader = ArtifactReader("./output/2024-01-15")
        df = reader.read_observations()
        manifest = reader.read_manifest()
    """

    def __init__(self, input_dir: Union[str, Path]):
        """
        Initialize reader with input directory.

        Args:
            input_dir: Directory containing artifacts
        """
        self.input_dir = Path(input_dir)
        if not self.input_dir.exists():
            raise FileNotFoundError(f"Directory not found: {input_dir}")

    def read_observations(self, prefix: str = "") -> pd.DataFrame:
        """
        Read observations file.

        Tries parquet first, then csv.
        """
        file_prefix = f"{prefix}_" if prefix else ""

        parquet_path = self.input_dir / f"{file_prefix}observations.parquet"
        csv_path = self.input_dir / f"{file_prefix}observations.csv"

        if parquet_path.exists():
            return pd.read_parquet(parquet_path)
        elif csv_path.exists():
            return pd.read_csv(csv_path)
        else:
            raise FileNotFoundError(
                f"No observations file found in {self.input_dir}"
            )

    def read_manifest(self, prefix: str = "") -> dict:
        """Read manifest file."""
        file_prefix = f"{prefix}_" if prefix else ""
        manifest_path = self.input_dir / f"{file_prefix}manifest.json"

        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")

        with open(manifest_path) as f:
            return json.load(f)
