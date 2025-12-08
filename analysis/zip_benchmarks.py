#!/usr/bin/env python3
"""
PRISM BENCHMARK ZIP PACKAGER
----------------------------
Creates a ZIP archive of all benchmark outputs (report + plots).
"""

import zipfile
from pathlib import Path
import yaml

BASE = Path(__file__).resolve().parents[1]
CONFIG = yaml.safe_load(open(BASE / "config.yaml"))
OUTPUT = Path(CONFIG["output"]["dir"]) / "benchmarks"

ZIP_PATH = OUTPUT / "PRISM_BENCHMARK_SUITE.zip"


def main():
    if not OUTPUT.exists():
        print(f"[PRISM] Error: Output directory does not exist: {OUTPUT}")
        print("[PRISM] Please run benchmark_report.py first.")
        return

    files_to_zip = list(OUTPUT.glob("*.md")) + list(OUTPUT.glob("*.png"))

    if not files_to_zip:
        print("[PRISM] Error: No files found to zip.")
        print("[PRISM] Please run benchmark_report.py first.")
        return

    print(f"[PRISM] Creating benchmark ZIP archive...")
    print(f"[PRISM] Output: {ZIP_PATH}")

    with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(files_to_zip):
            z.write(f, f.name)
            print(f"  + {f.name}")

    print(f"\n[PRISM] Benchmark Suite zipped → {ZIP_PATH}")
    print(f"[PRISM] Archive contains {len(files_to_zip)} files")


if __name__ == "__main__":
    main()
