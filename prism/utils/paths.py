"""
PRISM Path Resolution

Single source of truth for all filesystem paths.
All scripts must import from here — never use Path(".") or os.getcwd().
"""

from pathlib import Path

# Resolve repo root from this file's location
# prism/utils/paths.py -> parents[2] = repo root
REPO_ROOT = Path(__file__).resolve().parents[2]

# Canonical directories
DATA_DIR = REPO_ROOT / "data"
RESULTS_DIR = REPO_ROOT / "results"
SCRIPTS_DIR = REPO_ROOT / "scripts"
PRISM_DIR = REPO_ROOT / "prism"

# Sanity check — fail fast if we're misconfigured
if not PRISM_DIR.exists():
    raise RuntimeError(
        f"REPO_ROOT misidentified — aborting.\n"
        f"Expected prism/ at {PRISM_DIR}, but it doesn't exist."
    )


def ensure_dirs() -> None:
    """Create standard directories if they don't exist."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
