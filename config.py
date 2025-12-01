"""
PRISM Engine Configuration
==========================

Project metadata and version information.
"""

__version__ = "0.1.0"
__author__ = "PRISM Contributors"
__project__ = "PRISM Engine"

# Project paths are computed relative to this file
from pathlib import Path
import os


def get_project_root() -> Path:
    """
    Get project root directory, works in scripts, notebooks, and Colab.

    Resolution order:
    1. PROJECT_ROOT environment variable (if set)
    2. Google Colab auto-detection (via COLAB_RELEASE_TAG env var)
    3. Relative to this config file (standard case)

    Returns:
        Path to the project root directory
    """
    # 1. Check for explicit PROJECT_ROOT environment variable
    env_root = os.environ.get('PROJECT_ROOT')
    if env_root:
        env_path = Path(env_root)
        if env_path.exists():
            return env_path

    # 2. Check if running in Google Colab (use env var, not path check)
    if 'COLAB_RELEASE_TAG' in os.environ or 'COLAB_GPU' in os.environ:
        # Try common Colab locations for the project
        project_name = 'prism-engine'
        colab_locations = [
            Path.home() / project_name,  # ~/prism-engine
            Path.cwd() / project_name,   # ./prism-engine
            Path.cwd(),                  # Current directory (if already in project)
        ]
        # Also check Drive if mounted (Colab mounts Drive at /content/drive)
        drive_base = Path('/content/drive/MyDrive')
        if drive_base.exists():
            colab_locations.insert(0, drive_base / project_name)

        for loc in colab_locations:
            if loc.exists() and (loc / 'config.py').exists():
                return loc

    # 3. Standard case: relative to this config file
    return Path(__file__).parent.resolve()


PROJECT_ROOT = get_project_root()

# Standard project directories
DATA_DIR = PROJECT_ROOT / 'data'
OUTPUT_DIR = PROJECT_ROOT / '06_output'
LOGS_DIR = PROJECT_ROOT / 'logs'


def ensure_dir(path: Path) -> Path:
    """Create directory if it doesn't exist and return the path."""
    path.mkdir(parents=True, exist_ok=True)
    return path
