"""
PRISM Output Path Configuration
===============================

Cross-platform output directory that works on Mac, Windows, Chromebook, Linux.
Automatically finds Google Drive or falls back to local folder.

Usage in any script:
    from output_config import OUTPUT_DIR
    my_file = OUTPUT_DIR / "results.csv"
"""

import os
import platform
from pathlib import Path


def find_google_drive() -> Path | None:
    """
    Find Google Drive path across different operating systems.
    Returns None if Google Drive not found.
    """
    system = platform.system()
    home = Path.home()
    
    # Possible Google Drive locations by OS
    candidates = []
    
    if system == "Darwin":  # macOS
        # New Google Drive (Drive for Desktop)
        cloud_storage = home / "Library/CloudStorage"
        if cloud_storage.exists():
            for folder in cloud_storage.iterdir():
                if folder.name.startswith("GoogleDrive"):
                    # Could be "GoogleDrive-email@gmail.com" format
                    my_drive = folder / "My Drive"
                    if my_drive.exists():
                        candidates.append(my_drive)
                    # Or just the folder itself
                    candidates.append(folder)
        
        # Old Google Drive location
        candidates.append(home / "Google Drive")
        candidates.append(home / "Google Drive/My Drive")
        
    elif system == "Windows":
        # Windows Google Drive locations
        candidates.extend([
            Path("G:/My Drive"),
            Path("G:/"),
            home / "Google Drive",
            home / "Google Drive/My Drive",
            Path(os.environ.get("USERPROFILE", "")) / "Google Drive",
        ])
        
        # Check common drive letters
        for letter in ["G", "H", "D", "E"]:
            candidates.append(Path(f"{letter}:/My Drive"))
            candidates.append(Path(f"{letter}:/Google Drive"))
            
    elif system == "Linux":  # Includes Chromebook (Crostini)
        # Chromebook - Google Drive via Linux integration
        candidates.extend([
            home / "GoogleDrive",
            home / "google-drive",
            Path("/mnt/chromeos/GoogleDrive/MyDrive"),
            home / "MyDrive",
            # Mounted via rclone or other tools
            home / "gdrive",
            Path("/mnt/gdrive"),
        ])
    
    # Return first existing path
    for path in candidates:
        if path and path.exists():
            return path
    
    return None


def get_output_dir() -> Path:
    """
    Get the output directory for PRISM.
    
    Priority:
    1. PRISM_OUTPUT_DIR environment variable (if set)
    2. Google Drive / prism_output (if Google Drive found)
    3. ~/prism_output (local fallback)
    """
    
    # 1. Check environment variable first (allows override)
    env_path = os.environ.get("PRISM_OUTPUT_DIR")
    if env_path:
        path = Path(env_path)
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    # 2. Try Google Drive
    gdrive = find_google_drive()
    if gdrive:
        output_path = gdrive / "prism_output"
        try:
            output_path.mkdir(parents=True, exist_ok=True)
            return output_path
        except PermissionError:
            pass  # Fall through to local
    
    # 3. Local fallback
    local_path = Path.home() / "prism_output"
    local_path.mkdir(parents=True, exist_ok=True)
    return local_path


def get_data_dir() -> Path:
    """
    Get the data directory for PRISM (databases, panels).
    Same logic as output but separate folder.
    """
    env_path = os.environ.get("PRISM_DATA_DIR")
    if env_path:
        path = Path(env_path)
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    gdrive = find_google_drive()
    if gdrive:
        data_path = gdrive / "prism_data"
        try:
            data_path.mkdir(parents=True, exist_ok=True)
            return data_path
        except PermissionError:
            pass
    
    local_path = Path.home() / "prism_data"
    local_path.mkdir(parents=True, exist_ok=True)
    return local_path


# Export these as constants for easy import
OUTPUT_DIR = get_output_dir()
DATA_DIR = get_data_dir()


def print_config():
    """Print current configuration (for debugging)."""
    print("=" * 50)
    print("PRISM Path Configuration")
    print("=" * 50)
    print(f"System:      {platform.system()}")
    print(f"Google Drive: {find_google_drive() or 'NOT FOUND'}")
    print(f"OUTPUT_DIR:  {OUTPUT_DIR}")
    print(f"DATA_DIR:    {DATA_DIR}")
    print("=" * 50)


if __name__ == "__main__":
    print_config()
