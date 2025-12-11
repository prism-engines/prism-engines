import os
from pathlib import Path

def get_db_path() -> Path:
    """
    Return the path to the primary Prism database.
    """
    base_dir = Path(os.environ.get("PRISM_DB_DIR", str(Path.home() / "prism_data")))
    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir / "prism.db"

