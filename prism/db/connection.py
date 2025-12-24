"""
PRISM Database Connection

DEPRECATED: This module is deprecated. Use prism.db.open instead.

    from prism.db.open import open_prism_db
    conn = open_prism_db()

This module contained runtime schema creation logic which violates the
migrations-only schema policy. All new code should use open_prism_db().
"""

import warnings
import duckdb
from pathlib import Path
from typing import Optional
import logging

from prism.utils.paths import DATA_DIR

logger = logging.getLogger(__name__)

warnings.warn(
    "prism.db.connection is deprecated. Use 'from prism.db.open import open_prism_db' instead.",
    DeprecationWarning,
    stacklevel=2
)


# Canonical database location (absolute path, non-negotiable)
DB_PATH = DATA_DIR / "prism.duckdb"
DEFAULT_DB_PATH = DB_PATH  # Alias for backwards compatibility
SCHEMA_PATH = Path(__file__).resolve().parent / "schema.sql"


def get_db_path() -> Path:
    """
    Return the canonical database path.

    All DuckDB connections must resolve to this path.
    No relative guessing. No alternate fallbacks. No secondary DBs.
    """
    return DEFAULT_DB_PATH


class DatabaseConnection:
    """
    DuckDB connection manager for PRISM.
    
    Usage:
        db = DatabaseConnection()
        conn = db.connect()
        conn.execute("SELECT * FROM data.raw_indicators LIMIT 10")
        
    Or as context manager:
        with DatabaseConnection() as conn:
            conn.execute("SELECT * FROM data.raw_indicators")
    """
    
    def __init__(self, db_path: Optional[Path] = None):
        """
        Initialize database connection.
        
        Args:
            db_path: Path to DuckDB file. Defaults to data/prism.duckdb
        """
        self.db_path = Path(db_path) if db_path else DEFAULT_DB_PATH
        self._connection: Optional[duckdb.DuckDBPyConnection] = None
        
    def connect(self) -> duckdb.DuckDBPyConnection:
        """
        Get database connection. Creates DB and schema if needed.
        
        Returns:
            DuckDB connection object
        """
        if self._connection is not None:
            return self._connection
            
        # Ensure data directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Connect (creates file if doesn't exist)
        self._connection = duckdb.connect(str(self.db_path))
        logger.info(f"Connected to database: {self.db_path}")
        
        # Initialize schema if needed
        self._ensure_schema()
        
        return self._connection
    
    def _ensure_schema(self) -> None:
        """Create schema if tables don't exist."""
        if self._connection is None:
            return
            
        # Check if raw schema exists
        result = self._connection.execute("""
            SELECT COUNT(*) 
            FROM information_schema.schemata 
            WHERE schema_name = 'raw'
        """).fetchone()
        
        if result[0] == 0:
            logger.info("Initializing database schema...")
            self._run_schema_script()
            logger.info("Schema initialized.")
    
    def _run_schema_script(self) -> None:
        """Execute the schema.sql file."""
        if not SCHEMA_PATH.exists():
            raise FileNotFoundError(f"Schema file not found: {SCHEMA_PATH}")
            
        schema_sql = SCHEMA_PATH.read_text()
        
        # DuckDB can execute multiple statements
        self._connection.execute(schema_sql)
    
    def close(self) -> None:
        """Close the database connection."""
        if self._connection is not None:
            self._connection.close()
            self._connection = None
            logger.info("Database connection closed.")
    
    def __enter__(self) -> duckdb.DuckDBPyConnection:
        """Context manager entry."""
        return self.connect()
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()


# Convenience function for quick access
def get_connection(db_path: Optional[Path] = None, read_only: bool = False) -> duckdb.DuckDBPyConnection:
    """
    Get a database connection.

    Args:
        db_path: Optional path to database file
        read_only: If True, open in read-only mode (no schema init)

    Returns:
        DuckDB connection

    Note:
        Caller is responsible for closing the connection.
        For automatic cleanup, use DatabaseConnection as context manager.
    """
    if read_only:
        # Read-only connections skip schema initialization
        path = Path(db_path) if db_path else DB_PATH
        return duckdb.connect(str(path), read_only=True)

    db = DatabaseConnection(db_path)
    return db.connect()

# Canonical public API (preferred)
def connect(read_only: bool = False):
    """
    Return a DuckDB connection using the canonical local database.
    """
    return get_connection(read_only=read_only)
