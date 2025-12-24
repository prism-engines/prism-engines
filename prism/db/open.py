import os
from pathlib import Path
import duckdb

REPO_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = Path(
    os.environ.get(
        "PRISM_DUCKDB_PATH",
        REPO_ROOT / "data" / "prism.duckdb"
    )
)
EXPECTED_SCHEMA_VERSION = 1  # increment only when schema changes


def open_prism_db():
    # --- existence check (never create implicitly) ---
    if not DB_PATH.exists():
        raise RuntimeError(
            "PRISM DB does not exist.\n"
            "Run: prism-db-build"
        )

    con = duckdb.connect(DB_PATH)

    # --- schema-aware table check ---
    tables = con.execute("""
        SELECT table_schema, table_name
        FROM information_schema.tables
        WHERE table_schema NOT IN ('information_schema', 'pg_catalog')
    """).fetchall()

    if not tables:
        raise RuntimeError(
            "PRISM DB exists but has no user tables.\n"
            "Run: prism-db-build"
        )

    # --- schema version check ---
    try:
        version = con.execute(
            "SELECT schema_version FROM meta.schema_version"
        ).fetchone()
    except duckdb.CatalogException:
        raise RuntimeError(
            "PRISM DB missing meta.schema_version.\n"
            "Run: prism-db-build"
        )

    if version is None or version[0] != EXPECTED_SCHEMA_VERSION:
        raise RuntimeError(
            f"Schema mismatch (found {version}, expected {EXPECTED_SCHEMA_VERSION}).\n"
            "Run: prism-db-build"
        )

    return con
