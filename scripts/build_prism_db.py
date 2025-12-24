#!/usr/bin/env python3
"""
PRISM Database Builder

Builds the database from scratch by:
1. Creating all schemas
2. Creating schema_version table
3. Applying all migrations in order

Usage:
    python scripts/build_prism_db.py
"""

from pathlib import Path
import duckdb
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = REPO_ROOT / "data" / "prism.duckdb"
MIGRATIONS_DIR = REPO_ROOT / "prism" / "db" / "migrations"
SCHEMA_VERSION = 1

print("=== PRISM DB BUILD START ===")

# Hard delete
if DB_PATH.exists():
    print("Deleting existing PRISM DB")
    DB_PATH.unlink()

# Ensure data directory exists
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

con = duckdb.connect(str(DB_PATH))

print("Creating schemas")

# All schemas used by PRISM
con.execute("CREATE SCHEMA IF NOT EXISTS meta;")
con.execute("CREATE SCHEMA IF NOT EXISTS raw;")
con.execute("CREATE SCHEMA IF NOT EXISTS clean;")
con.execute("CREATE SCHEMA IF NOT EXISTS data;")
con.execute("CREATE SCHEMA IF NOT EXISTS derived;")
con.execute("CREATE SCHEMA IF NOT EXISTS phase1;")
con.execute("CREATE SCHEMA IF NOT EXISTS domains;")
con.execute("CREATE SCHEMA IF NOT EXISTS structure;")

print("Creating schema version table")

con.execute("""
CREATE TABLE meta.schema_version (
    schema_version INTEGER,
    built_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
""")

con.execute(
    "INSERT INTO meta.schema_version VALUES (?, CURRENT_TIMESTAMP)",
    [SCHEMA_VERSION]
)

# Discover and apply migrations
print(f"\nDiscovering migrations in {MIGRATIONS_DIR}")

if not MIGRATIONS_DIR.exists():
    print(f"ERROR: Migrations directory not found: {MIGRATIONS_DIR}")
    con.close()
    sys.exit(1)

migration_files = sorted(MIGRATIONS_DIR.glob("*.sql"))

if not migration_files:
    print("WARNING: No migration files found")
else:
    print(f"Found {len(migration_files)} migration files")

failed = False
for migration_file in migration_files:
    print(f"  Applying: {migration_file.name}")
    try:
        sql = migration_file.read_text()
        con.execute(sql)
    except Exception as e:
        print(f"  ERROR: Migration failed: {migration_file.name}")
        print(f"         {e}")
        failed = True
        break

if failed:
    print("\n=== PRISM DB BUILD FAILED ===")
    con.close()
    sys.exit(1)

print(f"\nApplied {len(migration_files)} migrations successfully")

con.close()
print("\n=== PRISM DB BUILD COMPLETE ===")
