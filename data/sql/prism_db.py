import os
import sqlite3
import pandas as pd
from .db_path import get_db_path


# -------------------------------------------------
# CONNECTION
# -------------------------------------------------

def get_connection():
    """Return a SQLite connection to the Prism DB."""
    path = get_db_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return sqlite3.connect(path)


# -------------------------------------------------
# INITIALIZATION / MIGRATIONS
# -------------------------------------------------

def initialize_db():
    """Create an empty DB and enable WAL mode."""
    conn = get_connection()
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.commit()
    conn.close()


def run_migration_file(path):
    """Execute a single SQL migration file."""
    conn = get_connection()
    with open(path, "r") as f:
        sql = f.read()
    conn.executescript(sql)
    conn.commit()
    conn.close()


def run_all_migrations():
    """Run all migrations found in the migrations folder."""
    base_dir = os.path.dirname(__file__)
    mig_dir = os.path.join(base_dir, "migrations")

    if not os.path.exists(mig_dir):
        print("⚠ No migrations directory found.")
        return

    files = sorted(f for f in os.listdir(mig_dir) if f.endswith(".sql"))
    print(f"Found {len(files)} migrations.")

    for f in files:
        path = os.path.join(mig_dir, f)
        print(f"→ Running {f}")
        run_migration_file(path)

    print("✔ All migrations applied.")


# -------------------------------------------------
# DATA WRITE HELPERS
# -------------------------------------------------

def write_dataframe(df: pd.DataFrame, table: str):
    """
    Write a DataFrame into a SQL table.
    Required columns:
      • market_prices → ticker, date, value
      • econ_values   → series_id, date, value
    """
    conn = get_connection()

    if "date" in df.columns:
        df["date"] = df["date"].astype(str)

    df.to_sql(table, conn, if_exists="append", index=False)
    conn.close()


# -------------------------------------------------
# UNIFIED INDICATOR LOADER
# -------------------------------------------------

def load_indicator(name: str) -> pd.DataFrame:
    """
    Load data from market_prices or econ_values.
    Output: indicator, date, value
    """

    conn = get_connection()

    query = """
        SELECT ticker AS indicator, date, value
        FROM market_prices
        WHERE ticker = ?
        
        UNION ALL
        
        SELECT series_id AS indicator, date, value
        FROM econ_values
        WHERE series_id = ?
        
        ORDER BY date ASC;
    """

    df = pd.read_sql(query, conn, params=[name, name])
    conn.close()
    return df


# -------------------------------------------------
# GENERAL UTILITIES
# -------------------------------------------------

def query(sql: str, params=None) -> pd.DataFrame:
    """Run an arbitrary SQL query and return a DataFrame."""
    conn = get_connection()
    df = pd.read_sql(sql, conn, params=params)
    conn.close()
    return df


def export_to_csv(table: str, filepath: str):
    """Export any SQL table to CSV."""
    conn = get_connection()
    df = pd.read_sql(f"SELECT * FROM {table}", conn)
    df.to_csv(filepath, index=False)
    conn.close()

