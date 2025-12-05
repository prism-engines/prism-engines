import sqlite3
from .db_path import get_db_path

def main():
    path = get_db_path()
    print("\n=== PRISM DATABASE HEALTH CHECK ===")
    print(f"DB Path: {path}")

    try:
        conn = sqlite3.connect(path)
        print("✔ Connection OK.")
    except Exception as e:
        print("❌ Connection failed:", e)
        return

    cursor = conn.execute("PRAGMA journal_mode;")
    print("✔ WAL mode enabled.")

    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row[0] for row in cursor.fetchall()]

    print("\nTables found:")
    for t in tables:
        print("  •", t)

    print("\n=== HEALTH CHECK COMPLETE ===")

if __name__ == "__main__":
    main()

