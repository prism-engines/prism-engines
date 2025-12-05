import os
from .db_path import get_db_path
from .prism_db import initialize_db, run_all_migrations

def main():
    db_path = get_db_path()
    if os.path.exists(db_path):
        print(f"Removing old DB at {db_path}")
        os.remove(db_path)

    print("Creating fresh Prism DB…")
    initialize_db()

    print("Running migrations…")
    run_all_migrations()

    print("\n✔ Prism DB rebuilt successfully")
    print(f"→ Location: {db_path}")

if __name__ == "__main__":
    main()

