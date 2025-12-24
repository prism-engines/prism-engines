from pathlib import Path
from prism.db.open import open_prism_db

def main():
    con = open_prism_db()

    schema_path = (
        Path(__file__).resolve().parents[1]
        / "prism"
        / "db"
        / "schema.sql"
    )

    with open(schema_path, "r") as f:
        con.execute(f.read())

    print("PRISM database initialized.")

if __name__ == "__main__":
    main()
