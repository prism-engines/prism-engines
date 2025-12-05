import os

def main():
    print("\n=== PRISM DB SANITY CHECK ===")

    mig_dir = os.path.join(os.path.dirname(__file__), "migrations")

    files = sorted(f for f in os.listdir(mig_dir) if f.endswith(".sql"))
    print(f"✔ Found {len(files)} migration files.")

    print("Sanity check complete.")

if __name__ == "__main__":
    main()

