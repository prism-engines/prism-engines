import json
import logging
from fetch.hybrid_fetcher import HybridFetcher

logging.basicConfig(level=logging.INFO)

def main():
    fetcher = HybridFetcher()

    with open("data/registry/market_registry.json") as f:
        instruments = json.load(f)["instruments"]

    print("\n============================================")
    print("   PRISM FULL HYBRID FETCH TEST START")
    print("============================================\n")

    for inst in instruments:
        key = inst["key"]
        ticker = inst["ticker"]

        print(f"--- Testing {key.upper()} ({ticker}) ---")

        df = fetcher.fetch_single(ticker)

        if df is None or df.empty:
            print(f"❌ FAIL — {key} returned no data\n")
        else:
            print(f"✅ PASS — {len(df)} rows\n")

    print("\n============================================")
    print("         HYBRID FETCH TEST COMPLETE")
    print("============================================\n")


if __name__ == "__main__":
    main()
