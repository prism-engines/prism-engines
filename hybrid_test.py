from fetch.fetcher_router import HybridFetcher

hf = HybridFetcher()

print("\n=== Test 1: SPY (should use Stooq) ===")
df1 = hf.fetch_single("SPY.US")
print("Rows:", len(df1))

print("\n=== Test 2: VIX (Stooq missing → Yahoo fallback) ===")
df2 = hf.fetch_single("VIX.US")
print("Rows:", len(df2))
print(df2.head())
