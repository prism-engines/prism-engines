from fetch import YahooFetcher

print("=== YAHOO FETCHER TEST ===")

fetcher = YahooFetcher()

df = fetcher.fetch_single(
    "SPY",
    start_date="2020-01-01",
    end_date=None
)

print("\nHEAD:")
print(df.head())

print("\nTAIL:")
print(df.tail())

print("\nSHAPE:", df.shape)
print("COLUMNS:", df.columns.tolist())
print("DATERANGE:", df["date"].min(), "→", df["date"].max())

print("\n=== TEST COMPLETE ===")
