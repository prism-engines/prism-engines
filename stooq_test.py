# stooq_test.py
# PRISM – Stooq Fetcher Test

from fetch.fetcher_stooq import StooqFetcher
from data.sql.prism_db import write_dataframe, query

print("\n=== Testing Stooq Fetch ===")

# IMPORTANT: Stooq requires ticker format like SPY.US, QQQ.US, etc.
ticker = "SPY.US"

fetcher = StooqFetcher()
df = fetcher.fetch_single(ticker)

if df is None or df.empty:
    print("❌ No data returned — check ticker format.")
    exit()

print("\nFetched sample rows:")
print(df.head())

# Add required ticker column before writing into SQL
df["ticker"] = "SPY_STOOQ"

# Reorder for DB schema
df = df[["ticker", "date", "value"]]

print("\n→ Writing SPY_STOOQ into market_prices…")
write_dataframe(df, "market_prices")

print("\n=== DB Rows Inserted (first 5) ===")
print(query("SELECT * FROM market_prices WHERE ticker='SPY_STOOQ' LIMIT 5;"))

print("\n✔ Stooq test completed.")

