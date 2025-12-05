import yfinance as yf
from data.sql.db import connect, add_indicator

print("=== PRISM YAHOO FETCH TEST (SPY → indicator_values) ===")

# 1. Ensure indicator metadata exists
indicator = "SPY"
system = "market"

indicator_id = add_indicator(
    name=indicator,
    system=system,
    frequency="daily",
    source="yahoo",
    units="price",
    description="SPY daily close from Yahoo"
)
print(f"Registered indicator '{indicator}' / '{system}' with ID:", indicator_id)

# 2. Fetch last 60 days of SPY data
df = yf.download("SPY", period="60d", interval="1d")
print("Fetched rows:", len(df))
print(df.tail())

# 3. Flatten MultiIndex columns (level 0 only: Close, High, Low, Open, Volume)
df.columns = df.columns.get_level_values(0)

# 4. Extract Close column
close_df = df[["Close"]].reset_index()
close_df["date"] = close_df["Date"].dt.strftime("%Y-%m-%d")
close_df = close_df[["date", "Close"]].rename(columns={"Close": "value"})

# 5. Insert into indicator_values table
conn = connect()
rows = 0
for _, row in close_df.iterrows():
    conn.execute(
        """
        INSERT OR REPLACE INTO indicator_values
            (indicator, system, date, value, value_2, adjusted_value)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (indicator, system, row["date"], float(row["value"]), None, None)
    )
    rows += 1

conn.commit()
conn.close()

print(f"Inserted {rows} rows into indicator_values for {indicator}/{system}.")
print("=== YAHOO FETCH TEST COMPLETE ===")
