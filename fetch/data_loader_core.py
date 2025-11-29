
import os
import pandas as pd
import yfinance as yf
from datetime import datetime
from fredapi import Fred

# Import userdata for Colab secrets
try:
    from google.colab import userdata
except ImportError:
    userdata = None

# Detect if running in Colab or locally
try:
    import google.colab
    IN_COLAB = True
    BASE_DIR = "/content/drive/MyDrive/prism_engine"
except ImportError:
    IN_COLAB = False
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Allow override via environment variable
BASE_DIR = os.environ.get("PRISM_ENGINE_BASE_DIR", BASE_DIR)

DATA_DIR = os.path.join(BASE_DIR, "data_raw")
REGISTRY_PATH = os.path.join(BASE_DIR, "registry", "prism_metric_registry.json")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(os.path.dirname(REGISTRY_PATH), exist_ok=True)

class CoreDataLoader:

    def __init__(self):
        self.fred = None
        if os.path.exists(REGISTRY_PATH):
            # Load registry as a DataFrame directly from the JSON array
            self.registry = pd.read_json(REGISTRY_PATH)
        else:
            self.registry = None

    def init_fred(self):
        api_key = None
        if userdata: # Check if running in Colab and userdata is available
            try:
                api_key = userdata.get("FRED_API_KEY")
            except:
                pass
        if not api_key: # Fallback to os.environ if not found via userdata or not in Colab
            api_key = os.environ.get("FRED_API_KEY") or os.environ.get("FRED_API")

        if api_key is None:
            raise ValueError("⚠️ FRED_API_KEY not set. Please set it in Colab secrets (named FRED_API_KEY) or as an environment variable.")
        self.fred = Fred(api_key)

    # -----------------------------------------
    # UTIL — FIX MULTIINDEX & EMPTY DFS
    # -----------------------------------------
    def _sanitize(self, df, ticker):
        if df is None or len(df) == 0:
            print(f"⚠️ {ticker} returned EMPTY dataset — creating placeholder.")
            return pd.DataFrame({"date": [], ticker.lower(): []})

        # flatten multiindex columns
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = ['_'.join([str(i) for i in col if i]) for col in df.columns]

        # normalize name of price column
        cols = [c.lower() for c in df.columns]
        df.columns = cols

        # enforce 'date'
        if "date" not in df.columns:
            raise ValueError(f"❌ '{ticker}' missing date column")

        return df

    # -----------------------------------------
    # FRED FETCH
    # -----------------------------------------
    def fetch_fred(self, ticker):
        try:
            series = self.fred.get_series(ticker)
            df = pd.DataFrame({"date": series.index, ticker.lower(): series.values})
            df["date"] = pd.to_datetime(df["date"])
            return self._sanitize(df, ticker)
        except Exception as e:
            print(f"❌ FRED error for {ticker}: {e}")
            return None

    # -----------------------------------------
    # YAHOO FETCH
    # -----------------------------------------
    def fetch_yahoo(self, ticker):
        try:
            df = yf.download(ticker, auto_adjust=True, progress=False)

            if df is None or len(df) == 0:
                return self._sanitize(None, ticker)

            df = df.reset_index()
            df = df[["Date", "Close"]].rename(columns={"Date": "date", "Close": ticker.lower()})
            df["date"] = pd.to_datetime(df["date"])

            return self._sanitize(df, ticker)
        except Exception as e:
            print(f"❌ Yahoo error for {ticker}: {e}")
            return self._sanitize(None, ticker)

    # -----------------------------------------
    # SAVE
    # -----------------------------------------
    def save(self, df, name):
        out = f"{DATA_DIR}/{name.lower()}.csv"
        df.to_csv(out, index=False)
        print(f"✔ Saved {name} → {out}")

    # -----------------------------------------
    # BUILD MASTER PANEL
    # -----------------------------------------
    def build_master_panel(self, dfs):
        # Filter out empty dataframes before merging
        dfs = [df for df in dfs if not df.empty]

        if not dfs:
            print("⚠️ No dataframes to merge for master panel.")
            return pd.DataFrame()

        panel = dfs[0]

        for df in dfs[1:]:
            # Ensure both sides only have SINGLE-LEVEL columns
            df.columns = df.columns.astype(str)
            panel.columns = panel.columns.astype(str)

            panel = pd.merge(panel, df, on="date", how="outer")

        panel = panel.sort_values("date")
        panel.columns = [c.lower() for c in panel.columns]
        return panel

    # -----------------------------------------
    # MAIN FETCH ALL
    # -----------------------------------------
    def fetch_all(self):
        if self.registry is None:
            raise ValueError("❌ Registry missing!")

        self.init_fred()

        dfs = []

        for _, row in self.registry.iterrows():
            ticker = row["ticker"]
            source = row["source"].lower()

            if source == "fred":
                df = self.fetch_fred(ticker)
            elif source == "yahoo":
                df = self.fetch_yahoo(ticker)
            else:
                print(f"⚠️ Unknown source for {ticker}")
                continue

            if df is not None and not df.empty:
                self.save(df, ticker)
                dfs.append(df)

        if not dfs:
            print("⚠️ No data fetched. Master panel cannot be built.")
            return pd.DataFrame()

        master = self.build_master_panel(dfs)
        out = f"{DATA_DIR}/master_panel.csv"
        master.to_csv(out, index=False)

        print(f"\n📁 MASTER PANEL CREATED → {out}\n")
        return master
