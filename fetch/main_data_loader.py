
# main_data_loader.py

# --- Initial Setup ---
import os
import sys
import json
import pandas as pd
import yfinance as yf
from datetime import datetime
from fredapi import Fred

# NOTE: Google Drive is expected to be mounted externally in the Colab notebook
# before running this script if executed in a Colab environment.
print("✔ Google Drive assumed mounted.")

# Set FRED API Key from environment or load from env file
# Priority: 1) Already set env var, 2) Load from env file
if "FRED_API_KEY" not in os.environ:
    env_file = os.path.join(os.path.dirname(__file__), 'env')
    if os.path.exists(env_file):
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()
        print("✔ Environment variables loaded from env file.")
    else:
        print("⚠️ No env file found. Please set FRED_API_KEY environment variable.")

# Use FRED_API_KEY (standardized name)
if "FRED_API_KEY" in os.environ:
    os.environ["FRED_API"] = os.environ["FRED_API_KEY"]  # For backward compatibility
    print("✔ FRED_API_KEY loaded successfully.")
else:
    print("⚠️ FRED_API_KEY not set. Data fetching may fail.")

# --- Configuration Paths ---
# Detect if running in Colab or locally
try:
    import google.colab
    IN_COLAB = True
    BASE_DIR = "/content/drive/MyDrive/financial_engine"
except ImportError:
    IN_COLAB = False
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Allow override via environment variable
BASE_DIR = os.environ.get("FINANCIAL_ENGINE_BASE_DIR", BASE_DIR)

DATA_DIR = os.path.join(BASE_DIR, "data_raw")
REGISTRY_PATH = os.path.join(BASE_DIR, "registry", "prism_metric_registry.json")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(os.path.dirname(REGISTRY_PATH), exist_ok=True)

print(f"✔ Base directory: {BASE_DIR}")
print(f"✔ Data directory: {DATA_DIR}")
print(f"✔ Registry path: {REGISTRY_PATH}")


# --- Create / Correct prism_metric_registry.json ---
print("\nCreating/updating prism_metric_registry.json...")
registry_data = [
    {"key": "dgs10", "source": "fred", "ticker": "DGS10"},
    {"key": "dgs2", "source": "fred", "ticker": "DGS2"},
    {"key": "dgs3mo", "source": "fred", "ticker": "DGS3MO"},
    {"key": "t10y2y", "source": "fred", "ticker": "T10Y2Y"},
    {"key": "t10y3m", "source": "fred", "ticker": "T10Y3M"},
    {"key": "cpi", "source": "fred", "ticker": "CPIAUCSL"},
    {"key": "cpi_core", "source": "fred", "ticker": "CPILFESL"},
    {"key": "ppi", "source": "fred", "ticker": "PPIACO"},
    {"key": "unrate", "source": "fred", "ticker": "UNRATE"},
    {"key": "payrolls", "source": "fred", "ticker": "PAYEMS"},
    {"key": "industrial_production", "source": "fred", "ticker": "INDPRO"},
    {"key": "housing_starts", "source": "fred", "ticker": "HOUST"},
    {"key": "permits", "source": "fred", "ticker": "PERMIT"},
    {"key": "m2", "source": "fred", "ticker": "M2SL"},
    {"key": "fed_balance_sheet", "source": "fred", "ticker": "WALCL"},
    {"key": "anfci", "source": "fred", "ticker": "ANFCI"},
    {"key": "nfci", "source": "fred", "ticker": "NFCI"},
    {"key": "bcom", "source": "yahoo", "ticker": "BCOM"},
    {"key": "spy", "source": "yahoo", "ticker": "SPY"},
    {"key": "qqq", "source": "yahoo", "ticker": "QQQ"},
    {"key": "iwm", "source": "yahoo", "ticker": "IWM"},
    {"key": "dxy", "source": "yahoo", "ticker": "DXY"},
    {"key": "vix", "source": "yahoo", "ticker": "VIX"},
    {"key": "gld", "source": "yahoo", "ticker": "GLD"},
    {"key": "slv", "source": "yahoo", "ticker": "SLV"},
    {"key": "uso", "source": "yahoo", "ticker": "USO"},
    {"key": "bnd", "source": "yahoo", "ticker": "BND"},
    {"key": "tlt", "source": "yahoo", "ticker": "TLT"},
    {"key": "shy", "source": "yahoo", "ticker": "SHY"},
    {"key": "ief", "source": "yahoo", "ticker": "IEF"},
    {"key": "tip", "source": "yahoo", "ticker": "TIP"},
    {"key": "lqd", "source": "yahoo", "ticker": "LQD"},
    {"key": "hyg", "source": "yahoo", "ticker": "HYG"},
    {"key": "xlu", "source": "yahoo", "ticker": "XLU"} # Added XLU here
]

with open(REGISTRY_PATH, "w") as f:
    json.dump(registry_data, f, indent=4)
print("✔ prism_metric_registry.json written successfully!")


# --- CoreDataLoader Class Definition (from the latest patch) ---
class CoreDataLoader:

    def __init__(self):
        self.fred = None
        if os.path.exists(REGISTRY_PATH):
            self.registry = pd.read_json(REGISTRY_PATH)
        else:
            self.registry = None

    def init_fred(self):
        # FRED_API key is expected to be set in os.environ before running this script
        api_key = os.environ.get("FRED_API")

        if api_key is None:
            raise ValueError("⚠️ FRED_API not set. Please ensure it's set as an environment variable.")
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


# --- Execute Data Loading and Build Master Panel ---
print("\nStarting data loading process...")
loader = CoreDataLoader()
panel = loader.fetch_all()

print("\n--- Raw Panel Head ---")
print(panel.head().to_string())

print("\n--- Missing Values in Raw Panel ---")
print(panel.isnull().sum().to_string())

# --- Imputation ---
print("\nApplying forward fill (ffill) for missing values...")
panel_ffilled = panel.ffill()
print("✔ Forward fill applied.")

print("\nApplying backward fill (bfill) for remaining leading NaNs...")
panel_filled = panel_ffilled.bfill()
print("✔ Backward fill applied.")

print("\n--- Panel DataFrame after Imputation Head ---")
print(panel_filled.head().to_string())

print("\n--- Missing Values after Imputation (should be 0) ---")
print(panel_filled.isnull().sum().to_string())

# --- Slicing ---
print("\nSlicing panel from '1975-01-01' onwards...")
panel_filled['date'] = pd.to_datetime(panel_filled['date']) # Ensure date column is datetime
panel_sliced = panel_filled[panel_filled['date'] >= '1975-01-01'].copy()
print("✔ Panel sliced.")

# --- Final Verification ---
print("\n--- Final Panel Sliced Head ---")
print(panel_sliced.head().to_string())

print("\n--- Shape of Sliced Panel ---")
print(panel_sliced.shape)

print("\n--- Missing Values after Slicing (should be 0) ---")
print(panel_sliced.isnull().sum().to_string())
