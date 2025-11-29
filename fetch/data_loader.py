"""PRISM Data Loader"""

import os
import sys
import json
import pandas as pd
from fredapi import Fred
import yfinance as yf
from typing import Optional

# Import config - handle both relative and absolute imports
try:
    from .config import DATA_RAW, METRIC_REGISTRY
except ImportError:
    # Add parent directory to path for standalone execution
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from config import DATA_RAW, METRIC_REGISTRY

def load_registry():
    if not METRIC_REGISTRY.exists():
        raise FileNotFoundError(f"Registry missing: {METRIC_REGISTRY}")
    with open(METRIC_REGISTRY, "r") as f:
        return json.load(f)

class PRISMDataLoader:
    def __init__(self, fred_api_key: Optional[str] = None):
        if fred_api_key is None:
            fred_api_key = os.getenv("FRED_API_KEY")
        self.fred = Fred(api_key=fred_api_key) if fred_api_key else None
        
    def fetch_fred(self, ticker: str):
        if not self.fred:
            raise ValueError("FRED API key not set")
        series = self.fred.get_series(ticker)
        df = series.to_frame("value")
        df.index.name = "date"
        return df.reset_index()
    
    def fetch_yahoo(self, ticker: str):
        df = yf.download(ticker, progress=False)
        if df.empty:
            raise ValueError(f"No data for {ticker}")
        if "Adj Close" in df.columns:
            df = df[["Adj Close"]].rename(columns={"Adj Close": "value"})
        else:
            df = df[["Close"]].rename(columns={"Close": "value"})
        df.index.name = "date"
        return df.reset_index()
