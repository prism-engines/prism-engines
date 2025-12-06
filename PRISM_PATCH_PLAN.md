# PRISM Engine Patch Plan
## Critical Fixes for Data Fetching Pipeline

**Date:** December 5, 2025  
**Total Issues:** 9 bugs identified  
**Priority:** High - these block data fetching entirely

---

## Summary of Issues

| # | Issue | Severity | File(s) Affected |
|---|-------|----------|------------------|
| 1 | `fetch_all()` returns `None` → crashes `len()` | 🔴 Critical | `fetcher_yahoo.py`, `start/fetcher.py` |
| 2 | Registry uses `id`, fetcher expects `symbol` | 🔴 Critical | `fetcher_yahoo.py` |
| 3 | Failed tickers cause early exit | 🟠 High | `fetcher_yahoo.py` |
| 4 | Default interval may be `1m` (intraday) | 🟠 High | `fetcher_yahoo.py` |
| 5 | PyYAML missing from requirements | 🟡 Medium | `requirements.txt` |
| 6 | Fetcher interface mismatch | 🟠 High | `start/fetcher.py` |
| 7 | Logging assumes non-None results | 🟡 Medium | `start/fetcher.py` |
| 8 | Fallback ticker map not applied | 🟡 Medium | `fetcher_yahoo.py` |
| 9 | Missing dependencies in requirements | 🟡 Medium | `requirements.txt` |

---

## Fix 1: `fetch_all()` Must Always Return Dict (Never None)

**Problem:** When all tickers fail, `fetch_all()` returns `None`. Then `start/fetcher.py` calls `len(results)` which crashes with `TypeError: object of type 'NoneType' has no len()`.

**File:** `fetch/fetcher_yahoo.py`

**Current code (line ~250):**
```python
return merged
```

**Fixed code:**
```python
return merged if merged is not None else pd.DataFrame()
```

Also fix the early returns at lines 208 and 218 - they return `None` but should return empty DataFrame:

```python
# Line 208: change
return None
# to:
return pd.DataFrame()

# Line 218: change  
return None
# to:
return pd.DataFrame()
```

---

## Fix 2: Extract Ticker from Registry Properly

**Problem:** YAML registry stores ticker in `params.ticker`, but `fetch_all()` looks for `item["symbol"]` or `item["ticker"]` at the top level.

**File:** `fetch/fetcher_yahoo.py`

**Current code (lines 210-214):**
```python
tickers = [
    item.get("symbol") or item.get("ticker")
    for item in registry["market"]
    if item.get("symbol") or item.get("ticker")
]
```

**Fixed code:**
```python
tickers = []
for item in registry["market"]:
    # Try multiple locations where ticker might be stored
    ticker = (
        item.get("ticker") or 
        item.get("symbol") or 
        item.get("params", {}).get("ticker") or
        item.get("name", "").upper()
    )
    if ticker:
        tickers.append(ticker)
```

---

## Fix 3: Handle Failed Tickers Gracefully

**Problem:** When tickers like `^GVZ`, `ZW=F` fail, the whole batch might abort.

**File:** `fetch/fetcher_yahoo.py`

The existing fallback logic (lines 233-248) is correct, but we need to ensure it always returns a dict, not None. The fix from #1 handles this.

**Additional improvement** - add known-bad ticker filtering:

```python
# Add near top of file (after imports):
KNOWN_DELISTED_TICKERS = {
    "^GVZ",  # Gold Volatility Index - often unavailable
    # Add others as discovered
}

# Then in fetch_all, filter before fetching:
tickers = [t for t in tickers if t not in KNOWN_DELISTED_TICKERS]
```

---

## Fix 4: Force Daily Interval by Default

**Problem:** Default interval might be `1m` which pulls massive intraday data.

**File:** `fetch/fetcher_yahoo.py`

The signature already defaults to `"1d"` at line 191:
```python
def fetch_all(
    self,
    registry: dict,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    interval: str = "1d",  # ✅ Already correct
    **kwargs
)
```

**Verify:** Check that `fetch_single()` also defaults to `"1d"` (it does at line 44). ✅ No change needed.

---

## Fix 5: Add PyYAML to Requirements

**Problem:** YAML registry loading fails without PyYAML installed.

**File:** `requirements.txt`

**Add these lines:**
```
pyyaml>=6.0
```

---

## Fix 6: Fetcher Interface is Already Aligned

**Problem reported:** `fetch_all()` signature mismatch.

**Actual status:** Looking at the code, `start/fetcher.py` correctly passes `registry` at line 56-61:
```python
results = fetcher.fetch_all(
    registry=registry,
    start_date=start_date,
    end_date=end_date,
    write_to_db=write_to_db
)
```

⚠️ **BUT** - `YahooFetcher.fetch_all()` doesn't accept `write_to_db` parameter! This will be silently ignored via `**kwargs` but it's confusing.

**Recommended:** Either add `write_to_db` support to `YahooFetcher.fetch_all()` or remove it from the call in `start/fetcher.py`.

---

## Fix 7: Safe Logging for Possibly-None Results

**Problem:** `len(results)` crashes when results is `None`.

**File:** `start/fetcher.py`

**Current code (line 63):**
```python
logger.info(f"Market fetch complete: {len(results)} instruments")
```

**Fixed code:**
```python
count = len(results) if results is not None else 0
logger.info(f"Market fetch complete: {count} instruments")
```

**Also fix line 90:**
```python
count = len(results) if results is not None else 0
logger.info(f"Economic fetch complete: {count} series")
```

**And lines 266-267:**
```python
market_count = len(results['market']) if results.get('market') else 0
econ_count = len(results['economic']) if results.get('economic') else 0
logger.info(f"Market instruments: {market_count}")
logger.info(f"Economic series: {econ_count}")
```

---

## Fix 8: Apply Fallback Ticker Map

**Problem:** `yahoo_fallback_map.json` exists but isn't used.

**File:** `fetch/fetcher_yahoo.py`

**Add this near the top of the file:**
```python
import json

def load_fallback_map():
    """Load ticker fallback mappings."""
    fallback_path = Path(__file__).parent.parent / "data" / "registry" / "yahoo_fallback_map.json"
    if fallback_path.exists():
        with open(fallback_path) as f:
            return json.load(f)
    return {}

FALLBACK_MAP = load_fallback_map()
```

**Then in `fetch_single()`, add ticker remapping (around line 61):**
```python
def fetch_single(
    self,
    ticker: str,
    ...
) -> Optional[pd.DataFrame]:
    # Apply fallback mapping if available
    original_ticker = ticker
    ticker = FALLBACK_MAP.get(ticker, ticker)
    if ticker != original_ticker:
        logger.info(f"Remapped ticker {original_ticker} -> {ticker}")
    
    # ... rest of method
```

---

## Fix 9: Complete Requirements.txt

**File:** `requirements.txt`

**Add these missing dependencies:**
```
pyyaml>=6.0
tenacity>=8.0
```

---

## Complete Patched Files

Below are the complete fixed files ready to copy:

### Fixed `fetch/fetcher_yahoo.py` (key sections)

```python
"""
Yahoo Finance Fetcher - Stock, ETF, and market data
"""

from pathlib import Path
from typing import Optional, Any, List
import pandas as pd
import logging
import json

from .fetcher_base import BaseFetcher

logger = logging.getLogger(__name__)


def _load_fallback_map():
    """Load ticker fallback mappings."""
    fallback_path = Path(__file__).parent.parent / "data" / "registry" / "yahoo_fallback_map.json"
    if fallback_path.exists():
        try:
            with open(fallback_path) as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load fallback map: {e}")
    return {}


FALLBACK_MAP = _load_fallback_map()

# Tickers known to be problematic
KNOWN_PROBLEMATIC_TICKERS = {
    "^GVZ",  # Gold Volatility - often unavailable
}


class YahooFetcher(BaseFetcher):
    """
    Fetcher for Yahoo Finance data.
    """

    def __init__(self, checkpoint_dir: Optional[Path] = None):
        super().__init__(checkpoint_dir)

    def validate_response(self, response: Any) -> bool:
        """Validate Yahoo Finance response."""
        if response is None:
            return False
        if isinstance(response, pd.DataFrame) and response.empty:
            return False
        return True

    def fetch_single(
        self,
        ticker: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        interval: str = "1d",
        **kwargs
    ) -> Optional[pd.DataFrame]:
        """Fetch data for a single Yahoo Finance ticker."""
        import yfinance as yf

        # Apply fallback mapping
        original_ticker = ticker
        ticker = FALLBACK_MAP.get(ticker, ticker)
        if ticker != original_ticker:
            logger.info(f"Remapped ticker {original_ticker} -> {ticker}")

        try:
            df = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                interval=interval,
                auto_adjust=True,
                progress=False
            )

            if not self.validate_response(df):
                logger.warning(f"No data returned for {ticker}")
                return None

            df = df.reset_index()
            column_map = {
                "Date": "date",
                "Open": f"{ticker.lower()}_open",
                "High": f"{ticker.lower()}_high",
                "Low": f"{ticker.lower()}_low",
                "Close": ticker.lower(),
                "Volume": f"{ticker.lower()}_volume"
            }
            df = df.rename(columns=column_map)
            return self.sanitize_dataframe(df, ticker)

        except Exception as e:
            logger.error(f"Yahoo error for {ticker}: {e}")
            return None

    def fetch_multiple(
        self,
        tickers: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        """Fetch multiple tickers efficiently in one call."""
        import yfinance as yf

        # Filter problematic tickers
        tickers = [t for t in tickers if t not in KNOWN_PROBLEMATIC_TICKERS]
        
        if not tickers:
            return pd.DataFrame()

        try:
            df = yf.download(
                tickers,
                start=start_date,
                end=end_date,
                auto_adjust=True,
                progress=False
            )

            if not self.validate_response(df):
                return pd.DataFrame()

            if isinstance(df.columns, pd.MultiIndex):
                df = df["Close"]

            df = df.reset_index()
            df.columns = ["date"] + [t.lower() for t in tickers]
            return df

        except Exception as e:
            logger.error(f"Yahoo batch error: {e}")
            return pd.DataFrame()

    def fetch_all(
        self,
        registry: dict,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        interval: str = "1d",
        **kwargs
    ) -> pd.DataFrame:
        """
        Unified Yahoo Finance fetcher for the entire market registry.
        
        ALWAYS returns a DataFrame (empty if no data), never None.
        """
        if "market" not in registry:
            logger.error("Registry missing 'market' section.")
            return pd.DataFrame()  # Changed from None

        # Extract tickers from various possible locations
        tickers = []
        for item in registry["market"]:
            ticker = (
                item.get("ticker") or 
                item.get("symbol") or 
                item.get("params", {}).get("ticker") or
                item.get("name", "").upper()
            )
            if ticker and ticker not in KNOWN_PROBLEMATIC_TICKERS:
                tickers.append(ticker)

        if not tickers:
            logger.error("No Yahoo tickers found in registry['market']")
            return pd.DataFrame()  # Changed from None

        # Attempt batch fetch first
        try:
            df = self.fetch_multiple(
                tickers=tickers,
                start_date=start_date,
                end_date=end_date
            )
            if not df.empty:
                return df
        except Exception as e:
            logger.warning(f"Batch fetch failed, falling back to single mode: {e}")

        # Fallback: fetch each individually
        merged = None
        for t in tickers:
            df_t = self.fetch_single(
                t,
                start_date=start_date,
                end_date=end_date,
                interval=interval
            )
            if df_t is None:
                logger.warning(f"Skipping ticker (no data): {t}")
                continue

            if merged is None:
                merged = df_t
            else:
                merged = pd.merge(merged, df_t, on="date", how="outer")

        # CRITICAL: Always return DataFrame, never None
        return merged if merged is not None else pd.DataFrame()
```

### Fixed `start/fetcher.py` (key sections)

```python
def fetch_market(registry, start_date=None, end_date=None, write_to_db=True):
    """Fetch all enabled market instruments."""
    logger.info("=" * 60)
    logger.info("FETCHING MARKET DATA")
    logger.info("=" * 60)

    fetcher = YahooFetcher()
    results = fetcher.fetch_all(
        registry=registry,
        start_date=start_date,
        end_date=end_date
        # Note: removed write_to_db as YahooFetcher doesn't use it
    )
    
    # Safe length calculation
    count = len(results) if results is not None and not results.empty else 0
    logger.info(f"Market fetch complete: {count} instruments")
    return results


def fetch_economic(start_date=None, end_date=None, write_to_db=True):
    """Fetch all enabled economic series."""
    logger.info("=" * 60)
    logger.info("FETCHING ECONOMIC DATA")
    logger.info("=" * 60)
    
    fetcher = FREDFetcher()
    results = fetcher.fetch_all(
        write_to_db=write_to_db,
        start_date=start_date,
        end_date=end_date
    )
    
    # Safe length calculation
    count = len(results) if results else 0
    logger.info(f"Economic fetch complete: {count} series")
    return results
```

And in `main()`:
```python
    # Summary - safe length calculations
    logger.info("=" * 60)
    logger.info("FETCH SUMMARY")
    logger.info("=" * 60)
    
    market_count = 0
    if results.get('market') is not None:
        if isinstance(results['market'], pd.DataFrame):
            market_count = len(results['market'].columns) - 1  # subtract date column
        elif isinstance(results['market'], dict):
            market_count = len(results['market'])
    
    econ_count = len(results.get('economic', {}))
    
    logger.info(f"Market instruments: {market_count}")
    logger.info(f"Economic series: {econ_count}")
```

### Fixed `requirements.txt`

```
pandas>=2.0
numpy>=1.26
scipy>=1.11
scikit-learn>=1.3
fredapi>=0.5
yfinance>=0.2.30
requests>=2.31
matplotlib>=3.8
seaborn>=0.13
tqdm>=4.66
pyarrow>=14.0
python-dateutil>=2.8
pyyaml>=6.0
tenacity>=8.0
```

---

## Quick Test After Patching

```bash
cd prism-engine-main
pip install -r requirements.txt
python start/fetcher.py --test
python start/fetcher.py --market --start-date 2024-01-01
```

---

## Notes for Claude Code

When applying these patches:
1. Apply fixes in order (1→9)
2. Test after each major fix
3. The return type change (`None` → `pd.DataFrame()`) is the most critical
4. Registry ticker extraction fix is second priority

