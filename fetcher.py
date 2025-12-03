"""
PRISM Data Fetcher
==================

Downloads a comprehensive market universe covering:
- Broad indices
- All 11 sectors
- Factors (momentum, value, size, quality)
- Fixed income (duration spectrum)
- Commodities
- International
- Volatility
- Macro indicators (from FRED)

Usage:
    # Import directly
    from fetcher import fetch_all, fetch_equities, fetch_macro

    # Or run the file
    panel = fetch_all()  # Downloads everything

Selective fetching:
    fetch_equities()
    fetch_fixed_income()
    fetch_macro()
"""

import pandas as pd
import numpy as np
import os
import warnings
from datetime import datetime, timedelta

# Suppress yfinance FutureWarnings
warnings.filterwarnings('ignore', category=FutureWarning, module='yfinance')

# Try to import yfinance
try:
    import yfinance as yf
    HAS_YF = True
except ImportError:
    HAS_YF = False
    print("Installing yfinance...")
    os.system('pip install yfinance -q')
    import yfinance as yf
    HAS_YF = True

# Try pandas_datareader for FRED
try:
    from pandas_datareader import data as pdr
    HAS_PDR = True
except ImportError:
    HAS_PDR = False
    print("Installing pandas-datareader...")
    os.system('pip install pandas-datareader -q')
    from pandas_datareader import data as pdr
    HAS_PDR = True


# =============================================================================
# UNIVERSE DEFINITIONS
# =============================================================================

UNIVERSE = {
    # BROAD MARKET
    'broad': {
        'SPY': 'S&P 500',
        'QQQ': 'Nasdaq 100',
        'IWM': 'Russell 2000',
        'DIA': 'Dow 30',
        'VTI': 'Total US Market',
    },
    
    # SECTORS (all 11 GICS sectors)
    'sectors': {
        'XLK': 'Technology',
        'XLF': 'Financials',
        'XLE': 'Energy',
        'XLV': 'Healthcare',
        'XLI': 'Industrials',
        'XLP': 'Consumer Staples',
        'XLY': 'Consumer Discretionary',
        'XLU': 'Utilities',           # Your 20%! 🔌
        'XLB': 'Materials',
        'XLRE': 'Real Estate',
        'XLC': 'Communications',
    },
    
    # FACTORS
    'factors': {
        'MTUM': 'Momentum',
        'VLUE': 'Value',
        'SIZE': 'Size',
        'QUAL': 'Quality',
        'USMV': 'Low Volatility',
    },
    
    # FIXED INCOME (duration spectrum)
    'fixed_income': {
        'SHY': 'Treasury 1-3yr',
        'IEF': 'Treasury 7-10yr',
        'TLT': 'Treasury 20+yr',
        'TIP': 'TIPS (Inflation)',
        'LQD': 'Investment Grade Corp',
        'HYG': 'High Yield Corp',
        'AGG': 'Total Bond Market',
        'BND': 'Total Bond (Vanguard)',
        'EMB': 'Emerging Mkt Bonds',
    },
    
    # COMMODITIES
    'commodities': {
        'GLD': 'Gold',
        'SLV': 'Silver',
        'USO': 'Oil',
        'UNG': 'Natural Gas',
        'DBA': 'Agriculture',
        'DBB': 'Base Metals',
        'PDBC': 'Commodity Basket',
    },
    
    # INTERNATIONAL
    'international': {
        'EFA': 'Developed Markets',
        'EEM': 'Emerging Markets',
        'VEU': 'All World ex-US',
        'FXI': 'China',
        'EWJ': 'Japan',
        'EWG': 'Germany',
        'EWU': 'UK',
    },
    
    # VOLATILITY & ALTERNATIVES
    'volatility': {
        'VXX': 'VIX Short-term',
        'VIXY': 'VIX Futures',
    },
    
    # CURRENCIES (via ETFs)
    'currencies': {
        'UUP': 'US Dollar Bull',
        'FXE': 'Euro',
        'FXY': 'Yen',
        'FXB': 'British Pound',
    },
    
    # THEMATIC (Optional - comment out if not needed)
    'thematic': {
        'ARKK': 'Innovation',
        'ICLN': 'Clean Energy',
        'TAN': 'Solar',
        'LIT': 'Lithium/Battery',
        'SOXX': 'Semiconductors',
        'XBI': 'Biotech',
        'KRE': 'Regional Banks',
        'XHB': 'Homebuilders',
    },
}

# FRED Macro Series
FRED_SERIES = {
    # Interest Rates
    'DGS10': '10Y Treasury Yield',
    'DGS2': '2Y Treasury Yield',
    'DGS3MO': '3M Treasury Yield',
    'T10Y2Y': '10Y-2Y Spread',
    'T10Y3M': '10Y-3M Spread',
    
    # Inflation
    'CPIAUCSL': 'CPI All Urban',
    'CPILFESL': 'Core CPI',
    'PPIACO': 'PPI All Commodities',
    
    # Employment
    'UNRATE': 'Unemployment Rate',
    'PAYEMS': 'Nonfarm Payrolls',
    'ICSA': 'Initial Claims',
    
    # Growth
    'INDPRO': 'Industrial Production',
    'HOUST': 'Housing Starts',
    'PERMIT': 'Building Permits',
    'RSAFS': 'Retail Sales',
    
    # Money & Credit
    'M2SL': 'M2 Money Supply',
    'WALCL': 'Fed Balance Sheet',
    
    # Financial Conditions
    'NFCI': 'Chicago Fed Financial Conditions',
    'ANFCI': 'Adjusted NFCI',
    'BAMLH0A0HYM2': 'High Yield Spread',
    
    # Dollar
    'DTWEXBGS': 'Trade Weighted Dollar',
}


# =============================================================================
# FETCHER FUNCTIONS
# =============================================================================

def fetch_yahoo(tickers: list, start: str = '2000-01-01', end: str = None) -> pd.DataFrame:
    """Fetch data from Yahoo Finance."""
    if end is None:
        end = datetime.now().strftime('%Y-%m-%d')
    
    print(f"Fetching {len(tickers)} tickers from Yahoo Finance...")
    
    data = {}
    failed = []
    
    for i, ticker in enumerate(tickers):
        try:
            df = yf.download(ticker, start=start, end=end, progress=False)
            if len(df) > 0:
                # Handle new yfinance MultiIndex format (v0.2.40+)
                # auto_adjust is now True by default, so use 'Close' instead of 'Adj Close'
                if isinstance(df.columns, pd.MultiIndex):
                    data[ticker] = df['Close'][ticker]
                else:
                    data[ticker] = df['Close']
                print(f"  ✓ {ticker} ({len(df)} days)", end='\r')
            else:
                failed.append(ticker)
        except Exception as e:
            failed.append(ticker)
    
    print(f"\n  Downloaded {len(data)}/{len(tickers)} tickers")
    if failed:
        print(f"  Failed: {failed}")
    
    return pd.DataFrame(data)


def fetch_fred(series: list, start: str = '2000-01-01', end: str = None) -> pd.DataFrame:
    """Fetch data from FRED."""
    if end is None:
        end = datetime.now().strftime('%Y-%m-%d')
    
    print(f"Fetching {len(series)} series from FRED...")
    
    data = {}
    failed = []
    
    for s in series:
        try:
            df = pdr.DataReader(s, 'fred', start, end)
            if len(df) > 0:
                data[s] = df.iloc[:, 0]
                print(f"  ✓ {s}", end='\r')
            else:
                failed.append(s)
        except Exception as e:
            failed.append(s)
    
    print(f"\n  Downloaded {len(data)}/{len(series)} series")
    if failed:
        print(f"  Failed: {failed}")
    
    return pd.DataFrame(data)


# =============================================================================
# HIGH-LEVEL FETCH FUNCTIONS
# =============================================================================

def fetch_equities(start: str = '2000-01-01', categories: list = None) -> pd.DataFrame:
    """
    Fetch equity ETFs.
    
    Args:
        start: Start date
        categories: List of categories to fetch (default: all)
                   Options: 'broad', 'sectors', 'factors', 'international', 'thematic'
    """
    if categories is None:
        categories = ['broad', 'sectors', 'factors', 'international', 'thematic']
    
    tickers = []
    for cat in categories:
        if cat in UNIVERSE:
            tickers.extend(UNIVERSE[cat].keys())
    
    return fetch_yahoo(tickers, start=start)


def fetch_fixed_income(start: str = '2000-01-01') -> pd.DataFrame:
    """Fetch fixed income ETFs."""
    tickers = list(UNIVERSE['fixed_income'].keys())
    return fetch_yahoo(tickers, start=start)


def fetch_commodities(start: str = '2000-01-01') -> pd.DataFrame:
    """Fetch commodity ETFs."""
    tickers = list(UNIVERSE['commodities'].keys())
    return fetch_yahoo(tickers, start=start)


def fetch_currencies(start: str = '2000-01-01') -> pd.DataFrame:
    """Fetch currency ETFs."""
    tickers = list(UNIVERSE['currencies'].keys())
    return fetch_yahoo(tickers, start=start)


def fetch_volatility(start: str = '2010-01-01') -> pd.DataFrame:
    """Fetch volatility products (shorter history)."""
    tickers = list(UNIVERSE['volatility'].keys())
    return fetch_yahoo(tickers, start=start)


def fetch_macro(start: str = '2000-01-01') -> pd.DataFrame:
    """Fetch macro data from FRED."""
    series = list(FRED_SERIES.keys())
    return fetch_fred(series, start=start)


def fetch_all(start: str = '2005-01-01', save_path: str = None) -> pd.DataFrame:
    """
    Fetch everything and combine into one panel.
    
    Args:
        start: Start date (2005 default for better ETF coverage)
        save_path: Path to save CSV (optional)
    
    Returns:
        Combined DataFrame with all data
    """
    print("="*50)
    print("PRISM DATA FETCHER")
    print("="*50)
    print(f"Fetching data from {start} to today\n")
    
    # Fetch all categories
    equities = fetch_equities(start)
    fixed_income = fetch_fixed_income(start)
    commodities = fetch_commodities(start)
    currencies = fetch_currencies(start)
    
    # Combine market data
    market_data = pd.concat([equities, fixed_income, commodities, currencies], axis=1)
    
    # Fetch macro
    macro = fetch_macro(start)
    
    # Macro is often daily but with gaps - forward fill
    macro = macro.ffill()
    
    # Align to market data dates
    combined = market_data.join(macro, how='left')
    combined = combined.ffill()
    
    print(f"\n{'='*50}")
    print(f"COMPLETE: {combined.shape[1]} indicators, {combined.shape[0]} days")
    print(f"Date range: {combined.index[0]} to {combined.index[-1]}")
    print(f"{'='*50}")
    
    # Save if path provided
    if save_path:
        combined.to_csv(save_path)
        print(f"\n✓ Saved to {save_path}")
    
    return combined


def fetch_custom(tickers: list, start: str = '2000-01-01') -> pd.DataFrame:
    """
    Fetch custom list of tickers.
    
    Usage:
        my_data = fetch_custom(['AAPL', 'MSFT', 'GOOGL', 'XLU'])
    """
    return fetch_yahoo(tickers, start=start)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def list_universe():
    """Print the full universe."""
    total = 0
    print("\nPRISM UNIVERSE")
    print("="*50)
    for category, tickers in UNIVERSE.items():
        print(f"\n{category.upper()} ({len(tickers)})")
        for ticker, name in tickers.items():
            print(f"  {ticker:<8} {name}")
        total += len(tickers)
    print(f"\n{'='*50}")
    print(f"Total ETFs: {total}")
    print(f"FRED series: {len(FRED_SERIES)}")


def add_ticker(category: str, ticker: str, name: str):
    """
    Add a ticker to the universe.
    
    Usage:
        add_ticker('sectors', 'XLU', 'Utilities')  # Already there!
        add_ticker('thematic', 'QCLN', 'Clean Energy')
    """
    if category not in UNIVERSE:
        UNIVERSE[category] = {}
    UNIVERSE[category][ticker] = name
    print(f"✓ Added {ticker} ({name}) to {category}")


def quick_fetch(save: bool = True) -> pd.DataFrame:
    """
    Quick fetch with sensible defaults.
    Saves to data/raw/master_panel.csv in PRISM folder.
    """
    from pathlib import Path

    # Find PRISM root relative to this file
    if '__file__' in dir():
        script_dir = Path(__file__).parent.resolve()
    else:
        script_dir = Path('.').resolve()

    # Look for data/raw directory relative to script location
    possible_roots = [
        script_dir,  # If fetcher.py is in prism-engine root
        script_dir.parent,  # If fetcher.py is in a subdirectory
        Path('.').resolve(),  # Current working directory
    ]

    save_path = None
    if save:
        for root in possible_roots:
            raw_dir = root / 'data' / 'raw'
            if raw_dir.exists():
                save_path = str(raw_dir / 'master_panel.csv')
                break

        if save_path is None:
            # Create data/raw in script directory if it doesn't exist
            raw_dir = script_dir / 'data' / 'raw'
            raw_dir.mkdir(parents=True, exist_ok=True)
            save_path = str(raw_dir / 'master_panel.csv')

    return fetch_all(start='2010-01-01', save_path=save_path)


# =============================================================================
# PRINT SUMMARY
# =============================================================================

if __name__ != '__main__':
    # When exec'd or imported
    total_etfs = sum(len(v) for v in UNIVERSE.values())
    print(f"✓ Fetcher loaded: {total_etfs} ETFs + {len(FRED_SERIES)} FRED series")
    print()
    print("Quick start:")
    print("  panel = quick_fetch()        # Fetch everything & save")
    print("  panel = fetch_equities()     # Just equities")
    print("  panel = fetch_custom(['SPY', 'XLU', 'GLD'])")
    print()
    print("  list_universe()              # See all available tickers")
    print("  add_ticker('thematic', 'ARKK', 'Innovation')  # Add custom")
