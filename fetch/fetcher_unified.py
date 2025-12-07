"""
PRISM Unified Data Fetcher
===========================
Combined fetcher that uses the right source for each data type:
- FRED: Economic indicators (GDP, employment, inflation, rates)
- Tiingo: Market data (stocks, ETFs, sectors, crypto)

This provides a single interface while using the optimal data source
for each indicator type.

Usage:
    from fetch.fetcher_unified import PRISMFetcher

    fetcher = PRISMFetcher(
        fred_api_key='YOUR_FRED_KEY',
        tiingo_api_key='YOUR_TIINGO_KEY'
    )

    # Fetch by panel
    economy_df = fetcher.fetch_panel('economy')  # Uses FRED
    market_df = fetcher.fetch_panel('market')    # Uses Tiingo

    # Fetch everything
    all_data = fetcher.fetch_all()

Author: PRISM Engine Project
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path
import logging
import os

logger = logging.getLogger(__name__)

# =============================================================================
# FRED INDICATORS - Economic data that FRED actually provides
# =============================================================================
# These are VERIFIED working FRED symbols

FRED_INDICATORS = {
    'economy': {
        'description': 'Macroeconomic indicators from FRED',
        'icon': '🏛️',
        'indicators': {
            # --- Growth & Production ---
            'gdp_real':       ('GDPC1',       'Real GDP',                       'growth',     'Q'),
            'gdp_potential':  ('GDPPOT',      'Potential GDP',                  'growth',     'Q'),
            'indpro':         ('INDPRO',      'Industrial Production Index',    'growth',     'M'),
            'cap_util':       ('TCU',         'Capacity Utilization',           'growth',     'M'),
            'housing_starts': ('HOUST',       'Housing Starts',                 'growth',     'M'),
            'retail_sales':   ('RSAFS',       'Retail Sales',                   'growth',     'M'),

            # --- Employment & Labor ---
            'payrolls':       ('PAYEMS',      'Nonfarm Payrolls',               'labor',      'M'),
            'unemployment':   ('UNRATE',      'Unemployment Rate',              'labor',      'M'),
            'lfpr':           ('CIVPART',     'Labor Force Participation Rate', 'labor',      'M'),
            'emp_pop':        ('EMRATIO',     'Employment-Population Ratio',    'labor',      'M'),
            'init_claims':    ('ICSA',        'Initial Unemployment Claims',    'labor',      'W'),
            'job_openings':   ('JTSJOL',      'Job Openings (JOLTS)',           'labor',      'M'),

            # --- Prices & Inflation ---
            'pce':            ('PCEPI',       'PCE Price Index',                'inflation',  'M'),
            'pce_core':       ('PCEPILFE',    'Core PCE Price Index',           'inflation',  'M'),
            'cpi':            ('CPIAUCSL',    'CPI All Urban Consumers',        'inflation',  'M'),
            'cpi_core':       ('CPILFESL',    'Core CPI',                       'inflation',  'M'),
            'infl_expect':    ('T10YIE',      '10-Year Breakeven Inflation',    'inflation',  'D'),

            # --- Monetary Policy & Rates ---
            'fed_funds':      ('DFF',         'Effective Fed Funds Rate',       'monetary',   'D'),
            'treasury_10y':   ('DGS10',       '10-Year Treasury Rate',          'monetary',   'D'),
            'treasury_2y':    ('DGS2',        '2-Year Treasury Rate',           'monetary',   'D'),
            'yield_curve':    ('T10Y2Y',      '10Y-2Y Treasury Spread',         'monetary',   'D'),
            'fed_balance':    ('WALCL',       'Fed Balance Sheet',              'monetary',   'W'),
            'm2':             ('M2SL',        'M2 Money Supply',                'monetary',   'M'),

            # --- Sentiment (what's actually available) ---
            'consumer_sent':  ('UMCSENT',     'U Michigan Consumer Sentiment',  'sentiment',  'M'),
        },
    },

    'global': {
        'description': 'Global commodities, currencies, and trade data',
        'icon': '🌍',
        'indicators': {
            # --- Commodities ---
            'oil_wti':        ('DCOILWTICO',  'WTI Crude Oil Price',            'commodities', 'D'),
            'oil_brent':      ('DCOILBRENTEU','Brent Crude Oil Price',          'commodities', 'D'),

            # --- Currency ---
            'usd_eur':        ('DEXUSEU',     'USD/EUR Exchange Rate',          'currency',    'D'),
            'usd_index':      ('DTWEXBGS',    'Trade Weighted USD Index',       'currency',    'D'),

            # --- Trade ---
            'trade_balance':  ('BOPGSTB',     'Trade Balance',                  'trade',       'M'),
        },
    },

    'credit': {
        'description': 'Credit conditions and financial stress',
        'icon': '💳',
        'indicators': {
            # --- Credit Spreads (verified working) ---
            'hy_spread':      ('BAMLH0A0HYM2','High Yield OAS Spread',          'spreads',     'D'),
            'ig_spread':      ('BAMLC0A0CM',  'Investment Grade OAS Spread',    'spreads',     'D'),

            # --- Volatility & Stress ---
            'vix':            ('VIXCLS',      'VIX Volatility Index',           'volatility',  'D'),
            'stress_idx':     ('STLFSI4',     'St. Louis Financial Stress',     'volatility',  'W'),

            # --- Rates ---
            'mortgage_30y':   ('MORTGAGE30US','30-Year Mortgage Rate',          'rates',       'W'),

            # --- Debt ---
            'bus_loans':      ('BUSLOANS',    'Commercial & Industrial Loans',  'debt',        'M'),
            'consumer_credit':('TOTALSL',     'Total Consumer Credit',          'debt',        'M'),
            'fed_debt_gdp':   ('GFDEGDQ188S', 'Federal Debt as % of GDP',       'debt',        'Q'),
        },
    },
}

# =============================================================================
# TIINGO/MARKET INDICATORS - What we get from Tiingo/Stooq
# =============================================================================

MARKET_INDICATORS = {
    'market': {
        'description': 'Market data from Tiingo (ETFs as sector proxies)',
        'icon': '📈',
        'indicators': {
            # --- Major Index ETFs ---
            'spy':            ('SPY',    'S&P 500 ETF',                    'indexes',    'D'),
            'qqq':            ('QQQ',    'NASDAQ 100 ETF',                 'indexes',    'D'),
            'dia':            ('DIA',    'Dow Jones ETF',                  'indexes',    'D'),
            'iwm':            ('IWM',    'Russell 2000 ETF',               'indexes',    'D'),

            # --- Sector ETFs (instead of broken FRED sector indices) ---
            'xlk':            ('XLK',    'Technology Sector ETF',          'sectors',    'D'),
            'xlf':            ('XLF',    'Financials Sector ETF',          'sectors',    'D'),
            'xle':            ('XLE',    'Energy Sector ETF',              'sectors',    'D'),
            'xlv':            ('XLV',    'Health Care Sector ETF',         'sectors',    'D'),
            'xli':            ('XLI',    'Industrials Sector ETF',         'sectors',    'D'),
            'xlu':            ('XLU',    'Utilities Sector ETF',           'sectors',    'D'),
            'xlp':            ('XLP',    'Consumer Staples Sector ETF',    'sectors',    'D'),
            'xly':            ('XLY',    'Consumer Discretionary ETF',     'sectors',    'D'),
            'xlb':            ('XLB',    'Materials Sector ETF',           'sectors',    'D'),
            'xlre':           ('XLRE',   'Real Estate Sector ETF',         'sectors',    'D'),
            'xlc':            ('XLC',    'Communication Services ETF',     'sectors',    'D'),

            # --- Bond ETFs ---
            'tlt':            ('TLT',    '20+ Year Treasury ETF',          'bonds',      'D'),
            'ief':            ('IEF',    '7-10 Year Treasury ETF',         'bonds',      'D'),
            'shy':            ('SHY',    '1-3 Year Treasury ETF',          'bonds',      'D'),
            'hyg':            ('HYG',    'High Yield Bond ETF',            'bonds',      'D'),
            'lqd':            ('LQD',    'Investment Grade Bond ETF',      'bonds',      'D'),

            # --- Commodity ETFs ---
            'gld':            ('GLD',    'Gold ETF',                       'commodities', 'D'),
            'slv':            ('SLV',    'Silver ETF',                     'commodities', 'D'),
            'uso':            ('USO',    'Oil ETF',                        'commodities', 'D'),

            # --- Crypto ---
            'btc':            ('btcusd', 'Bitcoin',                        'crypto',     'D'),
            'eth':            ('ethusd', 'Ethereum',                       'crypto',     'D'),
        },
    },
}


# =============================================================================
# PRISM UNIFIED FETCHER
# =============================================================================

class PRISMFetcher:
    """
    Unified data fetcher for PRISM Engine.

    Routes requests to the appropriate data source:
    - FRED for economic data
    - Tiingo for market data (ETFs, crypto)
    """

    def __init__(
        self,
        fred_api_key: Optional[str] = None,
        tiingo_api_key: Optional[str] = None,
        start_date: str = '2000-01-01',
        end_date: Optional[str] = None
    ):
        """
        Initialize the unified fetcher.

        Parameters:
            fred_api_key: FRED API key
            tiingo_api_key: Tiingo API key
            start_date: Default start date
            end_date: Default end date (today if None)
        """
        self.fred_api_key = fred_api_key or os.environ.get('FRED_API_KEY', '3fd12c9d0fa4d7fd3c858b72251e3388')
        self.tiingo_api_key = tiingo_api_key or os.environ.get('TIINGO_API_KEY', '39a87d4f616a7432dbc92533eeed14c238f6d159')

        self.start_date = start_date
        self.end_date = end_date or datetime.today().strftime('%Y-%m-%d')

        # Initialize FRED
        self._fred = None

        # Initialize Tiingo
        self._tiingo = None

        # Data storage
        self.data: Dict[str, pd.DataFrame] = {}
        self.metadata: Dict[str, Dict] = {}
        self.failed: List[tuple] = []

    @property
    def fred(self):
        """Lazy load FRED client."""
        if self._fred is None:
            try:
                from fredapi import Fred
                self._fred = Fred(api_key=self.fred_api_key)
            except ImportError:
                raise ImportError("fredapi required: pip install fredapi")
        return self._fred

    @property
    def tiingo(self):
        """Lazy load Tiingo fetcher."""
        if self._tiingo is None:
            from fetch.fetcher_tiingo import TiingoFetcher
            self._tiingo = TiingoFetcher(api_key=self.tiingo_api_key)
        return self._tiingo

    def _get_panel_config(self, panel_name: str) -> tuple:
        """Get panel config and determine source."""
        if panel_name in FRED_INDICATORS:
            return FRED_INDICATORS[panel_name], 'fred'
        elif panel_name in MARKET_INDICATORS:
            return MARKET_INDICATORS[panel_name], 'tiingo'
        else:
            raise ValueError(f"Unknown panel: {panel_name}")

    def fetch_series_fred(
        self,
        fred_symbol: str,
        key: str,
        start: Optional[str] = None,
        end: Optional[str] = None
    ) -> Optional[pd.Series]:
        """Fetch a single FRED series."""
        start = start or self.start_date
        end = end or self.end_date

        try:
            series = self.fred.get_series(fred_symbol, start, end)
            logger.debug(f"✓ {key} ({fred_symbol}): {len(series)} obs")
            return series
        except Exception as e:
            logger.warning(f"✗ {key} ({fred_symbol}): {str(e)[:60]}")
            self.failed.append((key, fred_symbol, str(e)))
            return None

    def fetch_series_tiingo(
        self,
        ticker: str,
        key: str,
        is_crypto: bool = False,
        start: Optional[str] = None,
        end: Optional[str] = None
    ) -> Optional[pd.Series]:
        """Fetch a single Tiingo series."""
        start = start or self.start_date
        end = end or self.end_date

        try:
            if is_crypto:
                df = self.tiingo.fetch_crypto(ticker, start_date=start, end_date=end)
            else:
                df = self.tiingo.fetch_single(ticker, start_date=start, end_date=end)

            if df is not None and not df.empty:
                # Extract the value column
                value_col = [c for c in df.columns if c != 'date'][0]
                series = df.set_index('date')[value_col]
                series.name = key
                logger.debug(f"✓ {key} ({ticker}): {len(series)} obs")
                return series
            else:
                logger.warning(f"✗ {key} ({ticker}): No data")
                self.failed.append((key, ticker, "No data returned"))
                return None
        except Exception as e:
            logger.warning(f"✗ {key} ({ticker}): {str(e)[:60]}")
            self.failed.append((key, ticker, str(e)))
            return None

    def fetch_panel(
        self,
        panel_name: str,
        indicators: Optional[List[str]] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
        verbose: bool = True
    ) -> pd.DataFrame:
        """
        Fetch all indicators for a panel.

        Parameters:
            panel_name: 'economy', 'global', 'credit', or 'market'
            indicators: Specific indicators to fetch (None = all)
            start: Start date
            end: End date
            verbose: Print progress

        Returns:
            DataFrame with all panel indicators
        """
        panel_config, source = self._get_panel_config(panel_name)

        if verbose:
            print(f"\n{'='*60}")
            print(f"{panel_config['icon']} Fetching {panel_name.upper()} panel (source: {source})")
            print(f"{'='*60}")

        panel_indicators = panel_config['indicators']

        # Filter to requested indicators
        if indicators:
            panel_indicators = {k: v for k, v in panel_indicators.items() if k in indicators}

        results = {}
        for key, (symbol, desc, category, freq) in panel_indicators.items():
            if source == 'fred':
                series = self.fetch_series_fred(symbol, key, start, end)
            else:
                is_crypto = category == 'crypto'
                series = self.fetch_series_tiingo(symbol, key, is_crypto, start, end)

            if series is not None:
                results[key] = series
                self.metadata[key] = {
                    'symbol': symbol,
                    'description': desc,
                    'category': category,
                    'frequency': freq,
                    'panel': panel_name,
                    'source': source,
                    'observations': len(series),
                }
                if verbose:
                    print(f"  ✓ {key:20s} {len(series):5d} obs  ({category})")
            elif verbose:
                print(f"  ✗ {key:20s} FAILED")

        if results:
            df = pd.DataFrame(results)
            df.index = pd.to_datetime(df.index)
            self.data[panel_name] = df
            return df

        return pd.DataFrame()

    def fetch_all(self, verbose: bool = True) -> Dict[str, pd.DataFrame]:
        """Fetch all panels."""
        if verbose:
            print("\n" + "="*60)
            print("PRISM UNIFIED FETCHER")
            print(f"Date Range: {self.start_date} to {self.end_date}")
            print("="*60)

        results = {}

        # Fetch FRED panels
        for panel_name in FRED_INDICATORS:
            df = self.fetch_panel(panel_name, verbose=verbose)
            if not df.empty:
                results[panel_name] = df

        # Fetch market panel
        for panel_name in MARKET_INDICATORS:
            df = self.fetch_panel(panel_name, verbose=verbose)
            if not df.empty:
                results[panel_name] = df

        self._print_summary()
        return results

    def get_combined(
        self,
        panels: Optional[List[str]] = None,
        frequency: str = 'M'
    ) -> pd.DataFrame:
        """Combine panel data into a single DataFrame."""
        panels = panels or list(self.data.keys())

        all_series = {}
        for panel_name in panels:
            if panel_name in self.data:
                df = self.data[panel_name]
                for col in df.columns:
                    all_series[col] = df[col]

        if not all_series:
            return pd.DataFrame()

        combined = pd.DataFrame(all_series)

        if frequency:
            combined = combined.resample(frequency).last()

        return combined

    def get_sectors(self) -> pd.DataFrame:
        """Get all sector ETF data."""
        if 'market' not in self.data:
            return pd.DataFrame()

        sector_keys = [k for k, m in self.metadata.items()
                      if m.get('category') == 'sectors']
        df = self.data['market']
        return df[[k for k in sector_keys if k in df.columns]]

    def _print_summary(self):
        """Print fetch summary."""
        print("\n" + "="*60)
        print("FETCH SUMMARY")
        print("="*60)

        total_fred = sum(len(p['indicators']) for p in FRED_INDICATORS.values())
        total_market = sum(len(p['indicators']) for p in MARKET_INDICATORS.values())
        total_fetched = len(self.metadata)

        print(f"Total indicators: {total_fred + total_market}")
        print(f"Successfully fetched: {total_fetched}")
        print(f"Failed: {len(self.failed)}")

        if self.failed:
            print("\nFailed series:")
            for key, symbol, error in self.failed[:10]:
                print(f"  - {key} ({symbol})")

        print("\nBy panel:")
        for panel_name, df in self.data.items():
            source = 'FRED' if panel_name in FRED_INDICATORS else 'Tiingo'
            print(f"  {panel_name:12s}: {len(df.columns):3d} indicators ({source})")

    @staticmethod
    def list_indicators(panel: Optional[str] = None) -> pd.DataFrame:
        """List all available indicators."""
        rows = []

        # FRED indicators
        for panel_name, panel_config in FRED_INDICATORS.items():
            if panel and panel_name != panel:
                continue
            for key, (symbol, desc, category, freq) in panel_config['indicators'].items():
                rows.append({
                    'key': key,
                    'symbol': symbol,
                    'description': desc,
                    'category': category,
                    'frequency': freq,
                    'panel': panel_name,
                    'source': 'fred'
                })

        # Market indicators
        for panel_name, panel_config in MARKET_INDICATORS.items():
            if panel and panel_name != panel:
                continue
            for key, (symbol, desc, category, freq) in panel_config['indicators'].items():
                rows.append({
                    'key': key,
                    'symbol': symbol,
                    'description': desc,
                    'category': category,
                    'frequency': freq,
                    'panel': panel_name,
                    'source': 'tiingo'
                })

        return pd.DataFrame(rows)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def fetch_prism_data(
    fred_api_key: Optional[str] = None,
    tiingo_api_key: Optional[str] = None,
    panels: Optional[List[str]] = None,
    output_dir: Optional[str] = None,
    start_date: str = '2000-01-01'
) -> PRISMFetcher:
    """Quick-start function to fetch PRISM data."""
    fetcher = PRISMFetcher(
        fred_api_key=fred_api_key,
        tiingo_api_key=tiingo_api_key,
        start_date=start_date
    )

    if panels:
        for panel in panels:
            fetcher.fetch_panel(panel)
    else:
        fetcher.fetch_all()

    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        combined = fetcher.get_combined()
        combined.to_csv(output_path / 'prism_combined.csv')
        print(f"Saved to {output_path / 'prism_combined.csv'}")

    return fetcher


if __name__ == '__main__':
    print("""
    PRISM UNIFIED FETCHER - Fixed Version
    =====================================

    Uses the RIGHT source for each data type:
    - FRED: Economic data (GDP, employment, inflation, rates)
    - Tiingo: Market data (ETFs, sectors, crypto)

    Usage:
        fetcher = PRISMFetcher()
        economy_df = fetcher.fetch_panel('economy')  # FRED
        market_df = fetcher.fetch_panel('market')    # Tiingo
    """)

    # Show indicator inventory
    df = PRISMFetcher.list_indicators()
    print("\nIndicator inventory:")
    print(df.groupby(['panel', 'source']).size())
