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

        # Normalize ticker for Tiingo (remove .US suffix if present)
        clean_ticker = ticker.replace('.US', '')

        try:
            if is_crypto:
                df = self.tiingo.fetch_crypto(clean_ticker, start_date=start, end_date=end)
            else:
                df = self.tiingo.fetch_single(clean_ticker, start_date=start, end_date=end)

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

    def save_to_db(self, verbose: bool = True) -> Dict[str, int]:
        """
        Save all fetched data to indicator_values table (Schema v2).

        This writes to the universal indicator_values table, NOT the
        deprecated market_prices or econ_values tables.

        Returns:
            Dict mapping indicator names to rows written
        """
        from data.sql.db_connector import add_indicator, write_dataframe

        results = {}

        for panel_name, df in self.data.items():
            if df.empty:
                continue

            source = 'fred' if panel_name in FRED_INDICATORS else 'tiingo'

            for indicator_name in df.columns:
                # Get metadata for this indicator
                meta = self.metadata.get(indicator_name, {})

                # Register indicator in indicators table
                add_indicator(
                    name=indicator_name,
                    system=meta.get('category', 'market'),
                    source=source,
                    frequency=meta.get('frequency', 'daily'),
                    description=meta.get('description'),
                    metadata={
                        'symbol': meta.get('symbol'),
                        'panel': panel_name,
                    }
                )

                # Prepare data for indicator_values table
                series = df[indicator_name].dropna()
                if series.empty:
                    continue

                indicator_df = pd.DataFrame({
                    'indicator_name': indicator_name,
                    'date': series.index,
                    'value': series.values,
                })

                # Write to indicator_values
                rows = write_dataframe(
                    indicator_df,
                    table='indicator_values',
                    provenance=source,
                    quality_flag='verified'
                )

                results[indicator_name] = rows

                if verbose:
                    print(f"  + {indicator_name}: {rows} rows saved")

        return results

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
    import argparse
    import sys

    # Add project root to path
    PROJECT_ROOT = Path(__file__).parent.parent
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    parser = argparse.ArgumentParser(
        description='PRISM Unified Fetcher - Fetch economic (FRED) and market (Tiingo) data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python fetcher_unified.py --fetch all
    python fetcher_unified.py --fetch economy
    python fetcher_unified.py --fetch market
    python fetcher_unified.py --fetch economy market
    python fetcher_unified.py --fetch all --start-date 2010-01-01
    python fetcher_unified.py --list
    python fetcher_unified.py --list --panel economy

Panels:
    economy  - FRED macroeconomic data (GDP, employment, inflation)
    global   - FRED global data (commodities, currencies, trade)
    credit   - FRED credit/financial stress (spreads, VIX, debt)
    market   - Tiingo market data (ETFs, sectors, bonds, crypto)
    all      - All panels
        """
    )

    parser.add_argument(
        '--fetch', '-f',
        nargs='*',
        default=['all'],
        metavar='PANEL',
        help='Panels to fetch: economy, global, credit, market, or all (default: all)'
    )
    parser.add_argument(
        '--start-date', '-s',
        type=str,
        default='2000-01-01',
        help='Start date (YYYY-MM-DD, default: 2000-01-01)'
    )
    parser.add_argument(
        '--end-date', '-e',
        type=str,
        default=None,
        help='End date (YYYY-MM-DD, default: today)'
    )
    parser.add_argument(
        '--list', '-l',
        action='store_true',
        help='List available indicators'
    )
    parser.add_argument(
        '--panel', '-p',
        type=str,
        help='Filter --list to specific panel'
    )
    parser.add_argument(
        '--dry-run', '-n',
        action='store_true',
        help='Show what would be fetched without actually fetching'
    )
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Fetch data but do not save to database'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output'
    )

    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )

    # List mode
    if args.list:
        df = PRISMFetcher.list_indicators(panel=args.panel)
        print("\n" + "="*70)
        print("PRISM INDICATOR INVENTORY")
        print("="*70)

        if args.panel:
            print(f"\nPanel: {args.panel}")
        else:
            print("\nAll panels:")

        # Summary by panel
        print("\nBy Panel:")
        for (panel, source), group in df.groupby(['panel', 'source']):
            print(f"  {panel:12s} ({source:6s}): {len(group):3d} indicators")

        print("\nIndicators:")
        print("-"*70)
        for _, row in df.iterrows():
            print(f"  {row['key']:20s} {row['symbol']:12s} {row['description'][:35]}")

        print(f"\nTotal: {len(df)} indicators")
        sys.exit(0)

    # Fetch mode
    if args.fetch or args.fetch == []:
        panels_to_fetch = args.fetch if args.fetch else ['all']

        # Expand 'all'
        if 'all' in panels_to_fetch:
            panels_to_fetch = list(FRED_INDICATORS.keys()) + list(MARKET_INDICATORS.keys())

        # Validate panels
        valid_panels = set(FRED_INDICATORS.keys()) | set(MARKET_INDICATORS.keys())
        for p in panels_to_fetch:
            if p not in valid_panels and p != 'all':
                print(f"ERROR: Unknown panel '{p}'")
                print(f"Valid panels: {', '.join(valid_panels)}")
                sys.exit(1)

        # Dry run
        if args.dry_run:
            print("\n" + "="*60)
            print("DRY RUN - Would fetch:")
            print("="*60)
            for panel in panels_to_fetch:
                if panel in FRED_INDICATORS:
                    config = FRED_INDICATORS[panel]
                    source = 'FRED'
                else:
                    config = MARKET_INDICATORS[panel]
                    source = 'Tiingo'

                print(f"\n{config['icon']} {panel.upper()} ({source}):")
                for key, (symbol, desc, cat, freq) in config['indicators'].items():
                    print(f"    {key:20s} -> {symbol}")

            print(f"\nDate range: {args.start_date} to {args.end_date or 'today'}")
            print("\nRun without --dry-run to actually fetch.")
            sys.exit(0)

        # Actually fetch
        print("\n" + "="*60)
        print("PRISM UNIFIED FETCHER")
        print("="*60)
        print(f"Date range: {args.start_date} to {args.end_date or 'today'}")
        print(f"Panels: {', '.join(panels_to_fetch)}")
        print(f"Save to DB: {not args.no_save}")
        print("="*60)

        fetcher = PRISMFetcher(
            start_date=args.start_date,
            end_date=args.end_date
        )

        # Fetch each panel
        for panel in panels_to_fetch:
            try:
                fetcher.fetch_panel(panel, verbose=True)
            except Exception as e:
                logger.error(f"Failed to fetch {panel}: {e}")

        # Save to database
        if not args.no_save and fetcher.data:
            print("\n" + "="*60)
            print("SAVING TO DATABASE")
            print("="*60)

            from data.sql.db_path import get_db_path
            from data.sql.db_connector import init_database

            db_path = get_db_path()
            print(f"Database: {db_path}")

            # Ensure DB is initialized
            init_database()

            # Save data
            save_results = fetcher.save_to_db(verbose=args.verbose)

            total_rows = sum(save_results.values())
            print(f"\nTotal: {len(save_results)} indicators, {total_rows:,} rows saved")

            # Log to fetch_log
            try:
                from data.sql.db_connector import log_fetch
                for indicator_name, rows in save_results.items():
                    meta = fetcher.metadata.get(indicator_name, {})
                    log_fetch(
                        indicator_name=indicator_name,
                        source=meta.get('source', 'unknown'),
                        rows_fetched=rows,
                        status='success'
                    )
            except Exception as e:
                logger.warning(f"Could not log fetch: {e}")

        # Print summary
        fetcher._print_summary()

        print("\n✅ Fetch complete!")
        sys.exit(0)

    # No action specified
    parser.print_help()
    print("\n" + "="*60)
    print("Quick Start:")
    print("  python fetcher_unified.py --fetch all")
    print("  python fetcher_unified.py --list")
    print("="*60)
