"""
PRISM Unified FRED Fetcher
===========================
Single-source data fetcher with extensible panel architecture.

Replaces the multi-source (Stooq/Yahoo/FRED) approach with FRED-only,
eliminating fallback complexity while maintaining comprehensive coverage.

ARCHITECTURE:
- Panels are modular and easily extensible
- Currently active: 'economy' (comprehensive macro/market data)
- Future panels can be added: 'climate', 'housing', 'traffic', 'chemistry', etc.
- Each panel has its own indicators organized by category

Current 'economy' panel covers:
- Market indexes, sectors, volatility, credit spreads
- GDP, employment, inflation, Fed policy, sentiment
- Global commodities, forex, trade balance
- Debt levels and credit conditions

Usage:
    from fetch.fetcher_unified import PRISMFetcher

    fetcher = PRISMFetcher(api_key='YOUR_KEY')

    # Fetch economy panel
    economy_df = fetcher.fetch_panel('economy')

    # Fetch by category within a panel
    sectors_df = fetcher.get_category('economy', 'sectors')
    labor_df = fetcher.get_category('economy', 'labor')

    # Get combined DataFrame
    df = fetcher.get_combined(frequency='M')

    # List available panels
    panels = PRISMFetcher.list_panels()

Adding New Panels:
    Simply add a new entry to PRISM_REGISTRY with:
    - 'description': Panel description
    - 'icon': Display icon
    - 'indicators': Dict of key -> (fred_symbol, description, category, frequency)

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
# PRISM INDICATOR REGISTRY - UNIFIED ECONOMY PANEL
# =============================================================================
# All indicators consolidated into single "economy" panel
# Each indicator maps: key -> (fred_symbol, description, category, frequency)

PRISM_REGISTRY = {
    # =========================================================================
    # ECONOMY PANEL - All PRISM Indicators Unified
    # =========================================================================
    'economy': {
        'description': 'Unified economic and market indicators from FRED',
        'icon': '🏛️',
        'indicators': {
            # --- Major Market Indexes ---
            'sp500':        ('SP500',           'S&P 500 Index',                    'indexes',     'D'),
            'nasdaq':       ('NASDAQCOM',       'NASDAQ Composite',                 'indexes',     'D'),
            'djia':         ('DJIA',            'Dow Jones Industrial Average',     'indexes',     'D'),

            # --- GICS Sectors (S&P 500 Sector Indexes) ---
            'sect_energy':      ('SPXENE',  'S&P 500 Energy Sector',                'sectors',     'D'),
            'sect_materials':   ('SPXMAT',  'S&P 500 Materials Sector',             'sectors',     'D'),
            'sect_industrials': ('SPXIND',  'S&P 500 Industrials Sector',           'sectors',     'D'),
            'sect_cons_disc':   ('SPXDCE',  'S&P 500 Consumer Discretionary',       'sectors',     'D'),
            'sect_cons_stpl':   ('SPXCS',   'S&P 500 Consumer Staples',             'sectors',     'D'),
            'sect_healthcare':  ('SPXHC',   'S&P 500 Health Care Sector',           'sectors',     'D'),
            'sect_financials':  ('SPXFL',   'S&P 500 Financials Sector',            'sectors',     'D'),
            'sect_tech':        ('SPXIT',   'S&P 500 Information Technology',       'sectors',     'D'),
            'sect_comm':        ('SPXCSE',  'S&P 500 Communication Services',       'sectors',     'D'),
            'sect_utilities':   ('SPXUT',   'S&P 500 Utilities Sector',             'sectors',     'D'),
            'sect_realestate':  ('SPXRE',   'S&P 500 Real Estate Sector',           'sectors',     'D'),

            # --- Volatility & Stress ---
            'vix':          ('VIXCLS',          'VIX Volatility Index',             'volatility',  'D'),
            'stress_idx':   ('STLFSI',          'St. Louis Financial Stress Index', 'volatility',  'W'),

            # --- Credit Conditions ---
            'hy_spread':    ('BAMLH0A0HYM2',    'High Yield OAS Spread',            'credit',      'D'),
            'ted_spread':   ('TEDRATE',         'TED Spread',                       'credit',      'D'),

            # --- Housing & Real Estate ---
            'mortgage_30y': ('MORTGAGE30US',    '30-Year Mortgage Rate',            'housing',     'W'),
            'home_price':   ('CSUSHPINSA',      'Case-Shiller Home Price Index',    'housing',     'M'),

            # --- Household Financial Health ---
            'fin_oblig':    ('DRSFRST',         'Financial Obligations Ratio',      'household',   'Q'),

            # --- Growth & Production ---
            'gdp_real':     ('GDPC1',           'Real GDP',                         'growth',      'Q'),
            'gdp_potential':('GDPPOT',          'Potential GDP',                    'growth',      'Q'),
            'indpro':       ('INDPRO',          'Industrial Production Index',      'growth',      'M'),
            'cap_util':     ('TCU',             'Capacity Utilization',             'growth',      'M'),
            'housing_starts':('HOUST',          'Housing Starts',                   'growth',      'M'),
            'retail_sales': ('RRSFS',           'Real Retail Sales',                'growth',      'M'),
            'cap_orders':   ('M0349AUSM189SNBR','Nondefense Capital Goods Orders',  'growth',      'M'),

            # --- Employment & Labor ---
            'payrolls':     ('PAYEMS',          'Nonfarm Payrolls',                 'labor',       'M'),
            'unemployment': ('UNRATE',          'Unemployment Rate',                'labor',       'M'),
            'lfpr':         ('CIVPART',         'Labor Force Participation Rate',   'labor',       'M'),
            'emp_pop':      ('EMRATIO',         'Employment-Population Ratio',      'labor',       'M'),
            'init_claims':  ('ICSA',            'Initial Unemployment Claims',      'labor',       'W'),
            'job_openings': ('JTSJOL',          'Job Openings (JOLTS)',             'labor',       'M'),

            # --- Prices & Inflation ---
            'pce':          ('PCEPI',           'PCE Price Index',                  'inflation',   'M'),
            'pce_core':     ('PCEPILFE',        'Core PCE Price Index',             'inflation',   'M'),
            'cpi':          ('CPIAUCSL',        'CPI All Urban Consumers',          'inflation',   'M'),
            'cpi_core':     ('CPILFESL',        'Core CPI',                         'inflation',   'M'),
            'infl_expect':  ('T10YIE',          '10-Year Breakeven Inflation',      'inflation',   'D'),

            # --- Monetary Policy & Rates ---
            'fed_funds':    ('DFF',             'Effective Fed Funds Rate',         'monetary',    'D'),
            'treasury_10y': ('DGS10',           '10-Year Treasury Rate',            'monetary',    'D'),
            'yield_curve':  ('T10Y2YM',         '10Y-2Y Treasury Spread',           'monetary',    'D'),
            'fed_balance':  ('WALCL',           'Fed Balance Sheet',                'monetary',    'W'),
            'm2':           ('M2SL',            'M2 Money Supply',                  'monetary',    'M'),
            'bank_reserves':('TOTRESNS',        'Total Bank Reserves',              'monetary',    'M'),

            # --- Sentiment ---
            'consumer_sent':('UMCSENT',         'U Michigan Consumer Sentiment',    'sentiment',   'M'),
            'ism_mfg_new':  ('NAPMNEWO',        'ISM Manufacturing New Orders',     'sentiment',   'M'),
            'ism_svc':      ('ISMS',            'ISM Services PMI',                 'sentiment',   'M'),

            # --- Commodities ---
            'oil_wti':      ('DCOILWTICO',      'WTI Crude Oil Price',              'commodities', 'D'),
            'nat_gas':      ('WPU0911',         'Natural Gas PPI',                  'commodities', 'M'),
            'commodity_idx':('PALLFNFINDEXM',   'Global Commodity Price Index',     'commodities', 'M'),

            # --- Currency ---
            'usd_eur':      ('DEXUSEU',         'USD/EUR Exchange Rate',            'currency',    'D'),
            'usd_index':    ('DTWEXBGS',        'Trade Weighted USD Index',         'currency',    'D'),

            # --- Trade ---
            'trade_balance':('BOPGSTB',         'Trade Balance',                    'trade',       'M'),
            'imports':      ('IMPGSE',          'US Imports',                       'trade',       'Q'),
            'exports':      ('EXPGSE',          'US Exports',                       'trade',       'Q'),

            # --- Debt & Credit ---
            'bus_loans':    ('BUSLOANS',        'Commercial & Industrial Loans',    'debt',        'M'),
            'consumer_cred':('CCLACBW027SBOG',  'Consumer Credit Outstanding',      'debt',        'M'),
            'fed_debt':     ('GFDEBTN',         'Federal Debt: Total Public',       'debt',        'Q'),
            'total_debt':   ('TCMDO',           'Total Credit Market Debt',         'debt',        'Q'),
            'debt_gdp':     ('GFDEGDQ188S',     'Federal Debt as % of GDP',         'debt',        'Q'),
        },
    },
}


# =============================================================================
# PRISM FETCHER CLASS
# =============================================================================

class PRISMFetcher:
    """
    Unified FRED data fetcher for PRISM Engine.

    Single source for all economic and market data consolidated
    into one panel. Eliminates multi-source fallback complexity.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        start_date: str = '1970-01-01',
        end_date: Optional[str] = None
    ):
        """
        Initialize the PRISM fetcher.

        Parameters:
            api_key: FRED API key (or set FRED_API_KEY env variable)
            start_date: Default start date for data fetches
            end_date: Default end date (defaults to today)
        """
        self.api_key = api_key or os.environ.get('FRED_API_KEY')
        if not self.api_key:
            raise ValueError(
                "FRED API key required. Set FRED_API_KEY env variable or pass api_key parameter.\n"
                "Get a free key at: https://fred.stlouisfed.org/docs/api/api_key.html"
            )

        self.start_date = start_date
        self.end_date = end_date or datetime.today().strftime('%Y-%m-%d')

        # Initialize FRED connection
        try:
            from fredapi import Fred
            self.fred = Fred(api_key=self.api_key)
        except ImportError:
            raise ImportError("fredapi required: pip install fredapi")

        # Storage for fetched data
        self.data: Dict[str, pd.DataFrame] = {}
        self.metadata: Dict[str, Dict] = {}
        self.failed: List[tuple] = []

    # -------------------------------------------------------------------------
    # Core Fetch Methods
    # -------------------------------------------------------------------------

    def fetch_series(
        self,
        fred_symbol: str,
        key: str,
        start: Optional[str] = None,
        end: Optional[str] = None
    ) -> Optional[pd.Series]:
        """
        Fetch a single FRED series.

        Parameters:
            fred_symbol: FRED series ID
            key: PRISM indicator key
            start: Start date (uses default if None)
            end: End date (uses default if None)

        Returns:
            pandas Series or None if failed
        """
        start = start or self.start_date
        end = end or self.end_date

        try:
            series = self.fred.get_series(fred_symbol, start, end)
            logger.debug(f"Fetched {key} ({fred_symbol}): {len(series)} obs")
            return series
        except Exception as e:
            logger.warning(f"Failed {key} ({fred_symbol}): {str(e)[:60]}")
            self.failed.append((key, fred_symbol, str(e)))
            return None

    def fetch_panel(
        self,
        panel_name: str = 'economy',
        indicators: Optional[List[str]] = None,
        categories: Optional[List[str]] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
        verbose: bool = True
    ) -> pd.DataFrame:
        """
        Fetch indicators from the unified economy panel.

        Parameters:
            panel_name: Panel name (currently only 'economy' supported)
            indicators: List of indicator keys (None = all)
            categories: List of categories to fetch (e.g., ['labor', 'inflation'])
            start: Start date
            end: End date
            verbose: Print progress

        Returns:
            DataFrame with all panel indicators
        """
        if panel_name not in PRISM_REGISTRY:
            raise ValueError(f"Unknown panel: {panel_name}. Valid: {list(PRISM_REGISTRY.keys())}")

        panel = PRISM_REGISTRY[panel_name]
        panel_indicators = panel['indicators']

        if verbose:
            print(f"\n{'='*60}")
            print(f"{panel['icon']} Fetching {panel_name.upper()} panel")
            print(f"{'='*60}")

        # Filter by categories if specified
        if categories:
            panel_indicators = {
                k: v for k, v in panel_indicators.items()
                if v[2] in categories
            }

        # Filter to requested indicators
        if indicators:
            panel_indicators = {k: v for k, v in panel_indicators.items() if k in indicators}

        # Fetch each series
        results = {}
        for key, (fred_symbol, desc, category, freq) in panel_indicators.items():
            series = self.fetch_series(fred_symbol, key, start, end)
            if series is not None:
                results[key] = series
                self.metadata[key] = {
                    'fred_symbol': fred_symbol,
                    'description': desc,
                    'category': category,
                    'frequency': freq,
                    'panel': panel_name,
                    'observations': len(series),
                }
                if verbose:
                    print(f"  + {key:20s} {len(series):5d} obs  ({category})")
            elif verbose:
                print(f"  x {key:20s} FAILED")

        if results:
            df = pd.DataFrame(results)
            df.index = pd.to_datetime(df.index)
            self.data[panel_name] = df
            return df

        return pd.DataFrame()

    def fetch_all(self, verbose: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Fetch all indicators from all available panels.

        Returns:
            Dict mapping panel name to DataFrame
        """
        if verbose:
            print("\n" + "="*60)
            print("PRISM UNIFIED FRED FETCHER")
            print(f"Date Range: {self.start_date} to {self.end_date}")
            print(f"Available panels: {list(PRISM_REGISTRY.keys())}")
            print("="*60)

        results = {}
        for panel_name in PRISM_REGISTRY:
            df = self.fetch_panel(panel_name, verbose=verbose)
            if not df.empty:
                results[panel_name] = df

        self._print_summary()
        return results

    # -------------------------------------------------------------------------
    # Data Access Methods
    # -------------------------------------------------------------------------

    def get_combined(
        self,
        panels: Optional[List[str]] = None,
        frequency: str = 'M'
    ) -> pd.DataFrame:
        """
        Combine panel data into a single DataFrame.

        Parameters:
            panels: List of panels to include (None = all fetched)
            frequency: Resample frequency ('D', 'W', 'M', 'Q')

        Returns:
            Combined DataFrame
        """
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

        # Resample to common frequency
        if frequency:
            combined = combined.resample(frequency).last()

        return combined

    def get_category(
        self,
        panel: str,
        category: str
    ) -> pd.DataFrame:
        """
        Get indicators for a specific category within a panel.

        Parameters:
            panel: Panel name (e.g., 'economy')
            category: Category name (e.g., 'sectors', 'labor', 'commodities')

        Returns:
            DataFrame with category indicators
        """
        if panel not in self.data:
            return pd.DataFrame()

        # Find indicators in this category
        category_keys = [
            key for key, meta in self.metadata.items()
            if meta.get('panel') == panel and meta.get('category') == category
        ]

        df = self.data[panel]
        return df[[k for k in category_keys if k in df.columns]]

    def get_sectors(self) -> pd.DataFrame:
        """Get all GICS sector data from economy panel."""
        return self.get_category('economy', 'sectors')

    def get_indexes(self) -> pd.DataFrame:
        """Get major market indexes from economy panel."""
        return self.get_category('economy', 'indexes')

    def get_labor(self) -> pd.DataFrame:
        """Get employment and labor market data from economy panel."""
        return self.get_category('economy', 'labor')

    def get_inflation(self) -> pd.DataFrame:
        """Get inflation and price data from economy panel."""
        return self.get_category('economy', 'inflation')

    def get_monetary(self) -> pd.DataFrame:
        """Get monetary policy and rates data from economy panel."""
        return self.get_category('economy', 'monetary')

    # -------------------------------------------------------------------------
    # Export Methods
    # -------------------------------------------------------------------------

    def save_to_csv(
        self,
        output_dir: str,
        combined: bool = True,
        by_panel: bool = True,
        prefix: str = 'prism'
    ) -> Path:
        """
        Save data to CSV files.

        Parameters:
            output_dir: Output directory path
            combined: Save combined DataFrame (monthly resampled)
            by_panel: Save separate file per panel
            prefix: Filename prefix

        Returns:
            Path to output directory
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        if combined and self.data:
            combined_df = self.get_combined(frequency='M')
            filepath = output_path / f'{prefix}_combined.csv'
            combined_df.to_csv(filepath)
            print(f"\n+ Saved: {filepath} ({combined_df.shape})")

        if by_panel:
            for panel_name, df in self.data.items():
                filepath = output_path / f'{prefix}_{panel_name}.csv'
                df.to_csv(filepath)
                print(f"+ Saved: {filepath} ({df.shape})")

        # Save metadata
        if self.metadata:
            meta_df = pd.DataFrame(self.metadata).T
            meta_path = output_path / f'{prefix}_metadata.csv'
            meta_df.to_csv(meta_path)
            print(f"+ Saved: {meta_path}")

        return output_path

    def to_registry_yaml(self, output_path: str) -> str:
        """
        Export metadata as PRISM registry YAML format.

        Parameters:
            output_path: Path for YAML file

        Returns:
            Path to created file
        """
        import yaml

        registry = {}
        for panel_name, panel_config in PRISM_REGISTRY.items():
            panel_data = {
                'name': panel_name.title(),
                'description': panel_config['description'],
                'icon': panel_config['icon'],
                'data_source': {'primary': 'fred', 'api_key_env': 'FRED_API_KEY'},
                'indicators': []
            }

            for key, (symbol, desc, category, freq) in panel_config['indicators'].items():
                panel_data['indicators'].append({
                    'key': key,
                    'symbol': symbol,
                    'name': desc,
                    'category': category,
                    'frequency': freq,
                    'source': 'fred'
                })

            registry[panel_name] = panel_data

        with open(output_path, 'w') as f:
            yaml.dump(registry, f, default_flow_style=False, sort_keys=False)

        print(f"+ Registry exported: {output_path}")
        return output_path

    # -------------------------------------------------------------------------
    # Utility Methods
    # -------------------------------------------------------------------------

    def _print_summary(self):
        """Print fetch summary."""
        print("\n" + "="*60)
        print("FETCH SUMMARY")
        print("="*60)

        total_requested = sum(len(p['indicators']) for p in PRISM_REGISTRY.values())
        total_fetched = len(self.metadata)

        print(f"Total indicators: {total_requested}")
        print(f"Successfully fetched: {total_fetched}")
        print(f"Failed: {len(self.failed)}")

        if self.failed:
            print("\nFailed series:")
            for key, symbol, error in self.failed[:10]:
                print(f"  - {key} ({symbol})")
            if len(self.failed) > 10:
                print(f"  ... and {len(self.failed) - 10} more")

        print("\nBy panel:")
        for panel_name, df in self.data.items():
            icon = PRISM_REGISTRY[panel_name]['icon']
            print(f"  {icon} {panel_name:12s}: {len(df.columns):3d} indicators, {len(df):5d} rows")

        # Count by category
        categories = {}
        for key, meta in self.metadata.items():
            cat = meta.get('category', 'other')
            categories[cat] = categories.get(cat, 0) + 1

        if categories:
            print("\nBy category:")
            for cat, count in sorted(categories.items()):
                print(f"  {cat:15s}: {count:3d} indicators")

    @staticmethod
    def list_panels() -> List[str]:
        """List all available panels."""
        return list(PRISM_REGISTRY.keys())

    @staticmethod
    def list_indicators(
        panel: Optional[str] = None,
        category: Optional[str] = None
    ) -> pd.DataFrame:
        """
        List all available indicators.

        Parameters:
            panel: Filter to specific panel (None = all panels)
            category: Filter to specific category (None = all)

        Returns:
            DataFrame with indicator information
        """
        rows = []
        panels_to_check = [panel] if panel else PRISM_REGISTRY.keys()

        for panel_name in panels_to_check:
            if panel_name not in PRISM_REGISTRY:
                continue
            panel_config = PRISM_REGISTRY[panel_name]
            for key, (symbol, desc, cat, freq) in panel_config['indicators'].items():
                if category and cat != category:
                    continue
                rows.append({
                    'key': key,
                    'fred_symbol': symbol,
                    'description': desc,
                    'category': cat,
                    'frequency': freq,
                    'panel': panel_name,
                })
        return pd.DataFrame(rows)

    @staticmethod
    def list_categories(panel: Optional[str] = None) -> List[str]:
        """
        List all available categories.

        Parameters:
            panel: Filter to specific panel (None = all panels)
        """
        categories = set()
        panels_to_check = [panel] if panel else PRISM_REGISTRY.keys()

        for panel_name in panels_to_check:
            if panel_name not in PRISM_REGISTRY:
                continue
            for key, (symbol, desc, cat, freq) in PRISM_REGISTRY[panel_name]['indicators'].items():
                categories.add(cat)
        return sorted(list(categories))


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def fetch_prism_data(
    api_key: Optional[str] = None,
    categories: Optional[List[str]] = None,
    output_dir: Optional[str] = None,
    start_date: str = '1970-01-01'
) -> PRISMFetcher:
    """
    Quick-start function to fetch PRISM data.

    Parameters:
        api_key: FRED API key
        categories: List of categories to fetch (None = all)
        output_dir: Save CSVs to this directory (optional)
        start_date: Data start date

    Returns:
        PRISMFetcher instance with data
    """
    fetcher = PRISMFetcher(api_key=api_key, start_date=start_date)

    if categories:
        fetcher.fetch_panel('economy', categories=categories)
    else:
        fetcher.fetch_all()

    if output_dir:
        fetcher.save_to_csv(output_dir)

    return fetcher


def fetch_economy_data(api_key: Optional[str] = None) -> pd.DataFrame:
    """Fetch all economy panel data."""
    fetcher = PRISMFetcher(api_key=api_key)
    return fetcher.fetch_panel('economy')


def fetch_sectors_only(api_key: Optional[str] = None) -> pd.DataFrame:
    """Fetch only GICS sector data."""
    fetcher = PRISMFetcher(api_key=api_key)
    fetcher.fetch_panel('economy', categories=['sectors'])
    return fetcher.get_sectors()


def fetch_macro_only(api_key: Optional[str] = None) -> pd.DataFrame:
    """Fetch core macro indicators (growth, labor, inflation, monetary)."""
    fetcher = PRISMFetcher(api_key=api_key)
    fetcher.fetch_panel('economy', categories=['growth', 'labor', 'inflation', 'monetary'])
    return fetcher.get_combined()


# =============================================================================
# MAIN - INVENTORY DISPLAY
# =============================================================================

if __name__ == '__main__':
    print("""
    ================================================================
    PRISM UNIFIED FRED FETCHER
    ================================================================
    Single-source data for PRISM Engine
    Extensible panel architecture - easily add new panels
    Currently active: 'economy' panel
    Replaces Stooq/Yahoo/FRED multi-source complexity
    ================================================================

    Usage:
    ------
    from fetch.fetcher_unified import PRISMFetcher, fetch_prism_data

    # Option 1: Fetch everything
    fetcher = fetch_prism_data(
        api_key=FRED_API_KEY,
        output_dir='./data'
    )

    # Option 2: Fetch by categories
    fetcher = PRISMFetcher(api_key=FRED_API_KEY)
    fetcher.fetch_panel('economy', categories=['labor', 'inflation'])

    # Option 3: Get combined monthly data
    df = fetcher.get_combined(frequency='M')

    # Option 4: Get specific data
    sectors = fetcher.get_sectors()
    labor = fetcher.get_labor()

    Adding New Panels:
    ------------------
    Add a new entry to PRISM_REGISTRY:

    'climate': {
        'description': 'Climate and environmental indicators',
        'icon': '🌡️',
        'indicators': {
            'co2_level': ('XXXXX', 'CO2 Levels', 'atmosphere', 'M'),
            ...
        }
    }
    """)

    # Print inventory for all panels
    print("\n" + "="*70)
    print("INDICATOR INVENTORY")
    print("="*70)

    total = 0
    for panel_name, panel_config in PRISM_REGISTRY.items():
        indicators = panel_config['indicators']
        print(f"\n{panel_config['icon']} {panel_name.upper()} PANEL ({len(indicators)} indicators)")
        print("-" * 60)

        # Group by category
        by_category = {}
        for key, (symbol, desc, category, freq) in indicators.items():
            if category not in by_category:
                by_category[category] = []
            by_category[category].append((key, symbol, desc, freq))

        for category, items in sorted(by_category.items()):
            print(f"\n  [{category}]")
            for key, symbol, desc, freq in items:
                print(f"    {key:20s} {symbol:20s} {freq}")

        total += len(indicators)

    print(f"\n{'='*70}")
    print(f"TOTAL INDICATORS: {total}")
    print(f"AVAILABLE PANELS: {', '.join(PRISM_REGISTRY.keys())}")
    print(f"{'='*70}")
