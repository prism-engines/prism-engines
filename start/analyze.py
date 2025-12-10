#!/usr/bin/env python3
"""
PRISM Unified Analysis Runner
=============================
Runs all configured lenses on economic/market data.

Usage:
    python analyze.py                    # Run with config defaults
    python analyze.py --start 2020-01-01 # Override start date
    python analyze.py --lenses granger regime pca  # Run specific lenses
    python analyze.py --list             # List available lenses

Reads settings from: prism_config.yaml
"""

import sys
import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np

# Setup paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name == 'start' else SCRIPT_DIR

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------

def load_config():
    """Load configuration from prism_config.yaml."""
    import yaml
    
    config_paths = [
        SCRIPT_DIR / 'prism_config.yaml',
        PROJECT_ROOT / 'prism_config.yaml',
        PROJECT_ROOT / 'start' / 'prism_config.yaml',
    ]
    
    for path in config_paths:
        if path.exists():
            with open(path) as f:
                return yaml.safe_load(f)
    
    return get_default_config()


def get_default_config():
    """Return default configuration."""
    return {
        'data': {
            'start_date': '2000-01-01',
            'end_date': None,
            'frequency': 'D',
            'min_coverage': 0.5,  # 50% coverage - balances data quality vs indicator count
        },
        'analysis': {
            'output_dir': 'output/analysis',
            'save_results': True,
            'lenses': [
                'granger', 'regime', 'clustering', 'wavelet', 'anomaly',
                'pca', 'network', 'transfer_entropy', 'mutual_info', 'magnitude',
                'decomposition', 'dmd', 'influence', 'tda'
            ],
        },
    }


# -----------------------------------------------------------------------------
# Normalization Functions
# -----------------------------------------------------------------------------

def normalize_data(df: pd.DataFrame, method: str) -> pd.DataFrame:
    """
    Apply normalization to numeric columns.
    
    Args:
        df: DataFrame with 'date' column and numeric indicators
        method: Normalization method
            - None/'none': No transformation (raw values)
            - 'zscore': Standardize to mean=0, std=1
            - 'diff': First difference (returns/changes)
            - 'pct': Percent change
            - 'rank': Rank transform (robust to outliers)
            - 'robust': Robust scaling (median/IQR)
            - 'log_diff': Log returns (for prices)
    
    Returns:
        Normalized DataFrame with same structure
    """
    if method is None or method == 'none':
        return df
    
    # Preserve date column
    date_col = None
    if 'date' in df.columns:
        date_col = df['date'].copy()
        numeric_df = df.drop(columns=['date'])
    else:
        numeric_df = df.copy()
    
    if method == 'zscore':
        # Standardize: (x - mean) / std
        result = (numeric_df - numeric_df.mean()) / numeric_df.std()
        
    elif method == 'diff':
        # First difference - makes non-stationary data stationary
        result = numeric_df.diff()
        
    elif method == 'pct':
        # Percent change
        result = numeric_df.pct_change()
        
    elif method == 'rank':
        # Rank transform - robust to outliers
        result = numeric_df.rank(pct=True)
        
    elif method == 'robust':
        # Robust scaling using median and IQR
        median = numeric_df.median()
        q75 = numeric_df.quantile(0.75)
        q25 = numeric_df.quantile(0.25)
        iqr = q75 - q25
        iqr = iqr.replace(0, 1)  # Avoid division by zero
        result = (numeric_df - median) / iqr
        
    elif method == 'log_diff':
        # Log returns - standard for price data
        result = np.log(numeric_df / numeric_df.shift(1))
        
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    # Drop NaN rows created by diff/pct operations
    if method in ('diff', 'pct', 'log_diff'):
        result = result.iloc[1:]
        if date_col is not None:
            date_col = date_col.iloc[1:]
    
    # Re-attach date column
    if date_col is not None:
        result.insert(0, 'date', date_col.values)
    
    return result.reset_index(drop=True)


# -----------------------------------------------------------------------------
# Lens Registry
# -----------------------------------------------------------------------------

LENS_REGISTRY = {
    'granger': {
        'module': 'engine_core.lenses.granger_lens',
        'class': 'GrangerLens',
        'description': 'Granger causality testing',
        'normalize': 'diff',  # Requires stationary data
    },
    'regime': {
        'module': 'engine_core.lenses.regime_switching_lens',
        'class': 'RegimeSwitchingLens',
        'description': 'Regime detection and switching',
        'normalize': 'zscore',  # Standardized for regime detection
    },
    'clustering': {
        'module': 'engine_core.lenses.clustering_lens',
        'class': 'ClusteringLens',
        'description': 'Hierarchical clustering',
        'normalize': 'zscore',  # Standardized for distance calculations
    },
    'wavelet': {
        'module': 'engine_core.lenses.wavelet_lens',
        'class': 'WaveletLens',
        'description': 'Multi-scale wavelet analysis',
        'normalize': None,  # Works on raw levels to detect cycles
    },
    'anomaly': {
        'module': 'engine_core.lenses.anomaly_lens',
        'class': 'AnomalyLens',
        'description': 'Anomaly detection',
        'normalize': 'robust',  # Robust scaling for outlier detection
    },
    'pca': {
        'module': 'engine_core.lenses.pca_lens',
        'class': 'PCALens',
        'description': 'Principal Component Analysis',
        'normalize': 'zscore',  # Required - PCA is scale-sensitive
    },
    'network': {
        'module': 'engine_core.lenses.network_lens',
        'class': 'NetworkLens',
        'description': 'Network/graph analysis',
        'normalize': 'zscore',  # Standardized for correlation-based networks
    },
    'transfer_entropy': {
        'module': 'engine_core.lenses.transfer_entropy_lens',
        'class': 'TransferEntropyLens',
        'description': 'Information flow analysis',
        'normalize': 'diff',  # Requires stationary data
    },
    'mutual_info': {
        'module': 'engine_core.lenses.mutual_info_lens',
        'class': 'MutualInfoLens',
        'description': 'Mutual information dependencies',
        'normalize': 'rank',  # Rank transform captures nonlinear relationships
    },
    'magnitude': {
        'module': 'engine_core.lenses.magnitude_lens',
        'class': 'MagnitudeLens',
        'description': 'Basic magnitude/volatility',
        'normalize': 'pct',  # Percent changes for volatility
    },
    'decomposition': {
        'module': 'engine_core.lenses.decomposition_lens',
        'class': 'DecompositionLens',
        'description': 'Time series decomposition',
        'normalize': None,  # Works on raw levels
    },
    'dmd': {
        'module': 'engine_core.lenses.dmd_lens',
        'class': 'DMDLens',
        'description': 'Dynamic Mode Decomposition',
        'normalize': 'zscore',  # Standardized for mode extraction
    },
    'influence': {
        'module': 'engine_core.lenses.influence_lens',
        'class': 'InfluenceLens',
        'description': 'Influence/impact analysis',
        'normalize': 'diff',  # Stationary for influence detection
    },
    'tda': {
        'module': 'engine_core.lenses.tda_lens',
        'class': 'TDALens',
        'description': 'Topological Data Analysis',
        'normalize': 'zscore',  # Standardized for distance calculations
    },
}


def get_lens(name: str):
    """Dynamically import and return a lens class instance."""
    if name not in LENS_REGISTRY:
        raise ValueError(f"Unknown lens: {name}")
    
    info = LENS_REGISTRY[name]
    
    import importlib
    module = importlib.import_module(info['module'])
    lens_class = getattr(module, info['class'])
    
    return lens_class()


# -----------------------------------------------------------------------------
# Data Loading
# -----------------------------------------------------------------------------

def normalize_date_column(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure DataFrame has a 'date' column (not just as index)."""
    if df.empty:
        return df
    
    # If 'date' column already exists, we're good
    if 'date' in df.columns:
        return df
    
    # Check for case variations
    for col in df.columns:
        if col.lower() == 'date':
            df = df.rename(columns={col: 'date'})
            return df
    
    # If index is datetime, reset it to create 'date' column
    if isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index()
        # Rename the index column to 'date' if needed
        if df.columns[0] != 'date':
            df = df.rename(columns={df.columns[0]: 'date'})
        return df
    
    # If index has a name that looks like a date column
    if df.index.name and 'date' in df.index.name.lower():
        df = df.reset_index()
        df = df.rename(columns={df.columns[0]: 'date'})
        return df
    
    # Last resort: if index looks like dates, use it
    try:
        if len(df.index) > 0:
            pd.to_datetime(df.index[0])
            df = df.reset_index()
            df = df.rename(columns={df.columns[0]: 'date'})
            return df
    except:
        pass
    
    return df


def load_data(config: dict) -> pd.DataFrame:
    """Load data from database or fallback sources."""
    data_cfg = config.get('data', {})
    start_date = data_cfg.get('start_date', '2000-01-01')
    end_date = data_cfg.get('end_date')
    
    print(f"\n📥 Loading data...")
    print(f"   Date range: {start_date} to {end_date or 'today'}")
    
    # Try database first
    try:
        from data.sql.db_connector import load_all_indicators_wide
        df = load_all_indicators_wide(start_date=start_date, end_date=end_date)
        
        if not df.empty:
            df = normalize_date_column(df)
            print(f"   Loaded from DB: {df.shape[0]} rows × {df.shape[1]-1} indicators")
            return df
    except Exception as e:
        print(f"   DB load failed: {e}")
    
    # Try runtime loader
    try:
        from panel.runtime_loader import load_calibrated_panel
        df = load_calibrated_panel(start_date=start_date, end_date=end_date)
        
        if not df.empty:
            df = normalize_date_column(df)
            print(f"   Loaded from panel: {df.shape[0]} rows × {df.shape[1]-1} indicators")
            return df
    except Exception as e:
        print(f"   Panel load failed: {e}")
    
    # Try CSV fallback
    csv_paths = [
        PROJECT_ROOT / 'data' / 'economic_data.csv',
        PROJECT_ROOT / 'data' / 'panel_data.csv',
    ]
    
    for csv_path in csv_paths:
        if csv_path.exists():
            df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
            df = normalize_date_column(df)
            print(f"   Loaded from CSV: {df.shape[0]} rows × {df.shape[1]-1} indicators")
            return df
    
    raise RuntimeError("No data source available")


def preprocess_data(df: pd.DataFrame, min_coverage: float = 0.7) -> pd.DataFrame:
    """
    Preprocess data to handle missing values.
    
    Note: Forward fill is now handled at the DB level in load_all_indicators_wide().
    
    Args:
        df: Raw data with 'date' column (already forward-filled from DB)
        min_coverage: Minimum fraction of non-null values required (0-1)
    
    Returns:
        Cleaned DataFrame ready for analysis
    """
    if df.empty:
        return df
    
    # Separate date column
    date_col = df['date'] if 'date' in df.columns else None
    numeric_df = df.select_dtypes(include=[np.number])
    
    # Calculate coverage (data is already forward-filled from DB)
    coverage = numeric_df.notna().mean()
    good_indicators = coverage[coverage >= min_coverage].index.tolist()
    
    if len(good_indicators) == 0:
        print(f"   ⚠️  No indicators meet {min_coverage:.0%} coverage threshold")
        # Fall back to top 20 by coverage
        good_indicators = coverage.nlargest(20).index.tolist()
    
    filtered_df = numeric_df[good_indicators].copy()
    
    # Drop any remaining rows with NaN (typically at the start before first data point)
    filtered_df = filtered_df.dropna()
    
    # Re-attach date column
    if date_col is not None:
        # Align date with filtered data
        filtered_df.insert(0, 'date', date_col.iloc[filtered_df.index].values)
    
    dropped = len(coverage) - len(good_indicators)
    if dropped > 0:
        print(f"   📊 Filtered: {len(good_indicators)} indicators (dropped {dropped} with <{min_coverage:.0%} coverage)")
    print(f"   📊 After preprocessing: {filtered_df.shape[0]} rows × {len(good_indicators)} indicators")
    
    return filtered_df.reset_index(drop=True)


# -----------------------------------------------------------------------------
# Analysis Runner
# -----------------------------------------------------------------------------

class AnalysisRunner:
    """Runs multiple lenses and aggregates results."""
    
    def __init__(self, config: dict):
        self.config = config
        self.results: Dict[str, Any] = {}
        self.rankings: Dict[str, pd.DataFrame] = {}
        self.errors: Dict[str, str] = {}
    
    def run_lens(self, name: str, df: pd.DataFrame) -> Optional[Dict]:
        """Run a single lens with appropriate normalization."""
        try:
            lens = get_lens(name)
            
            # Get lens-specific params from config
            lens_params = self.config.get('analysis', {}).get(name, {})
            
            # Apply lens-specific normalization
            normalize_method = LENS_REGISTRY[name].get('normalize')
            if normalize_method:
                df_normalized = normalize_data(df, normalize_method)
                # Drop any rows with NaN/inf from normalization
                numeric_cols = df_normalized.select_dtypes(include=[np.number]).columns
                df_normalized = df_normalized.replace([np.inf, -np.inf], np.nan)
                df_normalized = df_normalized.dropna(subset=numeric_cols)
            else:
                df_normalized = df
            
            # Dynamic adjustment for PCA - cap n_components to data dimensions
            if name == 'pca' and 'n_components' in lens_params:
                n_features = len([c for c in df_normalized.columns if c != 'date'])
                lens_params['n_components'] = min(lens_params['n_components'], n_features)
            
            result = lens.analyze(df_normalized, **lens_params)
            
            # Try to get rankings (using normalized data)
            try:
                rankings = lens.rank_indicators(df_normalized)
                if isinstance(rankings, pd.DataFrame):
                    # Ensure index has indicator names
                    if rankings.index.dtype == 'int64' and 'indicator' not in rankings.columns:
                        # Index is integers, try to map to indicator names
                        indicator_cols = [c for c in df_normalized.columns if c != 'date']
                        if len(indicator_cols) == len(rankings):
                            rankings.index = indicator_cols
                    self.rankings[name] = rankings
                elif isinstance(rankings, pd.Series):
                    # Convert Series to DataFrame, preserving index
                    if rankings.index.dtype == 'int64':
                        # Index is integers, try to map to indicator names
                        indicator_cols = [c for c in df_normalized.columns if c != 'date']
                        if len(indicator_cols) == len(rankings):
                            rankings.index = indicator_cols
                    self.rankings[name] = rankings.to_frame(name='score')
            except:
                pass
            
            return result
            
        except Exception as e:
            self.errors[name] = str(e)
            return None
    
    def run_all(self, df: pd.DataFrame, lenses: List[str]) -> Dict[str, Any]:
        """Run all specified lenses."""
        print(f"\n🔬 Running {len(lenses)} lenses...")
        
        for name in lenses:
            norm = LENS_REGISTRY.get(name, {}).get('normalize') or 'raw'
            print(f"\n   ▶ {name} [{norm}]...", end=" ")
            
            result = self.run_lens(name, df)
            
            if result:
                self.results[name] = result
                print(f"✓")
                self._print_highlights(name, result)
            else:
                print(f"✗ ({self.errors.get(name, 'unknown error')})")
        
        return self.results
    
    def _print_highlights(self, name: str, result: Dict):
        """Print key findings from a lens."""
        if name == 'granger':
            n_sig = result.get('n_significant', 0)
            leaders = list(result.get('top_leaders', []))[:3]
            print(f"       Significant pairs: {n_sig}, Leaders: {leaders}")
            
        elif name == 'regime':
            n_regimes = result.get('n_regimes', 0)
            current = result.get('current_regime_character', 'unknown')
            print(f"       Regimes: {n_regimes}, Current: {current}")
            
        elif name == 'clustering':
            n_clusters = result.get('n_clusters', 0)
            print(f"       Clusters: {n_clusters}")
            
        elif name == 'wavelet':
            periods = list(result.get('significant_periods', []))
            print(f"       Significant periods: {periods}")
            
        elif name == 'anomaly':
            n_anomalies = result.get('n_anomalies', 0)
            rate = result.get('anomaly_rate', 0)
            print(f"       Anomalies: {n_anomalies} ({rate:.1%})")
            
        elif name == 'pca':
            variance = list(result.get('explained_variance_ratio', []))
            if variance:
                cum = sum(variance[:3]) if len(variance) >= 3 else sum(variance)
                print(f"       Top 3 components explain: {cum:.1%}")
    
    def get_consensus_rankings(self) -> pd.DataFrame:
        """Aggregate rankings across all lenses."""
        if not self.rankings:
            return pd.DataFrame()
        
        # Combine all rankings
        all_scores = {}
        for lens_name, ranking_df in self.rankings.items():
            if isinstance(ranking_df, pd.DataFrame) and 'score' in ranking_df.columns:
                for idx, row in ranking_df.iterrows():
                    indicator = idx if isinstance(idx, str) else row.get('indicator', str(idx))
                    if indicator not in all_scores:
                        all_scores[indicator] = {}
                    all_scores[indicator][lens_name] = row['score']
        
        if not all_scores:
            return pd.DataFrame()
        
        df = pd.DataFrame(all_scores).T
        df['mean_score'] = df.mean(axis=1)
        df['n_lenses'] = df.notna().sum(axis=1) - 1
        df = df.sort_values('mean_score', ascending=False)
        
        return df
    
    def save_results(self, output_dir: Path):
        """Save results to files."""
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save raw results as JSON
        def clean_for_json(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.int64, np.float64)):
                return float(obj)
            elif isinstance(obj, pd.DataFrame):
                return obj.to_dict('records')
            elif isinstance(obj, pd.Series):
                return obj.to_dict()
            elif isinstance(obj, dict):
                return {k: clean_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [clean_for_json(v) for v in obj]
            return obj
        
        results_path = output_dir / f'analysis_results_{timestamp}.json'
        with open(results_path, 'w') as f:
            json.dump(clean_for_json(self.results), f, indent=2, default=str)
        print(f"\n   Saved results: {results_path}")
        
        # Save consensus rankings
        consensus = self.get_consensus_rankings()
        if not consensus.empty:
            rankings_path = output_dir / f'consensus_rankings_{timestamp}.csv'
            consensus.to_csv(rankings_path)
            print(f"   Saved rankings: {rankings_path}")
    
    def print_summary(self):
        """Print analysis summary."""
        print("\n" + "=" * 60)
        print("ANALYSIS SUMMARY")
        print("=" * 60)
        
        print(f"\n  Lenses run: {len(self.results)}")
        print(f"  Errors: {len(self.errors)}")
        
        if self.errors:
            print("\n  Failed lenses:")
            for name, error in self.errors.items():
                print(f"    - {name}: {error[:50]}...")
        
        # Consensus rankings
        consensus = self.get_consensus_rankings()
        if not consensus.empty:
            print("\n  Top 10 Indicators (Consensus):")
            for i, (indicator, row) in enumerate(consensus.head(10).iterrows()):
                score = row.get('mean_score', 0)
                n = int(row.get('n_lenses', 0))
                print(f"    {i+1}. {indicator}: {score:.3f} ({n} lenses)")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='PRISM Unified Analysis Runner')
    parser.add_argument('--start', type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, help='End date (YYYY-MM-DD)')
    parser.add_argument('--lenses', nargs='*', help='Specific lenses to run')
    parser.add_argument('--coverage', type=float, help='Min coverage threshold 0-1 (default 0.5)')
    parser.add_argument('--list', action='store_true', help='List available lenses')
    parser.add_argument('--no-save', action='store_true', help='Do not save results to files')
    parser.add_argument('--no-db', action='store_true', help='Do not save results to database')
    parser.add_argument('--quiet', '-q', action='store_true', help='Minimal output')
    
    args = parser.parse_args()
    
    # List mode
    if args.list:
        print("\nAvailable Lenses:")
        print("-" * 65)
        print(f"  {'Name':<18} {'Normalize':<10} Description")
        print("-" * 65)
        for name, info in LENS_REGISTRY.items():
            norm = info.get('normalize', 'none') or 'none'
            print(f"  {name:<18} {norm:<10} {info['description']}")
        return 0
    
    # Load config
    config = load_config()
    
    # Override with CLI args
    if args.start:
        config['data']['start_date'] = args.start
    if args.end:
        config['data']['end_date'] = args.end
    if args.coverage:
        config['data']['min_coverage'] = args.coverage
    
    # Determine lenses to run
    if args.lenses:
        lenses = args.lenses
    else:
        lenses = config.get('analysis', {}).get('lenses', list(LENS_REGISTRY.keys()))
    
    # Filter out temporal (use temporal.py for that)
    lenses = [l for l in lenses if l != 'temporal']
    
    print("=" * 60)
    print("🔬 PRISM UNIFIED ANALYSIS")
    print(f"   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # Load data
    try:
        df = load_data(config)
    except Exception as e:
        print(f"\n❌ Failed to load data: {e}")
        return 1
    
    # Preprocess data
    min_coverage = config.get('data', {}).get('min_coverage', 0.7)
    df = preprocess_data(df, min_coverage=min_coverage)
    
    if df.empty:
        print("\n❌ No data after preprocessing")
        return 1
    
    # Run analysis
    runner = AnalysisRunner(config)
    runner.run_all(df, lenses)
    
    # Save results
    if not args.no_save and config.get('analysis', {}).get('save_results', True):
        output_dir = Path(config.get('analysis', {}).get('output_dir', 'output/analysis'))
        runner.save_results(output_dir)
    
    # Save to database
    if not args.no_db:
        try:
            from data.sql.db_connector import save_analysis_run
            
            # Gather normalize methods
            normalize_methods = {
                name: LENS_REGISTRY.get(name, {}).get('normalize', 'none')
                for name in lenses
            }
            
            # Get data dimensions
            n_indicators = len([c for c in df.columns if c != 'date'])
            n_rows = len(df)
            
            run_id = save_analysis_run(
                start_date=config.get('data', {}).get('start_date', '2000-01-01'),
                end_date=config.get('data', {}).get('end_date'),
                n_indicators=n_indicators,
                n_rows=n_rows,
                n_lenses=len(runner.results),
                n_errors=len(runner.errors),
                config=config,
                lens_results=runner.results,
                lens_errors=runner.errors,
                rankings=runner.rankings,
                consensus=runner.get_consensus_rankings(),
                normalize_methods=normalize_methods,
            )
            print(f"   💾 Saved to database: run_id={run_id}")
        except Exception as e:
            print(f"   ⚠️  DB save failed: {e}")
    
    # Print summary
    runner.print_summary()
    
    print("\n✅ Analysis complete!")
    return 0


if __name__ == '__main__':
    sys.exit(main())
