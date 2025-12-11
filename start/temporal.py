#!/usr/bin/env python3
"""
PRISM Temporal Analysis Runner
==============================
Runs rolling-window analysis for time-varying patterns.

Usage:
    python temporal.py                    # Run with config defaults
    python temporal.py --windows 63 126   # Custom window sizes
    python temporal.py --step 10          # Custom step size

Reads settings from: prism_config.yaml
"""

import sys
import argparse
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
    
    return {
        'data': {'start_date': '2000-01-01', 'end_date': None},
        'temporal': {
            'output_dir': 'output/temporal',
            'windows': [63, 126, 252],
            'step_size': 21,
        },
    }


# -----------------------------------------------------------------------------
# Data Loading
# -----------------------------------------------------------------------------

def load_data(config: dict) -> pd.DataFrame:
    """Load data from database or fallback sources."""
    data_cfg = config.get('data', {})
    start_date = data_cfg.get('start_date', '2000-01-01')
    end_date = data_cfg.get('end_date')
    
    print(f"\n📥 Loading data...")
    print(f"   Date range: {start_date} to {end_date or 'today'}")
    
    # Try database first
    try:
        from data.duckdb_connector import load_all_indicators_wide
        df = load_all_indicators_wide(start_date=start_date, end_date=end_date)
        
        if not df.empty:
            print(f"   Loaded from DB: {df.shape[0]} rows × {df.shape[1]} indicators")
            return df
    except Exception as e:
        print(f"   DB load failed: {e}")
    
    # Try runtime loader
    try:
        from panel.runtime_loader import load_calibrated_panel
        df = load_calibrated_panel(start_date=start_date, end_date=end_date)
        
        if not df.empty:
            print(f"   Loaded from panel: {df.shape[0]} rows × {df.shape[1]} indicators")
            return df
    except Exception as e:
        print(f"   Panel load failed: {e}")
    
    raise RuntimeError("No data source available")


# -----------------------------------------------------------------------------
# Temporal Lenses
# -----------------------------------------------------------------------------

def rolling_volatility(returns: pd.DataFrame, window: int) -> pd.DataFrame:
    """Calculate rolling volatility for each indicator."""
    return returns.rolling(window).std()


def rolling_correlation(returns: pd.DataFrame, window: int) -> pd.Series:
    """Calculate rolling mean pairwise correlation."""
    def calc_mean_corr(window_data):
        corr = window_data.corr()
        # Get upper triangle (excluding diagonal)
        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
        return corr.where(mask).stack().mean()
    
    results = []
    for i in range(window, len(returns)):
        window_data = returns.iloc[i-window:i]
        mean_corr = calc_mean_corr(window_data)
        results.append({'date': returns.index[i], 'mean_correlation': mean_corr})
    
    return pd.DataFrame(results).set_index('date')['mean_correlation']


def rolling_dispersion(returns: pd.DataFrame, window: int) -> pd.Series:
    """Calculate rolling cross-sectional dispersion."""
    return returns.rolling(window).apply(lambda x: x.std(axis=0).mean())


def detect_regime_changes(series: pd.Series, threshold: float = 2.0) -> List[int]:
    """Detect regime changes using z-score threshold."""
    zscore = (series - series.rolling(63).mean()) / series.rolling(63).std()
    changes = []
    
    for i in range(len(zscore)):
        if abs(zscore.iloc[i]) > threshold:
            changes.append(i)
    
    return changes


# -----------------------------------------------------------------------------
# Temporal Runner
# -----------------------------------------------------------------------------

class TemporalRunner:
    """Runs rolling-window temporal analysis."""
    
    def __init__(self, config: dict):
        self.config = config
        self.results: Dict[str, pd.DataFrame] = {}
    
    def run(self, df: pd.DataFrame, windows: List[int], step: int) -> Dict[str, Any]:
        """Run temporal analysis across multiple windows."""
        print(f"\n⏱️ Running temporal analysis...")
        print(f"   Windows: {windows}")
        print(f"   Step size: {step}")
        
        # Calculate returns
        returns = df.pct_change().dropna(how='all')
        
        for window in windows:
            print(f"\n   Window: {window} days")
            
            # Rolling volatility
            vol = rolling_volatility(returns, window)
            self.results[f'volatility_{window}'] = vol
            print(f"     ✓ Volatility")
            
            # Rolling correlation
            try:
                corr = rolling_correlation(returns, window)
                self.results[f'correlation_{window}'] = corr
                print(f"     ✓ Correlation")
            except Exception as e:
                print(f"     ✗ Correlation: {e}")
            
            # Rolling dispersion
            try:
                disp = returns.rolling(window).std().mean(axis=1)
                self.results[f'dispersion_{window}'] = disp
                print(f"     ✓ Dispersion")
            except Exception as e:
                print(f"     ✗ Dispersion: {e}")
        
        return self.results
    
    def find_consensus_events(self, threshold: float = 0.8) -> pd.DataFrame:
        """Find dates where multiple metrics agree on elevated stress."""
        # Combine volatility metrics
        vol_cols = [k for k in self.results.keys() if k.startswith('volatility')]
        
        if not vol_cols:
            return pd.DataFrame()
        
        # Standardize each metric
        standardized = {}
        for key, data in self.results.items():
            if isinstance(data, pd.DataFrame):
                mean_series = data.mean(axis=1)
            else:
                mean_series = data
            
            zscore = (mean_series - mean_series.mean()) / mean_series.std()
            standardized[key] = zscore
        
        # Find consensus spikes
        combined = pd.DataFrame(standardized)
        consensus = combined.mean(axis=1)
        
        # Flag high consensus dates
        events = consensus[consensus > threshold * consensus.std()]
        
        return events.to_frame('consensus_score')
    
    def save_results(self, output_dir: Path):
        """Save results to files."""
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        for name, data in self.results.items():
            if isinstance(data, (pd.DataFrame, pd.Series)):
                path = output_dir / f'{name}_{timestamp}.csv'
                data.to_csv(path)
        
        # Save consensus events
        events = self.find_consensus_events()
        if not events.empty:
            events_path = output_dir / f'consensus_events_{timestamp}.csv'
            events.to_csv(events_path)
            print(f"\n   Saved consensus events: {events_path}")
        
        print(f"\n   Saved {len(self.results)} result files to {output_dir}")
    
    def print_summary(self):
        """Print temporal analysis summary."""
        print("\n" + "=" * 60)
        print("TEMPORAL ANALYSIS SUMMARY")
        print("=" * 60)
        
        print(f"\n  Metrics computed: {len(self.results)}")
        
        # Find consensus events
        events = self.find_consensus_events()
        if not events.empty:
            print(f"\n  High-stress events detected: {len(events)}")
            print("\n  Top 5 consensus events:")
            for i, (date, row) in enumerate(events.nlargest(5, 'consensus_score').iterrows()):
                score = row['consensus_score']
                print(f"    {i+1}. {date.strftime('%Y-%m-%d') if hasattr(date, 'strftime') else date}: {score:.2f}")
        else:
            print("\n  No high-stress events detected")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='PRISM Temporal Analysis Runner')
    parser.add_argument('--start', type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, help='End date (YYYY-MM-DD)')
    parser.add_argument('--windows', nargs='*', type=int, help='Window sizes in days')
    parser.add_argument('--step', type=int, help='Step size in days')
    parser.add_argument('--no-save', action='store_true', help='Do not save results')
    
    args = parser.parse_args()
    
    # Load config
    config = load_config()
    temporal_cfg = config.get('temporal', {})
    
    # Override with CLI args
    if args.start:
        config['data']['start_date'] = args.start
    if args.end:
        config['data']['end_date'] = args.end
    
    windows = args.windows or temporal_cfg.get('windows', [63, 126, 252])
    step = args.step or temporal_cfg.get('step_size', 21)
    
    print("=" * 60)
    print("⏱️ PRISM TEMPORAL ANALYSIS")
    print(f"   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # Load data
    try:
        df = load_data(config)
    except Exception as e:
        print(f"\n❌ Failed to load data: {e}")
        return 1
    
    # Run temporal analysis
    runner = TemporalRunner(config)
    runner.run(df, windows, step)
    
    # Save results
    if not args.no_save:
        output_dir = Path(temporal_cfg.get('output_dir', 'output/temporal'))
        runner.save_results(output_dir)
    
    # Print summary
    runner.print_summary()
    
    print("\n✅ Temporal analysis complete!")
    return 0


if __name__ == '__main__':
    sys.exit(main())
