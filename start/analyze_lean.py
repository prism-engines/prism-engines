#!/usr/bin/env python3
"""
PRISM Lean Analysis
===================
Streamlined 7-lens analysis based on lens geometry findings.

These 7 lenses provide independent perspectives with minimal redundancy:
  1. pca        - Structure (representative of the structure cluster)
  2. granger    - Causality (orthogonal to everything)
  3. transfer_entropy - Information flow (independent)
  4. anomaly    - Stress/risk detection (negatively correlated with structure)
  5. magnitude  - Volatility patterns (independent)
  6. decomposition - Macro trends (unique findings)
  7. mutual_info - Nonlinear relationships (semi-independent)

Dropped due to redundancy with PCA (r > 0.8):
  - clustering, network, regime (all correlate 0.82-0.87 with each other)

Dropped for other reasons:
  - wavelet (moderate correlation, no unique findings)
  - dmd (moderate correlation with structure)
  - influence (overlaps with granger/transfer_entropy)
  - tda (memory intensive, often crashes)

Usage:
    python analyze_lean.py                    # Full date range
    python analyze_lean.py --start 2020-01-01 # From specific date
    python analyze_lean.py --period covid     # Named period
    python analyze_lean.py --quick            # Recent data only
"""

import sys
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd

# Setup paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name == 'start' else SCRIPT_DIR
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# LEAN LENS CONFIGURATION
# =============================================================================

# The 7 independent lenses
LEAN_LENSES = [
    'pca',              # Structure (represents clustering/network/regime cluster)
    'granger',          # Causality - orthogonal to all others
    'transfer_entropy', # Information flow - independent
    'anomaly',          # Stress/risk - negatively correlated with structure
    'magnitude',        # Volatility - independent
    'decomposition',    # Macro trends - unique findings
    'mutual_info',      # Nonlinear relationships - semi-independent
]

# Normalization per lens (from PRISM research)
LEAN_NORMALIZATION = {
    'pca': 'zscore',           # Scale-sensitive
    'granger': 'diff',         # Requires stationarity
    'transfer_entropy': 'diff', # Requires stationarity
    'anomaly': 'robust',       # Outlier detection
    'magnitude': 'pct',        # Percent changes for volatility
    'decomposition': 'none',   # Raw levels for trend extraction
    'mutual_info': 'rank',     # Rank transform for nonlinear
}

# What each lens uniquely contributes (from geometry analysis)
LENS_INSIGHTS = {
    'pca': 'Dominant factors driving covariance structure',
    'granger': 'Predictive causality (who leads whom)',
    'transfer_entropy': 'Information flow direction and magnitude',
    'anomaly': 'Stress indicators (spreads, volatility regimes)',
    'magnitude': 'Volatility clustering and extreme moves',
    'decomposition': 'Macro cycles (unemployment, yield curve)',
    'mutual_info': 'Nonlinear dependencies (tech, bonds)',
}

# Named periods for quick analysis
NAMED_PERIODS = {
    'dot_com': ('2000-01-01', '2002-12-31'),
    'pre_gfc': ('2005-01-01', '2007-06-30'),
    'gfc': ('2007-07-01', '2009-03-31'),
    'recovery': ('2009-04-01', '2012-12-31'),
    'bull': ('2013-01-01', '2019-12-31'),
    'covid': ('2020-01-01', '2020-06-30'),
    'post_covid': ('2020-07-01', '2021-12-31'),
    'inflation': ('2022-01-01', '2023-12-31'),
    'recent': ('2024-01-01', None),
}


# =============================================================================
# DATA LOADING
# =============================================================================

def load_data(start_date: str = '2000-01-01', end_date: Optional[str] = None,
              coverage_threshold: float = 0.7) -> pd.DataFrame:
    """Load and filter indicator data."""
    from data.sql.db_connector import load_all_indicators_wide
    
    df = load_all_indicators_wide(start_date=start_date, end_date=end_date, ffill=True)
    
    if df.empty:
        return df
    
    # Ensure date is a column, not index
    if 'date' not in df.columns and df.index.name == 'date':
        df = df.reset_index()
    
    # Filter by coverage
    n_rows = len(df)
    numeric_cols = [c for c in df.columns if c != 'date']
    coverage = df[numeric_cols].notna().sum() / n_rows
    good_cols = [c for c in numeric_cols if coverage[c] >= coverage_threshold]
    
    # Keep date + good columns
    df = df[['date'] + good_cols] if 'date' in df.columns else df[good_cols]
    
    # Drop rows with any NaN in numeric columns
    df = df.dropna(subset=good_cols)
    
    return df


# =============================================================================
# NORMALIZATION
# =============================================================================

def normalize_data(df: pd.DataFrame, method: str) -> pd.DataFrame:
    """Apply normalization to numeric columns."""
    if method == 'none' or method is None:
        return df.copy()
    
    result = df.copy()
    numeric_cols = [c for c in df.columns if c != 'date']
    
    if method == 'zscore':
        for col in numeric_cols:
            mean = result[col].mean()
            std = result[col].std()
            if std > 0:
                result[col] = (result[col] - mean) / std
    
    elif method == 'diff':
        for col in numeric_cols:
            result[col] = result[col].diff()
        result = result.iloc[1:]  # Drop first NaN row
    
    elif method == 'pct':
        for col in numeric_cols:
            result[col] = result[col].pct_change()
        result = result.iloc[1:]
        result = result.replace([np.inf, -np.inf], np.nan)
    
    elif method == 'rank':
        for col in numeric_cols:
            result[col] = result[col].rank(pct=True)
    
    elif method == 'robust':
        for col in numeric_cols:
            median = result[col].median()
            q75, q25 = result[col].quantile([0.75, 0.25])
            iqr = q75 - q25
            if iqr > 0:
                result[col] = (result[col] - median) / iqr
    
    return result


# =============================================================================
# ANALYSIS ENGINE
# =============================================================================

class LeanAnalyzer:
    """Streamlined analyzer using 7 independent lenses."""
    
    def __init__(self):
        self.results = {}
        self.rankings = {}
        self.errors = {}
    
    def run_lens(self, name: str, df: pd.DataFrame) -> Optional[Dict]:
        """Run a single lens with appropriate normalization."""
        from lenses import get_lens
        
        try:
            lens = get_lens(name)
            
            # Apply lens-specific normalization
            norm_method = LEAN_NORMALIZATION.get(name, 'none')
            df_norm = normalize_data(df, norm_method)
            
            # Clean up any remaining NaN/inf
            numeric_cols = df_norm.select_dtypes(include=[np.number]).columns
            df_norm = df_norm.replace([np.inf, -np.inf], np.nan)
            df_norm = df_norm.dropna(subset=numeric_cols)
            
            if len(df_norm) < 100:
                raise ValueError(f"Insufficient data after normalization: {len(df_norm)} rows")
            
            # Run lens
            result = lens.analyze(df_norm)
            self.results[name] = result
            
            # Get rankings
            try:
                rankings = lens.rank_indicators(df_norm)
                if isinstance(rankings, pd.DataFrame):
                    # Ensure indicator names in index
                    if rankings.index.dtype == 'int64':
                        indicator_cols = [c for c in df_norm.columns if c != 'date']
                        if len(indicator_cols) == len(rankings):
                            rankings.index = indicator_cols
                    self.rankings[name] = rankings
                elif isinstance(rankings, pd.Series):
                    if rankings.index.dtype == 'int64':
                        indicator_cols = [c for c in df_norm.columns if c != 'date']
                        if len(indicator_cols) == len(rankings):
                            rankings.index = indicator_cols
                    self.rankings[name] = rankings.to_frame(name='score')
            except Exception:
                pass
            
            return result
            
        except Exception as e:
            self.errors[name] = str(e)
            return None
    
    def run_all(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Run all 7 lean lenses."""
        print(f"\n🔬 Running {len(LEAN_LENSES)} independent lenses...")
        
        for name in LEAN_LENSES:
            norm = LEAN_NORMALIZATION.get(name, 'raw')
            print(f"\n   ▶ {name} [{norm}]...", end=" ", flush=True)
            
            result = self.run_lens(name, df)
            
            if result:
                print("✓", end="")
                self._print_lens_summary(name, result)
            else:
                print(f"✗ ({self.errors.get(name, 'unknown error')})")
        
        return self.results
    
    def _print_lens_summary(self, name: str, result: Dict):
        """Print brief summary for each lens."""
        summaries = {
            'pca': lambda r: f"Top 3 explain {r.get('explained_variance_top3', 0)*100:.1f}%",
            'granger': lambda r: f"Leaders: {r.get('top_leaders', [])[:3]}",
            'transfer_entropy': lambda r: f"Hubs: {list(r.get('information_hubs', {}).keys())[:3]}",
            'anomaly': lambda r: f"Anomalies: {r.get('n_anomalies', 0)} ({r.get('anomaly_pct', 0):.1f}%)",
            'magnitude': lambda r: f"High vol: {list(r.get('top_volatile', {}).keys())[:3] if isinstance(r.get('top_volatile'), dict) else []}",
            'decomposition': lambda r: f"Trend strength: {r.get('avg_trend_strength', 0):.2f}",
            'mutual_info': lambda r: f"Strongest pair: {r.get('indicator_1', '?')}-{r.get('indicator_2', '?')}",
        }
        
        if name in summaries:
            try:
                summary = summaries[name](result)
                print(f"\n       {summary}")
            except Exception:
                pass
    
    def compute_consensus(self) -> pd.DataFrame:
        """Compute consensus rankings across lenses."""
        if not self.rankings:
            return pd.DataFrame()
        
        # Collect all indicators
        all_indicators = set()
        for ranking_df in self.rankings.values():
            all_indicators.update(ranking_df.index.tolist())
        
        # Build score matrix
        scores = pd.DataFrame(index=list(all_indicators))
        
        for lens_name, ranking_df in self.rankings.items():
            if 'score' in ranking_df.columns:
                # Normalize scores to [0, 1]
                s = ranking_df['score']
                if s.max() != s.min():
                    s_norm = (s - s.min()) / (s.max() - s.min())
                else:
                    s_norm = s * 0 + 0.5
                scores[lens_name] = s_norm
        
        # Compute mean score and count
        consensus = pd.DataFrame({
            'mean_score': scores.mean(axis=1),
            'n_lenses': scores.notna().sum(axis=1),
        })
        
        consensus = consensus.sort_values('mean_score', ascending=False)
        consensus['rank'] = range(1, len(consensus) + 1)
        
        return consensus


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='PRISM Lean Analysis (7 independent lenses)')
    parser.add_argument('--start', '-s', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', '-e', help='End date (YYYY-MM-DD)')
    parser.add_argument('--period', '-p', choices=list(NAMED_PERIODS.keys()),
                        help='Named period to analyze')
    parser.add_argument('--quick', '-q', action='store_true',
                        help='Quick mode: last 2 years only')
    parser.add_argument('--coverage', '-c', type=float, default=0.7,
                        help='Minimum coverage threshold (default: 0.7)')
    parser.add_argument('--no-db', action='store_true',
                        help='Skip saving to database')
    
    args = parser.parse_args()
    
    # Determine date range
    if args.period:
        start_date, end_date = NAMED_PERIODS[args.period]
    elif args.quick:
        start_date = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')
        end_date = None
    else:
        start_date = args.start or '2000-01-01'
        end_date = args.end
    
    # Header
    print("=" * 60)
    print("🎯 PRISM LEAN ANALYSIS")
    print(f"   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    print("\n   7 Independent Lenses (based on geometry analysis):")
    for lens in LEAN_LENSES:
        print(f"     • {lens}: {LENS_INSIGHTS[lens]}")
    
    # Load data
    print(f"\n📥 Loading data...")
    print(f"   Date range: {start_date} to {end_date or 'today'}")
    
    df = load_data(start_date, end_date, args.coverage)
    
    if df.empty:
        print("   ❌ No data loaded!")
        return 1
    
    n_indicators = len([c for c in df.columns if c != 'date'])
    print(f"   Loaded: {len(df)} rows × {n_indicators} indicators")
    
    # Run analysis
    analyzer = LeanAnalyzer()
    results = analyzer.run_all(df)
    
    # Compute consensus
    consensus = analyzer.compute_consensus()
    
    # Summary
    print("\n")
    print("=" * 60)
    print("LEAN ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"\n  Lenses run: {len(results)}")
    print(f"  Errors: {len(analyzer.errors)}")
    
    if not consensus.empty:
        print(f"\n  Top 10 Indicators (Consensus of {len(LEAN_LENSES)} independent lenses):")
        for i, (indicator, row) in enumerate(consensus.head(10).iterrows()):
            print(f"    {i+1:2}. {indicator}: {row['mean_score']:.3f} ({int(row['n_lenses'])} lenses)")
    
    # Save to database
    if not args.no_db:
        try:
            from data.sql.db_connector import save_analysis_run
            
            run_id = save_analysis_run(
                start_date=start_date,
                end_date=end_date,
                n_indicators=n_indicators,
                n_rows=len(df),
                n_lenses=len(results),
                n_errors=len(analyzer.errors),
                config={'mode': 'lean', 'lenses': LEAN_LENSES},
                lens_results=analyzer.results,
                lens_errors=analyzer.errors,
                rankings=analyzer.rankings,
                consensus=consensus,
                normalize_methods=LEAN_NORMALIZATION,
            )
            print(f"\n   💾 Saved to database: run_id={run_id}")
        except Exception as e:
            print(f"\n   ⚠️  Could not save to DB: {e}")
    
    print("\n✅ Lean analysis complete!")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
