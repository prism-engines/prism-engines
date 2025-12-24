#!/usr/bin/env python3
"""
PRISM CROSS-DOMAIN OBSERVATORY
==============================

The big overnight run. Analyzes geometric structure across:
- Finance (2005-2025)
- Climate (1948-2025)  
- Epidemiology (2005-2025)

Questions we're answering:
1. Does each domain have distinct geometric states?
2. Do state transitions correlate across domains?
3. Is there universal structure in complex systems?
4. Does El Niño geometry look like market stress?
5. Do flu outbreaks have geometric signatures?

Run time: Hours to days depending on parameters.

Usage:
    python scripts/cross_domain_analysis.py
    python scripts/cross_domain_analysis.py --quick  # Fast test run
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# PRISM imports
from prism.db.connection import get_db_path, get_connection
from prism.structure import StructureExtractor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# ============================================================
# DATA LOADING
# ============================================================

def load_finance_data(conn, start_date='2005-01-01') -> pd.DataFrame:
    """Load finance indicators."""
    df = conn.execute(f'''
        SELECT date, indicator_id, value
        FROM clean.indicators
        WHERE date >= '{start_date}'
        AND indicator_id IN ('SPY', 'XLK', 'XLF', 'XLE', 'XLU', 'AGG', 'GLD', 'VIXCLS')
    ''').fetchdf()
    
    if len(df) == 0:
        return pd.DataFrame()
    
    pivot = df.pivot(index='date', columns='indicator_id', values='value')
    pivot.index = pd.to_datetime(pivot.index)
    return pivot.dropna()


def load_climate_data(conn, start_date='1950-01-01') -> pd.DataFrame:
    """Load climate indicators."""
    df = conn.execute(f'''
        SELECT date, indicator_id, value
        FROM domains.indicators
        WHERE domain = 'climate'
        AND date >= '{start_date}'
    ''').fetchdf()
    
    if len(df) == 0:
        return pd.DataFrame()
    
    pivot = df.pivot(index='date', columns='indicator_id', values='value')
    pivot.index = pd.to_datetime(pivot.index)
    return pivot.dropna()


def load_epi_data(conn, start_date='2005-01-01') -> pd.DataFrame:
    """Load epidemiology indicators."""
    df = conn.execute(f'''
        SELECT date, indicator_id, value
        FROM domains.indicators
        WHERE domain = 'epidemiology'
        AND date >= '{start_date}'
    ''').fetchdf()
    
    if len(df) == 0:
        return pd.DataFrame()
    
    pivot = df.pivot(index='date', columns='indicator_id', values='value')
    pivot.index = pd.to_datetime(pivot.index)
    return pivot.dropna()


# ============================================================
# STRUCTURE EXTRACTION (Per Domain)
# ============================================================

def extract_domain_structure(
    df: pd.DataFrame,
    domain_name: str,
    windows: List[int] = [63, 126, 252],
    n_states: List[int] = [3, 4, 5],
    step: int = 5
) -> Dict[str, Any]:
    """
    Extract geometric structure for a domain across multiple parameters.
    
    Returns dict with:
    - geometry: raw geometry metrics
    - states: cluster labels for each n_states
    - transitions: state change points
    - fingerprint: summary statistics
    """
    logger.info(f"Extracting structure for {domain_name}: {len(df)} rows, {len(df.columns)} indicators")
    
    results = {
        'domain': domain_name,
        'n_rows': len(df),
        'n_indicators': len(df.columns),
        'indicators': list(df.columns),
        'date_range': (str(df.index.min().date()), str(df.index.max().date())),
        'windows': {},
    }
    
    for window in windows:
        if len(df) < window + 50:
            logger.warning(f"Skipping window {window} for {domain_name} - not enough data")
            continue
            
        logger.info(f"  Window {window} ({window/21:.0f} months)...")
        
        extractor = StructureExtractor(df)
        
        try:
            geometry = extractor.extract_geometry(
                window=window,
                step=step,
                verbose=False
            )
            
            window_results = {
                'n_points': len(geometry),
                'geometry_stats': {},
                'states': {},
            }
            
            # Store geometry statistics
            for col in geometry.columns:
                window_results['geometry_stats'][col] = {
                    'mean': float(geometry[col].mean()),
                    'std': float(geometry[col].std()),
                    'min': float(geometry[col].min()),
                    'max': float(geometry[col].max()),
                }
            
            # Cluster into different numbers of states
            for n in n_states:
                try:
                    state_labels = extractor.cluster_geometry(n_states=n)
                    
                    # Find transitions
                    transitions = []
                    prev_state = None
                    for i, (date, state) in enumerate(zip(geometry.index, state_labels)):
                        if state != prev_state:
                            transitions.append({
                                'date': str(date.date()) if hasattr(date, 'date') else str(date),
                                'from_state': int(prev_state) if prev_state is not None else None,
                                'to_state': int(state)
                            })
                            prev_state = state
                    
                    # State characteristics
                    state_chars = {}
                    for state_id in range(n):
                        mask = state_labels == state_id
                        if mask.sum() > 0:
                            state_geo = geometry[mask]
                            state_chars[int(state_id)] = {
                                'count': int(mask.sum()),
                                'pct': float(mask.sum() / len(mask) * 100),
                                'pc1_variance': float(state_geo['pca__variance_pc1'].mean()) if 'pca__variance_pc1' in state_geo else None,
                                'eff_dim': float(state_geo['pca__effective_dimensionality'].mean()) if 'pca__effective_dimensionality' in state_geo else None,
                                'avg_corr': float(state_geo['cross_correlation__avg_abs_correlation'].mean()) if 'cross_correlation__avg_abs_correlation' in state_geo else None,
                            }
                    
                    window_results['states'][n] = {
                        'n_transitions': len(transitions),
                        'transitions': transitions,
                        'state_characteristics': state_chars,
                        'labels': [int(x) for x in state_labels],
                        'dates': [str(d.date()) if hasattr(d, 'date') else str(d) for d in geometry.index],
                    }
                    
                except Exception as e:
                    logger.error(f"Clustering failed for n={n}: {e}")
            
            results['windows'][window] = window_results
            
        except Exception as e:
            logger.error(f"Structure extraction failed for window {window}: {e}")
    
    return results


# ============================================================
# CROSS-DOMAIN ANALYSIS
# ============================================================

def align_domain_states(
    domain_results: Dict[str, Dict],
    window: int = 126,
    n_states: int = 4
) -> pd.DataFrame:
    """
    Align state labels across domains on common dates.
    
    Returns DataFrame with date index and state columns for each domain.
    """
    aligned = {}
    
    for domain_name, results in domain_results.items():
        if window not in results['windows']:
            continue
        if n_states not in results['windows'][window]['states']:
            continue
            
        states = results['windows'][window]['states'][n_states]
        dates = pd.to_datetime(states['dates'])
        labels = states['labels']
        
        aligned[domain_name] = pd.Series(labels, index=dates, name=domain_name)
    
    if len(aligned) < 2:
        return pd.DataFrame()
    
    df = pd.DataFrame(aligned)
    return df.dropna()


def compute_state_correlation(aligned_states: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute correlation between domain state transitions.
    """
    if len(aligned_states) < 10:
        return {'error': 'Not enough aligned data'}
    
    results = {
        'n_aligned_dates': len(aligned_states),
        'date_range': (str(aligned_states.index.min().date()), str(aligned_states.index.max().date())),
        'state_correlations': {},
        'transition_correlations': {},
    }
    
    domains = aligned_states.columns.tolist()
    
    # State value correlations
    for i, d1 in enumerate(domains):
        for d2 in domains[i+1:]:
            corr, pval = stats.spearmanr(aligned_states[d1], aligned_states[d2])
            results['state_correlations'][f'{d1}_vs_{d2}'] = {
                'correlation': float(corr),
                'p_value': float(pval),
                'significant': pval < 0.05
            }
    
    # Transition correlations (did state change on same dates?)
    for domain in domains:
        aligned_states[f'{domain}_changed'] = (aligned_states[domain].diff() != 0).astype(int)
    
    change_cols = [c for c in aligned_states.columns if '_changed' in c]
    for i, c1 in enumerate(change_cols):
        for c2 in change_cols[i+1:]:
            d1 = c1.replace('_changed', '')
            d2 = c2.replace('_changed', '')
            
            # Count co-occurrences
            both_changed = ((aligned_states[c1] == 1) & (aligned_states[c2] == 1)).sum()
            either_changed = ((aligned_states[c1] == 1) | (aligned_states[c2] == 1)).sum()
            
            jaccard = both_changed / either_changed if either_changed > 0 else 0
            
            results['transition_correlations'][f'{d1}_vs_{d2}'] = {
                'both_changed': int(both_changed),
                'either_changed': int(either_changed),
                'jaccard_similarity': float(jaccard),
            }
    
    return results


def find_cross_domain_events(
    aligned_states: pd.DataFrame,
    lookback: int = 30
) -> List[Dict]:
    """
    Find dates where multiple domains transitioned.
    """
    events = []
    
    domains = [c for c in aligned_states.columns if '_changed' not in c]
    
    for domain in domains:
        aligned_states[f'{domain}_changed'] = (aligned_states[domain].diff() != 0).astype(int)
    
    change_cols = [f'{d}_changed' for d in domains]
    aligned_states['n_domains_changed'] = aligned_states[change_cols].sum(axis=1)
    
    # Find dates where 2+ domains changed
    multi_change = aligned_states[aligned_states['n_domains_changed'] >= 2]
    
    for date, row in multi_change.iterrows():
        changed_domains = [d for d in domains if row[f'{d}_changed'] == 1]
        
        event = {
            'date': str(date.date()),
            'n_domains': int(row['n_domains_changed']),
            'domains_changed': changed_domains,
            'states': {d: int(row[d]) for d in domains}
        }
        events.append(event)
    
    return events


# ============================================================
# HISTORICAL EVENT ANALYSIS
# ============================================================

HISTORICAL_EVENTS = {
    # Financial
    'dot_com_crash': '2000-03-10',
    'sept_11': '2001-09-11',
    'gfc_start': '2007-08-09',
    'bear_stearns': '2008-03-16',
    'lehman': '2008-09-15',
    'gfc_bottom': '2009-03-09',
    'flash_crash': '2010-05-06',
    'eu_debt_crisis': '2011-08-05',
    'china_deval': '2015-08-11',
    'covid_start': '2020-01-20',
    'covid_crash': '2020-03-16',
    'covid_bottom': '2020-03-23',
    'inflation_spike': '2022-06-10',
    'svb_collapse': '2023-03-10',
    
    # Climate (El Niño events)
    'el_nino_1997': '1997-11-01',
    'la_nina_1998': '1998-07-01',
    'el_nino_2015': '2015-11-01',
    'el_nino_2023': '2023-06-01',
    
    # Epidemiology
    'h1n1_pandemic': '2009-04-15',
    'covid_pandemic': '2020-03-11',
    'flu_season_2017': '2017-12-01',
}


def analyze_event_geometry(
    domain_results: Dict[str, Dict],
    events: Dict[str, str],
    window: int = 126,
    n_states: int = 4
) -> Dict[str, Any]:
    """
    What was the geometric state at each historical event?
    """
    event_analysis = {}
    
    for event_name, event_date in events.items():
        event_date = pd.Timestamp(event_date)
        event_analysis[event_name] = {'date': str(event_date.date()), 'states': {}}
        
        for domain_name, results in domain_results.items():
            if window not in results['windows']:
                continue
            if n_states not in results['windows'][window]['states']:
                continue
            
            states = results['windows'][window]['states'][n_states]
            dates = pd.to_datetime(states['dates'])
            labels = states['labels']
            
            # Find closest date
            date_diffs = abs(dates - event_date)
            closest_idx = date_diffs.argmin()
            
            if date_diffs[closest_idx].days < 30:  # Within 30 days
                event_analysis[event_name]['states'][domain_name] = {
                    'state': int(labels[closest_idx]),
                    'date_found': str(dates[closest_idx].date()),
                    'days_diff': int(date_diffs[closest_idx].days)
                }
    
    return event_analysis


# ============================================================
# UNIVERSAL STRUCTURE ANALYSIS
# ============================================================

def compare_geometry_distributions(domain_results: Dict[str, Dict], window: int = 126) -> Dict[str, Any]:
    """
    Compare the distribution of geometry metrics across domains.
    
    Are the shapes similar even if the data is different?
    """
    comparison = {}
    
    metrics = [
        'pca__variance_pc1',
        'pca__effective_dimensionality', 
        'cross_correlation__avg_abs_correlation',
    ]
    
    for metric in metrics:
        comparison[metric] = {}
        
        domain_values = {}
        for domain_name, results in domain_results.items():
            if window not in results['windows']:
                continue
            
            stats = results['windows'][window]['geometry_stats']
            if metric in stats:
                domain_values[domain_name] = stats[metric]
        
        if len(domain_values) >= 2:
            # KS test between domains
            domains = list(domain_values.keys())
            comparison[metric]['domain_stats'] = domain_values
            comparison[metric]['ks_tests'] = {}
            
            # We'd need raw values for KS test - for now just compare means
            means = [v['mean'] for v in domain_values.values()]
            stds = [v['std'] for v in domain_values.values()]
            
            comparison[metric]['mean_range'] = (min(means), max(means))
            comparison[metric]['std_range'] = (min(stds), max(stds))
            comparison[metric]['mean_cv'] = np.std(means) / np.mean(means) if np.mean(means) != 0 else None
    
    return comparison


# ============================================================
# MAIN ANALYSIS
# ============================================================

def run_full_analysis(
    db_path: str = None,
    windows: List[int] = [63, 126, 252],
    n_states: List[int] = [3, 4, 5],
    step: int = 5,
    output_dir: str = 'results/cross_domain'
) -> Dict[str, Any]:
    """
    Run the full cross-domain analysis.
    """
    start_time = datetime.now()

    logger.info("=" * 70)
    logger.info("PRISM CROSS-DOMAIN OBSERVATORY ANALYSIS")
    logger.info("=" * 70)
    logger.info(f"Started: {start_time}")
    logger.info(f"Windows: {windows}")
    logger.info(f"States: {n_states}")
    logger.info(f"Step: {step}")

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Connect to database
    conn = get_connection(Path(db_path) if db_path else None)
    
    # Load data
    logger.info("\n" + "=" * 70)
    logger.info("LOADING DATA")
    logger.info("=" * 70)
    
    finance_df = load_finance_data(conn)
    logger.info(f"Finance: {len(finance_df)} rows, {len(finance_df.columns)} indicators")
    
    climate_df = load_climate_data(conn)
    logger.info(f"Climate: {len(climate_df)} rows, {len(climate_df.columns)} indicators")
    
    epi_df = load_epi_data(conn)
    logger.info(f"Epidemiology: {len(epi_df)} rows, {len(epi_df.columns)} indicators")
    
    conn.close()
    
    # Extract structure for each domain
    logger.info("\n" + "=" * 70)
    logger.info("EXTRACTING DOMAIN STRUCTURE")
    logger.info("=" * 70)
    
    domain_results = {}
    
    if len(finance_df) > 0:
        domain_results['finance'] = extract_domain_structure(
            finance_df, 'finance', windows=windows, n_states=n_states, step=step
        )
    
    if len(climate_df) > 0:
        domain_results['climate'] = extract_domain_structure(
            climate_df, 'climate', windows=windows, n_states=n_states, step=step
        )
    
    if len(epi_df) > 0:
        domain_results['epi'] = extract_domain_structure(
            epi_df, 'epi', windows=windows, n_states=n_states, step=step
        )
    
    # Cross-domain analysis
    logger.info("\n" + "=" * 70)
    logger.info("CROSS-DOMAIN ANALYSIS")
    logger.info("=" * 70)
    
    cross_domain_results = {}
    
    for window in windows:
        for n in n_states:
            key = f'w{window}_s{n}'
            logger.info(f"Analyzing window={window}, states={n}")
            
            aligned = align_domain_states(domain_results, window=window, n_states=n)
            
            if len(aligned) > 0:
                cross_domain_results[key] = {
                    'window': window,
                    'n_states': n,
                    'correlation': compute_state_correlation(aligned),
                    'multi_domain_events': find_cross_domain_events(aligned),
                }
    
    # Historical event analysis
    logger.info("\n" + "=" * 70)
    logger.info("HISTORICAL EVENT ANALYSIS")
    logger.info("=" * 70)
    
    event_analysis = analyze_event_geometry(
        domain_results, HISTORICAL_EVENTS, window=126, n_states=4
    )
    
    # Universal structure comparison
    logger.info("\n" + "=" * 70)
    logger.info("UNIVERSAL STRUCTURE COMPARISON")
    logger.info("=" * 70)
    
    universal_comparison = compare_geometry_distributions(domain_results, window=126)
    
    # Compile results
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    full_results = {
        'metadata': {
            'started': str(start_time),
            'completed': str(end_time),
            'duration_seconds': duration,
            'duration_human': f"{duration/60:.1f} minutes",
            'parameters': {
                'windows': windows,
                'n_states': n_states,
                'step': step,
            }
        },
        'domain_results': domain_results,
        'cross_domain': cross_domain_results,
        'historical_events': event_analysis,
        'universal_structure': universal_comparison,
    }
    
    # Save results
    output_file = Path(output_dir) / f'cross_domain_analysis_{start_time.strftime("%Y%m%d_%H%M%S")}.json'
    
    with open(output_file, 'w') as f:
        json.dump(full_results, f, indent=2, default=str)
    
    logger.info(f"\nResults saved to: {output_file}")
    
    # Print summary
    print_summary(full_results)
    
    return full_results


def print_summary(results: Dict[str, Any]):
    """Print human-readable summary."""
    print("\n" + "=" * 70)
    print("CROSS-DOMAIN ANALYSIS SUMMARY")
    print("=" * 70)
    
    meta = results['metadata']
    print(f"\nDuration: {meta['duration_human']}")
    
    # Domain summaries
    print("\n" + "-" * 50)
    print("DOMAIN SUMMARIES")
    print("-" * 50)
    
    for domain, data in results['domain_results'].items():
        print(f"\n{domain.upper()}:")
        print(f"  Indicators: {data['n_indicators']}")
        print(f"  Rows: {data['n_rows']}")
        print(f"  Date range: {data['date_range'][0]} to {data['date_range'][1]}")
        
        if 126 in data['windows'] and 4 in data['windows'][126]['states']:
            states = data['windows'][126]['states'][4]
            print(f"  Transitions (w=126, s=4): {states['n_transitions']}")
    
    # Cross-domain correlations
    print("\n" + "-" * 50)
    print("CROSS-DOMAIN STATE CORRELATIONS (w=126, s=4)")
    print("-" * 50)
    
    if 'w126_s4' in results['cross_domain']:
        cd = results['cross_domain']['w126_s4']
        
        if 'correlation' in cd and 'state_correlations' in cd['correlation']:
            for pair, stats in cd['correlation']['state_correlations'].items():
                sig = "***" if stats['significant'] else ""
                print(f"  {pair}: r={stats['correlation']:.3f} (p={stats['p_value']:.4f}) {sig}")
        
        if 'multi_domain_events' in cd:
            events = cd['multi_domain_events']
            print(f"\n  Multi-domain transitions: {len(events)}")
            if events:
                print("  Recent multi-domain events:")
                for e in events[-5:]:
                    print(f"    {e['date']}: {e['domains_changed']}")
    
    # Historical events
    print("\n" + "-" * 50)
    print("GEOMETRY AT HISTORICAL EVENTS")
    print("-" * 50)
    
    key_events = ['lehman', 'covid_crash', 'el_nino_2015', 'h1n1_pandemic']
    
    for event in key_events:
        if event in results['historical_events']:
            data = results['historical_events'][event]
            print(f"\n  {event} ({data['date']}):")
            for domain, state_info in data['states'].items():
                print(f"    {domain}: State {state_info['state']}")
    
    # Universal structure
    print("\n" + "-" * 50)
    print("UNIVERSAL STRUCTURE COMPARISON")
    print("-" * 50)
    
    for metric, data in results['universal_structure'].items():
        if 'domain_stats' in data:
            print(f"\n  {metric.split('__')[1]}:")
            for domain, stats in data['domain_stats'].items():
                print(f"    {domain}: mean={stats['mean']:.3f}, std={stats['std']:.3f}")
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PRISM Cross-Domain Analysis")
    parser.add_argument("--quick", action="store_true", help="Quick test run")
    parser.add_argument("--db", default=str(get_db_path()), help="Database path")
    parser.add_argument("--output", default="results/cross_domain", help="Output directory")
    
    args = parser.parse_args()
    
    if args.quick:
        # Quick test: small windows, fewer states
        windows = [63]
        n_states = [3, 4]
        step = 10
    else:
        # Full analysis: multiple windows and state counts
        windows = [21, 42, 63, 126, 252]  # 1mo, 2mo, 3mo, 6mo, 1yr
        n_states = [2, 3, 4, 5, 6, 7, 8]
        step = 5
    
    results = run_full_analysis(
        db_path=args.db,
        windows=windows,
        n_states=n_states,
        step=step,
        output_dir=args.output
    )
