#!/usr/bin/env python3
"""
PRISM Unified Calibration Config
=================================

Combines results from all calibration steps into one master config.

Fixes the over-aggressive redundancy filtering and creates a
balanced, usable configuration.

Usage:
    python calibration_unified.py
    
Output:
    calibration/master_config.json - The final tuned configuration
"""

import sys
from pathlib import Path

if __name__ == "__main__":
    _script_dir = Path(__file__).parent.parent
    import os
    os.chdir(_script_dir)
    if str(_script_dir) not in sys.path:
        sys.path.insert(0, str(_script_dir))

import json
import numpy as np
import pandas as pd
from collections import defaultdict

from output_config import OUTPUT_DIR, DATA_DIR

CALIBRATION_DIR = OUTPUT_DIR / "calibration"


def load_calibration_files():
    """Load all calibration outputs."""
    
    files = {}
    
    # Lens weights
    lens_file = CALIBRATION_DIR / "lens_weights.json"
    if lens_file.exists():
        with open(lens_file) as f:
            files['lens_weights'] = json.load(f)
    
    # Lens scores
    lens_scores = CALIBRATION_DIR / "lens_scores.csv"
    if lens_scores.exists():
        files['lens_scores'] = pd.read_csv(lens_scores, index_col=0)
    
    # Indicator tiers
    tiers_file = CALIBRATION_DIR / "indicator_tiers.json"
    if tiers_file.exists():
        with open(tiers_file) as f:
            files['indicator_tiers'] = json.load(f)
    
    # Indicator scores
    ind_scores = CALIBRATION_DIR / "indicator_scores.csv"
    if ind_scores.exists():
        files['indicator_scores'] = pd.read_csv(ind_scores, index_col=0)
    
    # Redundancy map
    redundancy = CALIBRATION_DIR / "redundancy_map.csv"
    if redundancy.exists():
        files['redundancy'] = pd.read_csv(redundancy)
    
    # Window config
    window_file = CALIBRATION_DIR / "optimal_windows.json"
    if window_file.exists():
        with open(window_file) as f:
            files['optimal_windows'] = json.load(f)
    
    # Consensus events
    events_file = CALIBRATION_DIR / "consensus_events.csv"
    if events_file.exists():
        files['events'] = pd.read_csv(events_file)
    
    return files


def build_master_config(files: dict) -> dict:
    """Build the master configuration from all calibration data."""
    
    config = {
        'version': '1.0',
        'generated': pd.Timestamp.now().isoformat(),
        'lenses': {},
        'indicators': {},
        'windows': {},
        'thresholds': {},
    }
    
    # === LENS CONFIGURATION ===
    if 'lens_weights' in files:
        weights = files['lens_weights']
        scores = files.get('lens_scores', pd.DataFrame())
        
        for lens, weight in weights.items():
            lens_info = {
                'weight': weight,
                'use': weight >= 0.5,  # Use if weight >= 0.5
                'tier': 1 if weight >= 1.0 else 2 if weight >= 0.7 else 3,
            }
            
            if not scores.empty and lens in scores.index:
                lens_info['hit_rate'] = float(scores.loc[lens, 'hit_rate'])
                lens_info['avg_lead_time'] = float(scores.loc[lens, 'avg_lead_time'])
            
            config['lenses'][lens] = lens_info
    
    # === INDICATOR CONFIGURATION ===
    if 'indicator_tiers' in files and 'indicator_scores' in files:
        tiers = files['indicator_tiers']
        scores = files['indicator_scores']
        windows = files.get('optimal_windows', {})
        redundancy = files.get('redundancy', pd.DataFrame())
        
        # Build redundancy lookup (keep best from each pair)
        redundant_to = {}
        if not redundancy.empty:
            for _, row in redundancy.iterrows():
                ind1, ind2 = row['indicator_1'], row['indicator_2']
                
                # Keep the one with higher score
                score1 = scores.loc[ind1, 'total_score'] if ind1 in scores.index else 0
                score2 = scores.loc[ind2, 'total_score'] if ind2 in scores.index else 0
                
                if score1 >= score2:
                    redundant_to[ind2] = ind1
                else:
                    redundant_to[ind1] = ind2
        
        for indicator, tier_info in tiers.items():
            tier = tier_info['tier']
            score = tier_info['score']
            
            # Get window info
            window_info = windows.get(indicator, {})
            optimal_window = window_info.get('optimal_window', 63)
            ind_type = window_info.get('type', 'medium')
            
            # Determine if we should use this indicator
            is_redundant = indicator in redundant_to
            
            # More lenient: use Tier 1, 2, and non-redundant Tier 3
            use_indicator = (
                (tier <= 2) or  # Always use Tier 1 and 2
                (tier == 3 and not is_redundant and score > 0.1)  # Tier 3 if not redundant
            )
            
            config['indicators'][indicator] = {
                'tier': tier,
                'score': score,
                'use': use_indicator,
                'optimal_window': optimal_window,
                'type': ind_type,
                'redundant_to': redundant_to.get(indicator),
            }
    
    # === WINDOW CONFIGURATION ===
    config['windows'] = {
        'default': 63,
        'by_type': {
            'fast': 10,    # Based on actual data
            'medium': 21,
            'slow': 126,
        },
        'multi_window': [21, 63, 126],  # For multi-resolution analysis
    }
    
    # === THRESHOLD CONFIGURATION ===
    if 'events' in files:
        events = files['events']
        if not events.empty:
            # Derive thresholds from actual events
            config['thresholds'] = {
                'consensus_warning': float(events['peak_consensus'].quantile(0.5)),
                'consensus_danger': float(events['peak_consensus'].quantile(0.75)),
                'rank_warning': 10,
                'rank_danger': 5,
            }
    else:
        config['thresholds'] = {
            'consensus_warning': 0.15,
            'consensus_danger': 0.25,
            'rank_warning': 10,
            'rank_danger': 5,
        }
    
    return config


def summarize_config(config: dict):
    """Print a summary of the configuration."""
    
    print("\n" + "=" * 70)
    print("📋 MASTER CALIBRATION CONFIGURATION")
    print("=" * 70)
    
    # Lenses
    print("\n🔬 LENS CONFIGURATION:")
    active_lenses = [l for l, info in config['lenses'].items() if info['use']]
    inactive_lenses = [l for l, info in config['lenses'].items() if not info['use']]
    
    print(f"\n   Active lenses ({len(active_lenses)}):")
    for lens in sorted(active_lenses, key=lambda x: config['lenses'][x]['weight'], reverse=True):
        info = config['lenses'][lens]
        print(f"      ✅ {lens:20} weight={info['weight']:.2f}")
    
    if inactive_lenses:
        print(f"\n   Inactive lenses ({len(inactive_lenses)}):")
        for lens in inactive_lenses:
            info = config['lenses'][lens]
            print(f"      ❌ {lens:20} weight={info['weight']:.2f}")
    
    # Indicators
    print("\n📊 INDICATOR CONFIGURATION:")
    
    tier_counts = defaultdict(int)
    use_counts = {'use': 0, 'skip': 0}
    
    for ind, info in config['indicators'].items():
        tier_counts[info['tier']] += 1
        use_counts['use' if info['use'] else 'skip'] += 1
    
    print(f"\n   By tier:")
    for tier in sorted(tier_counts.keys()):
        print(f"      Tier {tier}: {tier_counts[tier]} indicators")
    
    print(f"\n   Usage:")
    print(f"      ✅ Active:   {use_counts['use']} indicators")
    print(f"      ❌ Inactive: {use_counts['skip']} indicators")
    
    # Top indicators by tier
    print(f"\n   Top Tier 1 indicators:")
    tier_1 = [(ind, info) for ind, info in config['indicators'].items() 
              if info['tier'] == 1 and info['use']]
    tier_1_sorted = sorted(tier_1, key=lambda x: x[1]['score'], reverse=True)
    
    for ind, info in tier_1_sorted[:10]:
        window = info['optimal_window']
        print(f"      • {ind:25} (window={window}d, score={info['score']:.3f})")
    
    # Windows
    print("\n⏱️ WINDOW CONFIGURATION:")
    print(f"   Default: {config['windows']['default']} days")
    print(f"   By type:")
    for ind_type, window in config['windows']['by_type'].items():
        print(f"      {ind_type}: {window} days")
    print(f"   Multi-resolution: {config['windows']['multi_window']}")
    
    # Thresholds
    print("\n🚨 THRESHOLD CONFIGURATION:")
    for name, value in config['thresholds'].items():
        print(f"   {name}: {value}")
    
    print("\n" + "=" * 70)


def create_summary_report(config: dict):
    """Create a markdown summary report."""
    
    report = f"""# PRISM Calibration Report

Generated: {config['generated']}

## Lens Configuration

| Lens | Weight | Use | Tier |
|------|--------|-----|------|
"""
    
    for lens in sorted(config['lenses'].keys(), 
                       key=lambda x: config['lenses'][x]['weight'], reverse=True):
        info = config['lenses'][lens]
        use = "✅" if info['use'] else "❌"
        report += f"| {lens} | {info['weight']:.2f} | {use} | {info['tier']} |\n"
    
    report += """
## Indicator Summary

"""
    
    tier_1 = [(ind, info) for ind, info in config['indicators'].items() 
              if info['tier'] == 1 and info['use']]
    tier_1_sorted = sorted(tier_1, key=lambda x: x[1]['score'], reverse=True)
    
    report += "### Tier 1 (Core) Indicators\n\n"
    report += "| Indicator | Score | Window | Type |\n"
    report += "|-----------|-------|--------|------|\n"
    
    for ind, info in tier_1_sorted:
        report += f"| {ind} | {info['score']:.3f} | {info['optimal_window']}d | {info['type']} |\n"
    
    report += f"""
## Window Configuration

- **Default window**: {config['windows']['default']} days
- **Fast indicators**: {config['windows']['by_type']['fast']} days
- **Medium indicators**: {config['windows']['by_type']['medium']} days  
- **Slow indicators**: {config['windows']['by_type']['slow']} days

## Thresholds

- Warning (rank): {config['thresholds']['rank_warning']}
- Danger (rank): {config['thresholds']['rank_danger']}
- Consensus warning: {config['thresholds']['consensus_warning']:.3f}
- Consensus danger: {config['thresholds']['consensus_danger']:.3f}
"""
    
    report_path = CALIBRATION_DIR / "calibration_report.md"
    report_path.write_text(report)
    print(f"   ✅ {report_path}")


def main():
    print("=" * 70)
    print("🔧 PRISM UNIFIED CALIBRATION")
    print("=" * 70)
    
    print("\n📥 Loading calibration files...")
    files = load_calibration_files()
    
    print(f"   Found: {list(files.keys())}")
    
    print("\n🔨 Building master configuration...")
    config = build_master_config(files)
    
    # Save master config
    print("\n💾 Saving master configuration...")
    
    config_path = CALIBRATION_DIR / "master_config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2, default=str)
    print(f"   ✅ {config_path}")
    
    create_summary_report(config)
    
    summarize_config(config)
    
    # Count what we're using
    active_lenses = sum(1 for l in config['lenses'].values() if l['use'])
    active_indicators = sum(1 for i in config['indicators'].values() if i['use'])
    
    print("\n✅ UNIFIED CALIBRATION COMPLETE")
    print(f"\n   📊 Active lenses: {active_lenses}")
    print(f"   📈 Active indicators: {active_indicators}")
    print(f"\n   📁 Config saved to: {config_path}")
    
    return config


if __name__ == "__main__":
    main()
