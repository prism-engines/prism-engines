# VCF-Research: Adapter for Existing VCF Output
# Connects your current magnitude-based VCF to influence analysis

"""
This adapter helps you understand:
1. What magnitude tells you vs. what it doesn't
2. How to add influence analysis to your existing pipeline
3. Compare magnitude-based regime detection with influence-based detection
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple
import warnings
warnings.filterwarnings('ignore')


class VCFMagnitudeAnalyzer:
    """
    Analyze existing VCF magnitude output
    """
    
    def __init__(self, vcf_results_df: pd.DataFrame):
        """
        vcf_results_df: DataFrame with columns ['vcf_magnitude', 'theta_deg', 'phi_deg']
                        and datetime index
        """
        self.vcf_results = vcf_results_df
        
    def detect_regimes_by_magnitude(self, 
                                    high_threshold: float = 2.0,
                                    low_threshold: float = 0.5) -> pd.DataFrame:
        """
        Classify regimes based on magnitude
        
        High magnitude = High coherence/volatility = CRISIS or strong trend
        Low magnitude = Low coherence = CONSOLIDATION or stability
        
        But this doesn't tell you WHAT's causing it!
        """
        magnitude = self.vcf_results['vcf_magnitude']
        
        regimes = pd.DataFrame(index=self.vcf_results.index)
        regimes['magnitude'] = magnitude
        
        # Simple regime classification
        regimes['regime'] = 'NORMAL'
        regimes.loc[magnitude > high_threshold, 'regime'] = 'HIGH_COHERENCE'
        regimes.loc[magnitude < low_threshold, 'regime'] = 'LOW_COHERENCE'
        
        # Volatility
        regimes['magnitude_velocity'] = magnitude.diff().abs()
        regimes['magnitude_volatility'] = magnitude.rolling(20).std()
        
        return regimes
    
    def identify_crisis_periods(self, threshold: float = 3.0) -> pd.DataFrame:
        """
        Find periods of extreme magnitude
        """
        magnitude = self.vcf_results['vcf_magnitude']
        
        crisis_periods = []
        in_crisis = False
        crisis_start = None
        
        for date, mag in magnitude.items():
            if mag > threshold and not in_crisis:
                # Start of crisis
                in_crisis = True
                crisis_start = date
                max_mag = mag
                
            elif mag > threshold and in_crisis:
                # Continue crisis, track max
                max_mag = max(max_mag, mag)
                
            elif mag <= threshold and in_crisis:
                # End of crisis
                crisis_periods.append({
                    'start': crisis_start,
                    'end': date,
                    'max_magnitude': max_mag,
                    'duration_days': (date - crisis_start).days
                })
                in_crisis = False
        
        return pd.DataFrame(crisis_periods)
    
    def magnitude_evolution_by_period(self) -> Dict:
        """
        Break down magnitude statistics by major historical periods
        """
        periods = {
            'Pre-WWII (1913-1938)': ('1913', '1938'),
            'WWII Era (1939-1945)': ('1939', '1945'),
            'Post-War (1946-1960)': ('1946', '1960'),
            'Stable 60s-70s (1961-1979)': ('1961', '1979'),
            'Volatility 80s (1980-1989)': ('1980', '1989'),
            '90s Boom (1990-1999)': ('1990', '1999'),
            'Dot-com + 9/11 (2000-2002)': ('2000', '2002'),
            'Housing Bubble (2003-2007)': ('2003', '2007'),
            'Financial Crisis (2008-2009)': ('2008', '2009'),
            'Recovery (2010-2019)': ('2010', '2019'),
            'COVID + Current (2020-2025)': ('2020', '2025')
        }
        
        results = {}
        for period_name, (start, end) in periods.items():
            try:
                period_data = self.vcf_results.loc[start:end, 'vcf_magnitude']
                results[period_name] = {
                    'mean': period_data.mean(),
                    'std': period_data.std(),
                    'max': period_data.max(),
                    'min': period_data.min(),
                    'n_obs': len(period_data)
                }
            except:
                results[period_name] = {'error': 'No data for this period'}
        
        return results
    
    def compare_magnitude_vs_velocity(self, window: int = 20) -> pd.DataFrame:
        """
        Magnitude tells you the level of coherence
        Velocity tells you the rate of change
        
        Both are important for regime detection
        """
        magnitude = self.vcf_results['vcf_magnitude']
        
        comparison = pd.DataFrame(index=self.vcf_results.index)
        comparison['magnitude'] = magnitude
        comparison['velocity'] = magnitude.diff().abs()
        comparison['acceleration'] = comparison['velocity'].diff().abs()
        comparison['rolling_volatility'] = magnitude.rolling(window).std()
        
        # Classify dynamics
        high_mag = magnitude > magnitude.quantile(0.75)
        high_vel = comparison['velocity'] > comparison['velocity'].quantile(0.75)
        
        comparison['regime_type'] = 'Stable'
        comparison.loc[high_mag & ~high_vel, 'regime_type'] = 'High_Coherence_Stable'
        comparison.loc[~high_mag & high_vel, 'regime_type'] = 'Low_Coherence_Changing'
        comparison.loc[high_mag & high_vel, 'regime_type'] = 'Crisis_Rapid_Change'
        
        return comparison


class VCFInfluenceConnector:
    """
    Connect VCF magnitude analysis with influence analysis
    
    This shows the gap: magnitude tells you "something is happening"
    but influence tells you "WHAT is causing it"
    """
    
    def __init__(self, vcf_results_df: pd.DataFrame, indicator_panel_df: pd.DataFrame):
        """
        vcf_results_df: Your VCF magnitude output
        indicator_panel_df: The underlying indicators (what you need for influence!)
        """
        self.vcf_results = vcf_results_df
        self.indicator_panel = indicator_panel_df
        
        # Align dates
        common_dates = self.vcf_results.index.intersection(self.indicator_panel.index)
        self.vcf_results = self.vcf_results.loc[common_dates]
        self.indicator_panel = self.indicator_panel.loc[common_dates]
        
    def analyze_high_magnitude_drivers(self, n_periods: int = 10) -> pd.DataFrame:
        """
        For the N highest magnitude periods, identify which indicators were most active
        
        This is the key insight: magnitude shows "something happening",
        influence shows "what's driving it"
        """
        from influence_analysis import InfluenceRanking
        
        # Get top N magnitude periods
        top_periods = self.vcf_results['vcf_magnitude'].nlargest(n_periods)
        
        # For each period, compute influence
        ranker = InfluenceRanking(self.indicator_panel)
        influence_scores = ranker.composite_influence_score(window=12)
        
        results = []
        for date, magnitude in top_periods.items():
            if date in influence_scores.index:
                # Get top 5 influencers at this date
                scores = influence_scores.loc[date].sort_values(ascending=False)
                
                results.append({
                    'date': date,
                    'magnitude': magnitude,
                    'top_1_driver': scores.index[0],
                    'top_1_score': scores.iloc[0],
                    'top_2_driver': scores.index[1],
                    'top_2_score': scores.iloc[1],
                    'top_3_driver': scores.index[2],
                    'top_3_score': scores.iloc[2],
                    'concentration': scores.iloc[0] / scores.iloc[:3].sum()
                })
        
        return pd.DataFrame(results)
    
    def magnitude_vs_influence_correlation(self) -> Dict:
        """
        Do certain indicators correlate with high VCF magnitude?
        
        This helps identify which indicators are "magnitude drivers"
        """
        from scipy.stats import pearsonr
        
        magnitude = self.vcf_results['vcf_magnitude']
        
        correlations = {}
        for col in self.indicator_panel.columns:
            indicator = self.indicator_panel[col]
            
            # Correlation with magnitude
            corr, p_value = pearsonr(magnitude, indicator)
            
            # Correlation with magnitude CHANGES (volatility)
            mag_changes = magnitude.diff().abs()
            ind_changes = indicator.diff().abs()
            vol_corr, vol_p = pearsonr(mag_changes.dropna(), ind_changes.dropna())
            
            correlations[col] = {
                'level_correlation': corr,
                'level_p_value': p_value,
                'volatility_correlation': vol_corr,
                'volatility_p_value': vol_p
            }
        
        # Sort by absolute volatility correlation
        sorted_cors = sorted(correlations.items(), 
                           key=lambda x: abs(x[1]['volatility_correlation']), 
                           reverse=True)
        
        return dict(sorted_cors)
    
    def regime_stability_analysis(self) -> pd.DataFrame:
        """
        Compare magnitude-based regimes with influence-based regimes
        
        Do they identify the same regime changes?
        """
        from influence_analysis import TemporalInfluenceTracking
        
        # Magnitude-based regimes
        magnitude = self.vcf_results['vcf_magnitude']
        mag_regimes = pd.cut(magnitude, bins=3, labels=['Low', 'Medium', 'High'])
        mag_changes = (mag_regimes != mag_regimes.shift(1)).astype(int)
        
        # Influence-based regimes
        tracker = TemporalInfluenceTracking(self.indicator_panel)
        influence_regimes = tracker.detect_influence_regime_changes()
        
        # Align and compare
        common_dates = mag_changes.index.intersection(influence_regimes['date'])
        
        comparison = pd.DataFrame({
            'date': common_dates,
            'magnitude_regime_change': mag_changes.loc[common_dates].values,
            'influence_regime_change': influence_regimes.set_index('date').loc[common_dates, 'regime_change'].values
        })
        
        comparison['both_agree'] = (comparison['magnitude_regime_change'] == 
                                   comparison['influence_regime_change'])
        
        return comparison


def visualize_vcf_magnitude_history(vcf_results: pd.DataFrame, output_file: str = None):
    """
    Comprehensive visualization of VCF magnitude over time
    """
    fig, axes = plt.subplots(4, 1, figsize=(16, 12))
    
    magnitude = vcf_results['vcf_magnitude']
    
    # 1. Full time series
    ax = axes[0]
    ax.plot(magnitude.index, magnitude.values, linewidth=0.5, color='navy', alpha=0.7)
    ax.axhline(y=2.0, color='red', linestyle='--', alpha=0.5, label='High threshold')
    ax.axhline(y=0.5, color='green', linestyle='--', alpha=0.5, label='Low threshold')
    ax.set_title('VCF Magnitude: Full History (1913-2025)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Magnitude')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Annotate major events
    events = {
        '1929-10': 'Great Depression',
        '1987-10': 'Black Monday',
        '1998-08': 'Asian Crisis',
        '2008-09': 'Lehman',
        '2020-03': 'COVID Crash'
    }
    for date_str, event in events.items():
        if date_str in magnitude.index.strftime('%Y-%m'):
            idx = magnitude.index.get_loc(date_str, method='nearest')
            ax.annotate(event, xy=(magnitude.index[idx], magnitude.iloc[idx]),
                       xytext=(10, 10), textcoords='offset points',
                       fontsize=8, ha='left',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                       arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    # 2. Rolling statistics
    ax = axes[1]
    rolling_mean = magnitude.rolling(252).mean()  # ~1 year
    rolling_std = magnitude.rolling(252).std()
    ax.plot(rolling_mean.index, rolling_mean.values, label='1-year mean', linewidth=2)
    ax.fill_between(rolling_mean.index, 
                    rolling_mean.values - rolling_std.values,
                    rolling_mean.values + rolling_std.values,
                    alpha=0.3, label='±1 std')
    ax.set_title('VCF Magnitude: Rolling Statistics', fontsize=12, fontweight='bold')
    ax.set_ylabel('Magnitude')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Distribution by decade
    ax = axes[2]
    decades = (magnitude.index.year // 10) * 10
    decade_data = [magnitude[decades == d].values for d in sorted(set(decades))]
    ax.boxplot(decade_data, labels=[f"{d}s" for d in sorted(set(decades))])
    ax.set_title('VCF Magnitude Distribution by Decade', fontsize=12, fontweight='bold')
    ax.set_ylabel('Magnitude')
    ax.set_xlabel('Decade')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 4. Recent detail (2020-present)
    ax = axes[3]
    recent = magnitude.loc['2020':]
    ax.plot(recent.index, recent.values, linewidth=1.5, color='darkred')
    ax.axhline(y=recent.mean(), color='blue', linestyle='--', alpha=0.7, 
              label=f'Mean: {recent.mean():.2f}')
    ax.set_title('VCF Magnitude: Recent Period (2020-2025)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Magnitude')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {output_file}")
    
    return fig


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    """
    Demonstrate the gap between magnitude and influence analysis
    """
    
    # Load VCF results
    vcf_results = pd.read_csv('/mnt/user-data/uploads/vcf_results_full_ZSCORE.csv',
                              parse_dates=['date'], index_col='date')
    
    print("=" * 70)
    print("VCF MAGNITUDE ANALYSIS")
    print("=" * 70)
    
    # Analyze magnitude
    analyzer = VCFMagnitudeAnalyzer(vcf_results)
    
    # Regime detection by magnitude
    print("\n1. MAGNITUDE-BASED REGIME DETECTION")
    print("-" * 70)
    regimes = analyzer.detect_regimes_by_magnitude(high_threshold=2.0, low_threshold=0.5)
    regime_counts = regimes['regime'].value_counts()
    print("Regime distribution:")
    print(regime_counts)
    
    # Crisis periods
    print("\n2. CRISIS PERIODS (Magnitude > 3.0)")
    print("-" * 70)
    crises = analyzer.identify_crisis_periods(threshold=3.0)
    print(f"Found {len(crises)} crisis periods")
    print(crises.to_string(index=False))
    
    # Historical evolution
    print("\n3. MAGNITUDE BY HISTORICAL PERIOD")
    print("-" * 70)
    evolution = analyzer.magnitude_evolution_by_period()
    for period, stats in evolution.items():
        if 'error' not in stats:
            print(f"\n{period}:")
            print(f"  Mean: {stats['mean']:.3f}, Max: {stats['max']:.3f}")
    
    # Dynamics
    print("\n4. MAGNITUDE vs VELOCITY ANALYSIS")
    print("-" * 70)
    dynamics = analyzer.compare_magnitude_vs_velocity()
    print("\nRegime type distribution:")
    print(dynamics['regime_type'].value_counts())
    
    print("\n" + "=" * 70)
    print("KEY INSIGHT:")
    print("-" * 70)
    print("Magnitude tells you THAT something is happening.")
    print("Influence analysis tells you WHAT is causing it.")
    print("\nTo add influence analysis, you need the underlying indicator panel.")
    print("=" * 70)
    
    # Create visualization
    visualize_vcf_magnitude_history(vcf_results, 
                                    output_file='/mnt/user-data/outputs/vcf_magnitude_history.png')
