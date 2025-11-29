# VCF Research: Influence Analysis Module
# Determines which indicators drive system dynamics at any given time

"""
This module answers the key questions:
1. Which indicators have the strongest influence RIGHT NOW?
2. What is the relative magnitude of each indicator's contribution?
3. How does influence change over time?
4. Which pairs/groups of indicators are most coherent?

This shifts focus from "what regime are we in?" to 
"what's actually driving the system?"
"""

import numpy as np
import pandas as pd
from scipy.stats import zscore
from scipy.signal import hilbert
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


class InfluenceRanking:
    """
    Rank indicators by their influence on system dynamics
    """
    
    def __init__(self, panel_df: pd.DataFrame):
        """
        panel_df: DataFrame with all 56 indicators (or however many)
        Each column is an indicator, rows are time points
        """
        self.panel = panel_df
        self.n_indicators = panel_df.shape[1]
        self.n_time = panel_df.shape[0]
        
    def magnitude_influence(self, window: int = 12) -> pd.DataFrame:
        """
        Which indicators have the largest magnitude changes?
        High magnitude = high influence potential
        
        Returns: DataFrame with rolling magnitude for each indicator
        """
        # Compute rolling magnitude (absolute rate of change)
        magnitudes = {}
        
        for col in self.panel.columns:
            series = self.panel[col].values
            
            # Rolling standard deviation (volatility)
            rolling_mag = pd.Series(series).rolling(window).std()
            
            # Normalize to make comparable across indicators
            rolling_mag = rolling_mag / (np.std(series) + 1e-10)
            
            magnitudes[col] = rolling_mag.values
        
        return pd.DataFrame(magnitudes, index=self.panel.index)
    
    def velocity_influence(self, window: int = 12) -> pd.DataFrame:
        """
        Which indicators are changing fastest?
        High velocity = driving dynamics
        
        Returns: DataFrame with rolling velocity for each indicator
        """
        velocities = {}
        
        for col in self.panel.columns:
            series = self.panel[col].values
            
            # First derivative (velocity)
            velocity = np.gradient(series)
            
            # Rolling magnitude of velocity
            rolling_vel = pd.Series(np.abs(velocity)).rolling(window).mean()
            
            # Normalize
            rolling_vel = rolling_vel / (np.std(velocity) + 1e-10)
            
            velocities[col] = rolling_vel.values
        
        return pd.DataFrame(velocities, index=self.panel.index)
    
    def coherence_leadership(self, window: int = 24) -> pd.DataFrame:
        """
        Which indicators lead the system?
        High coherence with others = high influence
        
        For each indicator, compute its average coherence with all others
        """
        from scipy.signal import coherence
        
        leadership_scores = {}
        
        for i, col in enumerate(self.panel.columns):
            signal = self.panel[col].values
            
            # Compute coherence with all other indicators
            coherences = []
            
            for j, other_col in enumerate(self.panel.columns):
                if i == j:
                    continue
                
                other_signal = self.panel[other_col].values
                
                # Rolling coherence
                rolling_coh = []
                for t in range(window, len(signal)):
                    local_sig = signal[t-window:t]
                    local_other = other_signal[t-window:t]
                    
                    f, Cxy = coherence(local_sig, local_other, fs=1.0, 
                                      nperseg=min(16, window//2))
                    rolling_coh.append(np.mean(Cxy))
                
                coherences.append(rolling_coh)
            
            # Average coherence with all others
            avg_coherence = np.mean(coherences, axis=0)
            
            # Pad to match original length
            leadership_scores[col] = np.concatenate([
                np.full(window, np.nan),
                avg_coherence
            ])
        
        return pd.DataFrame(leadership_scores, index=self.panel.index)
    
    def granger_causality_influence(self, max_lag: int = 6) -> pd.DataFrame:
        """
        Which indicators Granger-cause others?
        Strong causality = high influence
        
        Returns: Matrix where element [i,j] = influence of i on j
        """
        from statsmodels.tsa.stattools import grangercausalitytests
        
        influence_matrix = np.zeros((self.n_indicators, self.n_indicators))
        
        for i, col_i in enumerate(self.panel.columns):
            for j, col_j in enumerate(self.panel.columns):
                if i == j:
                    continue
                
                try:
                    # Prepare data for Granger test
                    data = self.panel[[col_j, col_i]].dropna()
                    
                    if len(data) < max_lag + 10:
                        continue
                    
                    # Granger causality test
                    result = grangercausalitytests(data, max_lag, verbose=False)
                    
                    # Extract p-values and take minimum (strongest causality)
                    p_values = [result[lag][0]['ssr_ftest'][1] 
                               for lag in range(1, max_lag + 1)]
                    min_p = np.min(p_values)
                    
                    # Convert to influence score (lower p-value = higher influence)
                    influence_matrix[i, j] = 1.0 - min_p
                    
                except:
                    influence_matrix[i, j] = 0.0
        
        return pd.DataFrame(
            influence_matrix,
            index=self.panel.columns,
            columns=self.panel.columns
        )
    
    def composite_influence_score(self, window: int = 12) -> pd.DataFrame:
        """
        Combine multiple influence measures into single score
        
        Returns: DataFrame with time-varying influence score for each indicator
        """
        # Get individual influence measures
        mag_influence = self.magnitude_influence(window)
        vel_influence = self.velocity_influence(window)
        coh_leadership = self.coherence_leadership(window)
        
        # Normalize each to [0, 1]
        mag_norm = (mag_influence - mag_influence.min()) / (mag_influence.max() - mag_influence.min() + 1e-10)
        vel_norm = (vel_influence - vel_influence.min()) / (vel_influence.max() - vel_influence.min() + 1e-10)
        coh_norm = (coh_leadership - coh_leadership.min()) / (coh_leadership.max() - coh_leadership.min() + 1e-10)
        
        # Weighted average (you can adjust weights)
        composite = 0.4 * mag_norm + 0.3 * vel_norm + 0.3 * coh_norm
        
        return composite
    
    def top_influencers(self, date: pd.Timestamp = None, n_top: int = 10) -> pd.DataFrame:
        """
        Get the top N most influential indicators at a specific date
        If date is None, uses the most recent date
        """
        composite_scores = self.composite_influence_score()
        
        if date is None:
            date = self.panel.index[-1]
        
        # Get scores at this date
        scores_at_date = composite_scores.loc[date]
        
        # Sort and get top N
        top_indicators = scores_at_date.nlargest(n_top)
        
        return pd.DataFrame({
            'indicator': top_indicators.index,
            'influence_score': top_indicators.values,
            'rank': range(1, n_top + 1)
        })


class PairwiseCoherence:
    """
    Analyze which pairs/groups of indicators move together
    """
    
    def __init__(self, panel_df: pd.DataFrame):
        self.panel = panel_df
        
    def coherence_matrix(self, window: int = 24) -> pd.DataFrame:
        """
        Compute pairwise coherence between all indicators
        
        Returns: n_indicators x n_indicators matrix of coherence scores
        """
        from scipy.signal import coherence
        
        n = len(self.panel.columns)
        coh_matrix = np.zeros((n, n))
        
        for i, col_i in enumerate(self.panel.columns):
            for j, col_j in enumerate(self.panel.columns):
                if i == j:
                    coh_matrix[i, j] = 1.0
                    continue
                
                # Compute coherence
                signal_i = self.panel[col_i].values
                signal_j = self.panel[col_j].values
                
                # Use recent window
                recent_i = signal_i[-window:]
                recent_j = signal_j[-window:]
                
                f, Cxy = coherence(recent_i, recent_j, fs=1.0, 
                                  nperseg=min(16, window//2))
                
                coh_matrix[i, j] = np.mean(Cxy)
        
        return pd.DataFrame(
            coh_matrix,
            index=self.panel.columns,
            columns=self.panel.columns
        )
    
    def coherent_clusters(self, threshold: float = 0.7) -> List[List[str]]:
        """
        Find clusters of highly coherent indicators
        
        Returns: List of clusters (each cluster is a list of indicator names)
        """
        coh_matrix = self.coherence_matrix()
        
        # Simple clustering: indicators are in same cluster if coherence > threshold
        from scipy.cluster.hierarchy import linkage, fcluster
        from scipy.spatial.distance import squareform
        
        # Convert coherence to distance
        distance = 1.0 - coh_matrix.values
        distance = np.clip(distance, 0, 2)  # Ensure valid distances
        
        # Hierarchical clustering
        condensed_dist = squareform(distance)
        linkage_matrix = linkage(condensed_dist, method='average')
        
        # Form clusters
        cluster_labels = fcluster(linkage_matrix, t=1-threshold, criterion='distance')
        
        # Group indicators by cluster
        clusters = {}
        for i, label in enumerate(cluster_labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(self.panel.columns[i])
        
        return list(clusters.values())
    
    def most_coherent_pairs(self, n_pairs: int = 10) -> pd.DataFrame:
        """
        Find the N most coherent indicator pairs
        """
        coh_matrix = self.coherence_matrix()
        
        # Extract upper triangle (avoid duplicates)
        pairs = []
        n = len(coh_matrix)
        
        for i in range(n):
            for j in range(i+1, n):
                pairs.append({
                    'indicator_1': coh_matrix.index[i],
                    'indicator_2': coh_matrix.columns[j],
                    'coherence': coh_matrix.iloc[i, j]
                })
        
        # Sort by coherence
        pairs_df = pd.DataFrame(pairs)
        pairs_df = pairs_df.sort_values('coherence', ascending=False)
        
        return pairs_df.head(n_pairs)


class TemporalInfluenceTracking:
    """
    Track how influence changes over time
    Detect regime shifts based on changes in influence patterns
    """
    
    def __init__(self, panel_df: pd.DataFrame):
        self.panel = panel_df
        self.influence_ranker = InfluenceRanking(panel_df)
        
    def influence_evolution(self, window: int = 12) -> pd.DataFrame:
        """
        Track top influencers over time
        
        Returns: DataFrame showing which indicators were most influential at each time
        """
        composite_scores = self.influence_ranker.composite_influence_score(window)
        
        # For each time point, get top 5 influencers
        evolution = []
        
        for date in composite_scores.index[window:]:
            scores = composite_scores.loc[date].sort_values(ascending=False)
            
            evolution.append({
                'date': date,
                'top_1': scores.index[0],
                'top_1_score': scores.iloc[0],
                'top_2': scores.index[1],
                'top_2_score': scores.iloc[1],
                'top_3': scores.index[2],
                'top_3_score': scores.iloc[2],
                'top_1_dominance': scores.iloc[0] / (scores.iloc[1] + 1e-10)
            })
        
        return pd.DataFrame(evolution)
    
    def detect_influence_regime_changes(self, threshold: float = 0.5) -> pd.DataFrame:
        """
        Detect when the dominant influencers change significantly
        (Alternative to regime detection based on coherence)
        """
        evolution = self.influence_evolution()
        
        # Detect changes in top influencer
        top_changes = evolution['top_1'] != evolution['top_1'].shift(1)
        
        # Detect large changes in dominance
        dominance_changes = np.abs(
            evolution['top_1_dominance'] - evolution['top_1_dominance'].shift(1)
        ) > threshold
        
        regime_changes = top_changes | dominance_changes
        
        return pd.DataFrame({
            'date': evolution['date'],
            'regime_change': regime_changes,
            'top_influencer': evolution['top_1'],
            'dominance': evolution['top_1_dominance']
        })
    
    def influence_persistence(self, indicator: str, window: int = 24) -> pd.Series:
        """
        How consistently does an indicator maintain high influence?
        
        Returns: Time series of rolling influence persistence score
        """
        composite_scores = self.influence_ranker.composite_influence_score()
        
        if indicator not in composite_scores.columns:
            raise ValueError(f"Indicator {indicator} not found")
        
        indicator_score = composite_scores[indicator]
        
        # Rolling percentile rank
        persistence = indicator_score.rolling(window).apply(
            lambda x: np.sum(x > np.median(x)) / len(x)
        )
        
        return persistence


class ComparativeInfluenceAnalysis:
    """
    Compare influence patterns across different engines or time periods
    Perfect for comparing your engine vs Gemini's
    """
    
    @staticmethod
    def compare_rankings(ranking_a: pd.DataFrame, ranking_b: pd.DataFrame) -> Dict:
        """
        Compare two sets of influence rankings
        
        ranking_a, ranking_b: DataFrames with 'indicator' and 'influence_score' columns
        
        Returns: Dictionary with comparison metrics
        """
        # Rank correlation
        from scipy.stats import spearmanr, kendalltau
        
        # Merge on indicator
        merged = ranking_a.merge(ranking_b, on='indicator', suffixes=('_a', '_b'))
        
        spearman_corr, _ = spearmanr(
            merged['influence_score_a'],
            merged['influence_score_b']
        )
        
        kendall_corr, _ = kendalltau(
            merged['influence_score_a'],
            merged['influence_score_b']
        )
        
        # Top K overlap
        top_5_a = set(ranking_a.nsmallest(5, 'rank')['indicator'])
        top_5_b = set(ranking_b.nsmallest(5, 'rank')['indicator'])
        top_5_overlap = len(top_5_a & top_5_b) / 5.0
        
        # Largest disagreements
        merged['score_diff'] = np.abs(
            merged['influence_score_a'] - merged['influence_score_b']
        )
        disagreements = merged.nlargest(5, 'score_diff')[
            ['indicator', 'influence_score_a', 'influence_score_b', 'score_diff']
        ]
        
        return {
            'spearman_correlation': spearman_corr,
            'kendall_correlation': kendall_corr,
            'top_5_overlap': top_5_overlap,
            'largest_disagreements': disagreements
        }
    
    @staticmethod
    def visualize_influence_comparison(ranking_a: pd.DataFrame, ranking_b: pd.DataFrame,
                                      labels: Tuple[str, str] = ('Engine A', 'Engine B')):
        """
        Generate comparison visualization data
        
        Returns: DataFrame suitable for plotting
        """
        merged = ranking_a.merge(ranking_b, on='indicator', suffixes=('_a', '_b'))
        
        comparison = pd.DataFrame({
            'indicator': merged['indicator'],
            f'{labels[0]}_score': merged['influence_score_a'],
            f'{labels[1]}_score': merged['influence_score_b'],
            'difference': merged['influence_score_a'] - merged['influence_score_b'],
            'agreement': np.abs(merged['influence_score_a'] - merged['influence_score_b']) < 0.1
        })
        
        return comparison.sort_values('difference', key=abs, ascending=False)


# ============================================================================
# INTEGRATED INFLUENCE ANALYSIS ENGINE
# ============================================================================

class InfluenceAnalysisEngine:
    """
    Unified interface for all influence analysis
    """
    
    def __init__(self, panel_df: pd.DataFrame):
        self.panel = panel_df
        self.ranker = InfluenceRanking(panel_df)
        self.pairwise = PairwiseCoherence(panel_df)
        self.temporal = TemporalInfluenceTracking(panel_df)
        
    def full_influence_report(self, date: pd.Timestamp = None, 
                             n_top: int = 10) -> Dict:
        """
        Complete influence analysis for a given date
        """
        if date is None:
            date = self.panel.index[-1]
        
        return {
            'top_influencers': self.ranker.top_influencers(date, n_top),
            'coherent_pairs': self.pairwise.most_coherent_pairs(n_top),
            'coherent_clusters': self.pairwise.coherent_clusters(),
            'granger_influence': self.ranker.granger_causality_influence(),
            'influence_evolution': self.temporal.influence_evolution().tail(12)
        }
    
    def detect_what_drives_dynamics(self, window: int = 12) -> pd.DataFrame:
        """
        THE KEY QUESTION: What's driving the system right now?
        
        Returns: Time series showing dominant drivers and their relative strength
        """
        composite_scores = self.ranker.composite_influence_score(window)
        
        results = []
        for date in composite_scores.index[window:]:
            scores = composite_scores.loc[date].sort_values(ascending=False)
            
            # Top 3 drivers
            top_3 = scores.head(3)
            total_top_3 = top_3.sum()
            
            results.append({
                'date': date,
                'primary_driver': top_3.index[0],
                'primary_magnitude': top_3.iloc[0],
                'secondary_driver': top_3.index[1],
                'secondary_magnitude': top_3.iloc[1],
                'tertiary_driver': top_3.index[2],
                'tertiary_magnitude': top_3.iloc[2],
                'concentration': top_3.iloc[0] / total_top_3,  # How concentrated is influence?
                'top_3_total_influence': total_top_3
            })
        
        return pd.DataFrame(results)


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    """
    Example: Analyze which indicators drive dynamics
    """
    
    # Create sample data (replace with your 56 indicators)
    dates = pd.date_range('2010-01-01', periods=200, freq='M')
    panel = pd.DataFrame({
        'CPI': np.cumsum(np.random.randn(200) * 0.2) + 100,
        'Yield_10Y': 3 + np.sin(np.linspace(0, 8*np.pi, 200)) + np.random.randn(200) * 0.3,
        'DXY': np.cumsum(np.random.randn(200) * 0.5) + 95,
        'SP500': np.cumsum(np.random.randn(200) * 2) + 3000,
        'XLF': np.cumsum(np.random.randn(200) * 2) + 50,
        'XLE': np.cumsum(np.random.randn(200) * 2.5) + 45,
        'AGG': np.cumsum(np.random.randn(200) * 0.8) + 105
    }, index=dates)
    
    # Initialize engine
    engine = InfluenceAnalysisEngine(panel)
    
    print("=" * 70)
    print("INFLUENCE ANALYSIS: WHAT'S DRIVING THE SYSTEM?")
    print("=" * 70)
    
    # 1. Current top influencers
    print("\n1. TOP INFLUENCERS (Most Recent)")
    print("-" * 70)
    top_influencers = engine.ranker.top_influencers(n_top=5)
    print(top_influencers.to_string(index=False))
    
    # 2. Most coherent pairs
    print("\n2. MOST COHERENT INDICATOR PAIRS")
    print("-" * 70)
    coherent_pairs = engine.pairwise.most_coherent_pairs(n_pairs=5)
    print(coherent_pairs.to_string(index=False))
    
    # 3. What drives dynamics over time
    print("\n3. PRIMARY DRIVERS OVER TIME (Last 12 months)")
    print("-" * 70)
    drivers = engine.detect_what_drives_dynamics().tail(12)
    print(drivers[['date', 'primary_driver', 'primary_magnitude', 
                   'concentration']].to_string(index=False))
    
    # 4. Influence regime changes
    print("\n4. INFLUENCE REGIME CHANGES")
    print("-" * 70)
    regime_changes = engine.temporal.detect_influence_regime_changes()
    n_changes = regime_changes['regime_change'].sum()
    print(f"Detected {n_changes} regime changes based on influence shifts")
    
    print("\n" + "=" * 70)
    print("This is the analysis Gemini suggested:")
    print("Focus on WHICH indicators drive dynamics, not just regime labels")
    print("=" * 70)
