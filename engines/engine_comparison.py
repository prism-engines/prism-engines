# VCF Research: Engine Comparison Framework
# Compare different analytical approaches on the same data

"""
This module provides tools to:
1. Run multiple engines on the same dataset
2. Compare their outputs systematically
3. Identify where they agree/disagree
4. Determine which engine provides more actionable insights
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Callable
import json


class EngineComparator:
    """
    Compare outputs from different analytical engines
    """
    
    def __init__(self, panel_df: pd.DataFrame):
        """
        panel_df: The shared dataset that all engines will analyze
        """
        self.panel = panel_df
        self.engine_results = {}
        
    def register_engine(self, name: str, engine_func: Callable, **kwargs):
        """
        Register an engine for comparison
        
        Parameters:
        -----------
        name : str
            Name of the engine (e.g., "VCF_Vector", "Gemini_Influence")
        engine_func : Callable
            Function that takes panel_df and returns analysis results
        **kwargs : 
            Additional parameters to pass to engine_func
        """
        print(f"Running {name} engine...")
        results = engine_func(self.panel, **kwargs)
        self.engine_results[name] = results
        print(f"✓ {name} complete")
        
    def compare_influence_rankings(self) -> pd.DataFrame:
        """
        Compare how different engines rank indicator influence
        
        Expects each engine to return a 'top_influencers' DataFrame
        with columns: ['indicator', 'influence_score', 'rank']
        """
        if len(self.engine_results) < 2:
            raise ValueError("Need at least 2 engines to compare")
        
        # Get rankings from each engine
        rankings = {}
        for engine_name, results in self.engine_results.items():
            if 'top_influencers' in results:
                rankings[engine_name] = results['top_influencers']
        
        # Create comparison matrix
        all_indicators = set()
        for ranking in rankings.values():
            all_indicators.update(ranking['indicator'].tolist())
        
        comparison = pd.DataFrame({'indicator': list(all_indicators)})
        
        for engine_name, ranking in rankings.items():
            # Merge rankings
            comparison = comparison.merge(
                ranking[['indicator', 'influence_score', 'rank']],
                on='indicator',
                how='left',
                suffixes=('', f'_{engine_name}')
            )
            comparison = comparison.rename(columns={
                'influence_score': f'{engine_name}_score',
                'rank': f'{engine_name}_rank'
            })
        
        # Fill NaN with low scores for indicators not in top N
        score_cols = [col for col in comparison.columns if col.endswith('_score')]
        comparison[score_cols] = comparison[score_cols].fillna(0)
        
        # Calculate agreement metrics
        rank_cols = [col for col in comparison.columns if col.endswith('_rank')]
        comparison['rank_variance'] = comparison[rank_cols].var(axis=1)
        comparison['score_variance'] = comparison[score_cols].var(axis=1)
        
        return comparison.sort_values('rank_variance')
    
    def identify_agreements(self, threshold: float = 0.8) -> Dict:
        """
        Find indicators that all engines agree are important
        
        Parameters:
        -----------
        threshold : float
            Minimum correlation required to consider engines "in agreement"
        
        Returns:
        --------
        Dict with agreement statistics
        """
        comparison = self.compare_influence_rankings()
        score_cols = [col for col in comparison.columns if col.endswith('_score')]
        
        # Correlation between engine scores
        from scipy.stats import spearmanr
        
        engine_names = list(self.engine_results.keys())
        correlations = {}
        
        for i, eng_a in enumerate(engine_names):
            for eng_b in engine_names[i+1:]:
                col_a = f'{eng_a}_score'
                col_b = f'{eng_b}_score'
                
                if col_a in comparison.columns and col_b in comparison.columns:
                    corr, _ = spearmanr(comparison[col_a], comparison[col_b])
                    correlations[f'{eng_a} vs {eng_b}'] = corr
        
        # Find indicators where engines agree (low rank variance)
        high_agreement = comparison[comparison['rank_variance'] < 1.0]
        
        return {
            'correlation_matrix': correlations,
            'high_agreement_indicators': high_agreement['indicator'].tolist(),
            'disagreement_indicators': comparison.nlargest(5, 'rank_variance')['indicator'].tolist()
        }
    
    def identify_disagreements(self, n_top: int = 10) -> pd.DataFrame:
        """
        Find the biggest disagreements between engines
        
        These are the cases where engines have very different interpretations
        """
        comparison = self.compare_influence_rankings()
        
        # Biggest disagreements = high score variance
        disagreements = comparison.nlargest(n_top, 'score_variance')
        
        return disagreements
    
    def consensus_ranking(self, method: str = 'average') -> pd.DataFrame:
        """
        Create a consensus ranking across all engines
        
        Parameters:
        -----------
        method : str
            'average': Average the scores
            'rank_average': Average the ranks (Borda count)
            'intersection': Only include indicators all engines ranked highly
        """
        comparison = self.compare_influence_rankings()
        
        if method == 'average':
            score_cols = [col for col in comparison.columns if col.endswith('_score')]
            comparison['consensus_score'] = comparison[score_cols].mean(axis=1)
            
        elif method == 'rank_average':
            rank_cols = [col for col in comparison.columns if col.endswith('_rank')]
            comparison['consensus_rank'] = comparison[rank_cols].mean(axis=1)
            comparison['consensus_score'] = 1.0 / (comparison['consensus_rank'] + 1)
            
        elif method == 'intersection':
            # Only indicators in top 10 for ALL engines
            rank_cols = [col for col in comparison.columns if col.endswith('_rank')]
            comparison['max_rank'] = comparison[rank_cols].max(axis=1)
            comparison = comparison[comparison['max_rank'] <= 10]
            comparison['consensus_score'] = 1.0 / (comparison['max_rank'] + 1)
        
        return comparison.sort_values('consensus_score', ascending=False)
    
    def temporal_agreement(self) -> pd.DataFrame:
        """
        Track agreement between engines over time
        
        Expects engines to return 'primary_driver' time series
        """
        temporal_data = {}
        
        for engine_name, results in self.engine_results.items():
            if 'temporal_drivers' in results:
                temporal_data[engine_name] = results['temporal_drivers']
        
        if len(temporal_data) < 2:
            return pd.DataFrame()
        
        # Merge temporal data
        merged = None
        for engine_name, data in temporal_data.items():
            data = data[['date', 'primary_driver']].copy()
            data = data.rename(columns={'primary_driver': f'{engine_name}_driver'})
            
            if merged is None:
                merged = data
            else:
                merged = merged.merge(data, on='date', how='outer')
        
        # Calculate agreement at each time point
        driver_cols = [col for col in merged.columns if col.endswith('_driver')]
        
        def calculate_agreement(row):
            drivers = [row[col] for col in driver_cols if pd.notna(row[col])]
            if len(drivers) == 0:
                return 0.0
            # Agreement = fraction of engines that agree on most common driver
            most_common = max(set(drivers), key=drivers.count)
            return drivers.count(most_common) / len(drivers)
        
        merged['agreement'] = merged.apply(calculate_agreement, axis=1)
        
        return merged
    
    def generate_comparison_report(self, output_file: str = None) -> Dict:
        """
        Generate comprehensive comparison report
        """
        report = {
            'engines_compared': list(self.engine_results.keys()),
            'n_engines': len(self.engine_results),
            'ranking_comparison': self.compare_influence_rankings().to_dict(),
            'agreement_analysis': self.identify_agreements(),
            'top_disagreements': self.identify_disagreements().to_dict(),
            'consensus_ranking': self.consensus_ranking().head(10).to_dict(),
        }
        
        # Add temporal agreement if available
        temporal_agreement = self.temporal_agreement()
        if not temporal_agreement.empty:
            report['temporal_agreement'] = {
                'mean_agreement': temporal_agreement['agreement'].mean(),
                'min_agreement': temporal_agreement['agreement'].min(),
                'periods_of_disagreement': temporal_agreement[
                    temporal_agreement['agreement'] < 0.5
                ]['date'].tolist()
            }
        
        # Save to file if requested
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            print(f"Report saved to {output_file}")
        
        return report
    
    def visualize_comparison(self) -> Dict[str, pd.DataFrame]:
        """
        Generate data for visualization
        
        Returns DataFrames suitable for plotting
        """
        viz_data = {}
        
        # 1. Ranking comparison scatter plot
        comparison = self.compare_influence_rankings()
        score_cols = [col for col in comparison.columns if col.endswith('_score')]
        viz_data['scatter_data'] = comparison[['indicator'] + score_cols]
        
        # 2. Agreement over time
        temporal = self.temporal_agreement()
        if not temporal.empty:
            viz_data['temporal_agreement'] = temporal[['date', 'agreement']]
        
        # 3. Heatmap of score differences
        score_diff_matrix = pd.DataFrame()
        engine_names = list(self.engine_results.keys())
        
        for i, eng_a in enumerate(engine_names):
            for eng_b in enumerate(engine_names[i+1:]):
                diff_col = f'{eng_a}_vs_{eng_b[1]}'
                col_a = f'{eng_a}_score'
                col_b = f'{eng_b[1]}_score'
                
                if col_a in comparison.columns and col_b in comparison.columns:
                    score_diff_matrix[diff_col] = np.abs(
                        comparison[col_a] - comparison[col_b]
                    )
        
        viz_data['difference_heatmap'] = score_diff_matrix
        
        return viz_data


class EngineEvaluator:
    """
    Evaluate engine performance using objective criteria
    """
    
    @staticmethod
    def evaluate_predictive_power(engine_results: Dict, 
                                  future_panel: pd.DataFrame,
                                  target_col: str = 'SP500',
                                  horizon: int = 1) -> Dict:
        """
        Evaluate if top influencers actually predict future movements
        
        Parameters:
        -----------
        engine_results : Dict
            Results from an engine (must include 'top_influencers')
        future_panel : pd.DataFrame
            Future data to test predictions against
        target_col : str
            Which indicator to predict
        horizon : int
            How many periods ahead to predict
        """
        if 'top_influencers' not in engine_results:
            return {'error': 'No top_influencers in results'}
        
        top_indicators = engine_results['top_influencers']['indicator'].tolist()[:5]
        
        # Simple test: Do top influencers' current values correlate with
        # future target movements?
        
        target_future = future_panel[target_col].shift(-horizon)
        target_change = target_future.pct_change()
        
        correlations = {}
        for indicator in top_indicators:
            if indicator in future_panel.columns:
                indicator_current = future_panel[indicator]
                corr = indicator_current.corr(target_change)
                correlations[indicator] = corr
        
        return {
            'indicator_correlations': correlations,
            'mean_correlation': np.mean(list(correlations.values())),
            'max_correlation': np.max(list(correlations.values()))
        }
    
    @staticmethod
    def evaluate_stability(engine_results_t1: Dict, 
                          engine_results_t2: Dict) -> Dict:
        """
        Evaluate if engine results are stable across time periods
        
        Compare rankings from two different time periods
        """
        if 'top_influencers' not in engine_results_t1 or 'top_influencers' not in engine_results_t2:
            return {'error': 'Missing top_influencers'}
        
        ranking_t1 = engine_results_t1['top_influencers']
        ranking_t2 = engine_results_t2['top_influencers']
        
        # Rank correlation
        merged = ranking_t1.merge(ranking_t2, on='indicator', suffixes=('_t1', '_t2'))
        
        from scipy.stats import spearmanr
        corr, _ = spearmanr(merged['rank_t1'], merged['rank_t2'])
        
        # Top-5 overlap
        top5_t1 = set(ranking_t1.head(5)['indicator'])
        top5_t2 = set(ranking_t2.head(5)['indicator'])
        overlap = len(top5_t1 & top5_t2) / 5.0
        
        return {
            'rank_correlation': corr,
            'top5_overlap': overlap,
            'stable': corr > 0.7 and overlap > 0.6
        }
    
    @staticmethod
    def evaluate_interpretability(engine_results: Dict) -> Dict:
        """
        Evaluate how interpretable the results are
        
        Criteria:
        - Clear primary driver (not too diffuse)
        - Reasonable number of important indicators
        - Stability of influence scores
        """
        if 'top_influencers' not in engine_results:
            return {'error': 'Missing top_influencers'}
        
        top_influencers = engine_results['top_influencers']
        
        # Concentration: How much influence is in top indicator?
        total_score = top_influencers['influence_score'].sum()
        top1_score = top_influencers.iloc[0]['influence_score']
        concentration = top1_score / total_score if total_score > 0 else 0
        
        # Effective number of influencers (Shannon entropy)
        scores = top_influencers['influence_score'].values
        scores_norm = scores / scores.sum()
        entropy = -np.sum(scores_norm * np.log(scores_norm + 1e-10))
        effective_n = np.exp(entropy)
        
        return {
            'concentration': concentration,
            'effective_n_influencers': effective_n,
            'interpretable': 0.3 < concentration < 0.7 and 2 < effective_n < 8
        }


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

def example_vcf_engine(panel_df, **kwargs):
    """
    Example wrapper for your VCF engine
    """
    from influence_analysis import InfluenceAnalysisEngine
    
    engine = InfluenceAnalysisEngine(panel_df)
    
    return {
        'top_influencers': engine.ranker.top_influencers(n_top=10),
        'temporal_drivers': engine.detect_what_drives_dynamics()
    }


def example_gemini_engine(panel_df, **kwargs):
    """
    Example wrapper for Gemini's approach
    (Replace with actual Gemini implementation)
    """
    # Placeholder - replace with actual Gemini engine
    from influence_analysis import InfluenceRanking
    
    ranker = InfluenceRanking(panel_df)
    
    # Gemini might use different weighting or methods
    # This is just a placeholder
    return {
        'top_influencers': ranker.top_influencers(n_top=10),
        'temporal_drivers': pd.DataFrame()  # Replace with actual
    }


if __name__ == "__main__":
    """
    Example comparison between your engine and Gemini's
    """
    
    # Create sample data
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
    
    # Initialize comparator
    comparator = EngineComparator(panel)
    
    # Register engines
    comparator.register_engine('VCF_Influence', example_vcf_engine)
    comparator.register_engine('Gemini_Approach', example_gemini_engine)
    
    print("=" * 70)
    print("ENGINE COMPARISON REPORT")
    print("=" * 70)
    
    # 1. Compare rankings
    print("\n1. RANKING COMPARISON")
    print("-" * 70)
    ranking_comparison = comparator.compare_influence_rankings()
    print(ranking_comparison.head(10).to_string(index=False))
    
    # 2. Agreement analysis
    print("\n2. AGREEMENT ANALYSIS")
    print("-" * 70)
    agreements = comparator.identify_agreements()
    print(f"Correlation: {agreements['correlation_matrix']}")
    print(f"\nHigh agreement indicators: {agreements['high_agreement_indicators'][:5]}")
    
    # 3. Disagreements
    print("\n3. BIGGEST DISAGREEMENTS")
    print("-" * 70)
    disagreements = comparator.identify_disagreements(n_top=5)
    print(disagreements[['indicator', 'score_variance']].to_string(index=False))
    
    # 4. Consensus ranking
    print("\n4. CONSENSUS RANKING")
    print("-" * 70)
    consensus = comparator.consensus_ranking(method='average')
    print(consensus[['indicator', 'consensus_score']].head(10).to_string(index=False))
    
    print("\n" + "=" * 70)
    print("Use this framework to compare your engine vs Gemini's systematically")
    print("=" * 70)
