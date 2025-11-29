# Mathematical Lens Comparison Framework
# Apply different mathematical approaches to the same data and compare insights

"""
Core Philosophy:
- NO regime labels
- NO pillar groupings
- NO predetermined structure
- JUST different mathematical lenses looking at raw data
- Then compare: where do they agree? disagree? what does each uniquely see?

This is pure mathematical exploration, not market timing.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple, Callable
import warnings
import os
import json
warnings.filterwarnings('ignore')


# =============================================================================
# LENS 1: VECTOR MAGNITUDE
# =============================================================================

class MagnitudeLens:
    """
    Simplest lens: L2 norm across all indicators
    Answers: "How much is happening overall?"
    """

    def __init__(self, name: str = "Magnitude"):
        self.name = name

    def analyze(self, panel: pd.DataFrame) -> Dict:
        """
        Returns:
        - magnitude: overall system state
        - contribution: how much each indicator contributes to magnitude
        """
        # Normalize
        panel_norm = (panel - panel.mean()) / panel.std()

        # Overall magnitude
        magnitude = np.sqrt((panel_norm ** 2).sum(axis=1))

        # Individual contributions (what % of total magnitude squared)
        contributions = {}
        for date in panel.index:
            squared_sum = (panel_norm.loc[date] ** 2).sum()
            contrib = (panel_norm.loc[date] ** 2) / squared_sum if squared_sum > 0 else panel_norm.loc[date] * 0
            contributions[date] = contrib.to_dict()

        contributions_df = pd.DataFrame(contributions).T

        return {
            'magnitude': magnitude,
            'contributions': contributions_df,
            'method': 'L2 Euclidean norm'
        }

    def top_indicators(self, result: Dict, date: pd.Timestamp, n: int = 5) -> List[Tuple[str, float]]:
        """Get top N most important indicators at given date"""
        if date not in result['contributions'].index:
            return []

        contrib = result['contributions'].loc[date].sort_values(ascending=False)
        return list(zip(contrib.index[:n], contrib.values[:n]))


# =============================================================================
# LENS 2: PRINCIPAL COMPONENT ANALYSIS
# =============================================================================

class PCALens:
    """
    Linear dimensionality reduction
    Answers: "What are the natural factors in this data?"
    """

    def __init__(self, name: str = "PCA"):
        self.name = name

    def analyze(self, panel: pd.DataFrame) -> Dict:
        """
        Returns:
        - n_components: how many factors explain 90% variance
        - loadings: which indicators load on which factors
        - scores: factor scores over time
        - importance: which indicators matter most (sum of squared loadings)
        """
        from sklearn.decomposition import PCA

        # Normalize
        panel_norm = (panel - panel.mean()) / panel.std()
        panel_clean = panel_norm.dropna()

        # Fit PCA
        pca = PCA()
        scores = pca.fit_transform(panel_clean)

        # How many components for 90% variance?
        cumvar = pca.explained_variance_ratio_.cumsum()
        n_components = (cumvar < 0.90).sum() + 1

        # Loadings (which indicators → which components)
        loadings = pd.DataFrame(
            pca.components_[:n_components].T,
            index=panel_clean.columns,
            columns=[f'PC{i+1}' for i in range(n_components)]
        )

        # Indicator importance = sum of squared loadings
        importance = (loadings ** 2).sum(axis=1).sort_values(ascending=False)

        # PC scores over time
        pc_scores = pd.DataFrame(
            scores[:, :n_components],
            index=panel_clean.index,
            columns=[f'PC{i+1}' for i in range(n_components)]
        )

        return {
            'n_components': n_components,
            'loadings': loadings,
            'pc_scores': pc_scores,
            'importance': importance,
            'explained_variance': pca.explained_variance_ratio_[:n_components],
            'method': 'Principal Component Analysis'
        }

    def top_indicators(self, result: Dict, date: pd.Timestamp, n: int = 5) -> List[Tuple[str, float]]:
        """Top indicators by overall importance (not time-specific for PCA)"""
        importance = result['importance']
        return list(zip(importance.index[:n], importance.values[:n]))


# =============================================================================
# LENS 3: GRANGER CAUSALITY
# =============================================================================

class GrangerLens:
    """
    Temporal causality
    Answers: "Which indicators predict/cause others?"
    """

    def __init__(self, name: str = "Granger", max_lag: int = 6):
        self.name = name
        self.max_lag = max_lag

    def analyze(self, panel: pd.DataFrame) -> Dict:
        """
        Returns:
        - causality_matrix: [i,j] = strength of i causing j
        - out_degree: how much each indicator causes others (source strength)
        - in_degree: how much each is caused by others (sink strength)
        """
        from statsmodels.tsa.stattools import grangercausalitytests

        n = len(panel.columns)
        causality_matrix = np.zeros((n, n))

        for i, col_i in enumerate(panel.columns):
            for j, col_j in enumerate(panel.columns):
                if i == j:
                    continue

                try:
                    # Granger test: does i cause j?
                    data = panel[[col_j, col_i]].dropna()

                    if len(data) < self.max_lag + 10:
                        continue

                    result = grangercausalitytests(data, self.max_lag, verbose=False)

                    # Get minimum p-value across lags
                    p_values = [result[lag][0]['ssr_ftest'][1] for lag in range(1, self.max_lag + 1)]
                    min_p = np.min(p_values)

                    # Convert to causality strength (1 - p_value)
                    causality_matrix[i, j] = 1.0 - min_p

                except:
                    causality_matrix[i, j] = 0.0

        causality_df = pd.DataFrame(
            causality_matrix,
            index=panel.columns,
            columns=panel.columns
        )

        # Out-degree: how much does this indicator cause others?
        out_degree = causality_df.sum(axis=1).sort_values(ascending=False)

        # In-degree: how much is this caused by others?
        in_degree = causality_df.sum(axis=0).sort_values(ascending=False)

        return {
            'causality_matrix': causality_df,
            'out_degree': out_degree,  # "drivers"
            'in_degree': in_degree,    # "followers"
            'method': 'Granger Causality'
        }

    def top_indicators(self, result: Dict, date: pd.Timestamp, n: int = 5) -> List[Tuple[str, float]]:
        """Top causal drivers (not time-specific for Granger)"""
        drivers = result['out_degree']
        return list(zip(drivers.index[:n], drivers.values[:n]))


# =============================================================================
# LENS 4: DYNAMIC MODE DECOMPOSITION
# =============================================================================

class DMDLens:
    """
    Identifies oscillatory modes and growth/decay patterns
    Answers: "What are the dominant temporal patterns?"
    """

    def __init__(self, name: str = "DMD"):
        self.name = name

    def analyze(self, panel: pd.DataFrame) -> Dict:
        """
        Returns:
        - modes: spatial patterns
        - frequencies: temporal frequencies
        - growth_rates: growth/decay rates
        - mode_importance: which modes are strongest
        """
        from scipy.linalg import svd, eig

        # Normalize
        panel_norm = (panel - panel.mean()) / panel.std()
        X = panel_norm.dropna().T.values

        # DMD
        X1 = X[:, :-1]
        X2 = X[:, 1:]

        # SVD of X1
        U, s, Vt = svd(X1, full_matrices=False)

        # Truncate to significant modes
        r = min(10, len(s))  # Keep top 10 modes
        U = U[:, :r]
        s = s[:r]
        Vt = Vt[:r, :]

        # DMD operator
        S_inv = np.diag(1.0 / s)
        A_tilde = U.T @ X2 @ Vt.T @ S_inv

        # Eigendecomposition
        eigenvalues, eigenvectors = eig(A_tilde)

        # DMD modes
        modes = X2 @ Vt.T @ S_inv @ eigenvectors

        # Frequencies and growth rates
        dt = 1.0  # assuming unit time steps
        frequencies = np.log(eigenvalues).imag / (2 * np.pi * dt)
        growth_rates = np.log(np.abs(eigenvalues)) / dt

        # Mode amplitudes (importance)
        amplitudes = np.abs(np.linalg.lstsq(modes, X[:, 0], rcond=None)[0])

        # Create importance ranking
        mode_importance = pd.Series(amplitudes, index=[f'Mode_{i+1}' for i in range(len(amplitudes))])
        mode_importance = mode_importance.sort_values(ascending=False)

        # Which indicators participate most in dominant modes?
        mode_participation = np.abs(modes).sum(axis=1)
        indicator_importance = pd.Series(mode_participation, index=panel_norm.columns).sort_values(ascending=False)

        return {
            'n_modes': r,
            'frequencies': frequencies,
            'growth_rates': growth_rates,
            'mode_importance': mode_importance,
            'indicator_importance': indicator_importance,
            'method': 'Dynamic Mode Decomposition'
        }

    def top_indicators(self, result: Dict, date: pd.Timestamp, n: int = 5) -> List[Tuple[str, float]]:
        """Top indicators by modal participation"""
        importance = result['indicator_importance']
        return list(zip(importance.index[:n], importance.values[:n]))


# =============================================================================
# LENS 5: ROLLING INFLUENCE (What I built before)
# =============================================================================

class InfluenceLens:
    """
    Time-varying influence scores
    Answers: "Which indicators are most active/volatile right now?"
    """

    def __init__(self, name: str = "Influence", window: int = 12):
        self.name = name
        self.window = window

    def analyze(self, panel: pd.DataFrame) -> Dict:
        """
        Returns:
        - influence_scores: time-varying importance for each indicator
        - concentration: how concentrated is influence over time
        """
        # Normalize
        panel_norm = (panel - panel.mean()) / panel.std()

        # Rolling magnitude (volatility)
        rolling_influence = {}

        for col in panel.columns:
            # Rolling std deviation (activity level)
            rolling_std = panel_norm[col].rolling(self.window).std()

            # Absolute z-score (current deviation from mean)
            current_z = panel_norm[col].abs()

            # Combined influence: volatility × current deviation
            influence = rolling_std * current_z

            rolling_influence[col] = influence

        influence_df = pd.DataFrame(rolling_influence)

        # Normalize each row to sum to 1 (relative influence)
        influence_normalized = influence_df.div(influence_df.sum(axis=1), axis=0)

        # Concentration (Herfindahl index)
        concentration = (influence_normalized ** 2).sum(axis=1)

        return {
            'influence_scores': influence_normalized,
            'concentration': concentration,
            'method': 'Rolling volatility × current deviation'
        }

    def top_indicators(self, result: Dict, date: pd.Timestamp, n: int = 5) -> List[Tuple[str, float]]:
        """Top indicators at specific date"""
        if date not in result['influence_scores'].index:
            return []

        scores = result['influence_scores'].loc[date].sort_values(ascending=False)
        return list(zip(scores.index[:n], scores.values[:n]))


# =============================================================================
# LENS 6: MUTUAL INFORMATION
# =============================================================================

class MutualInformationLens:
    """
    Information-theoretic dependencies
    Answers: "Which indicators share the most information?"
    """

    def __init__(self, name: str = "MutualInfo"):
        self.name = name

    def analyze(self, panel: pd.DataFrame) -> Dict:
        """
        Returns:
        - mi_matrix: pairwise mutual information
        - mi_sum: total information shared by each indicator
        - redundancy: which indicators are most redundant
        """
        from sklearn.feature_selection import mutual_info_regression

        panel_clean = panel.dropna()
        n = len(panel_clean.columns)
        mi_matrix = np.zeros((n, n))

        for i, col_i in enumerate(panel_clean.columns):
            X = panel_clean.drop(columns=[col_i]).values
            y = panel_clean[col_i].values

            # Mutual information with all other variables
            mi_scores = mutual_info_regression(X, y, random_state=42)

            # Fill matrix
            other_cols = [c for c in panel_clean.columns if c != col_i]
            for j, col_j in enumerate(other_cols):
                j_idx = panel_clean.columns.get_loc(col_j)
                mi_matrix[i, j_idx] = mi_scores[j]

        mi_df = pd.DataFrame(mi_matrix, index=panel_clean.columns, columns=panel_clean.columns)

        # Sum of MI with others (information centrality)
        mi_sum = mi_df.sum(axis=1).sort_values(ascending=False)

        return {
            'mi_matrix': mi_df,
            'information_centrality': mi_sum,
            'method': 'Mutual Information'
        }

    def top_indicators(self, result: Dict, date: pd.Timestamp, n: int = 5) -> List[Tuple[str, float]]:
        """Top by information centrality (not time-specific)"""
        centrality = result['information_centrality']
        return list(zip(centrality.index[:n], centrality.values[:n]))


# =============================================================================
# META-LAYER: LENS COMPARATOR
# =============================================================================

class LensComparator:
    """
    Compare multiple mathematical lenses on the same data

    This is the meta-layer that shows:
    - Where do different methods agree?
    - Where do they disagree?
    - What does each uniquely see?
    """

    def __init__(self, panel: pd.DataFrame):
        self.panel = panel
        self.lenses = {}
        self.results = {}

    def add_lens(self, lens: object):
        """Add a mathematical lens to compare"""
        self.lenses[lens.name] = lens
        print(f"✓ Added lens: {lens.name}")

    def run_all(self):
        """Run all lenses on the data"""
        print(f"\n{'='*70}")
        print(f"Running {len(self.lenses)} mathematical lenses on data")
        print(f"Data: {self.panel.shape[0]} observations × {self.panel.shape[1]} indicators")
        print(f"{'='*70}\n")

        for name, lens in self.lenses.items():
            print(f"Running {name}...")
            try:
                self.results[name] = lens.analyze(self.panel)
                print(f"  ✓ Complete")
            except Exception as e:
                print(f"  ✗ Error: {e}")
                self.results[name] = None

        print(f"\n✅ All lenses complete\n")
        return self.results

    def compare_at_date(self, date: pd.Timestamp, n_top: int = 5) -> pd.DataFrame:
        """
        Compare what different lenses say are the top indicators at a specific date
        """
        comparison = {}

        for lens_name, lens in self.lenses.items():
            if self.results[lens_name] is None:
                continue

            top_indicators = lens.top_indicators(self.results[lens_name], date, n_top)

            # Store as dict for easier comparison
            comparison[lens_name] = {
                ind: score for ind, score in top_indicators
            }

        # Create DataFrame
        all_indicators = set()
        for lens_results in comparison.values():
            all_indicators.update(lens_results.keys())

        comp_df = pd.DataFrame(index=sorted(all_indicators))

        for lens_name, lens_results in comparison.items():
            comp_df[lens_name] = pd.Series(lens_results)

        # Add rank columns
        for lens_name in comparison.keys():
            comp_df[f'{lens_name}_rank'] = comp_df[lens_name].rank(ascending=False)

        return comp_df.sort_values(by=list(comparison.keys())[0], ascending=False)

    def agreement_matrix(self) -> pd.DataFrame:
        """
        Compute how much different lenses agree with each other

        For each pair of lenses, compute rank correlation of their top indicators
        """
        from scipy.stats import spearmanr

        lens_names = list(self.lenses.keys())
        n = len(lens_names)
        agreement = np.zeros((n, n))

        # Get overall importance from each lens
        importance_rankings = {}
        for lens_name in lens_names:
            if self.results[lens_name] is None:
                continue

            # Extract overall importance (method-dependent)
            if 'importance' in self.results[lens_name]:
                importance_rankings[lens_name] = self.results[lens_name]['importance']
            elif 'out_degree' in self.results[lens_name]:
                importance_rankings[lens_name] = self.results[lens_name]['out_degree']
            elif 'information_centrality' in self.results[lens_name]:
                importance_rankings[lens_name] = self.results[lens_name]['information_centrality']
            elif 'indicator_importance' in self.results[lens_name]:
                importance_rankings[lens_name] = self.results[lens_name]['indicator_importance']

        # Compute pairwise correlations
        for i, lens_i in enumerate(lens_names):
            for j, lens_j in enumerate(lens_names):
                if i == j:
                    agreement[i, j] = 1.0
                    continue

                if lens_i not in importance_rankings or lens_j not in importance_rankings:
                    agreement[i, j] = np.nan
                    continue

                # Align indicators
                common_indicators = importance_rankings[lens_i].index.intersection(
                    importance_rankings[lens_j].index
                )

                if len(common_indicators) < 3:
                    agreement[i, j] = np.nan
                    continue

                rank_i = importance_rankings[lens_i][common_indicators]
                rank_j = importance_rankings[lens_j][common_indicators]

                corr, _ = spearmanr(rank_i, rank_j)
                agreement[i, j] = corr

        return pd.DataFrame(agreement, index=lens_names, columns=lens_names)

    def consensus_indicators(self, n_top: int = 5) -> pd.DataFrame:
        """
        Which indicators do MOST lenses agree are important?
        """
        # Collect rankings from all lenses
        all_rankings = []

        for lens_name in self.lenses.keys():
            if self.results[lens_name] is None:
                continue

            # Get top indicators (method varies by lens)
            if 'importance' in self.results[lens_name]:
                ranking = self.results[lens_name]['importance']
            elif 'out_degree' in self.results[lens_name]:
                ranking = self.results[lens_name]['out_degree']
            elif 'information_centrality' in self.results[lens_name]:
                ranking = self.results[lens_name]['information_centrality']
            elif 'indicator_importance' in self.results[lens_name]: # CORRECTED LINE
                ranking = self.results[lens_name]['indicator_importance']
            else:
                continue

            all_rankings.append(ranking)

        # Combine rankings (average rank)
        consensus_df = pd.DataFrame()
        for i, ranking in enumerate(all_rankings):
            lens_name = list(self.lenses.keys())[i]
            consensus_df[lens_name] = ranking

        # Average rank across lenses
        consensus_df['mean_score'] = consensus_df.mean(axis=1, skipna=True)
        consensus_df['std_score'] = consensus_df.std(axis=1, skipna=True)
        consensus_df['n_lenses'] = consensus_df.notna().sum(axis=1)

        # Sort by mean score
        consensus_df = consensus_df.sort_values('mean_score', ascending=False)

        return consensus_df.head(n_top)

    def unique_insights(self) -> Dict:
        """
        What does each lens see that others don't?

        For each lens, find indicators it ranks highly but others don't
        """
        unique = {}

        for lens_name in self.lenses.keys():
            if self.results[lens_name] is None:
                continue

            # Get this lens's top indicators
            if 'importance' in self.results[lens_name]:
                this_ranking = self.results[lens_name]['importance']
            elif 'out_degree' in self.results[lens_name]:
                this_ranking = self.results[lens_name]['out_degree']
            elif 'information_centrality' in self.results[lens_name]:
                this_ranking = self.results[lens_name]['information_centrality']
            elif 'indicator_importance' in self.results[lens_name]:
                this_ranking = self.results[lens_name]['indicator_importance']
            else:
                continue

            # Get top 10 from this lens
            top_10_this = set(this_ranking.head(10).index)

            # Get top 10 from all other lenses
            top_10_others = set()
            for other_lens in self.lenses.keys():
                if other_lens == lens_name or self.results[other_lens] is None:
                    continue

                if 'importance' in self.results[other_lens]:
                    other_ranking = self.results[other_lens]['importance']
                elif 'out_degree' in self.results[other_lens]:
                    other_ranking = self.results[other_lens]['out_degree']
                elif 'information_centrality' in self.results[other_lens]:
                    other_ranking = self.results[other_lens]['information_centrality']
                elif 'indicator_importance' in self.results[other_lens]:
                    other_ranking = self.results[other_lens]['indicator_importance']
                else:
                    continue

                top_10_others.update(other_ranking.head(10).index)

            # Unique to this lens
            unique[lens_name] = list(top_10_this - top_10_others)

        return unique


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def run_full_lens_analysis(panel: pd.DataFrame,
                           date_to_analyze: pd.Timestamp = None) -> LensComparator:
    """
    Convenience function: run all lenses and generate comparison
    """
    comparator = LensComparator(panel)

    # Add all available lenses
    comparator.add_lens(MagnitudeLens())
    comparator.add_lens(PCALens())
    comparator.add_lens(GrangerLens())
    comparator.add_lens(DMDLens())
    comparator.add_lens(InfluenceLens())
    comparator.add_lens(MutualInformationLens())

    # Run all
    comparator.run_all()

    # Generate comparisons
    print("\n" + "="*70)
    print("LENS COMPARISON ANALYSIS")
    print("="*70)

    # Agreement matrix
    print("\nLens Agreement Matrix (Spearman correlation):")
    print(comparator.agreement_matrix().to_string())

    # Consensus indicators
    print("\nConsensus Indicators (agreed upon by most lenses):")
    print(comparator.consensus_indicators(n_top=10).to_string())

    # Unique insights
    print("\nUnique Insights by Lens:")
    unique = comparator.unique_insights()
    for lens_name, indicators in unique.items():
        if indicators:
            print(f"  {lens_name}: {indicators}")

    # Date-specific comparison if provided
    if date_to_analyze and date_to_analyze in panel.index:
        print(f"\nComparison at {date_to_analyze.strftime('%Y-%m-%d')}:")
        print(comparator.compare_at_date(date_to_analyze, n_top=5).to_string())

    return comparator

def save_lens_analysis_results(normalized_data_df: pd.DataFrame,
                               current_regime_dict: Dict,
                               comparator_object: LensComparator,
                               output_base_dir: str = '/content/drive/MyDrive/prism_engine/outputs'):
    """
    Saves the results of the lens analysis to specified output directory.

    Args:
        normalized_data_df: The DataFrame containing the normalized data.
        current_regime_dict: The dictionary containing current regime details.
        comparator_object: The LensComparator object with analysis results.
        output_base_dir: The base directory to save outputs.
    """
    os.makedirs(output_base_dir, exist_ok=True)
    print(f"\nSaving analysis outputs to: {output_base_dir}")

    # 1. Save Normalized Data
    normalized_path = os.path.join(output_base_dir, 'normalized_data.csv')
    normalized_data_df.to_csv(normalized_path)
    print(f"✓ Saved normalized data to {normalized_path}")

    # 2. Save Current Regime Details
    current_regime_series = pd.Series(current_regime_dict)
    current_regime_path = os.path.join(output_base_dir, 'current_regime.csv')
    current_regime_series.to_csv(current_regime_path, header=False)
    print(f"✓ Saved current regime details to {current_regime_path}")

    # 3. Save LensComparator outputs
    if comparator_object is not None:
        # Agreement Matrix
        agreement_matrix_path = os.path.join(output_base_dir, 'lens_agreement_matrix.csv')
        comparator_object.agreement_matrix().to_csv(agreement_matrix_path)
        print(f"✓ Saved lens agreement matrix to {agreement_matrix_path}")

        # Consensus Indicators
        consensus_indicators_path = os.path.join(output_base_dir, 'consensus_indicators.csv')
        comparator_object.consensus_indicators().to_csv(consensus_indicators_path)
        print(f"✓ Saved consensus indicators to {consensus_indicators_path}")

        # Unique Insights (save as JSON)
        unique_insights_path = os.path.join(output_base_dir, 'unique_insights.json')
        with open(unique_insights_path, 'w') as f:
            json.dump(comparator_object.unique_insights(), f, indent=4)
        print(f"✓ Saved unique insights to {unique_insights_path}")
    else:
        print("✗ Comparator object is None, skipping saving of comparator outputs.")

    print("✓ All requested analysis outputs have been saved!")


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    """
    Example: Compare mathematical lenses on sample data
    """

    # Create sample data (replace with your actual data)
    dates = pd.date_range('2015-01-01', periods=200, freq='M')
    panel = pd.DataFrame({
        'GDP': np.cumsum(np.random.randn(200) * 0.3),
        'CPI': np.cumsum(np.random.randn(200) * 0.2),
        'VIX': 15 + np.random.randn(200) * 5,
        'SPY': np.cumsum(np.random.randn(200) * 2) + 300,
        'DGS10': 2 + np.random.randn(200) * 0.5,
        'M2': np.cumsum(np.random.randn(200) * 0.4),
    }, index=dates)

    # Run full analysis
    comparator = run_full_lens_analysis(panel, date_to_analyze=panel.index[-1])

    print("\n" + "="*70)
    print("This shows how DIFFERENT MATHEMATICAL METHODS see the SAME data")
    print("No regimes, no pillars - just pure mathematical perspectives")
    print("="*70)

    # Example of how to call the saving function with placeholder data
    # Replace 'normalized' and 'current_regime' with your actual variables if using this block in the notebook
    # For this example, we'll create dummy data to show the function call
    dummy_normalized_data = pd.DataFrame(np.random.randn(10, 3), columns=['A', 'B', 'C'])
    dummy_current_regime = {'regime': 'TEST', 'confidence': 0.9}

    # To actually save the results from the run_full_lens_analysis above, you would use:
    # save_lens_analysis_results(dummy_normalized_data, dummy_current_regime, comparator)
    # NOTE: 'normalized' and 'current_regime' are not defined in this __main__ block directly,
    # they are from the broader Colab notebook execution context. If this file were run standalone,
    # you'd need to generate or load them first.

    # For the purpose of integrating into this .py file, and assuming 'normalized' and 'current_regime'
    # are available from some external context, we'll call it with the comparator directly.
    # However, if this were run as a standalone script, you'd need to pass the actual
    # normalized data and current regime dict generated earlier.

    # --- Placeholder for the actual save call within the __main__ block (if all data was locally generated) ---
    # If you had `normalized` and `current_regime` defined within this `if __name__ == "__main__":` block,
    # you would call:
    # save_lens_analysis_results(normalized, current_regime, comparator)

    # As an example, calling with the comparator from this __main__ block and some dummy data
    # to show the function signature.
    print("\n--- Demonstrating save_lens_analysis_results function call ---")
    save_lens_analysis_results(panel, {'regime': 'EXAMPLE', 'confidence': 0.8}, comparator)