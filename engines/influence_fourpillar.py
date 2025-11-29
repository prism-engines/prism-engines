# VCF 4-Pillar Geometry → Influence Analysis Connector
# Specifically for your VCF-RESEARCH structure

"""
Your VCF Setup:
- 4 Pillars: Macro, Liquidity, Risk, Equity
- Geometry: Theta (Macro/Liquidity), Phi (Equity/Risk)
- Coherence: Signal strength for each plane

This connects your pillar structure to influence analysis
"""

import pandas as pd
import numpy as np
import os


class VCF_4Pillar_Analyzer:
    """
    Analyze your 4-pillar VCF structure
    """
    
    def __init__(self, geometry_file: str, normalized_panel_file: str):
        """
        geometry_file: geometry_engine_full.csv (your output)
        normalized_panel_file: normalized_panel_monthly.csv (your inputs)
        """
        self.geometry = pd.read_csv(geometry_file, parse_dates=['date'], index_col='date')
        self.panel = pd.read_csv(normalized_panel_file, parse_dates=['date'], index_col='date')
        
        # Align dates
        common_dates = self.geometry.index.intersection(self.panel.index)
        self.geometry = self.geometry.loc[common_dates]
        self.panel = self.panel.loc[common_dates]
        
        print(f"✓ Loaded geometry: {len(self.geometry)} observations")
        print(f"✓ Loaded panel: {len(self.panel)} observations with {len(self.panel.columns)} metrics")
        print(f"✓ Common dates: {len(common_dates)}")
    
    def get_pillar_metrics(self, registry_path: str) -> dict:
        """
        Extract which metrics belong to which pillar from registry
        """
        import json
        
        with open(registry_path, 'r') as f:
            registry = json.load(f)
        
        pillar_metrics = {
            'Macro': [],
            'Liquidity': [],
            'Risk': [],
            'Equity': []
        }
        
        category_map = {
            'Macro': 'Macro',
            'Labor': 'Macro',  # Labor goes into Macro pillar
            'Liquidity': 'Liquidity',
            'Volatility': 'Risk',
            'Equities': 'Equity'
        }
        
        for metric_id, info in registry.items():
            category = info.get('category')
            if category in category_map:
                pillar = category_map[category]
                if metric_id in self.panel.columns:
                    pillar_metrics[pillar].append(metric_id)
        
        return pillar_metrics
    
    def analyze_pillar_influence(self, pillar_metrics: dict) -> pd.DataFrame:
        """
        For each pillar, determine which underlying metrics drive it most
        
        This answers: "When Macro score is high, which metrics (GDP, CPI, etc.) 
        contribute most?"
        """
        results = []
        
        for date in self.panel.index:
            row = {'date': date}
            
            # Get pillar scores at this date
            if date in self.geometry.index:
                row['theta_deg'] = self.geometry.loc[date, 'theta_deg']
                row['phi_deg'] = self.geometry.loc[date, 'phi_deg']
                row['coherence_theta'] = self.geometry.loc[date, 'coherence_theta']
                row['coherence_phi'] = self.geometry.loc[date, 'coherence_phi']
            
            # For each pillar, find which metric contributes most
            for pillar_name, metrics in pillar_metrics.items():
                if not metrics:
                    continue
                
                # Get values for all metrics in this pillar
                values = self.panel.loc[date, metrics]
                
                # Find metric with highest absolute value (most extreme)
                if values.notna().any():
                    abs_values = values.abs()
                    top_metric = abs_values.idxmax()
                    top_value = values[top_metric]
                    
                    row[f'{pillar_name}_top_metric'] = top_metric
                    row[f'{pillar_name}_top_value'] = top_value
                    row[f'{pillar_name}_avg'] = values.mean()
            
            results.append(row)
        
        return pd.DataFrame(results)
    
    def find_covid_crash_drivers(self, influence_df: pd.DataFrame) -> pd.DataFrame:
        """
        What drove the COVID crash in your 4-pillar structure?
        """
        covid_period = influence_df.loc['2020-02':'2020-04']
        
        print("=" * 70)
        print("COVID CRASH ANALYSIS (Feb-Apr 2020)")
        print("=" * 70)
        
        if len(covid_period) == 0:
            print("No data for COVID period!")
            return pd.DataFrame()
        
        print(f"\nCoherence during crash:")
        print(f"  Theta (Macro/Liq): {covid_period['coherence_theta'].mean():.2f}")
        print(f"  Phi (Equity/Risk): {covid_period['coherence_phi'].mean():.2f}")
        
        print(f"\nDominant metrics per pillar:")
        for pillar in ['Macro', 'Liquidity', 'Risk', 'Equity']:
            top_col = f'{pillar}_top_metric'
            val_col = f'{pillar}_top_value'
            
            if top_col in covid_period.columns:
                # Most common top metric
                top_metric = covid_period[top_col].mode()[0] if len(covid_period[top_col].mode()) > 0 else 'N/A'
                avg_val = covid_period[val_col].mean()
                print(f"  {pillar}: {top_metric} (avg: {avg_val:.2f}σ)")
        
        return covid_period
    
    def theta_phi_decomposition(self) -> pd.DataFrame:
        """
        Decompose theta and phi into their component contributions
        
        Theta = arctan2(macro_z, liquidity_z)
        Phi = arctan2(equity_z, risk_z)
        
        This shows which pillar dominates at each time
        """
        decomp = pd.DataFrame(index=self.geometry.index)
        
        # Get z-scores for each pillar
        decomp['macro_score'] = self.geometry['macro_score']
        decomp['liquidity_score'] = self.geometry['liquidity_score']
        decomp['risk_score'] = self.geometry['risk_score']
        decomp['equity_score'] = self.geometry['equity_score']
        
        # Determine which pillar dominates
        decomp['theta'] = self.geometry['theta_deg']
        decomp['phi'] = self.geometry['phi_deg']
        
        # Theta decomposition
        decomp['theta_regime'] = 'Balanced'
        decomp.loc[decomp['theta'] > 45, 'theta_regime'] = 'Macro Dominant'
        decomp.loc[decomp['theta'] < -45, 'theta_regime'] = 'Both Weak'
        decomp.loc[(decomp['theta'] > -45) & (decomp['theta'] < 45), 'theta_regime'] = 'Liquidity Dominant'
        
        # Phi decomposition
        decomp['phi_regime'] = 'Balanced'
        decomp.loc[decomp['phi'] > 45, 'phi_regime'] = 'Risk-On'
        decomp.loc[decomp['phi'] < -45, 'phi_regime'] = 'Risk-Off'
        
        # Magnitude of influence
        decomp['macro_influence'] = decomp['macro_score'].abs()
        decomp['liquidity_influence'] = decomp['liquidity_score'].abs()
        decomp['risk_influence'] = decomp['risk_score'].abs()
        decomp['equity_influence'] = decomp['equity_score'].abs()
        
        return decomp
    
    def compare_with_coherence_magnitude(self, vcf_magnitude_file: str) -> pd.DataFrame:
        """
        Compare your 4-pillar coherence with the magnitude from your earlier upload
        
        vcf_magnitude_file: vcf_results_full_ZSCORE.csv
        """
        vcf_mag = pd.read_csv(vcf_magnitude_file, parse_dates=['date'], index_col='date')
        
        # Align dates
        common = self.geometry.index.intersection(vcf_mag.index)
        
        comparison = pd.DataFrame(index=common)
        comparison['coherence_theta'] = self.geometry.loc[common, 'coherence_theta']
        comparison['coherence_phi'] = self.geometry.loc[common, 'coherence_phi']
        comparison['vcf_magnitude'] = vcf_mag.loc[common, 'vcf_magnitude']
        
        # Combined coherence
        comparison['combined_coherence'] = np.sqrt(
            comparison['coherence_theta']**2 + comparison['coherence_phi']**2
        )
        
        # Correlation
        corr = comparison['combined_coherence'].corr(comparison['vcf_magnitude'])
        
        print("=" * 70)
        print("COHERENCE vs MAGNITUDE COMPARISON")
        print("=" * 70)
        print(f"\nCorrelation: {corr:.3f}")
        print(f"\nSummary stats:")
        print(comparison.describe())
        
        return comparison


def run_full_4pillar_analysis(base_dir: str):
    """
    Complete analysis of your 4-pillar VCF structure
    
    base_dir: Path to your VCF-RESEARCH directory
    """
    # File paths
    geometry_file = os.path.join(base_dir, "geometry", "geometry_engine_full.csv")
    panel_file = os.path.join(base_dir, "data_clean", "normalized_panel_monthly.csv")
    registry_file = os.path.join(base_dir, "registry", "vcf_metric_registry.json")
    
    print("=" * 70)
    print("VCF 4-PILLAR INFLUENCE ANALYSIS")
    print("=" * 70)
    print()
    
    # Initialize analyzer
    analyzer = VCF_4Pillar_Analyzer(geometry_file, panel_file)
    
    # Get pillar composition
    print("\n" + "=" * 70)
    print("PILLAR COMPOSITION")
    print("=" * 70)
    pillar_metrics = analyzer.get_pillar_metrics(registry_file)
    for pillar, metrics in pillar_metrics.items():
        print(f"\n{pillar} ({len(metrics)} metrics):")
        for m in metrics:
            print(f"  - {m}")
    
    # Analyze which metrics drive each pillar
    print("\n" + "=" * 70)
    print("COMPUTING METRIC-LEVEL INFLUENCE")
    print("=" * 70)
    influence_df = analyzer.analyze_pillar_influence(pillar_metrics)
    
    # COVID crash analysis
    print("\n")
    covid_analysis = analyzer.find_covid_crash_drivers(influence_df)
    
    # Theta/Phi decomposition
    print("\n" + "=" * 70)
    print("THETA/PHI REGIME DECOMPOSITION")
    print("=" * 70)
    decomp = analyzer.theta_phi_decomposition()
    
    print("\nCurrent regimes (last 12 months):")
    recent = decomp.tail(12)
    print("\nTheta regimes:")
    print(recent['theta_regime'].value_counts())
    print("\nPhi regimes:")
    print(recent['phi_regime'].value_counts())
    
    # Latest state
    print("\n" + "=" * 70)
    print("CURRENT STATE (Most Recent)")
    print("=" * 70)
    latest = decomp.iloc[-1]
    print(f"\nDate: {decomp.index[-1].strftime('%B %Y')}")
    print(f"\nPillar Scores:")
    print(f"  Macro:     {latest['macro_score']:>7.2f}σ")
    print(f"  Liquidity: {latest['liquidity_score']:>7.2f}σ")
    print(f"  Risk:      {latest['risk_score']:>7.2f}σ")
    print(f"  Equity:    {latest['equity_score']:>7.2f}σ")
    
    print(f"\nRegimes:")
    print(f"  Theta: {latest['theta_regime']}")
    print(f"  Phi:   {latest['phi_regime']}")
    
    print(f"\nInfluence (absolute values):")
    print(f"  Macro:     {latest['macro_influence']:.2f}")
    print(f"  Liquidity: {latest['liquidity_influence']:.2f}")
    print(f"  Risk:      {latest['risk_influence']:.2f}")
    print(f"  Equity:    {latest['equity_influence']:.2f}")
    
    # Determine dominant pillar
    influences = {
        'Macro': latest['macro_influence'],
        'Liquidity': latest['liquidity_influence'],
        'Risk': latest['risk_influence'],
        'Equity': latest['equity_influence']
    }
    dominant_pillar = max(influences, key=influences.get)
    print(f"\n🎯 DOMINANT PILLAR: {dominant_pillar} ({influences[dominant_pillar]:.2f})")
    
    return {
        'analyzer': analyzer,
        'pillar_metrics': pillar_metrics,
        'influence_df': influence_df,
        'decomposition': decomp,
        'covid_analysis': covid_analysis
    }


# ==============================================================================
# EXAMPLE USAGE
# ==============================================================================

if __name__ == "__main__":
    """
    Run this with your VCF-RESEARCH directory
    """
    
    # If running in Google Colab
    # BASE_DIR = "/content/drive/MyDrive/VCF-RESEARCH"
    
    # If running locally, adjust path
    BASE_DIR = "./VCF-RESEARCH"  # Change to your path
    
    if os.path.exists(BASE_DIR):
        results = run_full_4pillar_analysis(BASE_DIR)
        
        # Save results
        output_dir = os.path.join(BASE_DIR, "influence_analysis")
        os.makedirs(output_dir, exist_ok=True)
        
        results['influence_df'].to_csv(
            os.path.join(output_dir, "pillar_influence_by_date.csv")
        )
        results['decomposition'].to_csv(
            os.path.join(output_dir, "theta_phi_decomposition.csv")
        )
        
        print("\n" + "=" * 70)
        print("✅ Analysis complete! Results saved to:")
        print(f"   {output_dir}/")
        print("=" * 70)
        
    else:
        print(f"❌ Directory not found: {BASE_DIR}")
        print("\nPlease update BASE_DIR to point to your VCF-RESEARCH folder")
        print("\nFor Google Colab:")
        print('  BASE_DIR = "/content/drive/MyDrive/VCF-RESEARCH"')
        print("\nFor local:")
        print('  BASE_DIR = "/path/to/your/VCF-RESEARCH"')
