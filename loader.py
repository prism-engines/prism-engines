"""
Lens Loader - Bulletproof Edition
==================================

Handles all the Python import nonsense so you can just run lenses.

Usage:
    # Import the loader
    from loader import run_lens, run_all_lenses, quick_analysis

    # Run a single lens
    result = run_lens('magnitude', panel_clean)

    # Or run all lenses
    results = run_all_lenses(panel_clean)

    # Quick consensus analysis
    consensus = quick_analysis(panel_clean)
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================================
# SETUP PATHS
# ============================================================================

def _find_engine_root():
    """Find the engine root directory relative to this file."""
    # Start from this file's location
    if '__file__' in dir():
        script_dir = Path(__file__).resolve().parent
    else:
        script_dir = Path('.').resolve()

    # Check possible locations relative to script
    candidates = [
        script_dir,  # loader.py is in root
        script_dir.parent,  # loader.py is in a subdirectory
        Path('.').resolve(),  # Current working directory
    ]

    for candidate in candidates:
        if (candidate / '05_engine' / 'lenses').exists():
            return candidate

    return None


ENGINE_ROOT = _find_engine_root()

if ENGINE_ROOT is None:
    print("Warning: Could not find engine folder!")
    print("    Set ENGINE_ROOT manually or ensure you're in the project directory")
else:
    print(f"ENGINE_ROOT = {ENGINE_ROOT}")
    sys.path.insert(0, str(ENGINE_ROOT))

LENSES_PATH = ENGINE_ROOT / '05_engine' / 'lenses' if ENGINE_ROOT else None


# ============================================================================
# STANDALONE LENS IMPLEMENTATIONS
# ============================================================================
# These work without any imports - completely self-contained

class BaseLens:
    """Base class for all lenses."""
    name = "base"
    
    def analyze(self, panel: pd.DataFrame) -> dict:
        raise NotImplementedError
    
    def top_indicators(self, result: dict, n: int = 10) -> list:
        if 'importance' in result:
            imp = result['importance']
            if isinstance(imp, pd.Series):
                return list(imp.sort_values(ascending=False).head(n).items())
            elif isinstance(imp, dict):
                sorted_imp = sorted(imp.items(), key=lambda x: x[1], reverse=True)
                return sorted_imp[:n]
        return []


class MagnitudeLens(BaseLens):
    """Measures indicator importance by L2 norm (total magnitude of movement)."""
    name = "magnitude"
    
    def analyze(self, panel: pd.DataFrame) -> dict:
        # Normalize each column
        normalized = (panel - panel.mean()) / panel.std()
        
        # L2 norm for each column
        magnitude = np.sqrt((normalized ** 2).sum())
        
        # Also compute recent vs historical magnitude
        mid = len(panel) // 2
        recent_mag = np.sqrt((normalized.iloc[mid:] ** 2).sum())
        historical_mag = np.sqrt((normalized.iloc[:mid] ** 2).sum())
        
        return {
            'importance': magnitude,
            'magnitude': magnitude,
            'recent_magnitude': recent_mag,
            'historical_magnitude': historical_mag,
            'magnitude_change': (recent_mag - historical_mag) / historical_mag,
        }


class PCALens(BaseLens):
    """Uses PCA to find indicators that explain the most variance."""
    name = "pca"
    
    def analyze(self, panel: pd.DataFrame, n_components: int = 5) -> dict:
        # Standardize
        X = (panel - panel.mean()) / panel.std()
        X = X.fillna(0)
        
        # Manual PCA via SVD
        U, S, Vt = np.linalg.svd(X.values, full_matrices=False)
        
        # Explained variance
        explained_var = (S ** 2) / (len(X) - 1)
        explained_var_ratio = explained_var / explained_var.sum()
        
        # Loadings (correlation of each variable with each PC)
        loadings = Vt[:n_components].T * S[:n_components]
        
        # Importance = sum of absolute loadings across top PCs
        importance = pd.Series(
            np.abs(loadings).sum(axis=1),
            index=panel.columns
        )
        
        return {
            'importance': importance,
            'explained_variance_ratio': explained_var_ratio[:n_components],
            'cumulative_variance': np.cumsum(explained_var_ratio)[:n_components],
            'loadings': pd.DataFrame(loadings, index=panel.columns, 
                                      columns=[f'PC{i+1}' for i in range(n_components)]),
            'n_components_for_90pct': np.searchsorted(np.cumsum(explained_var_ratio), 0.9) + 1,
        }


class InfluenceLens(BaseLens):
    """Measures influence as volatility × deviation from mean."""
    name = "influence"
    
    def analyze(self, panel: pd.DataFrame, window: int = 20) -> dict:
        # Rolling volatility
        volatility = panel.rolling(window=window).std()
        
        # Deviation from rolling mean
        rolling_mean = panel.rolling(window=window).mean()
        deviation = np.abs(panel - rolling_mean)
        
        # Influence = volatility × deviation
        influence = (volatility * deviation).mean()
        
        # Normalize to 0-1
        importance = (influence - influence.min()) / (influence.max() - influence.min())
        
        return {
            'importance': importance,
            'influence': influence,
            'avg_volatility': volatility.mean(),
            'avg_deviation': deviation.mean(),
        }


class ClusteringLens(BaseLens):
    """Groups indicators by correlation, identifies cluster representatives."""
    name = "clustering"
    
    def analyze(self, panel: pd.DataFrame, n_clusters: int = None) -> dict:
        # Correlation matrix
        corr = panel.corr()
        
        # Simple clustering: hierarchical-ish via correlation
        if n_clusters is None:
            n_clusters = max(2, len(panel.columns) // 4)
        
        # Use correlation as distance
        dist = 1 - np.abs(corr.values)
        
        # Simple k-means-ish on correlation structure
        np.random.seed(42)
        
        # Initialize cluster centers randomly
        centers_idx = np.random.choice(len(panel.columns), n_clusters, replace=False)
        
        # Assign to nearest center (by correlation distance)
        labels = np.zeros(len(panel.columns), dtype=int)
        for iteration in range(10):
            # Assign
            for i in range(len(panel.columns)):
                distances = [dist[i, c] for c in centers_idx]
                labels[i] = np.argmin(distances)
            
            # Update centers (most central in each cluster)
            for k in range(n_clusters):
                cluster_members = np.where(labels == k)[0]
                if len(cluster_members) > 0:
                    # Find most central (lowest avg distance to others in cluster)
                    avg_dist = [dist[m, cluster_members].mean() for m in cluster_members]
                    centers_idx[k] = cluster_members[np.argmin(avg_dist)]
        
        # Importance = how representative of cluster (inverse avg distance)
        importance_vals = []
        for i in range(len(panel.columns)):
            cluster = labels[i]
            cluster_members = np.where(labels == cluster)[0]
            avg_dist = dist[i, cluster_members].mean()
            importance_vals.append(1 / (1 + avg_dist))
        
        importance = pd.Series(importance_vals, index=panel.columns)
        
        return {
            'importance': importance,
            'labels': pd.Series(labels, index=panel.columns),
            'n_clusters': n_clusters,
            'cluster_centers': [panel.columns[c] for c in centers_idx],
            'correlation_matrix': corr,
        }


class DecompositionLens(BaseLens):
    """Decomposes time series into trend, seasonal, and residual components."""
    name = "decomposition"
    
    def analyze(self, panel: pd.DataFrame, period: int = 252) -> dict:
        results = {}
        importance_vals = []
        
        for col in panel.columns:
            series = panel[col].dropna()
            
            if len(series) < period * 2:
                # Not enough data for decomposition
                importance_vals.append(0)
                continue
            
            # Simple moving average trend
            trend = series.rolling(window=period, center=True).mean()
            
            # Detrended
            detrended = series - trend
            
            # Seasonal: average by position in period
            seasonal = pd.Series(index=series.index, dtype=float)
            for i in range(period):
                mask = np.arange(len(series)) % period == i
                seasonal.iloc[mask] = detrended.iloc[mask].mean()
            
            # Residual
            residual = detrended - seasonal
            
            # Importance: ratio of trend variance to total variance
            # Higher = more predictable/structured
            trend_var = trend.var()
            total_var = series.var()
            
            importance_vals.append(trend_var / total_var if total_var > 0 else 0)
            
            results[col] = {
                'trend_strength': trend_var / total_var if total_var > 0 else 0,
                'seasonal_strength': seasonal.var() / total_var if total_var > 0 else 0,
                'residual_strength': residual.var() / total_var if total_var > 0 else 0,
            }
        
        importance = pd.Series(importance_vals, index=panel.columns)
        
        return {
            'importance': importance,
            'decomposition': results,
        }


# ============================================================================
# LENS REGISTRY
# ============================================================================

BUILTIN_LENSES = {
    'magnitude': MagnitudeLens,
    'pca': PCALens,
    'influence': InfluenceLens,
    'clustering': ClusteringLens,
    'decomposition': DecompositionLens,
}

def get_available_lenses():
    """List all available lenses."""
    available = list(BUILTIN_LENSES.keys())

    # Check for additional lenses in the lenses folder
    if LENSES_PATH and LENSES_PATH.exists():
        for f in LENSES_PATH.iterdir():
            if f.name.endswith('_lens.py') and f.name != 'base_lens.py':
                name = f.name.replace('_lens.py', '')
                if name not in available:
                    available.append(name + ' (file)')

    return available


def load_lens(name: str):
    """
    Load a lens by name.

    First tries builtin lenses, then tries to load from file.
    """
    # Try builtin first
    if name in BUILTIN_LENSES:
        return BUILTIN_LENSES[name]()

    # Try loading from file
    if LENSES_PATH:
        lens_file = LENSES_PATH / f'{name}_lens.py'
        if lens_file.exists():
            # Read the file and extract the class
            content = lens_file.read_text()
            
            # Create a namespace and exec
            namespace = {
                'pd': pd,
                'np': np,
                'BaseLens': BaseLens,
            }
            
            # Remove relative imports
            lines = content.split('\n')
            cleaned_lines = []
            for line in lines:
                if line.strip().startswith('from .'):
                    # Skip relative imports
                    continue
                if line.strip().startswith('from base_lens'):
                    continue
                cleaned_lines.append(line)
            
            cleaned_content = '\n'.join(cleaned_lines)
            
            try:
                exec(cleaned_content, namespace)
                
                # Find the lens class
                for key, val in namespace.items():
                    if isinstance(val, type) and key.endswith('Lens') and key != 'BaseLens':
                        return val()
            except Exception as e:
                print(f"  Warning: Could not load {name} from file: {e}")
    
    raise ValueError(f"Unknown lens: {name}")


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def run_lens(name: str, panel: pd.DataFrame) -> dict:
    """Run a single lens and return results."""
    lens = load_lens(name)
    return lens.analyze(panel)


def run_all_lenses(panel: pd.DataFrame, lenses: list = None) -> dict:
    """Run multiple lenses and return all results."""
    if lenses is None:
        lenses = list(BUILTIN_LENSES.keys())
    
    results = {}
    for name in lenses:
        try:
            print(f"  Running {name}...", end=" ")
            results[name] = run_lens(name, panel)
            print("✓")
        except Exception as e:
            print(f"✗ ({e})")
    
    return results


def compute_consensus(results: dict) -> pd.DataFrame:
    """Compute consensus rankings from multiple lens results."""
    rankings = {}
    
    for lens_name, result in results.items():
        if 'importance' in result:
            imp = result['importance']
            if isinstance(imp, pd.Series):
                rankings[lens_name] = imp.rank(ascending=False)
    
    if not rankings:
        return pd.DataFrame()
    
    rank_df = pd.DataFrame(rankings)
    rank_df['avg_rank'] = rank_df.mean(axis=1)
    rank_df['std_rank'] = rank_df.std(axis=1)
    
    return rank_df.sort_values('avg_rank')


def quick_analysis(panel: pd.DataFrame) -> pd.DataFrame:
    """
    One-liner to run all lenses and get consensus.
    
    Usage:
        consensus = quick_analysis(panel_clean)
        print(consensus.head(10))
    """
    print("Running analysis...")
    results = run_all_lenses(panel)
    consensus = compute_consensus(results)
    print(f"\n✓ Done! Top indicator: {consensus.index[0]}")
    return consensus


def load_panel_from_sql(
    tickers: list = None,
    start_date: str = None,
    end_date: str = None
) -> pd.DataFrame:
    """
    Load panel data from SQL database.
    
    Args:
        tickers: List of tickers to load (optional, defaults to all)
        start_date: Start date filter (optional)
        end_date: End date filter (optional)
    
    Returns:
        DataFrame with date index and ticker columns
    
    Usage:
        panel = load_panel_from_sql()
        panel = load_panel_from_sql(tickers=['spy', 'qqq', 'gld'])
    """
    try:
        from data.sql import SQLDataManager
        db = SQLDataManager()
        panel = db.load_cleaned_panel(
            tickers=tickers,
            start_date=start_date,
            end_date=end_date
        )
        if panel.empty:
            # Try loading from stored panel
            panel = db.load_panel('master_panel')
        return panel
    except ImportError as e:
        print(f"Warning: Could not import SQL module: {e}")
        return pd.DataFrame()
    except Exception as e:
        print(f"Warning: Could not load from SQL: {e}")
        return pd.DataFrame()


def load_panel_from_csv(path: str = None) -> pd.DataFrame:
    """
    Load panel data from CSV file.
    
    Args:
        path: Path to CSV file (optional, defaults to data/raw/master_panel.csv)
    
    Returns:
        DataFrame with date index
    
    Usage:
        panel = load_panel_from_csv()
        panel = load_panel_from_csv('data/raw/custom_panel.csv')
    """
    if path is None:
        if ENGINE_ROOT:
            path = ENGINE_ROOT / 'data' / 'raw' / 'master_panel.csv'
        else:
            path = Path('data/raw/master_panel.csv')
    
    path = Path(path)
    if not path.exists():
        print(f"Warning: File not found: {path}")
        return pd.DataFrame()
    
    return pd.read_csv(path, index_col=0, parse_dates=True)


def load_panel(use_sql: bool = True, **kwargs) -> pd.DataFrame:
    """
    Load panel data, trying SQL first then falling back to CSV.
    
    Args:
        use_sql: Try to load from SQL first
        **kwargs: Additional arguments passed to load functions
    
    Returns:
        DataFrame with panel data
    
    Usage:
        panel = load_panel()                    # Try SQL, fall back to CSV
        panel = load_panel(use_sql=False)       # Load from CSV only
        panel = load_panel(tickers=['spy'])     # Load specific tickers from SQL
    """
    panel = pd.DataFrame()
    
    if use_sql:
        panel = load_panel_from_sql(**kwargs)
    
    if panel.empty:
        csv_path = kwargs.pop('path', None) if kwargs else None
        panel = load_panel_from_csv(csv_path)
    
    return panel


# ============================================================================
# PRINT STATUS
# ============================================================================

print(f"✓ Loaded {len(BUILTIN_LENSES)} builtin lenses: {list(BUILTIN_LENSES.keys())}")
print()
print("Usage:")
print("  panel = load_panel()                    # Load data (SQL or CSV)")
print("  panel = load_panel_from_sql()           # Load from SQL only")
print("  panel = load_panel_from_csv()           # Load from CSV only")
print("  result = run_lens('magnitude', panel)")
print("  results = run_all_lenses(panel)")
print("  consensus = quick_analysis(panel)")
