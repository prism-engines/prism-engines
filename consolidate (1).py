"""
PRISM Engine - Data Consolidation
=================================

This module handles:
1. Loading raw indicator data
2. Grouping indicators into categories
3. Weighting indicators WITHIN each category (algorithmic, no human bias)
4. Weighting categories AGAINST each other (algorithmic)

The goal: REMOVE human judgment from "which indicators matter"

WEIGHTING METHODS:
    equal    - All indicators weighted equally (baseline)
    variance - Stable indicators get more weight
    entropy  - High information content = more weight
    pca      - Weight by contribution to first principal component
    auto     - Try all methods, pick best for each category
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import yaml


# ============================================================
# INDICATOR CATEGORIES
# ============================================================
# These define what TYPE of indicator each one is.
# The categories are fixed, but indicators within them can vary.

DEFAULT_CATEGORIES = {
    "monetary": {
        "description": "Money supply, Fed policy, interest rates",
        "indicators": ["M2SL", "WALCL", "DGS2", "DGS10", "FEDFUNDS", "T10Y2Y"]
    },
    "employment": {
        "description": "Jobs, unemployment, labor market",
        "indicators": ["PAYEMS", "UNRATE", "ICSA", "JOLTS", "AWHAETP"]
    },
    "housing": {
        "description": "Housing starts, permits, prices",
        "indicators": ["HOUST", "PERMIT", "CSUSHPISA", "MORTGAGE30US"]
    },
    "inflation": {
        "description": "Price indices",
        "indicators": ["CPIAUCSL", "CPILFESL", "PCEPI", "PPIFIS"]
    },
    "sentiment": {
        "description": "Market sentiment, financial conditions",
        "indicators": ["VIX", "NFCI", "UMCSENT", "AAII_BULL"]
    },
    "real_economy": {
        "description": "GDP, production, consumption",
        "indicators": ["GDP", "INDPRO", "RSXFS", "DGORDER"]
    },
    "markets": {
        "description": "Market indices and sectors",
        "indicators": ["SP500", "DXY", "AGG", "XLU", "XLF", "XLE"]
    }
}


# ============================================================
# LOADING
# ============================================================

def load_and_consolidate(
    data_dir: Path,
    weight_method: str = "auto",
    focus_category: Optional[str] = None,
    since_year: Optional[int] = None,
    categories_file: Optional[Path] = None
) -> Tuple[pd.DataFrame, Dict]:
    """
    Load data and organize into weighted categories.
    
    Args:
        data_dir: Path to clean data folder
        weight_method: How to weight indicators (equal, variance, entropy, pca, auto)
        focus_category: If set, only load this category
        since_year: Filter data to this year onward
        categories_file: Custom categories YAML (optional)
    
    Returns:
        (consolidated_df, categories_dict)
    """
    
    # Load category definitions
    if categories_file and categories_file.exists():
        with open(categories_file) as f:
            categories = yaml.safe_load(f)
    else:
        categories = DEFAULT_CATEGORIES.copy()
    
    # Filter to focus category if specified
    if focus_category:
        if focus_category not in categories:
            raise ValueError(f"Unknown category: {focus_category}. Options: {list(categories.keys())}")
        categories = {focus_category: categories[focus_category]}
    
    # Load data
    data_file = find_data_file(data_dir)
    print(f"  Loading: {data_file.name}")
    
    df = pd.read_csv(data_file, index_col=0, parse_dates=True)
    
    # Filter by date
    if since_year:
        df = df[df.index.year >= since_year]
        print(f"  Filtered to {since_year}+: {len(df)} rows")
    
    # Match available indicators to categories
    available = set(df.columns)
    matched_categories = {}
    
    for cat_name, cat_info in categories.items():
        expected = set(cat_info["indicators"])
        found = expected & available
        
        if found:
            matched_categories[cat_name] = {
                "description": cat_info["description"],
                "indicators": list(found),
                "missing": list(expected - found)
            }
            print(f"  {cat_name}: {len(found)}/{len(expected)} indicators")
    
    if not matched_categories:
        raise ValueError("No indicators matched any category!")
    
    # Calculate weights for each category
    print(f"  Weighting method: {weight_method}")
    
    for cat_name, cat_info in matched_categories.items():
        indicators = cat_info["indicators"]
        cat_df = df[indicators].dropna()
        
        if weight_method == "auto":
            weights = auto_weight(cat_df)
        elif weight_method == "equal":
            weights = equal_weight(cat_df)
        elif weight_method == "variance":
            weights = variance_weight(cat_df)
        elif weight_method == "entropy":
            weights = entropy_weight(cat_df)
        elif weight_method == "pca":
            weights = pca_weight(cat_df)
        else:
            weights = equal_weight(cat_df)
        
        matched_categories[cat_name]["weights"] = weights
    
    return df, matched_categories


def find_data_file(data_dir: Path) -> Path:
    """Find the data file in the directory."""
    candidates = [
        "prism_data_clean.csv",
        "clean_data.csv",
        "data.csv",
        "indicators.csv",
        "consolidated.csv"
    ]
    
    for name in candidates:
        path = data_dir / name
        if path.exists():
            return path
    
    # Try any CSV
    csvs = list(data_dir.glob("*.csv"))
    if csvs:
        return csvs[0]
    
    raise FileNotFoundError(f"No data file found in {data_dir}")


# ============================================================
# WEIGHTING METHODS
# ============================================================
# All methods return a dict: {indicator_name: weight}
# Weights are normalized to sum to 1

def equal_weight(df: pd.DataFrame) -> Dict[str, float]:
    """All indicators weighted equally."""
    n = len(df.columns)
    return {col: 1.0 / n for col in df.columns}


def variance_weight(df: pd.DataFrame) -> Dict[str, float]:
    """
    Inverse variance weighting.
    Stable (low variance) indicators get MORE weight.
    Intuition: Noisy indicators are less reliable.
    """
    # Normalize first to make variances comparable
    normalized = (df - df.mean()) / df.std()
    variances = normalized.var()
    
    # Inverse variance (add small epsilon to avoid division by zero)
    inv_var = 1.0 / (variances + 1e-10)
    
    # Normalize to sum to 1
    weights = inv_var / inv_var.sum()
    
    return weights.to_dict()


def entropy_weight(df: pd.DataFrame) -> Dict[str, float]:
    """
    Information entropy weighting.
    High entropy (more information) = MORE weight.
    Intuition: Indicators that vary meaningfully carry more signal.
    """
    weights = {}
    
    for col in df.columns:
        series = df[col].dropna()
        
        # Discretize into bins for entropy calculation
        n_bins = min(20, len(series) // 10)
        if n_bins < 2:
            weights[col] = 1.0
            continue
        
        hist, _ = np.histogram(series, bins=n_bins, density=True)
        hist = hist + 1e-10  # Avoid log(0)
        hist = hist / hist.sum()  # Normalize
        
        # Shannon entropy
        entropy = -np.sum(hist * np.log2(hist))
        weights[col] = entropy
    
    # Normalize
    total = sum(weights.values())
    return {k: v / total for k, v in weights.items()}


def pca_weight(df: pd.DataFrame) -> Dict[str, float]:
    """
    Weight by first principal component loading.
    Indicators that contribute most to the main pattern get MORE weight.
    Intuition: What's driving the category?
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    
    # Drop NaN rows for PCA
    clean_df = df.dropna()
    
    if len(clean_df) < 10:
        return equal_weight(df)
    
    # Standardize
    scaler = StandardScaler()
    scaled = scaler.fit_transform(clean_df)
    
    # PCA
    pca = PCA(n_components=1)
    pca.fit(scaled)
    
    # Absolute loadings (direction doesn't matter for importance)
    loadings = np.abs(pca.components_[0])
    
    # Normalize
    weights = loadings / loadings.sum()
    
    return dict(zip(df.columns, weights))


def redundancy_weight(df: pd.DataFrame) -> Dict[str, float]:
    """
    Inverse correlation weighting.
    Low correlation with peers = MORE weight (unique information).
    Intuition: Don't double-count redundant indicators.
    """
    corr = df.corr().abs()
    
    # Mean correlation with all other indicators
    np.fill_diagonal(corr.values, np.nan)
    mean_corr = corr.mean(axis=1)
    
    # Inverse correlation (low = good = high weight)
    inv_corr = 1.0 / (mean_corr + 0.1)  # Add 0.1 to avoid explosion for uncorrelated
    
    # Normalize
    weights = inv_corr / inv_corr.sum()
    
    return weights.to_dict()


def auto_weight(df: pd.DataFrame) -> Dict[str, float]:
    """
    Automatically select the best weighting method for this data.
    
    Logic:
    - If high multicollinearity → use redundancy weighting
    - If some indicators very noisy → use variance weighting
    - Otherwise → use PCA weighting
    """
    corr = df.corr().abs()
    np.fill_diagonal(corr.values, np.nan)
    mean_corr = np.nanmean(corr.values)
    
    variances = df.std() / df.mean().abs()  # Coefficient of variation
    cv_range = variances.max() / (variances.min() + 1e-10)
    
    if mean_corr > 0.7:
        # High correlation → reduce redundancy
        method = "redundancy"
        weights = redundancy_weight(df)
    elif cv_range > 5:
        # Very different noise levels → variance weight
        method = "variance"
        weights = variance_weight(df)
    else:
        # Default to PCA
        method = "pca"
        weights = pca_weight(df)
    
    return weights


# ============================================================
# CATEGORY CONSOLIDATION
# ============================================================

def consolidate_category(df: pd.DataFrame, weights: Dict[str, float]) -> pd.Series:
    """
    Combine multiple indicators into a single category score.
    
    Args:
        df: DataFrame with indicator columns
        weights: Dict of {indicator: weight}
    
    Returns:
        Single weighted series representing the category
    """
    available = [col for col in weights.keys() if col in df.columns]
    
    if not available:
        raise ValueError("No weighted indicators found in data")
    
    # Normalize each indicator to z-scores (comparable scale)
    normalized = pd.DataFrame()
    for col in available:
        normalized[col] = (df[col] - df[col].mean()) / df[col].std()
    
    # Weighted sum
    result = pd.Series(0.0, index=df.index)
    for col in available:
        result += normalized[col] * weights[col]
    
    return result


def build_consolidated_view(
    df: pd.DataFrame, 
    categories: Dict
) -> pd.DataFrame:
    """
    Build a consolidated DataFrame with one column per category.
    
    This is a SIMPLIFIED view of the data:
    - 74 indicators → 7 category scores
    - Easier to see relationships
    - Weights are already applied
    """
    result = pd.DataFrame(index=df.index)
    
    for cat_name, cat_info in categories.items():
        indicators = cat_info["indicators"]
        weights = cat_info["weights"]
        
        cat_df = df[[col for col in indicators if col in df.columns]]
        result[cat_name] = consolidate_category(cat_df, weights)
    
    return result.dropna()


# ============================================================
# CROSS-CATEGORY WEIGHTING
# ============================================================

def weight_categories(
    category_scores: pd.DataFrame,
    method: str = "auto"
) -> Dict[str, float]:
    """
    Weight categories against each other.
    
    This determines: How much does 'monetary' matter vs 'housing' vs etc.
    
    Same methods as indicator weighting - the hierarchy is consistent.
    """
    if method == "equal":
        return equal_weight(category_scores)
    elif method == "variance":
        return variance_weight(category_scores)
    elif method == "entropy":
        return entropy_weight(category_scores)
    elif method == "pca":
        return pca_weight(category_scores)
    elif method == "auto":
        return auto_weight(category_scores)
    else:
        return equal_weight(category_scores)
