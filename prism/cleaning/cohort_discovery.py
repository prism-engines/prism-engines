"""
PRISM Cohort Discovery Agent

Discovers structural cohorts from indicator data based on:
- Missingness topology (how gaps are distributed)
- Cadence (daily, weekly, monthly)
- Temporal overlap (for window viability, not clustering)

Cohorts are groups of indicators that can be meaningfully analyzed together
because they share compatible structure - NOT because they have similar values.

Usage:
    agent = CohortDiscoveryAgent(threshold=0.5)
    result = agent.run(df_wide)
    print(agent.explain())
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist


# =============================================================================
# Result Dataclasses
# =============================================================================

@dataclass
class CohortProfile:
    """Profile for a single cohort."""
    cohort_id: str
    indicators: List[str]
    n_indicators: int
    
    # Missingness
    mean_missing_rate: float      # Raw missing rate on global grid
    mean_missing_adjusted: float  # Cadence-adjusted missing rate
    
    # Cadence
    inferred_cadence: str         # "daily", "weekly", "monthly", "irregular"
    median_dt_days: float
    
    # Temporal overlap
    overlap_ratio: float          # How much indicators overlap in time
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    
    # Confidence
    confidence: float = 1.0       # Cohort assignment confidence


@dataclass
class CohortDiscoveryResult:
    """Result of cohort discovery."""
    cohorts: Dict[str, CohortProfile]
    n_cohorts: int
    n_indicators: int
    threshold_used: float
    features_used: List[str]
    
    # Per-indicator assignments
    indicator_to_cohort: Dict[str, str] = field(default_factory=dict)


# =============================================================================
# Feature Extraction
# =============================================================================

def compute_indicator_features(series: pd.Series, global_index: pd.DatetimeIndex) -> Dict[str, float]:
    """
    Compute 12 structural features for an indicator.
    
    A) Missingness Topology (features 1-4)
    B) Cadence (features 5-8)
    C) Temporal Overlap (features 9-12) - for gating only
    
    FIX: Uses business-day grid for daily series to avoid weekend/holiday inflation.
    """
    features = {}
    
    # First pass: compute raw cadence to decide reference grid
    valid_dates = series.dropna().index
    if len(valid_dates) > 1:
        dt_days = np.diff(valid_dates).astype('timedelta64[D]').astype(float)
        dt_days = dt_days[dt_days > 0]
        raw_median_dt = np.median(dt_days) if len(dt_days) > 0 else 1.0
    else:
        raw_median_dt = 1.0
    
    # FIX #1: Use business-day grid for daily series (median_dt <= 2)
    # This prevents weekend/holiday inflation of missing rates
    if raw_median_dt <= 2 and len(valid_dates) > 0:
        ref_index = pd.date_range(
            valid_dates.min(),
            valid_dates.max(),
            freq="B"  # Business days only
        )
        aligned = series.reindex(ref_index)
    else:
        aligned = series.reindex(global_index)
    
    valid_mask = ~aligned.isna()
    n_total = len(aligned)
    n_valid = valid_mask.sum()
    
    # === A) MISSINGNESS TOPOLOGY ===
    
    # 1. Missing rate (on global grid)
    features["missing_rate"] = 1.0 - (n_valid / n_total) if n_total > 0 else 1.0
    
    # 2-4. Gap analysis
    if n_valid > 1:
        valid_indices = np.where(valid_mask.values)[0]
        gaps = np.diff(valid_indices)
        gaps = gaps[gaps > 1] - 1  # Convert to gap lengths
        
        if len(gaps) > 0:
            max_gap = gaps.max()
            features["max_gap_ratio"] = max_gap / n_total
            features["gap_count_rate"] = len(gaps) / n_total
            
            # Gap length entropy
            gap_counts = np.bincount(gaps.astype(int))
            gap_probs = gap_counts[gap_counts > 0] / gap_counts.sum()
            features["gap_len_entropy"] = -np.sum(gap_probs * np.log(gap_probs + 1e-10))
        else:
            features["max_gap_ratio"] = 0.0
            features["gap_count_rate"] = 0.0
            features["gap_len_entropy"] = 0.0
    else:
        features["max_gap_ratio"] = 1.0
        features["gap_count_rate"] = 0.0
        features["gap_len_entropy"] = 0.0
    
    # === B) CADENCE ===
    
    if n_valid > 1:
        valid_dates = global_index[valid_mask]
        dt_days = np.diff(valid_dates).astype('timedelta64[D]').astype(float)
        dt_days = dt_days[dt_days > 0]
        
        if len(dt_days) > 0:
            # 5. Median inter-arrival time
            features["median_dt_days"] = np.median(dt_days)
            
            # 6. Cadence entropy (regularity)
            dt_counts = np.bincount(np.clip(dt_days.astype(int), 0, 365))
            dt_probs = dt_counts[dt_counts > 0] / dt_counts.sum()
            features["dt_entropy"] = -np.sum(dt_probs * np.log(dt_probs + 1e-10))
            
            # 7. Coefficient of variation
            features["dt_cv"] = np.std(dt_days) / (np.mean(dt_days) + 1e-10)
            
            # 8. Staleness rate - now measures LATE releases, not calendar sparsity
            # FIX #2: Changed from (dt > 1) to (dt > median * 1.5)
            # This captures irregular/late data, not routine weekends
            late_threshold = features["median_dt_days"] * 1.5
            features["staleness_rate"] = (dt_days > late_threshold).sum() / len(dt_days)
        else:
            features["median_dt_days"] = 1.0
            features["dt_entropy"] = 0.0
            features["dt_cv"] = 0.0
            features["staleness_rate"] = 0.0
    else:
        features["median_dt_days"] = 1.0
        features["dt_entropy"] = 0.0
        features["dt_cv"] = 0.0
        features["staleness_rate"] = 1.0
    
    # === C) TEMPORAL OVERLAP (for gating, not clustering) ===
    
    if n_valid > 0:
        valid_indices = np.where(valid_mask.values)[0]
        features["start_pos"] = valid_indices[0] / n_total
        features["end_pos"] = valid_indices[-1] / n_total
        features["coverage_ratio"] = (valid_indices[-1] - valid_indices[0] + 1) / n_total
        features["effective_obs_log"] = np.log1p(n_valid)
    else:
        features["start_pos"] = 0.0
        features["end_pos"] = 0.0
        features["coverage_ratio"] = 0.0
        features["effective_obs_log"] = 0.0
    
    return features


def compute_adjusted_missing_rate(missing_rate: float, median_dt: float) -> float:
    """
    Compute cadence-adjusted missing rate.
    
    A monthly series with 96% "missing" on a daily grid is actually
    ~0% missing on its own cadence.
    """
    expected_density = 1.0 / max(median_dt, 1.0)  # daily=1.0, monthly≈0.033
    actual_density = 1.0 - missing_rate
    
    if expected_density > 0:
        completeness = min(1.0, actual_density / expected_density)
        return 1.0 - completeness
    return missing_rate


def infer_cadence(median_dt: float) -> str:
    """Infer cadence label from median inter-arrival time."""
    if median_dt <= 1.5:
        return "daily"
    elif median_dt <= 8:
        return "weekly"
    elif median_dt <= 35:
        return "monthly"
    elif median_dt <= 100:
        return "quarterly"
    else:
        return "irregular"


# =============================================================================
# Clustering
# =============================================================================

def cluster_indicators(
    features_df: pd.DataFrame,
    threshold: Optional[float] = None,
    n_clusters: Optional[int] = None,
) -> Tuple[np.ndarray, float]:
    """
    Cluster indicators using hierarchical clustering on structural features.
    
    Only uses features 1-8 (missingness + cadence).
    Features 9-12 (temporal overlap) are for post-cohort gating.
    """
    # Select clustering features (NOT temporal overlap)
    # FIX #2: staleness_rate removed - was redundant with missing_rate
    clustering_features = [
        "missing_rate", "max_gap_ratio", "gap_count_rate", "gap_len_entropy",
        "median_dt_days", "dt_entropy", "dt_cv"
    ]
    
    # Feature weights
    # FIX #2: Removed staleness_rate - redundant with missing_rate
    weights = {
        # Missingness (55%)
        "missing_rate": 0.20,
        "max_gap_ratio": 0.15,
        "gap_count_rate": 0.10,
        "gap_len_entropy": 0.10,
        # Cadence (45%)
        "median_dt_days": 0.25,
        "dt_entropy": 0.10,
        "dt_cv": 0.10,
        # staleness_rate removed - was duplicating missing_rate signal
    }
    
    X = features_df[clustering_features].copy()
    
    # Robust scaling (median/IQR)
    for col in X.columns:
        median = X[col].median()
        iqr = X[col].quantile(0.75) - X[col].quantile(0.25)
        if iqr > 0:
            X[col] = (X[col] - median) / iqr
        else:
            X[col] = 0.0
    
    # Apply weights
    for col in X.columns:
        X[col] = X[col] * weights.get(col, 1.0)
    
    # Handle edge cases
    if len(X) == 1:
        return np.array([0]), 0.0
    
    # Hierarchical clustering
    Z = linkage(X.values, method='ward')
    
    # Determine threshold
    if n_clusters is not None:
        labels = fcluster(Z, n_clusters, criterion='maxclust') - 1
        threshold_used = 0.0
    else:
        max_dist = Z[:, 2].max()
        threshold_used = threshold if threshold is not None else max_dist * 0.5
        labels = fcluster(Z, threshold_used, criterion='distance') - 1
    
    return labels, threshold_used


# =============================================================================
# Cohort Discovery Agent
# =============================================================================

class CohortDiscoveryAgent:
    """
    Discovers structural cohorts from indicator data.
    
    Cohorts are groups of indicators with compatible structure:
    - Similar missingness patterns
    - Compatible cadence
    - Sufficient temporal overlap
    
    Usage:
        agent = CohortDiscoveryAgent(threshold=0.5)
        result = agent.run(df_wide)
        print(agent.explain())
    """
    
    def __init__(
        self,
        threshold: Optional[float] = None,
        n_clusters: Optional[int] = None,
    ):
        self.threshold = threshold
        self.n_clusters = n_clusters
        self._result: Optional[CohortDiscoveryResult] = None
        self._features_df: Optional[pd.DataFrame] = None
    
    def run(self, df_wide: pd.DataFrame) -> CohortDiscoveryResult:
        """
        Discover cohorts from wide-format DataFrame.
        
        Args:
            df_wide: DataFrame with DatetimeIndex, columns are indicators
            
        Returns:
            CohortDiscoveryResult with cohort profiles
        """
        global_index = df_wide.index
        indicators = list(df_wide.columns)
        
        # Compute features for each indicator
        features_list = []
        for ind in indicators:
            features = compute_indicator_features(df_wide[ind], global_index)
            features["indicator"] = ind
            features_list.append(features)
        
        self._features_df = pd.DataFrame(features_list).set_index("indicator")
        
        # Cluster indicators
        labels, threshold_used = cluster_indicators(
            self._features_df,
            threshold=self.threshold,
            n_clusters=self.n_clusters,
        )
        
        # Build cohort profiles
        cohorts = {}
        indicator_to_cohort = {}
        
        for cohort_id in np.unique(labels):
            mask = labels == cohort_id
            cohort_indicators = [ind for ind, m in zip(indicators, mask) if m]
            cohort_features = self._features_df.loc[cohort_indicators]
            
            # Compute cohort-level stats
            mean_missing = cohort_features["missing_rate"].mean()
            median_dt = cohort_features["median_dt_days"].mean()
            mean_missing_adj = compute_adjusted_missing_rate(mean_missing, median_dt)
            
            # Compute overlap
            start_positions = cohort_features["start_pos"].values
            end_positions = cohort_features["end_pos"].values
            
            # Overlap = intersection / union of temporal ranges
            latest_start = start_positions.max()
            earliest_end = end_positions.min()
            earliest_start = start_positions.min()
            latest_end = end_positions.max()
            
            intersection = max(0, earliest_end - latest_start)
            union = latest_end - earliest_start
            overlap_ratio = intersection / union if union > 0 else 0.0
            
            # Confidence based on cluster tightness
            # FIX #3: Add singleton penalty - singletons shouldn't be "perfectly confident"
            if len(cohort_indicators) > 1:
                # Use std of key features as inverse confidence
                feature_std = cohort_features[["missing_rate", "median_dt_days"]].std().mean()
                base_confidence = max(0.5, 1.0 - feature_std)
            else:
                base_confidence = 0.8  # Singletons start lower
            
            # Size penalty: singletons and small cohorts get reduced confidence
            size_factor = min(1.0, np.log1p(len(cohort_indicators)) / np.log1p(3))
            confidence = base_confidence * size_factor
            
            profile = CohortProfile(
                cohort_id=f"cohort_{cohort_id}",
                indicators=cohort_indicators,
                n_indicators=len(cohort_indicators),
                mean_missing_rate=mean_missing,
                mean_missing_adjusted=mean_missing_adj,
                inferred_cadence=infer_cadence(median_dt),
                median_dt_days=median_dt,
                overlap_ratio=overlap_ratio,
                confidence=confidence,
            )
            
            cohorts[f"cohort_{cohort_id}"] = profile
            for ind in cohort_indicators:
                indicator_to_cohort[ind] = f"cohort_{cohort_id}"
        
        self._result = CohortDiscoveryResult(
            cohorts=cohorts,
            n_cohorts=len(cohorts),
            n_indicators=len(indicators),
            threshold_used=threshold_used,
            features_used=list(self._features_df.columns),
            indicator_to_cohort=indicator_to_cohort,
        )
        
        return self._result
    
    def explain(self) -> str:
        """Return human-readable explanation of discovered cohorts."""
        if self._result is None:
            return "No cohort discovery has been run yet."
        
        lines = [
            "Cohort Discovery Results",
            "=" * 50,
            f"Total indicators: {self._result.n_indicators}",
            f"Cohorts found: {self._result.n_cohorts}",
            f"Threshold used: {self._result.threshold_used:.3f}",
            "",
        ]
        
        for cohort_id, profile in self._result.cohorts.items():
            lines.append(f"{cohort_id}:")
            lines.append(f"  Indicators: {', '.join(profile.indicators)}")
            lines.append(f"  Cadence: {profile.inferred_cadence} (median dt: {profile.median_dt_days:.1f} days)")
            lines.append(f"  Missing (raw): {profile.mean_missing_rate:.1%}")
            lines.append(f"  Missing (adj): {profile.mean_missing_adjusted:.1%}")
            lines.append(f"  Overlap: {profile.overlap_ratio:.1%}")
            lines.append(f"  Confidence: {profile.confidence:.2f}")
            lines.append("")
        
        return "\n".join(lines)


# =============================================================================
# Convenience Function
# =============================================================================

def discover_cohorts(
    df_wide: pd.DataFrame,
    threshold: float = 0.5,
) -> CohortDiscoveryResult:
    """Convenience function to discover cohorts."""
    agent = CohortDiscoveryAgent(threshold=threshold)
    return agent.run(df_wide)
