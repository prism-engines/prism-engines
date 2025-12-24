"""
PRISM Cohort Discovery Agent

Discovers structural cohorts from data — not metadata.
Cohorts are sets of indicators with compatible structural properties,
such that running engines on them jointly does not violate assumptions.

Cohort Features:
    Clustering (9 features): Missingness topology + Cadence
    Gating (4 features): Temporal overlap — used post-cohort, not for clustering

Pipeline Position:
    AUDIT → COHORT_DISCOVERY → ENGINE_RUN (per cohort) → GEOMETRY (per cohort)

This agent is read-only. No data mutation. Only routing changes.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.stats import entropy

logger = logging.getLogger(__name__)


# =============================================================================
# Feature Weights
# =============================================================================

# Features used for CLUSTERING (structural compatibility)
CLUSTERING_WEIGHTS = {
    # Missingness topology (55%) — engines die on alignment collapse
    "missing_rate_adjusted": 0.18,  # Cadence-adjusted
    "max_gap_ratio": 0.15,
    "gap_count_rate": 0.12,
    "gap_len_entropy": 0.10,
    
    # Cadence (45%) — determines if series can be analyzed together
    "median_dt_days": 0.15,
    "dt_entropy": 0.10,
    "dt_cv": 0.10,
    "staleness_rate": 0.10,
}

# Features computed but NOT used for clustering — used for post-cohort gating
GATING_FEATURES = ["start_pos", "end_pos", "coverage_ratio", "effective_obs_log"]

CLUSTERING_ORDER = list(CLUSTERING_WEIGHTS.keys())
ALL_FEATURES = CLUSTERING_ORDER + GATING_FEATURES


# =============================================================================
# Result Dataclasses
# =============================================================================

@dataclass
class IndicatorFeatures:
    """Structural features for a single indicator."""
    indicator_id: str
    
    # Missingness topology (for clustering)
    missing_rate: float = 0.0           # Raw missing rate
    missing_rate_adjusted: float = 0.0  # Cadence-adjusted missing rate
    max_gap_ratio: float = 0.0
    gap_count_rate: float = 0.0
    gap_len_entropy: float = 0.0
    
    # Cadence (for clustering)
    median_dt_days: float = 1.0
    dt_entropy: float = 0.0
    dt_cv: float = 0.0
    staleness_rate: float = 0.0
    
    # Temporal overlap (for GATING, not clustering)
    start_pos: float = 0.0
    end_pos: float = 1.0
    coverage_ratio: float = 1.0
    effective_obs_log: float = 0.0
    
    def to_clustering_vector(self) -> np.ndarray:
        """Return only clustering features as array."""
        return np.array([getattr(self, f) for f in CLUSTERING_ORDER])
    
    def to_dict(self) -> Dict[str, float]:
        """Return all features as dict."""
        return {f: getattr(self, f) for f in ALL_FEATURES}


@dataclass
class CohortProfile:
    """Profile for a discovered cohort."""
    cohort_id: str
    indicators: List[str]
    
    # Aggregate stats
    mean_missing_rate: float = 0.0
    mean_missing_adjusted: float = 0.0  # Cadence-adjusted
    mean_density: float = 0.0
    inferred_cadence: str = "unknown"  # "daily", "weekly", "monthly", "irregular"
    
    # Quality metrics (per-cohort, not global)
    confidence: float = 0.0
    internal_distance: float = 0.0  # How tight is the cluster
    n_indicators: int = 0
    
    # Gating info (for engine routing)
    earliest_start: float = 0.0
    latest_end: float = 1.0
    overlap_ratio: float = 1.0  # How much do indicators overlap?
    
    def __post_init__(self):
        self.n_indicators = len(self.indicators)


@dataclass
class CohortDiscoveryResult:
    """Result from the Cohort Discovery Agent."""
    cohorts: Dict[str, CohortProfile]
    assignments: Dict[str, str]  # indicator_id -> cohort_id
    features: Dict[str, IndicatorFeatures]
    
    # Metadata
    n_cohorts: int = 0
    n_indicators: int = 0
    linkage_threshold: float = 0.0
    
    # No global confidence — per-cohort only
    warnings: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        self.n_cohorts = len(self.cohorts)
        self.n_indicators = len(self.assignments)
    
    def get_cohort_indicators(self, cohort_id: str) -> List[str]:
        """Get indicators for a specific cohort."""
        return self.cohorts.get(cohort_id, CohortProfile(cohort_id, [])).indicators


# =============================================================================
# Cadence-Adjusted Missingness
# =============================================================================

def compute_adjusted_missing_rate(missing_rate: float, median_dt: float) -> float:
    """
    Normalize missingness by inferred cadence.
    
    Monthly indicator with 96% 'missing' on daily grid 
    is actually ~0% missing on its own cadence.
    
    Args:
        missing_rate: Raw missing rate on global grid
        median_dt: Median inter-arrival time in days
    
    Returns:
        Cadence-adjusted missing rate [0, 1]
    """
    if median_dt <= 0:
        return missing_rate
    
    # Expected density for this cadence
    # daily = 1.0, weekly = 0.14, monthly = 0.033
    expected_density = 1.0 / median_dt
    
    # Actual density (inverse of missing rate)
    actual_density = 1.0 - missing_rate
    
    # How well does it hit its own expected cadence?
    # If actual >= expected, it's complete for its cadence
    if expected_density > 0:
        completeness = min(1.0, actual_density / expected_density)
        return 1.0 - completeness
    
    return missing_rate


# =============================================================================
# Feature Computation
# =============================================================================

def compute_indicator_features(
    series: pd.Series,
    panel_min_date: pd.Timestamp,
    panel_max_date: pd.Timestamp,
    panel_span_days: int,
) -> IndicatorFeatures:
    """
    Compute all structural features for a single indicator.
    
    Args:
        series: Single indicator series (may contain NaN)
        panel_min_date: Earliest date in full panel
        panel_max_date: Latest date in full panel
        panel_span_days: Total days in panel span
    
    Returns:
        IndicatorFeatures with clustering + gating features
    """
    indicator_id = series.name if series.name else "unknown"
    features = IndicatorFeatures(indicator_id=indicator_id)
    
    total_days = len(series)
    if total_days == 0:
        return features
    
    # Valid observations
    valid_mask = ~series.isna()
    valid_count = valid_mask.sum()
    
    if valid_count == 0:
        features.missing_rate = 1.0
        features.missing_rate_adjusted = 1.0
        features.max_gap_ratio = 1.0
        return features
    
    # ===================
    # A) Missingness Topology
    # ===================
    
    # 1. missing_rate (raw)
    features.missing_rate = 1 - (valid_count / total_days)
    
    # Find gaps (runs of NaN)
    is_missing = series.isna().values
    gap_lengths = []
    current_gap = 0
    
    for val in is_missing:
        if val:
            current_gap += 1
        else:
            if current_gap > 0:
                gap_lengths.append(current_gap)
            current_gap = 0
    if current_gap > 0:
        gap_lengths.append(current_gap)
    
    # 2. max_gap_ratio
    features.max_gap_ratio = max(gap_lengths) / total_days if gap_lengths else 0.0
    
    # 3. gap_count_rate
    features.gap_count_rate = len(gap_lengths) / total_days if total_days > 0 else 0.0
    
    # 4. gap_len_entropy (normalized Shannon entropy of gap lengths)
    if gap_lengths and len(set(gap_lengths)) > 1:
        gap_counts = pd.Series(gap_lengths).value_counts(normalize=True)
        features.gap_len_entropy = entropy(gap_counts) / np.log(len(gap_counts))
    else:
        features.gap_len_entropy = 0.0
    
    # ===================
    # B) Cadence & Time-Step Structure
    # ===================
    
    # Get valid observation dates
    valid_dates = series.index[valid_mask]
    
    if len(valid_dates) >= 2:
        # Inter-arrival deltas in days
        deltas = np.diff(valid_dates).astype('timedelta64[D]').astype(float)
        
        # 5. median_dt_days
        features.median_dt_days = float(np.median(deltas))
        
        # 6. dt_entropy (bucket deltas and compute entropy)
        # Buckets: 1, 2, 3, 4-6, 7, 8-14, 15-31, 32+
        def bucket_delta(d):
            if d <= 1:
                return 0
            elif d <= 2:
                return 1
            elif d <= 3:
                return 2
            elif d <= 6:
                return 3
            elif d <= 7:
                return 4
            elif d <= 14:
                return 5
            elif d <= 31:
                return 6
            else:
                return 7
        
        bucketed = [bucket_delta(d) for d in deltas]
        bucket_counts = pd.Series(bucketed).value_counts(normalize=True)
        if len(bucket_counts) > 1:
            features.dt_entropy = entropy(bucket_counts) / np.log(8)  # 8 buckets
        else:
            features.dt_entropy = 0.0
        
        # 7. dt_cv (coefficient of variation)
        mean_dt = np.mean(deltas)
        std_dt = np.std(deltas)
        features.dt_cv = std_dt / mean_dt if mean_dt > 0 else 0.0
        
        # 8. staleness_rate (fraction of days that would be forward-filled)
        features.staleness_rate = 1 - (valid_count / total_days)
    
    # Compute cadence-adjusted missing rate
    features.missing_rate_adjusted = compute_adjusted_missing_rate(
        features.missing_rate, 
        features.median_dt_days
    )
    
    # ===================
    # C) Temporal Overlap / Gating Features (NOT for clustering)
    # ===================
    
    first_valid = valid_dates.min() if len(valid_dates) > 0 else panel_min_date
    last_valid = valid_dates.max() if len(valid_dates) > 0 else panel_max_date
    
    # 9. start_pos (GATING ONLY)
    if panel_span_days > 0:
        features.start_pos = (first_valid - panel_min_date).days / panel_span_days
    
    # 10. end_pos (GATING ONLY)
    if panel_span_days > 0:
        features.end_pos = (last_valid - panel_min_date).days / panel_span_days
    
    # 11. coverage_ratio (GATING ONLY)
    indicator_span = (last_valid - first_valid).days + 1
    if indicator_span > 0:
        features.coverage_ratio = valid_count / indicator_span
    
    # 12. effective_obs_log (GATING ONLY)
    features.effective_obs_log = np.log1p(valid_count)
    
    return features


def compute_panel_features(df: pd.DataFrame) -> Dict[str, IndicatorFeatures]:
    """
    Compute features for all indicators in panel.
    
    Args:
        df: Wide panel DataFrame (rows=dates, cols=indicators)
    
    Returns:
        Dict mapping indicator_id to IndicatorFeatures
    """
    features = {}
    
    panel_min_date = df.index.min()
    panel_max_date = df.index.max()
    panel_span_days = (panel_max_date - panel_min_date).days + 1
    
    for col in df.columns:
        features[col] = compute_indicator_features(
            df[col],
            panel_min_date,
            panel_max_date,
            panel_span_days,
        )
    
    return features


# =============================================================================
# Clustering
# =============================================================================

def build_feature_matrix(
    features: Dict[str, IndicatorFeatures],
    robust_scale: bool = True,
) -> Tuple[np.ndarray, List[str]]:
    """
    Build feature matrix from indicator features.
    Uses ONLY clustering features (not gating features).
    
    Args:
        features: Dict of indicator features
        robust_scale: Whether to apply robust scaling (median/IQR)
    
    Returns:
        (feature_matrix, indicator_ids)
    """
    indicator_ids = list(features.keys())
    n = len(indicator_ids)
    
    # Build raw matrix using ONLY clustering features
    X = np.zeros((n, len(CLUSTERING_ORDER)))
    for i, ind_id in enumerate(indicator_ids):
        X[i] = features[ind_id].to_clustering_vector()
    
    # Robust scaling (median / IQR)
    if robust_scale and n > 1:
        for j in range(X.shape[1]):
            col = X[:, j]
            median = np.median(col)
            q75, q25 = np.percentile(col, [75, 25])
            iqr = q75 - q25
            if iqr > 0:
                X[:, j] = (col - median) / iqr
            else:
                X[:, j] = 0.0
    
    # Apply sqrt(weight) to columns for weighted euclidean
    weights = np.array([CLUSTERING_WEIGHTS[f] for f in CLUSTERING_ORDER])
    X = X * np.sqrt(weights)
    
    return X, indicator_ids


def cluster_indicators(
    X: np.ndarray,
    indicator_ids: List[str],
    method: str = "ward",
    threshold: Optional[float] = None,
    n_clusters: Optional[int] = None,
) -> Dict[str, str]:
    """
    Cluster indicators into cohorts.
    """
    n = len(indicator_ids)
    
    if n <= 1:
        return {indicator_ids[0]: "cohort_0"} if n == 1 else {}
    
    # Hierarchical clustering
    Z = linkage(X, method=method)
    
    # Determine clusters
    if n_clusters is not None:
        labels = fcluster(Z, t=n_clusters, criterion='maxclust')
    elif threshold is not None:
        labels = fcluster(Z, t=threshold, criterion='distance')
    else:
        # Auto threshold: use 50% of max distance
        max_dist = Z[-1, 2]
        threshold = max_dist * 0.5
        labels = fcluster(Z, t=threshold, criterion='distance')
    
    # Build assignment dict
    assignments = {}
    for i, ind_id in enumerate(indicator_ids):
        assignments[ind_id] = f"cohort_{labels[i] - 1}"  # 0-indexed
    
    return assignments


def infer_cadence(median_dt: float) -> str:
    """Infer cadence label from median inter-arrival time."""
    if median_dt <= 1.5:
        return "daily"
    elif median_dt <= 5.5:
        return "weekly"
    elif median_dt <= 35:
        return "monthly"
    elif median_dt <= 100:
        return "quarterly"
    else:
        return "irregular"


def build_cohort_profiles(
    assignments: Dict[str, str],
    features: Dict[str, IndicatorFeatures],
    X: np.ndarray,
    indicator_ids: List[str],
) -> Dict[str, CohortProfile]:
    """
    Build cohort profiles from assignments.
    
    Args:
        assignments: indicator_id -> cohort_id mapping
        features: Indicator features
        X: Feature matrix (clustering features only)
        indicator_ids: List of indicator IDs
    
    Returns:
        Dict of CohortProfile objects
    """
    # Group indicators by cohort
    cohort_indicators: Dict[str, List[str]] = {}
    for ind_id, cohort_id in assignments.items():
        if cohort_id not in cohort_indicators:
            cohort_indicators[cohort_id] = []
        cohort_indicators[cohort_id].append(ind_id)
    
    # Build profiles
    profiles = {}
    indicator_idx = {ind: i for i, ind in enumerate(indicator_ids)}
    
    for cohort_id, indicators in cohort_indicators.items():
        # Aggregate features
        cohort_features = [features[ind] for ind in indicators]
        
        mean_missing = np.mean([f.missing_rate for f in cohort_features])
        mean_missing_adj = np.mean([f.missing_rate_adjusted for f in cohort_features])
        mean_density = np.mean([f.coverage_ratio for f in cohort_features])
        mean_dt = np.mean([f.median_dt_days for f in cohort_features])
        
        # Gating info (from gating features)
        earliest_start = min(f.start_pos for f in cohort_features)
        latest_end = max(f.end_pos for f in cohort_features)
        
        # Overlap ratio: how much do all indicators share?
        latest_start = max(f.start_pos for f in cohort_features)
        earliest_end = min(f.end_pos for f in cohort_features)
        if latest_end > earliest_start:
            overlap = max(0, earliest_end - latest_start) / (latest_end - earliest_start)
        else:
            overlap = 0.0
        
        # Internal distance (cohort tightness) — using clustering features only
        if len(indicators) > 1:
            indices = [indicator_idx[ind] for ind in indicators]
            cohort_X = X[indices]
            centroid = cohort_X.mean(axis=0)
            distances = np.linalg.norm(cohort_X - centroid, axis=1)
            internal_distance = float(np.mean(distances))
        else:
            internal_distance = 0.0
        
        # Confidence (inverse of internal distance, bounded)
        # Now based on structural similarity, not temporal position
        confidence = 1.0 / (1.0 + internal_distance)
        
        profiles[cohort_id] = CohortProfile(
            cohort_id=cohort_id,
            indicators=indicators,
            mean_missing_rate=mean_missing,
            mean_missing_adjusted=mean_missing_adj,
            mean_density=mean_density,
            inferred_cadence=infer_cadence(mean_dt),
            confidence=confidence,
            internal_distance=internal_distance,
            earliest_start=earliest_start,
            latest_end=latest_end,
            overlap_ratio=overlap,
        )
    
    return profiles


# =============================================================================
# Main Agent Class
# =============================================================================

class CohortDiscoveryAgent:
    """
    Discovers structural cohorts from data.
    
    Usage:
        agent = CohortDiscoveryAgent()
        result = agent.run(clean_df)
        
        for cohort_id, profile in result.cohorts.items():
            print(f"{cohort_id}: {profile.indicators}")
            engine.run(clean_df[profile.indicators])
    """
    
    name = "cohort_discovery"
    
    def __init__(
        self,
        registry=None,
        method: str = "ward",
        threshold: Optional[float] = None,
        n_clusters: Optional[int] = None,
        min_cohort_size: int = 1,
    ):
        """
        Initialize CohortDiscoveryAgent.
        
        Args:
            registry: Optional DiagnosticRegistry for audit trail
            method: Clustering method ('ward', 'average', 'complete')
            threshold: Distance threshold for clustering
            n_clusters: Target number of clusters (alternative to threshold)
            min_cohort_size: Minimum indicators per cohort
        """
        self._registry = registry
        self.method = method
        self.threshold = threshold
        self.n_clusters = n_clusters
        self.min_cohort_size = min_cohort_size
        self._last_result: Optional[CohortDiscoveryResult] = None
    
    def run(self, df: pd.DataFrame) -> CohortDiscoveryResult:
        """
        Discover cohorts from panel data.
        
        Args:
            df: Wide panel DataFrame (rows=dates, cols=indicators)
        
        Returns:
            CohortDiscoveryResult with cohort assignments and profiles
        """
        logger.info(f"Discovering cohorts for {len(df.columns)} indicators...")
        
        # Compute features
        features = compute_panel_features(df)
        
        # Build feature matrix (clustering features only)
        X, indicator_ids = build_feature_matrix(features, robust_scale=True)
        
        # Cluster
        assignments = cluster_indicators(
            X, indicator_ids,
            method=self.method,
            threshold=self.threshold,
            n_clusters=self.n_clusters,
        )
        
        # Build profiles
        profiles = build_cohort_profiles(assignments, features, X, indicator_ids)
        
        # Build result (no global confidence)
        result = CohortDiscoveryResult(
            cohorts=profiles,
            assignments=assignments,
            features=features,
            linkage_threshold=self.threshold or 0.0,
        )
        
        # Log to registry if available
        if self._registry is not None:
            self._registry.store(self.name, "n_cohorts", result.n_cohorts)
            self._registry.store(self.name, "assignments", assignments)
            for cid, profile in profiles.items():
                self._registry.store(self.name, f"{cid}_confidence", profile.confidence)
        
        logger.info(f"Discovered {result.n_cohorts} cohorts")
        
        self._last_result = result
        return result
    
    def explain(self) -> str:
        """Return human-readable summary of cohort discovery."""
        if self._last_result is None:
            return "No cohort discovery has been run yet."
        
        lines = ["COHORT DISCOVERY SUMMARY", "=" * 50]
        
        result = self._last_result
        lines.append(f"Total indicators: {result.n_indicators}")
        lines.append(f"Cohorts discovered: {result.n_cohorts}")
        lines.append("")
        
        for cohort_id, profile in sorted(result.cohorts.items()):
            lines.append(f"{cohort_id}:")
            lines.append(f"  Indicators:    {', '.join(sorted(profile.indicators))}")
            lines.append(f"  Cadence:       {profile.inferred_cadence}")
            lines.append(f"  Missing (raw): {profile.mean_missing_rate:.1%}")
            lines.append(f"  Missing (adj): {profile.mean_missing_adjusted:.1%}")
            lines.append(f"  Overlap:       {profile.overlap_ratio:.1%}")
            lines.append(f"  Confidence:    {profile.confidence:.2f}")
            lines.append("")
        
        return "\n".join(lines)


# =============================================================================
# Convenience Function
# =============================================================================

def discover_cohorts(
    df: pd.DataFrame,
    n_clusters: Optional[int] = None,
    threshold: Optional[float] = None,
) -> CohortDiscoveryResult:
    """
    Convenience function to discover cohorts.
    
    Args:
        df: Wide panel DataFrame
        n_clusters: Target number of clusters
        threshold: Distance threshold
    
    Returns:
        CohortDiscoveryResult
    """
    agent = CohortDiscoveryAgent(
        n_clusters=n_clusters,
        threshold=threshold,
    )
    return agent.run(df)


if __name__ == "__main__":
    # Quick test with synthetic data
    import pandas as pd
    import numpy as np
    
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=1000, freq="D")
    
    # Create panel with different structures
    df = pd.DataFrame(index=dates)
    
    # Daily indicators (dense)
    df["SPY"] = np.random.randn(1000).cumsum()
    df["QQQ"] = np.random.randn(1000).cumsum() + df["SPY"] * 0.8
    df["VIX"] = np.abs(np.random.randn(1000)) * 20
    
    # Weekly-ish (some gaps)
    weekly = np.random.randn(1000).cumsum()
    weekly[::7] = np.nan  # Every 7th day missing
    df["WEEKLY"] = weekly
    
    # Monthly (very sparse on daily grid)
    monthly = np.full(1000, np.nan)
    monthly[::30] = np.random.randn(len(monthly[::30])).cumsum()
    df["MONTHLY"] = monthly
    
    # Run cohort discovery
    agent = CohortDiscoveryAgent(n_clusters=3)
    result = agent.run(df)
    print(agent.explain())