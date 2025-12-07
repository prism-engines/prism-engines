"""
Hidden Variation Detector (HVD)
===============================

Advisory-only layer that uses similarity metrics and unsupervised ML to:
1. Find highly similar indicators (potential double-count / redundancy)
2. Find behavior clusters (groups of indicators that move together)
3. Flag divergences (family members that stop behaving alike)

IMPORTANT: This module is READ-ONLY and ADVISORY.
- No changes to PRISM geometry math
- No changes to weights / engine outputs
- Only warnings, clusters, and summary objects for UI and logs

Usage:
    from analysis.hidden_variation_detector import build_hvd_report
    
    report = build_hvd_report(panel_df, family_manager=fm)
    for warning in report.warnings:
        print(warning)

Author: PRISM Project
Version: 1.0.0
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict
import warnings


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class SimilarityResult:
    """Result of pairwise similarity analysis."""
    pair: Tuple[str, str]
    window: str                    # e.g. 'full', '5y', '1y'
    correlation: float
    distance: float                # e.g. 1 - |corr|
    flags: List[str] = field(default_factory=list)  # e.g. ['HIGH_SIMILARITY', 'DIVERGENCE']
    
    def __post_init__(self):
        if self.flags is None:
            self.flags = []


@dataclass
class ClusterResult:
    """Result of clustering analysis."""
    cluster_id: int
    members: List[str]
    method: str                    # e.g. 'kmeans', 'hierarchical', 'correlation'
    centroid_indicator: Optional[str] = None
    intra_cluster_correlation: Optional[float] = None
    notes: str = ""


@dataclass
class FamilyDivergence:
    """Detected divergence within an indicator family."""
    family_id: str
    member_1: str
    member_2: str
    correlation: float
    window: str
    expected_correlation: float = 0.80
    severity: str = "medium"       # 'low', 'medium', 'high'


@dataclass
class HVDReport:
    """Complete Hidden Variation Detector report."""
    indicators: List[str]
    similarity_results: List[SimilarityResult]
    clusters: List[ClusterResult]
    family_divergences: List[FamilyDivergence]
    warnings: List[str]
    meta: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def high_similarity_pairs(self) -> List[SimilarityResult]:
        """Get pairs flagged as HIGH_SIMILARITY."""
        return [r for r in self.similarity_results if 'HIGH_SIMILARITY' in r.flags]
    
    @property
    def divergent_pairs(self) -> List[SimilarityResult]:
        """Get pairs flagged as DIVERGENCE."""
        return [r for r in self.similarity_results if 'DIVERGENCE' in r.flags]
    
    def summary(self) -> Dict[str, Any]:
        """Generate summary statistics."""
        return {
            'total_indicators': len(self.indicators),
            'high_similarity_count': len(self.high_similarity_pairs),
            'cluster_count': len(self.clusters),
            'divergence_count': len(self.family_divergences),
            'warning_count': len(self.warnings)
        }


# =============================================================================
# Core Functions
# =============================================================================

def compute_similarity_matrix(
    panel: pd.DataFrame,
    min_overlap: int = 252,
    method: str = 'pearson'
) -> Dict[Tuple[str, str], float]:
    """
    Compute pairwise correlations between columns.
    
    Only uses overlapping non-NaN periods. Returns dict[(col1, col2)] = corr.
    
    Args:
        panel: DataFrame with indicators as columns
        min_overlap: Minimum overlapping observations required
        method: Correlation method ('pearson', 'spearman', 'kendall')
        
    Returns:
        Dictionary mapping (col1, col2) tuples to correlation values
    """
    similarity = {}
    columns = list(panel.columns)
    
    for i, col1 in enumerate(columns):
        for col2 in columns[i+1:]:
            # Get overlapping non-NaN values
            mask = panel[col1].notna() & panel[col2].notna()
            overlap = mask.sum()
            
            if overlap < min_overlap:
                continue
            
            series1 = panel.loc[mask, col1]
            series2 = panel.loc[mask, col2]
            
            try:
                if method == 'pearson':
                    corr = np.corrcoef(series1.values, series2.values)[0, 1]
                elif method == 'spearman':
                    from scipy.stats import spearmanr
                    corr, _ = spearmanr(series1.values, series2.values)
                elif method == 'kendall':
                    from scipy.stats import kendalltau
                    corr, _ = kendalltau(series1.values, series2.values)
                else:
                    corr = np.corrcoef(series1.values, series2.values)[0, 1]
                
                if not np.isnan(corr):
                    similarity[(col1, col2)] = corr
                    
            except Exception:
                continue
    
    return similarity


def detect_high_similarity_pairs(
    similarity: Dict[Tuple[str, str], float],
    threshold: float = 0.90
) -> List[SimilarityResult]:
    """
    Flag pairs above threshold as HIGH_SIMILARITY.
    
    Args:
        similarity: Dictionary of pairwise correlations
        threshold: Correlation threshold for flagging
        
    Returns:
        List of SimilarityResult objects for high-similarity pairs
    """
    results = []
    
    for (col1, col2), corr in similarity.items():
        if abs(corr) >= threshold:
            results.append(SimilarityResult(
                pair=(col1, col2),
                window='full',
                correlation=round(corr, 4),
                distance=round(1 - abs(corr), 4),
                flags=['HIGH_SIMILARITY']
            ))
    
    # Sort by absolute correlation descending
    results.sort(key=lambda x: abs(x.correlation), reverse=True)
    
    return results


def detect_rolling_divergence(
    panel: pd.DataFrame,
    pair: Tuple[str, str],
    window_days: int = 252,
    divergence_threshold: float = 0.50,
    baseline_threshold: float = 0.80
) -> Optional[SimilarityResult]:
    """
    Detect if a historically correlated pair has diverged recently.
    
    Args:
        panel: DataFrame with indicators
        pair: Tuple of (col1, col2)
        window_days: Rolling window size
        divergence_threshold: Correlation below this = divergence
        baseline_threshold: Expected baseline correlation
        
    Returns:
        SimilarityResult with DIVERGENCE flag if detected, else None
    """
    col1, col2 = pair
    
    if col1 not in panel.columns or col2 not in panel.columns:
        return None
    
    # Get overlapping data
    mask = panel[col1].notna() & panel[col2].notna()
    if mask.sum() < window_days * 2:
        return None
    
    data = panel.loc[mask, [col1, col2]].copy()
    
    # Full period correlation
    full_corr = data[col1].corr(data[col2])
    
    # Recent window correlation
    recent_data = data.tail(window_days)
    recent_corr = recent_data[col1].corr(recent_data[col2])
    
    # Check for divergence
    if full_corr >= baseline_threshold and recent_corr < divergence_threshold:
        return SimilarityResult(
            pair=pair,
            window=f'{window_days}d',
            correlation=round(recent_corr, 4),
            distance=round(1 - abs(recent_corr), 4),
            flags=['DIVERGENCE', 'HISTORICALLY_CORRELATED']
        )
    
    return None


def detect_clusters_correlation(
    panel: pd.DataFrame,
    threshold: float = 0.70,
    min_cluster_size: int = 2
) -> List[ClusterResult]:
    """
    Cluster indicators based on correlation similarity.
    
    Uses hierarchical-like approach: group indicators with correlation > threshold.
    No external dependencies required.
    
    Args:
        panel: DataFrame with indicators as columns
        threshold: Minimum correlation to be in same cluster
        min_cluster_size: Minimum members for a valid cluster
        
    Returns:
        List of ClusterResult objects
    """
    columns = list(panel.columns)
    n = len(columns)
    
    if n < 2:
        return []
    
    # Build correlation matrix
    corr_matrix = panel.corr(method='pearson')
    
    # Simple clustering via connected components with correlation threshold
    visited = set()
    clusters = []
    cluster_id = 0
    
    for col in columns:
        if col in visited:
            continue
        
        # BFS to find all connected (correlated) indicators
        cluster = []
        queue = [col]
        
        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            
            visited.add(current)
            cluster.append(current)
            
            # Find connected nodes
            for other in columns:
                if other not in visited:
                    corr = corr_matrix.loc[current, other]
                    if not np.isnan(corr) and abs(corr) >= threshold:
                        queue.append(other)
        
        if len(cluster) >= min_cluster_size:
            # Calculate intra-cluster correlation
            if len(cluster) > 1:
                sub_corr = corr_matrix.loc[cluster, cluster]
                # Average off-diagonal correlation
                mask = ~np.eye(len(cluster), dtype=bool)
                avg_corr = sub_corr.values[mask].mean()
            else:
                avg_corr = 1.0
            
            clusters.append(ClusterResult(
                cluster_id=cluster_id,
                members=cluster,
                method='correlation_threshold',
                intra_cluster_correlation=round(avg_corr, 4),
                notes=f"Indicators with pairwise |corr| >= {threshold}"
            ))
            cluster_id += 1
    
    return clusters


def detect_clusters_kmeans(
    panel: pd.DataFrame,
    n_clusters: int = 4,
    random_state: int = 42
) -> List[ClusterResult]:
    """
    Cluster indicators using KMeans on standardized returns.
    
    Falls back to correlation-based clustering if sklearn unavailable.
    
    Args:
        panel: DataFrame with indicators as columns
        n_clusters: Number of clusters
        random_state: Random seed for reproducibility
        
    Returns:
        List of ClusterResult objects
    """
    try:
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA
    except ImportError:
        # Fall back to correlation-based clustering
        return detect_clusters_correlation(panel)
    
    # Prepare data: transpose so indicators are rows
    # Use returns to avoid scale issues
    returns = panel.pct_change().dropna()
    
    if returns.shape[0] < 30 or returns.shape[1] < n_clusters:
        return detect_clusters_correlation(panel)
    
    # Transpose: columns become samples for clustering
    X = returns.T.values
    
    # Handle NaN
    X = np.nan_to_num(X, nan=0.0)
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Optional: reduce dimensions if many time steps
    if X_scaled.shape[1] > 50:
        pca = PCA(n_components=min(50, X_scaled.shape[0] - 1))
        X_scaled = pca.fit_transform(X_scaled)
    
    # Cluster
    actual_n_clusters = min(n_clusters, X_scaled.shape[0])
    kmeans = KMeans(n_clusters=actual_n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    
    # Build results
    columns = list(panel.columns)
    cluster_members = defaultdict(list)
    
    for col, label in zip(columns, labels):
        cluster_members[label].append(col)
    
    results = []
    for cluster_id, members in cluster_members.items():
        if len(members) >= 1:
            results.append(ClusterResult(
                cluster_id=cluster_id,
                members=members,
                method='kmeans',
                notes=f"KMeans with {actual_n_clusters} clusters"
            ))
    
    return results


def detect_family_divergences(
    panel: pd.DataFrame,
    family_manager: Any,
    divergence_threshold: float = 0.70,
    window_days: int = 756  # ~3 years
) -> Tuple[List[FamilyDivergence], List[str]]:
    """
    Detect divergences within indicator families.
    
    Args:
        panel: DataFrame with indicators
        family_manager: FamilyManager instance (optional)
        divergence_threshold: Correlation below this = divergence
        window_days: Window for recent correlation
        
    Returns:
        Tuple of (divergences, warnings)
    """
    divergences = []
    warnings_list = []
    
    if family_manager is None:
        return divergences, warnings_list
    
    try:
        families = family_manager.families
    except AttributeError:
        return divergences, warnings_list
    
    panel_cols = set(panel.columns)
    
    for family_id, family in families.items():
        try:
            members_in_panel = [m for m in family.members if m in panel_cols]
        except AttributeError:
            continue
        
        if len(members_in_panel) < 2:
            continue
        
        # Warning: multiple family members present
        warnings_list.append(
            f"Family '{family_id}': members {members_in_panel} present in panel. "
            f"May overweight signal unless handled explicitly."
        )
        
        # Check correlation between family members
        for i, m1 in enumerate(members_in_panel):
            for m2 in members_in_panel[i+1:]:
                # Full correlation
                mask = panel[m1].notna() & panel[m2].notna()
                if mask.sum() < 100:
                    continue
                
                full_corr = panel.loc[mask, m1].corr(panel.loc[mask, m2])
                
                # Recent correlation
                recent_mask = mask & (mask.cumsum() > mask.sum() - window_days)
                if recent_mask.sum() > 50:
                    recent_corr = panel.loc[recent_mask, m1].corr(panel.loc[recent_mask, m2])
                else:
                    recent_corr = full_corr
                
                # Check for divergence
                if recent_corr < divergence_threshold:
                    severity = 'high' if recent_corr < 0.4 else 'medium' if recent_corr < 0.6 else 'low'
                    
                    divergences.append(FamilyDivergence(
                        family_id=family_id,
                        member_1=m1,
                        member_2=m2,
                        correlation=round(recent_corr, 4),
                        window=f'{window_days}d',
                        severity=severity
                    ))
                    
                    warnings_list.append(
                        f"DIVERGENCE in family '{family_id}': {m1} vs {m2} "
                        f"corr={recent_corr:.2f} (last {window_days} days). "
                        f"These should typically move together."
                    )
    
    return divergences, warnings_list


def build_hvd_report(
    panel: pd.DataFrame,
    family_manager: Any = None,
    similarity_threshold: float = 0.90,
    cluster_method: str = 'auto',
    n_clusters: int = 4,
    min_overlap: int = 252
) -> HVDReport:
    """
    Build complete Hidden Variation Detector report.
    
    This is the main orchestrator function that:
    1. Computes similarity matrix
    2. Detects high-similarity pairs
    3. Clusters indicators
    4. Detects family divergences (if family_manager provided)
    5. Generates warnings
    
    Args:
        panel: DataFrame with indicators as columns
        family_manager: Optional FamilyManager for family-aware analysis
        similarity_threshold: Threshold for HIGH_SIMILARITY flag
        cluster_method: 'kmeans', 'correlation', or 'auto'
        n_clusters: Number of clusters for KMeans
        min_overlap: Minimum overlapping observations for similarity
        
    Returns:
        Complete HVDReport
    """
    indicators = list(panel.columns)
    warnings_list = []
    
    # Safety checks
    if panel.empty:
        warnings_list.append("Panel is empty - no analysis performed")
        return HVDReport(
            indicators=[],
            similarity_results=[],
            clusters=[],
            family_divergences=[],
            warnings=warnings_list,
            meta={'status': 'empty_panel'}
        )
    
    if len(indicators) < 2:
        warnings_list.append(f"Only {len(indicators)} indicator(s) - need at least 2 for comparison")
        return HVDReport(
            indicators=indicators,
            similarity_results=[],
            clusters=[],
            family_divergences=[],
            warnings=warnings_list,
            meta={'status': 'insufficient_indicators'}
        )
    
    if len(panel) < min_overlap:
        warnings_list.append(
            f"Panel has only {len(panel)} rows - need at least {min_overlap} for reliable analysis"
        )
    
    # Step 1: Compute similarity matrix
    similarity = compute_similarity_matrix(panel, min_overlap=min(min_overlap, len(panel) // 2))
    
    # Step 2: Detect high-similarity pairs
    high_sim_results = detect_high_similarity_pairs(similarity, threshold=similarity_threshold)
    
    if high_sim_results:
        warnings_list.append(
            f"Found {len(high_sim_results)} high-similarity pair(s) (corr >= {similarity_threshold})"
        )
    
    # Step 3: Cluster indicators
    if cluster_method == 'kmeans':
        clusters = detect_clusters_kmeans(panel, n_clusters=n_clusters)
    elif cluster_method == 'correlation':
        clusters = detect_clusters_correlation(panel)
    else:  # auto
        clusters = detect_clusters_kmeans(panel, n_clusters=n_clusters)
        if not clusters:
            clusters = detect_clusters_correlation(panel)
    
    # Step 4: Detect family divergences
    family_divergences, family_warnings = detect_family_divergences(panel, family_manager)
    warnings_list.extend(family_warnings)
    
    # Step 5: Build metadata
    meta = {
        'status': 'completed',
        'panel_shape': panel.shape,
        'similarity_threshold': similarity_threshold,
        'cluster_method': cluster_method,
        'pairs_analyzed': len(similarity),
        'family_manager_available': family_manager is not None
    }
    
    return HVDReport(
        indicators=indicators,
        similarity_results=high_sim_results,
        clusters=clusters,
        family_divergences=family_divergences,
        warnings=warnings_list,
        meta=meta
    )


# =============================================================================
# Pretty Printing
# =============================================================================

def print_hvd_summary(report: HVDReport, max_warnings: int = 10, max_pairs: int = 5):
    """
    Print a formatted summary of the HVD report.
    
    Args:
        report: HVDReport to summarize
        max_warnings: Maximum warnings to display
        max_pairs: Maximum high-similarity pairs to display
    """
    print("\n" + "=" * 60)
    print("HIDDEN VARIATION DETECTOR SUMMARY")
    print("=" * 60)
    
    summary = report.summary()
    print(f"\nIndicators analyzed: {summary['total_indicators']}")
    print(f"High-similarity pairs: {summary['high_similarity_count']}")
    print(f"Clusters found: {summary['cluster_count']}")
    print(f"Family divergences: {summary['divergence_count']}")
    
    # Warnings
    if report.warnings:
        print(f"\n⚠️  Warnings ({min(len(report.warnings), max_warnings)} of {len(report.warnings)}):")
        for w in report.warnings[:max_warnings]:
            print(f"   • {w}")
    
    # High-similarity pairs
    high_sim = report.high_similarity_pairs
    if high_sim:
        print(f"\n🔴 High-Similarity Pairs (top {min(len(high_sim), max_pairs)}):")
        for r in high_sim[:max_pairs]:
            a, b = r.pair
            print(f"   {a} ~ {b}: corr={r.correlation:.3f}")
    else:
        print("\n✅ No high-similarity pairs above threshold")
    
    # Clusters
    if report.clusters:
        print(f"\n📊 Indicator Clusters ({len(report.clusters)}):")
        for c in report.clusters:
            members_str = ', '.join(c.members[:5])
            if len(c.members) > 5:
                members_str += f" (+{len(c.members) - 5} more)"
            corr_str = f", avg_corr={c.intra_cluster_correlation:.2f}" if c.intra_cluster_correlation else ""
            print(f"   Cluster {c.cluster_id}: [{members_str}]{corr_str}")
    
    # Family divergences
    if report.family_divergences:
        print(f"\n🔶 Family Divergences ({len(report.family_divergences)}):")
        for d in report.family_divergences:
            print(f"   {d.family_id}: {d.member_1} vs {d.member_2} corr={d.correlation:.2f} [{d.severity}]")
    
    print("\n" + "=" * 60)


# =============================================================================
# CLI / Standalone Testing
# =============================================================================

if __name__ == "__main__":
    print("Hidden Variation Detector - Standalone Test")
    print("=" * 60)
    
    # Create synthetic test data
    np.random.seed(42)
    n = 500
    
    # Base signals
    trend = np.linspace(0, 10, n)
    cycle = np.sin(np.linspace(0, 8 * np.pi, n))
    noise = np.random.randn(n) * 0.5
    
    # Create indicators with known relationships
    panel = pd.DataFrame({
        # Cluster 1: Trend-following (should be highly correlated)
        'trend_a': trend + noise,
        'trend_b': trend * 1.1 + noise * 0.8,  # ~0.95+ correlation with trend_a
        'trend_c': trend * 0.9 + np.random.randn(n) * 0.3,
        
        # Cluster 2: Cyclical (should cluster together)
        'cycle_a': cycle + noise * 0.3,
        'cycle_b': cycle * 1.2 + np.random.randn(n) * 0.2,
        
        # Cluster 3: Independent
        'random_a': np.random.randn(n),
        'random_b': np.random.randn(n),
        
        # Divergent pair (was correlated, now not)
        'diverge_a': np.concatenate([trend[:300] + noise[:300], np.random.randn(200)]),
        'diverge_b': trend + noise * 0.5,
    })
    
    print(f"\nTest panel: {panel.shape[0]} rows, {panel.shape[1]} columns")
    
    # Run HVD
    report = build_hvd_report(panel, similarity_threshold=0.85)
    
    # Print summary
    print_hvd_summary(report)
    
    # Verify expected results
    print("\n" + "-" * 60)
    print("VERIFICATION:")
    
    high_sim_pairs = {(r.pair[0], r.pair[1]) for r in report.high_similarity_pairs}
    
    # trend_a and trend_b should be flagged
    if ('trend_a', 'trend_b') in high_sim_pairs or ('trend_b', 'trend_a') in high_sim_pairs:
        print("✅ trend_a ~ trend_b correctly flagged as high-similarity")
    else:
        print("❌ trend_a ~ trend_b NOT flagged (expected high correlation)")
    
    # random_a and random_b should NOT be flagged
    if ('random_a', 'random_b') not in high_sim_pairs and ('random_b', 'random_a') not in high_sim_pairs:
        print("✅ random_a ~ random_b correctly NOT flagged")
    else:
        print("❌ random_a ~ random_b incorrectly flagged")
    
    # Should have clusters
    if len(report.clusters) >= 2:
        print(f"✅ Multiple clusters detected ({len(report.clusters)})")
    else:
        print(f"⚠️  Only {len(report.clusters)} cluster(s) detected")
    
    print("\n" + "=" * 60)
    print("Test completed!")
