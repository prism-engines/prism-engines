"""
PRISM Cohort Multi-View Geometry Engine

Computes geometry from multiple viewpoints to separate trivial correlation-driven
structure from genuine structural findings.

Views:
1. Full System - All indicators, all correlations (may be trivial)
2. Residual - After removing cluster means (reveals hidden structure)
3. Within-Cluster - Geometry within each correlation cluster
4. Intrinsic-Only - Using only non-relational engines (no correlation influence)

Key Insight:
- Correlation clusters are KNOWN structure (bonds cluster, equities cluster)
- What's INTERESTING is when the geometry disagrees with correlation
- Multiple views expose these disagreements

Cross-validated by: Claude
Date: December 2024
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Set
from datetime import datetime
import logging
from itertools import combinations
from collections import defaultdict

import duckdb

logger = logging.getLogger(__name__)


# =============================================================================
# THRESHOLDS
# =============================================================================

CLUSTER_CORRELATION_THRESHOLD = 0.60  # Min correlation to form cluster
PC_VARIANCE_THRESHOLD = 0.15  # Min variance to report a PC


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class IndicatorProfile:
    """Complete profile of an indicator across all views."""

    indicator_id: str

    # INVARIANT - computed once, never changes
    intrinsic: Dict[str, float] = field(default_factory=dict)

    # COHORT-DEPENDENT - computed per view
    relational_views: Dict[str, Dict[str, float]] = field(default_factory=dict)
    # {
    #   'full_system': {pc1_loading, centrality, ...},
    #   'residual': {pc1_loading, centrality, ...},
    #   'within_cohort': {pc1_loading, centrality, ...},
    #   'intrinsic_geometry': {nearest_neighbor, cluster, ...},
    # }

    cluster_id: Optional[str] = None


@dataclass
class ClusterInfo:
    """Information about a detected correlation cluster."""

    cluster_id: str
    members: List[str]
    mean_correlation: float
    within_cluster_dimension: float = 0.0
    within_cluster_pc1_variance: float = 0.0


@dataclass
class ViewGeometry:
    """Geometry computed for a single view."""

    view_name: str
    indicators: List[str]
    n_indicators: int

    # PCA structure
    effective_dimension: float = 0.0
    pc_variances: List[float] = field(default_factory=list)
    pc_interpretations: List[str] = field(default_factory=list)

    # Indicator positions
    pc_loadings: Dict[str, List[float]] = field(default_factory=dict)
    centrality: Dict[str, float] = field(default_factory=dict)

    # Clustering
    n_clusters: int = 0
    cluster_assignments: Dict[str, str] = field(default_factory=dict)


@dataclass
class MultiViewResult:
    """Complete multi-view geometry analysis result."""

    window_start: str
    window_end: str
    indicators: List[str]

    # Indicator profiles
    profiles: Dict[str, IndicatorProfile] = field(default_factory=dict)

    # Detected clusters
    correlation_clusters: List[ClusterInfo] = field(default_factory=list)

    # View geometries
    full_system: Optional[ViewGeometry] = None
    residual: Optional[ViewGeometry] = None
    within_cluster: Dict[str, ViewGeometry] = field(default_factory=dict)
    intrinsic_geometry: Optional[ViewGeometry] = None

    # Divergence findings
    divergences: List[Dict[str, Any]] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"Multi-View Geometry: {self.window_start} to {self.window_end}",
            f"Indicators: {len(self.indicators)}",
            f"Correlation clusters: {len(self.correlation_clusters)}",
            f"Divergences found: {len(self.divergences)}",
        ]
        return "\n".join(lines)


# =============================================================================
# MULTI-VIEW GEOMETRY ENGINE
# =============================================================================

class CohortMultiViewEngine:
    """
    Computes geometry from multiple viewpoints based on cohort structure.
    """

    def __init__(self, conn: duckdb.DuckDBPyConnection):
        self.conn = conn

    def compute_all_views(
        self,
        indicators: List[str],
        window_start: str,
        window_end: str
    ) -> MultiViewResult:
        """
        Compute geometry from all viewpoints.

        Args:
            indicators: List of indicator IDs
            window_start: Analysis window start
            window_end: Analysis window end

        Returns:
            MultiViewResult with all views and divergences
        """
        logger.info(f"Computing cohort multi-view geometry for {len(indicators)} indicators")

        result = MultiViewResult(
            window_start=window_start,
            window_end=window_end,
            indicators=indicators,
        )

        # 1. Load data
        price_data = self._load_price_data(indicators, window_start, window_end)
        descriptor_data = self._load_descriptors(indicators)

        # 2. Compute intrinsic profiles (once, invariant)
        for ind in indicators:
            profile = IndicatorProfile(indicator_id=ind)
            if ind in descriptor_data:
                profile.intrinsic = descriptor_data[ind]
            result.profiles[ind] = profile

        # 3. Compute correlation matrix
        corr_matrix, aligned_indicators = self._compute_correlation_matrix(price_data)
        if corr_matrix is None:
            logger.warning("Could not compute correlation matrix")
            return result

        # 4. Full system geometry
        result.full_system = self._compute_view_geometry(
            'full_system', corr_matrix, aligned_indicators
        )
        self._update_profiles_from_view(result.profiles, result.full_system, 'full_system')

        # 5. Detect correlation clusters
        result.correlation_clusters = self._detect_clusters(
            corr_matrix, aligned_indicators
        )

        # Assign cluster IDs to profiles
        for cluster in result.correlation_clusters:
            for ind in cluster.members:
                if ind in result.profiles:
                    result.profiles[ind].cluster_id = cluster.cluster_id

        # 6. Residual geometry (remove cluster structure)
        if result.correlation_clusters:
            residual_corr = self._compute_residual_correlation(
                corr_matrix, aligned_indicators, result.correlation_clusters
            )
            result.residual = self._compute_view_geometry(
                'residual', residual_corr, aligned_indicators
            )
            self._update_profiles_from_view(result.profiles, result.residual, 'residual')

        # 7. Within-cluster geometry (per cluster)
        for cluster in result.correlation_clusters:
            if len(cluster.members) >= 3:
                cluster_indices = [aligned_indicators.index(m) for m in cluster.members
                                  if m in aligned_indicators]
                if len(cluster_indices) >= 3:
                    cluster_corr = corr_matrix[np.ix_(cluster_indices, cluster_indices)]
                    view = self._compute_view_geometry(
                        f'within_{cluster.cluster_id}', cluster_corr, cluster.members
                    )
                    result.within_cluster[cluster.cluster_id] = view
                    cluster.within_cluster_dimension = view.effective_dimension
                    if view.pc_variances:
                        cluster.within_cluster_pc1_variance = view.pc_variances[0]

        # 8. Intrinsic-only geometry (using descriptor similarity, not correlation)
        if descriptor_data:
            intrinsic_sim = self._compute_intrinsic_similarity_matrix(
                aligned_indicators, descriptor_data
            )
            if intrinsic_sim is not None:
                result.intrinsic_geometry = self._compute_view_geometry(
                    'intrinsic', intrinsic_sim, aligned_indicators
                )
                self._update_profiles_from_view(
                    result.profiles, result.intrinsic_geometry, 'intrinsic_geometry'
                )

        # 9. Detect divergences
        result.divergences = self._detect_divergences(
            result, corr_matrix, aligned_indicators
        )

        logger.info(f"Completed multi-view analysis: {len(result.correlation_clusters)} clusters, "
                   f"{len(result.divergences)} divergences")

        return result

    def _load_price_data(
        self,
        indicators: List[str],
        window_start: str,
        window_end: str
    ) -> Dict[str, Dict]:
        """Load price data for all indicators."""
        data = {}
        for ind in indicators:
            try:
                rows = self.conn.execute("""
                    SELECT date, value FROM data.indicators
                    WHERE indicator_id = ? AND date >= ? AND date <= ?
                    ORDER BY date
                """, [ind, window_start, window_end]).fetchall()
                if rows:
                    data[ind] = {r[0]: r[1] for r in rows}
            except Exception as e:
                logger.warning(f"Could not load data for {ind}: {e}")
        return data

    def _load_descriptors(
        self,
        indicators: List[str]
    ) -> Dict[str, Dict[str, float]]:
        """Load intrinsic descriptors for all indicators."""
        descriptors = {}
        try:
            placeholders = ','.join(['?']*len(indicators))
            result = self.conn.execute(f"""
                SELECT indicator_id, dimension, value
                FROM derived.geometry_descriptors
                WHERE indicator_id IN ({placeholders})
            """, indicators).fetchall()

            for ind, dim, value in result:
                if value is None:
                    continue
                if ind not in descriptors:
                    descriptors[ind] = {}
                descriptors[ind][dim] = value

        except Exception as e:
            logger.warning(f"Could not load descriptors: {e}")

        return descriptors

    def _compute_correlation_matrix(
        self,
        data: Dict[str, Dict]
    ) -> Tuple[Optional[np.ndarray], List[str]]:
        """Compute correlation matrix from price data."""
        indicators = list(data.keys())
        if len(indicators) < 2:
            return None, indicators

        # Find common dates
        common_dates = set(data[indicators[0]].keys())
        for ind in indicators[1:]:
            common_dates &= set(data[ind].keys())
        common_dates = sorted(common_dates)

        if len(common_dates) < 20:
            return None, indicators

        # Build price matrix
        prices = np.array([[data[ind][d] for d in common_dates] for ind in indicators])
        prices = np.maximum(prices, 1e-10)
        returns = np.diff(np.log(prices), axis=1)

        # Compute correlation
        corr_matrix = np.corrcoef(returns)

        return corr_matrix, indicators

    def _compute_view_geometry(
        self,
        view_name: str,
        corr_matrix: np.ndarray,
        indicators: List[str]
    ) -> ViewGeometry:
        """Compute geometry for a single view."""
        n = len(indicators)
        view = ViewGeometry(
            view_name=view_name,
            indicators=indicators,
            n_indicators=n,
        )

        # PCA
        eigenvalues, eigenvectors = np.linalg.eigh(corr_matrix)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = np.maximum(eigenvalues[idx], 0)
        eigenvectors = eigenvectors[:, idx]

        total_var = np.sum(eigenvalues)
        if total_var > 0:
            explained = eigenvalues / total_var
            view.pc_variances = explained[:5].tolist()

            # Effective dimension
            cumvar = np.cumsum(explained)
            view.effective_dimension = float(np.searchsorted(cumvar, 0.9) + 1)

            # PC loadings
            for i, ind in enumerate(indicators):
                view.pc_loadings[ind] = eigenvectors[i, :3].tolist()

            # Centrality (mean absolute correlation)
            for i, ind in enumerate(indicators):
                row = np.abs(corr_matrix[i, :])
                row[i] = 0  # Exclude self
                view.centrality[ind] = float(np.mean(row))

        return view

    def _detect_clusters(
        self,
        corr_matrix: np.ndarray,
        indicators: List[str]
    ) -> List[ClusterInfo]:
        """Detect correlation clusters via connected components."""
        n = len(indicators)
        adjacency = np.abs(corr_matrix) > CLUSTER_CORRELATION_THRESHOLD
        np.fill_diagonal(adjacency, False)

        visited = set()
        clusters = []
        cluster_id = 0

        for start in range(n):
            if start in visited:
                continue

            # BFS to find connected component
            component = []
            queue = [start]

            while queue:
                node = queue.pop(0)
                if node in visited:
                    continue
                visited.add(node)
                component.append(node)

                for neighbor in range(n):
                    if adjacency[node, neighbor] and neighbor not in visited:
                        queue.append(neighbor)

            if len(component) >= 2:
                members = [indicators[i] for i in component]

                # Compute mean within-cluster correlation
                sub_corr = corr_matrix[np.ix_(component, component)]
                upper = sub_corr[np.triu_indices(len(component), k=1)]
                mean_corr = float(np.mean(np.abs(upper))) if len(upper) > 0 else 0

                cluster_name = chr(65 + cluster_id)  # A, B, C, ...
                clusters.append(ClusterInfo(
                    cluster_id=f"cluster_{cluster_name}",
                    members=members,
                    mean_correlation=mean_corr,
                ))
                cluster_id += 1

        return clusters

    def _compute_residual_correlation(
        self,
        corr_matrix: np.ndarray,
        indicators: List[str],
        clusters: List[ClusterInfo]
    ) -> np.ndarray:
        """Compute residual correlation after removing cluster structure."""
        n = len(indicators)
        residual = corr_matrix.copy()

        # For each cluster, remove the mean correlation
        for cluster in clusters:
            indices = [indicators.index(m) for m in cluster.members if m in indicators]
            if len(indices) < 2:
                continue

            # Get the cluster's mean correlation
            mean_corr = cluster.mean_correlation

            # Subtract mean from within-cluster correlations
            for i in indices:
                for j in indices:
                    if i != j:
                        residual[i, j] -= mean_corr * np.sign(corr_matrix[i, j])

        # Normalize and clip
        np.fill_diagonal(residual, 1.0)
        residual = np.clip(residual, -1.0, 1.0)

        return residual

    def _compute_intrinsic_similarity_matrix(
        self,
        indicators: List[str],
        descriptors: Dict[str, Dict[str, float]]
    ) -> Optional[np.ndarray]:
        """Compute similarity matrix based on intrinsic descriptors only."""
        n = len(indicators)
        sim_matrix = np.eye(n)

        for i, ind1 in enumerate(indicators):
            for j, ind2 in enumerate(indicators):
                if i >= j:
                    continue

                if ind1 not in descriptors or ind2 not in descriptors:
                    continue

                d1 = descriptors[ind1]
                d2 = descriptors[ind2]

                # Common descriptors
                common = set(d1.keys()) & set(d2.keys())
                if not common:
                    continue

                # Cosine similarity
                v1 = np.array([d1[k] for k in sorted(common)])
                v2 = np.array([d2[k] for k in sorted(common)])

                n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
                if n1 > 0 and n2 > 0:
                    sim = float(np.dot(v1, v2) / (n1 * n2))
                    sim_matrix[i, j] = sim
                    sim_matrix[j, i] = sim

        return sim_matrix

    def _update_profiles_from_view(
        self,
        profiles: Dict[str, IndicatorProfile],
        view: ViewGeometry,
        view_name: str
    ):
        """Update indicator profiles with view-specific metrics."""
        for ind in view.indicators:
            if ind in profiles:
                profiles[ind].relational_views[view_name] = {
                    'centrality': view.centrality.get(ind, 0),
                    'pc1_loading': view.pc_loadings.get(ind, [0])[0] if view.pc_loadings.get(ind) else 0,
                    'pc2_loading': view.pc_loadings.get(ind, [0, 0])[1] if len(view.pc_loadings.get(ind, [])) > 1 else 0,
                }

    def _detect_divergences(
        self,
        result: MultiViewResult,
        corr_matrix: np.ndarray,
        indicators: List[str]
    ) -> List[Dict[str, Any]]:
        """Detect divergences between correlation and intrinsic views."""
        divergences = []

        if result.intrinsic_geometry is None:
            return divergences

        intrinsic_sim = self._compute_intrinsic_similarity_matrix(
            indicators,
            {ind: p.intrinsic for ind, p in result.profiles.items()}
        )
        if intrinsic_sim is None:
            return divergences

        for i, ind1 in enumerate(indicators):
            for j, ind2 in enumerate(indicators):
                if i >= j:
                    continue

                corr = corr_matrix[i, j]
                intr_sim = intrinsic_sim[i, j]

                # Uncorrelated but intrinsically similar
                if abs(corr) < 0.3 and intr_sim > 0.7:
                    divergences.append({
                        'type': 'uncorrelated_but_similar',
                        'indicator_1': ind1,
                        'indicator_2': ind2,
                        'correlation': float(corr),
                        'intrinsic_similarity': float(intr_sim),
                        'interpretation': (
                            f"{ind1} and {ind2} move independently (corr={corr:.2f}) "
                            f"but have nearly identical intrinsic dynamics (sim={intr_sim:.2f})"
                        )
                    })

                # Correlated but intrinsically different
                if abs(corr) > 0.7 and intr_sim < 0.3:
                    divergences.append({
                        'type': 'correlated_but_different',
                        'indicator_1': ind1,
                        'indicator_2': ind2,
                        'correlation': float(corr),
                        'intrinsic_similarity': float(intr_sim),
                        'interpretation': (
                            f"{ind1} and {ind2} move together (corr={corr:.2f}) "
                            f"but have different intrinsic dynamics (sim={intr_sim:.2f})"
                        )
                    })

        return divergences


# =============================================================================
# CLI
# =============================================================================

def main():
    """CLI for running cohort multi-view geometry analysis."""
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s'
    )

    parser = argparse.ArgumentParser(description='PRISM Cohort Multi-View Geometry Engine')
    parser.add_argument('--window-start', required=True, help='Window start date')
    parser.add_argument('--window-end', required=True, help='Window end date')
    parser.add_argument('--indicators', nargs='+', help='Specific indicators')
    parser.add_argument('--db', default='data/prism.duckdb', help='Database path')
    args = parser.parse_args()

    conn = duckdb.connect(args.db)
    engine = CohortMultiViewEngine(conn)

    if args.indicators:
        indicators = args.indicators
    else:
        indicators = conn.execute("""
            SELECT DISTINCT indicator_id FROM data.indicators
            WHERE indicator_id IN ('IEF', 'LQD', 'TLT', 'SHY', 'HYG', 'AGG', 'TIP',
                                   'GDP', 'M2SL', 'VIXY', 'DGS1', 'SLV', 'XLI', 'XLK',
                                   'SPY', 'QQQ')
        """).fetchall()
        indicators = [r[0] for r in indicators]

    result = engine.compute_all_views(indicators, args.window_start, args.window_end)

    # Output
    print("\n" + "=" * 70)
    print("MULTI-VIEW GEOMETRY REPORT")
    print("=" * 70)
    print(f"Window: {result.window_start} to {result.window_end}")
    print(f"Indicators: {len(result.indicators)}")

    # Clusters
    print("\n" + "-" * 70)
    print("CORRELATION CLUSTERS DETECTED:")
    print("-" * 70)
    if result.correlation_clusters:
        for cluster in result.correlation_clusters:
            print(f"  {cluster.cluster_id}: {', '.join(cluster.members)}")
            print(f"    Mean correlation: {cluster.mean_correlation:.2f}")
            if cluster.within_cluster_dimension > 0:
                print(f"    Within-cluster dimension: {cluster.within_cluster_dimension:.1f}")
    else:
        print("  No clusters detected (threshold=0.60)")

    # Full system view
    if result.full_system:
        print("\n" + "-" * 70)
        print("VIEW 1 - FULL SYSTEM:")
        print("-" * 70)
        print(f"  Effective dimension: {result.full_system.effective_dimension:.1f}")
        for i, var in enumerate(result.full_system.pc_variances[:3]):
            if var > PC_VARIANCE_THRESHOLD:
                trivial = " <- TRIVIAL (correlation-driven)" if i == 0 and var > 0.4 else ""
                print(f"  PC{i+1}: {var:.1%}{trivial}")

    # Residual view
    if result.residual:
        print("\n" + "-" * 70)
        print("VIEW 2 - RESIDUAL (after removing cluster means):")
        print("-" * 70)
        print(f"  Effective dimension: {result.residual.effective_dimension:.1f}")
        for i, var in enumerate(result.residual.pc_variances[:3]):
            if var > PC_VARIANCE_THRESHOLD:
                interesting = " <- INTERESTING" if i == 0 else ""
                print(f"  PC{i+1}: {var:.1%}{interesting}")

    # Intrinsic view
    if result.intrinsic_geometry:
        print("\n" + "-" * 70)
        print("VIEW 3 - INTRINSIC GEOMETRY (descriptor-based, no correlations):")
        print("-" * 70)
        print(f"  Effective dimension: {result.intrinsic_geometry.effective_dimension:.1f}")
        for i, var in enumerate(result.intrinsic_geometry.pc_variances[:3]):
            if var > PC_VARIANCE_THRESHOLD:
                print(f"  PC{i+1}: {var:.1%}")

    # Divergences
    if result.divergences:
        print("\n" + "-" * 70)
        print("DIVERGENCE FLAGS:")
        print("-" * 70)
        for div in result.divergences[:10]:
            print(f"\n  ⚠️  {div['indicator_1']} and {div['indicator_2']}")
            print(f"      Correlation: {div['correlation']:.2f}")
            print(f"      Intrinsic similarity: {div['intrinsic_similarity']:.2f}")
            print(f"      {div['interpretation']}")

    if not result.divergences:
        print("\n  No divergences detected.")

    conn.close()


if __name__ == "__main__":
    main()
