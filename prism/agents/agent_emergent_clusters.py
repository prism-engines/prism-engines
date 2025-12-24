"""
PRISM Emergent Cluster Detection

Let the math find the masses. No human-defined thresholds.

Multiple approaches that converge on the same question:
"Where are the natural groupings in this data?"

Methods:
1. EIGENGAP: Spectral clustering with automatic k detection
2. DENDROGRAM: Hierarchical clustering with gap analysis
3. MODULARITY: Network community detection
4. GRAVITATIONAL: N-body simulation - let similar points attract
5. PERSISTENT HOMOLOGY: Topological features that survive across scales

Each method votes on cluster structure. Consensus = robust finding.

Usage:
    detector = EmergentClusterDetector()
    result = detector.detect_clusters(similarity_matrix, indicator_ids)

    print(f"Found {result.n_clusters} natural clusters")
    print(f"Confidence: {result.confidence}")
    for cluster in result.clusters:
        print(f"  {cluster.members}")

Author: Jason (PRISM Project)
Date: December 2024
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set, Any
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class EmergentCluster:
    """A cluster found by letting the math decide."""
    cluster_id: int
    members: List[str]

    # How it was found
    detection_method: str

    # Strength metrics
    internal_density: float      # How tight is it?
    external_separation: float   # How far from others?
    stability: float             # How robust across methods?

    # Centroid
    centroid: np.ndarray = None

    # Mass (could be count, could be weighted)
    mass: float = 0.0


@dataclass
class ClusteringResult:
    """Result of emergent cluster detection."""

    # The clusters
    n_clusters: int
    clusters: List[EmergentCluster]
    assignments: Dict[str, int]  # indicator_id -> cluster_id

    # Confidence
    confidence: float            # 0-1, how sure are we?
    method_agreement: float      # Do methods agree?

    # Per-method results
    method_results: Dict[str, List[List[str]]]

    # The "why"
    eigengap_location: int       # Where was the gap?
    dendrogram_cut_height: float # Where did we cut?
    modularity_score: float      # How modular is the network?

    # Singletons (not in any cluster)
    singletons: List[str]

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "EMERGENT CLUSTER DETECTION RESULTS",
            "=" * 60,
            "",
            f"Natural clusters found: {self.n_clusters}",
            f"Confidence: {self.confidence:.1%}",
            f"Method agreement: {self.method_agreement:.1%}",
            "",
            "CLUSTERS:",
        ]

        for c in self.clusters:
            lines.append(f"  Cluster {c.cluster_id}:")
            lines.append(f"    Members: {', '.join(c.members)}")
            lines.append(f"    Internal density: {c.internal_density:.3f}")
            lines.append(f"    Separation: {c.external_separation:.3f}")
            lines.append(f"    Stability: {c.stability:.1%}")
            lines.append("")

        if self.singletons:
            lines.append(f"SINGLETONS: {', '.join(self.singletons)}")
            lines.append("")

        lines.append("DETECTION DETAILS:")
        lines.append(f"  Eigengap at: {self.eigengap_location}")
        lines.append(f"  Dendrogram cut: {self.dendrogram_cut_height:.3f}")
        lines.append(f"  Modularity: {self.modularity_score:.3f}")

        return '\n'.join(lines)


# =============================================================================
# EIGENGAP METHOD
# =============================================================================

class EigengapDetector:
    """
    Find clusters using spectral gap.

    The number of clusters = location of largest gap in Laplacian eigenvalues.
    No threshold needed - the gap tells us.
    """

    def detect(
        self,
        similarity_matrix: np.ndarray,
        indicator_ids: List[str],
        max_clusters: int = None
    ) -> Tuple[List[List[str]], int, float]:
        """
        Detect clusters via eigengap.

        Returns:
            (clusters, gap_location, gap_size)
        """
        n = len(indicator_ids)
        max_clusters = max_clusters or n // 2

        # Build graph Laplacian
        # Ensure similarity is non-negative
        W = np.maximum(similarity_matrix, 0)
        np.fill_diagonal(W, 0)  # No self-loops

        # Degree matrix
        D = np.diag(np.sum(W, axis=1))

        # Normalized Laplacian: L = I - D^(-1/2) W D^(-1/2)
        D_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(np.diag(D), 1e-10)))
        L = np.eye(n) - D_inv_sqrt @ W @ D_inv_sqrt

        # Eigenvalues (sorted ascending)
        try:
            eigenvalues = np.linalg.eigvalsh(L)
            eigenvalues = np.sort(eigenvalues)
        except:
            return [[ind] for ind in indicator_ids], n, 0.0

        # Find largest gap
        gaps = np.diff(eigenvalues[:max_clusters + 1])

        if len(gaps) == 0:
            return [[ind] for ind in indicator_ids], n, 0.0

        gap_location = np.argmax(gaps) + 1  # +1 because diff reduces length
        gap_size = gaps[gap_location - 1]

        # Cluster using k = gap_location
        k = max(1, min(gap_location, n - 1))

        clusters = self._spectral_cluster(W, indicator_ids, k)

        return clusters, gap_location, gap_size

    def _spectral_cluster(
        self,
        W: np.ndarray,
        indicator_ids: List[str],
        k: int
    ) -> List[List[str]]:
        """Perform spectral clustering with k clusters."""
        from scipy.cluster.hierarchy import linkage, fcluster

        n = len(indicator_ids)

        # Build normalized Laplacian
        D = np.diag(np.sum(W, axis=1))
        D_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(np.diag(D), 1e-10)))
        L = np.eye(n) - D_inv_sqrt @ W @ D_inv_sqrt

        # Get first k eigenvectors
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(L)
            idx = np.argsort(eigenvalues)[:k]
            V = eigenvectors[:, idx]
        except:
            return [[ind] for ind in indicator_ids]

        # Normalize rows
        row_norms = np.linalg.norm(V, axis=1, keepdims=True)
        V = V / np.maximum(row_norms, 1e-10)

        # Cluster in spectral space
        try:
            from scipy.cluster.vq import kmeans2
            _, labels = kmeans2(V, k, minit='++')
        except:
            # Fallback to hierarchical
            if V.shape[0] > 1:
                Z = linkage(V, method='ward')
                labels = fcluster(Z, k, criterion='maxclust') - 1
            else:
                labels = np.array([0])

        # Build cluster lists
        clusters = defaultdict(list)
        for ind_id, label in zip(indicator_ids, labels):
            clusters[label].append(ind_id)

        return list(clusters.values())


# =============================================================================
# DENDROGRAM GAP METHOD
# =============================================================================

class DendrogramGapDetector:
    """
    Find clusters using hierarchical clustering with automatic cut.

    Cut where the biggest jump in merge distance occurs.
    That's where natural clusters exist.
    """

    def detect(
        self,
        distance_matrix: np.ndarray,
        indicator_ids: List[str]
    ) -> Tuple[List[List[str]], float, List[float]]:
        """
        Detect clusters via dendrogram gap.

        Returns:
            (clusters, cut_height, merge_distances)
        """
        from scipy.cluster.hierarchy import linkage, fcluster
        from scipy.spatial.distance import squareform

        n = len(indicator_ids)

        if n < 2:
            return [[ind] for ind in indicator_ids], 0.0, []

        # Convert to condensed form
        try:
            condensed = squareform(distance_matrix, checks=False)
        except:
            return [[ind] for ind in indicator_ids], 0.0, []

        # Hierarchical clustering
        Z = linkage(condensed, method='average')

        # Merge distances (4th column of Z)
        merge_distances = Z[:, 2]

        if len(merge_distances) < 2:
            return [[ind] for ind in indicator_ids], 0.0, list(merge_distances)

        # Find biggest gap
        gaps = np.diff(merge_distances)
        gap_idx = np.argmax(gaps)

        # Cut just above the biggest gap
        cut_height = merge_distances[gap_idx] + gaps[gap_idx] / 2

        # Get clusters
        labels = fcluster(Z, cut_height, criterion='distance')

        # Build cluster lists
        clusters = defaultdict(list)
        for ind_id, label in zip(indicator_ids, labels):
            clusters[label].append(ind_id)

        return list(clusters.values()), cut_height, list(merge_distances)


# =============================================================================
# MODULARITY METHOD
# =============================================================================

class ModularityDetector:
    """
    Find clusters using network modularity optimization.

    Modularity measures: edges within clusters vs expected by chance.
    High modularity = natural community structure.
    """

    def detect(
        self,
        similarity_matrix: np.ndarray,
        indicator_ids: List[str],
        resolution: float = 1.0
    ) -> Tuple[List[List[str]], float]:
        """
        Detect clusters via modularity optimization.

        Returns:
            (clusters, modularity_score)
        """
        n = len(indicator_ids)

        # Build adjacency matrix (threshold weak edges)
        A = np.maximum(similarity_matrix, 0)
        np.fill_diagonal(A, 0)

        # Try different numbers of clusters, pick highest modularity
        best_modularity = -1
        best_clusters = [[ind] for ind in indicator_ids]

        for k in range(2, min(n, 10)):
            clusters = self._partition(A, indicator_ids, k)
            modularity = self._compute_modularity(A, clusters, indicator_ids)

            if modularity > best_modularity:
                best_modularity = modularity
                best_clusters = clusters

        # Compare to all-in-one
        single_modularity = self._compute_modularity(A, [indicator_ids], indicator_ids)
        if single_modularity > best_modularity:
            best_modularity = single_modularity
            best_clusters = [indicator_ids]

        return best_clusters, best_modularity

    def _partition(
        self,
        A: np.ndarray,
        indicator_ids: List[str],
        k: int
    ) -> List[List[str]]:
        """Partition into k clusters using spectral method."""
        from scipy.cluster.hierarchy import linkage, fcluster

        n = len(indicator_ids)

        # Use Laplacian eigenvectors
        D = np.diag(np.sum(A, axis=1))
        L = D - A

        try:
            eigenvalues, eigenvectors = np.linalg.eigh(L)
            idx = np.argsort(eigenvalues)[1:k+1]  # Skip first (trivial)
            V = eigenvectors[:, idx]
        except:
            return [[ind] for ind in indicator_ids]

        # Cluster
        if V.shape[0] > 1 and V.shape[1] > 0:
            Z = linkage(V, method='ward')
            labels = fcluster(Z, k, criterion='maxclust')
        else:
            labels = np.zeros(n, dtype=int)

        clusters = defaultdict(list)
        for ind_id, label in zip(indicator_ids, labels):
            clusters[label].append(ind_id)

        return list(clusters.values())

    def _compute_modularity(
        self,
        A: np.ndarray,
        clusters: List[List[str]],
        indicator_ids: List[str]
    ) -> float:
        """Compute modularity score."""
        id_to_idx = {ind: i for i, ind in enumerate(indicator_ids)}
        n = len(indicator_ids)
        m = np.sum(A) / 2  # Total edge weight

        if m == 0:
            return 0.0

        # Degree of each node
        k = np.sum(A, axis=1)

        Q = 0.0
        for cluster in clusters:
            for i_id in cluster:
                for j_id in cluster:
                    i = id_to_idx[i_id]
                    j = id_to_idx[j_id]
                    Q += A[i, j] - (k[i] * k[j]) / (2 * m)

        Q /= (2 * m)
        return Q


# =============================================================================
# GRAVITATIONAL SIMULATION
# =============================================================================

class GravitationalDetector:
    """
    N-body simulation: let similar indicators attract.

    No threshold. Let the system find equilibrium.
    Where particles coalesce = cluster.
    """

    def detect(
        self,
        similarity_matrix: np.ndarray,
        indicator_ids: List[str],
        n_steps: int = 100,
        dt: float = 0.1,
        merge_threshold: float = 0.1
    ) -> Tuple[List[List[str]], List[np.ndarray]]:
        """
        Detect clusters via gravitational simulation.

        Returns:
            (clusters, final_positions)
        """
        n = len(indicator_ids)

        # Initialize positions (random or from MDS)
        positions = self._initial_positions(similarity_matrix, n)
        velocities = np.zeros_like(positions)

        # Mass proportional to total similarity
        masses = np.sum(similarity_matrix, axis=1)
        masses = masses / np.max(masses)  # Normalize

        # Simulation
        for step in range(n_steps):
            # Compute forces
            forces = np.zeros_like(positions)

            for i in range(n):
                for j in range(i + 1, n):
                    # Direction
                    r = positions[j] - positions[i]
                    dist = np.linalg.norm(r)

                    if dist < 1e-6:
                        continue

                    r_hat = r / dist

                    # Attraction proportional to similarity
                    # Repulsion at very close range (prevents collapse)
                    attraction = similarity_matrix[i, j] / (dist + 0.1)
                    repulsion = 0.01 / (dist ** 2 + 0.01) if dist < 0.2 else 0

                    force_magnitude = attraction - repulsion
                    force = force_magnitude * r_hat

                    forces[i] += force * masses[j]
                    forces[j] -= force * masses[i]

            # Update velocities and positions
            velocities = 0.9 * velocities + forces * dt  # Damping
            positions += velocities * dt

        # Find clusters from final positions
        clusters = self._cluster_from_positions(
            positions, indicator_ids, merge_threshold
        )

        return clusters, positions

    def _initial_positions(
        self,
        similarity_matrix: np.ndarray,
        n: int
    ) -> np.ndarray:
        """Initialize positions using MDS."""
        from scipy.spatial.distance import squareform

        # Convert similarity to distance
        distance_matrix = 1 - np.clip(similarity_matrix, 0, 1)
        np.fill_diagonal(distance_matrix, 0)

        try:
            from sklearn.manifold import MDS
            mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
            positions = mds.fit_transform(distance_matrix)
        except:
            # Random initialization
            positions = np.random.randn(n, 2)

        return positions

    def _cluster_from_positions(
        self,
        positions: np.ndarray,
        indicator_ids: List[str],
        threshold: float
    ) -> List[List[str]]:
        """Find clusters from final positions using DBSCAN-like approach."""
        n = len(indicator_ids)

        # Compute pairwise distances in position space
        distances = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(positions[i] - positions[j])
                distances[i, j] = d
                distances[j, i] = d

        # Adaptive threshold: median distance / 2
        if threshold is None:
            threshold = np.median(distances[distances > 0]) / 2

        # Simple clustering: union-find
        parent = list(range(n))

        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        for i in range(n):
            for j in range(i + 1, n):
                if distances[i, j] < threshold:
                    union(i, j)

        # Build clusters
        clusters = defaultdict(list)
        for i, ind_id in enumerate(indicator_ids):
            clusters[find(i)].append(ind_id)

        return list(clusters.values())


# =============================================================================
# PERSISTENT HOMOLOGY (Simplified)
# =============================================================================

class PersistenceDetector:
    """
    Find clusters that persist across scales.

    Vary the threshold and see which groupings survive.
    Persistent = robust = real structure.
    """

    def detect(
        self,
        distance_matrix: np.ndarray,
        indicator_ids: List[str],
        n_thresholds: int = 50
    ) -> Tuple[List[List[str]], Dict[str, float]]:
        """
        Detect clusters via persistence.

        Returns:
            (clusters, persistence_scores)
        """
        n = len(indicator_ids)

        # Range of thresholds
        max_dist = np.max(distance_matrix)
        thresholds = np.linspace(0.01, max_dist, n_thresholds)

        # Track cluster counts at each threshold
        cluster_counts = []
        cluster_history = []

        for thresh in thresholds:
            clusters = self._threshold_cluster(distance_matrix, indicator_ids, thresh)
            cluster_counts.append(len(clusters))
            cluster_history.append(clusters)

        # Find most persistent number of clusters
        # (the one that survives longest)
        count_persistence = defaultdict(int)
        for count in cluster_counts:
            count_persistence[count] += 1

        # Most common cluster count (excluding 1 and n)
        valid_counts = {k: v for k, v in count_persistence.items() if 1 < k < n}

        if valid_counts:
            best_k = max(valid_counts, key=valid_counts.get)
        else:
            best_k = 1

        # Get clusters at threshold where we have best_k clusters
        for i, count in enumerate(cluster_counts):
            if count == best_k:
                final_clusters = cluster_history[i]
                break
        else:
            final_clusters = [[ind] for ind in indicator_ids]

        # Compute persistence scores per indicator
        persistence_scores = {}
        for ind_id in indicator_ids:
            # How many thresholds was this indicator in the same cluster as its final cluster-mates?
            final_mates = set()
            for cluster in final_clusters:
                if ind_id in cluster:
                    final_mates = set(cluster)
                    break

            score = 0
            for clusters in cluster_history:
                for cluster in clusters:
                    if ind_id in cluster:
                        current_mates = set(cluster)
                        overlap = len(final_mates & current_mates) / len(final_mates | current_mates)
                        score += overlap
                        break

            persistence_scores[ind_id] = score / n_thresholds

        return final_clusters, persistence_scores

    def _threshold_cluster(
        self,
        distance_matrix: np.ndarray,
        indicator_ids: List[str],
        threshold: float
    ) -> List[List[str]]:
        """Cluster at a specific distance threshold."""
        n = len(indicator_ids)

        # Union-find
        parent = list(range(n))

        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        for i in range(n):
            for j in range(i + 1, n):
                if distance_matrix[i, j] <= threshold:
                    union(i, j)

        clusters = defaultdict(list)
        for i, ind_id in enumerate(indicator_ids):
            clusters[find(i)].append(ind_id)

        return list(clusters.values())


# =============================================================================
# MAIN DETECTOR (ENSEMBLE)
# =============================================================================

class EmergentClusterDetector:
    """
    Ensemble cluster detection: let multiple methods vote.

    No human thresholds. The math decides.
    """

    def __init__(self):
        self.eigengap = EigengapDetector()
        self.dendrogram = DendrogramGapDetector()
        self.modularity = ModularityDetector()
        self.gravitational = GravitationalDetector()
        self.persistence = PersistenceDetector()

    def detect_clusters(
        self,
        similarity_matrix: np.ndarray,
        indicator_ids: List[str],
        methods: List[str] = None
    ) -> ClusteringResult:
        """
        Detect natural clusters using ensemble of methods.

        Args:
            similarity_matrix: n x n similarity (not distance)
            indicator_ids: List of indicator names
            methods: Which methods to use (default: all)

        Returns:
            ClusteringResult with natural clusters
        """
        n = len(indicator_ids)

        if methods is None:
            methods = ['eigengap', 'dendrogram', 'modularity', 'gravitational', 'persistence']

        # Convert similarity to distance for methods that need it
        distance_matrix = 1 - np.clip(similarity_matrix, 0, 1)
        np.fill_diagonal(distance_matrix, 0)

        # Run each method
        method_results = {}
        method_metadata = {}

        if 'eigengap' in methods:
            try:
                clusters, gap_loc, gap_size = self.eigengap.detect(
                    similarity_matrix, indicator_ids
                )
                method_results['eigengap'] = clusters
                method_metadata['eigengap'] = {'gap_location': gap_loc, 'gap_size': gap_size}
            except Exception as e:
                logger.warning(f"Eigengap failed: {e}")

        if 'dendrogram' in methods:
            try:
                clusters, cut_height, _ = self.dendrogram.detect(
                    distance_matrix, indicator_ids
                )
                method_results['dendrogram'] = clusters
                method_metadata['dendrogram'] = {'cut_height': cut_height}
            except Exception as e:
                logger.warning(f"Dendrogram failed: {e}")

        if 'modularity' in methods:
            try:
                clusters, mod_score = self.modularity.detect(
                    similarity_matrix, indicator_ids
                )
                method_results['modularity'] = clusters
                method_metadata['modularity'] = {'score': mod_score}
            except Exception as e:
                logger.warning(f"Modularity failed: {e}")

        if 'gravitational' in methods:
            try:
                clusters, positions = self.gravitational.detect(
                    similarity_matrix, indicator_ids
                )
                method_results['gravitational'] = clusters
                method_metadata['gravitational'] = {'positions': positions}
            except Exception as e:
                logger.warning(f"Gravitational failed: {e}")

        if 'persistence' in methods:
            try:
                clusters, persistence_scores = self.persistence.detect(
                    distance_matrix, indicator_ids
                )
                method_results['persistence'] = clusters
                method_metadata['persistence'] = {'scores': persistence_scores}
            except Exception as e:
                logger.warning(f"Persistence failed: {e}")

        # Consensus clustering
        final_clusters, confidence, agreement = self._consensus(
            method_results, indicator_ids
        )

        # Build EmergentCluster objects
        clusters = []
        assignments = {}

        for i, members in enumerate(final_clusters):
            if len(members) == 1:
                continue  # Will be singleton

            # Compute cluster metrics
            internal_density = self._compute_internal_density(
                members, similarity_matrix, indicator_ids
            )
            external_separation = self._compute_external_separation(
                members, final_clusters, similarity_matrix, indicator_ids
            )
            stability = self._compute_stability(members, method_results)

            cluster = EmergentCluster(
                cluster_id=i,
                members=members,
                detection_method='ensemble',
                internal_density=internal_density,
                external_separation=external_separation,
                stability=stability,
                mass=float(len(members)),
            )
            clusters.append(cluster)

            for member in members:
                assignments[member] = i

        # Identify singletons
        singletons = []
        for members in final_clusters:
            if len(members) == 1:
                singletons.append(members[0])
                assignments[members[0]] = -1

        return ClusteringResult(
            n_clusters=len(clusters),
            clusters=clusters,
            assignments=assignments,
            confidence=confidence,
            method_agreement=agreement,
            method_results={k: v for k, v in method_results.items()},
            eigengap_location=method_metadata.get('eigengap', {}).get('gap_location', 0),
            dendrogram_cut_height=method_metadata.get('dendrogram', {}).get('cut_height', 0),
            modularity_score=method_metadata.get('modularity', {}).get('score', 0),
            singletons=singletons,
        )

    def _consensus(
        self,
        method_results: Dict[str, List[List[str]]],
        indicator_ids: List[str]
    ) -> Tuple[List[List[str]], float, float]:
        """
        Build consensus clustering from multiple methods.

        Uses co-occurrence: how often are two indicators in the same cluster?
        """
        n = len(indicator_ids)
        id_to_idx = {ind: i for i, ind in enumerate(indicator_ids)}

        # Co-occurrence matrix
        cooccur = np.zeros((n, n))
        n_methods = len(method_results)

        if n_methods == 0:
            # No methods succeeded, each indicator is its own cluster
            return [[ind] for ind in indicator_ids], 0.0, 0.0

        for method, clusters in method_results.items():
            for cluster in clusters:
                for ind1 in cluster:
                    for ind2 in cluster:
                        i = id_to_idx.get(ind1)
                        j = id_to_idx.get(ind2)
                        if i is not None and j is not None:
                            cooccur[i, j] += 1

        # Normalize
        cooccur /= n_methods

        # Cluster based on co-occurrence
        # Threshold: together in > 50% of methods
        threshold = 0.5

        parent = list(range(n))

        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        for i in range(n):
            for j in range(i + 1, n):
                if cooccur[i, j] > threshold:
                    union(i, j)

        # Build clusters
        clusters = defaultdict(list)
        for i, ind_id in enumerate(indicator_ids):
            clusters[find(i)].append(ind_id)

        final_clusters = list(clusters.values())

        # Compute agreement (mean co-occurrence for same-cluster pairs)
        same_cluster_cooccur = []
        for cluster in final_clusters:
            for i, ind1 in enumerate(cluster):
                for ind2 in cluster[i+1:]:
                    idx1 = id_to_idx[ind1]
                    idx2 = id_to_idx[ind2]
                    same_cluster_cooccur.append(cooccur[idx1, idx2])

        agreement = np.mean(same_cluster_cooccur) if same_cluster_cooccur else 0.0

        # Confidence: agreement * method count factor
        confidence = agreement * (n_methods / 5)  # Max 5 methods
        confidence = np.clip(confidence, 0, 1)

        return final_clusters, confidence, agreement

    def _compute_internal_density(
        self,
        members: List[str],
        similarity_matrix: np.ndarray,
        indicator_ids: List[str]
    ) -> float:
        """Average similarity within cluster."""
        if len(members) < 2:
            return 1.0

        id_to_idx = {ind: i for i, ind in enumerate(indicator_ids)}

        total = 0
        count = 0
        for i, m1 in enumerate(members):
            for m2 in members[i+1:]:
                idx1 = id_to_idx.get(m1)
                idx2 = id_to_idx.get(m2)
                if idx1 is not None and idx2 is not None:
                    total += similarity_matrix[idx1, idx2]
                    count += 1

        return total / count if count > 0 else 0.0

    def _compute_external_separation(
        self,
        members: List[str],
        all_clusters: List[List[str]],
        similarity_matrix: np.ndarray,
        indicator_ids: List[str]
    ) -> float:
        """Average distance to other clusters."""
        id_to_idx = {ind: i for i, ind in enumerate(indicator_ids)}
        members_set = set(members)

        total = 0
        count = 0

        for other_cluster in all_clusters:
            if set(other_cluster) == members_set:
                continue  # Skip self

            for m1 in members:
                for m2 in other_cluster:
                    idx1 = id_to_idx.get(m1)
                    idx2 = id_to_idx.get(m2)
                    if idx1 is not None and idx2 is not None:
                        total += 1 - similarity_matrix[idx1, idx2]  # Distance
                        count += 1

        return total / count if count > 0 else 1.0

    def _compute_stability(
        self,
        members: List[str],
        method_results: Dict[str, List[List[str]]]
    ) -> float:
        """How consistently do methods agree on this cluster?"""
        members_set = set(members)
        n_methods = len(method_results)

        if n_methods == 0:
            return 0.0

        matches = 0
        for method, clusters in method_results.items():
            for cluster in clusters:
                cluster_set = set(cluster)
                # Jaccard similarity
                overlap = len(members_set & cluster_set) / len(members_set | cluster_set)
                if overlap > 0.8:  # Close match
                    matches += 1
                    break

        return matches / n_methods


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def find_natural_clusters(
    similarity_matrix: np.ndarray,
    indicator_ids: List[str]
) -> ClusteringResult:
    """
    One-liner to find natural clusters.

    Usage:
        corr = df.corr().values
        result = find_natural_clusters(corr, df.columns.tolist())
        print(result.summary())
    """
    detector = EmergentClusterDetector()
    return detector.detect_clusters(similarity_matrix, indicator_ids)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="PRISM Emergent Cluster Detection")
    parser.add_argument('--demo', action='store_true', help='Run demo')

    args = parser.parse_args()

    if args.demo:
        print("Running demo with synthetic data...\n")

        # Create synthetic similarity matrix with known structure
        # 3 clusters: [0,1,2], [3,4,5], [6,7]
        n = 8
        indicator_ids = [f'IND_{i}' for i in range(n)]

        # Base random
        np.random.seed(42)
        similarity = np.random.rand(n, n) * 0.3

        # Add cluster structure
        clusters = [[0,1,2], [3,4,5], [6,7]]
        for cluster in clusters:
            for i in cluster:
                for j in cluster:
                    similarity[i,j] = 0.8 + np.random.rand() * 0.15

        # Make symmetric
        similarity = (similarity + similarity.T) / 2
        np.fill_diagonal(similarity, 1)

        print("Synthetic similarity matrix (3 clusters expected):")
        print(pd.DataFrame(similarity, index=indicator_ids, columns=indicator_ids).round(2))
        print()

        # Detect
        result = find_natural_clusters(similarity, indicator_ids)
        print(result.summary())
    else:
        parser.print_help()
