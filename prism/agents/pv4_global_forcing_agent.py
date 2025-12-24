"""
PV4: Global Forcing Decomposition Agent

Separates global forcing from structural coupling by removing the dominant
system mode (PC1) and re-evaluating geometry.

Key Outputs:
    - Global Forcing Metric (GFM): PC1 variance / total variance
    - Raw vs residual domain coupling
    - Raw vs residual domain cohesion

Interpretation:
    - High GFM with collapsing residual cohesion = phase dispersion
    - Persistent residual coupling = structural alignment
"""

import numpy as np
from sklearn.decomposition import PCA
from dataclasses import dataclass
from typing import Dict


@dataclass
class PV4Result:
    """Result of PV4 Global Forcing analysis."""
    gfm: float  # Global Forcing Metric: PC1 variance / total variance
    pc1_vector: np.ndarray  # The dominant mode vector
    coupling_raw: Dict[str, float]  # Domain-pair coupling before PC1 removal
    coupling_residual: Dict[str, float]  # Domain-pair coupling after PC1 removal
    cohesion_raw: Dict[str, float]  # Within-domain cohesion before PC1 removal
    cohesion_residual: Dict[str, float]  # Within-domain cohesion after PC1 removal


class PV4GlobalForcingAgent:
    """
    PV4 Agent: Global Forcing Decomposition & Phase Coherence Analysis

    This agent computes geometry metrics before and after removing PC1
    to distinguish between:
    - Correlation driven by shared global forcing (high GFM, low residual coupling)
    - Structural alignment that persists beyond the global mode

    Usage:
        state_vectors = {"US_GDP": np.array([...]), "US_10Y": np.array([...]), ...}
        domains = {"US_GDP": "macro", "US_10Y": "bonds", ...}
        agent = PV4GlobalForcingAgent(state_vectors, domains)
        result = agent.run()
        print(f"GFM: {result.gfm:.3f}")
    """

    def __init__(self, state_vectors: Dict[str, np.ndarray], domains: Dict[str, str]):
        """
        Args:
            state_vectors: Dict mapping indicator IDs to their state vectors
            domains: Dict mapping indicator IDs to their domain labels
        """
        self.state_vectors = state_vectors
        self.domains = domains

    def run(self) -> PV4Result:
        """Execute PV4 analysis."""
        ids = list(self.state_vectors.keys())
        X = np.vstack([self.state_vectors[i] for i in ids])

        # Fit PCA to compute global mode
        pca = PCA()
        pca.fit(X)

        # Global Forcing Metric: fraction of variance in PC1
        sigma = pca.explained_variance_
        gfm = float(sigma[0] / sigma.sum())

        # Remove PC1 to get residuals
        pc1 = pca.components_[0]
        a1 = X @ pc1  # Projection onto PC1
        X_res = X - np.outer(a1, pc1)  # Residual after removing PC1

        return PV4Result(
            gfm=gfm,
            pc1_vector=pc1,
            coupling_raw=self._domain_coupling(X, ids),
            coupling_residual=self._domain_coupling(X_res, ids),
            cohesion_raw=self._domain_cohesion(X, ids),
            cohesion_residual=self._domain_cohesion(X_res, ids),
        )

    def _domain_centroids(self, X: np.ndarray, ids: list) -> Dict[str, np.ndarray]:
        """Compute centroid of each domain."""
        centroids = {}
        for i, vec in zip(ids, X):
            d = self.domains[i]
            centroids.setdefault(d, []).append(vec)
        return {d: np.mean(v, axis=0) for d, v in centroids.items()}

    def _domain_coupling(self, X: np.ndarray, ids: list) -> Dict[str, float]:
        """
        Compute coupling between domain pairs as cosine similarity
        of their centroids.
        """
        centroids = self._domain_centroids(X, ids)
        keys = list(centroids.keys())
        coupling = {}
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                a, b = keys[i], keys[j]
                v1, v2 = centroids[a], centroids[b]
                norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
                if norm1 > 1e-10 and norm2 > 1e-10:
                    sim = float(np.dot(v1, v2) / (norm1 * norm2))
                else:
                    sim = 0.0
                coupling[f"{a}__{b}"] = sim
        return coupling

    def _domain_cohesion(self, X: np.ndarray, ids: list) -> Dict[str, float]:
        """
        Compute within-domain cohesion as inverse of mean distance
        from domain centroid.
        """
        cohesion = {}
        centroids = self._domain_centroids(X, ids)
        for d, c in centroids.items():
            members = [X[i] for i, k in enumerate(ids) if self.domains[k] == d]
            if len(members) > 0:
                dist = np.mean([np.linalg.norm(v - c) for v in members])
                cohesion[d] = float(1.0 / (1.0 + dist))
            else:
                cohesion[d] = 0.0
        return cohesion


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)

    state_vectors = {
        "US_GDP": np.random.randn(7),
        "US_10Y": np.random.randn(7),
        "COVID_US": np.random.randn(7),
        "COVID_IT": np.random.randn(7),
        "USGS_EQ": np.random.randn(7),
    }

    domains = {
        "US_GDP": "macro",
        "US_10Y": "bonds",
        "COVID_US": "epidemiology",
        "COVID_IT": "epidemiology",
        "USGS_EQ": "geophysics",
    }

    agent = PV4GlobalForcingAgent(state_vectors, domains)
    result = agent.run()

    print("=" * 60)
    print("PV4: Global Forcing Decomposition")
    print("=" * 60)
    print(f"\nGlobal Forcing Metric (GFM): {result.gfm:.3f}")
    print(f"  -> PC1 explains {result.gfm*100:.1f}% of variance")

    print("\n--- Domain Coupling ---")
    print(f"{'Pair':<25} {'Raw':>10} {'Residual':>10} {'Delta':>10}")
    print("-" * 55)
    for pair in result.coupling_raw:
        raw = result.coupling_raw[pair]
        res = result.coupling_residual[pair]
        delta = res - raw
        print(f"{pair:<25} {raw:>10.3f} {res:>10.3f} {delta:>+10.3f}")

    print("\n--- Domain Cohesion ---")
    print(f"{'Domain':<15} {'Raw':>10} {'Residual':>10} {'Delta':>10}")
    print("-" * 45)
    for domain in result.cohesion_raw:
        raw = result.cohesion_raw[domain]
        res = result.cohesion_residual[domain]
        delta = res - raw
        print(f"{domain:<15} {raw:>10.3f} {res:>10.3f} {delta:>+10.3f}")
