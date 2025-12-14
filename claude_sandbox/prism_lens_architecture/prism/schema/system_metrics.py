# prism/schema/system_metrics.py
"""
System-level geometry metrics.

These are OBSERVATIONS derived from lens outputs, not interpretations.
A scientist looks at these and asks "why?" — PRISM does not answer that.

Metrics:
- effective_dim: How many independent dimensions explain variance?
- lens_agreement: How correlated are lens rankings?
- indicator_concentration: Is influence concentrated or distributed?
"""

from typing import Dict, List, TYPE_CHECKING
from itertools import combinations
import numpy as np

if TYPE_CHECKING:
    from prism.schema.observation import SystemSnapshot


def compute_system_metrics(snapshot: "SystemSnapshot") -> Dict[str, float]:
    """
    Derive system-level geometry from lens observations.

    These are OBSERVATIONS, not interpretations:
    - How aligned are the lenses?
    - How concentrated is influence?
    - What is the effective dimensionality?

    A scientist looks at these and asks "why?"
    PRISM does not answer that.
    """

    metrics = {}

    # 1. Effective dimensionality (PCA on indicator vectors)
    metrics["effective_dim"] = _effective_dimensionality(snapshot)

    # 2. Lens agreement (avg correlation between lens outputs)
    metrics["lens_agreement"] = _lens_agreement(snapshot)

    # 3. Indicator concentration (Herfindahl-like)
    metrics["indicator_concentration"] = _concentration(snapshot)

    # 4. Total observation count (sanity check)
    metrics["n_observations"] = sum(
        len(ig.observations) for ig in snapshot.indicators.values()
    )

    # 5. Indicator count
    metrics["n_indicators"] = len(snapshot.indicators)

    # 6. Lens count
    metrics["n_lenses"] = len(snapshot.lens_names)

    return metrics


def _effective_dimensionality(snapshot: "SystemSnapshot") -> float:
    """
    How many independent dimensions explain the variance?

    Low = indicators moving together (potential instability)
    High = diverse, uncorrelated signals

    Uses SVD entropy of normalized metric matrix.
    """
    # Build matrix: rows = indicators, cols = normalized metric values
    rows = []
    for ind_geo in snapshot.indicators.values():
        row = []
        for obs in sorted(ind_geo.observations, key=lambda o: o.lens):
            # Use std as canonical metric, fallback to entropy
            val = obs.metrics.get("std", obs.metrics.get("entropy", 0))
            row.append(val)
        if row:
            rows.append(row)

    if len(rows) < 2:
        return float(len(rows))

    # Ensure all rows have same length
    max_len = max(len(r) for r in rows)
    if any(len(r) != max_len for r in rows):
        # Pad with zeros if needed (shouldn't happen in normal use)
        rows = [r + [0] * (max_len - len(r)) for r in rows]

    if max_len < 2:
        return float(len(rows))

    matrix = np.array(rows, dtype=float)

    # Normalize columns
    col_std = matrix.std(axis=0)
    col_std[col_std == 0] = 1  # Avoid division by zero
    matrix = (matrix - matrix.mean(axis=0)) / col_std

    # SVD
    try:
        _, s, _ = np.linalg.svd(matrix, full_matrices=False)
        # Effective rank via entropy of singular values
        s_sum = s.sum()
        if s_sum == 0:
            return float(len(rows))
        s_norm = s / s_sum
        s_norm = s_norm[s_norm > 0]
        eff_dim = np.exp(-np.sum(s_norm * np.log(s_norm + 1e-12)))
        return float(eff_dim)
    except np.linalg.LinAlgError:
        return float(len(rows))


def _lens_agreement(snapshot: "SystemSnapshot") -> float:
    """
    How much do different lenses agree on indicator rankings?

    High = lenses see similar structure
    Low = lenses see different things (more information diversity)

    Uses average Spearman rank correlation between lens pairs.
    """
    # For each lens, rank indicators by a canonical metric
    lens_rankings = {}

    all_lenses = _get_all_lenses(snapshot)

    for lens in all_lenses:
        scores = []
        for ind_name, ind_geo in snapshot.indicators.items():
            for obs in ind_geo.observations:
                if obs.lens == lens:
                    # Use std or entropy as score
                    score = obs.metrics.get("std", obs.metrics.get("entropy", 0))
                    scores.append((ind_name, score))

        if scores:
            sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
            lens_rankings[lens] = [s[0] for s in sorted_scores]

    if len(lens_rankings) < 2:
        return 1.0

    # Average rank correlation between all lens pairs
    correlations = []

    for l1, l2 in combinations(lens_rankings.keys(), 2):
        r1, r2 = lens_rankings[l1], lens_rankings[l2]
        corr = _rank_correlation(r1, r2)
        if corr is not None:
            correlations.append(corr)

    return float(np.mean(correlations)) if correlations else 1.0


def _rank_correlation(rank1: List[str], rank2: List[str]) -> float:
    """
    Spearman-style rank correlation between two rankings.

    Returns value in [-1, 1] where 1 = perfect agreement.
    """
    # Find common elements
    common = set(rank1) & set(rank2)
    if len(common) < 2:
        return None

    # Filter to common elements preserving order
    r1 = [r for r in rank1 if r in common]
    r2 = [r for r in rank2 if r in common]

    n = len(r1)
    rank_map1 = {v: i for i, v in enumerate(r1)}
    rank_map2 = {v: i for i, v in enumerate(r2)}

    d_sq = sum((rank_map1[v] - rank_map2[v]) ** 2 for v in r1)

    # Spearman formula
    return 1 - (6 * d_sq) / (n * (n ** 2 - 1))


def _concentration(snapshot: "SystemSnapshot") -> float:
    """
    Is variance concentrated in few indicators or spread out?

    High (-> 1.0) = few indicators dominate
    Low (-> 0.0) = influence is distributed evenly

    Uses Herfindahl-Hirschman Index on std values.
    """
    stds = []
    for ind_geo in snapshot.indicators.values():
        # Average std across all lenses for this indicator
        ind_stds = []
        for obs in ind_geo.observations:
            if "std" in obs.metrics:
                ind_stds.append(obs.metrics["std"])
        if ind_stds:
            stds.append(np.mean(ind_stds))

    if not stds or sum(stds) == 0:
        return 0.0

    total = sum(stds)
    shares = [s / total for s in stds]
    hhi = sum(s ** 2 for s in shares)

    return float(hhi)


def _get_all_lenses(snapshot: "SystemSnapshot") -> set:
    """Get all unique lens names from snapshot."""
    return {
        obs.lens
        for ig in snapshot.indicators.values()
        for obs in ig.observations
    }
