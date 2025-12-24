"""
PRISM Correlation Decontamination Agent

Detects when geometry is just recapitulating correlation.
Flags trivial findings. Recomputes to find what's underneath.

Position in Pipeline:
    P2: Derived (per indicator)
        ↓
    P2.5: Correlation Decontamination Agent  ← THIS
        ↓
    P3: Structure (system geometry)
        ↓
    P4: Interpretation

What This Catches:
    | Cohort         | Naive Finding         | After Decontamination              |
    |----------------|----------------------|-----------------------------------|
    | Treasury bonds | "All same space"     | "TLT leads IEF by 2 days"         |
    | Tech stocks    | "FAANG clusters"     | "AAPL diverges on PC2 in earnings"|
    | Oil & gas      | "Energy moves together"| "Refiners lead producers by 1wk" |

The shared component is known. The residual is the finding.

Cross-validated by: Claude
Date: December 2024
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import logging

import duckdb

logger = logging.getLogger(__name__)


# =============================================================================
# THRESHOLDS
# =============================================================================

CORRELATION_DOMINANCE_THRESHOLD = 0.70  # Mean correlation indicating single-factor dominance
PC1_VARIANCE_THRESHOLD = 0.70  # If PC1 > 70%, cohort is dominated by shared factor
MIN_DIMENSION_EXPANSION = 2.0  # Residual dim / raw dim > 2 means hidden structure


# =============================================================================
# DECONTAMINATION RESULT
# =============================================================================

@dataclass
class DecontaminationResult:
    """Result of correlation decontamination analysis."""

    cohort_id: str
    window_start: str
    window_end: str
    indicators: List[str]

    # Contamination detection
    is_contaminated: bool = False
    mean_correlation: float = 0.0
    pc1_variance_explained: float = 0.0

    # Raw geometry metrics
    raw_effective_dimension: float = 0.0
    raw_cohorts: int = 0
    raw_network_density: float = 0.0

    # Residual geometry metrics (after PC1 removal)
    residual_effective_dimension: float = 0.0
    residual_cohorts: int = 0
    residual_network_density: float = 0.0
    residual_mean_correlation: float = 0.0

    # Structure discovery
    hidden_structure_found: bool = False
    dimension_expansion_ratio: float = 0.0

    # Correlation matrices
    raw_correlation_matrix: np.ndarray = None
    residual_correlation_matrix: np.ndarray = None

    # PC1 loadings (what the shared factor looks like)
    pc1_loadings: Dict[str, float] = field(default_factory=dict)

    # Rankings
    raw_rankings: List[Tuple[str, float]] = field(default_factory=list)
    residual_rankings: List[Tuple[str, float]] = field(default_factory=list)

    # Interpretation
    warning: Optional[str] = None
    interpretation: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for persistence."""
        return {
            'cohort_id': self.cohort_id,
            'window_start': self.window_start,
            'window_end': self.window_end,
            'indicators': self.indicators,
            'is_contaminated': self.is_contaminated,
            'mean_correlation': self.mean_correlation,
            'pc1_variance_explained': self.pc1_variance_explained,
            'raw_effective_dimension': self.raw_effective_dimension,
            'residual_effective_dimension': self.residual_effective_dimension,
            'hidden_structure_found': self.hidden_structure_found,
            'dimension_expansion_ratio': self.dimension_expansion_ratio,
            'warning': self.warning,
            'interpretation': self.interpretation,
        }


# =============================================================================
# CORRELATION DECONTAMINATION AGENT
# =============================================================================

class CorrelationDecontaminationAgent:
    """
    Detects when geometry is just recapitulating correlation.

    Flags trivial findings. Recomputes to find what's underneath.
    """

    def __init__(self, conn: duckdb.DuckDBPyConnection):
        self.conn = conn

    def analyze_cohort(
        self,
        cohort_indicators: List[str],
        window_start: str,
        window_end: str,
        cohort_id: str = "default"
    ) -> DecontaminationResult:
        """
        Analyze a cohort for correlation contamination.

        Args:
            cohort_indicators: List of indicator IDs in the cohort
            window_start: Analysis window start
            window_end: Analysis window end
            cohort_id: Identifier for this cohort

        Returns:
            DecontaminationResult with raw and residual analysis
        """
        logger.info(f"Analyzing cohort '{cohort_id}' with {len(cohort_indicators)} indicators")

        result = DecontaminationResult(
            cohort_id=cohort_id,
            window_start=window_start,
            window_end=window_end,
            indicators=cohort_indicators,
        )

        # Load price data
        data = self._load_indicator_data(cohort_indicators, window_start, window_end)
        if not data or len(data) < 2:
            logger.warning(f"Insufficient data for cohort {cohort_id}")
            result.interpretation = "Insufficient data for analysis"
            return result

        # Compute raw correlation matrix
        corr_matrix, aligned_indicators = self._compute_correlation_matrix(data)
        if corr_matrix is None:
            result.interpretation = "Could not compute correlations"
            return result

        result.raw_correlation_matrix = corr_matrix
        result.indicators = aligned_indicators

        # Step 1: Check if dominated by single factor
        result.mean_correlation = self._compute_mean_correlation(corr_matrix)
        pc1_variance, pc1_loadings, eigenvalues, eigenvectors = self._extract_pc1(corr_matrix)
        result.pc1_variance_explained = pc1_variance
        result.pc1_loadings = {ind: float(pc1_loadings[i]) for i, ind in enumerate(aligned_indicators)}

        # Compute raw geometry metrics
        result.raw_effective_dimension = self._compute_effective_dimension(eigenvalues)
        result.raw_cohorts = self._compute_cohort_count(corr_matrix)
        result.raw_network_density = self._compute_network_density(corr_matrix)

        # Step 2: Check contamination threshold
        if result.mean_correlation > CORRELATION_DOMINANCE_THRESHOLD or \
           result.pc1_variance_explained > PC1_VARIANCE_THRESHOLD:

            result.is_contaminated = True
            result.warning = 'correlation_dominated'

            logger.info(f"Cohort is correlation-dominated (mean_corr={result.mean_correlation:.2f}, "
                       f"PC1={result.pc1_variance_explained:.1%})")

            # Step 3: Extract shared component and compute residuals
            residual_corr = self._compute_residual_correlations(
                corr_matrix, eigenvectors, eigenvalues, n_factors_to_remove=1
            )
            result.residual_correlation_matrix = residual_corr

            # Step 4: Compute residual geometry
            residual_eigenvalues = np.linalg.eigvalsh(residual_corr)
            residual_eigenvalues = np.sort(residual_eigenvalues)[::-1]

            result.residual_effective_dimension = self._compute_effective_dimension(residual_eigenvalues)
            result.residual_cohorts = self._compute_cohort_count(residual_corr)
            result.residual_network_density = self._compute_network_density(residual_corr)
            result.residual_mean_correlation = self._compute_mean_correlation(residual_corr)

            # Step 5: Check for hidden structure
            if result.raw_effective_dimension > 0:
                result.dimension_expansion_ratio = (
                    result.residual_effective_dimension / result.raw_effective_dimension
                )
                if result.dimension_expansion_ratio >= MIN_DIMENSION_EXPANSION:
                    result.hidden_structure_found = True
                    logger.warning(f"HIDDEN STRUCTURE FOUND: Dimension expanded from "
                                  f"{result.raw_effective_dimension:.1f} to "
                                  f"{result.residual_effective_dimension:.1f}")

            # Generate interpretation
            result.interpretation = self._generate_interpretation(result)

        else:
            result.interpretation = (
                f"Cohort is NOT correlation-dominated (mean_corr={result.mean_correlation:.2f}, "
                f"PC1={result.pc1_variance_explained:.1%}). Geometry reflects true structure."
            )

        return result

    def _load_indicator_data(
        self,
        indicators: List[str],
        window_start: str,
        window_end: str
    ) -> Dict[str, Dict]:
        """Load price data for indicators."""
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

    def _compute_correlation_matrix(
        self,
        data: Dict[str, Dict]
    ) -> Tuple[Optional[np.ndarray], List[str]]:
        """Compute pairwise correlations from price data."""
        indicators = list(data.keys())
        n = len(indicators)

        if n < 2:
            return None, indicators

        # Find common dates
        common_dates = set(data[indicators[0]].keys())
        for ind in indicators[1:]:
            common_dates &= set(data[ind].keys())
        common_dates = sorted(common_dates)

        if len(common_dates) < 10:
            return None, indicators

        # Build return matrix
        prices = np.array([[data[ind][d] for d in common_dates] for ind in indicators])

        # Handle any zeros or negatives for log returns
        prices = np.maximum(prices, 1e-10)
        returns = np.diff(np.log(prices), axis=1)

        # Compute correlation
        corr_matrix = np.corrcoef(returns)

        return corr_matrix, indicators

    def _compute_mean_correlation(self, corr_matrix: np.ndarray) -> float:
        """Compute mean off-diagonal correlation."""
        n = corr_matrix.shape[0]
        if n < 2:
            return 0.0
        # Get upper triangle (excluding diagonal)
        upper = corr_matrix[np.triu_indices(n, k=1)]
        return float(np.mean(np.abs(upper)))

    def _extract_pc1(
        self,
        corr_matrix: np.ndarray
    ) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
        """Extract PC1 and return variance explained, loadings, all eigenvalues/vectors."""
        eigenvalues, eigenvectors = np.linalg.eigh(corr_matrix)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        total_var = np.sum(np.maximum(eigenvalues, 0))
        pc1_variance = eigenvalues[0] / total_var if total_var > 0 else 0
        pc1_loadings = eigenvectors[:, 0]

        return float(pc1_variance), pc1_loadings, eigenvalues, eigenvectors

    def _compute_residual_correlations(
        self,
        corr_matrix: np.ndarray,
        eigenvectors: np.ndarray,
        eigenvalues: np.ndarray,
        n_factors_to_remove: int = 1
    ) -> np.ndarray:
        """Remove dominant PCs and return residual correlation matrix."""
        residual_corr = corr_matrix.copy()

        for k in range(n_factors_to_remove):
            pc_k = eigenvectors[:, k:k+1]
            contribution = eigenvalues[k] * (pc_k @ pc_k.T)
            residual_corr -= contribution

        # Re-normalize
        diag = np.sqrt(np.abs(np.diag(residual_corr)))
        diag[diag == 0] = 1
        residual_corr = residual_corr / np.outer(diag, diag)
        np.fill_diagonal(residual_corr, 1.0)
        residual_corr = np.clip(residual_corr, -1.0, 1.0)

        return residual_corr

    def _compute_effective_dimension(self, eigenvalues: np.ndarray) -> float:
        """Compute effective dimension (components for 90% variance)."""
        eigenvalues = np.maximum(eigenvalues, 0)
        total = np.sum(eigenvalues)
        if total == 0:
            return 0.0
        explained_ratio = eigenvalues / total
        cumvar = np.cumsum(explained_ratio)
        return float(np.searchsorted(cumvar, 0.9) + 1)

    def _compute_cohort_count(self, corr_matrix: np.ndarray, threshold: float = 0.3) -> int:
        """Estimate number of cohorts via connected components."""
        n = corr_matrix.shape[0]
        adjacency = np.abs(corr_matrix) > threshold
        np.fill_diagonal(adjacency, False)

        visited = set()
        components = 0

        for start in range(n):
            if start in visited:
                continue
            components += 1
            queue = [start]
            while queue:
                node = queue.pop(0)
                if node in visited:
                    continue
                visited.add(node)
                for neighbor in range(n):
                    if adjacency[node, neighbor] and neighbor not in visited:
                        queue.append(neighbor)

        return components

    def _compute_network_density(self, corr_matrix: np.ndarray, threshold: float = 0.3) -> float:
        """Compute fraction of significant correlations."""
        n = corr_matrix.shape[0]
        if n < 2:
            return 0.0
        adjacency = np.abs(corr_matrix) > threshold
        np.fill_diagonal(adjacency, False)
        n_edges = np.sum(adjacency) / 2  # Undirected
        max_edges = n * (n - 1) / 2
        return float(n_edges / max_edges) if max_edges > 0 else 0.0

    def _generate_interpretation(self, result: DecontaminationResult) -> str:
        """Generate human-readable interpretation."""
        lines = [
            f"Cohort '{result.cohort_id}' is correlation-dominated:",
            f"  - Mean correlation: {result.mean_correlation:.2f}",
            f"  - PC1 explains: {result.pc1_variance_explained:.1%}",
            "",
            "Geometry reflects correlation, not structure.",
            "",
            "After removing shared factor (PC1):",
            f"  - Dimension: {result.raw_effective_dimension:.1f} -> {result.residual_effective_dimension:.1f}",
            f"  - Cohorts: {result.raw_cohorts} -> {result.residual_cohorts}",
            f"  - Density: {result.raw_network_density:.2f} -> {result.residual_network_density:.2f}",
        ]

        if result.hidden_structure_found:
            lines.extend([
                "",
                "*** HIDDEN STRUCTURE DETECTED ***",
                f"Dimension expanded {result.dimension_expansion_ratio:.1f}x after decontamination.",
                "Residual geometry reveals structure masked by shared factor.",
            ])

        return "\n".join(lines)

    def analyze_all_cohorts(
        self,
        window_start: str,
        window_end: str,
        cohort_definitions: Dict[str, List[str]] = None
    ) -> List[DecontaminationResult]:
        """
        Analyze multiple cohorts.

        If cohort_definitions not provided, treats all indicators as one cohort.
        """
        results = []

        if cohort_definitions:
            for cohort_id, indicators in cohort_definitions.items():
                result = self.analyze_cohort(indicators, window_start, window_end, cohort_id)
                results.append(result)
        else:
            # Get all indicators
            indicators = self.conn.execute("""
                SELECT DISTINCT indicator_id FROM data.indicators
            """).fetchall()
            indicators = [r[0] for r in indicators]

            result = self.analyze_cohort(indicators, window_start, window_end, "all")
            results.append(result)

        return results


# =============================================================================
# PERSISTENCE
# =============================================================================

def persist_decontamination_result(
    conn: duckdb.DuckDBPyConnection,
    result: DecontaminationResult
):
    """Persist decontamination result to database."""
    try:
        conn.execute("""
            INSERT INTO derived.decontamination_results
            (cohort_id, window_start, window_end, is_contaminated,
             mean_correlation, pc1_variance_explained,
             raw_effective_dimension, residual_effective_dimension,
             hidden_structure_found, dimension_expansion_ratio,
             warning, interpretation, computed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            result.cohort_id,
            result.window_start,
            result.window_end,
            result.is_contaminated,
            result.mean_correlation,
            result.pc1_variance_explained,
            result.raw_effective_dimension,
            result.residual_effective_dimension,
            result.hidden_structure_found,
            result.dimension_expansion_ratio,
            result.warning,
            result.interpretation,
            datetime.utcnow()
        ])
        logger.info(f"Persisted decontamination result for cohort {result.cohort_id}")
    except Exception as e:
        logger.error(f"Failed to persist decontamination result: {e}")


# =============================================================================
# CLI
# =============================================================================

def main():
    """CLI for running correlation decontamination analysis."""
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s'
    )

    parser = argparse.ArgumentParser(description='PRISM Correlation Decontamination Agent')
    parser.add_argument('--window-start', required=True, help='Window start date')
    parser.add_argument('--window-end', required=True, help='Window end date')
    parser.add_argument('--indicators', nargs='+', help='Specific indicators to analyze')
    parser.add_argument('--cohort-name', default='analysis', help='Name for this cohort')
    parser.add_argument('--db', default='data/prism.duckdb', help='Database path')
    args = parser.parse_args()

    conn = duckdb.connect(args.db)
    agent = CorrelationDecontaminationAgent(conn)

    if args.indicators:
        indicators = args.indicators
    else:
        # Use all available indicators
        indicators = conn.execute("""
            SELECT DISTINCT indicator_id FROM data.indicators
            WHERE indicator_id IN ('IEF', 'LQD', 'TLT', 'SHY', 'HYG', 'AGG', 'TIP',
                                   'GDP', 'M2SL', 'VIXY', 'DGS1', 'SLV', 'XLI', 'XLK')
        """).fetchall()
        indicators = [r[0] for r in indicators]

    print(f"Analyzing {len(indicators)} indicators: {indicators}")

    result = agent.analyze_cohort(
        indicators,
        args.window_start,
        args.window_end,
        args.cohort_name
    )

    # Output
    print("\n" + "=" * 70)
    print("CORRELATION DECONTAMINATION ANALYSIS")
    print("=" * 70)
    print(f"Cohort: {result.cohort_id}")
    print(f"Window: {result.window_start} to {result.window_end}")
    print(f"Indicators: {len(result.indicators)}")
    print()
    print(f"Contaminated: {'YES' if result.is_contaminated else 'No'}")
    print(f"Mean Correlation: {result.mean_correlation:.3f}")
    print(f"PC1 Variance: {result.pc1_variance_explained:.1%}")
    print()
    print(f"Raw Geometry:")
    print(f"  Effective Dimension: {result.raw_effective_dimension:.1f}")
    print(f"  Cohorts: {result.raw_cohorts}")
    print(f"  Network Density: {result.raw_network_density:.2f}")

    if result.is_contaminated:
        print()
        print(f"Residual Geometry (PC1 removed):")
        print(f"  Effective Dimension: {result.residual_effective_dimension:.1f}")
        print(f"  Cohorts: {result.residual_cohorts}")
        print(f"  Network Density: {result.residual_network_density:.2f}")
        print(f"  Mean Correlation: {result.residual_mean_correlation:.3f}")

        if result.hidden_structure_found:
            print()
            print("!" * 70)
            print("HIDDEN STRUCTURE DETECTED")
            print(f"Dimension expanded {result.dimension_expansion_ratio:.1f}x")
            print("!" * 70)

        print()
        print("PC1 Loadings (shared factor):")
        sorted_loadings = sorted(result.pc1_loadings.items(), key=lambda x: abs(x[1]), reverse=True)
        for ind, loading in sorted_loadings:
            print(f"  {ind}: {loading:+.3f}")

    print()
    print("-" * 70)
    print("INTERPRETATION:")
    print(result.interpretation)

    conn.close()


if __name__ == "__main__":
    main()
