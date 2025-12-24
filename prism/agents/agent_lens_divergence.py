"""
PRISM Lens Divergence Agent

Finds indicators that are:
- Similar in correlation, different in other views
- Different in correlation, similar in other views

These are the interesting cases that reveal structure invisible to correlation.

Pipeline position:
    P2: Derived (all engines run)
        ↓
    P2.5a: Correlation Decontamination (remove trivial clustering)
    P2.5b: Lens Divergence Detection (find interesting disagreements)  ← THIS
        ↓
    P3: Structure (now informed by divergence flags)
        ↓
    P4: Interpretation (reports BOTH correlation view AND divergent views)

The Paper Quote:
    "Traditional analysis groups indicators by correlation. PRISM reveals that
    uncorrelated indicators may share structural properties invisible to
    correlation, while correlated indicators may have fundamentally different
    dynamics. The multi-lens assembly exposes structure that no single method
    can see."

Cross-validated by: Claude
Date: December 2024
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import logging
from itertools import combinations

import duckdb

logger = logging.getLogger(__name__)


# =============================================================================
# THRESHOLDS
# =============================================================================

CORRELATION_SIMILAR_THRESHOLD = 0.70  # Considered "moving together"
CORRELATION_DIFFERENT_THRESHOLD = 0.30  # Considered "independent"
STRUCTURAL_SIMILAR_THRESHOLD = 0.70  # Considered "structurally similar"

# Engines that provide independent structural views (not correlation-based)
INDEPENDENT_ENGINES = [
    'hurst',       # Memory/persistence
    'entropy',     # Complexity/information
    'spectral',    # Frequency structure
    'wavelet',     # Multi-scale patterns
    'volatility',  # Risk dynamics
    'trend',       # Directional bias
    'stationarity', # Statistical stability
    'recurrence',  # Attractor structure
    'lyapunov',    # Chaos/predictability
]


# =============================================================================
# DIVERGENCE TYPES
# =============================================================================

@dataclass
class LensDivergence:
    """A single divergence finding between two indicators."""

    indicator_1: str
    indicator_2: str
    divergence_type: str  # 'correlated_but_different' or 'uncorrelated_but_similar'
    correlation: float
    divergent_engines: List[str]
    engine_similarities: Dict[str, float]
    interpretation: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            'indicator_1': self.indicator_1,
            'indicator_2': self.indicator_2,
            'divergence_type': self.divergence_type,
            'correlation': self.correlation,
            'divergent_engines': self.divergent_engines,
            'interpretation': self.interpretation,
        }


@dataclass
class LensDivergenceReport:
    """Complete lens divergence analysis report."""

    window_start: str
    window_end: str
    indicators: List[str]
    n_pairs_analyzed: int

    correlated_but_different: List[LensDivergence] = field(default_factory=list)
    uncorrelated_but_similar: List[LensDivergence] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"Lens Divergence Report: {self.window_start} to {self.window_end}",
            f"Indicators: {len(self.indicators)}",
            f"Pairs analyzed: {self.n_pairs_analyzed}",
            f"Correlated but structurally different: {len(self.correlated_but_different)}",
            f"Uncorrelated but structurally similar: {len(self.uncorrelated_but_similar)}",
        ]
        return "\n".join(lines)


# =============================================================================
# LENS DIVERGENCE AGENT
# =============================================================================

class LensDivergenceAgent:
    """
    Finds indicators that are:
    - Similar in correlation, different in other views
    - Different in correlation, similar in other views

    These are the interesting cases.
    """

    def __init__(self, conn: duckdb.DuckDBPyConnection):
        self.conn = conn
        self.engines = INDEPENDENT_ENGINES

    def analyze(
        self,
        indicators: List[str],
        window_start: str,
        window_end: str
    ) -> LensDivergenceReport:
        """
        Analyze all indicator pairs for lens divergences.

        Args:
            indicators: List of indicator IDs to analyze
            window_start: Analysis window start
            window_end: Analysis window end

        Returns:
            LensDivergenceReport with all divergences found
        """
        logger.info(f"Analyzing lens divergences for {len(indicators)} indicators")

        report = LensDivergenceReport(
            window_start=window_start,
            window_end=window_end,
            indicators=indicators,
            n_pairs_analyzed=0,
        )

        # Load correlation data
        correlations = self._load_correlations(indicators, window_start, window_end)

        # Load engine descriptors
        engine_descriptors = self._load_engine_descriptors(indicators, window_start, window_end)

        # Compute similarity per engine
        similarity_by_engine = {}
        for engine in self.engines:
            if engine in engine_descriptors:
                similarity_by_engine[engine] = self._compute_pairwise_similarity(
                    indicators, engine_descriptors[engine]
                )

        # Analyze all pairs
        pairs = list(combinations(indicators, 2))
        report.n_pairs_analyzed = len(pairs)

        for ind1, ind2 in pairs:
            pair_key = (ind1, ind2)
            corr = correlations.get(pair_key, correlations.get((ind2, ind1), 0.0))

            corr_similar = abs(corr) > CORRELATION_SIMILAR_THRESHOLD
            corr_different = abs(corr) < CORRELATION_DIFFERENT_THRESHOLD

            divergent_engines_similar = []  # Engines where structurally similar
            divergent_engines_different = []  # Engines where structurally different
            engine_sims = {}

            for engine, sim_matrix in similarity_by_engine.items():
                sim = sim_matrix.get(pair_key, sim_matrix.get((ind2, ind1), 0.0))
                engine_sims[engine] = sim

                if sim > STRUCTURAL_SIMILAR_THRESHOLD:
                    divergent_engines_similar.append(engine)
                else:
                    divergent_engines_different.append(engine)

            # Case 1: Correlated but structurally different
            if corr_similar and divergent_engines_different:
                interpretation = self._generate_correlated_different_interpretation(
                    ind1, ind2, corr, divergent_engines_different, engine_sims
                )
                divergence = LensDivergence(
                    indicator_1=ind1,
                    indicator_2=ind2,
                    divergence_type='correlated_but_different',
                    correlation=corr,
                    divergent_engines=divergent_engines_different,
                    engine_similarities=engine_sims,
                    interpretation=interpretation,
                )
                report.correlated_but_different.append(divergence)

            # Case 2: Uncorrelated but structurally similar
            if corr_different and divergent_engines_similar:
                interpretation = self._generate_uncorrelated_similar_interpretation(
                    ind1, ind2, corr, divergent_engines_similar, engine_sims
                )
                divergence = LensDivergence(
                    indicator_1=ind1,
                    indicator_2=ind2,
                    divergence_type='uncorrelated_but_similar',
                    correlation=corr,
                    divergent_engines=divergent_engines_similar,
                    engine_similarities=engine_sims,
                    interpretation=interpretation,
                )
                report.uncorrelated_but_similar.append(divergence)

        # Sort by number of divergent engines
        report.correlated_but_different.sort(
            key=lambda x: len(x.divergent_engines), reverse=True
        )
        report.uncorrelated_but_similar.sort(
            key=lambda x: len(x.divergent_engines), reverse=True
        )

        logger.info(f"Found {len(report.correlated_but_different)} correlated-but-different, "
                   f"{len(report.uncorrelated_but_similar)} uncorrelated-but-similar")

        return report

    def _load_correlations(
        self,
        indicators: List[str],
        window_start: str,
        window_end: str
    ) -> Dict[Tuple[str, str], float]:
        """Load pairwise correlations from database or compute from prices."""
        correlations = {}

        # Try loading from correlation_matrix
        try:
            result = self.conn.execute("""
                SELECT indicator_id_1, indicator_id_2, correlation
                FROM derived.correlation_matrix
                WHERE window_start = ? AND window_end = ?
            """, [window_start, window_end]).fetchall()

            for ind1, ind2, corr in result:
                if ind1 in indicators and ind2 in indicators:
                    correlations[(ind1, ind2)] = corr

            if correlations:
                return correlations
        except Exception:
            pass

        # Compute from price data
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
            except Exception:
                pass

        if len(data) < 2:
            return correlations

        # Find common dates
        common_dates = set(data[indicators[0]].keys()) if indicators[0] in data else set()
        for ind in indicators[1:]:
            if ind in data:
                common_dates &= set(data[ind].keys())
        common_dates = sorted(common_dates)

        if len(common_dates) < 10:
            return correlations

        # Compute returns
        for ind1, ind2 in combinations(indicators, 2):
            if ind1 not in data or ind2 not in data:
                continue

            v1 = [data[ind1][d] for d in common_dates]
            v2 = [data[ind2][d] for d in common_dates]

            # Log returns
            r1 = np.diff(np.log(np.maximum(v1, 1e-10)))
            r2 = np.diff(np.log(np.maximum(v2, 1e-10)))

            if len(r1) > 1:
                corr = np.corrcoef(r1, r2)[0, 1]
                correlations[(ind1, ind2)] = float(corr)

        return correlations

    def _load_engine_descriptors(
        self,
        indicators: List[str],
        window_start: str,
        window_end: str
    ) -> Dict[str, Dict[str, Dict[str, float]]]:
        """
        Load engine descriptors from derived.geometry_descriptors.

        Returns: {engine: {indicator: {descriptor: value}}}

        Note: The schema uses 'dimension' column for descriptor names.
        We group descriptors by prefix to infer engine groupings.
        """
        engine_descriptors = {}

        try:
            placeholders = ','.join(['?']*len(indicators))
            result = self.conn.execute(f"""
                SELECT indicator_id, dimension, value
                FROM derived.geometry_descriptors
                WHERE indicator_id IN ({placeholders})
            """, indicators).fetchall()

            # Group descriptors by category (infer engine from descriptor name)
            descriptor_to_engine = {
                'saturation': 'geometry',
                'asymmetry': 'geometry',
                'bounded': 'geometry',
                'determinism': 'recurrence',
                'entropy': 'entropy',
                'acf_1': 'autocorrelation',
                'acf_lag1': 'autocorrelation',
                'memory': 'autocorrelation',
                'hurst': 'hurst',
                'fat_tails': 'stats',
                'kurtosis': 'stats',
                'skewness': 'stats',
                'volatility': 'volatility',
                'trend': 'trend',
                'spectral': 'spectral',
                'wavelet': 'wavelet',
                'lyapunov': 'lyapunov',
            }

            for ind, dimension, value in result:
                if value is None:
                    continue

                # Infer engine from dimension name
                engine = 'other'
                dim_lower = dimension.lower()
                for key, eng in descriptor_to_engine.items():
                    if key in dim_lower:
                        engine = eng
                        break

                if engine not in engine_descriptors:
                    engine_descriptors[engine] = {}
                if ind not in engine_descriptors[engine]:
                    engine_descriptors[engine][ind] = {}
                engine_descriptors[engine][ind][dimension] = value

        except Exception as e:
            logger.warning(f"Could not load engine descriptors: {e}")

        return engine_descriptors

    def _compute_pairwise_similarity(
        self,
        indicators: List[str],
        engine_data: Dict[str, Dict[str, float]]
    ) -> Dict[Tuple[str, str], float]:
        """Compute pairwise similarity based on engine descriptor vectors."""
        similarities = {}

        for ind1, ind2 in combinations(indicators, 2):
            if ind1 not in engine_data or ind2 not in engine_data:
                continue

            d1 = engine_data[ind1]
            d2 = engine_data[ind2]

            # Get common descriptors
            common = set(d1.keys()) & set(d2.keys())
            if not common:
                continue

            # Compute cosine similarity on descriptor vectors
            v1 = np.array([d1[k] for k in sorted(common)])
            v2 = np.array([d2[k] for k in sorted(common)])

            # Normalize
            n1 = np.linalg.norm(v1)
            n2 = np.linalg.norm(v2)

            if n1 > 0 and n2 > 0:
                sim = float(np.dot(v1, v2) / (n1 * n2))
                similarities[(ind1, ind2)] = sim

        return similarities

    def _generate_correlated_different_interpretation(
        self,
        ind1: str,
        ind2: str,
        correlation: float,
        different_engines: List[str],
        engine_sims: Dict[str, float]
    ) -> str:
        """Generate interpretation for correlated but structurally different pairs."""
        engine_details = []
        for eng in different_engines[:3]:  # Top 3
            sim = engine_sims.get(eng, 0)
            engine_details.append(f"{eng} (sim={sim:.2f})")

        return (
            f"{ind1} and {ind2} move together (corr={correlation:.2f}) but have different "
            f"structural profiles in: {', '.join(engine_details)}. "
            f"Same direction, different dynamics."
        )

    def _generate_uncorrelated_similar_interpretation(
        self,
        ind1: str,
        ind2: str,
        correlation: float,
        similar_engines: List[str],
        engine_sims: Dict[str, float]
    ) -> str:
        """Generate interpretation for uncorrelated but structurally similar pairs."""
        engine_details = []
        for eng in similar_engines[:3]:  # Top 3
            sim = engine_sims.get(eng, 0)
            engine_details.append(f"{eng} (sim={sim:.2f})")

        return (
            f"{ind1} and {ind2} move independently (corr={correlation:.2f}) but share "
            f"structural properties in: {', '.join(engine_details)}. "
            f"Different directions, same underlying dynamics."
        )


# =============================================================================
# PERSISTENCE
# =============================================================================

def persist_lens_divergences(
    conn: duckdb.DuckDBPyConnection,
    report: LensDivergenceReport
):
    """Persist lens divergence findings to database."""
    try:
        for div in report.correlated_but_different + report.uncorrelated_but_similar:
            conn.execute("""
                INSERT INTO derived.lens_divergences
                (indicator_1, indicator_2, window_start, window_end,
                 divergence_type, correlation, divergent_engines, interpretation, computed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                div.indicator_1,
                div.indicator_2,
                report.window_start,
                report.window_end,
                div.divergence_type,
                div.correlation,
                ','.join(div.divergent_engines),
                div.interpretation,
                datetime.utcnow()
            ])
        logger.info(f"Persisted {len(report.correlated_but_different) + len(report.uncorrelated_but_similar)} divergences")
    except Exception as e:
        logger.error(f"Failed to persist divergences: {e}")


# =============================================================================
# CLI
# =============================================================================

def main():
    """CLI for running lens divergence analysis."""
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s'
    )

    parser = argparse.ArgumentParser(description='PRISM Lens Divergence Agent')
    parser.add_argument('--window-start', required=True, help='Window start date')
    parser.add_argument('--window-end', required=True, help='Window end date')
    parser.add_argument('--indicators', nargs='+', help='Specific indicators to analyze')
    parser.add_argument('--db', default='data/prism.duckdb', help='Database path')
    args = parser.parse_args()

    conn = duckdb.connect(args.db)
    agent = LensDivergenceAgent(conn)

    if args.indicators:
        indicators = args.indicators
    else:
        # Default set
        indicators = conn.execute("""
            SELECT DISTINCT indicator_id FROM data.indicators
            WHERE indicator_id IN ('IEF', 'LQD', 'TLT', 'SHY', 'HYG', 'AGG', 'TIP',
                                   'GDP', 'M2SL', 'VIXY', 'DGS1', 'SLV', 'XLI', 'XLK',
                                   'SPY', 'QQQ', 'VIX', 'UNRATE')
        """).fetchall()
        indicators = [r[0] for r in indicators]

    print(f"Analyzing {len(indicators)} indicators: {indicators}")

    report = agent.analyze(indicators, args.window_start, args.window_end)

    # Output
    print("\n" + "=" * 70)
    print("LENS DIVERGENCE REPORT")
    print("=" * 70)
    print(report.summary())

    if report.uncorrelated_but_similar:
        print("\n" + "-" * 70)
        print("UNCORRELATED BUT STRUCTURALLY SIMILAR:")
        print("-" * 70)
        for div in report.uncorrelated_but_similar[:10]:
            print(f"\n  {div.indicator_1} <-> {div.indicator_2}")
            print(f"    Correlation: {div.correlation:.2f} (independent)")
            print(f"    Similar in: {', '.join(div.divergent_engines)}")
            print(f"    {div.interpretation}")

    if report.correlated_but_different:
        print("\n" + "-" * 70)
        print("CORRELATED BUT STRUCTURALLY DIFFERENT:")
        print("-" * 70)
        for div in report.correlated_but_different[:10]:
            print(f"\n  {div.indicator_1} <-> {div.indicator_2}")
            print(f"    Correlation: {div.correlation:.2f} (move together)")
            print(f"    Different in: {', '.join(div.divergent_engines)}")
            print(f"    {div.interpretation}")

    if not report.uncorrelated_but_similar and not report.correlated_but_different:
        print("\nNo significant lens divergences found.")
        print("All pairs show consistent agreement between correlation and structural views.")

    conn.close()


if __name__ == "__main__":
    main()
