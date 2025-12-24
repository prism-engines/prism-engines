"""
PRISM Cross-Engine Agreement Agent

Determines whether multiple independent engines agree on structural change.
Agreement strengthens confidence - no voting, no averaging, no interpretation.

IMPORTANT CONSTRAINTS (NON-NEGOTIABLE):
1. Phase-0 data is immutable. No writes to Phase-0 tables.
2. Agents operate ONLY on DB-persisted engine outputs.
3. Agents are NOT real-time.
4. Agents do NOT rerun engines.
5. Agents do NOT introduce meaning or labels.
6. Agents write annotations only.
7. All runnable scripts must live at /scripts level.
8. DB paths must be resolved from repo root (no relative drift).
9. No parallel execution inside agents unless explicitly requested.

Author: PRISM Team
Version: 1.0.0
"""

import logging
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Set
from datetime import datetime, date
from itertools import combinations
import numpy as np

from prism.config import get_agreement_config

logger = logging.getLogger(__name__)


# Engine groupings for agreement analysis
ENGINE_GROUPS = {
    "structural": ["pca", "correlation"],
    "temporal": ["hmm", "dmd", "granger"],
    "scale": ["wavelet", "hurst"],
}


@dataclass
class EnginePairAgreement:
    """Agreement metrics between two engines."""
    engine_a: str
    engine_b: str
    agreement_type: str  # 'reinforcing', 'contradicting', 'neutral'
    agreement_strength: float
    correlation: Optional[float] = None
    transition_overlap: Optional[float] = None


@dataclass
class Contradiction:
    """A specific disagreement between engines."""
    contradiction_id: int
    engine_a: str
    engine_b: str
    dimension: str
    value_a: float
    value_b: float
    disagreement_magnitude: float
    severity: str  # 'low', 'medium', 'high'


@dataclass
class AgreementResult:
    """Result of agreement analysis for an indicator."""
    indicator_id: str
    window_years: float
    
    # Overall agreement
    agreement_score: float = 0.0
    n_engines: int = 0
    
    # Pair counts
    n_reinforcing_pairs: int = 0
    n_contradicting_pairs: int = 0
    n_neutral_pairs: int = 0
    
    # Group agreement
    structural_agreement: Optional[float] = None
    temporal_agreement: Optional[float] = None
    scale_agreement: Optional[float] = None
    
    # Confidence adjustment
    confidence_multiplier: float = 1.0
    
    # Details
    pair_agreements: List[EnginePairAgreement] = field(default_factory=list)
    contradictions: List[Contradiction] = field(default_factory=list)


@dataclass
class AgreementPolicy:
    """
    Configurable thresholds for agreement analysis.

    Parameters can be loaded from config/agreement.yaml via from_config().
    Default values are preserved for backward compatibility.
    """
    version: str = "1.0.0"

    # Agreement thresholds
    reinforcing_threshold: float = 0.3    # Correlation above this = reinforcing
    contradicting_threshold: float = -0.3  # Correlation below this = contradicting

    # Contradiction severity
    low_disagreement: float = 1.0         # Std devs for low severity
    medium_disagreement: float = 2.0      # Std devs for medium severity
    high_disagreement: float = 3.0        # Std devs for high severity

    # Confidence adjustment
    high_agreement_bonus: float = 1.2     # Multiplier for high agreement
    contradiction_penalty: float = 0.8    # Multiplier when contradictions exist

    @classmethod
    def from_config(cls) -> "AgreementPolicy":
        """
        Create policy from config/agreement.yaml.

        Falls back to defaults if config not found.
        """
        try:
            cfg = get_agreement_config()
        except FileNotFoundError:
            logger.warning("agreement.yaml not found, using defaults")
            return cls()

        return cls(
            version=cfg.get("version", "1.0.0"),
            reinforcing_threshold=cfg.get("reinforcing_threshold", 0.3),
            contradicting_threshold=cfg.get("contradicting_threshold", -0.3),
            low_disagreement=cfg.get("low_disagreement", 1.0),
            medium_disagreement=cfg.get("medium_disagreement", 2.0),
            high_disagreement=cfg.get("high_disagreement", 3.0),
            high_agreement_bonus=cfg.get("high_agreement_bonus", 1.2),
            contradiction_penalty=cfg.get("contradiction_penalty", 0.8),
        )


class CrossEngineAgreementAgent:
    """
    Detects structural coherence across multiple independent engines.
    
    This agent:
    - Compares geometry contributions from different engines
    - Detects reinforcement (engines agree on structure)
    - Detects contradiction (engines disagree)
    - Produces agreement scores and confidence adjustments
    
    This agent does NOT:
    - Vote or average across engines
    - Interpret what agreement means
    - Make predictions
    - Modify geometry or engine outputs
    """
    
    VERSION = "1.0.0"
    
    def __init__(self, policy: Optional[AgreementPolicy] = None):
        self.policy = policy or AgreementPolicy()
        self._conn = None

    @property
    def conn(self):
        if self._conn is None:
            from prism.db.open import open_prism_db
            self._conn = open_prism_db()
        return self._conn
    
    def analyze_run(self, run_id: str) -> List[AgreementResult]:
        """
        Analyze cross-engine agreement for all indicators in a run.
        
        Args:
            run_id: The run identifier
            
        Returns:
            List of AgreementResult objects
        """
        logger.info(f"Analyzing cross-engine agreement for run_id={run_id}")
        
        # Load geometry and engine mappings
        indicators = self._load_indicator_data(run_id)
        
        results = []
        for indicator_id, window_years, engine_data in indicators:
            result = self._analyze_agreement(indicator_id, window_years, engine_data)
            results.append(result)
        
        logger.info(f"Analyzed agreement for {len(results)} indicators")
        return results
    
    def analyze_and_persist(self, run_id: str) -> List[AgreementResult]:
        """Analyze agreement and persist results to DB."""
        results = self.analyze_run(run_id)
        self._persist_results(run_id, results)
        return results
    
    def _load_indicator_data(self, run_id: str) -> List[Tuple[str, float, Dict]]:
        """Load engine contributions for each indicator."""
        indicators = []
        
        # Load geometry vectors with engine contributions
        try:
            df = self.conn.execute("""
                SELECT 
                    gv.indicator_id, 
                    gv.window_years,
                    gv.vector,
                    gv.component_engines,
                    gv.component_weights
                FROM phase1.geometry_vectors gv
                WHERE gv.run_id = ?
            """, [run_id]).fetchdf()
        except Exception as e:
            logger.error(f"Could not load geometry: {e}")
            return []
        
        # Load coordinate frame to understand engine -> dimension mapping
        try:
            frame_df = self.conn.execute("""
                SELECT dimension_index, engine, engine_output_key
                FROM phase1.geometry_coordinate_frames
                WHERE run_id = ?
                ORDER BY dimension_index
            """, [run_id]).fetchdf()
            
            dimension_engine_map = {}
            for _, row in frame_df.iterrows():
                dimension_engine_map[row["dimension_index"]] = row["engine"]
        except Exception as e:
            logger.debug(f"Could not load coordinate frame: {e}")
            dimension_engine_map = {}
        
        # Group by (indicator_id, window_years)
        grouped = {}
        for _, row in df.iterrows():
            key = (row["indicator_id"], row["window_years"])
            
            vector = np.array(json.loads(row["vector"]))
            engines = row["component_engines"].split(",") if row["component_engines"] else []
            weights = json.loads(row["component_weights"]) if row["component_weights"] else None
            
            if key not in grouped:
                grouped[key] = {
                    "vectors": [],
                    "engines": set(),
                    "dimension_engine_map": dimension_engine_map,
                }
            
            grouped[key]["vectors"].append(vector)
            grouped[key]["engines"].update(engines)
        
        # Build output list
        for (indicator_id, window_years), data in grouped.items():
            # Compute mean vector for this indicator
            if data["vectors"]:
                mean_vector = np.mean(data["vectors"], axis=0)
                data["mean_vector"] = mean_vector
            
            indicators.append((indicator_id, window_years, data))
        
        return indicators
    
    def _analyze_agreement(
        self, 
        indicator_id: str, 
        window_years: float,
        engine_data: Dict
    ) -> AgreementResult:
        """Analyze agreement between engines for an indicator."""
        result = AgreementResult(
            indicator_id=indicator_id,
            window_years=window_years,
        )
        
        engines = list(engine_data.get("engines", []))
        result.n_engines = len(engines)
        
        if len(engines) < 2:
            # Can't compare with fewer than 2 engines
            return result
        
        mean_vector = engine_data.get("mean_vector")
        dim_engine_map = engine_data.get("dimension_engine_map", {})
        
        if mean_vector is None:
            return result
        
        # Extract per-engine contributions
        engine_contributions = self._extract_engine_contributions(
            mean_vector, dim_engine_map, engines
        )
        
        # Compare all pairs
        pair_agreements = []
        contradictions = []
        contradiction_id = 0
        
        for engine_a, engine_b in combinations(engines, 2):
            contrib_a = engine_contributions.get(engine_a, [])
            contrib_b = engine_contributions.get(engine_b, [])
            
            pair = self._compare_engines(
                engine_a, engine_b, contrib_a, contrib_b, mean_vector
            )
            pair_agreements.append(pair)
            
            # Check for contradictions
            if pair.agreement_type == "contradicting":
                for i, (va, vb) in enumerate(zip(contrib_a, contrib_b)):
                    if va is not None and vb is not None:
                        diff = abs(va - vb)
                        std = np.std([va, vb]) if va != vb else 1.0
                        mag = diff / (std + 1e-10)
                        
                        if mag > self.policy.low_disagreement:
                            contradiction_id += 1
                            severity = "low"
                            if mag > self.policy.high_disagreement:
                                severity = "high"
                            elif mag > self.policy.medium_disagreement:
                                severity = "medium"
                            
                            contradictions.append(Contradiction(
                                contradiction_id=contradiction_id,
                                engine_a=engine_a,
                                engine_b=engine_b,
                                dimension=f"dim_{i}",
                                value_a=float(va),
                                value_b=float(vb),
                                disagreement_magnitude=float(mag),
                                severity=severity,
                            ))
        
        result.pair_agreements = pair_agreements
        result.contradictions = contradictions
        
        # Count pair types
        result.n_reinforcing_pairs = sum(1 for p in pair_agreements if p.agreement_type == "reinforcing")
        result.n_contradicting_pairs = sum(1 for p in pair_agreements if p.agreement_type == "contradicting")
        result.n_neutral_pairs = sum(1 for p in pair_agreements if p.agreement_type == "neutral")
        
        # Compute group agreements
        result.structural_agreement = self._compute_group_agreement(
            pair_agreements, ENGINE_GROUPS["structural"]
        )
        result.temporal_agreement = self._compute_group_agreement(
            pair_agreements, ENGINE_GROUPS["temporal"]
        )
        result.scale_agreement = self._compute_group_agreement(
            pair_agreements, ENGINE_GROUPS["scale"]
        )
        
        # Compute overall agreement score
        total_pairs = len(pair_agreements)
        if total_pairs > 0:
            avg_strength = np.mean([p.agreement_strength for p in pair_agreements])
            reinforcing_ratio = result.n_reinforcing_pairs / total_pairs
            
            # Score: weighted combination
            result.agreement_score = float(
                0.5 * avg_strength +
                0.3 * reinforcing_ratio +
                0.2 * (1.0 - result.n_contradicting_pairs / total_pairs)
            )
        
        # Confidence multiplier
        if result.agreement_score > 0.7 and result.n_contradicting_pairs == 0:
            result.confidence_multiplier = self.policy.high_agreement_bonus
        elif result.n_contradicting_pairs > 0:
            result.confidence_multiplier = self.policy.contradiction_penalty
        
        return result
    
    def _extract_engine_contributions(
        self, 
        vector: np.ndarray,
        dim_engine_map: Dict[int, str],
        engines: List[str]
    ) -> Dict[str, List[Optional[float]]]:
        """Extract which vector components each engine contributed."""
        contributions = {e: [] for e in engines}
        
        for i, val in enumerate(vector):
            engine = dim_engine_map.get(i)
            if engine and engine in contributions:
                contributions[engine].append(float(val))
            else:
                # Unknown engine for this dimension
                for e in engines:
                    contributions[e].append(None)
        
        return contributions
    
    def _compare_engines(
        self,
        engine_a: str,
        engine_b: str,
        contrib_a: List[Optional[float]],
        contrib_b: List[Optional[float]],
        mean_vector: np.ndarray
    ) -> EnginePairAgreement:
        """Compare contributions from two engines."""
        # Get valid pairs
        valid_pairs = [
            (a, b) for a, b in zip(contrib_a, contrib_b)
            if a is not None and b is not None
        ]
        
        if not valid_pairs:
            return EnginePairAgreement(
                engine_a=engine_a,
                engine_b=engine_b,
                agreement_type="neutral",
                agreement_strength=0.0,
            )
        
        vals_a = np.array([p[0] for p in valid_pairs])
        vals_b = np.array([p[1] for p in valid_pairs])
        
        # Compute correlation
        if len(vals_a) > 1 and np.std(vals_a) > 0 and np.std(vals_b) > 0:
            correlation = float(np.corrcoef(vals_a, vals_b)[0, 1])
        else:
            correlation = 0.0
        
        # Determine agreement type
        if correlation >= self.policy.reinforcing_threshold:
            agreement_type = "reinforcing"
            agreement_strength = float((correlation - self.policy.reinforcing_threshold) / 
                                      (1.0 - self.policy.reinforcing_threshold))
        elif correlation <= self.policy.contradicting_threshold:
            agreement_type = "contradicting"
            agreement_strength = float((self.policy.contradicting_threshold - correlation) /
                                      (1.0 + self.policy.contradicting_threshold))
        else:
            agreement_type = "neutral"
            agreement_strength = float(1.0 - abs(correlation) / self.policy.reinforcing_threshold)
        
        return EnginePairAgreement(
            engine_a=engine_a,
            engine_b=engine_b,
            agreement_type=agreement_type,
            agreement_strength=min(1.0, max(0.0, agreement_strength)),
            correlation=correlation,
        )
    
    def _compute_group_agreement(
        self,
        pair_agreements: List[EnginePairAgreement],
        group_engines: List[str]
    ) -> Optional[float]:
        """Compute agreement within an engine group."""
        # Filter to pairs within the group
        group_pairs = [
            p for p in pair_agreements
            if p.engine_a in group_engines and p.engine_b in group_engines
        ]
        
        if not group_pairs:
            return None
        
        return float(np.mean([p.agreement_strength for p in group_pairs]))
    
    def _persist_results(self, run_id: str, results: List[AgreementResult]):
        """Persist agreement results to Phase-1 tables."""
        logger.info(f"Persisting {len(results)} agreement results")

        # Assert required tables exist (no runtime schema creation)
        from prism.db.schema_guard import assert_tables_exist
        assert_tables_exist(self.conn, [
            "phase1.cross_engine_agreement",
            "phase1.engine_pair_agreement",
            "phase1.engine_contradictions"
        ])
        
        for result in results:
            # Main agreement record
            self.conn.execute("""
                INSERT OR REPLACE INTO phase1.cross_engine_agreement
                (run_id, indicator_id, window_years, agreement_score, n_engines,
                 n_reinforcing_pairs, n_contradicting_pairs, n_neutral_pairs,
                 structural_agreement, temporal_agreement, scale_agreement,
                 confidence_multiplier, analyzer_version)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                run_id,
                result.indicator_id,
                result.window_years,
                result.agreement_score,
                result.n_engines,
                result.n_reinforcing_pairs,
                result.n_contradicting_pairs,
                result.n_neutral_pairs,
                result.structural_agreement,
                result.temporal_agreement,
                result.scale_agreement,
                result.confidence_multiplier,
                self.VERSION,
            ])
            
            # Pair agreements
            for pair in result.pair_agreements:
                self.conn.execute("""
                    INSERT OR REPLACE INTO phase1.engine_pair_agreement
                    (run_id, indicator_id, window_years, engine_a, engine_b,
                     agreement_type, agreement_strength, correlation, transition_overlap)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, [
                    run_id,
                    result.indicator_id,
                    result.window_years,
                    pair.engine_a,
                    pair.engine_b,
                    pair.agreement_type,
                    pair.agreement_strength,
                    pair.correlation,
                    pair.transition_overlap,
                ])
            
            # Contradictions
            for c in result.contradictions:
                self.conn.execute("""
                    INSERT OR REPLACE INTO phase1.engine_contradictions
                    (run_id, indicator_id, window_years, contradiction_id,
                     engine_a, engine_b, dimension, value_a, value_b,
                     disagreement_magnitude, severity)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, [
                    run_id,
                    result.indicator_id,
                    result.window_years,
                    c.contradiction_id,
                    c.engine_a,
                    c.engine_b,
                    c.dimension,
                    c.value_a,
                    c.value_b,
                    c.disagreement_magnitude,
                    c.severity,
                ])
        
        logger.info(f"Persisted agreement analysis to phase1 tables")
    
    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None


# =============================================================================
# CLI INTERFACE
# =============================================================================

def main():
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Analyze cross-engine agreement for structural coherence"
    )
    parser.add_argument("--run-id", required=True, help="Run ID to analyze")
    parser.add_argument("--dry-run", action="store_true", help="Analyze without persisting")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )

    agent = CrossEngineAgreementAgent()
    
    try:
        if args.dry_run:
            results = agent.analyze_run(args.run_id)
            print(f"\nCross-Engine Agreement Analysis (DRY RUN):")
        else:
            results = agent.analyze_and_persist(args.run_id)
            print(f"\nCross-Engine Agreement Analysis (PERSISTED):")
        
        print(f"  Total indicators analyzed: {len(results)}")
        
        if results:
            avg_agreement = np.mean([r.agreement_score for r in results])
            total_reinforcing = sum(r.n_reinforcing_pairs for r in results)
            total_contradicting = sum(r.n_contradicting_pairs for r in results)
            total_contradictions = sum(len(r.contradictions) for r in results)
            
            print(f"  Average agreement score: {avg_agreement:.3f}")
            print(f"  Total reinforcing pairs: {total_reinforcing}")
            print(f"  Total contradicting pairs: {total_contradicting}")
            print(f"  Total specific contradictions: {total_contradictions}")
            
            # High agreement indicators
            high_agreement = [r for r in results if r.agreement_score >= 0.7]
            print(f"  High agreement indicators: {len(high_agreement)}")
        
    finally:
        agent.close()


if __name__ == "__main__":
    main()
