"""
PRISM Temporal Persistence Agent

Determines whether constructed geometry persists over time or is transient motion.
Operates only on geometry - no raw engine access.

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
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, date, timedelta
import numpy as np

from prism.config import get_persistence_config

logger = logging.getLogger(__name__)


@dataclass
class Transition:
    """A detected transition in geometry space."""
    transition_id: int
    transition_date: date
    magnitude: float
    direction_vector: np.ndarray
    sharpness: float
    confidence: float
    pre_state_centroid: Optional[np.ndarray] = None
    post_state_centroid: Optional[np.ndarray] = None


@dataclass
class WindowContinuity:
    """Continuity metrics between adjacent windows."""
    window_start: date
    window_end: date
    next_window_start: Optional[date]
    step_distance: float
    velocity: float
    acceleration: float
    smoothness_score: float
    is_discontinuity: bool


@dataclass
class PersistenceResult:
    """Result of persistence analysis for an indicator."""
    indicator_id: str
    window_years: float
    
    # Scores
    persistence_score: float = 0.0
    stability_score: float = 0.0
    continuity_score: float = 0.0
    
    # Transitions
    transitions: List[Transition] = field(default_factory=list)
    n_transitions: int = 0
    transition_rate: float = 0.0
    avg_transition_magnitude: float = 0.0
    max_transition_magnitude: float = 0.0
    
    # Regime persistence
    avg_regime_duration: float = 0.0
    max_regime_duration: float = 0.0
    regime_entropy: float = 0.0
    
    # Boundaries
    edge_instability: float = 0.0
    boundary_artifact_flag: bool = False
    
    # Continuity details
    continuity_details: List[WindowContinuity] = field(default_factory=list)


@dataclass
class PersistencePolicy:
    """
    Configurable thresholds for persistence analysis.

    Parameters can be loaded from config/persistence.yaml via from_config().
    Default values are preserved for backward compatibility.
    """
    version: str = "1.0.0"

    # Transition detection
    transition_threshold: float = 1.5    # Std devs from mean step size
    min_transition_confidence: float = 0.3

    # Persistence thresholds
    high_persistence_threshold: float = 0.7
    low_persistence_threshold: float = 0.3

    # Continuity thresholds
    discontinuity_threshold: float = 3.0  # Std devs for discontinuity
    edge_window_fraction: float = 0.1     # Fraction of data to check for edge effects

    # Regime analysis
    min_regime_duration_days: int = 30

    @classmethod
    def from_config(cls) -> "PersistencePolicy":
        """
        Create policy from config/persistence.yaml.

        Falls back to defaults if config not found.
        """
        try:
            cfg = get_persistence_config()
        except FileNotFoundError:
            logger.warning("persistence.yaml not found, using defaults")
            return cls()

        return cls(
            version=cfg.get("version", "1.0.0"),
            transition_threshold=cfg.get("transition_threshold", 1.5),
            min_transition_confidence=cfg.get("min_transition_confidence", 0.3),
            high_persistence_threshold=cfg.get("high_persistence_threshold", 0.7),
            low_persistence_threshold=cfg.get("low_persistence_threshold", 0.3),
            discontinuity_threshold=cfg.get("discontinuity_threshold", 3.0),
            edge_window_fraction=cfg.get("edge_window_fraction", 0.1),
            min_regime_duration_days=cfg.get("min_regime_duration_days", 30),
        )


class TemporalPersistenceAgent:
    """
    Analyzes persistence and continuity of constructed geometry over time.
    
    This agent:
    - Reads geometry from phase1.geometry_vectors
    - Analyzes window-to-window continuity
    - Detects regime transitions
    - Computes persistence and stability scores
    - Flags boundary artifacts
    
    This agent does NOT:
    - Modify geometry
    - Reclassify math
    - Introduce meaning or interpretation
    """
    
    VERSION = "1.0.0"
    
    def __init__(self, policy: Optional[PersistencePolicy] = None):
        self.policy = policy or PersistencePolicy()
        self._conn = None

    @property
    def conn(self):
        if self._conn is None:
            from prism.db.open import open_prism_db
            self._conn = open_prism_db()
        return self._conn
    
    def analyze_run(self, run_id: str) -> List[PersistenceResult]:
        """
        Analyze persistence for all trajectories in a run.
        
        Args:
            run_id: The run identifier
            
        Returns:
            List of PersistenceResult objects
        """
        logger.info(f"Analyzing temporal persistence for run_id={run_id}")
        
        # Get all trajectories
        trajectories = self._load_trajectories(run_id)
        
        results = []
        for indicator_id, window_years, vectors in trajectories:
            result = self._analyze_trajectory(indicator_id, window_years, vectors)
            results.append(result)
        
        logger.info(f"Analyzed {len(results)} trajectories")
        return results
    
    def analyze_and_persist(self, run_id: str) -> List[PersistenceResult]:
        """Analyze persistence and persist results to DB."""
        results = self.analyze_run(run_id)
        self._persist_results(run_id, results)
        return results
    
    def _load_trajectories(self, run_id: str) -> List[Tuple[str, float, List[Dict]]]:
        """Load geometry vectors grouped by trajectory."""
        trajectories = []
        
        try:
            df = self.conn.execute("""
                SELECT indicator_id, window_years, window_start, window_end, vector
                FROM phase1.geometry_vectors
                WHERE run_id = ?
                ORDER BY indicator_id, window_years, window_start
            """, [run_id]).fetchdf()
        except Exception as e:
            logger.error(f"Could not load geometry: {e}")
            return []
        
        # Group by (indicator_id, window_years)
        current_key = None
        current_vectors = []
        
        for _, row in df.iterrows():
            key = (row["indicator_id"], row["window_years"])
            
            if key != current_key:
                if current_key is not None and current_vectors:
                    trajectories.append((*current_key, current_vectors))
                current_key = key
                current_vectors = []
            
            current_vectors.append({
                "window_start": row["window_start"],
                "window_end": row["window_end"],
                "vector": np.array(json.loads(row["vector"])),
            })
        
        # Don't forget last group
        if current_key is not None and current_vectors:
            trajectories.append((*current_key, current_vectors))
        
        return trajectories
    
    def _analyze_trajectory(
        self, 
        indicator_id: str, 
        window_years: float,
        vectors: List[Dict]
    ) -> PersistenceResult:
        """Analyze persistence of a single trajectory."""
        result = PersistenceResult(
            indicator_id=indicator_id,
            window_years=window_years,
        )
        
        if len(vectors) < 2:
            # Can't analyze persistence with fewer than 2 points
            return result
        
        # Sort by time
        vectors = sorted(vectors, key=lambda v: v["window_start"])
        
        # Compute step distances
        step_distances = []
        step_times = []
        
        for i in range(1, len(vectors)):
            v_prev = vectors[i-1]["vector"]
            v_curr = vectors[i]["vector"]
            
            dist = np.linalg.norm(v_curr - v_prev)
            step_distances.append(dist)
            
            # Time difference in days
            t_prev = vectors[i-1]["window_start"]
            t_curr = vectors[i]["window_start"]
            if isinstance(t_prev, str):
                t_prev = datetime.fromisoformat(t_prev).date()
            if isinstance(t_curr, str):
                t_curr = datetime.fromisoformat(t_curr).date()
            
            days = (t_curr - t_prev).days if t_curr and t_prev else 30
            step_times.append(max(days, 1))
        
        step_distances = np.array(step_distances)
        step_times = np.array(step_times)
        
        # Compute velocities
        velocities = step_distances / step_times
        
        # Compute accelerations
        accelerations = np.zeros(len(velocities))
        for i in range(1, len(velocities)):
            accelerations[i] = (velocities[i] - velocities[i-1]) / step_times[i]
        
        # Detect transitions
        mean_step = np.mean(step_distances)
        std_step = np.std(step_distances)
        threshold = mean_step + self.policy.transition_threshold * std_step
        
        transitions = []
        transition_id = 0
        
        for i, dist in enumerate(step_distances):
            if dist > threshold:
                transition_id += 1
                
                # Get transition date
                t_date = vectors[i+1]["window_start"]
                if isinstance(t_date, str):
                    t_date = datetime.fromisoformat(t_date).date()
                
                # Direction vector (normalized)
                direction = vectors[i+1]["vector"] - vectors[i]["vector"]
                direction_norm = direction / (np.linalg.norm(direction) + 1e-10)
                
                # Sharpness (how much larger than expected)
                sharpness = (dist - mean_step) / (std_step + 1e-10)
                
                # Confidence based on sharpness
                confidence = min(1.0, sharpness / 5.0)
                
                if confidence >= self.policy.min_transition_confidence:
                    transitions.append(Transition(
                        transition_id=transition_id,
                        transition_date=t_date,
                        magnitude=float(dist),
                        direction_vector=direction_norm,
                        sharpness=float(sharpness),
                        confidence=float(confidence),
                        pre_state_centroid=vectors[i]["vector"],
                        post_state_centroid=vectors[i+1]["vector"],
                    ))
        
        result.transitions = transitions
        result.n_transitions = len(transitions)
        
        # Transition statistics
        if transitions:
            mags = [t.magnitude for t in transitions]
            result.avg_transition_magnitude = float(np.mean(mags))
            result.max_transition_magnitude = float(np.max(mags))
            
            # Transition rate (per year)
            total_days = step_times.sum()
            result.transition_rate = len(transitions) / (total_days / 365.25)
        
        # Compute regime durations
        regime_durations = self._compute_regime_durations(vectors, transitions)
        if regime_durations:
            result.avg_regime_duration = float(np.mean(regime_durations))
            result.max_regime_duration = float(np.max(regime_durations))
            
            # Regime entropy
            durations = np.array(regime_durations)
            probs = durations / durations.sum()
            result.regime_entropy = float(-np.sum(probs * np.log(probs + 1e-10)))
        
        # Window-to-window continuity
        continuity_details = []
        smoothness_scores = []
        
        for i in range(len(vectors) - 1):
            smoothness = 1.0 - min(1.0, step_distances[i] / (mean_step * 3 + 1e-10))
            is_discontinuity = step_distances[i] > (mean_step + self.policy.discontinuity_threshold * std_step)
            
            smoothness_scores.append(smoothness)
            
            ws = vectors[i]["window_start"]
            we = vectors[i]["window_end"]
            nws = vectors[i+1]["window_start"]
            
            continuity_details.append(WindowContinuity(
                window_start=ws if isinstance(ws, date) else datetime.fromisoformat(ws).date() if ws else None,
                window_end=we if isinstance(we, date) else datetime.fromisoformat(we).date() if we else None,
                next_window_start=nws if isinstance(nws, date) else datetime.fromisoformat(nws).date() if nws else None,
                step_distance=float(step_distances[i]),
                velocity=float(velocities[i]),
                acceleration=float(accelerations[i]) if i < len(accelerations) else 0.0,
                smoothness_score=float(smoothness),
                is_discontinuity=is_discontinuity,
            ))
        
        result.continuity_details = continuity_details
        
        # Edge instability
        n_edge = max(1, int(len(step_distances) * self.policy.edge_window_fraction))
        edge_steps = np.concatenate([step_distances[:n_edge], step_distances[-n_edge:]])
        middle_steps = step_distances[n_edge:-n_edge] if len(step_distances) > 2*n_edge else step_distances
        
        edge_mean = np.mean(edge_steps)
        middle_mean = np.mean(middle_steps) if len(middle_steps) > 0 else edge_mean
        
        result.edge_instability = float(edge_mean / (middle_mean + 1e-10) - 1.0)
        result.boundary_artifact_flag = result.edge_instability > 0.5
        
        # Compute aggregate scores
        result.continuity_score = float(np.mean(smoothness_scores)) if smoothness_scores else 0.0
        result.stability_score = float(1.0 / (1.0 + np.std(step_distances) / (mean_step + 1e-10)))
        
        # Persistence score: combination of continuity, stability, and low transition rate
        transition_penalty = min(1.0, result.transition_rate / 2.0)  # High rate = low persistence
        result.persistence_score = float(
            0.4 * result.continuity_score +
            0.3 * result.stability_score +
            0.3 * (1.0 - transition_penalty)
        )
        
        return result
    
    def _compute_regime_durations(
        self, 
        vectors: List[Dict], 
        transitions: List[Transition]
    ) -> List[float]:
        """Compute duration of each regime (time between transitions)."""
        if not vectors:
            return []
        
        # Get transition dates
        transition_dates = [t.transition_date for t in transitions]
        
        # Get start and end dates
        start_date = vectors[0]["window_start"]
        end_date = vectors[-1]["window_end"]
        
        if isinstance(start_date, str):
            start_date = datetime.fromisoformat(start_date).date()
        if isinstance(end_date, str):
            end_date = datetime.fromisoformat(end_date).date()
        
        # Build regime boundaries
        boundaries = [start_date] + sorted(transition_dates) + [end_date]
        
        # Compute durations
        durations = []
        for i in range(len(boundaries) - 1):
            if boundaries[i] and boundaries[i+1]:
                days = (boundaries[i+1] - boundaries[i]).days
                durations.append(days)
        
        return durations
    
    def _persist_results(self, run_id: str, results: List[PersistenceResult]):
        """Persist persistence results to Phase-1 tables."""
        logger.info(f"Persisting {len(results)} persistence results")

        # Assert required tables exist (no runtime schema creation)
        from prism.db.schema_guard import assert_tables_exist
        assert_tables_exist(self.conn, [
            "phase1.temporal_persistence",
            "phase1.detected_transitions",
            "phase1.window_continuity"
        ])
        
        # Insert data
        for result in results:
            # Main persistence record
            self.conn.execute("""
                INSERT OR REPLACE INTO phase1.temporal_persistence
                (run_id, indicator_id, window_years, persistence_score, stability_score,
                 continuity_score, n_transitions, transition_rate, avg_transition_magnitude,
                 max_transition_magnitude, avg_regime_duration, max_regime_duration,
                 regime_entropy, edge_instability, boundary_artifact_flag, analyzer_version)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                run_id,
                result.indicator_id,
                result.window_years,
                result.persistence_score,
                result.stability_score,
                result.continuity_score,
                result.n_transitions,
                result.transition_rate,
                result.avg_transition_magnitude,
                result.max_transition_magnitude,
                result.avg_regime_duration,
                result.max_regime_duration,
                result.regime_entropy,
                result.edge_instability,
                result.boundary_artifact_flag,
                self.VERSION,
            ])
            
            # Transitions
            for t in result.transitions:
                self.conn.execute("""
                    INSERT OR REPLACE INTO phase1.detected_transitions
                    (run_id, indicator_id, window_years, transition_id, transition_date,
                     magnitude, direction_vector, sharpness, confidence,
                     pre_state_centroid, post_state_centroid)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, [
                    run_id,
                    result.indicator_id,
                    result.window_years,
                    t.transition_id,
                    t.transition_date.isoformat() if t.transition_date else None,
                    t.magnitude,
                    json.dumps(t.direction_vector.tolist()) if t.direction_vector is not None else None,
                    t.sharpness,
                    t.confidence,
                    json.dumps(t.pre_state_centroid.tolist()) if t.pre_state_centroid is not None else None,
                    json.dumps(t.post_state_centroid.tolist()) if t.post_state_centroid is not None else None,
                ])
            
            # Continuity details
            for c in result.continuity_details:
                self.conn.execute("""
                    INSERT OR REPLACE INTO phase1.window_continuity
                    (run_id, indicator_id, window_years, window_start, window_end,
                     next_window_start, step_distance, velocity, acceleration,
                     smoothness_score, is_discontinuity)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, [
                    run_id,
                    result.indicator_id,
                    result.window_years,
                    c.window_start.isoformat() if c.window_start else None,
                    c.window_end.isoformat() if c.window_end else None,
                    c.next_window_start.isoformat() if c.next_window_start else None,
                    c.step_distance,
                    c.velocity,
                    c.acceleration,
                    c.smoothness_score,
                    c.is_discontinuity,
                ])
        
        logger.info(f"Persisted persistence analysis to phase1 tables")
    
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
        description="Analyze temporal persistence and regime stability of geometry"
    )
    parser.add_argument("--run-id", required=True, help="Run ID to analyze")
    parser.add_argument("--dry-run", action="store_true", help="Analyze without persisting")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )

    agent = TemporalPersistenceAgent()
    
    try:
        if args.dry_run:
            results = agent.analyze_run(args.run_id)
            print(f"\nPersistence Analysis (DRY RUN):")
        else:
            results = agent.analyze_and_persist(args.run_id)
            print(f"\nPersistence Analysis (PERSISTED):")
        
        print(f"  Total trajectories analyzed: {len(results)}")
        
        # Summary stats
        if results:
            avg_persistence = np.mean([r.persistence_score for r in results])
            avg_stability = np.mean([r.stability_score for r in results])
            total_transitions = sum(r.n_transitions for r in results)
            artifacts = sum(1 for r in results if r.boundary_artifact_flag)
            
            print(f"  Average persistence score: {avg_persistence:.3f}")
            print(f"  Average stability score: {avg_stability:.3f}")
            print(f"  Total transitions detected: {total_transitions}")
            print(f"  Boundary artifacts flagged: {artifacts}")
        
    finally:
        agent.close()


if __name__ == "__main__":
    main()
