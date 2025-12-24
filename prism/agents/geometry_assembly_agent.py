"""
PRISM Geometry Assembly Agent

Constructs system geometry from validated engine outputs.
Deterministic construction only - no interpretation, no filtering.

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
from datetime import datetime, date
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class GeometryVector:
    """A single point in geometry space."""
    indicator_id: str
    window_years: float
    window_start: date
    window_end: date
    
    vector: np.ndarray
    component_engines: List[str]
    component_weights: Optional[np.ndarray] = None
    coverage_fraction: float = 1.0
    effective_confidence: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "indicator_id": self.indicator_id,
            "window_years": self.window_years,
            "window_start": self.window_start.isoformat() if isinstance(self.window_start, date) else self.window_start,
            "window_end": self.window_end.isoformat() if isinstance(self.window_end, date) else self.window_end,
            "vector": json.dumps(self.vector.tolist()),
            "vector_dim": len(self.vector),
            "component_engines": ",".join(self.component_engines),
            "component_weights": json.dumps(self.component_weights.tolist()) if self.component_weights is not None else None,
            "coverage_fraction": self.coverage_fraction,
            "effective_confidence": self.effective_confidence,
        }


@dataclass
class GeometryTrajectory:
    """Time series of geometry vectors for an indicator."""
    indicator_id: str
    window_years: float
    
    vectors: List[GeometryVector] = field(default_factory=list)
    
    @property
    def n_points(self) -> int:
        return len(self.vectors)
    
    @property
    def start_date(self) -> Optional[date]:
        if not self.vectors:
            return None
        return min(v.window_start for v in self.vectors)
    
    @property
    def end_date(self) -> Optional[date]:
        if not self.vectors:
            return None
        return max(v.window_end for v in self.vectors)
    
    def compute_trajectory_stats(self) -> Dict[str, float]:
        """Compute trajectory statistics."""
        if len(self.vectors) < 2:
            return {
                "total_distance": 0.0,
                "avg_step_size": 0.0,
                "max_step_size": 0.0,
                "trajectory_length": 0.0,
                "displacement": 0.0,
            }
        
        # Sort by window start
        sorted_vectors = sorted(self.vectors, key=lambda v: v.window_start)
        
        # Compute step distances
        step_distances = []
        for i in range(1, len(sorted_vectors)):
            dist = np.linalg.norm(sorted_vectors[i].vector - sorted_vectors[i-1].vector)
            step_distances.append(dist)
        
        total_distance = sum(step_distances)
        trajectory_length = total_distance
        displacement = np.linalg.norm(sorted_vectors[-1].vector - sorted_vectors[0].vector)
        
        return {
            "total_distance": float(total_distance),
            "avg_step_size": float(np.mean(step_distances)) if step_distances else 0.0,
            "max_step_size": float(max(step_distances)) if step_distances else 0.0,
            "trajectory_length": float(trajectory_length),
            "displacement": float(displacement),
        }


@dataclass
class CoordinateFrame:
    """Defines how engine outputs map to geometry dimensions."""
    frame_id: str
    dimension_map: List[Dict[str, str]]  # List of {engine, output_key, transform}
    
    def __post_init__(self):
        for i, dim in enumerate(self.dimension_map):
            dim["dimension_index"] = i


class GeometryAssemblyAgent:
    """
    Constructs system geometry from validated engine outputs.
    
    This agent:
    - Reads validated engine outputs from DB
    - Constructs geometry vectors from engine outputs
    - Builds trajectories through geometry space
    - Defines coordinate frames
    
    This agent does NOT:
    - Make admission decisions
    - Filter outputs
    - Introduce domain semantics
    - Label or interpret geometry
    """
    
    VERSION = "1.0.0"
    
    # Default coordinate frame: which engine outputs become geometry dimensions
    DEFAULT_FRAME = [
        {"engine": "pca", "output_key": "pc1_loading", "transform": None},
        {"engine": "pca", "output_key": "pc2_loading", "transform": None},
        {"engine": "pca", "output_key": "explained_variance_1", "transform": None},
        {"engine": "hmm", "output_key": "state_entropy", "transform": None},
        {"engine": "hmm", "output_key": "transition_rate", "transform": None},
        {"engine": "correlation", "output_key": "mean_correlation", "transform": None},
        {"engine": "correlation", "output_key": "correlation_std", "transform": None},
        {"engine": "hurst", "output_key": "hurst_exponent", "transform": None},
        {"engine": "wavelet", "output_key": "dominant_scale", "transform": "log"},
        {"engine": "dmd", "output_key": "dominant_frequency", "transform": None},
    ]
    
    def __init__(self, coordinate_frame: Optional[List[Dict]] = None):
        self.coordinate_frame = CoordinateFrame(
            frame_id="default",
            dimension_map=coordinate_frame or self.DEFAULT_FRAME
        )
        self._conn = None

    @property
    def conn(self):
        if self._conn is None:
            from prism.db.open import open_prism_db
            self._conn = open_prism_db()
        return self._conn
    
    def assemble_geometry(self, run_id: str) -> List[GeometryTrajectory]:
        """
        Assemble geometry for all indicators in a run.
        
        Args:
            run_id: The run identifier
            
        Returns:
            List of GeometryTrajectory objects
        """
        logger.info(f"Assembling geometry for run_id={run_id}")
        
        # Get all validated outputs
        validated = self._load_validated_outputs(run_id)
        
        # Group by (indicator_id, window_years)
        grouped = self._group_outputs(validated)
        
        # Build trajectories
        trajectories = []
        for (indicator_id, window_years), outputs in grouped.items():
            trajectory = self._build_trajectory(
                run_id, indicator_id, window_years, outputs
            )
            if trajectory.n_points > 0:
                trajectories.append(trajectory)
        
        logger.info(f"Assembled {len(trajectories)} trajectories")
        return trajectories
    
    def assemble_and_persist(self, run_id: str) -> List[GeometryTrajectory]:
        """Assemble geometry and persist to DB."""
        trajectories = self.assemble_geometry(run_id)
        self._persist_trajectories(run_id, trajectories)
        self._persist_coordinate_frame(run_id)
        return trajectories
    
    def _load_validated_outputs(self, run_id: str) -> List[Dict]:
        """Load validated engine outputs (only valid/degraded)."""
        outputs = []
        
        # Get validation results
        try:
            validations = self.conn.execute("""
                SELECT indicator_id, window_years, engine, 
                       validity_flag, confidence_penalty
                FROM phase1.engine_validation
                WHERE run_id = ?
                  AND validity_flag IN ('valid', 'degraded')
            """, [run_id]).fetchdf()
        except Exception as e:
            logger.warning(f"Could not load validations: {e}")
            validations = None
        
        # Build effective confidence map
        confidence_map = {}
        if validations is not None:
            for _, row in validations.iterrows():
                key = (row["indicator_id"], row["window_years"], row["engine"])
                conf = 1.0 - row["confidence_penalty"]
                if row["validity_flag"] == "degraded":
                    conf *= 0.5
                confidence_map[key] = conf
        
        # Load engine outputs
        engine_tables = [
            ("engine_pca", "pca"),
            ("engine_hmm", "hmm"),
            ("engine_correlation", "correlation"),
            ("engine_granger", "granger"),
            ("engine_wavelet", "wavelet"),
            ("engine_dmd", "dmd"),
            ("engine_hurst", "hurst"),
        ]
        
        for table_name, engine_name in engine_tables:
            try:
                df = self.conn.execute(f"""
                    SELECT * FROM {table_name}
                    WHERE run_id = ?
                """, [run_id]).fetchdf()
                
                for _, row in df.iterrows():
                    indicator_id = row.get("indicator_id", "unknown")
                    window_years = row.get("window_years", 0)
                    
                    # Check if validated (or accept if no validation data)
                    key = (indicator_id, window_years, engine_name)
                    if validations is not None and key not in confidence_map:
                        continue  # Skip non-validated
                    
                    outputs.append({
                        "engine": engine_name,
                        "indicator_id": indicator_id,
                        "window_years": window_years,
                        "window_start": row.get("window_start"),
                        "window_end": row.get("window_end"),
                        "data": row.to_dict(),
                        "effective_confidence": confidence_map.get(key, 1.0),
                    })
            except Exception as e:
                logger.debug(f"Table {table_name} not found: {e}")
                continue
        
        return outputs
    
    def _group_outputs(self, outputs: List[Dict]) -> Dict[Tuple, List[Dict]]:
        """Group outputs by (indicator_id, window_years)."""
        grouped = {}
        for output in outputs:
            key = (output["indicator_id"], output["window_years"])
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(output)
        return grouped
    
    def _build_trajectory(
        self, 
        run_id: str,
        indicator_id: str, 
        window_years: float,
        outputs: List[Dict]
    ) -> GeometryTrajectory:
        """Build a geometry trajectory from engine outputs."""
        trajectory = GeometryTrajectory(
            indicator_id=indicator_id,
            window_years=window_years,
        )
        
        # Group by time window
        by_window = {}
        for output in outputs:
            window_key = (output.get("window_start"), output.get("window_end"))
            if window_key not in by_window:
                by_window[window_key] = []
            by_window[window_key].append(output)
        
        # Build geometry vector for each time window
        for (window_start, window_end), window_outputs in by_window.items():
            vector = self._build_vector(window_outputs)
            if vector is not None:
                trajectory.vectors.append(vector)
        
        return trajectory
    
    def _build_vector(self, outputs: List[Dict]) -> Optional[GeometryVector]:
        """Build a single geometry vector from engine outputs."""
        if not outputs:
            return None
        
        # Get window info from first output
        window_start = outputs[0].get("window_start")
        window_end = outputs[0].get("window_end")
        indicator_id = outputs[0]["indicator_id"]
        window_years = outputs[0]["window_years"]
        
        # Build output map by engine
        engine_outputs = {}
        engine_confidences = {}
        for output in outputs:
            engine = output["engine"]
            engine_outputs[engine] = output["data"]
            engine_confidences[engine] = output.get("effective_confidence", 1.0)
        
        # Extract vector components according to coordinate frame
        components = []
        component_engines = []
        component_weights = []
        
        for dim_spec in self.coordinate_frame.dimension_map:
            engine = dim_spec["engine"]
            output_key = dim_spec["output_key"]
            transform = dim_spec.get("transform")
            
            if engine not in engine_outputs:
                # Missing engine - use NaN
                components.append(np.nan)
                component_engines.append(engine)
                component_weights.append(0.0)
                continue
            
            value = engine_outputs[engine].get(output_key)
            if value is None:
                components.append(np.nan)
                component_engines.append(engine)
                component_weights.append(0.0)
                continue
            
            # Apply transform if specified
            if transform == "log" and value > 0:
                value = np.log(value)
            elif transform == "sqrt" and value >= 0:
                value = np.sqrt(value)
            
            components.append(float(value))
            component_engines.append(engine)
            component_weights.append(engine_confidences.get(engine, 1.0))
        
        # Compute coverage and confidence
        valid_count = sum(1 for c in components if not np.isnan(c))
        coverage = valid_count / len(components) if components else 0.0
        
        # Replace NaN with 0 for vector math (but record coverage)
        vector = np.array(components)
        vector = np.nan_to_num(vector, nan=0.0)
        
        # Effective confidence is weighted average
        weights = np.array(component_weights)
        if np.sum(weights) > 0:
            effective_conf = np.sum(weights * (1 - np.isnan(np.array(components)).astype(float))) / np.sum(weights)
        else:
            effective_conf = 0.0
        
        return GeometryVector(
            indicator_id=indicator_id,
            window_years=window_years,
            window_start=window_start,
            window_end=window_end,
            vector=vector,
            component_engines=list(set(component_engines)),
            component_weights=np.array(component_weights),
            coverage_fraction=coverage,
            effective_confidence=float(effective_conf),
        )
    
    def _persist_trajectories(self, run_id: str, trajectories: List[GeometryTrajectory]):
        """Persist geometry to Phase-1 tables."""
        logger.info(f"Persisting {len(trajectories)} trajectories")

        # Assert required tables exist (no runtime schema creation)
        from prism.db.schema_guard import assert_tables_exist
        assert_tables_exist(self.conn, [
            "phase1.geometry_vectors",
            "phase1.geometry_trajectories"
        ])
        
        # Insert data
        for traj in trajectories:
            # Insert vectors
            for vec in traj.vectors:
                d = vec.to_dict()
                self.conn.execute("""
                    INSERT OR REPLACE INTO phase1.geometry_vectors
                    (run_id, indicator_id, window_years, window_start, window_end,
                     vector, vector_dim, component_engines, component_weights,
                     coverage_fraction, effective_confidence, assembler_version)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, [
                    run_id,
                    d["indicator_id"],
                    d["window_years"],
                    d["window_start"],
                    d["window_end"],
                    d["vector"],
                    d["vector_dim"],
                    d["component_engines"],
                    d["component_weights"],
                    d["coverage_fraction"],
                    d["effective_confidence"],
                    self.VERSION,
                ])
            
            # Insert trajectory stats
            stats = traj.compute_trajectory_stats()
            self.conn.execute("""
                INSERT OR REPLACE INTO phase1.geometry_trajectories
                (run_id, indicator_id, window_years, n_points, start_date, end_date,
                 total_distance, avg_step_size, max_step_size, trajectory_length,
                 displacement, assembler_version)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                run_id,
                traj.indicator_id,
                traj.window_years,
                traj.n_points,
                traj.start_date.isoformat() if traj.start_date else None,
                traj.end_date.isoformat() if traj.end_date else None,
                stats["total_distance"],
                stats["avg_step_size"],
                stats["max_step_size"],
                stats["trajectory_length"],
                stats["displacement"],
                self.VERSION,
            ])
        
        logger.info(f"Persisted geometry to phase1.geometry_vectors and phase1.geometry_trajectories")
    
    def _persist_coordinate_frame(self, run_id: str):
        """Persist coordinate frame definition."""
        # Assert required table exists (no runtime schema creation)
        from prism.db.schema_guard import assert_table_exists
        assert_table_exists(self.conn, "phase1.geometry_coordinate_frames")
        
        for dim in self.coordinate_frame.dimension_map:
            self.conn.execute("""
                INSERT OR REPLACE INTO phase1.geometry_coordinate_frames
                (run_id, frame_id, dimension_index, engine, engine_output_key, transform)
                VALUES (?, ?, ?, ?, ?, ?)
            """, [
                run_id,
                self.coordinate_frame.frame_id,
                dim["dimension_index"],
                dim["engine"],
                dim["output_key"],
                dim.get("transform"),
            ])
    
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
        description="Assemble system geometry from validated engine outputs"
    )
    parser.add_argument("--run-id", required=True, help="Run ID to assemble")
    parser.add_argument("--dry-run", action="store_true", help="Assemble without persisting")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )

    agent = GeometryAssemblyAgent()
    
    try:
        if args.dry_run:
            trajectories = agent.assemble_geometry(args.run_id)
            print(f"\nGeometry Assembly (DRY RUN):")
        else:
            trajectories = agent.assemble_and_persist(args.run_id)
            print(f"\nGeometry Assembly (PERSISTED):")
        
        print(f"  Total trajectories: {len(trajectories)}")
        
        total_points = sum(t.n_points for t in trajectories)
        print(f"  Total geometry points: {total_points}")
        
        if trajectories:
            avg_coverage = np.mean([
                np.mean([v.coverage_fraction for v in t.vectors])
                for t in trajectories if t.vectors
            ])
            print(f"  Average coverage: {avg_coverage:.2%}")
        
    finally:
        agent.close()


if __name__ == "__main__":
    main()
