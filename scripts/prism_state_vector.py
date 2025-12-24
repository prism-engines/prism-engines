"""
PRISM State Vector Assembly

The State Vector is the fundamental object in PRISM.
All upstream computation exists to construct it.
All downstream reasoning operates upon it.

For an indicator i, evaluated over sampling context w:

    S(i, w) = [ G, M, C, P, T, S, R ]

Where:
    G = Geometry      (shape, dimension, boundedness)
    M = Memory        (persistence, temporal dependence)
    C = Complexity    (entropy, information structure)
    P = Periodicity   (cycles, oscillatory structure)
    T = Tails         (extremes, fragility, stability)
    S = Structure     (coupling, relational position)
    R = Resistance    (emergent stiffness - INFERRED)

The Resistance component is not measured directly.
It emerges from the stability of the state vector across contexts.
This is where we detect "hidden mass" - unmeasured forces that
create observable deformations in the geometry.

Cross-validated by: Claude
Date: December 2024
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
from datetime import datetime
import logging
from collections import defaultdict

import duckdb

logger = logging.getLogger(__name__)


# =============================================================================
# STATE VECTOR COMPONENTS
# =============================================================================

class ComponentType(Enum):
    """The seven components of the PRISM State Vector."""
    GEOMETRY = "G"       # Shape, dimension, boundedness
    MEMORY = "M"         # Persistence, temporal dependence
    COMPLEXITY = "C"     # Entropy, information structure
    PERIODICITY = "P"    # Cycles, oscillatory structure
    TAILS = "T"          # Extremes, fragility
    STRUCTURE = "S"      # Coupling, relational position
    RESISTANCE = "R"     # Emergent stiffness (inferred)


# Mapping from derived.geometry_descriptors dimensions to State Vector components
DIMENSION_TO_COMPONENT = {
    # G: Geometry
    "saturation": ComponentType.GEOMETRY,
    "saturation_metric": ComponentType.GEOMETRY,
    "asymmetry": ComponentType.GEOMETRY,
    "bounded_accumulation": ComponentType.GEOMETRY,
    "level_mean": ComponentType.GEOMETRY,
    "level_std": ComponentType.GEOMETRY,
    "effective_dimension": ComponentType.GEOMETRY,
    "curvature": ComponentType.GEOMETRY,
    "trend_strength": ComponentType.GEOMETRY,
    "drift_rate": ComponentType.GEOMETRY,
    
    # M: Memory
    "long_memory": ComponentType.MEMORY,
    "hurst": ComponentType.MEMORY,
    "hurst_exponent": ComponentType.MEMORY,
    "acf_decay": ComponentType.MEMORY,
    "acf_lag1": ComponentType.MEMORY,
    "pacf_lag1": ComponentType.MEMORY,
    "regime_stickiness": ComponentType.MEMORY,
    "mean_reversion_speed": ComponentType.MEMORY,
    
    # C: Complexity
    "shannon_entropy": ComponentType.COMPLEXITY,
    "spectral_entropy": ComponentType.COMPLEXITY,
    "permutation_entropy": ComponentType.COMPLEXITY,
    "sample_entropy": ComponentType.COMPLEXITY,
    "approximate_entropy": ComponentType.COMPLEXITY,
    "complexity_ratio": ComponentType.COMPLEXITY,
    "recurrence_rate": ComponentType.COMPLEXITY,
    "determinism": ComponentType.COMPLEXITY,
    
    # P: Periodicity
    "periodicity": ComponentType.PERIODICITY,
    "periodicity_concentration": ComponentType.PERIODICITY,
    "dominant_frequency": ComponentType.PERIODICITY,
    "dominant_period": ComponentType.PERIODICITY,
    "spectral_peak_ratio": ComponentType.PERIODICITY,
    "harmonic_concentration": ComponentType.PERIODICITY,
    "wavelet_energy_ratio": ComponentType.PERIODICITY,
    "dmd_dominant_frequency": ComponentType.PERIODICITY,
    
    # T: Tails
    "fat_tails": ComponentType.TAILS,
    "excess_kurtosis": ComponentType.TAILS,
    "skewness": ComponentType.TAILS,
    "kurtosis": ComponentType.TAILS,
    "tail_dependence": ComponentType.TAILS,
    "var_95": ComponentType.TAILS,
    "cvar_95": ComponentType.TAILS,
    "max_drawdown": ComponentType.TAILS,
    "volatility_clustering": ComponentType.TAILS,
    "arch_effect": ComponentType.TAILS,
    "parameter_instability": ComponentType.TAILS,
    
    # S: Structure (from pairwise/cohort analysis)
    "cluster_membership": ComponentType.STRUCTURE,
    "cohort_loading": ComponentType.STRUCTURE,
    "pca_loading_pc1": ComponentType.STRUCTURE,
    "pca_loading_pc2": ComponentType.STRUCTURE,
    "correlation_to_market": ComponentType.STRUCTURE,
    "beta": ComponentType.STRUCTURE,
    "idiosyncratic_vol": ComponentType.STRUCTURE,
}


@dataclass
class StateVectorComponent:
    """A single component of the state vector with its dimensions."""
    component_type: ComponentType
    dimensions: Dict[str, float]  # dimension_name -> value
    confidence: float = 1.0       # How confident are we in this component
    n_sources: int = 0            # How many dimensions contributed
    
    def to_array(self, dimension_order: List[str] = None) -> np.ndarray:
        """Convert to numpy array for geometric operations."""
        if dimension_order is None:
            dimension_order = sorted(self.dimensions.keys())
        return np.array([self.dimensions.get(d, np.nan) for d in dimension_order])
    
    @property
    def magnitude(self) -> float:
        """L2 norm of the component."""
        values = [v for v in self.dimensions.values() if np.isfinite(v)]
        if not values:
            return 0.0
        return np.linalg.norm(values)


@dataclass
class PRISMStateVector:
    """
    The complete PRISM State Vector for an indicator at a sampling context.
    
    S(i, w) = [ G, M, C, P, T, S, R ]
    
    This is the fundamental object in PRISM.
    """
    indicator_id: str
    window_start: str
    window_end: str
    
    # The seven components
    geometry: StateVectorComponent = None       # G
    memory: StateVectorComponent = None         # M
    complexity: StateVectorComponent = None     # C
    periodicity: StateVectorComponent = None    # P
    tails: StateVectorComponent = None          # T
    structure: StateVectorComponent = None      # S
    resistance: StateVectorComponent = None     # R (inferred)
    
    # Metadata
    run_id: str = None
    computed_at: datetime = None
    quality_score: float = 1.0  # Overall vector quality
    
    def __post_init__(self):
        if self.computed_at is None:
            self.computed_at = datetime.now()
    
    @property
    def components(self) -> Dict[ComponentType, StateVectorComponent]:
        """Return all components as a dict."""
        return {
            ComponentType.GEOMETRY: self.geometry,
            ComponentType.MEMORY: self.memory,
            ComponentType.COMPLEXITY: self.complexity,
            ComponentType.PERIODICITY: self.periodicity,
            ComponentType.TAILS: self.tails,
            ComponentType.STRUCTURE: self.structure,
            ComponentType.RESISTANCE: self.resistance,
        }
    
    @property
    def is_complete(self) -> bool:
        """Check if all components have data."""
        return all(c is not None and c.n_sources > 0 
                   for c in self.components.values() 
                   if c is not None)
    
    @property
    def completeness(self) -> float:
        """Fraction of components with data."""
        components = [c for c in self.components.values() if c is not None]
        if not components:
            return 0.0
        return sum(1 for c in components if c.n_sources > 0) / len(components)
    
    def to_flat_vector(self) -> Tuple[np.ndarray, List[str]]:
        """
        Flatten all components into a single vector for geometric operations.
        
        Returns:
            (vector, dimension_names) tuple
        """
        values = []
        names = []
        
        for comp_type in ComponentType:
            comp = self.components.get(comp_type)
            if comp is not None and comp.dimensions:
                for dim_name, value in sorted(comp.dimensions.items()):
                    values.append(value if np.isfinite(value) else 0.0)
                    names.append(f"{comp_type.value}_{dim_name}")
        
        return np.array(values), names
    
    def distance_to(self, other: 'PRISMStateVector', metric: str = 'euclidean') -> float:
        """
        Compute distance to another state vector.
        
        Args:
            other: Another PRISMStateVector
            metric: 'euclidean', 'mahalanobis', 'cosine'
        
        Returns:
            Distance value
        """
        v1, names1 = self.to_flat_vector()
        v2, names2 = other.to_flat_vector()
        
        # Align dimensions
        common_dims = set(names1) & set(names2)
        if not common_dims:
            return np.inf
        
        idx1 = [names1.index(d) for d in common_dims]
        idx2 = [names2.index(d) for d in common_dims]
        
        v1_aligned = v1[idx1]
        v2_aligned = v2[idx2]
        
        if metric == 'euclidean':
            return np.linalg.norm(v1_aligned - v2_aligned)
        elif metric == 'cosine':
            dot = np.dot(v1_aligned, v2_aligned)
            norm = np.linalg.norm(v1_aligned) * np.linalg.norm(v2_aligned)
            if norm == 0:
                return 1.0
            return 1.0 - dot / norm
        elif metric == 'mahalanobis':
            # Simplified - would need covariance from population
            diff = v1_aligned - v2_aligned
            return np.sqrt(np.sum(diff ** 2))
        else:
            raise ValueError(f"Unknown metric: {metric}")
    
    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            f"PRISM State Vector: {self.indicator_id}",
            f"Window: {self.window_start} to {self.window_end}",
            f"Completeness: {self.completeness:.1%}",
            f"Quality: {self.quality_score:.2f}",
            "",
        ]
        
        for comp_type in ComponentType:
            comp = self.components.get(comp_type)
            if comp is not None and comp.dimensions:
                lines.append(f"  {comp_type.name} ({comp_type.value}): {comp.n_sources} dims, mag={comp.magnitude:.3f}")
                for dim, val in sorted(comp.dimensions.items())[:3]:
                    lines.append(f"    {dim}: {val:.4f}")
                if len(comp.dimensions) > 3:
                    lines.append(f"    ... +{len(comp.dimensions)-3} more")
        
        return "\n".join(lines)


# =============================================================================
# STATE VECTOR EVOLUTION (Temporal Dynamics)
# =============================================================================

@dataclass
class StateVectorMotion:
    """
    Describes how a state vector evolves between two time contexts.
    
    This captures the "motion" through state space.
    """
    indicator_id: str
    from_window: Tuple[str, str]  # (start, end)
    to_window: Tuple[str, str]
    
    # Motion metrics
    displacement: float = 0.0      # Distance moved
    velocity: np.ndarray = None    # Direction of motion
    component_changes: Dict[ComponentType, float] = None  # Per-component change
    
    # Anomaly detection
    expected_displacement: float = None  # Based on historical motion
    residual: float = None               # Actual - expected (hidden force indicator)
    
    def __post_init__(self):
        if self.component_changes is None:
            self.component_changes = {}


@dataclass  
class StateVectorTrajectory:
    """
    The complete evolution of an indicator's state vector over time.
    
    This is where we detect hidden mass - by observing deformations
    that cannot be explained by measured forces.
    """
    indicator_id: str
    vectors: List[PRISMStateVector] = field(default_factory=list)
    motions: List[StateVectorMotion] = field(default_factory=list)
    
    # Derived properties
    mean_velocity: float = None
    velocity_variance: float = None
    
    # Hidden mass indicators
    total_unexplained_motion: float = 0.0
    anomalous_windows: List[Tuple[str, str]] = field(default_factory=list)
    
    @property
    def resistance_score(self) -> float:
        """
        Compute the Resistance (R) component.
        
        Resistance = how stable is the state vector across contexts?
        High resistance = indicator is hard to move
        Low resistance = indicator responds easily to perturbations
        
        This is inferred from trajectory stability, not measured directly.
        """
        if len(self.vectors) < 2:
            return np.nan
        
        # Compute variance of displacements
        if not self.motions:
            return np.nan
        
        displacements = [m.displacement for m in self.motions if m.displacement is not None]
        if not displacements:
            return np.nan
        
        # Resistance is inverse of mobility
        # Normalize to [0, 1] where 1 = maximum resistance
        mean_disp = np.mean(displacements)
        if mean_disp == 0:
            return 1.0  # No motion = maximum resistance
        
        # Use exponential mapping to bound in [0, 1]
        # Higher displacement = lower resistance
        resistance = np.exp(-mean_disp)
        return float(np.clip(resistance, 0, 1))
    
    def detect_anomalous_deformation(self, threshold: float = 2.0) -> List[Dict]:
        """
        Detect windows where motion exceeds expected bounds.
        
        This is the "dark matter detection" - finding deformations
        that suggest unmeasured forces are acting on the system.
        
        Args:
            threshold: Number of standard deviations for anomaly
            
        Returns:
            List of anomaly records with window, residual, interpretation
        """
        anomalies = []
        
        if len(self.motions) < 3:
            return anomalies
        
        displacements = [m.displacement for m in self.motions 
                        if m.displacement is not None and np.isfinite(m.displacement)]
        
        if len(displacements) < 3:
            return anomalies
        
        mean_disp = np.mean(displacements)
        std_disp = np.std(displacements)
        
        if std_disp == 0:
            return anomalies
        
        for motion in self.motions:
            if motion.displacement is None:
                continue
            
            z_score = (motion.displacement - mean_disp) / std_disp
            
            if abs(z_score) > threshold:
                anomalies.append({
                    'indicator_id': self.indicator_id,
                    'from_window': motion.from_window,
                    'to_window': motion.to_window,
                    'displacement': motion.displacement,
                    'expected': mean_disp,
                    'z_score': z_score,
                    'interpretation': self._interpret_anomaly(z_score, motion),
                })
        
        return anomalies
    
    def _interpret_anomaly(self, z_score: float, motion: StateVectorMotion) -> str:
        """Generate interpretation of anomalous motion."""
        if z_score > 0:
            direction = "excessive"
            implication = "Hidden force pushing indicator away from equilibrium"
        else:
            direction = "suppressed"
            implication = "Hidden force constraining indicator motion"
        
        # Check which components changed most
        if motion.component_changes:
            sorted_changes = sorted(
                motion.component_changes.items(),
                key=lambda x: abs(x[1]) if x[1] is not None else 0,
                reverse=True
            )
            top_component = sorted_changes[0][0].name if sorted_changes else "UNKNOWN"
        else:
            top_component = "UNKNOWN"
        
        return f"{direction.upper()} motion (z={z_score:.2f}), largest change in {top_component}. {implication}"


# =============================================================================
# STATE VECTOR ASSEMBLER
# =============================================================================

class StateVectorAssembler:
    """
    Assembles PRISM State Vectors from derived phase outputs.
    
    This is the core integration layer that:
    1. Loads all derived descriptors for an indicator
    2. Maps dimensions to state vector components
    3. Normalizes within components for comparability
    4. Computes the Resistance (R) component from trajectory stability
    5. Detects anomalous deformations suggesting hidden mass
    """
    
    def __init__(self, conn: duckdb.DuckDBPyConnection):
        self.conn = conn
        self._normalization_stats = {}  # Cache for z-score normalization
    
    def assemble_vector(
        self,
        indicator_id: str,
        window_start: str,
        window_end: str,
        run_id: str = None
    ) -> PRISMStateVector:
        """
        Assemble a complete state vector for one indicator/window.
        
        Args:
            indicator_id: The indicator to assemble
            window_start: Window start date
            window_end: Window end date
            run_id: Optional specific run_id filter
            
        Returns:
            PRISMStateVector with all available components
        """
        # Load all descriptors for this indicator/window
        descriptors = self._load_descriptors(indicator_id, window_start, window_end, run_id)
        
        # Group by component
        component_data = defaultdict(dict)
        
        for dim_name, value in descriptors.items():
            comp_type = DIMENSION_TO_COMPONENT.get(dim_name)
            if comp_type is not None:
                component_data[comp_type][dim_name] = value
        
        # Build components
        def make_component(comp_type: ComponentType) -> StateVectorComponent:
            dims = component_data.get(comp_type, {})
            return StateVectorComponent(
                component_type=comp_type,
                dimensions=dims,
                n_sources=len(dims),
                confidence=1.0 if dims else 0.0
            )
        
        # Create state vector
        sv = PRISMStateVector(
            indicator_id=indicator_id,
            window_start=window_start,
            window_end=window_end,
            geometry=make_component(ComponentType.GEOMETRY),
            memory=make_component(ComponentType.MEMORY),
            complexity=make_component(ComponentType.COMPLEXITY),
            periodicity=make_component(ComponentType.PERIODICITY),
            tails=make_component(ComponentType.TAILS),
            structure=make_component(ComponentType.STRUCTURE),
            resistance=None,  # Computed from trajectory
            run_id=run_id,
        )
        
        # Calculate quality score
        sv.quality_score = sv.completeness
        
        return sv
    
    def assemble_trajectory(
        self,
        indicator_id: str,
        min_windows: int = 3
    ) -> Optional[StateVectorTrajectory]:
        """
        Assemble complete trajectory for an indicator across all windows.
        
        This is where Resistance (R) emerges and hidden mass is detected.
        
        Args:
            indicator_id: The indicator
            min_windows: Minimum windows required for trajectory
            
        Returns:
            StateVectorTrajectory with motion analysis and resistance
        """
        # Get all windows for this indicator
        windows = self._get_indicator_windows(indicator_id)
        
        if len(windows) < min_windows:
            logger.warning(f"Insufficient windows for {indicator_id}: {len(windows)} < {min_windows}")
            return None
        
        # Assemble state vector for each window
        trajectory = StateVectorTrajectory(indicator_id=indicator_id)
        
        for window_start, window_end in windows:
            sv = self.assemble_vector(indicator_id, window_start, window_end)
            if sv.completeness > 0.3:  # Require at least 30% data
                trajectory.vectors.append(sv)
        
        if len(trajectory.vectors) < min_windows:
            return None
        
        # Compute motion between consecutive windows
        for i in range(len(trajectory.vectors) - 1):
            v1 = trajectory.vectors[i]
            v2 = trajectory.vectors[i + 1]
            
            motion = self._compute_motion(v1, v2)
            trajectory.motions.append(motion)
        
        # Compute trajectory-level statistics
        if trajectory.motions:
            displacements = [m.displacement for m in trajectory.motions 
                           if m.displacement is not None]
            if displacements:
                trajectory.mean_velocity = np.mean(displacements)
                trajectory.velocity_variance = np.var(displacements)
        
        # Inject Resistance into all vectors
        resistance_score = trajectory.resistance_score
        for sv in trajectory.vectors:
            sv.resistance = StateVectorComponent(
                component_type=ComponentType.RESISTANCE,
                dimensions={'resistance_score': resistance_score},
                n_sources=1,
                confidence=0.8 if len(trajectory.motions) >= 5 else 0.5
            )
        
        # Detect anomalies (hidden mass indicators)
        anomalies = trajectory.detect_anomalous_deformation()
        trajectory.anomalous_windows = [(a['from_window'], a['to_window']) for a in anomalies]
        trajectory.total_unexplained_motion = sum(
            abs(a['displacement'] - a['expected']) for a in anomalies
        )
        
        return trajectory
    
    def _load_descriptors(
        self,
        indicator_id: str,
        window_start: str,
        window_end: str,
        run_id: str = None
    ) -> Dict[str, float]:
        """Load all descriptor dimensions for an indicator/window."""
        query = """
            SELECT dimension, value
            FROM derived.geometry_descriptors
            WHERE indicator_id = ?
              AND window_start = ?
              AND window_end = ?
        """
        params = [indicator_id, window_start, window_end]
        
        if run_id:
            query += " AND run_id = ?"
            params.append(run_id)
        
        try:
            result = self.conn.execute(query, params).fetchall()
            return {row[0]: row[1] for row in result if row[1] is not None}
        except Exception as e:
            logger.error(f"Failed to load descriptors: {e}")
            return {}
    
    def _get_indicator_windows(self, indicator_id: str) -> List[Tuple[str, str]]:
        """Get all windows for an indicator, sorted chronologically."""
        try:
            result = self.conn.execute("""
                SELECT DISTINCT window_start, window_end
                FROM derived.geometry_descriptors
                WHERE indicator_id = ?
                ORDER BY window_start
            """, [indicator_id]).fetchall()
            return [(str(r[0]), str(r[1])) for r in result]
        except Exception as e:
            logger.error(f"Failed to get windows: {e}")
            return []
    
    def _compute_motion(
        self,
        v1: PRISMStateVector,
        v2: PRISMStateVector
    ) -> StateVectorMotion:
        """Compute motion between two state vectors."""
        motion = StateVectorMotion(
            indicator_id=v1.indicator_id,
            from_window=(v1.window_start, v1.window_end),
            to_window=(v2.window_start, v2.window_end),
        )
        
        # Total displacement
        motion.displacement = v1.distance_to(v2, metric='euclidean')
        
        # Per-component changes
        for comp_type in ComponentType:
            if comp_type == ComponentType.RESISTANCE:
                continue  # Skip resistance (it's derived)
            
            c1 = v1.components.get(comp_type)
            c2 = v2.components.get(comp_type)
            
            if c1 is not None and c2 is not None:
                # Compute magnitude change
                motion.component_changes[comp_type] = c2.magnitude - c1.magnitude
        
        # Velocity vector (direction of motion)
        vec1, names1 = v1.to_flat_vector()
        vec2, names2 = v2.to_flat_vector()
        
        if len(vec1) == len(vec2) and len(vec1) > 0:
            motion.velocity = vec2 - vec1
        
        return motion


# =============================================================================
# HIDDEN MASS DETECTOR
# =============================================================================

class HiddenMassDetector:
    """
    Detects unmeasured forces ("hidden mass") from state vector deformations.
    
    The core insight: if indicators that should move together don't,
    or if an indicator moves in ways not predicted by its measured coupling,
    something unmeasured is acting on the system.
    
    This is analogous to gravitational lensing - we infer mass from
    the curvature it creates in the observable space.
    """
    
    def __init__(self, assembler: StateVectorAssembler):
        self.assembler = assembler
        self.conn = assembler.conn
    
    def detect_cohort_deformation(
        self,
        cohort_indicators: List[str],
        window_start: str,
        window_end: str
    ) -> Dict[str, Any]:
        """
        Detect deformation within a cohort of indicators.
        
        If cohort members have high coupling but divergent motion,
        there's hidden mass affecting the system.
        
        Args:
            cohort_indicators: List of indicator IDs in the cohort
            window_start: Analysis window start
            window_end: Analysis window end
            
        Returns:
            Dict with deformation analysis
        """
        # Assemble state vectors for all cohort members
        vectors = []
        for ind_id in cohort_indicators:
            sv = self.assembler.assemble_vector(ind_id, window_start, window_end)
            if sv.completeness > 0.3:
                vectors.append(sv)
        
        if len(vectors) < 2:
            return {'status': 'insufficient_data', 'n_vectors': len(vectors)}
        
        # Compute pairwise distances
        distances = []
        for i, v1 in enumerate(vectors):
            for v2 in vectors[i+1:]:
                dist = v1.distance_to(v2)
                distances.append({
                    'ind1': v1.indicator_id,
                    'ind2': v2.indicator_id,
                    'distance': dist
                })
        
        # Compute cohort centroid
        flat_vectors = [v.to_flat_vector()[0] for v in vectors]
        
        # Align to same dimensionality (take intersection)
        min_len = min(len(v) for v in flat_vectors)
        aligned = np.array([v[:min_len] for v in flat_vectors])
        
        centroid = np.mean(aligned, axis=0)
        
        # Distances from centroid
        centroid_distances = []
        for i, (v, sv) in enumerate(zip(aligned, vectors)):
            dist = np.linalg.norm(v - centroid)
            centroid_distances.append({
                'indicator_id': sv.indicator_id,
                'distance_to_centroid': dist
            })
        
        # Detect outliers (potential hidden mass indicators)
        dist_values = [d['distance_to_centroid'] for d in centroid_distances]
        mean_dist = np.mean(dist_values)
        std_dist = np.std(dist_values)
        
        outliers = []
        for d in centroid_distances:
            if std_dist > 0:
                z = (d['distance_to_centroid'] - mean_dist) / std_dist
                if abs(z) > 2.0:
                    outliers.append({
                        **d,
                        'z_score': z,
                        'interpretation': 'Indicator deviates from cohort - possible hidden influence'
                    })
        
        return {
            'status': 'complete',
            'n_vectors': len(vectors),
            'mean_cohort_distance': mean_dist,
            'cohort_dispersion': std_dist,
            'pairwise_distances': distances,
            'centroid_distances': centroid_distances,
            'outliers': outliers,
            'hidden_mass_indicator': len(outliers) > 0,
        }
    
    def detect_trajectory_anomaly(
        self,
        indicator_id: str,
        reference_indicators: List[str] = None
    ) -> Dict[str, Any]:
        """
        Detect anomalous trajectory for an indicator.
        
        Compares an indicator's motion to:
        1. Its own historical motion (self-consistency)
        2. Reference indicators' motion (cross-consistency)
        
        Deviations suggest hidden mass affecting this indicator
        differently than others.
        """
        # Get trajectory for target indicator
        trajectory = self.assembler.assemble_trajectory(indicator_id)
        
        if trajectory is None:
            return {'status': 'insufficient_data'}
        
        result = {
            'status': 'complete',
            'indicator_id': indicator_id,
            'n_windows': len(trajectory.vectors),
            'resistance_score': trajectory.resistance_score,
            'mean_velocity': trajectory.mean_velocity,
            'velocity_variance': trajectory.velocity_variance,
            'self_anomalies': trajectory.detect_anomalous_deformation(),
        }
        
        # Compare to reference indicators if provided
        if reference_indicators:
            cross_anomalies = []
            
            for ref_id in reference_indicators:
                ref_trajectory = self.assembler.assemble_trajectory(ref_id)
                if ref_trajectory is None:
                    continue
                
                # Compare velocity patterns
                target_velocities = [m.displacement for m in trajectory.motions 
                                    if m.displacement is not None]
                ref_velocities = [m.displacement for m in ref_trajectory.motions
                                 if m.displacement is not None]
                
                if target_velocities and ref_velocities:
                    # Correlation of motion
                    min_len = min(len(target_velocities), len(ref_velocities))
                    if min_len >= 3:
                        corr = np.corrcoef(
                            target_velocities[:min_len],
                            ref_velocities[:min_len]
                        )[0, 1]
                        
                        cross_anomalies.append({
                            'reference': ref_id,
                            'motion_correlation': corr,
                            'divergent': corr < 0.3,  # Low correlation = divergent
                        })
            
            result['cross_anomalies'] = cross_anomalies
            result['n_divergent_references'] = sum(1 for a in cross_anomalies if a['divergent'])
        
        # Interpret hidden mass
        self_anomaly_count = len(result['self_anomalies'])
        cross_divergent = result.get('n_divergent_references', 0)
        
        if self_anomaly_count > 2 or cross_divergent > len(reference_indicators or []) / 2:
            result['hidden_mass_interpretation'] = (
                f"Strong hidden mass signal: {self_anomaly_count} self-anomalies, "
                f"{cross_divergent} divergent references. "
                "Indicator responds to forces not captured by measured variables."
            )
        elif self_anomaly_count > 0 or cross_divergent > 0:
            result['hidden_mass_interpretation'] = (
                f"Moderate hidden mass signal: {self_anomaly_count} self-anomalies, "
                f"{cross_divergent} divergent references. "
                "Some unexplained motion detected."
            )
        else:
            result['hidden_mass_interpretation'] = (
                "No significant hidden mass detected. "
                "Indicator motion consistent with measured forces."
            )
        
        return result


# =============================================================================
# PERSISTENCE LAYER
# =============================================================================

def persist_state_vector(conn: duckdb.DuckDBPyConnection, sv: PRISMStateVector) -> bool:
    """
    Persist a state vector to the database.
    
    Creates the structure.state_vectors table if needed.
    """
    # Ensure table exists
    conn.execute("""
        CREATE TABLE IF NOT EXISTS structure.state_vectors (
            indicator_id VARCHAR NOT NULL,
            window_start DATE NOT NULL,
            window_end DATE NOT NULL,
            component VARCHAR NOT NULL,
            dimension VARCHAR NOT NULL,
            value DOUBLE,
            confidence DOUBLE,
            run_id VARCHAR,
            computed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (indicator_id, window_start, window_end, component, dimension)
        )
    """)
    
    now = datetime.now()
    rows_written = 0
    
    for comp_type, comp in sv.components.items():
        if comp is None or not comp.dimensions:
            continue
        
        for dim_name, value in comp.dimensions.items():
            try:
                conn.execute("""
                    INSERT INTO structure.state_vectors
                    (indicator_id, window_start, window_end, component, dimension, 
                     value, confidence, run_id, computed_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT DO UPDATE SET 
                        value = EXCLUDED.value,
                        confidence = EXCLUDED.confidence,
                        computed_at = EXCLUDED.computed_at
                """, [
                    sv.indicator_id, sv.window_start, sv.window_end,
                    comp_type.value, dim_name, value,
                    comp.confidence, sv.run_id, now
                ])
                rows_written += 1
            except Exception as e:
                logger.error(f"Failed to persist {sv.indicator_id}/{comp_type.value}/{dim_name}: {e}")
    
    return rows_written > 0


def persist_trajectory_analysis(
    conn: duckdb.DuckDBPyConnection,
    trajectory: StateVectorTrajectory,
    anomalies: List[Dict]
) -> bool:
    """Persist trajectory analysis and anomaly detection results."""
    # Ensure tables exist
    conn.execute("""
        CREATE TABLE IF NOT EXISTS structure.trajectory_analysis (
            indicator_id VARCHAR PRIMARY KEY,
            n_windows INTEGER,
            resistance_score DOUBLE,
            mean_velocity DOUBLE,
            velocity_variance DOUBLE,
            total_unexplained_motion DOUBLE,
            n_anomalous_windows INTEGER,
            analyzed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    conn.execute("""
        CREATE TABLE IF NOT EXISTS structure.hidden_mass_signals (
            indicator_id VARCHAR NOT NULL,
            from_window_start DATE,
            from_window_end DATE,
            to_window_start DATE,
            to_window_end DATE,
            displacement DOUBLE,
            expected DOUBLE,
            z_score DOUBLE,
            interpretation VARCHAR,
            detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    now = datetime.now()
    
    # Persist trajectory summary
    try:
        conn.execute("""
            INSERT INTO structure.trajectory_analysis
            (indicator_id, n_windows, resistance_score, mean_velocity,
             velocity_variance, total_unexplained_motion, n_anomalous_windows, analyzed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (indicator_id) DO UPDATE SET
                n_windows = EXCLUDED.n_windows,
                resistance_score = EXCLUDED.resistance_score,
                mean_velocity = EXCLUDED.mean_velocity,
                velocity_variance = EXCLUDED.velocity_variance,
                total_unexplained_motion = EXCLUDED.total_unexplained_motion,
                n_anomalous_windows = EXCLUDED.n_anomalous_windows,
                analyzed_at = EXCLUDED.analyzed_at
        """, [
            trajectory.indicator_id,
            len(trajectory.vectors),
            trajectory.resistance_score,
            trajectory.mean_velocity,
            trajectory.velocity_variance,
            trajectory.total_unexplained_motion,
            len(trajectory.anomalous_windows),
            now
        ])
    except Exception as e:
        logger.error(f"Failed to persist trajectory: {e}")
        return False
    
    # Persist anomalies (hidden mass signals)
    for anomaly in anomalies:
        try:
            from_start, from_end = anomaly['from_window']
            to_start, to_end = anomaly['to_window']
            
            conn.execute("""
                INSERT INTO structure.hidden_mass_signals
                (indicator_id, from_window_start, from_window_end,
                 to_window_start, to_window_end, displacement, expected,
                 z_score, interpretation, detected_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                anomaly['indicator_id'],
                from_start, from_end, to_start, to_end,
                anomaly['displacement'], anomaly['expected'],
                anomaly['z_score'], anomaly['interpretation'], now
            ])
        except Exception as e:
            logger.debug(f"Failed to persist anomaly: {e}")
    
    return True


# =============================================================================
# CLI INTERFACE
# =============================================================================

def main():
    """CLI entry point for state vector assembly."""
    import argparse
    from prism.db.open import open_prism_db
    
    parser = argparse.ArgumentParser(description='PRISM State Vector Assembly')
    parser.add_argument('--indicator', type=str, help='Specific indicator to analyze')
    parser.add_argument('--all', action='store_true', help='Analyze all indicators')
    parser.add_argument('--detect-hidden-mass', action='store_true', 
                       help='Run hidden mass detection')
    parser.add_argument('--max-indicators', type=int, default=None,
                       help='Limit number of indicators')
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s'
    )
    
    conn = open_prism_db()
    assembler = StateVectorAssembler(conn)
    
    if args.indicator:
        # Single indicator analysis
        logger.info(f"Analyzing indicator: {args.indicator}")
        
        trajectory = assembler.assemble_trajectory(args.indicator)
        
        if trajectory:
            print(f"\n{'='*60}")
            print(f"TRAJECTORY: {args.indicator}")
            print(f"{'='*60}")
            print(f"Windows: {len(trajectory.vectors)}")
            print(f"Resistance (R): {trajectory.resistance_score:.4f}")
            print(f"Mean velocity: {trajectory.mean_velocity:.4f}")
            print(f"Velocity variance: {trajectory.velocity_variance:.4f}")
            
            anomalies = trajectory.detect_anomalous_deformation()
            if anomalies:
                print(f"\nANOMALIES DETECTED ({len(anomalies)}):")
                for a in anomalies:
                    print(f"  {a['from_window']} → {a['to_window']}")
                    print(f"    z-score: {a['z_score']:.2f}")
                    print(f"    {a['interpretation']}")
            else:
                print("\nNo anomalies detected.")
            
            # Print latest state vector
            if trajectory.vectors:
                print(f"\nLatest State Vector:")
                print(trajectory.vectors[-1].summary())
        else:
            print(f"Could not assemble trajectory for {args.indicator}")
    
    elif args.all or args.detect_hidden_mass:
        # Batch analysis
        indicators = conn.execute("""
            SELECT DISTINCT indicator_id 
            FROM derived.geometry_descriptors
        """).fetchall()
        indicators = [r[0] for r in indicators]
        
        if args.max_indicators:
            indicators = indicators[:args.max_indicators]
        
        logger.info(f"Analyzing {len(indicators)} indicators...")
        
        results = []
        for ind_id in indicators:
            trajectory = assembler.assemble_trajectory(ind_id)
            if trajectory:
                anomalies = trajectory.detect_anomalous_deformation()
                
                results.append({
                    'indicator_id': ind_id,
                    'n_windows': len(trajectory.vectors),
                    'resistance': trajectory.resistance_score,
                    'n_anomalies': len(anomalies),
                    'unexplained_motion': trajectory.total_unexplained_motion,
                })
                
                # Persist
                persist_trajectory_analysis(conn, trajectory, anomalies)
        
        # Summary
        print(f"\n{'='*60}")
        print("STATE VECTOR ANALYSIS SUMMARY")
        print(f"{'='*60}")
        print(f"Indicators analyzed: {len(results)}")
        
        if results:
            avg_resistance = np.mean([r['resistance'] for r in results if r['resistance'] is not None])
            total_anomalies = sum(r['n_anomalies'] for r in results)
            
            print(f"Average resistance: {avg_resistance:.4f}")
            print(f"Total anomalies: {total_anomalies}")
            
            # Top hidden mass indicators
            by_anomalies = sorted(results, key=lambda x: x['n_anomalies'], reverse=True)
            print(f"\nTop hidden mass indicators:")
            for r in by_anomalies[:10]:
                if r['n_anomalies'] > 0:
                    print(f"  {r['indicator_id']}: {r['n_anomalies']} anomalies, R={r['resistance']:.3f}")
    
    conn.close()


if __name__ == "__main__":
    main()
