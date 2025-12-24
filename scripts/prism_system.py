"""
PRISM System Geometry

This module defines the BOUNDED GEOMETRY within which indicators exist.

An indicator doesn't behave in isolation. It exists within a system of
coupled indicators that constrain its motion. This module:

1. Constructs the system geometry from correlation/cohort structure
2. Positions each indicator within that geometry  
3. Assesses the constraints each indicator feels
4. Computes relevance metrics (centrality, influence, sensitivity)

The output is a GEOMETRIC CONTEXT for each indicator - the "box" it
operates within. The State Vector then describes behavior WITHIN that box.
Hidden mass emerges when motion violates the box constraints.

Conceptual Framework:
────────────────────

    SYSTEM GEOMETRY                 INDICATOR POSITION
    ───────────────                 ──────────────────
    
    ┌─────────────────────┐         • Distance from centroid
    │                     │         • Alignment with principal axes  
    │    Principal        │         • Cohort membership strength
    │    Axes (PCA)       │         • Coupling to neighbors
    │         ↑           │         
    │         │     •A    │         CONSTRAINTS FELT
    │         │  •B       │         ────────────────
    │    ─────┼─────→     │         • Freedom of motion
    │         │   •C      │         • Expected trajectory
    │         │           │         • Soft boundaries
    │    •D   │           │         
    │                     │         RELEVANCE
    └─────────────────────┘         ─────────
                                    • Centrality (position)
                                    • Influence (outward coupling)
                                    • Sensitivity (inward coupling)

Cross-validated by: Claude
Date: December 2024
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Set
from enum import Enum
from datetime import datetime
import logging
from collections import defaultdict

import duckdb

logger = logging.getLogger(__name__)


# =============================================================================
# GEOMETRIC POSITION
# =============================================================================

@dataclass
class GeometricPosition:
    """
    An indicator's position within the system geometry.
    
    This describes WHERE the indicator sits, not how it behaves.
    Behavior comes from the State Vector; position comes from here.
    """
    indicator_id: str
    window_start: str
    window_end: str
    
    # Position relative to system
    centroid_distance: float = 0.0       # Distance from system centroid
    centroid_direction: np.ndarray = None  # Unit vector toward indicator from centroid
    
    # Alignment with principal structure
    pc1_loading: float = 0.0             # Loading on first principal component
    pc2_loading: float = 0.0             # Loading on second principal component
    pc3_loading: float = 0.0             # Loading on third principal component
    alignment_angle: float = 0.0          # Angle from PC1 axis (radians)
    
    # Cohort positioning
    primary_cohort: str = None           # Strongest cohort membership
    cohort_centrality: float = 0.0       # How central within primary cohort
    cross_cohort_coupling: float = 0.0   # Coupling to other cohorts
    
    # Local neighborhood
    n_neighbors: int = 0                 # Number of strongly coupled neighbors
    mean_neighbor_distance: float = 0.0  # Average distance to neighbors
    isolation_score: float = 0.0         # How isolated (0=central, 1=peripheral)
    
    def to_vector(self) -> np.ndarray:
        """Flatten position to vector for distance calculations."""
        return np.array([
            self.centroid_distance,
            self.pc1_loading,
            self.pc2_loading,
            self.pc3_loading,
            self.cohort_centrality,
            self.cross_cohort_coupling,
            self.isolation_score,
        ])


@dataclass
class BehavioralConstraints:
    """
    The constraints an indicator feels given its geometric position.
    
    These are the "walls of the box" - the soft boundaries that
    define expected behavior. Violations suggest hidden forces.
    """
    indicator_id: str
    window_start: str
    window_end: str
    
    # Freedom of motion
    degrees_of_freedom: float = 0.0      # Effective dimensionality of allowed motion
    motion_volume: float = 0.0           # Volume of typical motion envelope
    
    # Expected trajectory characteristics
    expected_drift: np.ndarray = None    # Expected direction of motion
    expected_volatility: float = 0.0     # Expected magnitude of motion
    drift_confidence: float = 0.0        # Confidence in drift estimate
    
    # Coupling constraints
    leader_indicators: List[str] = field(default_factory=list)  # Who this follows
    follower_indicators: List[str] = field(default_factory=list)  # Who follows this
    sync_requirement: float = 0.0        # How synchronized must motion be
    
    # Boundary conditions
    soft_ceiling: float = None           # Upper behavioral bound
    soft_floor: float = None             # Lower behavioral bound
    mean_reversion_strength: float = 0.0  # Pull toward equilibrium


@dataclass
class IndicatorRelevance:
    """
    How relevant is this indicator to understanding the system?
    
    Relevance is RELATIONAL, not intrinsic. An indicator matters
    because of its position and coupling, not its own properties.
    """
    indicator_id: str
    window_start: str
    window_end: str
    
    # Centrality metrics
    structural_centrality: float = 0.0   # Position-based importance
    eigenvector_centrality: float = 0.0  # Network-based importance
    betweenness: float = 0.0             # Bridge between clusters
    
    # Influence metrics
    influence_radius: float = 0.0        # How far does motion propagate
    influence_strength: float = 0.0      # How strongly does it affect others
    granger_out_degree: int = 0          # Number of indicators it Granger-causes
    
    # Sensitivity metrics
    sensitivity_radius: float = 0.0      # How far away can perturbations come from
    sensitivity_strength: float = 0.0    # How strongly is it affected
    granger_in_degree: int = 0           # Number of indicators that Granger-cause it
    
    # Composite scores
    systemic_importance: float = 0.0     # Overall importance to system
    information_value: float = 0.0       # How much does observing it tell us
    
    @property
    def influence_sensitivity_ratio(self) -> float:
        """Ratio of influence to sensitivity. >1 = leader, <1 = follower."""
        if self.sensitivity_strength == 0:
            return float('inf') if self.influence_strength > 0 else 1.0
        return self.influence_strength / self.sensitivity_strength


@dataclass
class BoundedGeometry:
    """
    Complete geometric context for an indicator.
    
    This is the "box" within which the indicator operates.
    Combines position, constraints, and relevance.
    """
    indicator_id: str
    window_start: str
    window_end: str
    
    position: GeometricPosition = None
    constraints: BehavioralConstraints = None
    relevance: IndicatorRelevance = None
    
    # Computed properties
    geometric_fit: float = 1.0           # How well does indicator fit the geometry
    anomaly_potential: float = 0.0       # Likelihood of exhibiting anomalous motion
    
    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            f"Bounded Geometry: {self.indicator_id}",
            f"Window: {self.window_start} to {self.window_end}",
            "",
            "POSITION:",
            f"  Centroid distance: {self.position.centroid_distance:.3f}",
            f"  PC1 loading: {self.position.pc1_loading:.3f}",
            f"  Primary cohort: {self.position.primary_cohort}",
            f"  Isolation: {self.position.isolation_score:.3f}",
            "",
            "CONSTRAINTS:",
            f"  Degrees of freedom: {self.constraints.degrees_of_freedom:.2f}",
            f"  Expected volatility: {self.constraints.expected_volatility:.4f}",
            f"  Mean reversion: {self.constraints.mean_reversion_strength:.3f}",
            "",
            "RELEVANCE:",
            f"  Structural centrality: {self.relevance.structural_centrality:.3f}",
            f"  Influence strength: {self.relevance.influence_strength:.3f}",
            f"  Sensitivity strength: {self.relevance.sensitivity_strength:.3f}",
            f"  Systemic importance: {self.relevance.systemic_importance:.3f}",
        ]
        return "\n".join(lines)


# =============================================================================
# SYSTEM GEOMETRY
# =============================================================================

@dataclass
class SystemGeometry:
    """
    The complete geometry of the indicator system.
    
    This is the "space" within which all indicators exist.
    It defines the coordinate system, principal axes, clusters,
    and overall structure that constrains individual behavior.
    """
    window_start: str
    window_end: str
    
    # Dimensionality
    n_indicators: int = 0
    effective_dimension: float = 0.0     # Number of independent factors
    explained_variance_ratio: np.ndarray = None  # Variance per component
    
    # Principal structure
    principal_axes: np.ndarray = None    # Principal component directions
    centroid: np.ndarray = None          # System centroid in PC space
    
    # Cluster structure
    n_cohorts: int = 0
    cohort_ids: List[str] = field(default_factory=list)
    cohort_sizes: Dict[str, int] = field(default_factory=dict)
    cohort_centroids: Dict[str, np.ndarray] = field(default_factory=dict)
    
    # Coupling structure
    mean_correlation: float = 0.0
    correlation_dispersion: float = 0.0
    network_density: float = 0.0         # Fraction of significant couplings
    
    # Stability
    geometric_stability: float = 0.0     # How stable is this geometry over time
    eigenvalue_gaps: List[float] = field(default_factory=list)  # Gaps between PCs
    
    # Individual positions
    indicator_geometries: Dict[str, BoundedGeometry] = field(default_factory=dict)
    
    def get_indicator(self, indicator_id: str) -> Optional[BoundedGeometry]:
        """Get bounded geometry for a specific indicator."""
        return self.indicator_geometries.get(indicator_id)
    
    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            f"System Geometry: {self.window_start} to {self.window_end}",
            f"Indicators: {self.n_indicators}",
            f"Effective dimension: {self.effective_dimension:.2f}",
            f"Cohorts: {self.n_cohorts}",
            f"Mean correlation: {self.mean_correlation:.3f}",
            f"Network density: {self.network_density:.3f}",
            f"Geometric stability: {self.geometric_stability:.3f}",
        ]
        
        if self.explained_variance_ratio is not None:
            cumvar = np.cumsum(self.explained_variance_ratio)
            for i, (var, cum) in enumerate(zip(self.explained_variance_ratio[:5], cumvar[:5])):
                lines.append(f"  PC{i+1}: {var:.1%} ({cum:.1%} cumulative)")
        
        return "\n".join(lines)


# =============================================================================
# GEOMETRY CONSTRUCTOR
# =============================================================================

class SystemGeometryConstructor:
    """
    Constructs the system geometry from derived phase outputs.
    
    This is the primary class that builds the "bounded box" for
    each indicator based on correlation structure, PCA, cohorts,
    and coupling analysis.
    """
    
    def __init__(self, conn: duckdb.DuckDBPyConnection):
        self.conn = conn
    
    def construct(
        self,
        window_start: str,
        window_end: str,
        min_correlation: float = 0.3,
        min_indicators: int = 5
    ) -> Optional[SystemGeometry]:
        """
        Construct the complete system geometry for a window.
        
        Args:
            window_start: Window start date
            window_end: Window end date
            min_correlation: Minimum correlation for coupling
            min_indicators: Minimum indicators required
            
        Returns:
            SystemGeometry with all indicator positions and constraints
        """
        logger.info(f"Constructing system geometry for {window_start} to {window_end}")
        
        # Load correlation matrix
        corr_matrix, indicators = self._load_correlation_matrix(window_start, window_end)
        
        if corr_matrix is None or len(indicators) < min_indicators:
            logger.warning(f"Insufficient data for system geometry: {len(indicators) if indicators else 0} indicators")
            return None
        
        # Initialize geometry
        geometry = SystemGeometry(
            window_start=window_start,
            window_end=window_end,
            n_indicators=len(indicators),
        )
        
        # Compute principal structure
        self._compute_principal_structure(geometry, corr_matrix, indicators)
        
        # Compute cluster structure
        self._compute_cluster_structure(geometry, corr_matrix, indicators, min_correlation)
        
        # Compute coupling structure
        self._compute_coupling_structure(geometry, corr_matrix, indicators, min_correlation)
        
        # Position each indicator
        for i, ind_id in enumerate(indicators):
            bounded = self._compute_indicator_geometry(
                ind_id, i, geometry, corr_matrix, indicators,
                window_start, window_end, min_correlation
            )
            geometry.indicator_geometries[ind_id] = bounded
        
        logger.info(f"Constructed geometry: {geometry.n_indicators} indicators, "
                   f"dim={geometry.effective_dimension:.1f}, "
                   f"{geometry.n_cohorts} cohorts")
        
        return geometry
    
    def _load_correlation_matrix(
        self,
        window_start: str,
        window_end: str
    ) -> Tuple[Optional[np.ndarray], Optional[List[str]]]:
        """Load correlation matrix from derived phase."""
        try:
            # Get all correlations for this window
            result = self.conn.execute("""
                SELECT indicator_id_1, indicator_id_2, correlation
                FROM derived.correlation_matrix
                WHERE window_start = ? AND window_end = ?
            """, [window_start, window_end]).fetchall()
            
            if not result:
                return None, None
            
            # Build indicator list
            indicators = set()
            for row in result:
                indicators.add(row[0])
                indicators.add(row[1])
            indicators = sorted(list(indicators))
            
            n = len(indicators)
            ind_to_idx = {ind: i for i, ind in enumerate(indicators)}
            
            # Build correlation matrix
            corr_matrix = np.eye(n)
            for ind1, ind2, corr in result:
                i, j = ind_to_idx[ind1], ind_to_idx[ind2]
                corr_matrix[i, j] = corr
                corr_matrix[j, i] = corr
            
            return corr_matrix, indicators
            
        except Exception as e:
            logger.error(f"Failed to load correlation matrix: {e}")
            return None, None
    
    def _compute_principal_structure(
        self,
        geometry: SystemGeometry,
        corr_matrix: np.ndarray,
        indicators: List[str]
    ):
        """Compute PCA structure from correlation matrix."""
        try:
            # Eigendecomposition of correlation matrix
            eigenvalues, eigenvectors = np.linalg.eigh(corr_matrix)
            
            # Sort by descending eigenvalue
            idx = np.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]
            
            # Explained variance
            total_var = np.sum(eigenvalues)
            explained_ratio = eigenvalues / total_var if total_var > 0 else eigenvalues
            
            # Effective dimension (number of components for 90% variance)
            cumvar = np.cumsum(explained_ratio)
            effective_dim = np.searchsorted(cumvar, 0.9) + 1
            
            # Eigenvalue gaps
            gaps = []
            for i in range(min(5, len(eigenvalues)-1)):
                if eigenvalues[i+1] > 0:
                    gaps.append(eigenvalues[i] / eigenvalues[i+1])
            
            geometry.principal_axes = eigenvectors
            geometry.explained_variance_ratio = explained_ratio
            geometry.effective_dimension = float(effective_dim)
            geometry.eigenvalue_gaps = gaps
            
            # Project indicators to PC space and compute centroid
            # (Using eigenvectors as loadings)
            pc_loadings = eigenvectors[:, :min(10, len(eigenvalues))]
            geometry.centroid = np.mean(pc_loadings, axis=0)
            
        except Exception as e:
            logger.error(f"Failed to compute principal structure: {e}")
    
    def _compute_cluster_structure(
        self,
        geometry: SystemGeometry,
        corr_matrix: np.ndarray,
        indicators: List[str],
        min_correlation: float
    ):
        """Identify cohorts from correlation structure."""
        try:
            # Load cohort assignments if available
            result = self.conn.execute("""
                SELECT DISTINCT cohort_id
                FROM derived.cohort_descriptors
                WHERE window_start = ? AND window_end = ?
            """, [geometry.window_start, geometry.window_end]).fetchall()
            
            if result:
                cohort_ids = [r[0] for r in result]
                geometry.cohort_ids = cohort_ids
                geometry.n_cohorts = len(cohort_ids)
                
                # Get cohort sizes
                for cohort_id in cohort_ids:
                    members = self.conn.execute("""
                        SELECT DISTINCT indicator_id 
                        FROM meta.cohort_membership
                        WHERE cohort_id = ?
                    """, [cohort_id]).fetchall()
                    geometry.cohort_sizes[cohort_id] = len(members)
            else:
                # Simple clustering based on correlation threshold
                # Connected components of correlation graph
                n = len(indicators)
                adjacency = corr_matrix > min_correlation
                np.fill_diagonal(adjacency, False)
                
                # Find connected components (simple BFS)
                visited = set()
                components = []
                
                for start in range(n):
                    if start in visited:
                        continue
                    
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
                        components.append(component)
                
                geometry.n_cohorts = len(components)
                geometry.cohort_ids = [f"cohort_{i}" for i in range(len(components))]
                
                for i, component in enumerate(components):
                    geometry.cohort_sizes[f"cohort_{i}"] = len(component)
                    
        except Exception as e:
            logger.error(f"Failed to compute cluster structure: {e}")
    
    def _compute_coupling_structure(
        self,
        geometry: SystemGeometry,
        corr_matrix: np.ndarray,
        indicators: List[str],
        min_correlation: float
    ):
        """Compute overall coupling statistics."""
        n = len(indicators)
        
        # Get upper triangle (exclude diagonal)
        upper_tri = corr_matrix[np.triu_indices(n, k=1)]
        
        geometry.mean_correlation = float(np.mean(np.abs(upper_tri)))
        geometry.correlation_dispersion = float(np.std(upper_tri))
        
        # Network density = fraction of significant couplings
        n_significant = np.sum(np.abs(upper_tri) > min_correlation)
        n_possible = len(upper_tri)
        geometry.network_density = n_significant / n_possible if n_possible > 0 else 0
    
    def _compute_indicator_geometry(
        self,
        indicator_id: str,
        idx: int,
        geometry: SystemGeometry,
        corr_matrix: np.ndarray,
        indicators: List[str],
        window_start: str,
        window_end: str,
        min_correlation: float
    ) -> BoundedGeometry:
        """Compute complete bounded geometry for one indicator."""
        
        # POSITION
        position = GeometricPosition(
            indicator_id=indicator_id,
            window_start=window_start,
            window_end=window_end,
        )
        
        # PC loadings (from eigenvector rows)
        if geometry.principal_axes is not None:
            loadings = geometry.principal_axes[idx, :]
            position.pc1_loading = float(loadings[0]) if len(loadings) > 0 else 0.0
            position.pc2_loading = float(loadings[1]) if len(loadings) > 1 else 0.0
            position.pc3_loading = float(loadings[2]) if len(loadings) > 2 else 0.0
            
            # Distance from centroid
            if geometry.centroid is not None:
                indicator_pos = loadings[:len(geometry.centroid)]
                position.centroid_distance = float(np.linalg.norm(indicator_pos - geometry.centroid))
                
                # Direction to indicator from centroid
                direction = indicator_pos - geometry.centroid
                norm = np.linalg.norm(direction)
                position.centroid_direction = direction / norm if norm > 0 else direction
            
            # Alignment angle (angle from PC1 in PC1-PC2 plane)
            if len(loadings) >= 2:
                position.alignment_angle = float(np.arctan2(loadings[1], loadings[0]))
        
        # Neighbor analysis
        row = corr_matrix[idx, :]
        neighbors = np.where(np.abs(row) > min_correlation)[0]
        neighbors = neighbors[neighbors != idx]  # Exclude self
        position.n_neighbors = len(neighbors)
        
        if len(neighbors) > 0:
            neighbor_corrs = np.abs(row[neighbors])
            position.mean_neighbor_distance = float(1 - np.mean(neighbor_corrs))
        
        # Isolation score
        position.isolation_score = float(1 - np.mean(np.abs(row)))
        
        # CONSTRAINTS
        constraints = BehavioralConstraints(
            indicator_id=indicator_id,
            window_start=window_start,
            window_end=window_end,
        )
        
        # Degrees of freedom (based on how constrained by neighbors)
        # More neighbors = more constraints = fewer degrees of freedom
        constraints.degrees_of_freedom = max(1.0, geometry.effective_dimension - position.n_neighbors * 0.5)
        
        # Expected volatility (load from derived descriptors)
        vol = self._load_descriptor(indicator_id, window_start, window_end, 'level_std')
        constraints.expected_volatility = vol if vol is not None else 0.0
        
        # Mean reversion strength (from Hurst exponent)
        hurst = self._load_descriptor(indicator_id, window_start, window_end, 'hurst')
        if hurst is not None:
            # Hurst < 0.5 = mean reverting, > 0.5 = trending
            constraints.mean_reversion_strength = max(0, 1 - 2 * hurst)
        
        # RELEVANCE
        relevance = IndicatorRelevance(
            indicator_id=indicator_id,
            window_start=window_start,
            window_end=window_end,
        )
        
        # Structural centrality (inverse of isolation)
        relevance.structural_centrality = 1 - position.isolation_score
        
        # Eigenvector centrality (approximate from PC1 loading)
        relevance.eigenvector_centrality = abs(position.pc1_loading)
        
        # Influence and sensitivity (from pairwise Granger if available)
        granger_out, granger_in = self._load_granger_degrees(indicator_id, window_start, window_end)
        relevance.granger_out_degree = granger_out
        relevance.granger_in_degree = granger_in
        
        # Influence strength = mean absolute correlation to others
        relevance.influence_strength = float(np.mean(np.abs(row)))
        relevance.sensitivity_strength = relevance.influence_strength  # Symmetric for correlation
        
        # Systemic importance = combination of centrality and influence
        relevance.systemic_importance = (
            0.4 * relevance.structural_centrality +
            0.3 * relevance.eigenvector_centrality +
            0.3 * relevance.influence_strength
        )
        
        # Information value (how much variance it captures)
        if geometry.explained_variance_ratio is not None:
            # Weighted contribution to principal components
            info_value = 0
            for i, var_ratio in enumerate(geometry.explained_variance_ratio[:5]):
                if i < len(geometry.principal_axes[idx]):
                    info_value += var_ratio * abs(geometry.principal_axes[idx, i])
            relevance.information_value = float(info_value)
        
        # BUILD BOUNDED GEOMETRY
        bounded = BoundedGeometry(
            indicator_id=indicator_id,
            window_start=window_start,
            window_end=window_end,
            position=position,
            constraints=constraints,
            relevance=relevance,
        )
        
        # Geometric fit (how well does indicator conform to system structure)
        bounded.geometric_fit = 1 - position.isolation_score * 0.5
        
        # Anomaly potential (based on position and constraints)
        # High isolation + low constraints = high anomaly potential
        bounded.anomaly_potential = position.isolation_score * (1 / (1 + position.n_neighbors))
        
        return bounded
    
    def _load_descriptor(
        self,
        indicator_id: str,
        window_start: str,
        window_end: str,
        dimension: str
    ) -> Optional[float]:
        """Load a single descriptor value."""
        try:
            result = self.conn.execute("""
                SELECT value FROM derived.geometry_descriptors
                WHERE indicator_id = ? 
                  AND window_start = ?
                  AND window_end = ?
                  AND dimension = ?
                LIMIT 1
            """, [indicator_id, window_start, window_end, dimension]).fetchone()
            
            return float(result[0]) if result else None
        except:
            return None
    
    def _load_granger_degrees(
        self,
        indicator_id: str,
        window_start: str,
        window_end: str,
        p_threshold: float = 0.05
    ) -> Tuple[int, int]:
        """Load Granger causality in/out degrees."""
        try:
            # Out-degree (this indicator causes others)
            out_result = self.conn.execute("""
                SELECT COUNT(*) FROM derived.pairwise_descriptors
                WHERE indicator_id_1 = ?
                  AND window_start = ?
                  AND window_end = ?
                  AND dimension = 'granger_f_stat'
                  AND p_value < ?
            """, [indicator_id, window_start, window_end, p_threshold]).fetchone()
            
            # In-degree (others cause this indicator)
            in_result = self.conn.execute("""
                SELECT COUNT(*) FROM derived.pairwise_descriptors
                WHERE indicator_id_2 = ?
                  AND window_start = ?
                  AND window_end = ?
                  AND dimension = 'granger_f_stat'
                  AND p_value < ?
            """, [indicator_id, window_start, window_end, p_threshold]).fetchone()
            
            return (int(out_result[0]) if out_result else 0,
                    int(in_result[0]) if in_result else 0)
        except:
            return 0, 0


# =============================================================================
# PERSISTENCE
# =============================================================================

def persist_system_geometry(conn: duckdb.DuckDBPyConnection, geometry: SystemGeometry) -> bool:
    """Persist system geometry to database."""
    now = datetime.now()
    
    # Persist system-level geometry
    try:
        conn.execute("""
            INSERT INTO structure.system_geometry
            (window_start, window_end, n_indicators, effective_dimension,
             n_cohorts, mean_correlation, network_density, geometric_stability, computed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT DO UPDATE SET
                n_indicators = EXCLUDED.n_indicators,
                effective_dimension = EXCLUDED.effective_dimension,
                n_cohorts = EXCLUDED.n_cohorts,
                mean_correlation = EXCLUDED.mean_correlation,
                network_density = EXCLUDED.network_density,
                geometric_stability = EXCLUDED.geometric_stability,
                computed_at = EXCLUDED.computed_at
        """, [
            geometry.window_start, geometry.window_end,
            geometry.n_indicators, geometry.effective_dimension,
            geometry.n_cohorts, geometry.mean_correlation,
            geometry.network_density, geometry.geometric_stability, now
        ])
    except Exception as e:
        logger.error(f"Failed to persist system geometry: {e}")
        return False
    
    # Persist indicator positions
    for ind_id, bounded in geometry.indicator_geometries.items():
        try:
            conn.execute("""
                INSERT INTO structure.indicator_positions
                (indicator_id, window_start, window_end, centroid_distance,
                 pc1_loading, pc2_loading, pc3_loading, isolation_score,
                 degrees_of_freedom, structural_centrality, systemic_importance,
                 information_value, geometric_fit, anomaly_potential)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT DO UPDATE SET
                    centroid_distance = EXCLUDED.centroid_distance,
                    pc1_loading = EXCLUDED.pc1_loading,
                    pc2_loading = EXCLUDED.pc2_loading,
                    pc3_loading = EXCLUDED.pc3_loading,
                    isolation_score = EXCLUDED.isolation_score,
                    degrees_of_freedom = EXCLUDED.degrees_of_freedom,
                    structural_centrality = EXCLUDED.structural_centrality,
                    systemic_importance = EXCLUDED.systemic_importance,
                    information_value = EXCLUDED.information_value,
                    geometric_fit = EXCLUDED.geometric_fit,
                    anomaly_potential = EXCLUDED.anomaly_potential
            """, [
                ind_id, bounded.window_start, bounded.window_end,
                bounded.position.centroid_distance,
                bounded.position.pc1_loading,
                bounded.position.pc2_loading,
                bounded.position.pc3_loading,
                bounded.position.isolation_score,
                bounded.constraints.degrees_of_freedom,
                bounded.relevance.structural_centrality,
                bounded.relevance.systemic_importance,
                bounded.relevance.information_value,
                bounded.geometric_fit,
                bounded.anomaly_potential,
            ])
        except Exception as e:
            logger.debug(f"Failed to persist position for {ind_id}: {e}")
    
    return True


# =============================================================================
# CLI
# =============================================================================

def main():
    """CLI entry point for system geometry construction."""
    import argparse
    from prism.db.open import open_prism_db
    
    parser = argparse.ArgumentParser(description='PRISM System Geometry Constructor')
    parser.add_argument('--window-start', type=str, required=True, help='Window start date')
    parser.add_argument('--window-end', type=str, required=True, help='Window end date')
    parser.add_argument('--min-correlation', type=float, default=0.3, help='Min correlation for coupling')
    parser.add_argument('--indicator', type=str, help='Show detail for specific indicator')
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s'
    )
    
    conn = open_prism_db()
    constructor = SystemGeometryConstructor(conn)
    
    geometry = constructor.construct(
        args.window_start,
        args.window_end,
        min_correlation=args.min_correlation
    )
    
    if geometry is None:
        print("Failed to construct geometry")
        return
    
    print("\n" + "=" * 60)
    print(geometry.summary())
    print("=" * 60)
    
    if args.indicator:
        bounded = geometry.get_indicator(args.indicator)
        if bounded:
            print("\n" + bounded.summary())
        else:
            print(f"\nIndicator {args.indicator} not found in geometry")
    
    # Show top indicators by systemic importance
    print("\nTop 10 by Systemic Importance:")
    sorted_inds = sorted(
        geometry.indicator_geometries.values(),
        key=lambda x: x.relevance.systemic_importance,
        reverse=True
    )
    for i, bounded in enumerate(sorted_inds[:10]):
        print(f"  {i+1}. {bounded.indicator_id}: {bounded.relevance.systemic_importance:.3f}")
    
    # Persist
    persist_system_geometry(conn, geometry)
    print(f"\nGeometry persisted to structure.system_geometry")
    
    conn.close()


if __name__ == "__main__":
    main()
