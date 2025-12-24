"""
PRISM Physics-Based Behavioral Engine

Synthesizes two approaches:
1. Gemini's physics formulation: projection matrices, velocity/acceleration,
   kinetic energy, constraint tension (Euclidean baseline)
2. PRISM's innovations: Natural metric (Mahalanobis), 7-component state vector,
   emergent Resistance, Self-Influence as Lipschitz constant

The physics vocabulary comes from Gemini's structured approach.
The spectral amplification solution comes from Li et al. [2025].
The multi-lens assembly is PRISM's core contribution.

Hat tip: Gemini provided the clean projection-based formulation.
         Its Euclidean metric was upgraded to Natural metric per Li et al.

Usage:
    engine = PhysicsBehavioralEngine(geometry, state_covariance)
    state = engine.compute_state(indicator_id, current_sv, previous_sv)
    trajectory = engine.compute_trajectory(indicator_id, state_history)

Author: Jason (PRISM Project)
Date: December 2024
"""

import numpy as np
import scipy.linalg
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

class MetricType(Enum):
    """Distance metric for residual computation."""
    EUCLIDEAN = "euclidean"      # Gemini's original (spectral amplification risk)
    NATURAL = "natural"          # Li et al. 2025 (Mahalanobis)
    DIAGONAL = "diagonal"        # Variance-weighted (fast approximation)


@dataclass
class PhysicsConfig:
    """Configuration for the physics engine."""
    
    # Metric selection
    metric: MetricType = MetricType.NATURAL
    
    # Regularization for matrix inversion
    regularization: float = 1e-6
    
    # Minimum degrees of freedom (prevent division by zero)
    min_dof: float = 0.1
    
    # Resistance computation
    min_trajectory_length: int = 5
    resistance_decay: float = 0.9  # Exponential weighting for recent observations
    
    # Hidden mass thresholds
    hidden_mass_threshold: float = 2.0  # Standard deviations
    
    # Component names (the 7 lenses)
    component_names: List[str] = field(default_factory=lambda: [
        'geometry', 'memory', 'complexity', 'periodicity', 
        'tails', 'structure', 'resistance'
    ])


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class IndicatorState:
    """
    The dynamic state of an indicator within its geometric box.
    
    Combines Gemini's physics vocabulary with PRISM's multi-lens structure.
    
    Physics (from Gemini):
        - phase_position: Location in PC space
        - velocity: Rate of change in state space
        - acceleration: Second derivative (requires 3+ points)
        - kinetic_energy: Motion magnitude scaled by importance
        - constraint_tension: How hard indicator pushes against bounds
    
    Multi-Lens (from PRISM):
        - state_vector: The 7-component [G, M, C, P, T, S, R] representation
        - component_contributions: Which lens contributed what
    
    Hidden Mass (synthesis):
        - residual_motion: Magnitude of unexplained motion (Natural metric)
        - hidden_mass: Scaled residual indicating unmeasured forces
        - hidden_mass_direction: Vector direction of violation
    
    Stability (from PRISM + Li et al.):
        - resistance: Emergent stability from trajectory consistency
        - self_influence: Lipschitz constant (stability bound)
    """
    
    # Identity
    indicator_id: str
    window_id: str
    timestamp: datetime
    
    # === PHYSICS (Gemini formulation) ===
    phase_position: np.ndarray          # Position in PC space
    velocity: np.ndarray                # First derivative of position
    acceleration: np.ndarray            # Second derivative of position
    kinetic_energy: float               # 0.5 * m * v^2 (scaled)
    constraint_tension: float           # Residual / expected_volatility
    
    # === MULTI-LENS (PRISM) ===
    state_vector: np.ndarray            # [G, M, C, P, T, S, R]
    component_contributions: Dict[str, float]  # Per-component magnitude
    
    # === HIDDEN MASS (Synthesis) ===
    residual_motion: float              # ||v_obs - v_expected||_Σ (Natural!)
    residual_motion_euclidean: float    # ||v_obs - v_expected||_2 (for comparison)
    hidden_mass: float                  # Scaled by importance/DOF
    hidden_mass_direction: np.ndarray   # Unit vector of violation
    hidden_mass_significant: bool       # Above threshold?
    
    # === STABILITY (PRISM + Li et al.) ===
    resistance: float                   # Emergent from trajectory
    self_influence: float               # Lipschitz constant = S'Σ⁻¹S
    stability_class: str                # 'rigid', 'stable', 'fluid', 'volatile'
    
    # === DIAGNOSTICS ===
    metric_used: str                    # Which metric computed residual
    spectral_amplification_ratio: float # Euclidean/Natural (should be >> 1)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize for storage."""
        return {
            'indicator_id': self.indicator_id,
            'window_id': self.window_id,
            'timestamp': self.timestamp.isoformat(),
            'phase_position': self.phase_position.tolist(),
            'velocity': self.velocity.tolist(),
            'acceleration': self.acceleration.tolist(),
            'kinetic_energy': self.kinetic_energy,
            'constraint_tension': self.constraint_tension,
            'state_vector': self.state_vector.tolist(),
            'component_contributions': self.component_contributions,
            'residual_motion': self.residual_motion,
            'residual_motion_euclidean': self.residual_motion_euclidean,
            'hidden_mass': self.hidden_mass,
            'hidden_mass_direction': self.hidden_mass_direction.tolist(),
            'hidden_mass_significant': self.hidden_mass_significant,
            'resistance': self.resistance,
            'self_influence': self.self_influence,
            'stability_class': self.stability_class,
            'metric_used': self.metric_used,
            'spectral_amplification_ratio': self.spectral_amplification_ratio,
        }


@dataclass
class TrajectoryAnalysis:
    """
    Analysis of an indicator's trajectory through state space over time.
    
    Used to compute emergent Resistance and detect regime transitions.
    """
    indicator_id: str
    window_id: str
    
    # Trajectory statistics
    n_observations: int
    time_span_days: float
    
    # Motion statistics
    mean_velocity: np.ndarray
    velocity_variance: np.ndarray
    mean_acceleration: np.ndarray
    
    # Stability metrics
    trajectory_smoothness: float        # Low = jerky, High = smooth
    direction_consistency: float        # Cosine similarity of velocity vectors
    resistance: float                   # Emergent stability measure
    
    # Hidden mass accumulation
    total_hidden_mass: float
    hidden_mass_events: int             # Count of significant events
    hidden_mass_persistence: float      # How long does hidden mass last?
    
    # Regime indicators
    regime_transitions: int             # Detected direction changes
    current_regime_duration: int        # Steps since last transition
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize for storage."""
        return {
            'indicator_id': self.indicator_id,
            'window_id': self.window_id,
            'n_observations': self.n_observations,
            'time_span_days': self.time_span_days,
            'mean_velocity': self.mean_velocity.tolist(),
            'velocity_variance': self.velocity_variance.tolist(),
            'mean_acceleration': self.mean_acceleration.tolist(),
            'trajectory_smoothness': self.trajectory_smoothness,
            'direction_consistency': self.direction_consistency,
            'resistance': self.resistance,
            'total_hidden_mass': self.total_hidden_mass,
            'hidden_mass_events': self.hidden_mass_events,
            'hidden_mass_persistence': self.hidden_mass_persistence,
            'regime_transitions': self.regime_transitions,
            'current_regime_duration': self.current_regime_duration,
        }


# =============================================================================
# BOUNDED GEOMETRY INTERFACE
# =============================================================================

@dataclass
class BoundedGeometry:
    """
    Minimal interface to SystemGeometry for an individual indicator.
    
    This is what PhysicsBehavioralEngine needs from the geometry layer.
    """
    indicator_id: str
    
    # Position in system
    systemic_importance: float          # 0-1, how central
    degrees_of_freedom: float           # Effective DOF
    expected_volatility: float          # Historical vol for normalization
    
    # For constraint checking
    soft_ceiling: Optional[float] = None
    soft_floor: Optional[float] = None


@dataclass 
class SystemGeometryInterface:
    """
    Minimal interface to SystemGeometry for the physics engine.
    """
    window_id: str
    n_indicators: int
    effective_dimension: int
    
    # Principal structure
    principal_axes: np.ndarray          # Eigenvectors (columns)
    eigenvalues: np.ndarray             # Corresponding eigenvalues
    
    # Per-indicator bounds
    indicator_bounds: Dict[str, BoundedGeometry]
    
    def get_indicator(self, indicator_id: str) -> BoundedGeometry:
        """Get bounded geometry for indicator."""
        return self.indicator_bounds.get(indicator_id)


# =============================================================================
# PHYSICS-BASED BEHAVIORAL ENGINE
# =============================================================================

class PhysicsBehavioralEngine:
    """
    Computes State Vectors and Hidden Mass using physics formulation.
    
    Core insight from Gemini: Motion in state space can be decomposed into
    "expected" (projection onto system manifold) and "residual" (violation).
    The residual is Hidden Mass.
    
    Core insight from Li et al. [2025]: Euclidean metric suffers spectral
    amplification. Must use Natural metric (Mahalanobis) induced by the
    state covariance to get meaningful residual magnitudes.
    
    Core insight from PRISM: The state vector itself is a multi-lens
    assembly from heterogeneous equations. Hidden mass in THIS space
    means something different than hidden mass in raw observation space.
    """
    
    def __init__(
        self,
        geometry: SystemGeometryInterface,
        state_covariance: np.ndarray,
        config: Optional[PhysicsConfig] = None
    ):
        """
        Initialize the physics engine.
        
        Args:
            geometry: System geometry with principal axes and indicator bounds
            state_covariance: Covariance matrix of state vectors across population
                             Shape: (n_components, n_components) typically (7, 7)
            config: Engine configuration
        """
        self.geometry = geometry
        self.config = config or PhysicsConfig()
        
        # === GEMINI'S PROJECTION APPROACH ===
        # Project onto the system manifold (first k principal components)
        k = int(geometry.effective_dimension)
        self.manifold_basis = geometry.principal_axes[:, :k]
        self.projection_matrix = self.manifold_basis @ self.manifold_basis.T
        
        logger.info(f"Manifold basis: {self.manifold_basis.shape}, "
                   f"effective dimension: {k}")
        
        # === PRISM'S NATURAL METRIC (Li et al. 2025) ===
        self.state_covariance = state_covariance
        self._setup_natural_metric(state_covariance)
        
        # Trajectory history for resistance computation
        self._trajectory_cache: Dict[str, List[IndicatorState]] = {}
    
    def _setup_natural_metric(self, Sigma: np.ndarray):
        """
        Set up the Natural metric components.
        
        The Natural metric measures distance as:
            d_nat(x, y) = sqrt((x-y)' Σ⁻¹ (x-y))
        
        This is Mahalanobis distance, which:
        1. Whitens the space (all directions contribute equally)
        2. Eliminates spectral amplification
        3. Makes Self-Influence = Lipschitz constant
        """
        n = Sigma.shape[0]
        reg = self.config.regularization
        
        # Regularized inverse
        Sigma_reg = Sigma + reg * np.eye(n)
        
        try:
            self.Sigma_inv = np.linalg.inv(Sigma_reg)
            self.Sigma_inv_sqrt = scipy.linalg.sqrtm(self.Sigma_inv).real
            
            # Condition number (for diagnostics)
            eigvals = np.linalg.eigvalsh(Sigma_reg)
            self.condition_number = eigvals.max() / max(eigvals.min(), 1e-10)
            
            logger.info(f"Natural metric initialized. Condition number: {self.condition_number:.2e}")
            
        except np.linalg.LinAlgError as e:
            logger.error(f"Failed to invert covariance: {e}")
            # Fall back to diagonal
            self.Sigma_inv = np.diag(1.0 / (np.diag(Sigma) + reg))
            self.Sigma_inv_sqrt = np.sqrt(self.Sigma_inv)
            self.condition_number = np.nan
        
        # Diagonal variances for fast approximation
        self.variances = np.diag(Sigma)
    
    def _compute_distance(
        self, 
        v: np.ndarray, 
        metric: Optional[MetricType] = None
    ) -> Tuple[float, float]:
        """
        Compute vector magnitude under specified metric.
        
        Returns:
            (natural_distance, euclidean_distance)
        """
        metric = metric or self.config.metric
        
        # Always compute Euclidean for comparison
        d_euclidean = np.linalg.norm(v)
        
        if metric == MetricType.EUCLIDEAN:
            d_natural = d_euclidean
            
        elif metric == MetricType.NATURAL:
            # Mahalanobis: sqrt(v' Σ⁻¹ v)
            d_natural = np.sqrt(max(0, v.T @ self.Sigma_inv @ v))
            
        elif metric == MetricType.DIAGONAL:
            # Fast approximation: variance-weighted
            d_natural = np.sqrt(np.sum(v**2 / (self.variances + 1e-10)))
            
        else:
            d_natural = d_euclidean
        
        return d_natural, d_euclidean
    
    def _classify_stability(self, resistance: float, self_influence: float) -> str:
        """
        Classify stability based on resistance and self-influence.
        
        High resistance + low self-influence = rigid (stable, central)
        High resistance + high self-influence = stable (stable, peripheral)
        Low resistance + low self-influence = fluid (unstable, central)
        Low resistance + high self-influence = volatile (unstable, peripheral)
        """
        r_thresh = 0.5
        si_thresh = np.median([b.systemic_importance 
                              for b in self.geometry.indicator_bounds.values()])
        
        if resistance > r_thresh:
            return 'rigid' if self_influence < si_thresh else 'stable'
        else:
            return 'fluid' if self_influence < si_thresh else 'volatile'
    
    def _decompose_contributions(
        self, 
        state_vector: np.ndarray
    ) -> Dict[str, float]:
        """
        Decompose state vector into per-component contributions.
        
        Uses Natural metric to properly weight components.
        """
        names = self.config.component_names
        n = min(len(names), len(state_vector))
        
        contributions = {}
        total = 0.0
        
        for i in range(n):
            # Contribution in Natural metric
            if self.config.metric == MetricType.NATURAL:
                contrib = abs(state_vector[i]) * np.sqrt(self.Sigma_inv[i, i])
            else:
                contrib = abs(state_vector[i])
            contributions[names[i]] = contrib
            total += contrib
        
        # Normalize
        if total > 0:
            contributions = {k: v/total for k, v in contributions.items()}
        
        return contributions
    
    def compute_state(
        self,
        indicator_id: str,
        current_state: np.ndarray,
        previous_state: np.ndarray,
        dt: float = 1.0,
        previous_velocity: Optional[np.ndarray] = None
    ) -> IndicatorState:
        """
        Compute the dynamic state of an indicator.
        
        This is Gemini's physics formulation upgraded with PRISM's Natural metric.
        
        Args:
            indicator_id: Indicator identifier
            current_state: Current state vector [G, M, C, P, T, S, R]
            previous_state: Previous state vector
            dt: Time delta (default 1.0 = one step)
            previous_velocity: Previous velocity (for acceleration)
        
        Returns:
            IndicatorState with physics + multi-lens + hidden mass
        """
        # Get bounded geometry for this indicator
        bounded = self.geometry.get_indicator(indicator_id)
        if bounded is None:
            logger.warning(f"No geometry for {indicator_id}, using defaults")
            bounded = BoundedGeometry(
                indicator_id=indicator_id,
                systemic_importance=0.5,
                degrees_of_freedom=3.0,
                expected_volatility=1.0
            )
        
        # === GEMINI'S PHYSICS ===
        
        # Observed velocity in state space
        v_obs = (current_state - previous_state) / dt
        
        # Project onto system manifold (expected motion)
        v_expected = self.projection_matrix @ v_obs
        
        # Residual = forbidden motion = hidden mass signal
        v_residual = v_obs - v_expected
        
        # === PRISM'S NATURAL METRIC ===
        
        # Compute residual magnitude in Natural metric (not Euclidean!)
        residual_natural, residual_euclidean = self._compute_distance(v_residual)
        
        # Spectral amplification ratio (should be >> 1 if metric matters)
        amplification_ratio = (residual_euclidean / max(residual_natural, 1e-10) 
                              if residual_natural > 0 else 1.0)
        
        # === HIDDEN MASS COMPUTATION ===
        
        # Mass coefficient: important indicators with few DOF generate more mass
        mass_coefficient = (bounded.systemic_importance / 
                          max(self.config.min_dof, bounded.degrees_of_freedom))
        
        hidden_mass = residual_natural * mass_coefficient
        
        # Direction of violation (unit vector)
        if residual_natural > 1e-10:
            hidden_mass_direction = v_residual / residual_natural
        else:
            hidden_mass_direction = np.zeros_like(v_residual)
        
        # Significance threshold
        hidden_mass_significant = residual_natural > self.config.hidden_mass_threshold
        
        # === SELF-INFLUENCE (Lipschitz constant) ===
        
        # Self-influence = state' Σ⁻¹ state (squared norm in whitened space)
        self_influence = current_state.T @ self.Sigma_inv @ current_state
        
        # === PHYSICS DERIVED QUANTITIES ===
        
        # Phase position (projection onto PC space)
        phase_position = self.manifold_basis.T @ current_state
        
        # Velocity in PC space
        velocity_pc = self.manifold_basis.T @ v_obs
        
        # Acceleration (requires previous velocity)
        if previous_velocity is not None:
            acceleration = (v_obs - previous_velocity) / dt
            acceleration_pc = self.manifold_basis.T @ acceleration
        else:
            acceleration_pc = np.zeros_like(phase_position)
        
        # Kinetic energy (in Natural metric)
        v_natural, _ = self._compute_distance(v_obs)
        kinetic_energy = 0.5 * mass_coefficient * (v_natural ** 2)
        
        # Constraint tension
        constraint_tension = residual_natural / max(bounded.expected_volatility, 1e-10)
        
        # === RESISTANCE (from trajectory history) ===
        
        resistance = self._compute_resistance(indicator_id)
        
        # Stability classification
        stability_class = self._classify_stability(resistance, self_influence)
        
        # === COMPONENT CONTRIBUTIONS ===
        
        contributions = self._decompose_contributions(current_state)
        
        # === BUILD STATE OBJECT ===
        
        state = IndicatorState(
            indicator_id=indicator_id,
            window_id=self.geometry.window_id,
            timestamp=datetime.now(),
            
            # Physics
            phase_position=phase_position,
            velocity=velocity_pc,
            acceleration=acceleration_pc,
            kinetic_energy=kinetic_energy,
            constraint_tension=constraint_tension,
            
            # Multi-lens
            state_vector=current_state.copy(),
            component_contributions=contributions,
            
            # Hidden mass
            residual_motion=residual_natural,
            residual_motion_euclidean=residual_euclidean,
            hidden_mass=hidden_mass,
            hidden_mass_direction=hidden_mass_direction,
            hidden_mass_significant=hidden_mass_significant,
            
            # Stability
            resistance=resistance,
            self_influence=self_influence,
            stability_class=stability_class,
            
            # Diagnostics
            metric_used=self.config.metric.value,
            spectral_amplification_ratio=amplification_ratio,
        )
        
        # Cache for trajectory analysis
        self._cache_state(indicator_id, state)
        
        return state
    
    def _cache_state(self, indicator_id: str, state: IndicatorState):
        """Cache state for trajectory analysis."""
        if indicator_id not in self._trajectory_cache:
            self._trajectory_cache[indicator_id] = []
        
        cache = self._trajectory_cache[indicator_id]
        cache.append(state)
        
        # Keep only recent history
        max_history = 100
        if len(cache) > max_history:
            self._trajectory_cache[indicator_id] = cache[-max_history:]
    
    def _compute_resistance(self, indicator_id: str) -> float:
        """
        Compute emergent Resistance from trajectory history.
        
        Resistance measures how stable the indicator's trajectory is.
        High resistance = trajectory is predictable, changes slowly.
        Low resistance = trajectory is erratic, changes rapidly.
        
        This is PRISM's key emergent property: resistance cannot be
        measured directly from the state vector, only from its evolution.
        """
        cache = self._trajectory_cache.get(indicator_id, [])
        
        if len(cache) < self.config.min_trajectory_length:
            return 0.5  # Default for insufficient data
        
        # Get recent velocities
        velocities = [s.velocity for s in cache[-20:] if s.velocity is not None]
        
        if len(velocities) < 3:
            return 0.5
        
        velocities = np.array(velocities)
        
        # Method 1: Direction consistency (cosine similarity of consecutive velocities)
        direction_scores = []
        for i in range(1, len(velocities)):
            v1, v2 = velocities[i-1], velocities[i]
            norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
            if norm1 > 1e-10 and norm2 > 1e-10:
                cosine = np.dot(v1, v2) / (norm1 * norm2)
                direction_scores.append((cosine + 1) / 2)  # Map [-1,1] to [0,1]
        
        direction_consistency = np.mean(direction_scores) if direction_scores else 0.5
        
        # Method 2: Magnitude consistency (coefficient of variation)
        magnitudes = np.linalg.norm(velocities, axis=1)
        if magnitudes.mean() > 1e-10:
            cv = magnitudes.std() / magnitudes.mean()
            magnitude_consistency = 1.0 / (1.0 + cv)  # High CV = low consistency
        else:
            magnitude_consistency = 0.5
        
        # Method 3: Acceleration smoothness
        hidden_masses = [s.hidden_mass for s in cache[-20:]]
        mass_variance = np.var(hidden_masses) if len(hidden_masses) > 1 else 0
        smoothness = 1.0 / (1.0 + mass_variance)
        
        # Combine (weighted average)
        resistance = (
            0.4 * direction_consistency +
            0.3 * magnitude_consistency +
            0.3 * smoothness
        )
        
        return float(np.clip(resistance, 0, 1))
    
    def compute_trajectory(
        self,
        indicator_id: str,
        states: Optional[List[IndicatorState]] = None
    ) -> TrajectoryAnalysis:
        """
        Analyze full trajectory for an indicator.
        
        Args:
            indicator_id: Indicator identifier
            states: List of states (or use cached)
        
        Returns:
            TrajectoryAnalysis with aggregate statistics
        """
        if states is None:
            states = self._trajectory_cache.get(indicator_id, [])
        
        if len(states) < 2:
            logger.warning(f"Insufficient trajectory data for {indicator_id}")
            return self._empty_trajectory(indicator_id)
        
        # Extract arrays
        velocities = np.array([s.velocity for s in states])
        accelerations = np.array([s.acceleration for s in states])
        hidden_masses = np.array([s.hidden_mass for s in states])
        
        # Time span
        t0, t1 = states[0].timestamp, states[-1].timestamp
        time_span = (t1 - t0).total_seconds() / 86400  # Days
        
        # Motion statistics
        mean_velocity = velocities.mean(axis=0)
        velocity_variance = velocities.var(axis=0)
        mean_acceleration = accelerations.mean(axis=0)
        
        # Trajectory smoothness (inverse of acceleration variance)
        accel_var = np.mean(np.var(accelerations, axis=0))
        smoothness = 1.0 / (1.0 + accel_var)
        
        # Direction consistency
        direction_scores = []
        for i in range(1, len(velocities)):
            v1, v2 = velocities[i-1], velocities[i]
            norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
            if norm1 > 1e-10 and norm2 > 1e-10:
                cosine = np.dot(v1, v2) / (norm1 * norm2)
                direction_scores.append(cosine)
        
        direction_consistency = np.mean(direction_scores) if direction_scores else 0.0
        
        # Hidden mass analysis
        total_hidden_mass = hidden_masses.sum()
        hidden_mass_events = sum(1 for s in states if s.hidden_mass_significant)
        
        # Persistence: autocorrelation of hidden mass
        if len(hidden_masses) > 3:
            hm_centered = hidden_masses - hidden_masses.mean()
            autocorr = np.correlate(hm_centered, hm_centered, mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            if autocorr[0] > 0:
                persistence = autocorr[1] / autocorr[0] if len(autocorr) > 1 else 0
            else:
                persistence = 0
        else:
            persistence = 0
        
        # Regime transitions (sign changes in PC1 velocity)
        pc1_velocity = velocities[:, 0] if velocities.shape[1] > 0 else velocities.flatten()
        transitions = np.sum(np.diff(np.sign(pc1_velocity)) != 0)
        
        # Current regime duration
        if len(pc1_velocity) > 1:
            signs = np.sign(pc1_velocity)
            current_sign = signs[-1]
            duration = 1
            for i in range(len(signs) - 2, -1, -1):
                if signs[i] == current_sign:
                    duration += 1
                else:
                    break
        else:
            duration = len(states)
        
        # Resistance (from last state or compute)
        resistance = states[-1].resistance if states else 0.5
        
        return TrajectoryAnalysis(
            indicator_id=indicator_id,
            window_id=self.geometry.window_id,
            n_observations=len(states),
            time_span_days=time_span,
            mean_velocity=mean_velocity,
            velocity_variance=velocity_variance,
            mean_acceleration=mean_acceleration,
            trajectory_smoothness=smoothness,
            direction_consistency=direction_consistency,
            resistance=resistance,
            total_hidden_mass=total_hidden_mass,
            hidden_mass_events=hidden_mass_events,
            hidden_mass_persistence=persistence,
            regime_transitions=transitions,
            current_regime_duration=duration,
        )
    
    def _empty_trajectory(self, indicator_id: str) -> TrajectoryAnalysis:
        """Return empty trajectory analysis."""
        return TrajectoryAnalysis(
            indicator_id=indicator_id,
            window_id=self.geometry.window_id,
            n_observations=0,
            time_span_days=0,
            mean_velocity=np.array([]),
            velocity_variance=np.array([]),
            mean_acceleration=np.array([]),
            trajectory_smoothness=0,
            direction_consistency=0,
            resistance=0.5,
            total_hidden_mass=0,
            hidden_mass_events=0,
            hidden_mass_persistence=0,
            regime_transitions=0,
            current_regime_duration=0,
        )
    
    def detect_hidden_mass_events(
        self,
        threshold_sigma: float = 2.0
    ) -> List[Dict[str, Any]]:
        """
        Detect significant hidden mass events across all cached trajectories.
        
        Returns list of events with indicator, timestamp, magnitude, direction.
        """
        events = []
        
        for indicator_id, states in self._trajectory_cache.items():
            # Compute population statistics for this indicator
            masses = [s.hidden_mass for s in states]
            if len(masses) < 5:
                continue
            
            mean_mass = np.mean(masses)
            std_mass = np.std(masses)
            
            if std_mass < 1e-10:
                continue
            
            # Find outliers
            for state in states:
                z_score = (state.hidden_mass - mean_mass) / std_mass
                if z_score > threshold_sigma:
                    events.append({
                        'indicator_id': indicator_id,
                        'timestamp': state.timestamp,
                        'hidden_mass': state.hidden_mass,
                        'z_score': z_score,
                        'direction': state.hidden_mass_direction.tolist(),
                        'constraint_tension': state.constraint_tension,
                        'stability_class': state.stability_class,
                    })
        
        # Sort by z-score
        events.sort(key=lambda x: x['z_score'], reverse=True)
        
        return events
    
    def get_system_deformation(self) -> Dict[str, Any]:
        """
        Compute system-wide deformation field from all trajectories.
        
        Returns aggregate statistics about hidden mass across the system.
        """
        all_states = []
        for states in self._trajectory_cache.values():
            all_states.extend(states)
        
        if not all_states:
            return {'error': 'No cached states'}
        
        # Aggregate hidden mass
        masses = [s.hidden_mass for s in all_states]
        
        # By stability class
        by_class = {}
        for s in all_states:
            cls = s.stability_class
            if cls not in by_class:
                by_class[cls] = []
            by_class[cls].append(s.hidden_mass)
        
        class_stats = {
            cls: {'mean': np.mean(v), 'std': np.std(v), 'n': len(v)}
            for cls, v in by_class.items()
        }
        
        # Spectral amplification across system
        ratios = [s.spectral_amplification_ratio for s in all_states 
                 if np.isfinite(s.spectral_amplification_ratio)]
        
        return {
            'n_states': len(all_states),
            'n_indicators': len(self._trajectory_cache),
            'total_hidden_mass': sum(masses),
            'mean_hidden_mass': np.mean(masses),
            'std_hidden_mass': np.std(masses),
            'by_stability_class': class_stats,
            'mean_spectral_amplification': np.mean(ratios) if ratios else 1.0,
            'metric_used': self.config.metric.value,
            'condition_number': self.condition_number,
        }


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_engine_from_data(
    principal_axes: np.ndarray,
    eigenvalues: np.ndarray,
    state_vectors: np.ndarray,
    indicator_metadata: Dict[str, Dict],
    window_id: str = "default",
    metric: MetricType = MetricType.NATURAL
) -> PhysicsBehavioralEngine:
    """
    Create a PhysicsBehavioralEngine from raw data.
    
    Args:
        principal_axes: Eigenvectors from PCA, shape (n_components, n_components)
        eigenvalues: Eigenvalues from PCA
        state_vectors: Matrix of state vectors, shape (n_indicators, n_components)
        indicator_metadata: Dict mapping indicator_id to {importance, dof, volatility}
        window_id: Window identifier
        metric: Distance metric to use
    
    Returns:
        Configured PhysicsBehavioralEngine
    """
    # Compute effective dimension (explain 95% variance)
    total_var = eigenvalues.sum()
    cumvar = np.cumsum(eigenvalues) / total_var
    effective_dim = np.searchsorted(cumvar, 0.95) + 1
    
    # Build indicator bounds
    indicator_bounds = {}
    for ind_id, meta in indicator_metadata.items():
        indicator_bounds[ind_id] = BoundedGeometry(
            indicator_id=ind_id,
            systemic_importance=meta.get('importance', 0.5),
            degrees_of_freedom=meta.get('dof', 3.0),
            expected_volatility=meta.get('volatility', 1.0),
        )
    
    # Build geometry interface
    geometry = SystemGeometryInterface(
        window_id=window_id,
        n_indicators=len(indicator_metadata),
        effective_dimension=effective_dim,
        principal_axes=principal_axes,
        eigenvalues=eigenvalues,
        indicator_bounds=indicator_bounds,
    )
    
    # Compute state covariance
    state_covariance = np.cov(state_vectors.T)
    if state_covariance.ndim == 0:
        state_covariance = np.array([[state_covariance]])
    
    # Config
    config = PhysicsConfig(metric=metric)
    
    return PhysicsBehavioralEngine(geometry, state_covariance, config)


# =============================================================================
# MAIN (Example Usage)
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="PRISM Physics-Based Behavioral Engine",
        epilog="""
Example:
    python prism_physics_engine.py --demo
    
    # In code:
    engine = PhysicsBehavioralEngine(geometry, covariance)
    state = engine.compute_state('SPY', current_sv, previous_sv)
    print(f"Hidden mass: {state.hidden_mass:.4f}")
    print(f"Stability: {state.stability_class}")
        """
    )
    
    parser.add_argument('--demo', action='store_true', help='Run demonstration')
    parser.add_argument('--metric', choices=['natural', 'euclidean', 'diagonal'],
                       default='natural', help='Distance metric')
    
    args = parser.parse_args()
    
    if args.demo:
        print("=" * 60)
        print("PRISM Physics-Based Behavioral Engine - Demo")
        print("=" * 60)
        print()
        print("Hat tip: Gemini for the physics formulation")
        print("         Li et al. [2025] for the Natural metric")
        print()
        
        # Create synthetic data
        np.random.seed(42)
        n_indicators = 10
        n_components = 7  # G, M, C, P, T, S, R
        
        # Synthetic principal structure
        principal_axes = np.eye(n_components)
        eigenvalues = np.array([3.0, 2.0, 1.5, 1.0, 0.5, 0.3, 0.1])
        
        # Synthetic state vectors
        state_vectors = np.random.randn(n_indicators, n_components)
        
        # Synthetic metadata
        indicator_ids = [f"IND_{i}" for i in range(n_indicators)]
        metadata = {
            ind_id: {
                'importance': np.random.uniform(0.3, 1.0),
                'dof': np.random.uniform(1.0, 5.0),
                'volatility': np.random.uniform(0.5, 2.0),
            }
            for ind_id in indicator_ids
        }
        
        # Create engine
        metric = MetricType(args.metric)
        engine = create_engine_from_data(
            principal_axes=principal_axes,
            eigenvalues=eigenvalues,
            state_vectors=state_vectors,
            indicator_metadata=metadata,
            metric=metric
        )
        
        print(f"Engine created with {metric.value} metric")
        print(f"Condition number: {engine.condition_number:.2e}")
        print()
        
        # Compute states for a few indicators
        print("Computing states...")
        print("-" * 60)
        
        for i, ind_id in enumerate(indicator_ids[:3]):
            current_sv = state_vectors[i]
            previous_sv = current_sv + np.random.randn(n_components) * 0.1
            
            state = engine.compute_state(ind_id, current_sv, previous_sv)
            
            print(f"\n{ind_id}:")
            print(f"  Hidden Mass: {state.hidden_mass:.4f}")
            print(f"  Residual (Natural): {state.residual_motion:.4f}")
            print(f"  Residual (Euclidean): {state.residual_motion_euclidean:.4f}")
            print(f"  Spectral Amplification: {state.spectral_amplification_ratio:.2f}x")
            print(f"  Self-Influence: {state.self_influence:.4f}")
            print(f"  Stability: {state.stability_class}")
            print(f"  Significant: {state.hidden_mass_significant}")
        
        print()
        print("=" * 60)
        print("Demo complete.")
        print()
        print("Key insight: Compare 'Spectral Amplification' values.")
        print("Values >> 1 mean Natural metric is essential.")
        print("Values ≈ 1 mean Euclidean would have been fine.")
