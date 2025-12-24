"""
PRISM Phase 4: Interpretation Layer

Takes individual IndicatorStates from the Physics Engine and answers:
"What does this mean for the system as a whole?"

Key outputs:
- RegimeSignal: System-wide tension, entropy, rupture risk
- CohortStress: Which groups of indicators are under stress
- DeformationField: Where is the geometry being pushed

Hat tip: Gemini provided the RegimeMonitor structure with systemic tension,
         geometric entropy, and rupture risk formulation.

Phase Structure:
    P1 Data → P2 Derived → P3 Structure → P4 Interpretation → P5 Prediction
                                              ↑ YOU ARE HERE

Usage:
    from prism_interpretation import RegimeMonitor, InterpretationEngine
    
    monitor = RegimeMonitor(geometry)
    signal = monitor.evaluate_system(indicator_states)
    print(f"Status: {signal.status}, Rupture Risk: {signal.rupture_risk:.2%}")

Author: Jason (PRISM Project)
Date: December 2024
"""

import numpy as np
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Set
from enum import Enum
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS AND CONFIGURATION
# =============================================================================

class SystemStatus(Enum):
    """System-wide regime status."""
    STABLE = "STABLE"           # Normal operations
    STRESSED = "STRESSED"       # Elevated tension, monitoring advised
    CRITICAL = "CRITICAL"       # High rupture probability
    RUPTURE = "RUPTURE"         # Active regime transition
    RECOVERY = "RECOVERY"       # Post-rupture stabilization


class TensionSource(Enum):
    """Classification of tension sources."""
    INTERNAL = "internal"       # Tension from indicator relationships
    EXTERNAL = "external"       # Tension from hidden mass (unmeasured forces)
    STRUCTURAL = "structural"   # Tension from geometry instability
    CASCADING = "cascading"     # Tension propagating through cohorts


@dataclass
class InterpretationConfig:
    """Configuration for interpretation layer."""
    
    # Rupture thresholds
    rupture_threshold: float = 0.75
    critical_threshold: float = 0.6
    stressed_threshold: float = 0.3
    
    # Hidden mass significance
    hidden_mass_sigma: float = 2.0  # Standard deviations for "critical"
    
    # Entropy normalization
    max_entropy_indicators: int = 100  # For normalizing entropy
    
    # History length
    signal_history_length: int = 50
    
    # Cohort stress
    cohort_stress_threshold: float = 0.5


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class RegimeSignal:
    """
    System-wide regime signal at a point in time.
    
    This is Gemini's formulation with PRISM extensions.
    
    Core (Gemini):
        - systemic_tension: Aggregate hidden mass across system
        - geometric_entropy: Disorder in energy distribution
        - rupture_risk: Probability of regime transition
        - status: STABLE / STRESSED / CRITICAL / RUPTURE
    
    Extensions (PRISM):
        - tension_source: Where is tension coming from?
        - momentum: Is tension building or releasing?
        - critical_cohorts: Which indicator groups are stressed?
    """
    timestamp: datetime
    window_id: str
    
    # === GEMINI'S CORE ===
    systemic_tension: float             # Sum of hidden mass / n_indicators
    geometric_entropy: float            # Energy distribution disorder
    critical_indicators: List[str]      # Who's driving the tension
    rupture_risk: float                 # 0 to 1 probability
    status: SystemStatus                # Classified status
    
    # === PRISM EXTENSIONS ===
    tension_source: TensionSource       # Where is it coming from?
    tension_momentum: float             # Positive = building, Negative = releasing
    tension_acceleration: float         # Rate of change of momentum
    
    critical_cohorts: List[str]         # Which cohorts are stressed
    cohort_tension: Dict[str, float]    # Tension by cohort
    
    # Geometry health
    geometry_condition: float           # Condition number trend
    effective_dimension_change: float   # Is structure collapsing/expanding?
    
    # Hidden mass concentration
    hidden_mass_gini: float            # Inequality of hidden mass distribution
    hidden_mass_top_5: List[str]       # Top 5 contributors
    
    # Recovery indicators (if in RECOVERY status)
    recovery_progress: float           # 0 to 1
    stabilizing_indicators: List[str]  # Who's helping stabilize
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize for storage."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'window_id': self.window_id,
            'systemic_tension': self.systemic_tension,
            'geometric_entropy': self.geometric_entropy,
            'critical_indicators': self.critical_indicators,
            'rupture_risk': self.rupture_risk,
            'status': self.status.value,
            'tension_source': self.tension_source.value,
            'tension_momentum': self.tension_momentum,
            'tension_acceleration': self.tension_acceleration,
            'critical_cohorts': self.critical_cohorts,
            'cohort_tension': self.cohort_tension,
            'geometry_condition': self.geometry_condition,
            'effective_dimension_change': self.effective_dimension_change,
            'hidden_mass_gini': self.hidden_mass_gini,
            'hidden_mass_top_5': self.hidden_mass_top_5,
            'recovery_progress': self.recovery_progress,
            'stabilizing_indicators': self.stabilizing_indicators,
        }
    
    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            f"=== Regime Signal: {self.status.value} ===",
            f"Timestamp: {self.timestamp}",
            f"Rupture Risk: {self.rupture_risk:.1%}",
            f"Systemic Tension: {self.systemic_tension:.4f}",
            f"Geometric Entropy: {self.geometric_entropy:.4f}",
            f"Tension Source: {self.tension_source.value}",
            f"Momentum: {self.tension_momentum:+.4f} ({'building' if self.tension_momentum > 0 else 'releasing'})",
        ]
        
        if self.critical_indicators:
            lines.append(f"Critical Indicators: {', '.join(self.critical_indicators[:5])}")
        
        if self.critical_cohorts:
            lines.append(f"Critical Cohorts: {', '.join(self.critical_cohorts)}")
        
        return '\n'.join(lines)


@dataclass
class CohortStress:
    """Stress analysis for a cohort of indicators."""
    cohort_id: str
    cohort_name: str
    n_indicators: int
    
    # Stress metrics
    mean_hidden_mass: float
    max_hidden_mass: float
    tension_concentration: float    # How concentrated is stress?
    
    # Dynamics
    stress_momentum: float          # Building or releasing?
    coherence: float               # Are indicators moving together?
    
    # Key members
    most_stressed: List[str]
    most_stable: List[str]
    
    # Status
    is_critical: bool
    stress_rank: int               # 1 = most stressed cohort


@dataclass
class DeformationField:
    """
    System-wide deformation field showing where geometry is being pushed.
    
    Think of this as a "stress map" of the system.
    """
    timestamp: datetime
    window_id: str
    
    # Overall deformation
    total_deformation: float
    deformation_direction: np.ndarray   # Principal direction of deformation
    
    # By PC axis
    pc_deformations: Dict[int, float]   # Deformation along each PC
    dominant_pc: int                     # Which PC is most deformed?
    
    # Spatial structure
    deformation_clusters: List[List[str]]  # Indicators deforming together
    isolated_deformers: List[str]          # Indicators deforming alone
    
    # Interpretation
    deformation_type: str               # "compression", "expansion", "shear", "rotation"
    structural_risk: float              # Risk of permanent geometry change


# =============================================================================
# REGIME MONITOR (Gemini's Core + Extensions)
# =============================================================================

class RegimeMonitor:
    """
    Analyzes aggregate behavioral state to predict regime transitions.
    
    This is Gemini's formulation extended with:
    - Tension momentum tracking (is it building or releasing?)
    - Cohort-level analysis (which groups are stressed?)
    - Deformation field computation (where is geometry pushed?)
    - Recovery detection (post-rupture stabilization)
    
    Hat tip: Gemini for the core structure.
    """
    
    def __init__(
        self,
        geometry: Any,  # SystemGeometryInterface or similar
        config: Optional[InterpretationConfig] = None
    ):
        """
        Initialize the regime monitor.
        
        Args:
            geometry: System geometry with stability and cohort info
            config: Interpretation configuration
        """
        self.geometry = geometry
        self.config = config or InterpretationConfig()
        
        # Signal history for momentum computation
        self.history: List[RegimeSignal] = []
        
        # Cohort mapping (indicator -> cohort)
        self.cohort_map: Dict[str, str] = {}
        self._build_cohort_map()
        
        logger.info(f"RegimeMonitor initialized with {len(self.cohort_map)} indicators")
    
    def _build_cohort_map(self):
        """Build indicator to cohort mapping from geometry."""
        if hasattr(self.geometry, 'indicator_bounds'):
            for ind_id, bound in self.geometry.indicator_bounds.items():
                # Try to get cohort from bound
                cohort = getattr(bound, 'cohort', None) or 'default'
                self.cohort_map[ind_id] = cohort
    
    def evaluate_system(
        self,
        states: Dict[str, Any],  # Dict[str, IndicatorState]
        previous_geometry: Optional[Any] = None
    ) -> RegimeSignal:
        """
        Evaluate system-wide regime status.
        
        This is Gemini's core algorithm with PRISM extensions.
        
        Args:
            states: Dictionary of indicator_id -> IndicatorState
            previous_geometry: Previous geometry for dimension change detection
        
        Returns:
            RegimeSignal with full system analysis
        """
        if not states:
            return self._empty_signal()
        
        n_indicators = len(states)
        
        # === GEMINI'S CORE ===
        
        # 1. Systemic Tension (Integrated Hidden Mass)
        hidden_masses = [s.hidden_mass for s in states.values()]
        total_mass = sum(hidden_masses)
        avg_tension = total_mass / n_indicators
        
        # 2. Geometric Entropy
        # Distribution of kinetic energy across indicators
        energies = np.array([s.kinetic_energy for s in states.values()])
        energies = np.maximum(energies, 1e-10)  # Avoid log(0)
        p = energies / np.sum(energies)
        entropy = -np.sum(p * np.log(p))
        
        # Normalize entropy (max entropy = log(n))
        max_entropy = np.log(min(n_indicators, self.config.max_entropy_indicators))
        normalized_entropy = entropy / max(max_entropy, 1e-10)
        
        # 3. Critical Indicators (Hidden Mass > 2σ)
        mass_mean = np.mean(hidden_masses)
        mass_std = np.std(hidden_masses)
        threshold = mass_mean + self.config.hidden_mass_sigma * mass_std
        
        critical = [ind_id for ind_id, s in states.items() 
                   if s.hidden_mass > threshold]
        
        # 4. Rupture Risk
        stability = getattr(self.geometry, 'geometric_stability', 0.5)
        raw_risk = (avg_tension * normalized_entropy) / (stability + 0.1)
        rupture_risk = 1 / (1 + np.exp(-10 * (raw_risk - self.config.rupture_threshold)))
        
        # 5. Status Classification
        status = self._classify_status(rupture_risk)
        
        # === PRISM EXTENSIONS ===
        
        # 6. Tension Source Analysis
        tension_source = self._analyze_tension_source(states, critical)
        
        # 7. Momentum (from history)
        momentum, acceleration = self._compute_momentum(avg_tension)
        
        # 8. Cohort Analysis
        cohort_tension, critical_cohorts = self._analyze_cohorts(states)
        
        # 9. Geometry Health
        geometry_condition = getattr(self.geometry, 'condition_number', 1.0)
        
        if previous_geometry is not None:
            prev_dim = getattr(previous_geometry, 'effective_dimension', 
                              self.geometry.effective_dimension)
            dim_change = self.geometry.effective_dimension - prev_dim
        else:
            dim_change = 0.0
        
        # 10. Hidden Mass Concentration (Gini coefficient)
        gini = self._compute_gini(hidden_masses)
        
        # Top 5 contributors
        sorted_by_mass = sorted(states.items(), 
                               key=lambda x: x[1].hidden_mass, 
                               reverse=True)
        top_5 = [ind_id for ind_id, _ in sorted_by_mass[:5]]
        
        # 11. Recovery Detection (if applicable)
        recovery_progress = 0.0
        stabilizing = []
        if len(self.history) >= 3:
            recent_statuses = [s.status for s in self.history[-3:]]
            if SystemStatus.RUPTURE in recent_statuses and status != SystemStatus.RUPTURE:
                status = SystemStatus.RECOVERY
                recovery_progress = self._compute_recovery_progress()
                stabilizing = self._find_stabilizing_indicators(states)
        
        # Build signal
        signal = RegimeSignal(
            timestamp=datetime.now(),
            window_id=getattr(self.geometry, 'window_id', 'unknown'),
            
            # Gemini core
            systemic_tension=avg_tension,
            geometric_entropy=normalized_entropy,
            critical_indicators=critical,
            rupture_risk=rupture_risk,
            status=status,
            
            # PRISM extensions
            tension_source=tension_source,
            tension_momentum=momentum,
            tension_acceleration=acceleration,
            critical_cohorts=critical_cohorts,
            cohort_tension=cohort_tension,
            geometry_condition=geometry_condition,
            effective_dimension_change=dim_change,
            hidden_mass_gini=gini,
            hidden_mass_top_5=top_5,
            recovery_progress=recovery_progress,
            stabilizing_indicators=stabilizing,
        )
        
        # Update history
        self.history.append(signal)
        if len(self.history) > self.config.signal_history_length:
            self.history = self.history[-self.config.signal_history_length:]
        
        return signal
    
    def _classify_status(self, rupture_risk: float) -> SystemStatus:
        """Classify system status from rupture risk."""
        if rupture_risk >= 0.85:
            return SystemStatus.RUPTURE
        elif rupture_risk >= self.config.critical_threshold:
            return SystemStatus.CRITICAL
        elif rupture_risk >= self.config.stressed_threshold:
            return SystemStatus.STRESSED
        else:
            return SystemStatus.STABLE
    
    def _analyze_tension_source(
        self,
        states: Dict[str, Any],
        critical: List[str]
    ) -> TensionSource:
        """Determine primary source of tension."""
        if not critical:
            return TensionSource.INTERNAL
        
        # Check if critical indicators are in same cohort (cascading)
        critical_cohorts = set(self.cohort_map.get(ind, 'default') for ind in critical)
        
        if len(critical_cohorts) == 1 and len(critical) > 2:
            return TensionSource.CASCADING
        
        # Check if tension is from hidden mass (external)
        avg_hidden_mass = np.mean([states[ind].hidden_mass for ind in critical])
        avg_residual = np.mean([states[ind].residual_motion for ind in critical])
        
        if avg_hidden_mass > avg_residual * 0.5:
            return TensionSource.EXTERNAL
        
        # Check geometry stability
        stability = getattr(self.geometry, 'geometric_stability', 0.5)
        if stability < 0.3:
            return TensionSource.STRUCTURAL
        
        return TensionSource.INTERNAL
    
    def _compute_momentum(
        self,
        current_tension: float
    ) -> Tuple[float, float]:
        """Compute tension momentum and acceleration."""
        if len(self.history) < 2:
            return 0.0, 0.0
        
        tensions = [s.systemic_tension for s in self.history[-5:]]
        tensions.append(current_tension)
        
        # Momentum = first derivative (exponential weighted)
        weights = np.exp(np.linspace(-1, 0, len(tensions) - 1))
        weights /= weights.sum()
        
        diffs = np.diff(tensions)
        momentum = np.sum(diffs * weights)
        
        # Acceleration = second derivative
        if len(tensions) >= 3:
            momentums = [self.history[i].tension_momentum 
                        for i in range(-min(3, len(self.history)), 0)]
            momentums.append(momentum)
            acceleration = momentums[-1] - momentums[-2] if len(momentums) >= 2 else 0.0
        else:
            acceleration = 0.0
        
        return momentum, acceleration
    
    def _analyze_cohorts(
        self,
        states: Dict[str, Any]
    ) -> Tuple[Dict[str, float], List[str]]:
        """Analyze tension by cohort."""
        cohort_masses = defaultdict(list)
        
        for ind_id, state in states.items():
            cohort = self.cohort_map.get(ind_id, 'default')
            cohort_masses[cohort].append(state.hidden_mass)
        
        cohort_tension = {
            cohort: np.mean(masses) 
            for cohort, masses in cohort_masses.items()
        }
        
        # Critical cohorts = above threshold
        threshold = self.config.cohort_stress_threshold * np.mean(list(cohort_tension.values()))
        critical = [cohort for cohort, tension in cohort_tension.items()
                   if tension > threshold]
        
        return cohort_tension, critical
    
    def _compute_gini(self, values: List[float]) -> float:
        """Compute Gini coefficient of hidden mass distribution."""
        values = np.array(sorted(values))
        n = len(values)
        
        if n == 0 or values.sum() == 0:
            return 0.0
        
        index = np.arange(1, n + 1)
        return (2 * np.sum(index * values) - (n + 1) * values.sum()) / (n * values.sum())
    
    def _compute_recovery_progress(self) -> float:
        """Compute recovery progress after rupture."""
        if len(self.history) < 2:
            return 0.0
        
        # Find most recent rupture
        rupture_idx = -1
        for i in range(len(self.history) - 1, -1, -1):
            if self.history[i].status == SystemStatus.RUPTURE:
                rupture_idx = i
                break
        
        if rupture_idx < 0:
            return 1.0
        
        # Compare current tension to rupture tension
        rupture_tension = self.history[rupture_idx].systemic_tension
        current_tension = self.history[-1].systemic_tension
        
        if rupture_tension <= 0:
            return 1.0
        
        progress = 1.0 - (current_tension / rupture_tension)
        return float(np.clip(progress, 0, 1))
    
    def _find_stabilizing_indicators(
        self,
        states: Dict[str, Any]
    ) -> List[str]:
        """Find indicators that are helping stabilize the system."""
        # Stabilizing = low hidden mass + high resistance
        stabilizing = []
        
        for ind_id, state in states.items():
            if state.hidden_mass < 0.1 and state.resistance > 0.7:
                stabilizing.append(ind_id)
        
        return stabilizing[:5]  # Top 5
    
    def _empty_signal(self) -> RegimeSignal:
        """Return empty signal for edge cases."""
        return RegimeSignal(
            timestamp=datetime.now(),
            window_id=getattr(self.geometry, 'window_id', 'unknown'),
            systemic_tension=0.0,
            geometric_entropy=0.0,
            critical_indicators=[],
            rupture_risk=0.0,
            status=SystemStatus.STABLE,
            tension_source=TensionSource.INTERNAL,
            tension_momentum=0.0,
            tension_acceleration=0.0,
            critical_cohorts=[],
            cohort_tension={},
            geometry_condition=1.0,
            effective_dimension_change=0.0,
            hidden_mass_gini=0.0,
            hidden_mass_top_5=[],
            recovery_progress=0.0,
            stabilizing_indicators=[],
        )
    
    def get_trend(self, n_periods: int = 10) -> Dict[str, Any]:
        """Get trend analysis over recent history."""
        if len(self.history) < 2:
            return {'error': 'Insufficient history'}
        
        recent = self.history[-n_periods:]
        
        tensions = [s.systemic_tension for s in recent]
        risks = [s.rupture_risk for s in recent]
        entropies = [s.geometric_entropy for s in recent]
        
        return {
            'n_periods': len(recent),
            'tension_trend': np.polyfit(range(len(tensions)), tensions, 1)[0],
            'risk_trend': np.polyfit(range(len(risks)), risks, 1)[0],
            'entropy_trend': np.polyfit(range(len(entropies)), entropies, 1)[0],
            'status_sequence': [s.status.value for s in recent],
            'rupture_count': sum(1 for s in recent if s.status == SystemStatus.RUPTURE),
            'mean_tension': np.mean(tensions),
            'max_tension': np.max(tensions),
            'volatility': np.std(tensions),
        }


# =============================================================================
# DEFORMATION FIELD ANALYZER
# =============================================================================

class DeformationAnalyzer:
    """
    Analyzes the deformation field across the system.
    
    This is PRISM's addition to regime monitoring - understanding
    not just THAT the system is stressed, but WHERE and HOW.
    """
    
    def __init__(self, geometry: Any):
        self.geometry = geometry
    
    def compute_field(
        self,
        states: Dict[str, Any]
    ) -> DeformationField:
        """Compute system-wide deformation field."""
        if not states:
            return self._empty_field()
        
        # Collect hidden mass directions
        directions = []
        magnitudes = []
        
        for state in states.values():
            if hasattr(state, 'hidden_mass_direction'):
                directions.append(state.hidden_mass_direction)
                magnitudes.append(state.hidden_mass)
        
        if not directions:
            return self._empty_field()
        
        directions = np.array(directions)
        magnitudes = np.array(magnitudes)
        
        # Total deformation
        total_deformation = np.sum(magnitudes)
        
        # Principal direction (weighted average)
        weighted_dirs = directions * magnitudes[:, np.newaxis]
        mean_direction = np.sum(weighted_dirs, axis=0)
        norm = np.linalg.norm(mean_direction)
        if norm > 1e-10:
            mean_direction = mean_direction / norm
        
        # Deformation by PC axis
        n_pcs = min(directions.shape[1], 5)
        pc_deformations = {}
        for pc in range(n_pcs):
            pc_deformations[pc] = np.mean(np.abs(directions[:, pc]) * magnitudes)
        
        dominant_pc = max(pc_deformations.keys(), key=lambda k: pc_deformations[k])
        
        # Cluster analysis (which indicators deform together)
        clusters, isolated = self._cluster_deformations(directions, magnitudes, states)
        
        # Deformation type
        def_type = self._classify_deformation(directions, magnitudes)
        
        # Structural risk
        structural_risk = self._compute_structural_risk(total_deformation, directions)
        
        return DeformationField(
            timestamp=datetime.now(),
            window_id=getattr(self.geometry, 'window_id', 'unknown'),
            total_deformation=total_deformation,
            deformation_direction=mean_direction,
            pc_deformations=pc_deformations,
            dominant_pc=dominant_pc,
            deformation_clusters=clusters,
            isolated_deformers=isolated,
            deformation_type=def_type,
            structural_risk=structural_risk,
        )
    
    def _cluster_deformations(
        self,
        directions: np.ndarray,
        magnitudes: np.ndarray,
        states: Dict[str, Any]
    ) -> Tuple[List[List[str]], List[str]]:
        """Cluster indicators by deformation direction."""
        indicator_ids = list(states.keys())
        n = len(indicator_ids)
        
        if n < 3:
            return [], indicator_ids
        
        # Compute direction similarity matrix
        similarity = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                # Cosine similarity
                dot = np.dot(directions[i], directions[j])
                norm_i = np.linalg.norm(directions[i])
                norm_j = np.linalg.norm(directions[j])
                if norm_i > 1e-10 and norm_j > 1e-10:
                    sim = dot / (norm_i * norm_j)
                else:
                    sim = 0
                similarity[i, j] = similarity[j, i] = sim
        
        # Simple clustering: group indicators with similarity > 0.7
        threshold = 0.7
        visited = set()
        clusters = []
        isolated = []
        
        for i in range(n):
            if i in visited:
                continue
            
            # Find similar indicators
            cluster = [indicator_ids[i]]
            visited.add(i)
            
            for j in range(n):
                if j not in visited and similarity[i, j] > threshold:
                    cluster.append(indicator_ids[j])
                    visited.add(j)
            
            if len(cluster) > 1:
                clusters.append(cluster)
            else:
                isolated.append(cluster[0])
        
        return clusters, isolated
    
    def _classify_deformation(
        self,
        directions: np.ndarray,
        magnitudes: np.ndarray
    ) -> str:
        """Classify the type of deformation."""
        # Compute coherence (how aligned are deformations?)
        mean_dir = np.mean(directions * magnitudes[:, np.newaxis], axis=0)
        mean_norm = np.linalg.norm(mean_dir)
        total_mag = np.sum(magnitudes)
        
        if total_mag < 1e-10:
            return "minimal"
        
        coherence = mean_norm / total_mag
        
        # Check for compression/expansion (PC1 dominated)
        pc1_component = np.mean(np.abs(directions[:, 0]) * magnitudes) / total_mag
        
        if coherence > 0.7:
            # Highly aligned deformation
            if pc1_component > 0.5:
                if mean_dir[0] > 0:
                    return "expansion"
                else:
                    return "compression"
            else:
                return "shear"
        else:
            # Incoherent deformation
            return "rotation"
    
    def _compute_structural_risk(
        self,
        total_deformation: float,
        directions: np.ndarray
    ) -> float:
        """Compute risk of permanent geometry change."""
        # High deformation + incoherent directions = structural risk
        coherence = np.linalg.norm(np.mean(directions, axis=0))
        
        # Risk increases with deformation, decreases with coherence
        stability = getattr(self.geometry, 'geometric_stability', 0.5)
        
        raw_risk = total_deformation * (1 - coherence) / (stability + 0.1)
        
        return float(np.clip(raw_risk / 10, 0, 1))  # Normalize
    
    def _empty_field(self) -> DeformationField:
        """Return empty deformation field."""
        return DeformationField(
            timestamp=datetime.now(),
            window_id=getattr(self.geometry, 'window_id', 'unknown'),
            total_deformation=0.0,
            deformation_direction=np.array([]),
            pc_deformations={},
            dominant_pc=0,
            deformation_clusters=[],
            isolated_deformers=[],
            deformation_type="minimal",
            structural_risk=0.0,
        )


# =============================================================================
# INTERPRETATION ENGINE (Combines All)
# =============================================================================

class InterpretationEngine:
    """
    Master interpretation engine combining regime monitoring
    and deformation analysis.
    
    This is Phase 4 of PRISM - turning measurements into meaning.
    """
    
    def __init__(
        self,
        geometry: Any,
        config: Optional[InterpretationConfig] = None
    ):
        self.geometry = geometry
        self.config = config or InterpretationConfig()
        
        self.regime_monitor = RegimeMonitor(geometry, config)
        self.deformation_analyzer = DeformationAnalyzer(geometry)
        
        logger.info("InterpretationEngine initialized")
    
    def interpret(
        self,
        states: Dict[str, Any],
        previous_geometry: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Full interpretation of system state.
        
        Args:
            states: Dictionary of indicator_id -> IndicatorState
            previous_geometry: Previous geometry for comparison
        
        Returns:
            Complete interpretation including regime signal and deformation field
        """
        # Regime signal
        signal = self.regime_monitor.evaluate_system(states, previous_geometry)
        
        # Deformation field
        field = self.deformation_analyzer.compute_field(states)
        
        # Trend analysis
        trend = self.regime_monitor.get_trend()
        
        # Narrative generation
        narrative = self._generate_narrative(signal, field, trend)
        
        return {
            'signal': signal,
            'field': field,
            'trend': trend,
            'narrative': narrative,
            'timestamp': datetime.now().isoformat(),
        }
    
    def _generate_narrative(
        self,
        signal: RegimeSignal,
        field: DeformationField,
        trend: Dict[str, Any]
    ) -> str:
        """Generate human-readable interpretation narrative."""
        lines = []
        
        # Status line
        status_emoji = {
            SystemStatus.STABLE: "🟢",
            SystemStatus.STRESSED: "🟡",
            SystemStatus.CRITICAL: "🔴",
            SystemStatus.RUPTURE: "💥",
            SystemStatus.RECOVERY: "🔄",
        }
        
        emoji = status_emoji.get(signal.status, "⚪")
        lines.append(f"{emoji} System Status: {signal.status.value}")
        lines.append("")
        
        # Key metrics
        lines.append(f"Rupture Risk: {signal.rupture_risk:.1%}")
        lines.append(f"Systemic Tension: {signal.systemic_tension:.4f}")
        lines.append(f"Geometric Entropy: {signal.geometric_entropy:.4f}")
        lines.append("")
        
        # Tension analysis
        if signal.tension_momentum > 0.01:
            lines.append(f"⚠️ Tension is BUILDING ({signal.tension_momentum:+.4f})")
        elif signal.tension_momentum < -0.01:
            lines.append(f"✅ Tension is RELEASING ({signal.tension_momentum:+.4f})")
        else:
            lines.append("➡️ Tension is STABLE")
        
        lines.append(f"Tension Source: {signal.tension_source.value}")
        lines.append("")
        
        # Critical indicators
        if signal.critical_indicators:
            lines.append(f"Critical Indicators ({len(signal.critical_indicators)}):")
            for ind in signal.critical_indicators[:5]:
                lines.append(f"  • {ind}")
            if len(signal.critical_indicators) > 5:
                lines.append(f"  ... and {len(signal.critical_indicators) - 5} more")
            lines.append("")
        
        # Deformation
        if field.total_deformation > 0.1:
            lines.append(f"Deformation: {field.deformation_type} (PC{field.dominant_pc + 1} dominant)")
            lines.append(f"Structural Risk: {field.structural_risk:.1%}")
            if field.deformation_clusters:
                lines.append(f"Clustered Deformation: {len(field.deformation_clusters)} groups")
            lines.append("")
        
        # Trend
        if 'tension_trend' in trend:
            trend_dir = "↑" if trend['tension_trend'] > 0 else "↓"
            lines.append(f"10-Period Trend: Tension {trend_dir} ({trend['tension_trend']:+.4f}/period)")
        
        # Recovery
        if signal.status == SystemStatus.RECOVERY:
            lines.append("")
            lines.append(f"Recovery Progress: {signal.recovery_progress:.1%}")
            if signal.stabilizing_indicators:
                lines.append(f"Stabilizing Indicators: {', '.join(signal.stabilizing_indicators)}")
        
        return '\n'.join(lines)
    
    def quick_status(self, states: Dict[str, Any]) -> str:
        """Get quick one-line status."""
        signal = self.regime_monitor.evaluate_system(states)
        return f"{signal.status.value} | Risk: {signal.rupture_risk:.0%} | Tension: {signal.systemic_tension:.3f}"


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("PRISM Phase 4: Interpretation Layer")
    print("=" * 60)
    print()
    print("Hat tip: Gemini for the RegimeMonitor structure")
    print()
    print("Components:")
    print("  - RegimeMonitor: System-wide tension, entropy, rupture risk")
    print("  - DeformationAnalyzer: Where is geometry being pushed?")
    print("  - InterpretationEngine: Full narrative generation")
    print()
    print("Usage:")
    print("  from prism_interpretation import InterpretationEngine")
    print("  engine = InterpretationEngine(geometry)")
    print("  result = engine.interpret(indicator_states)")
    print("  print(result['narrative'])")
    print()
    print("Statuses: STABLE → STRESSED → CRITICAL → RUPTURE → RECOVERY")
