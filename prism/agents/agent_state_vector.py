"""
PRISM State Vector Agents

Agents that support state vector assembly, trajectory analysis,
and hidden mass detection.

These agents ensure that:
1. State vectors are well-formed (VectorValidation)
2. No component dominates (ComponentBalance)
3. Trajectories are coherent (TrajectoryCoherence)
4. Resistance scores are calibrated (ResistanceCalibration)
5. System-wide deformation is mapped (DeformationField)
6. Hidden mass is attributed (HiddenMassAttribution)

Agent Pipeline:
    vector_validation → component_balance → [assembly] →
    trajectory_coherence → resistance_calibration →
    deformation_field → hidden_mass_attribution

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

logger = logging.getLogger(__name__)


# =============================================================================
# COMMON TYPES
# =============================================================================

class ValidationStatus(Enum):
    """Validation outcome."""
    VALID = "valid"
    MARGINAL = "marginal"
    INVALID = "invalid"
    UNKNOWN = "unknown"


@dataclass
class ValidationResult:
    """Result of a validation check."""
    status: ValidationStatus
    score: float
    issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# 1. VECTOR VALIDATION AGENT
# =============================================================================

@dataclass
class VectorValidationConfig:
    """Configuration for vector validation."""
    min_components: int = 4              # At least 4 of 7 components
    min_dimensions_per_component: int = 1
    max_nan_ratio: float = 0.3           # Max 30% NaN values
    min_completeness: float = 0.5        # Min 50% overall completeness


class VectorValidationAgent:
    """
    Validates that assembled state vectors are well-formed.
    
    A state vector is the fundamental object in PRISM. If it's
    malformed, all downstream analysis is suspect.
    
    Checks:
    - Minimum component coverage (at least 4/7 components)
    - Dimension count per component
    - NaN ratio within bounds
    - Value sanity (no infinities, reasonable magnitudes)
    """
    
    def __init__(self, config: VectorValidationConfig = None):
        self.config = config or VectorValidationConfig()
    
    def validate(self, sv: 'PRISMStateVector') -> ValidationResult:
        """
        Validate a single state vector.
        
        Args:
            sv: PRISMStateVector to validate
            
        Returns:
            ValidationResult with status and issues
        """
        issues = []
        metadata = {}
        score = 1.0
        
        # Check 1: Component coverage
        components = sv.components
        n_with_data = sum(1 for c in components.values() 
                        if c is not None and c.n_sources > 0)
        
        metadata['components_with_data'] = n_with_data
        metadata['total_components'] = len(components)
        
        if n_with_data < self.config.min_components:
            issues.append(f"Insufficient components: {n_with_data}/{len(components)}")
            score -= 0.3
        
        # Check 2: Completeness
        completeness = sv.completeness
        metadata['completeness'] = completeness
        
        if completeness < self.config.min_completeness:
            issues.append(f"Low completeness: {completeness:.1%}")
            score -= 0.2
        
        # Check 3: NaN analysis
        flat_vector, names = sv.to_flat_vector()
        n_total = len(flat_vector)
        n_nan = np.sum(~np.isfinite(flat_vector))
        nan_ratio = n_nan / n_total if n_total > 0 else 0
        
        metadata['n_dimensions'] = n_total
        metadata['n_nan'] = n_nan
        metadata['nan_ratio'] = nan_ratio
        
        if nan_ratio > self.config.max_nan_ratio:
            issues.append(f"High NaN ratio: {nan_ratio:.1%}")
            score -= 0.2
        
        # Check 4: Value sanity
        finite_values = flat_vector[np.isfinite(flat_vector)]
        if len(finite_values) > 0:
            max_abs = np.max(np.abs(finite_values))
            metadata['max_abs_value'] = float(max_abs)
            
            if max_abs > 1e6:
                issues.append(f"Extreme values detected: max={max_abs:.2e}")
                score -= 0.1
        
        # Check 5: Per-component dimensions
        for comp_type, comp in components.items():
            if comp is not None and comp.n_sources > 0:
                if comp.n_sources < self.config.min_dimensions_per_component:
                    issues.append(f"{comp_type.name}: only {comp.n_sources} dimensions")
                    score -= 0.05
        
        # Determine status
        score = max(0, min(1, score))
        
        if score >= 0.8:
            status = ValidationStatus.VALID
        elif score >= 0.5:
            status = ValidationStatus.MARGINAL
        else:
            status = ValidationStatus.INVALID
        
        recommendations = []
        if status == ValidationStatus.INVALID:
            recommendations.append("Consider expanding data window or adding indicators")
        if nan_ratio > 0.1:
            recommendations.append("Some engines may have failed - check derived phase logs")
        
        return ValidationResult(
            status=status,
            score=score,
            issues=issues,
            recommendations=recommendations,
            metadata=metadata
        )


# =============================================================================
# 2. COMPONENT BALANCE AGENT
# =============================================================================

@dataclass
class ComponentBalanceConfig:
    """Configuration for component balance."""
    max_component_weight: float = 0.30   # No component > 30% of total
    normalization_method: str = "unit_variance"  # or "min_max", "none"
    min_variance_ratio: float = 0.01     # Min variance to include


class ComponentBalanceAgent:
    """
    Ensures no single component dominates the state vector.
    
    Problem: If Tails has 10x the magnitude of Geometry,
    distance calculations become effectively Tails-only.
    This defeats the multi-lens consensus principle.
    
    Solution:
    - Normalize each component to unit variance
    - Or weight by confidence scores  
    - Or use Mahalanobis distance
    
    This agent validates and optionally rebalances components.
    """
    
    def __init__(self, config: ComponentBalanceConfig = None):
        self.config = config or ComponentBalanceConfig()
    
    def analyze_balance(self, sv: 'PRISMStateVector') -> Dict[str, Any]:
        """
        Analyze component balance in a state vector.
        
        Returns:
            Dictionary with balance metrics and flags
        """
        magnitudes = {}
        
        for comp_type, comp in sv.components.items():
            if comp is not None and comp.n_sources > 0:
                magnitudes[comp_type.value] = comp.magnitude
        
        if not magnitudes:
            return {'status': 'no_data'}
        
        total = sum(magnitudes.values())
        
        if total == 0:
            return {'status': 'zero_total'}
        
        # Compute weights
        weights = {k: v / total for k, v in magnitudes.items()}
        
        result = {
            'magnitudes': magnitudes,
            'weights': weights,
            'max_weight': max(weights.values()),
            'min_weight': min(weights.values()),
            'weight_range': max(weights.values()) - min(weights.values()),
        }
        
        # Flags
        result['flags'] = []
        
        dominant = [k for k, v in weights.items() if v > self.config.max_component_weight]
        if dominant:
            result['flags'].append(f"Dominant components: {dominant}")
            result['dominant_components'] = dominant
        
        negligible = [k for k, v in weights.items() if v < 0.05]
        if negligible:
            result['flags'].append(f"Negligible components: {negligible}")
            result['negligible_components'] = negligible
        
        result['is_balanced'] = len(dominant) == 0
        
        return result
    
    def normalize_vector(
        self,
        sv: 'PRISMStateVector',
        population_stats: Dict[str, Dict[str, float]] = None
    ) -> 'PRISMStateVector':
        """
        Return a normalized copy of the state vector.
        
        Args:
            sv: State vector to normalize
            population_stats: Optional dict of {component: {mean, std}} for z-score
            
        Returns:
            New PRISMStateVector with normalized components
        """
        # This would create a new state vector with normalized values
        # For now, return original (normalization logic would go here)
        logger.warning("normalize_vector not fully implemented - returning original")
        return sv
    
    def validate(self, sv: 'PRISMStateVector') -> ValidationResult:
        """Validate component balance."""
        
        analysis = self.analyze_balance(sv)
        
        if analysis.get('status') in ['no_data', 'zero_total']:
            return ValidationResult(
                status=ValidationStatus.INVALID,
                score=0.0,
                issues=["Cannot analyze balance: no data"],
                metadata=analysis
            )
        
        issues = analysis.get('flags', [])
        
        score = 1.0
        if 'dominant_components' in analysis:
            score -= 0.1 * len(analysis['dominant_components'])
        if 'negligible_components' in analysis:
            score -= 0.05 * len(analysis['negligible_components'])
        
        score = max(0, min(1, score))
        
        if score >= 0.8:
            status = ValidationStatus.VALID
        elif score >= 0.5:
            status = ValidationStatus.MARGINAL
        else:
            status = ValidationStatus.INVALID
        
        return ValidationResult(
            status=status,
            score=score,
            issues=issues,
            metadata=analysis
        )


# =============================================================================
# 3. TRAJECTORY COHERENCE AGENT
# =============================================================================

@dataclass
class TrajectoryCoherenceConfig:
    """Configuration for trajectory coherence validation."""
    max_velocity_zscore: float = 4.0     # Flag jumps > 4 sigma
    min_windows: int = 3                 # Minimum windows for trajectory
    discontinuity_threshold: float = 3.0  # Z-score for discontinuity
    max_missing_windows: int = 2         # Max gaps in sequence


class TrajectoryCoherenceAgent:
    """
    Validates that trajectories are coherent and interpretable.
    
    A trajectory is the evolution of a state vector over time.
    It should be:
    - Chronologically ordered
    - Free of impossible jumps (teleportation)
    - Reasonably continuous (small gaps OK)
    - Long enough for meaningful analysis
    
    This validation is critical before computing Resistance,
    which depends on trajectory stability.
    """
    
    def __init__(self, config: TrajectoryCoherenceConfig = None):
        self.config = config or TrajectoryCoherenceConfig()
    
    def validate(self, trajectory: 'StateVectorTrajectory') -> ValidationResult:
        """
        Validate a state vector trajectory.
        
        Args:
            trajectory: StateVectorTrajectory to validate
            
        Returns:
            ValidationResult with coherence assessment
        """
        issues = []
        metadata = {}
        score = 1.0
        
        vectors = trajectory.vectors
        motions = trajectory.motions
        
        # Check 1: Minimum length
        n_windows = len(vectors)
        metadata['n_windows'] = n_windows
        
        if n_windows < self.config.min_windows:
            issues.append(f"Insufficient windows: {n_windows} < {self.config.min_windows}")
            score -= 0.4
        
        # Check 2: Chronological ordering
        if n_windows >= 2:
            dates = [v.window_start for v in vectors]
            is_ordered = all(dates[i] <= dates[i+1] for i in range(len(dates)-1))
            metadata['is_chronological'] = is_ordered
            
            if not is_ordered:
                issues.append("Windows are not chronologically ordered")
                score -= 0.3
        
        # Check 3: Velocity analysis (discontinuity detection)
        if motions:
            displacements = [m.displacement for m in motions 
                           if m.displacement is not None and np.isfinite(m.displacement)]
            
            if len(displacements) >= 3:
                mean_disp = np.mean(displacements)
                std_disp = np.std(displacements)
                
                metadata['mean_displacement'] = float(mean_disp)
                metadata['std_displacement'] = float(std_disp)
                
                if std_disp > 0:
                    z_scores = [(d - mean_disp) / std_disp for d in displacements]
                    discontinuities = sum(1 for z in z_scores if abs(z) > self.config.discontinuity_threshold)
                    
                    metadata['n_discontinuities'] = discontinuities
                    
                    if discontinuities > 0:
                        issues.append(f"Detected {discontinuities} discontinuities (z > {self.config.discontinuity_threshold})")
                        score -= 0.1 * discontinuities
        
        # Check 4: Missing data within trajectory
        completeness_scores = [v.completeness for v in vectors]
        low_completeness = sum(1 for c in completeness_scores if c < 0.3)
        
        metadata['n_low_completeness_windows'] = low_completeness
        
        if low_completeness > self.config.max_missing_windows:
            issues.append(f"Too many sparse windows: {low_completeness}")
            score -= 0.15
        
        # Determine status
        score = max(0, min(1, score))
        
        if score >= 0.8:
            status = ValidationStatus.VALID
        elif score >= 0.5:
            status = ValidationStatus.MARGINAL
        else:
            status = ValidationStatus.INVALID
        
        recommendations = []
        if 'n_discontinuities' in metadata and metadata['n_discontinuities'] > 0:
            recommendations.append("Review discontinuous windows for data quality issues")
        if low_completeness > 0:
            recommendations.append("Some windows have sparse data - consider longer windows")
        
        return ValidationResult(
            status=status,
            score=score,
            issues=issues,
            recommendations=recommendations,
            metadata=metadata
        )


# =============================================================================
# 4. RESISTANCE CALIBRATION AGENT
# =============================================================================

@dataclass
class ResistanceCalibrationConfig:
    """Configuration for resistance calibration."""
    min_resistance: float = 0.1
    max_resistance: float = 0.99
    domain_bounds: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        'finance': (0.2, 0.9),
        'climate': (0.6, 0.95),
        'epidemiology': (0.1, 0.8),
        'default': (0.1, 0.95),
    })
    min_confidence_windows: int = 5      # Need 5+ windows for high confidence


class ResistanceCalibrationAgent:
    """
    Calibrates and validates the Resistance (R) component.
    
    Resistance is EMERGENT - it's not measured directly, but
    computed from trajectory stability. This makes it crucial
    to validate:
    
    - R is within plausible bounds
    - R has sufficient confidence (enough windows)
    - R is consistent with domain expectations
    - R is comparable across indicators
    
    Without calibration, R=0.72 is meaningless. With calibration,
    we can say "this indicator is in the 85th percentile of resistance."
    """
    
    def __init__(self, config: ResistanceCalibrationConfig = None):
        self.config = config or ResistanceCalibrationConfig()
    
    def calibrate(
        self,
        trajectory: 'StateVectorTrajectory',
        domain: str = 'default',
        population_resistances: List[float] = None
    ) -> Dict[str, Any]:
        """
        Calibrate resistance for a trajectory.
        
        Args:
            trajectory: Trajectory with computed resistance
            domain: Domain for bounds lookup
            population_resistances: Optional list for percentile computation
            
        Returns:
            Calibration results including confidence interval
        """
        result = {
            'indicator_id': trajectory.indicator_id,
            'n_windows': len(trajectory.vectors),
        }
        
        # Get raw resistance
        r = trajectory.resistance_score
        result['resistance_raw'] = r
        
        if r is None or np.isnan(r):
            result['status'] = 'no_resistance'
            return result
        
        # Check bounds
        bounds = self.config.domain_bounds.get(domain, self.config.domain_bounds['default'])
        result['domain'] = domain
        result['domain_bounds'] = bounds
        
        in_bounds = bounds[0] <= r <= bounds[1]
        result['in_domain_bounds'] = in_bounds
        
        # Compute confidence
        n_windows = len(trajectory.vectors)
        if n_windows >= self.config.min_confidence_windows:
            confidence = 'high'
            confidence_score = 0.9
        elif n_windows >= 3:
            confidence = 'medium'
            confidence_score = 0.6
        else:
            confidence = 'low'
            confidence_score = 0.3
        
        result['confidence'] = confidence
        result['confidence_score'] = confidence_score
        
        # Compute percentile if population available
        if population_resistances and len(population_resistances) > 5:
            percentile = np.mean([1 if r >= other else 0 for other in population_resistances])
            result['percentile'] = float(percentile)
            
            # Interpretation
            if percentile > 0.9:
                result['interpretation'] = 'very_rigid'
            elif percentile > 0.7:
                result['interpretation'] = 'rigid'
            elif percentile > 0.3:
                result['interpretation'] = 'moderate'
            elif percentile > 0.1:
                result['interpretation'] = 'fluid'
            else:
                result['interpretation'] = 'very_fluid'
        
        result['status'] = 'calibrated'
        return result
    
    def validate(
        self,
        trajectory: 'StateVectorTrajectory',
        domain: str = 'default'
    ) -> ValidationResult:
        """Validate resistance for a trajectory."""
        
        calibration = self.calibrate(trajectory, domain)
        
        if calibration.get('status') == 'no_resistance':
            return ValidationResult(
                status=ValidationStatus.INVALID,
                score=0.0,
                issues=["No resistance computed"],
                metadata=calibration
            )
        
        issues = []
        score = calibration.get('confidence_score', 0.5)
        
        if not calibration.get('in_domain_bounds', True):
            issues.append(f"Resistance {calibration['resistance_raw']:.3f} outside domain bounds")
            score -= 0.2
        
        if calibration.get('confidence') == 'low':
            issues.append("Low confidence due to few windows")
        
        score = max(0, min(1, score))
        
        if score >= 0.7:
            status = ValidationStatus.VALID
        elif score >= 0.4:
            status = ValidationStatus.MARGINAL
        else:
            status = ValidationStatus.INVALID
        
        return ValidationResult(
            status=status,
            score=score,
            issues=issues,
            metadata=calibration
        )


# =============================================================================
# 5. DEFORMATION FIELD AGENT
# =============================================================================

class DeformationFieldAgent:
    """
    Analyzes system-wide geometric deformation.
    
    Individual indicator anomalies might be noise.
    Correlated anomalies across indicators are SIGNAL.
    
    This agent maps the "deformation field" - where in state space
    is the system under stress, expanding, contracting, or rotating?
    
    The deformation field is where hidden mass becomes visible
    at the system level, not just individual indicators.
    """
    
    def __init__(self, min_indicators: int = 5):
        self.min_indicators = min_indicators
    
    def compute_deformation_field(
        self,
        trajectories: List['StateVectorTrajectory'],
        window_idx: int = -1
    ) -> Dict[str, Any]:
        """
        Compute the deformation field across all indicators.
        
        Args:
            trajectories: List of trajectories to analyze
            window_idx: Which window transition to analyze (-1 = latest)
            
        Returns:
            Deformation field metrics
        """
        if len(trajectories) < self.min_indicators:
            return {'status': 'insufficient_indicators'}
        
        # Collect motion vectors for the specified window
        velocities = []
        indicator_ids = []
        
        for traj in trajectories:
            if traj.motions and len(traj.motions) > abs(window_idx):
                motion = traj.motions[window_idx]
                if motion.velocity is not None:
                    velocities.append(motion.velocity)
                    indicator_ids.append(traj.indicator_id)
        
        if len(velocities) < self.min_indicators:
            return {'status': 'insufficient_motions'}
        
        # Align velocities to same dimension
        min_dim = min(len(v) for v in velocities)
        velocities = np.array([v[:min_dim] for v in velocities])
        
        result = {
            'status': 'computed',
            'n_indicators': len(velocities),
            'n_dimensions': min_dim,
        }
        
        # Compute divergence (are indicators spreading or converging?)
        # Positive divergence = expansion, negative = contraction
        mean_velocity = np.mean(velocities, axis=0)
        deviations = velocities - mean_velocity
        
        # Measure spread
        result['mean_velocity_magnitude'] = float(np.linalg.norm(mean_velocity))
        result['velocity_dispersion'] = float(np.mean([np.linalg.norm(d) for d in deviations]))
        
        # Divergence proxy: correlation of position with velocity direction
        # If far indicators move farther, that's expansion
        # This is simplified - full divergence would need positions
        
        # Check for coherent motion (all moving same direction)
        velocity_norms = np.linalg.norm(velocities, axis=1, keepdims=True)
        velocity_norms[velocity_norms == 0] = 1  # Avoid division by zero
        unit_velocities = velocities / velocity_norms
        
        mean_direction = np.mean(unit_velocities, axis=0)
        coherence = np.linalg.norm(mean_direction)  # 1 = all same direction, 0 = random
        
        result['motion_coherence'] = float(coherence)
        
        if coherence > 0.7:
            result['field_type'] = 'coherent_drift'
            result['interpretation'] = 'System moving together - possible systematic factor'
        elif coherence < 0.3:
            result['field_type'] = 'dispersive'
            result['interpretation'] = 'Indicators moving independently - diversified regime'
        else:
            result['field_type'] = 'mixed'
            result['interpretation'] = 'Partial coherence - subsystems may be decoupling'
        
        # Identify outliers
        magnitudes = [np.linalg.norm(v) for v in velocities]
        mean_mag = np.mean(magnitudes)
        std_mag = np.std(magnitudes)
        
        outliers = []
        if std_mag > 0:
            for i, (ind_id, mag) in enumerate(zip(indicator_ids, magnitudes)):
                z = (mag - mean_mag) / std_mag
                if abs(z) > 2:
                    outliers.append({
                        'indicator_id': ind_id,
                        'magnitude': mag,
                        'z_score': z,
                        'type': 'fast_mover' if z > 0 else 'anchored'
                    })
        
        result['outliers'] = outliers
        result['n_outliers'] = len(outliers)
        
        return result
    
    def detect_stress_regions(
        self,
        trajectories: List['StateVectorTrajectory']
    ) -> List[Dict[str, Any]]:
        """
        Identify regions of state space under stress.
        
        Stress = where multiple indicators are being pulled
        in different directions (high local variance in velocity).
        
        Returns:
            List of stress region descriptions
        """
        # This would identify clusters of indicators with high velocity variance
        # Simplified implementation for now
        
        stress_regions = []
        
        # Compute per-indicator velocity variance
        for traj in trajectories:
            if len(traj.motions) < 3:
                continue
            
            displacements = [m.displacement for m in traj.motions 
                           if m.displacement is not None]
            
            if len(displacements) >= 3:
                var = np.var(displacements)
                if var > np.mean(displacements) * 2:  # High variance relative to mean
                    stress_regions.append({
                        'indicator_id': traj.indicator_id,
                        'velocity_variance': float(var),
                        'interpretation': 'Indicator experiencing high motion variance'
                    })
        
        return stress_regions


# =============================================================================
# 6. HIDDEN MASS ATTRIBUTION AGENT
# =============================================================================

@dataclass
class HiddenMassAttributionConfig:
    """Configuration for hidden mass attribution."""
    min_correlation_for_attribution: float = 0.5
    max_attributions: int = 5
    temporal_window_days: int = 30


class HiddenMassAttributionAgent:
    """
    Attempts to attribute detected hidden mass to candidate factors.
    
    When we detect anomalous deformation, we ask:
    "What unmeasured force could cause this?"
    
    This agent:
    - Correlates anomalies with external events (if provided)
    - Looks for patterns across indicators
    - Generates hypotheses about hidden factors
    - Does NOT prove causation - only suggests candidates
    
    Attribution is interpretive, not definitive.
    """
    
    def __init__(self, config: HiddenMassAttributionConfig = None):
        self.config = config or HiddenMassAttributionConfig()
    
    def attribute_anomaly(
        self,
        anomaly: Dict[str, Any],
        external_events: List[Dict[str, Any]] = None,
        peer_anomalies: List[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate candidate attributions for an anomaly.
        
        Args:
            anomaly: The anomaly to attribute (from detect_anomalous_deformation)
            external_events: Optional list of {date, description, category}
            peer_anomalies: Anomalies from other indicators at same time
            
        Returns:
            Ranked list of candidate attributions
        """
        attributions = []
        
        # Extract anomaly timing
        from_start = anomaly.get('from_window', (None, None))[0]
        to_start = anomaly.get('to_window', (None, None))[0]
        
        # Attribution 1: Temporal correlation with events
        if external_events and from_start:
            for event in external_events:
                event_date = event.get('date')
                if event_date:
                    # Check if event is within temporal window
                    # (Simplified - would need proper date math)
                    attributions.append({
                        'type': 'external_event',
                        'candidate': event.get('description', 'Unknown event'),
                        'category': event.get('category', 'unknown'),
                        'date': event_date,
                        'confidence': 0.5,  # Would compute proper correlation
                        'evidence': 'Temporal proximity'
                    })
        
        # Attribution 2: Peer indicator patterns
        if peer_anomalies:
            n_peers = len(peer_anomalies)
            if n_peers >= 3:
                # Multiple indicators anomalous at same time = systematic factor
                attributions.append({
                    'type': 'systematic_factor',
                    'candidate': f'Unknown systematic factor affecting {n_peers} indicators',
                    'confidence': min(0.8, 0.3 + 0.1 * n_peers),
                    'evidence': f'{n_peers} peer indicators also anomalous',
                    'peer_ids': [a.get('indicator_id') for a in peer_anomalies]
                })
            elif n_peers == 0:
                # Isolated anomaly = idiosyncratic factor
                attributions.append({
                    'type': 'idiosyncratic_factor',
                    'candidate': 'Indicator-specific hidden factor',
                    'confidence': 0.4,
                    'evidence': 'No peer indicators affected'
                })
        
        # Attribution 3: Anomaly characteristics
        z_score = anomaly.get('z_score', 0)
        if z_score > 3:
            attributions.append({
                'type': 'extreme_event',
                'candidate': 'Extreme/tail event',
                'confidence': 0.6,
                'evidence': f'Z-score of {z_score:.1f} suggests rare event'
            })
        
        # Sort by confidence
        attributions.sort(key=lambda x: x.get('confidence', 0), reverse=True)
        
        return attributions[:self.config.max_attributions]
    
    def generate_hypothesis(
        self,
        anomalies: List[Dict[str, Any]],
        geometry: 'SystemGeometry' = None
    ) -> Dict[str, Any]:
        """
        Generate a hypothesis about the hidden mass based on multiple anomalies.
        
        Args:
            anomalies: List of detected anomalies
            geometry: Optional system geometry for context
            
        Returns:
            Hypothesis about the hidden factor
        """
        if not anomalies:
            return {'status': 'no_anomalies'}
        
        # Count anomalies by component
        component_counts = {}
        for a in anomalies:
            # Would need to extract which component was most affected
            pass
        
        # Analyze temporal distribution
        n_anomalies = len(anomalies)
        
        # Generate hypothesis
        hypothesis = {
            'n_anomalies': n_anomalies,
            'hypothesis': 'Unknown',
            'confidence': 0.0,
            'evidence': [],
        }
        
        if n_anomalies >= 5:
            hypothesis['hypothesis'] = 'Persistent hidden factor affecting system'
            hypothesis['confidence'] = 0.7
            hypothesis['evidence'].append(f'{n_anomalies} anomalies detected')
        elif n_anomalies >= 2:
            hypothesis['hypothesis'] = 'Episodic hidden factor'
            hypothesis['confidence'] = 0.5
        else:
            hypothesis['hypothesis'] = 'Possible measurement artifact or rare event'
            hypothesis['confidence'] = 0.3
        
        return hypothesis


# =============================================================================
# AGENT ORCHESTRATOR
# =============================================================================

class StateVectorAgentOrchestrator:
    """
    Orchestrates all state vector-related agents.
    
    Pipeline:
    1. VectorValidation - per vector
    2. ComponentBalance - per vector
    3. TrajectoryCoherence - per trajectory
    4. ResistanceCalibration - per trajectory
    5. DeformationField - system-wide
    6. HiddenMassAttribution - per anomaly
    """
    
    def __init__(
        self,
        vector_config: VectorValidationConfig = None,
        balance_config: ComponentBalanceConfig = None,
        coherence_config: TrajectoryCoherenceConfig = None,
        resistance_config: ResistanceCalibrationConfig = None,
        attribution_config: HiddenMassAttributionConfig = None
    ):
        self.vector_agent = VectorValidationAgent(vector_config)
        self.balance_agent = ComponentBalanceAgent(balance_config)
        self.coherence_agent = TrajectoryCoherenceAgent(coherence_config)
        self.resistance_agent = ResistanceCalibrationAgent(resistance_config)
        self.deformation_agent = DeformationFieldAgent()
        self.attribution_agent = HiddenMassAttributionAgent(attribution_config)
    
    def validate_vector(self, sv: 'PRISMStateVector') -> Dict[str, Any]:
        """Validate a single state vector."""
        validation = self.vector_agent.validate(sv)
        balance = self.balance_agent.analyze_balance(sv)
        
        return {
            'indicator_id': sv.indicator_id,
            'window': f"{sv.window_start} to {sv.window_end}",
            'validation': {
                'status': validation.status.value,
                'score': validation.score,
                'issues': validation.issues,
            },
            'balance': balance,
            'overall_valid': validation.status == ValidationStatus.VALID and balance.get('is_balanced', True)
        }
    
    def validate_trajectory(
        self,
        trajectory: 'StateVectorTrajectory',
        domain: str = 'default'
    ) -> Dict[str, Any]:
        """Validate a complete trajectory."""
        coherence = self.coherence_agent.validate(trajectory)
        resistance = self.resistance_agent.validate(trajectory, domain)
        
        # Validate individual vectors
        vector_results = []
        for sv in trajectory.vectors:
            vector_results.append(self.validate_vector(sv))
        
        n_valid_vectors = sum(1 for v in vector_results if v['overall_valid'])
        
        return {
            'indicator_id': trajectory.indicator_id,
            'n_windows': len(trajectory.vectors),
            'coherence': {
                'status': coherence.status.value,
                'score': coherence.score,
                'issues': coherence.issues,
                'metadata': coherence.metadata,
            },
            'resistance': {
                'status': resistance.status.value,
                'score': resistance.score,
                'metadata': resistance.metadata,
            },
            'n_valid_vectors': n_valid_vectors,
            'vector_validity_ratio': n_valid_vectors / len(trajectory.vectors) if trajectory.vectors else 0,
        }
    
    def analyze_system(
        self,
        trajectories: List['StateVectorTrajectory'],
        domain: str = 'default'
    ) -> Dict[str, Any]:
        """Analyze the full system of trajectories."""
        
        # Validate each trajectory
        trajectory_results = {}
        all_resistances = []
        
        for traj in trajectories:
            result = self.validate_trajectory(traj, domain)
            trajectory_results[traj.indicator_id] = result
            
            if traj.resistance_score is not None:
                all_resistances.append(traj.resistance_score)
        
        # Re-calibrate with population
        if len(all_resistances) >= 5:
            for traj in trajectories:
                calibration = self.resistance_agent.calibrate(
                    traj, domain, all_resistances
                )
                trajectory_results[traj.indicator_id]['resistance_calibrated'] = calibration
        
        # Compute deformation field
        deformation = self.deformation_agent.compute_deformation_field(trajectories)
        
        # Collect all anomalies for attribution
        all_anomalies = []
        for traj in trajectories:
            anomalies = traj.detect_anomalous_deformation()
            for a in anomalies:
                a['indicator_id'] = traj.indicator_id
                all_anomalies.append(a)
        
        # Attribute anomalies
        attributions = []
        for anomaly in all_anomalies:
            # Find peer anomalies at same time
            peers = [a for a in all_anomalies 
                    if a['indicator_id'] != anomaly['indicator_id']
                    and a.get('from_window') == anomaly.get('from_window')]
            
            attrs = self.attribution_agent.attribute_anomaly(
                anomaly, peer_anomalies=peers
            )
            attributions.append({
                'anomaly': anomaly,
                'attributions': attrs
            })
        
        return {
            'n_trajectories': len(trajectories),
            'trajectory_results': trajectory_results,
            'deformation_field': deformation,
            'n_anomalies': len(all_anomalies),
            'attributions': attributions,
            'resistance_distribution': {
                'mean': float(np.mean(all_resistances)) if all_resistances else None,
                'std': float(np.std(all_resistances)) if all_resistances else None,
                'min': float(np.min(all_resistances)) if all_resistances else None,
                'max': float(np.max(all_resistances)) if all_resistances else None,
            }
        }
