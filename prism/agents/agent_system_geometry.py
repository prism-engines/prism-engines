"""
PRISM System Geometry Agents

Agents that support the construction and validation of system geometry.

These agents ensure that:
1. The geometry is well-formed and stable
2. Indicator positions are valid and meaningful
3. Constraints are properly calibrated
4. Relevance scores are interpretable

Agent Pipeline:
    geometry_validation → position_validation → constraint_calibration → relevance_scoring

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

logger = logging.getLogger(__name__)


# =============================================================================
# VALIDATION STATUS
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
    score: float  # 0-1, higher is better
    issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# GEOMETRY VALIDATION AGENT
# =============================================================================

@dataclass
class GeometryValidationConfig:
    """Configuration for geometry validation."""
    min_indicators: int = 5
    max_condition_number: float = 100.0  # For correlation matrix
    min_eigenvalue_ratio: float = 0.01   # λ_min / λ_max
    max_effective_dimension_ratio: float = 0.8  # d_eff / n
    min_correlation_variance: float = 0.01  # Correlations should vary
    required_eigenvalue_gap: float = 1.5   # Gap between PC1 and PC2


class GeometryValidationAgent:
    """
    Validates that the constructed system geometry is well-formed.
    
    Checks:
    - Sufficient indicators for meaningful geometry
    - Correlation matrix is well-conditioned
    - Eigenvalue structure is meaningful (not degenerate)
    - Geometry captures real structure (not noise)
    
    A poorly-conditioned geometry will produce unreliable positions
    and constraints, making all downstream analysis suspect.
    """
    
    def __init__(self, config: GeometryValidationConfig = None):
        self.config = config or GeometryValidationConfig()
    
    def validate(self, geometry: 'SystemGeometry') -> ValidationResult:
        """
        Validate a system geometry.
        
        Args:
            geometry: SystemGeometry object to validate
            
        Returns:
            ValidationResult with status and issues
        """
        issues = []
        score = 1.0
        metadata = {}
        
        # Check 1: Sufficient indicators
        if geometry.n_indicators < self.config.min_indicators:
            issues.append(f"Insufficient indicators: {geometry.n_indicators} < {self.config.min_indicators}")
            score -= 0.3
        
        # Check 2: Effective dimension is meaningful
        if geometry.effective_dimension is not None:
            dim_ratio = geometry.effective_dimension / geometry.n_indicators
            metadata['dimension_ratio'] = dim_ratio
            
            if dim_ratio > self.config.max_effective_dimension_ratio:
                issues.append(f"Effective dimension too high: {dim_ratio:.2%} of total")
                score -= 0.2
            elif dim_ratio < 0.1:
                issues.append(f"Effective dimension suspiciously low: {dim_ratio:.2%}")
                score -= 0.1
        
        # Check 3: Eigenvalue structure
        if geometry.explained_variance_ratio is not None:
            ev = geometry.explained_variance_ratio
            
            # Check for degenerate case (all eigenvalues equal)
            if len(ev) > 1 and np.std(ev) < self.config.min_correlation_variance:
                issues.append("Eigenvalue structure is degenerate (no dominant factors)")
                score -= 0.2
            
            # Check for meaningful gap between PC1 and PC2
            if len(ev) > 1 and ev[1] > 0:
                gap = ev[0] / ev[1]
                metadata['pc1_pc2_gap'] = gap
                
                if gap < self.config.required_eigenvalue_gap:
                    issues.append(f"Weak separation between PC1 and PC2: gap={gap:.2f}")
                    score -= 0.1
        
        # Check 4: Eigenvalue gaps (stored in geometry)
        if geometry.eigenvalue_gaps:
            metadata['eigenvalue_gaps'] = geometry.eigenvalue_gaps
        
        # Check 5: Network density
        if geometry.network_density is not None:
            metadata['network_density'] = geometry.network_density
            
            if geometry.network_density < 0.1:
                issues.append(f"Very sparse coupling network: density={geometry.network_density:.2%}")
                score -= 0.1
            elif geometry.network_density > 0.9:
                issues.append(f"Very dense coupling network: density={geometry.network_density:.2%}")
                # Not necessarily bad, but worth noting
        
        # Check 6: Correlation dispersion
        if geometry.correlation_dispersion is not None:
            metadata['correlation_dispersion'] = geometry.correlation_dispersion
            
            if geometry.correlation_dispersion < self.config.min_correlation_variance:
                issues.append("Correlations show little variation (uniform structure)")
                score -= 0.15
        
        # Determine status
        score = max(0, min(1, score))
        
        if score >= 0.8:
            status = ValidationStatus.VALID
        elif score >= 0.5:
            status = ValidationStatus.MARGINAL
        else:
            status = ValidationStatus.INVALID
        
        # Recommendations
        recommendations = []
        if status == ValidationStatus.MARGINAL:
            recommendations.append("Geometry is usable but interpret results with caution")
        if status == ValidationStatus.INVALID:
            recommendations.append("Consider expanding indicator set or time window")
        if 'pc1_pc2_gap' in metadata and metadata['pc1_pc2_gap'] < 2.0:
            recommendations.append("System may not have clear factor structure")
        
        return ValidationResult(
            status=status,
            score=score,
            issues=issues,
            recommendations=recommendations,
            metadata=metadata
        )


# =============================================================================
# POSITION VALIDATION AGENT
# =============================================================================

@dataclass
class PositionValidationConfig:
    """Configuration for position validation."""
    max_isolation_score: float = 0.95    # Completely isolated = suspect
    min_isolation_score: float = 0.05    # Completely central = suspect
    max_centroid_distance_zscore: float = 3.0  # Extreme outliers
    required_pc_loadings: int = 2        # At least 2 meaningful loadings


class PositionValidationAgent:
    """
    Validates indicator positions within the geometry.
    
    Checks:
    - Positions are not degenerate (all at centroid or all isolated)
    - No extreme outliers that might be data errors
    - PC loadings are meaningful
    - Cohort assignments are sensible
    
    Invalid positions suggest either data quality issues or
    indicators that don't belong in this system.
    """
    
    def __init__(self, config: PositionValidationConfig = None):
        self.config = config or PositionValidationConfig()
    
    def validate_indicator(
        self, 
        bounded: 'BoundedGeometry',
        population_stats: Dict[str, float] = None
    ) -> ValidationResult:
        """
        Validate a single indicator's position.
        
        Args:
            bounded: BoundedGeometry for the indicator
            population_stats: Optional population-level statistics for comparison
            
        Returns:
            ValidationResult
        """
        issues = []
        score = 1.0
        metadata = {}
        
        pos = bounded.position
        
        # Check 1: Isolation score bounds
        metadata['isolation_score'] = pos.isolation_score
        
        if pos.isolation_score > self.config.max_isolation_score:
            issues.append(f"Indicator is nearly completely isolated: {pos.isolation_score:.3f}")
            score -= 0.2
        elif pos.isolation_score < self.config.min_isolation_score:
            issues.append(f"Indicator is suspiciously central: {pos.isolation_score:.3f}")
            score -= 0.1  # Less severe - being central is usually fine
        
        # Check 2: Centroid distance (if population stats available)
        if population_stats and 'centroid_distance_mean' in population_stats:
            mean_dist = population_stats['centroid_distance_mean']
            std_dist = population_stats.get('centroid_distance_std', 1.0)
            
            if std_dist > 0:
                z_score = (pos.centroid_distance - mean_dist) / std_dist
                metadata['centroid_distance_zscore'] = z_score
                
                if abs(z_score) > self.config.max_centroid_distance_zscore:
                    issues.append(f"Extreme distance from centroid: z={z_score:.2f}")
                    score -= 0.2
        
        # Check 3: PC loadings
        loadings = [pos.pc1_loading, pos.pc2_loading, pos.pc3_loading]
        n_meaningful = sum(1 for l in loadings if abs(l) > 0.1)
        metadata['meaningful_loadings'] = n_meaningful
        
        if n_meaningful < self.config.required_pc_loadings:
            issues.append(f"Few meaningful PC loadings: {n_meaningful}")
            score -= 0.15
        
        # Check 4: Neighbor count
        metadata['n_neighbors'] = pos.n_neighbors
        
        if pos.n_neighbors == 0 and pos.isolation_score < 0.9:
            issues.append("No neighbors despite non-isolated position")
            score -= 0.1
        
        # Determine status
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
            metadata=metadata
        )
    
    def validate_all(
        self,
        geometry: 'SystemGeometry'
    ) -> Dict[str, ValidationResult]:
        """Validate all indicator positions."""
        
        # Compute population statistics
        positions = list(geometry.indicator_geometries.values())
        
        if not positions:
            return {}
        
        distances = [p.position.centroid_distance for p in positions]
        population_stats = {
            'centroid_distance_mean': np.mean(distances),
            'centroid_distance_std': np.std(distances),
        }
        
        results = {}
        for ind_id, bounded in geometry.indicator_geometries.items():
            results[ind_id] = self.validate_indicator(bounded, population_stats)
        
        return results


# =============================================================================
# CONSTRAINT CALIBRATION AGENT
# =============================================================================

@dataclass
class ConstraintCalibrationConfig:
    """Configuration for constraint calibration."""
    min_degrees_of_freedom: float = 1.0
    max_degrees_of_freedom: float = 10.0
    volatility_zscore_threshold: float = 3.0
    mean_reversion_bounds: Tuple[float, float] = (0.0, 1.0)


class ConstraintCalibrationAgent:
    """
    Calibrates and validates behavioral constraints.
    
    Ensures:
    - Degrees of freedom are within reasonable bounds
    - Expected volatility is calibrated to population
    - Mean-reversion strength is interpretable
    - Constraints are consistent with position
    
    Poorly calibrated constraints lead to meaningless conformance scores.
    """
    
    def __init__(self, config: ConstraintCalibrationConfig = None):
        self.config = config or ConstraintCalibrationConfig()
    
    def calibrate(
        self,
        bounded: 'BoundedGeometry',
        population_volatilities: List[float] = None
    ) -> ValidationResult:
        """
        Calibrate constraints for an indicator.
        
        Args:
            bounded: BoundedGeometry with constraints to calibrate
            population_volatilities: Volatilities of all indicators for comparison
            
        Returns:
            ValidationResult with calibration status
        """
        issues = []
        recommendations = []
        metadata = {}
        score = 1.0
        
        constraints = bounded.constraints
        
        # Check 1: Degrees of freedom bounds
        dof = constraints.degrees_of_freedom
        metadata['degrees_of_freedom'] = dof
        
        if dof < self.config.min_degrees_of_freedom:
            issues.append(f"Degrees of freedom too low: {dof:.2f}")
            recommendations.append("Indicator may be over-constrained")
            score -= 0.2
        elif dof > self.config.max_degrees_of_freedom:
            issues.append(f"Degrees of freedom too high: {dof:.2f}")
            recommendations.append("Indicator may be under-constrained")
            score -= 0.1
        
        # Check 2: Volatility calibration
        vol = constraints.expected_volatility
        metadata['expected_volatility'] = vol
        
        if population_volatilities and len(population_volatilities) > 3:
            mean_vol = np.mean(population_volatilities)
            std_vol = np.std(population_volatilities)
            
            if std_vol > 0:
                vol_z = (vol - mean_vol) / std_vol
                metadata['volatility_zscore'] = vol_z
                
                if abs(vol_z) > self.config.volatility_zscore_threshold:
                    issues.append(f"Extreme volatility: z={vol_z:.2f}")
                    score -= 0.15
        
        # Check 3: Mean reversion bounds
        mr = constraints.mean_reversion_strength
        metadata['mean_reversion_strength'] = mr
        
        min_mr, max_mr = self.config.mean_reversion_bounds
        if mr < min_mr or mr > max_mr:
            issues.append(f"Mean reversion out of bounds: {mr:.3f}")
            score -= 0.1
        
        # Check 4: Consistency with position
        # High centrality should imply more constraints (lower DoF)
        centrality = bounded.relevance.structural_centrality
        if centrality > 0.7 and dof > 5:
            issues.append("High centrality but few constraints")
            recommendations.append("Expected more constrained motion for central indicator")
            score -= 0.1
        
        # Determine status
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
            recommendations=recommendations,
            metadata=metadata
        )


# =============================================================================
# RELEVANCE SCORING AGENT
# =============================================================================

@dataclass
class RelevanceScoringConfig:
    """Configuration for relevance scoring."""
    centrality_weight: float = 0.4
    eigenvector_weight: float = 0.3
    information_weight: float = 0.3
    min_relevance_threshold: float = 0.1
    high_relevance_threshold: float = 0.7


class RelevanceScoringAgent:
    """
    Computes and validates relevance scores.
    
    Relevance is RELATIONAL - it depends on position and coupling,
    not on intrinsic indicator properties.
    
    This agent:
    - Computes composite relevance from components
    - Validates score distributions
    - Identifies high/low relevance indicators
    - Flags anomalous relevance patterns
    """
    
    def __init__(self, config: RelevanceScoringConfig = None):
        self.config = config or RelevanceScoringConfig()
    
    def compute_relevance(
        self,
        bounded: 'BoundedGeometry'
    ) -> float:
        """
        Compute composite relevance score.
        
        Args:
            bounded: BoundedGeometry with relevance components
            
        Returns:
            Composite relevance score in [0, 1]
        """
        rel = bounded.relevance
        
        # Weighted combination
        score = (
            self.config.centrality_weight * rel.structural_centrality +
            self.config.eigenvector_weight * rel.eigenvector_centrality +
            self.config.information_weight * rel.information_value
        )
        
        return float(np.clip(score, 0, 1))
    
    def analyze_distribution(
        self,
        geometry: 'SystemGeometry'
    ) -> Dict[str, Any]:
        """
        Analyze relevance distribution across all indicators.
        
        Returns:
            Dictionary with distribution statistics and flags
        """
        scores = []
        for bounded in geometry.indicator_geometries.values():
            scores.append(self.compute_relevance(bounded))
        
        if not scores:
            return {'status': 'no_data'}
        
        scores = np.array(scores)
        
        result = {
            'n_indicators': len(scores),
            'mean_relevance': float(np.mean(scores)),
            'std_relevance': float(np.std(scores)),
            'min_relevance': float(np.min(scores)),
            'max_relevance': float(np.max(scores)),
            'median_relevance': float(np.median(scores)),
        }
        
        # Count high/low relevance
        result['n_high_relevance'] = int(np.sum(scores >= self.config.high_relevance_threshold))
        result['n_low_relevance'] = int(np.sum(scores <= self.config.min_relevance_threshold))
        
        # Flags
        result['flags'] = []
        
        if result['std_relevance'] < 0.05:
            result['flags'].append("Relevance scores show little variation")
        
        if result['n_high_relevance'] == 0:
            result['flags'].append("No high-relevance indicators identified")
        
        if result['n_low_relevance'] > len(scores) * 0.5:
            result['flags'].append("Majority of indicators have low relevance")
        
        # Identify top/bottom indicators
        indicator_scores = [
            (ind_id, self.compute_relevance(bounded))
            for ind_id, bounded in geometry.indicator_geometries.items()
        ]
        indicator_scores.sort(key=lambda x: x[1], reverse=True)
        
        result['top_5'] = indicator_scores[:5]
        result['bottom_5'] = indicator_scores[-5:]
        
        return result
    
    def validate_relevance(
        self,
        bounded: 'BoundedGeometry'
    ) -> ValidationResult:
        """Validate relevance for a single indicator."""
        
        relevance = self.compute_relevance(bounded)
        rel = bounded.relevance
        
        issues = []
        metadata = {
            'composite_relevance': relevance,
            'structural_centrality': rel.structural_centrality,
            'eigenvector_centrality': rel.eigenvector_centrality,
            'information_value': rel.information_value,
        }
        
        # Check for inconsistencies
        if rel.structural_centrality > 0.7 and rel.eigenvector_centrality < 0.2:
            issues.append("High structural but low eigenvector centrality")
        
        if rel.influence_strength > 0.5 and rel.granger_out_degree == 0:
            issues.append("High correlation influence but no Granger causality")
        
        # Influence/sensitivity balance
        ratio = rel.influence_sensitivity_ratio
        metadata['influence_sensitivity_ratio'] = ratio
        
        if ratio > 5:
            metadata['role'] = 'strong_leader'
        elif ratio > 1.5:
            metadata['role'] = 'leader'
        elif ratio < 0.2:
            metadata['role'] = 'strong_follower'
        elif ratio < 0.67:
            metadata['role'] = 'follower'
        else:
            metadata['role'] = 'balanced'
        
        score = 1.0 - 0.1 * len(issues)
        score = max(0, min(1, score))
        
        status = ValidationStatus.VALID if score >= 0.8 else ValidationStatus.MARGINAL
        
        return ValidationResult(
            status=status,
            score=score,
            issues=issues,
            metadata=metadata
        )


# =============================================================================
# GEOMETRY STABILITY AGENT
# =============================================================================

class GeometryStabilityAgent:
    """
    Assesses how stable the system geometry is over time.
    
    A stable geometry means positions and constraints are reliable.
    An unstable geometry means the "box" is changing shape,
    making behavioral interpretation difficult.
    
    This agent compares geometries across adjacent windows.
    """
    
    def __init__(self, stability_threshold: float = 0.7):
        self.stability_threshold = stability_threshold
    
    def compare_geometries(
        self,
        geom1: 'SystemGeometry',
        geom2: 'SystemGeometry'
    ) -> Dict[str, float]:
        """
        Compare two geometries for stability.
        
        Returns:
            Dictionary of stability metrics
        """
        result = {}
        
        # Compare effective dimension
        if geom1.effective_dimension and geom2.effective_dimension:
            dim_change = abs(geom1.effective_dimension - geom2.effective_dimension)
            result['dimension_stability'] = 1.0 / (1.0 + dim_change)
        
        # Compare explained variance structure
        if geom1.explained_variance_ratio is not None and geom2.explained_variance_ratio is not None:
            ev1 = geom1.explained_variance_ratio[:5]
            ev2 = geom2.explained_variance_ratio[:5]
            
            min_len = min(len(ev1), len(ev2))
            if min_len > 0:
                diff = np.linalg.norm(ev1[:min_len] - ev2[:min_len])
                result['eigenvalue_stability'] = 1.0 / (1.0 + diff)
        
        # Compare mean correlation
        if geom1.mean_correlation and geom2.mean_correlation:
            corr_change = abs(geom1.mean_correlation - geom2.mean_correlation)
            result['correlation_stability'] = 1.0 / (1.0 + corr_change * 5)
        
        # Compare network density
        if geom1.network_density and geom2.network_density:
            density_change = abs(geom1.network_density - geom2.network_density)
            result['density_stability'] = 1.0 / (1.0 + density_change * 5)
        
        # Compare indicator positions
        common_indicators = set(geom1.indicator_geometries.keys()) & set(geom2.indicator_geometries.keys())
        
        if common_indicators:
            position_diffs = []
            for ind_id in common_indicators:
                pos1 = geom1.indicator_geometries[ind_id].position
                pos2 = geom2.indicator_geometries[ind_id].position
                
                # Compare PC loadings
                loadings1 = np.array([pos1.pc1_loading, pos1.pc2_loading, pos1.pc3_loading])
                loadings2 = np.array([pos2.pc1_loading, pos2.pc2_loading, pos2.pc3_loading])
                
                diff = np.linalg.norm(loadings1 - loadings2)
                position_diffs.append(diff)
            
            mean_position_diff = np.mean(position_diffs)
            result['position_stability'] = 1.0 / (1.0 + mean_position_diff)
        
        # Composite stability
        if result:
            result['composite_stability'] = np.mean(list(result.values()))
        
        return result
    
    def is_stable(self, stability_metrics: Dict[str, float]) -> bool:
        """Check if geometry is stable based on metrics."""
        if 'composite_stability' not in stability_metrics:
            return True  # Assume stable if we can't measure
        
        return stability_metrics['composite_stability'] >= self.stability_threshold


# =============================================================================
# AGENT ORCHESTRATOR
# =============================================================================

class GeometryAgentOrchestrator:
    """
    Orchestrates all geometry-related agents.
    
    Runs the validation pipeline:
    1. Geometry validation
    2. Position validation (per indicator)
    3. Constraint calibration (per indicator)
    4. Relevance scoring
    """
    
    def __init__(
        self,
        geometry_config: GeometryValidationConfig = None,
        position_config: PositionValidationConfig = None,
        constraint_config: ConstraintCalibrationConfig = None,
        relevance_config: RelevanceScoringConfig = None
    ):
        self.geometry_agent = GeometryValidationAgent(geometry_config)
        self.position_agent = PositionValidationAgent(position_config)
        self.constraint_agent = ConstraintCalibrationAgent(constraint_config)
        self.relevance_agent = RelevanceScoringAgent(relevance_config)
        self.stability_agent = GeometryStabilityAgent()
    
    def validate_geometry(
        self,
        geometry: 'SystemGeometry'
    ) -> Dict[str, Any]:
        """
        Run full validation pipeline on a geometry.
        
        Returns:
            Dictionary with all validation results
        """
        results = {
            'window_start': geometry.window_start,
            'window_end': geometry.window_end,
            'n_indicators': geometry.n_indicators,
        }
        
        # Step 1: Validate geometry structure
        geom_result = self.geometry_agent.validate(geometry)
        results['geometry_validation'] = {
            'status': geom_result.status.value,
            'score': geom_result.score,
            'issues': geom_result.issues,
            'metadata': geom_result.metadata,
        }
        
        # Step 2: Validate positions
        position_results = self.position_agent.validate_all(geometry)
        valid_positions = sum(1 for r in position_results.values() if r.status == ValidationStatus.VALID)
        results['position_validation'] = {
            'n_valid': valid_positions,
            'n_marginal': sum(1 for r in position_results.values() if r.status == ValidationStatus.MARGINAL),
            'n_invalid': sum(1 for r in position_results.values() if r.status == ValidationStatus.INVALID),
        }
        
        # Step 3: Calibrate constraints
        volatilities = [
            b.constraints.expected_volatility 
            for b in geometry.indicator_geometries.values()
            if b.constraints.expected_volatility > 0
        ]
        
        constraint_scores = []
        for bounded in geometry.indicator_geometries.values():
            result = self.constraint_agent.calibrate(bounded, volatilities)
            constraint_scores.append(result.score)
        
        results['constraint_calibration'] = {
            'mean_score': float(np.mean(constraint_scores)) if constraint_scores else 0,
            'min_score': float(np.min(constraint_scores)) if constraint_scores else 0,
        }
        
        # Step 4: Analyze relevance
        relevance_analysis = self.relevance_agent.analyze_distribution(geometry)
        results['relevance_analysis'] = relevance_analysis
        
        # Overall assessment
        overall_score = np.mean([
            geom_result.score,
            valid_positions / geometry.n_indicators if geometry.n_indicators > 0 else 0,
            results['constraint_calibration']['mean_score'],
        ])
        
        results['overall_score'] = float(overall_score)
        results['overall_status'] = (
            'valid' if overall_score >= 0.8 else
            'marginal' if overall_score >= 0.5 else
            'invalid'
        )
        
        return results


# =============================================================================
# PERSISTENCE
# =============================================================================

def persist_geometry_validation(
    conn,
    geometry: 'SystemGeometry',
    validation_results: Dict[str, Any]
) -> bool:
    """Persist geometry validation results to database."""
    
    conn.execute("""
        CREATE TABLE IF NOT EXISTS meta.geometry_validation (
            window_start DATE NOT NULL,
            window_end DATE NOT NULL,
            n_indicators INTEGER,
            overall_score DOUBLE,
            overall_status VARCHAR,
            geometry_score DOUBLE,
            n_valid_positions INTEGER,
            mean_constraint_score DOUBLE,
            mean_relevance DOUBLE,
            validated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (window_start, window_end)
        )
    """)
    
    try:
        conn.execute("""
            INSERT INTO meta.geometry_validation
            (window_start, window_end, n_indicators, overall_score, overall_status,
             geometry_score, n_valid_positions, mean_constraint_score, mean_relevance)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT DO UPDATE SET
                overall_score = EXCLUDED.overall_score,
                overall_status = EXCLUDED.overall_status,
                geometry_score = EXCLUDED.geometry_score,
                n_valid_positions = EXCLUDED.n_valid_positions,
                mean_constraint_score = EXCLUDED.mean_constraint_score,
                mean_relevance = EXCLUDED.mean_relevance,
                validated_at = CURRENT_TIMESTAMP
        """, [
            geometry.window_start,
            geometry.window_end,
            geometry.n_indicators,
            validation_results.get('overall_score'),
            validation_results.get('overall_status'),
            validation_results.get('geometry_validation', {}).get('score'),
            validation_results.get('position_validation', {}).get('n_valid'),
            validation_results.get('constraint_calibration', {}).get('mean_score'),
            validation_results.get('relevance_analysis', {}).get('mean_relevance'),
        ])
        return True
    except Exception as e:
        logger.error(f"Failed to persist geometry validation: {e}")
        return False
