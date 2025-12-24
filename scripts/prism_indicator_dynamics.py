"""
PRISM Indicator Dynamics Within System Geometry

Tracks individual indicators relative to evolving system geometry:
- Where is the indicator positioned in the geometry?
- What motion does that position predict?
- What motion actually occurred?
- The difference is Hidden Mass.

Key insight: Hidden Mass is RELATIVE to position in geometry.
An indicator at the center has different expected behavior than one at the edge.
The same motion might be normal for one position, anomalous for another.

The system geometry is the "box". Indicators are particles in the box.
The box itself evolves over time. We track both.

Usage:
    tracker = IndicatorDynamicsTracker('prism.db')
    dynamics = tracker.compute_indicator_dynamics(
        indicator_id='SPY',
        start_date=datetime(2007, 1, 1),
        end_date=datetime(2009, 12, 31)
    )
    
    # See when hidden mass accumulated
    for event in dynamics.hidden_mass_events:
        print(f"{event.date}: {event.magnitude:.2f} - {event.interpretation}")

Author: Jason (PRISM Project)
Date: December 2024
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class IndicatorSnapshot:
    """
    Single point in time for an indicator within the geometry.
    """
    timestamp: datetime
    indicator_id: str
    
    # Position in geometry
    position: np.ndarray              # PC coordinates
    distance_from_centroid: float     # How far from center?
    position_percentile: float        # 0-100, where in distribution?
    
    # Motion
    velocity: np.ndarray              # Actual motion
    expected_velocity: np.ndarray     # Motion predicted by position
    residual_velocity: np.ndarray     # velocity - expected
    
    # Mass decomposition
    total_mass: float                 # Indicator's contribution to system
    explained_mass: float             # Part explained by geometry
    hidden_mass: float                # Unexplained part
    hidden_mass_ratio: float          # hidden / total
    
    # Context
    system_tension: float             # System-wide tension at this moment
    cohort_id: str                    # Which cohort?
    cohort_tension: float             # Tension within cohort
    
    # Classification
    behavior: str                     # 'normal', 'drifting', 'stressed', 'anomalous'
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp.isoformat(),
            'indicator_id': self.indicator_id,
            'position': self.position.tolist(),
            'distance_from_centroid': self.distance_from_centroid,
            'position_percentile': self.position_percentile,
            'velocity': self.velocity.tolist(),
            'expected_velocity': self.expected_velocity.tolist(),
            'residual_velocity': self.residual_velocity.tolist(),
            'total_mass': self.total_mass,
            'explained_mass': self.explained_mass,
            'hidden_mass': self.hidden_mass,
            'hidden_mass_ratio': self.hidden_mass_ratio,
            'system_tension': self.system_tension,
            'cohort_id': self.cohort_id,
            'cohort_tension': self.cohort_tension,
            'behavior': self.behavior,
        }


@dataclass
class HiddenMassEvent:
    """
    A significant hidden mass event for an indicator.
    """
    timestamp: datetime
    indicator_id: str
    
    # Magnitude
    hidden_mass: float
    z_score: float                    # How many std from normal?
    
    # Direction
    direction: np.ndarray             # Which dimensions?
    dominant_dimension: str           # PC1, PC2, etc.
    
    # Context
    system_status: str                # STABLE, STRESSED, CRITICAL
    preceding_tension: float          # Was system already stressed?
    
    # Interpretation
    interpretation: str               # Human-readable
    
    # Persistence
    duration_frames: int              # How long did it last?
    peak_magnitude: float             # Highest value during event
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp.isoformat(),
            'indicator_id': self.indicator_id,
            'hidden_mass': self.hidden_mass,
            'z_score': self.z_score,
            'direction': self.direction.tolist(),
            'dominant_dimension': self.dominant_dimension,
            'system_status': self.system_status,
            'preceding_tension': self.preceding_tension,
            'interpretation': self.interpretation,
            'duration_frames': self.duration_frames,
            'peak_magnitude': self.peak_magnitude,
        }


@dataclass
class IndicatorDynamics:
    """
    Complete dynamics of an indicator over a time period.
    """
    indicator_id: str
    start_date: datetime
    end_date: datetime
    
    # Time series
    snapshots: List[IndicatorSnapshot]
    
    # Aggregated metrics
    mean_hidden_mass: float
    max_hidden_mass: float
    total_hidden_mass_accumulation: float
    
    # Events
    hidden_mass_events: List[HiddenMassEvent]
    n_anomalous_frames: int
    
    # Position statistics
    mean_distance_from_centroid: float
    position_volatility: float        # How much did position jump around?
    cohort_changes: int               # How many times changed cohort?
    
    # Behavior profile
    pct_normal: float
    pct_drifting: float
    pct_stressed: float
    pct_anomalous: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'indicator_id': self.indicator_id,
            'start_date': self.start_date.isoformat(),
            'end_date': self.end_date.isoformat(),
            'n_snapshots': len(self.snapshots),
            'mean_hidden_mass': self.mean_hidden_mass,
            'max_hidden_mass': self.max_hidden_mass,
            'total_hidden_mass_accumulation': self.total_hidden_mass_accumulation,
            'n_hidden_mass_events': len(self.hidden_mass_events),
            'n_anomalous_frames': self.n_anomalous_frames,
            'mean_distance_from_centroid': self.mean_distance_from_centroid,
            'position_volatility': self.position_volatility,
            'cohort_changes': self.cohort_changes,
            'behavior_profile': {
                'normal': self.pct_normal,
                'drifting': self.pct_drifting,
                'stressed': self.pct_stressed,
                'anomalous': self.pct_anomalous,
            },
            'snapshots': [s.to_dict() for s in self.snapshots],
            'events': [e.to_dict() for e in self.hidden_mass_events],
        }
    
    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            f"=== Indicator Dynamics: {self.indicator_id} ===",
            f"Period: {self.start_date.date()} to {self.end_date.date()}",
            f"Frames: {len(self.snapshots)}",
            "",
            f"Hidden Mass:",
            f"  Mean: {self.mean_hidden_mass:.4f}",
            f"  Max:  {self.max_hidden_mass:.4f}",
            f"  Events: {len(self.hidden_mass_events)}",
            "",
            f"Position:",
            f"  Mean distance from center: {self.mean_distance_from_centroid:.4f}",
            f"  Volatility: {self.position_volatility:.4f}",
            f"  Cohort changes: {self.cohort_changes}",
            "",
            f"Behavior Profile:",
            f"  Normal:    {self.pct_normal:.1f}%",
            f"  Drifting:  {self.pct_drifting:.1f}%",
            f"  Stressed:  {self.pct_stressed:.1f}%",
            f"  Anomalous: {self.pct_anomalous:.1f}%",
        ]
        
        if self.hidden_mass_events:
            lines.append("")
            lines.append("Significant Events:")
            for event in self.hidden_mass_events[:5]:
                lines.append(f"  {event.timestamp.date()}: {event.interpretation}")
        
        return '\n'.join(lines)


@dataclass
class SystemIndicatorMatrix:
    """
    All indicators tracked together over time.
    
    This is the full picture: system geometry + all indicator dynamics.
    """
    start_date: datetime
    end_date: datetime
    n_frames: int
    n_indicators: int
    
    # Per-indicator dynamics
    indicators: Dict[str, IndicatorDynamics]
    
    # System-level time series
    system_tension_series: List[Tuple[datetime, float]]
    system_hidden_mass_series: List[Tuple[datetime, float]]
    
    # Cross-indicator analysis
    hidden_mass_correlations: Dict[Tuple[str, str], float]
    cascade_events: List[Dict[str, Any]]  # When hidden mass propagated
    
    # Rankings
    most_hidden_mass: List[str]       # Indicators with most accumulation
    most_volatile: List[str]          # Indicators with most position change
    most_stable: List[str]            # Indicators with least hidden mass


# =============================================================================
# INDICATOR DYNAMICS TRACKER
# =============================================================================

class IndicatorDynamicsTracker:
    """
    Tracks individual indicators within evolving system geometry.
    
    The key insight: hidden mass is RELATIVE to position.
    
    1. System geometry defines expected behavior for each position
    2. Indicator position determines what "normal" motion looks like
    3. Hidden mass = deviation from position-appropriate behavior
    
    An indicator at the edge of the geometry is expected to behave differently
    than one at the center. We measure deviation from expectation, not raw motion.
    """
    
    def __init__(self, db_path: str):
        """
        Initialize tracker.
        
        Args:
            db_path: Path to PRISM DuckDB database
        """
        self.db_path = Path(db_path)
        
        # Will be populated by geometry frames
        self._geometry_frames = None
        self._indicator_positions = None
        self._system_covariance = None
    
    def load_geometry_frames(
        self,
        frames_path: str = None,
        start_date: datetime = None,
        end_date: datetime = None
    ):
        """
        Load pre-computed geometry frames.
        
        Either from JSON (from TemporalGeometryTracker) or compute fresh.
        """
        if frames_path and Path(frames_path).exists():
            with open(frames_path) as f:
                data = json.load(f)
            self._geometry_frames = data.get('frames', [])
            logger.info(f"Loaded {len(self._geometry_frames)} geometry frames")
        else:
            # Compute fresh
            from prism_temporal_geometry import TemporalGeometryTracker
            
            tracker = TemporalGeometryTracker(str(self.db_path))
            frames = tracker.compute_trajectory(start_date, end_date)
            self._geometry_frames = [f.to_dict() for f in frames]
            logger.info(f"Computed {len(self._geometry_frames)} geometry frames")
    
    def compute_indicator_dynamics(
        self,
        indicator_id: str,
        start_date: datetime,
        end_date: datetime,
        geometry_frames: List[Dict] = None
    ) -> IndicatorDynamics:
        """
        Compute full dynamics for a single indicator.
        
        Args:
            indicator_id: Indicator to track
            start_date: Start of period
            end_date: End of period
            geometry_frames: Pre-computed geometry frames (optional)
        
        Returns:
            IndicatorDynamics with complete analysis
        """
        frames = geometry_frames or self._geometry_frames
        
        if not frames:
            raise ValueError("No geometry frames available. Call load_geometry_frames first.")
        
        snapshots = []
        previous_position = None
        hidden_mass_series = []
        behaviors = []
        cohorts = []
        
        for frame in frames:
            # Parse timestamp
            timestamp = datetime.fromisoformat(frame['window_center'])
            
            if timestamp < start_date or timestamp > end_date:
                continue
            
            # Check if indicator is in this frame
            positions = frame.get('positions', {})
            if indicator_id not in positions:
                continue
            
            position = np.array(positions[indicator_id])
            
            # Compute metrics
            snapshot = self._compute_snapshot(
                indicator_id=indicator_id,
                timestamp=timestamp,
                frame=frame,
                position=position,
                previous_position=previous_position
            )
            
            snapshots.append(snapshot)
            hidden_mass_series.append(snapshot.hidden_mass)
            behaviors.append(snapshot.behavior)
            cohorts.append(snapshot.cohort_id)
            
            previous_position = position
        
        if not snapshots:
            logger.warning(f"No data for indicator {indicator_id}")
            return self._empty_dynamics(indicator_id, start_date, end_date)
        
        # Detect hidden mass events
        events = self._detect_hidden_mass_events(snapshots, hidden_mass_series)
        
        # Compute aggregates
        hidden_masses = [s.hidden_mass for s in snapshots]
        distances = [s.distance_from_centroid for s in snapshots]
        
        # Position volatility: std of position changes
        if len(snapshots) > 1:
            position_changes = []
            for i in range(1, len(snapshots)):
                p1 = snapshots[i-1].position
                p2 = snapshots[i].position
                min_dim = min(len(p1), len(p2))
                position_changes.append(np.linalg.norm(p2[:min_dim] - p1[:min_dim]))
            position_volatility = np.std(position_changes) if position_changes else 0
        else:
            position_volatility = 0
        
        # Cohort changes
        cohort_changes = sum(1 for i in range(1, len(cohorts)) if cohorts[i] != cohorts[i-1])
        
        # Behavior profile
        n = len(behaviors)
        behavior_counts = {b: behaviors.count(b) for b in ['normal', 'drifting', 'stressed', 'anomalous']}
        
        return IndicatorDynamics(
            indicator_id=indicator_id,
            start_date=start_date,
            end_date=end_date,
            snapshots=snapshots,
            mean_hidden_mass=np.mean(hidden_masses),
            max_hidden_mass=np.max(hidden_masses),
            total_hidden_mass_accumulation=np.sum(hidden_masses),
            hidden_mass_events=events,
            n_anomalous_frames=behavior_counts.get('anomalous', 0),
            mean_distance_from_centroid=np.mean(distances),
            position_volatility=position_volatility,
            cohort_changes=cohort_changes,
            pct_normal=100 * behavior_counts.get('normal', 0) / n,
            pct_drifting=100 * behavior_counts.get('drifting', 0) / n,
            pct_stressed=100 * behavior_counts.get('stressed', 0) / n,
            pct_anomalous=100 * behavior_counts.get('anomalous', 0) / n,
        )
    
    def _compute_snapshot(
        self,
        indicator_id: str,
        timestamp: datetime,
        frame: Dict,
        position: np.ndarray,
        previous_position: Optional[np.ndarray]
    ) -> IndicatorSnapshot:
        """Compute single snapshot for indicator."""
        
        # Get all positions for this frame
        all_positions = frame.get('positions', {})
        all_positions_matrix = np.array([
            np.array(p) for p in all_positions.values()
        ])
        
        # Centroid
        if len(all_positions_matrix) > 0:
            centroid = np.mean(all_positions_matrix, axis=0)
            min_dim = min(len(position), len(centroid))
            distance_from_centroid = np.linalg.norm(position[:min_dim] - centroid[:min_dim])
            
            # Position percentile
            all_distances = [
                np.linalg.norm(np.array(p)[:min_dim] - centroid[:min_dim]) 
                for p in all_positions.values()
            ]
            position_percentile = 100 * np.mean([d <= distance_from_centroid for d in all_distances])
        else:
            centroid = np.zeros_like(position)
            distance_from_centroid = 0
            position_percentile = 50
        
        # Velocity
        if previous_position is not None:
            min_dim = min(len(position), len(previous_position))
            velocity = position[:min_dim] - previous_position[:min_dim]
        else:
            velocity = np.zeros_like(position)
        
        # Expected velocity based on position
        # Indicators far from center are expected to move toward center (mean reversion)
        # Indicators near center are expected to move slowly
        expected_velocity = self._compute_expected_velocity(
            position, centroid, distance_from_centroid, frame
        )
        
        # Residual
        min_dim = min(len(velocity), len(expected_velocity))
        residual_velocity = velocity[:min_dim] - expected_velocity[:min_dim]
        
        # Mass decomposition
        total_mass = np.linalg.norm(velocity)
        explained_mass = np.linalg.norm(expected_velocity[:min_dim])
        hidden_mass = np.linalg.norm(residual_velocity)
        hidden_mass_ratio = hidden_mass / max(total_mass, 1e-10)
        
        # System context
        system_tension = frame.get('systemic_tension', 0)
        
        # Cohort
        cohort_assignments = frame.get('cohort_assignments', {})
        cohort_id = cohort_assignments.get(indicator_id, 'unknown')
        
        # Cohort tension
        cohort_tension = self._compute_cohort_tension(
            indicator_id, cohort_id, frame
        )
        
        # Classify behavior
        behavior = self._classify_behavior(
            hidden_mass, hidden_mass_ratio, distance_from_centroid, system_tension
        )
        
        return IndicatorSnapshot(
            timestamp=timestamp,
            indicator_id=indicator_id,
            position=position,
            distance_from_centroid=distance_from_centroid,
            position_percentile=position_percentile,
            velocity=velocity,
            expected_velocity=expected_velocity,
            residual_velocity=residual_velocity,
            total_mass=total_mass,
            explained_mass=explained_mass,
            hidden_mass=hidden_mass,
            hidden_mass_ratio=hidden_mass_ratio,
            system_tension=system_tension,
            cohort_id=cohort_id,
            cohort_tension=cohort_tension,
            behavior=behavior,
        )
    
    def _compute_expected_velocity(
        self,
        position: np.ndarray,
        centroid: np.ndarray,
        distance: float,
        frame: Dict
    ) -> np.ndarray:
        """
        Compute expected velocity based on position.
        
        Key insight: expected behavior depends on WHERE you are.
        
        - Near center: expect slow drift
        - Far from center: expect mean reversion (toward center)
        - High system tension: expect larger movements
        """
        min_dim = min(len(position), len(centroid))
        
        # Direction toward centroid
        to_center = centroid[:min_dim] - position[:min_dim]
        to_center_norm = np.linalg.norm(to_center)
        
        if to_center_norm < 1e-10:
            return np.zeros(min_dim)
        
        to_center_unit = to_center / to_center_norm
        
        # Mean reversion strength increases with distance
        # But also depends on system stability
        stability = frame.get('geometric_stability', 0.5)
        
        # Expected magnitude: distance * reversion_rate * stability_factor
        reversion_rate = 0.1  # 10% toward center per frame
        expected_magnitude = distance * reversion_rate * stability
        
        # Expected velocity points toward center with this magnitude
        expected_velocity = to_center_unit * expected_magnitude
        
        return expected_velocity
    
    def _compute_cohort_tension(
        self,
        indicator_id: str,
        cohort_id: str,
        frame: Dict
    ) -> float:
        """Compute tension within indicator's cohort."""
        
        cohort_tensions = frame.get('cohort_tension', {})
        return cohort_tensions.get(cohort_id, 0)
    
    def _classify_behavior(
        self,
        hidden_mass: float,
        hidden_mass_ratio: float,
        distance: float,
        system_tension: float
    ) -> str:
        """
        Classify indicator behavior.
        
        normal:    Low hidden mass, expected for position
        drifting:  Moderate hidden mass, slow drift from expected
        stressed:  High hidden mass OR high system tension
        anomalous: Very high hidden mass, unexpected motion
        """
        if hidden_mass_ratio < 0.2 and hidden_mass < 0.5:
            return 'normal'
        elif hidden_mass_ratio < 0.4 and hidden_mass < 1.0:
            return 'drifting'
        elif hidden_mass_ratio < 0.6 or system_tension > 0.5:
            return 'stressed'
        else:
            return 'anomalous'
    
    def _detect_hidden_mass_events(
        self,
        snapshots: List[IndicatorSnapshot],
        hidden_mass_series: List[float]
    ) -> List[HiddenMassEvent]:
        """
        Detect significant hidden mass events.
        
        An event is when hidden mass exceeds 2 sigma for multiple frames.
        """
        if len(hidden_mass_series) < 5:
            return []
        
        mean_hm = np.mean(hidden_mass_series)
        std_hm = np.std(hidden_mass_series)
        
        if std_hm < 1e-10:
            return []
        
        threshold = mean_hm + 2 * std_hm
        
        events = []
        in_event = False
        event_start = None
        event_snapshots = []
        
        for i, snapshot in enumerate(snapshots):
            if snapshot.hidden_mass > threshold:
                if not in_event:
                    in_event = True
                    event_start = i
                    event_snapshots = [snapshot]
                else:
                    event_snapshots.append(snapshot)
            else:
                if in_event:
                    # Event ended, record it
                    event = self._create_event(event_snapshots, mean_hm, std_hm)
                    events.append(event)
                    in_event = False
                    event_snapshots = []
        
        # Handle event that extends to end
        if in_event and event_snapshots:
            event = self._create_event(event_snapshots, mean_hm, std_hm)
            events.append(event)
        
        return events
    
    def _create_event(
        self,
        snapshots: List[IndicatorSnapshot],
        mean_hm: float,
        std_hm: float
    ) -> HiddenMassEvent:
        """Create a hidden mass event from snapshots."""
        
        peak_snapshot = max(snapshots, key=lambda s: s.hidden_mass)
        
        # Z-score
        z_score = (peak_snapshot.hidden_mass - mean_hm) / max(std_hm, 1e-10)
        
        # Direction
        direction = peak_snapshot.residual_velocity
        
        # Dominant dimension
        if len(direction) > 0:
            dominant_idx = np.argmax(np.abs(direction))
            dominant_dimension = f"PC{dominant_idx + 1}"
        else:
            dominant_dimension = "unknown"
        
        # Interpretation
        interpretation = self._interpret_event(
            peak_snapshot, z_score, dominant_dimension
        )
        
        # System status at event
        if peak_snapshot.system_tension > 0.6:
            system_status = "CRITICAL"
        elif peak_snapshot.system_tension > 0.3:
            system_status = "STRESSED"
        else:
            system_status = "STABLE"
        
        return HiddenMassEvent(
            timestamp=peak_snapshot.timestamp,
            indicator_id=peak_snapshot.indicator_id,
            hidden_mass=peak_snapshot.hidden_mass,
            z_score=z_score,
            direction=direction,
            dominant_dimension=dominant_dimension,
            system_status=system_status,
            preceding_tension=peak_snapshot.system_tension,
            interpretation=interpretation,
            duration_frames=len(snapshots),
            peak_magnitude=peak_snapshot.hidden_mass,
        )
    
    def _interpret_event(
        self,
        snapshot: IndicatorSnapshot,
        z_score: float,
        dominant_dimension: str
    ) -> str:
        """Generate human-readable interpretation of event."""
        
        severity = "extreme" if z_score > 3 else "significant" if z_score > 2 else "moderate"
        
        if snapshot.distance_from_centroid > 1.5:
            position_desc = "peripheral"
        elif snapshot.distance_from_centroid > 0.5:
            position_desc = "intermediate"
        else:
            position_desc = "central"
        
        motion_type = "unexplained motion" if snapshot.hidden_mass_ratio > 0.5 else "unexpected direction"
        
        return f"{severity.capitalize()} {motion_type} for {position_desc} indicator ({dominant_dimension} dominant, z={z_score:.1f})"
    
    def _empty_dynamics(
        self,
        indicator_id: str,
        start_date: datetime,
        end_date: datetime
    ) -> IndicatorDynamics:
        """Return empty dynamics for missing data."""
        return IndicatorDynamics(
            indicator_id=indicator_id,
            start_date=start_date,
            end_date=end_date,
            snapshots=[],
            mean_hidden_mass=0,
            max_hidden_mass=0,
            total_hidden_mass_accumulation=0,
            hidden_mass_events=[],
            n_anomalous_frames=0,
            mean_distance_from_centroid=0,
            position_volatility=0,
            cohort_changes=0,
            pct_normal=0,
            pct_drifting=0,
            pct_stressed=0,
            pct_anomalous=0,
        )
    
    def compute_all_indicators(
        self,
        start_date: datetime,
        end_date: datetime,
        geometry_frames: List[Dict] = None
    ) -> SystemIndicatorMatrix:
        """
        Compute dynamics for all indicators.
        
        Returns full system picture with cross-indicator analysis.
        """
        frames = geometry_frames or self._geometry_frames
        
        if not frames:
            raise ValueError("No geometry frames")
        
        # Get all indicator IDs
        all_indicators = set()
        for frame in frames:
            all_indicators.update(frame.get('positions', {}).keys())
        
        logger.info(f"Computing dynamics for {len(all_indicators)} indicators")
        
        # Compute per-indicator dynamics
        indicator_dynamics = {}
        for ind_id in all_indicators:
            dynamics = self.compute_indicator_dynamics(
                indicator_id=ind_id,
                start_date=start_date,
                end_date=end_date,
                geometry_frames=frames
            )
            indicator_dynamics[ind_id] = dynamics
        
        # System-level time series
        system_tension_series = [
            (datetime.fromisoformat(f['window_center']), f.get('systemic_tension', 0))
            for f in frames
        ]
        
        system_hidden_mass_series = [
            (datetime.fromisoformat(f['window_center']), f.get('total_hidden_mass', 0))
            for f in frames
        ]
        
        # Rankings
        by_hidden_mass = sorted(
            indicator_dynamics.items(),
            key=lambda x: x[1].total_hidden_mass_accumulation,
            reverse=True
        )
        most_hidden_mass = [ind_id for ind_id, _ in by_hidden_mass[:10]]
        
        by_volatility = sorted(
            indicator_dynamics.items(),
            key=lambda x: x[1].position_volatility,
            reverse=True
        )
        most_volatile = [ind_id for ind_id, _ in by_volatility[:10]]
        
        by_stability = sorted(
            indicator_dynamics.items(),
            key=lambda x: x[1].mean_hidden_mass
        )
        most_stable = [ind_id for ind_id, _ in by_stability[:10]]
        
        # TODO: Hidden mass correlations and cascade detection
        # (Requires more sophisticated cross-indicator analysis)
        
        return SystemIndicatorMatrix(
            start_date=start_date,
            end_date=end_date,
            n_frames=len(frames),
            n_indicators=len(all_indicators),
            indicators=indicator_dynamics,
            system_tension_series=system_tension_series,
            system_hidden_mass_series=system_hidden_mass_series,
            hidden_mass_correlations={},
            cascade_events=[],
            most_hidden_mass=most_hidden_mass,
            most_volatile=most_volatile,
            most_stable=most_stable,
        )


# =============================================================================
# VISUALIZATION HELPERS
# =============================================================================

def plot_indicator_dynamics(dynamics: IndicatorDynamics, output_path: str = None):
    """Plot indicator dynamics over time."""
    
    import matplotlib.pyplot as plt
    
    if not dynamics.snapshots:
        print("No data to plot")
        return
    
    dates = [s.timestamp for s in dynamics.snapshots]
    hidden_mass = [s.hidden_mass for s in dynamics.snapshots]
    distance = [s.distance_from_centroid for s in dynamics.snapshots]
    system_tension = [s.system_tension for s in dynamics.snapshots]
    
    # Behavior colors
    behavior_colors = {
        'normal': 'green',
        'drifting': 'yellow',
        'stressed': 'orange',
        'anomalous': 'red'
    }
    colors = [behavior_colors[s.behavior] for s in dynamics.snapshots]
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    
    # Hidden mass
    axes[0].scatter(dates, hidden_mass, c=colors, s=20, alpha=0.7)
    axes[0].plot(dates, hidden_mass, 'b-', alpha=0.3)
    axes[0].axhline(y=dynamics.mean_hidden_mass, color='gray', linestyle='--', alpha=0.5)
    axes[0].set_ylabel('Hidden Mass')
    axes[0].set_title(f'Indicator Dynamics: {dynamics.indicator_id}')
    
    # Mark events
    for event in dynamics.hidden_mass_events:
        axes[0].axvline(x=event.timestamp, color='red', alpha=0.3)
    
    # Distance from centroid
    axes[1].plot(dates, distance, 'purple', linewidth=1.5)
    axes[1].fill_between(dates, distance, alpha=0.3, color='purple')
    axes[1].set_ylabel('Distance from Centroid')
    
    # System tension (context)
    axes[2].plot(dates, system_tension, 'brown', linewidth=1.5)
    axes[2].fill_between(dates, system_tension, alpha=0.3, color='brown')
    axes[2].axhline(y=0.5, color='orange', linestyle='--', alpha=0.5)
    axes[2].set_ylabel('System Tension')
    axes[2].set_xlabel('Date')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {output_path}")
    else:
        plt.show()
    
    plt.close()


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="PRISM Indicator Dynamics Tracker")
    
    parser.add_argument('--db', type=str, default='prism.db', help='Database path')
    parser.add_argument('--geometry', type=str, help='Pre-computed geometry JSON')
    parser.add_argument('--indicator', type=str, help='Specific indicator to analyze')
    parser.add_argument('--start', type=str, required=True, help='Start date')
    parser.add_argument('--end', type=str, required=True, help='End date')
    parser.add_argument('--output', type=str, help='Output file path')
    parser.add_argument('--plot', action='store_true', help='Generate plot')
    
    args = parser.parse_args()
    
    start_date = datetime.strptime(args.start, '%Y-%m-%d')
    end_date = datetime.strptime(args.end, '%Y-%m-%d')
    
    tracker = IndicatorDynamicsTracker(args.db)
    
    # Load geometry
    if args.geometry:
        tracker.load_geometry_frames(args.geometry)
    else:
        tracker.load_geometry_frames(start_date=start_date, end_date=end_date)
    
    if args.indicator:
        # Single indicator
        dynamics = tracker.compute_indicator_dynamics(
            args.indicator, start_date, end_date
        )
        print(dynamics.summary())
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(dynamics.to_dict(), f, indent=2)
        
        if args.plot:
            plot_path = args.output.replace('.json', '.png') if args.output else f'{args.indicator}_dynamics.png'
            plot_indicator_dynamics(dynamics, plot_path)
    
    else:
        # All indicators
        matrix = tracker.compute_all_indicators(start_date, end_date)
        
        print(f"\n=== System Indicator Matrix ===")
        print(f"Period: {start_date.date()} to {end_date.date()}")
        print(f"Frames: {matrix.n_frames}")
        print(f"Indicators: {matrix.n_indicators}")
        print(f"\nMost Hidden Mass Accumulation:")
        for ind in matrix.most_hidden_mass[:5]:
            d = matrix.indicators[ind]
            print(f"  {ind}: {d.total_hidden_mass_accumulation:.2f}")
        
        print(f"\nMost Volatile (position changes):")
        for ind in matrix.most_volatile[:5]:
            d = matrix.indicators[ind]
            print(f"  {ind}: volatility={d.position_volatility:.4f}")
        
        print(f"\nMost Stable (lowest hidden mass):")
        for ind in matrix.most_stable[:5]:
            d = matrix.indicators[ind]
            print(f"  {ind}: mean_hm={d.mean_hidden_mass:.4f}")
