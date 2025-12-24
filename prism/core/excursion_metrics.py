"""
Excursion Metrics - Time outside homeostatic band.

Tracks:
    - Streak lengths (consecutive days outside)
    - Churn rate (entries/exits per period)
    - Dwell time distribution
    - Time since last homeostasis

Usage:
    from prism.core import ExcursionTracker, ExcursionSummary

    tracker = ExcursionTracker(lookback_days=252)
    for ts, dist, inside in observations:
        tracker.update(ts, dist, inside)

    summary = tracker.compute_summary()
    print(f"Currently outside: {summary.currently_outside}")
    print(f"Current streak: {summary.current_streak_length} days")
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ExcursionSummary:
    """Summary of excursion behavior over a period."""

    # Current state
    currently_outside: bool
    current_distance: float
    current_streak_length: int  # days in current state
    time_since_homeostasis: int  # days since last inside band

    # Historical metrics (over lookback window)
    total_days_outside: int
    pct_time_outside: float
    longest_streak_outside: int

    # Churn
    n_entries: int  # crossed from inside to outside
    n_exits: int  # crossed from outside to inside
    churn_rate: float  # (entries + exits) per 90 days

    # Baseline comparison
    churn_rate_baseline: float
    churn_rate_ratio: float  # current / baseline

    # Dwell distribution
    mean_excursion_length: float
    median_excursion_length: float
    excursion_length_90pct: float  # 90th percentile streak

    @property
    def is_elevated_churn(self) -> bool:
        """True if churn rate is above baseline."""
        return self.churn_rate_ratio > 1.5

    @property
    def is_persistent_excursion(self) -> bool:
        """True if current excursion is unusually long."""
        return (
            self.currently_outside and
            self.current_streak_length > self.excursion_length_90pct
        )

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "currently_outside": self.currently_outside,
            "current_distance": self.current_distance,
            "current_streak_length": self.current_streak_length,
            "time_since_homeostasis": self.time_since_homeostasis,
            "total_days_outside": self.total_days_outside,
            "pct_time_outside": self.pct_time_outside,
            "longest_streak_outside": self.longest_streak_outside,
            "n_entries": self.n_entries,
            "n_exits": self.n_exits,
            "churn_rate": self.churn_rate,
            "churn_rate_baseline": self.churn_rate_baseline,
            "churn_rate_ratio": self.churn_rate_ratio,
            "mean_excursion_length": self.mean_excursion_length,
            "median_excursion_length": self.median_excursion_length,
            "excursion_length_90pct": self.excursion_length_90pct,
            "is_elevated_churn": self.is_elevated_churn,
            "is_persistent_excursion": self.is_persistent_excursion,
        }


@dataclass
class ExcursionEvent:
    """A single excursion event (period outside band)."""
    start_idx: int
    end_idx: int
    length: int
    max_distance: float
    mean_distance: float

    @property
    def is_significant(self) -> bool:
        """True if excursion lasted > 5 days."""
        return self.length > 5


# =============================================================================
# Main Class
# =============================================================================

class ExcursionTracker:
    """
    Track excursions from homeostatic band.

    Input: series of (timestamp, distance, inside_band)
    Output: ExcursionSummary

    Usage:
        tracker = ExcursionTracker(lookback_days=252)
        for ts, dist, inside in observations:
            tracker.update(ts, dist, inside)
        summary = tracker.compute_summary()
    """

    def __init__(self, lookback_days: int = 252):
        """
        Initialize tracker.

        Args:
            lookback_days: Window for computing statistics (default 252)
        """
        self.lookback_days = lookback_days
        self._distances: List[float] = []
        self._inside: List[bool] = []
        self._timestamps: List[str] = []

    def update(
        self,
        timestamp: str,
        distance: float,
        inside_band: bool,
    ) -> None:
        """
        Add new observation.

        Args:
            timestamp: Observation timestamp
            distance: Distance from band center
            inside_band: Whether inside the homeostatic band
        """
        self._timestamps.append(timestamp)
        self._distances.append(distance)
        self._inside.append(inside_band)

    def update_batch(
        self,
        timestamps: List[str],
        distances: List[float],
        inside_band: List[bool],
    ) -> None:
        """Add multiple observations at once."""
        for ts, d, inside in zip(timestamps, distances, inside_band):
            self.update(ts, d, inside)

    def compute_summary(
        self,
        baseline_churn: Optional[float] = None,
    ) -> ExcursionSummary:
        """
        Compute excursion summary over lookback window.

        Args:
            baseline_churn: Optional baseline churn rate for comparison

        Returns:
            ExcursionSummary with all metrics
        """
        if len(self._inside) < 2:
            raise ValueError("Need at least 2 observations")

        # Recent window
        window = self._inside[-self.lookback_days:]
        distances = self._distances[-self.lookback_days:]

        inside_arr = np.array(window)
        outside_arr = ~inside_arr

        # Current state
        currently_outside = bool(outside_arr[-1])
        current_distance = distances[-1]

        # Current streak
        current_streak = self._current_streak(outside_arr)

        # Time since homeostasis
        if currently_outside:
            time_since = self._time_since_last_true(inside_arr)
        else:
            time_since = 0

        # Total time outside
        total_outside = int(outside_arr.sum())
        pct_outside = total_outside / len(window)

        # Longest streak
        longest = self._longest_streak(outside_arr)

        # Churn (entries/exits)
        entries, exits = self._count_transitions(inside_arr)
        churn = (entries + exits) / (len(window) / 90)  # per 90 days

        # Baseline
        if baseline_churn is None:
            baseline_churn = self._estimate_baseline_churn()
        churn_ratio = churn / baseline_churn if baseline_churn > 0 else 1.0

        # Dwell distribution
        excursion_lengths = self._get_excursion_lengths(outside_arr)

        return ExcursionSummary(
            currently_outside=currently_outside,
            current_distance=current_distance,
            current_streak_length=current_streak,
            time_since_homeostasis=time_since,
            total_days_outside=total_outside,
            pct_time_outside=pct_outside,
            longest_streak_outside=longest,
            n_entries=entries,
            n_exits=exits,
            churn_rate=churn,
            churn_rate_baseline=baseline_churn,
            churn_rate_ratio=churn_ratio,
            mean_excursion_length=float(np.mean(excursion_lengths)) if excursion_lengths else 0,
            median_excursion_length=float(np.median(excursion_lengths)) if excursion_lengths else 0,
            excursion_length_90pct=float(np.percentile(excursion_lengths, 90)) if excursion_lengths else 0,
        )

    def get_excursion_events(self) -> List[ExcursionEvent]:
        """
        Get list of all excursion events (periods outside band).

        Returns:
            List of ExcursionEvent objects
        """
        if len(self._inside) < 2:
            return []

        events = []
        outside_arr = np.array([not x for x in self._inside])
        distances = np.array(self._distances)

        in_excursion = False
        start_idx = 0

        for i, is_outside in enumerate(outside_arr):
            if is_outside and not in_excursion:
                # Start of excursion
                in_excursion = True
                start_idx = i
            elif not is_outside and in_excursion:
                # End of excursion
                in_excursion = False
                excursion_distances = distances[start_idx:i]
                events.append(ExcursionEvent(
                    start_idx=start_idx,
                    end_idx=i - 1,
                    length=i - start_idx,
                    max_distance=float(np.max(excursion_distances)),
                    mean_distance=float(np.mean(excursion_distances)),
                ))

        # Handle ongoing excursion
        if in_excursion:
            excursion_distances = distances[start_idx:]
            events.append(ExcursionEvent(
                start_idx=start_idx,
                end_idx=len(outside_arr) - 1,
                length=len(outside_arr) - start_idx,
                max_distance=float(np.max(excursion_distances)),
                mean_distance=float(np.mean(excursion_distances)),
            ))

        return events

    def clear(self) -> None:
        """Clear all stored observations."""
        self._distances = []
        self._inside = []
        self._timestamps = []

    @property
    def n_observations(self) -> int:
        """Number of observations stored."""
        return len(self._inside)

    # =========================================================================
    # Private Methods
    # =========================================================================

    def _current_streak(self, arr: np.ndarray) -> int:
        """Length of current streak (same value as last)."""
        if len(arr) == 0:
            return 0
        current = arr[-1]
        streak = 1
        for i in range(len(arr) - 2, -1, -1):
            if arr[i] == current:
                streak += 1
            else:
                break
        return streak

    def _time_since_last_true(self, arr: np.ndarray) -> int:
        """Time since last True value."""
        for i in range(len(arr) - 1, -1, -1):
            if arr[i]:
                return len(arr) - 1 - i
        return len(arr)

    def _longest_streak(self, arr: np.ndarray) -> int:
        """Longest consecutive True streak."""
        max_streak = 0
        current = 0
        for val in arr:
            if val:
                current += 1
                max_streak = max(max_streak, current)
            else:
                current = 0
        return max_streak

    def _count_transitions(
        self,
        inside_arr: np.ndarray,
    ) -> Tuple[int, int]:
        """Count entries (in→out) and exits (out→in)."""
        entries = 0
        exits = 0
        for i in range(1, len(inside_arr)):
            if inside_arr[i - 1] and not inside_arr[i]:
                entries += 1  # was inside, now outside
            elif not inside_arr[i - 1] and inside_arr[i]:
                exits += 1  # was outside, now inside
        return entries, exits

    def _get_excursion_lengths(self, outside_arr: np.ndarray) -> List[int]:
        """Get lengths of all excursion streaks."""
        lengths = []
        current = 0
        for val in outside_arr:
            if val:
                current += 1
            elif current > 0:
                lengths.append(current)
                current = 0
        if current > 0:
            lengths.append(current)
        return lengths

    def _estimate_baseline_churn(self) -> float:
        """Estimate baseline churn from full history."""
        if len(self._inside) < 90:
            return 4.0  # default: ~4 transitions per 90 days
        full_arr = np.array(self._inside)
        entries, exits = self._count_transitions(full_arr)
        return (entries + exits) / (len(full_arr) / 90)


# =============================================================================
# Convenience Functions
# =============================================================================

def compute_excursion_rate(
    inside_band: List[bool],
    lookback: int = 252,
) -> float:
    """
    Compute percentage of time outside band.

    Args:
        inside_band: List of inside/outside states
        lookback: Days to consider

    Returns:
        Fraction of time outside [0, 1]
    """
    recent = inside_band[-lookback:]
    return 1.0 - (sum(recent) / len(recent))


def count_boundary_crossings(
    inside_band: List[bool],
    lookback: int = 252,
) -> int:
    """
    Count number of times crossed the band boundary.

    Args:
        inside_band: List of inside/outside states
        lookback: Days to consider

    Returns:
        Total number of crossings
    """
    recent = inside_band[-lookback:]
    crossings = 0
    for i in range(1, len(recent)):
        if recent[i] != recent[i - 1]:
            crossings += 1
    return crossings
