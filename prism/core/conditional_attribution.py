"""
Conditional Attribution - Which indicators lead excursions?

Answers:
- When D > threshold and velocity positive, which indicators move first?
- What is the effect size and lead time for each indicator?

Usage:
    from prism.core import ConditionalAttribution, AttributionReport

    attr = ConditionalAttribution(distance_threshold=1.5, velocity_threshold=0.05)
    report = attr.compute_attribution(
        distances=distance_series,
        velocities=velocity_series,
        indicators={"vix": vix_series, "credit_spread": spread_series}
    )

    print(f"Top leading indicators: {report.top_leading_indicators}")
    for ind in report.by_effect_size[:5]:
        print(f"  {ind.indicator_name}: effect={ind.effect_size:.2f}")
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class IndicatorAttribution:
    """Attribution for a single indicator."""
    indicator_name: str

    # Effect size
    mean_change: float  # E[ΔX | condition]
    std_change: float
    effect_size: float  # mean / std (standardized)

    # Lead time
    mean_lead_days: float  # how many days before condition was met
    median_lead_days: float

    # Stability
    bootstrap_ci_low: float
    bootstrap_ci_high: float
    is_significant: bool  # CI doesn't cross zero

    # Ranking
    rank_effect_size: int
    rank_lead_time: int

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "indicator_name": self.indicator_name,
            "mean_change": self.mean_change,
            "std_change": self.std_change,
            "effect_size": self.effect_size,
            "mean_lead_days": self.mean_lead_days,
            "median_lead_days": self.median_lead_days,
            "bootstrap_ci_low": self.bootstrap_ci_low,
            "bootstrap_ci_high": self.bootstrap_ci_high,
            "is_significant": self.is_significant,
            "rank_effect_size": self.rank_effect_size,
            "rank_lead_time": self.rank_lead_time,
        }


@dataclass
class AttributionReport:
    """Full attribution report for a condition."""
    condition_description: str  # e.g., "D > 1.5 and v > 0"
    n_episodes: int

    # Ranked indicators
    by_effect_size: List[IndicatorAttribution]
    by_lead_time: List[IndicatorAttribution]

    # Top movers
    top_leading_indicators: List[str]
    top_effect_indicators: List[str]

    @property
    def has_data(self) -> bool:
        """True if report has attribution data."""
        return self.n_episodes > 0 and len(self.by_effect_size) > 0

    @property
    def n_significant(self) -> int:
        """Number of indicators with significant effect."""
        return sum(1 for a in self.by_effect_size if a.is_significant)

    def get_indicator(self, name: str) -> Optional[IndicatorAttribution]:
        """Get attribution for specific indicator."""
        for a in self.by_effect_size:
            if a.indicator_name == name:
                return a
        return None

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "condition_description": self.condition_description,
            "n_episodes": self.n_episodes,
            "by_effect_size": [a.to_dict() for a in self.by_effect_size],
            "by_lead_time": [a.to_dict() for a in self.by_lead_time],
            "top_leading_indicators": self.top_leading_indicators,
            "top_effect_indicators": self.top_effect_indicators,
            "n_significant": self.n_significant,
        }


# =============================================================================
# Main Class
# =============================================================================

class ConditionalAttribution:
    """
    Compute which indicators lead excursions.

    Identifies:
    - Which indicators move first before excursion episodes
    - Effect size (standardized change) of each indicator
    - Lead time (how early the indicator signals)

    Usage:
        attr = ConditionalAttribution(distance_threshold=1.5)
        report = attr.compute_attribution(distances, velocities, indicators)
    """

    def __init__(
        self,
        distance_threshold: float = 1.5,
        velocity_threshold: float = 0.05,
        horizon: int = 20,
        lead_window: int = 10,
    ):
        """
        Initialize conditional attribution.

        Args:
            distance_threshold: D threshold for excursion condition
            velocity_threshold: v threshold for escalation condition
            horizon: Days to measure indicator change over
            lead_window: Days before condition to look for early signals
        """
        self.distance_threshold = distance_threshold
        self.velocity_threshold = velocity_threshold
        self.horizon = horizon
        self.lead_window = lead_window

    def compute_attribution(
        self,
        distances: np.ndarray,
        velocities: np.ndarray,
        indicators: Dict[str, np.ndarray],
        timestamps: Optional[List[str]] = None,
        bootstrap_n: int = 100,
    ) -> AttributionReport:
        """
        Compute indicator attribution for escalation episodes.

        Args:
            distances: D(t) series
            velocities: v(t) series
            indicators: dict of indicator_name -> values
            timestamps: optional timestamps
            bootstrap_n: bootstrap samples for CI

        Returns:
            AttributionReport with ranked indicators
        """
        distances = np.array(distances)
        velocities = np.array(velocities)

        # Validate lengths
        n = len(distances)
        if len(velocities) != n:
            raise ValueError("distances and velocities must have same length")
        for name, vals in indicators.items():
            if len(vals) != n:
                raise ValueError(f"Indicator {name} has wrong length: {len(vals)} vs {n}")

        # Find escalation episodes
        condition = (
            (distances > self.distance_threshold) &
            (velocities > self.velocity_threshold)
        )
        episode_starts = self._find_episode_starts(condition)

        if len(episode_starts) < 5:
            return self._empty_report("Insufficient episodes")

        # Compute attribution for each indicator
        attributions = []
        for name, values in indicators.items():
            attr = self._compute_single_attribution(
                name, np.array(values), episode_starts, bootstrap_n
            )
            attributions.append(attr)

        # Rank by effect size
        by_effect = sorted(
            attributions, key=lambda x: abs(x.effect_size), reverse=True
        )
        for i, a in enumerate(by_effect):
            a.rank_effect_size = i + 1

        # Rank by lead time (lower is better - earlier signal)
        by_lead = sorted(
            attributions, key=lambda x: -x.mean_lead_days  # negative so larger lead = better
        )
        for i, a in enumerate(by_lead):
            a.rank_lead_time = i + 1

        return AttributionReport(
            condition_description=f"D > {self.distance_threshold} and v > {self.velocity_threshold}",
            n_episodes=len(episode_starts),
            by_effect_size=by_effect,
            by_lead_time=by_lead,
            top_leading_indicators=[a.indicator_name for a in by_lead[:5]],
            top_effect_indicators=[a.indicator_name for a in by_effect[:5]],
        )

    def compute_attribution_for_condition(
        self,
        condition: np.ndarray,
        indicators: Dict[str, np.ndarray],
        condition_description: str = "custom",
        bootstrap_n: int = 100,
    ) -> AttributionReport:
        """
        Compute attribution for a custom boolean condition.

        Args:
            condition: Boolean array indicating when condition is met
            indicators: dict of indicator_name -> values
            condition_description: Description for the report
            bootstrap_n: bootstrap samples for CI

        Returns:
            AttributionReport with ranked indicators
        """
        condition = np.array(condition)
        episode_starts = self._find_episode_starts(condition)

        if len(episode_starts) < 5:
            return self._empty_report("Insufficient episodes")

        attributions = []
        for name, values in indicators.items():
            attr = self._compute_single_attribution(
                name, np.array(values), episode_starts, bootstrap_n
            )
            attributions.append(attr)

        by_effect = sorted(
            attributions, key=lambda x: abs(x.effect_size), reverse=True
        )
        for i, a in enumerate(by_effect):
            a.rank_effect_size = i + 1

        by_lead = sorted(
            attributions, key=lambda x: -x.mean_lead_days
        )
        for i, a in enumerate(by_lead):
            a.rank_lead_time = i + 1

        return AttributionReport(
            condition_description=condition_description,
            n_episodes=len(episode_starts),
            by_effect_size=by_effect,
            by_lead_time=by_lead,
            top_leading_indicators=[a.indicator_name for a in by_lead[:5]],
            top_effect_indicators=[a.indicator_name for a in by_effect[:5]],
        )

    # =========================================================================
    # Private Methods
    # =========================================================================

    def _find_episode_starts(self, condition: np.ndarray) -> List[int]:
        """Find starts of condition episodes."""
        starts = []
        in_episode = False
        for i, c in enumerate(condition):
            if c and not in_episode:
                starts.append(i)
                in_episode = True
            elif not c:
                in_episode = False
        return starts

    def _compute_single_attribution(
        self,
        name: str,
        values: np.ndarray,
        episode_starts: List[int],
        bootstrap_n: int,
    ) -> IndicatorAttribution:
        """Compute attribution for single indicator."""
        changes = []
        lead_times = []

        for start in episode_starts:
            if start < self.lead_window:
                continue
            if start + self.horizon >= len(values):
                continue

            # Change over horizon
            change = values[start + self.horizon] - values[start]
            changes.append(change)

            # Lead time: when did indicator start moving?
            pre_window = values[start - self.lead_window:start]
            if len(pre_window) == 0:
                lead_times.append(0)
                continue

            threshold = np.std(pre_window) * 0.5 if np.std(pre_window) > 0 else 0.01

            lead = 0  # default: no early signal
            for i, v in enumerate(pre_window):
                if abs(v - pre_window[0]) > threshold:
                    lead = self.lead_window - i
                    break
            lead_times.append(lead)

        if not changes:
            return self._empty_attribution(name)

        mean_change = float(np.mean(changes))
        std_change = float(np.std(changes))
        effect_size = mean_change / (std_change + 1e-10)

        # Bootstrap CI for effect size
        boot_effects = []
        for _ in range(bootstrap_n):
            idx = np.random.choice(len(changes), len(changes), replace=True)
            boot_changes = [changes[i] for i in idx]
            boot_mean = np.mean(boot_changes)
            boot_std = np.std(boot_changes)
            boot_effects.append(boot_mean / (boot_std + 1e-10))

        ci = np.percentile(boot_effects, [2.5, 97.5]) if boot_effects else [0, 0]
        significant = (ci[0] > 0) or (ci[1] < 0)

        return IndicatorAttribution(
            indicator_name=name,
            mean_change=mean_change,
            std_change=std_change,
            effect_size=float(effect_size),
            mean_lead_days=float(np.mean(lead_times)) if lead_times else 0.0,
            median_lead_days=float(np.median(lead_times)) if lead_times else 0.0,
            bootstrap_ci_low=float(ci[0]),
            bootstrap_ci_high=float(ci[1]),
            is_significant=bool(significant),
            rank_effect_size=0,
            rank_lead_time=0,
        )

    def _empty_attribution(self, name: str) -> IndicatorAttribution:
        """Return empty attribution for indicator with no data."""
        return IndicatorAttribution(
            indicator_name=name,
            mean_change=0.0,
            std_change=0.0,
            effect_size=0.0,
            mean_lead_days=0.0,
            median_lead_days=0.0,
            bootstrap_ci_low=0.0,
            bootstrap_ci_high=0.0,
            is_significant=False,
            rank_effect_size=999,
            rank_lead_time=999,
        )

    def _empty_report(self, reason: str) -> AttributionReport:
        """Return empty report when insufficient data."""
        return AttributionReport(
            condition_description=reason,
            n_episodes=0,
            by_effect_size=[],
            by_lead_time=[],
            top_leading_indicators=[],
            top_effect_indicators=[],
        )


# =============================================================================
# Convenience Functions
# =============================================================================

def compute_indicator_lead(
    indicator: np.ndarray,
    condition: np.ndarray,
    lead_window: int = 10,
) -> float:
    """
    Compute average lead time for an indicator before condition.

    Args:
        indicator: Indicator values
        condition: Boolean condition array
        lead_window: Window to look back for early signals

    Returns:
        Average lead time in periods
    """
    episode_starts = []
    in_episode = False
    for i, c in enumerate(condition):
        if c and not in_episode:
            episode_starts.append(i)
            in_episode = True
        elif not c:
            in_episode = False

    lead_times = []
    for start in episode_starts:
        if start < lead_window:
            continue

        pre_window = indicator[start - lead_window:start]
        if len(pre_window) == 0:
            continue

        threshold = np.std(pre_window) * 0.5 if np.std(pre_window) > 0 else 0.01

        for i, v in enumerate(pre_window):
            if abs(v - pre_window[0]) > threshold:
                lead_times.append(lead_window - i)
                break

    return float(np.mean(lead_times)) if lead_times else 0.0


def rank_indicators_by_effect(
    indicators: Dict[str, np.ndarray],
    condition: np.ndarray,
    horizon: int = 20,
) -> List[str]:
    """
    Rank indicators by effect size during condition episodes.

    Args:
        indicators: Dict of indicator name -> values
        condition: Boolean condition array
        horizon: Periods to measure change over

    Returns:
        List of indicator names ranked by absolute effect size
    """
    episode_starts = []
    in_episode = False
    for i, c in enumerate(condition):
        if c and not in_episode:
            episode_starts.append(i)
            in_episode = True
        elif not c:
            in_episode = False

    effects = {}
    for name, values in indicators.items():
        changes = []
        for start in episode_starts:
            if start + horizon < len(values):
                changes.append(values[start + horizon] - values[start])

        if changes:
            mean_change = np.mean(changes)
            std_change = np.std(changes) + 1e-10
            effects[name] = abs(mean_change / std_change)
        else:
            effects[name] = 0.0

    return sorted(effects.keys(), key=lambda x: effects[x], reverse=True)
