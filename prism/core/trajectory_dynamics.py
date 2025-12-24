"""
Trajectory Dynamics - Velocity, acceleration, and hazard estimation.

Answers:
- Is the system moving toward or away from homeostasis?
- Is excursion reverting or escalating?
- What are the conditional hazards?

Usage:
    from prism.core import TrajectoryAnalyzer, TrajectoryState

    analyzer = TrajectoryAnalyzer(band_radius=1.5)
    for ts, distance in observations:
        analyzer.update(ts, distance)

    state = analyzer.compute_trajectory_state()
    print(f"Type: {state.trajectory_type}, Velocity: {state.velocity}")

    mr = analyzer.estimate_mean_reversion()
    print(f"Mean-reverting: {mr.is_mean_reverting}, β={mr.beta:.3f}")
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Literal
import numpy as np


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class TrajectoryState:
    """Current trajectory state at time t."""
    timestamp: str
    distance: float  # D(t)
    velocity: float  # v(t) = D(t) - D(t-1)
    acceleration: float  # a(t) = v(t) - v(t-1)

    # Classification (no labels, just mechanics)
    trajectory_type: Literal["reverting", "escalating", "stalling", "inside"]

    # Confidence
    velocity_significant: bool  # |v| > noise threshold
    acceleration_significant: bool

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "timestamp": self.timestamp,
            "distance": self.distance,
            "velocity": self.velocity,
            "acceleration": self.acceleration,
            "trajectory_type": self.trajectory_type,
            "velocity_significant": self.velocity_significant,
            "acceleration_significant": self.acceleration_significant,
        }


@dataclass
class MeanReversionEstimate:
    """Mean-reversion coefficient β and uncertainty."""
    beta: float  # ΔD = α + β*(D - r) + ε; β < 0 = mean-reverting
    beta_std: float  # bootstrap uncertainty
    beta_ci_low: float  # 95% CI
    beta_ci_high: float

    alpha: float  # intercept
    r_squared: float
    n_samples: int

    # Interpretation
    is_mean_reverting: bool  # β significantly < 0
    is_unstable: bool  # β significantly > 0
    pull_strength: float  # |β| when mean-reverting, 0 otherwise

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "beta": self.beta,
            "beta_std": self.beta_std,
            "beta_ci_low": self.beta_ci_low,
            "beta_ci_high": self.beta_ci_high,
            "alpha": self.alpha,
            "r_squared": self.r_squared,
            "n_samples": self.n_samples,
            "is_mean_reverting": self.is_mean_reverting,
            "is_unstable": self.is_unstable,
            "pull_strength": self.pull_strength,
        }


@dataclass
class HazardEstimates:
    """Conditional hazard estimates for re-entry vs escalation."""

    # Current level
    current_level: int  # L1, L2, L3 (distance bins)
    current_velocity_sign: int  # -1, 0, +1

    # Hazards (probabilities over horizon h)
    p_reentry_4wk: float  # P(re-enter band | current state)
    p_reentry_12wk: float
    p_escalate_4wk: float  # P(move to higher level | current state)
    p_escalate_12wk: float

    # Expected times
    expected_time_to_reentry: float  # days
    expected_time_to_escalation: float

    # Historical basis
    n_similar_episodes: int  # how many historical episodes inform this

    @property
    def reentry_more_likely_4wk(self) -> bool:
        """True if re-entry more likely than escalation at 4wk."""
        return self.p_reentry_4wk > self.p_escalate_4wk

    @property
    def reentry_more_likely_12wk(self) -> bool:
        """True if re-entry more likely than escalation at 12wk."""
        return self.p_reentry_12wk > self.p_escalate_12wk

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "current_level": self.current_level,
            "current_velocity_sign": self.current_velocity_sign,
            "p_reentry_4wk": self.p_reentry_4wk,
            "p_reentry_12wk": self.p_reentry_12wk,
            "p_escalate_4wk": self.p_escalate_4wk,
            "p_escalate_12wk": self.p_escalate_12wk,
            "expected_time_to_reentry": self.expected_time_to_reentry,
            "expected_time_to_escalation": self.expected_time_to_escalation,
            "n_similar_episodes": self.n_similar_episodes,
            "reentry_more_likely_4wk": self.reentry_more_likely_4wk,
            "reentry_more_likely_12wk": self.reentry_more_likely_12wk,
        }


@dataclass
class PullStrength:
    """Homeostatic pull strength at current distance."""
    distance: float
    expected_delta_d: float  # E[ΔD | D]
    pull_strength: float  # -expected_delta_d (positive = pulling back)
    half_life_estimate: float  # estimated days to halve distance
    confidence: float  # based on sample size at this distance

    @property
    def is_pulling_back(self) -> bool:
        """True if there's positive pull toward homeostasis."""
        return self.pull_strength > 0

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "distance": self.distance,
            "expected_delta_d": self.expected_delta_d,
            "pull_strength": self.pull_strength,
            "half_life_estimate": self.half_life_estimate,
            "confidence": self.confidence,
            "is_pulling_back": self.is_pulling_back,
        }


# =============================================================================
# Main Class
# =============================================================================

class TrajectoryAnalyzer:
    """
    Analyze trajectory dynamics of deformation.

    Computes:
    - Velocity and acceleration
    - Mean-reversion estimates
    - Conditional hazards
    - Pull strength

    Usage:
        analyzer = TrajectoryAnalyzer(band_radius=1.5)
        for ts, distance in observations:
            analyzer.update(ts, distance)

        state = analyzer.compute_trajectory_state()
        mr = analyzer.estimate_mean_reversion()
        hazards = analyzer.estimate_hazards()
    """

    def __init__(
        self,
        band_radius: float,
        level_thresholds: Tuple[float, float] = (1.0, 2.0),  # multiples of r
    ):
        """
        Initialize trajectory analyzer.

        Args:
            band_radius: Radius of homeostatic band (distance threshold)
            level_thresholds: Thresholds for escalation levels as multiples of r
                              L1: r to r*(1+t[0]), L2: r*(1+t[0]) to r*(1+t[1]), L3: > r*(1+t[1])
        """
        self.band_radius = band_radius
        self.level_thresholds = level_thresholds
        self._distance_history: List[float] = []
        self._timestamps: List[str] = []

    def update(self, timestamp: str, distance: float) -> None:
        """Add new distance observation."""
        self._timestamps.append(timestamp)
        self._distance_history.append(distance)

    def update_batch(
        self,
        timestamps: List[str],
        distances: List[float],
    ) -> None:
        """Add multiple observations at once."""
        for ts, d in zip(timestamps, distances):
            self.update(ts, d)

    def compute_trajectory_state(
        self,
        noise_threshold: float = 0.05,
    ) -> TrajectoryState:
        """
        Compute current trajectory state.

        Args:
            noise_threshold: Threshold for significant velocity/acceleration

        Returns:
            TrajectoryState with current dynamics
        """
        if len(self._distance_history) < 3:
            raise ValueError("Need at least 3 observations for trajectory")

        d = self._distance_history[-1]
        d_prev = self._distance_history[-2]
        d_prev2 = self._distance_history[-3]

        v = d - d_prev
        v_prev = d_prev - d_prev2
        a = v - v_prev

        # Classify trajectory
        inside = d <= self.band_radius
        if inside:
            ttype = "inside"
        elif v < -noise_threshold:
            ttype = "reverting"
        elif v > noise_threshold:
            ttype = "escalating"
        else:
            ttype = "stalling"

        return TrajectoryState(
            timestamp=self._timestamps[-1],
            distance=d,
            velocity=v,
            acceleration=a,
            trajectory_type=ttype,
            velocity_significant=abs(v) > noise_threshold,
            acceleration_significant=abs(a) > noise_threshold,
        )

    def estimate_mean_reversion(
        self,
        bootstrap_n: int = 200,
    ) -> MeanReversionEstimate:
        """
        Estimate mean-reversion coefficient β.

        Model: ΔD(t) = α + β * (D(t-1) - r) + ε
        β < 0: mean-reverting (pulled back to band)
        β ≈ 0: random walk (no pull)
        β > 0: unstable (tends to diverge)

        Args:
            bootstrap_n: Number of bootstrap samples for CI

        Returns:
            MeanReversionEstimate with β and uncertainty
        """
        if len(self._distance_history) < 20:
            raise ValueError("Need at least 20 observations")

        D = np.array(self._distance_history)
        delta_D = np.diff(D)
        D_lagged = D[:-1] - self.band_radius

        # OLS fit
        X = np.column_stack([np.ones(len(D_lagged)), D_lagged])
        y = delta_D

        coeffs, residuals, _, _ = np.linalg.lstsq(X, y, rcond=None)
        alpha, beta = coeffs

        y_pred = X @ coeffs
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        # Bootstrap CI
        betas = []
        for _ in range(bootstrap_n):
            idx = np.random.choice(len(y), len(y), replace=True)
            X_b, y_b = X[idx], y[idx]
            try:
                c, _, _, _ = np.linalg.lstsq(X_b, y_b, rcond=None)
                betas.append(c[1])
            except Exception:
                pass

        beta_std = float(np.std(betas)) if betas else 0.0
        beta_ci = np.percentile(betas, [2.5, 97.5]) if betas else [beta, beta]

        # Interpretation
        is_mr = beta_ci[1] < 0  # 97.5th percentile < 0
        is_unstable = beta_ci[0] > 0  # 2.5th percentile > 0
        pull = abs(beta) if is_mr else 0.0

        return MeanReversionEstimate(
            beta=float(beta),
            beta_std=beta_std,
            beta_ci_low=float(beta_ci[0]),
            beta_ci_high=float(beta_ci[1]),
            alpha=float(alpha),
            r_squared=float(r_squared),
            n_samples=len(y),
            is_mean_reverting=bool(is_mr),
            is_unstable=bool(is_unstable),
            pull_strength=float(pull),
        )

    def estimate_hazards(
        self,
        horizon_4wk: int = 20,
        horizon_12wk: int = 60,
    ) -> HazardEstimates:
        """
        Estimate conditional hazards for re-entry vs escalation.

        Based on historical episodes at similar distance/velocity.

        Args:
            horizon_4wk: Days for 4-week horizon
            horizon_12wk: Days for 12-week horizon

        Returns:
            HazardEstimates with probabilities and expected times
        """
        if len(self._distance_history) < 100:
            # Not enough history, return uninformative priors
            return self._default_hazards()

        D = np.array(self._distance_history)
        current_d = D[-1]
        current_v = D[-1] - D[-2] if len(D) > 1 else 0

        # Current level
        level = self._get_level(current_d)
        v_sign = int(np.sign(current_v))

        # Find similar historical episodes
        similar_idx = self._find_similar_episodes(D, current_d, current_v)

        if len(similar_idx) < 10:
            return self._default_hazards(level, v_sign)

        # Compute outcomes for similar episodes
        reentry_4wk = 0
        reentry_12wk = 0
        escalate_4wk = 0
        escalate_12wk = 0
        valid_count = 0

        for idx in similar_idx:
            if idx + horizon_12wk >= len(D):
                continue

            valid_count += 1

            # Did it re-enter within horizons?
            future_4wk = D[idx + 1:idx + horizon_4wk + 1]
            future_12wk = D[idx + 1:idx + horizon_12wk + 1]

            if np.any(future_4wk <= self.band_radius):
                reentry_4wk += 1
            if np.any(future_12wk <= self.band_radius):
                reentry_12wk += 1

            # Did it escalate?
            next_level_threshold = self._level_threshold(level + 1)
            if np.any(future_4wk >= next_level_threshold):
                escalate_4wk += 1
            if np.any(future_12wk >= next_level_threshold):
                escalate_12wk += 1

        n = max(valid_count, 1)

        return HazardEstimates(
            current_level=level,
            current_velocity_sign=v_sign,
            p_reentry_4wk=reentry_4wk / n,
            p_reentry_12wk=reentry_12wk / n,
            p_escalate_4wk=escalate_4wk / n,
            p_escalate_12wk=escalate_12wk / n,
            expected_time_to_reentry=self._expected_time_to_reentry(D, similar_idx),
            expected_time_to_escalation=self._expected_time_to_escalation(D, similar_idx, level),
            n_similar_episodes=valid_count,
        )

    def compute_pull_strength(self, distance: Optional[float] = None) -> PullStrength:
        """
        Compute homeostatic pull strength at given distance.

        Pull = -E[ΔD | D]
        Positive pull means system tends to move back toward band.

        Args:
            distance: Distance to evaluate at (default: current distance)

        Returns:
            PullStrength with expected delta and half-life
        """
        if distance is None:
            if len(self._distance_history) == 0:
                raise ValueError("No history and no distance provided")
            distance = self._distance_history[-1]

        D = np.array(self._distance_history)
        if len(D) < 10:
            return PullStrength(
                distance=distance,
                expected_delta_d=0.0,
                pull_strength=0.0,
                half_life_estimate=float('inf'),
                confidence=0.0,
            )

        delta_D = np.diff(D)
        D_lagged = D[:-1]

        # Find points near this distance
        tolerance = 0.2 * max(distance, 0.1)
        near_idx = np.abs(D_lagged - distance) < tolerance

        if near_idx.sum() < 5:
            return PullStrength(
                distance=distance,
                expected_delta_d=0.0,
                pull_strength=0.0,
                half_life_estimate=float('inf'),
                confidence=0.0,
            )

        expected_delta = float(np.mean(delta_D[near_idx]))
        pull = -expected_delta  # positive pull = moving back

        # Half-life estimate
        if pull > 0 and distance > self.band_radius:
            # Time to halve the excess distance
            excess = distance - self.band_radius
            half_life = float(np.log(2) / pull * excess) if pull > 1e-10 else float('inf')
        else:
            half_life = float('inf')

        return PullStrength(
            distance=distance,
            expected_delta_d=expected_delta,
            pull_strength=pull,
            half_life_estimate=half_life,
            confidence=min(1.0, near_idx.sum() / 50),
        )

    def get_velocity_series(self) -> np.ndarray:
        """Get velocity series from distance history."""
        if len(self._distance_history) < 2:
            return np.array([])
        return np.diff(self._distance_history)

    def get_acceleration_series(self) -> np.ndarray:
        """Get acceleration series from distance history."""
        if len(self._distance_history) < 3:
            return np.array([])
        return np.diff(self._distance_history, n=2)

    def clear(self) -> None:
        """Clear all stored history."""
        self._distance_history = []
        self._timestamps = []

    @property
    def n_observations(self) -> int:
        """Number of observations stored."""
        return len(self._distance_history)

    # =========================================================================
    # Private Methods
    # =========================================================================

    def _get_level(self, d: float) -> int:
        """Get level (0, 1, 2, or 3) based on distance."""
        if d <= self.band_radius:
            return 0
        for i, thresh in enumerate(self.level_thresholds):
            if d <= self.band_radius * (1 + thresh):
                return i + 1
        return len(self.level_thresholds) + 1

    def _level_threshold(self, level: int) -> float:
        """Get distance threshold for entering a level."""
        if level <= 0:
            return 0.0
        if level == 1:
            return self.band_radius
        if level <= len(self.level_thresholds) + 1:
            return self.band_radius * (1 + self.level_thresholds[level - 2])
        return float('inf')

    def _find_similar_episodes(
        self,
        D: np.ndarray,
        current_d: float,
        current_v: float,
        d_tolerance: float = 0.2,
        v_sign_match: bool = True,
    ) -> List[int]:
        """Find historical episodes with similar distance/velocity."""
        similar = []
        for i in range(1, len(D) - 1):
            d = D[i]
            v = D[i] - D[i - 1]

            # Distance match
            if abs(d - current_d) > d_tolerance * max(current_d, 0.1):
                continue

            # Velocity sign match
            if v_sign_match and np.sign(v) != np.sign(current_v):
                continue

            similar.append(i)

        return similar

    def _expected_time_to_reentry(
        self,
        D: np.ndarray,
        similar_idx: List[int],
    ) -> float:
        """Expected time to re-enter band from similar episodes."""
        times = []
        for idx in similar_idx:
            for t in range(1, min(252, len(D) - idx)):
                if D[idx + t] <= self.band_radius:
                    times.append(t)
                    break
        return float(np.mean(times)) if times else float('inf')

    def _expected_time_to_escalation(
        self,
        D: np.ndarray,
        similar_idx: List[int],
        current_level: int,
    ) -> float:
        """Expected time to escalate to next level."""
        threshold = self._level_threshold(current_level + 1)
        times = []
        for idx in similar_idx:
            for t in range(1, min(252, len(D) - idx)):
                if D[idx + t] >= threshold:
                    times.append(t)
                    break
        return float(np.mean(times)) if times else float('inf')

    def _default_hazards(
        self,
        level: int = 1,
        v_sign: int = 0,
    ) -> HazardEstimates:
        """Default hazards when history is insufficient."""
        return HazardEstimates(
            current_level=level,
            current_velocity_sign=v_sign,
            p_reentry_4wk=0.3,
            p_reentry_12wk=0.5,
            p_escalate_4wk=0.2,
            p_escalate_12wk=0.3,
            expected_time_to_reentry=60.0,
            expected_time_to_escalation=90.0,
            n_similar_episodes=0,
        )


# =============================================================================
# Convenience Functions
# =============================================================================

def compute_velocity(
    distances: np.ndarray,
    smoothing: int = 1,
) -> np.ndarray:
    """
    Compute velocity series from distance series.

    Args:
        distances: (n_samples,) array of distances
        smoothing: Moving average window for smoothing

    Returns:
        (n_samples - 1,) array of velocities
    """
    velocities = np.diff(distances)

    if smoothing > 1 and len(velocities) >= smoothing:
        kernel = np.ones(smoothing) / smoothing
        velocities = np.convolve(velocities, kernel, mode='valid')

    return velocities


def classify_trajectory(
    distance: float,
    velocity: float,
    band_radius: float,
    noise_threshold: float = 0.05,
) -> str:
    """
    Classify trajectory type based on distance and velocity.

    Args:
        distance: Current distance from center
        velocity: Current velocity (change in distance)
        band_radius: Radius of homeostatic band
        noise_threshold: Threshold for significant velocity

    Returns:
        One of: "inside", "reverting", "escalating", "stalling"
    """
    if distance <= band_radius:
        return "inside"
    elif velocity < -noise_threshold:
        return "reverting"
    elif velocity > noise_threshold:
        return "escalating"
    else:
        return "stalling"
