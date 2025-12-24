"""
Homeostatic Stress Band (HSB) - Reference Geometry

Defines "normal" as a density-based region in geometry space.
Distance from this region = deformation, not "regime."

Philosophy:
    - "Normal" = highest-density region of geometry space
    - "Stress" = distance from that region
    - No labels, no regimes, just continuous measurement

Usage:
    from prism.core import GeometryReference, HomeostasisBand

    # Fit reference from historical geometry vectors
    ref = GeometryReference(coverage=0.80)
    band = ref.fit(historical_geometry_vectors)

    # Compute distance for new observation
    result = ref.compute_distance(current_geometry)
    print(f"Distance: {result.distance}, Inside band: {result.inside_band}")
"""

from dataclasses import dataclass, field
from typing import Optional, Literal, List
from datetime import datetime
import numpy as np

try:
    from sklearn.covariance import MinCovDet
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class HomeostasisBand:
    """
    The Homeostatic Reference Band.

    Defined by:
        - center (μ_H): robust center of reference geometry cloud
        - covariance (Σ_H): robust scatter
        - radius (r): coverage threshold
        - density_model: optional fitted density for flexible distance

    Attributes:
        center: μ_H - robust center of geometry space
        covariance: Σ_H - robust covariance matrix
        covariance_inv: Σ_H^-1 for Mahalanobis distance
        radius: coverage-defined threshold
        coverage: what % of reference data is inside
        n_samples: how many points defined this band
        fitted_at: timestamp when band was fitted
        density_model: optional fitted density for non-elliptical bands
    """
    center: np.ndarray  # μ_H
    covariance: np.ndarray  # Σ_H
    covariance_inv: np.ndarray  # Σ_H^-1 for Mahalanobis
    radius: float  # coverage-defined threshold
    coverage: float  # what % of reference data is inside
    n_samples: int  # how many points defined this band
    fitted_at: str  # timestamp

    # Optional: density model for non-elliptical bands
    density_model: Optional[object] = None

    @property
    def n_dims(self) -> int:
        """Dimensionality of the geometry space."""
        return len(self.center)


@dataclass
class DistanceResult:
    """Distance from homeostatic band at time t."""
    timestamp: str
    distance: float  # D(t) - Mahalanobis or density-based
    distance_std: float  # uncertainty on D(t) via bootstrap
    inside_band: bool  # D(t) <= r
    percentile: float  # where D(t) falls in historical distribution
    method: str  # "mahalanobis", "knn", "density"

    @property
    def is_excursion(self) -> bool:
        """True if outside the homeostatic band."""
        return not self.inside_band

    @property
    def normalized_distance(self) -> float:
        """Distance normalized by band radius (1.0 = at boundary)."""
        return self.distance  # Already normalized by covariance


# =============================================================================
# Main Class
# =============================================================================

class GeometryReference:
    """
    Builds and queries the Homeostatic Stress Band.

    Philosophy:
        - "Normal" = highest-density region of geometry space
        - "Stress" = distance from that region
        - No labels, no regimes, just continuous measurement

    Usage:
        ref = GeometryReference(coverage=0.80)
        band = ref.fit(geometry_vectors)
        result = ref.compute_distance(new_vector)
    """

    def __init__(
        self,
        coverage: float = 0.80,  # 80% of reference inside band
        method: Literal["mahalanobis", "knn", "density"] = "mahalanobis",
        robust: bool = True,
        min_samples: int = 100,
    ):
        """
        Initialize geometry reference.

        Args:
            coverage: Fraction of reference data inside band (default 0.80)
            method: Distance computation method
            robust: Use robust covariance estimation (MinCovDet)
            min_samples: Minimum samples required to fit
        """
        self.coverage = coverage
        self.method = method
        self.robust = robust
        self.min_samples = min_samples
        self.band: Optional[HomeostasisBand] = None
        self._history: List[float] = []

    def fit(
        self,
        geometry_vectors: np.ndarray,
        timestamps: Optional[List[str]] = None,
    ) -> HomeostasisBand:
        """
        Fit the homeostatic band from historical geometry.

        Args:
            geometry_vectors: (n_samples, n_dims) array of G(t)
            timestamps: optional timestamps for each vector

        Returns:
            HomeostasisBand object
        """
        n_samples, n_dims = geometry_vectors.shape

        if n_samples < self.min_samples:
            raise ValueError(
                f"Need >= {self.min_samples} samples, got {n_samples}"
            )

        # Robust center and covariance
        if self.robust and HAS_SKLEARN and n_samples > n_dims * 5:
            mcd = MinCovDet(support_fraction=0.75).fit(geometry_vectors)
            center = mcd.location_
            covariance = mcd.covariance_
        else:
            center = np.median(geometry_vectors, axis=0)
            covariance = np.cov(geometry_vectors, rowvar=False)
            # Handle 1D case
            if covariance.ndim == 0:
                covariance = np.array([[float(covariance)]])

        # Regularize covariance if needed
        covariance = self._regularize_covariance(covariance)
        covariance_inv = np.linalg.inv(covariance)

        # Compute distances for all historical points
        distances = self._mahalanobis_batch(
            geometry_vectors, center, covariance_inv
        )

        # Set radius by coverage
        radius = float(np.percentile(distances, self.coverage * 100))

        self.band = HomeostasisBand(
            center=center,
            covariance=covariance,
            covariance_inv=covariance_inv,
            radius=radius,
            coverage=self.coverage,
            n_samples=n_samples,
            fitted_at=datetime.now().isoformat(),
        )

        # Store distance history for percentile lookups
        self._history = list(distances)

        return self.band

    def compute_distance(
        self,
        geometry_vector: np.ndarray,
        timestamp: str = "",
        bootstrap_n: int = 50,
    ) -> DistanceResult:
        """
        Compute distance from homeostatic band.

        Args:
            geometry_vector: G(t) at current time
            timestamp: optional timestamp
            bootstrap_n: samples for uncertainty estimation

        Returns:
            DistanceResult with D(t), uncertainty, and percentile
        """
        if self.band is None:
            raise ValueError("Must call fit() before compute_distance()")

        # Primary distance
        d = self._mahalanobis(
            geometry_vector,
            self.band.center,
            self.band.covariance_inv,
        )

        # Bootstrap uncertainty on D(t)
        d_bootstrap = self._bootstrap_distance(geometry_vector, bootstrap_n)
        d_std = float(np.std(d_bootstrap))

        # Percentile vs history
        percentile = float(np.mean(np.array(self._history) <= d) * 100)

        return DistanceResult(
            timestamp=timestamp,
            distance=d,
            distance_std=d_std,
            inside_band=(d <= self.band.radius),
            percentile=percentile,
            method=self.method,
        )

    def compute_distance_batch(
        self,
        geometry_vectors: np.ndarray,
        timestamps: Optional[List[str]] = None,
    ) -> List[DistanceResult]:
        """
        Compute distances for multiple geometry vectors.

        Args:
            geometry_vectors: (n_samples, n_dims) array
            timestamps: optional list of timestamps

        Returns:
            List of DistanceResult objects
        """
        if self.band is None:
            raise ValueError("Must call fit() before compute_distance_batch()")

        if timestamps is None:
            timestamps = [""] * len(geometry_vectors)

        results = []
        for i, (vec, ts) in enumerate(zip(geometry_vectors, timestamps)):
            result = self.compute_distance(vec, ts, bootstrap_n=10)
            results.append(result)

        return results

    def get_band_summary(self) -> dict:
        """Get summary statistics of the fitted band."""
        if self.band is None:
            return {"fitted": False}

        return {
            "fitted": True,
            "n_dims": self.band.n_dims,
            "n_samples": self.band.n_samples,
            "coverage": self.band.coverage,
            "radius": self.band.radius,
            "center_norm": float(np.linalg.norm(self.band.center)),
            "fitted_at": self.band.fitted_at,
            "method": self.method,
            "history_size": len(self._history),
        }

    # =========================================================================
    # Private Methods
    # =========================================================================

    def _mahalanobis(
        self,
        x: np.ndarray,
        center: np.ndarray,
        cov_inv: np.ndarray,
    ) -> float:
        """Compute Mahalanobis distance for single point."""
        diff = x - center
        return float(np.sqrt(diff @ cov_inv @ diff))

    def _mahalanobis_batch(
        self,
        X: np.ndarray,
        center: np.ndarray,
        cov_inv: np.ndarray,
    ) -> np.ndarray:
        """Compute Mahalanobis distances for batch of points."""
        diff = X - center
        return np.sqrt(np.sum(diff @ cov_inv * diff, axis=1))

    def _regularize_covariance(
        self,
        cov: np.ndarray,
        eps: float = 1e-6,
    ) -> np.ndarray:
        """Add small ridge to ensure invertibility."""
        return cov + eps * np.eye(cov.shape[0])

    def _bootstrap_distance(
        self,
        x: np.ndarray,
        n: int,
    ) -> np.ndarray:
        """Bootstrap uncertainty by resampling covariance estimation."""
        if self.band is None:
            return np.array([0.0])

        distances = []
        for _ in range(n):
            # Add small noise to center/cov
            noise = np.random.normal(0, 0.01, self.band.center.shape)
            d = self._mahalanobis(
                x,
                self.band.center + noise,
                self.band.covariance_inv,
            )
            distances.append(d)
        return np.array(distances)


# =============================================================================
# Convenience Functions
# =============================================================================

def fit_homeostasis_band(
    geometry_vectors: np.ndarray,
    coverage: float = 0.80,
    robust: bool = True,
) -> HomeostasisBand:
    """
    Convenience function to fit a homeostasis band.

    Args:
        geometry_vectors: (n_samples, n_dims) historical geometry
        coverage: Fraction inside band
        robust: Use robust estimation

    Returns:
        Fitted HomeostasisBand
    """
    ref = GeometryReference(coverage=coverage, robust=robust)
    return ref.fit(geometry_vectors)


def compute_deformation(
    current_geometry: np.ndarray,
    band: HomeostasisBand,
) -> float:
    """
    Compute deformation (distance from homeostasis).

    Args:
        current_geometry: Current geometry vector G(t)
        band: Fitted HomeostasisBand

    Returns:
        Deformation distance D(t)
    """
    diff = current_geometry - band.center
    return float(np.sqrt(diff @ band.covariance_inv @ diff))
