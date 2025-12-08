"""
Geometric Framework Module
==========================

Define the metric structure of the indicator space.
Critical for any physics-inspired interpretation.

Key components:
- Metric tensor definition
- Distance computation
- Geodesic paths
- Curvature measures (if needed)

IMPORTANT DISCLAIMER:
The geometric framework is an analytical convenience, not a claim about
fundamental physics. The indicator space does not have intrinsic geometric
structure - we impose geometry for computational purposes.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum


class MetricType(Enum):
    """Available metric tensor types."""
    EUCLIDEAN = 'euclidean'
    MAHALANOBIS = 'mahalanobis'
    WEIGHTED = 'weighted'
    CORRELATION = 'correlation'


@dataclass
class GeometryReport:
    """Report on geometric properties of the space."""
    metric_type: str
    n_dimensions: int
    metric_tensor: np.ndarray
    is_positive_definite: bool
    eigenvalues: np.ndarray
    condition_number: float
    effective_dimensions: int  # Based on eigenvalue analysis
    notes: List[str] = field(default_factory=list)


class MetricTensor:
    """
    Metric tensor for the indicator space.

    The metric defines how we measure distances between points
    (states of the system) in the indicator space.

    Mathematical Background:
    -----------------------
    The metric tensor g_ij defines the infinitesimal distance:

        ds² = Σ_ij g_ij dx_i dx_j

    For Euclidean metric: g_ij = δ_ij (Kronecker delta)
    For Mahalanobis: g_ij = Σ^{-1}_ij (inverse covariance)

    The choice of metric affects:
    - Distance measurements
    - Clustering results
    - Geodesic paths
    - Volume elements

    DISCLAIMER:
    This is an imposed structure, not discovered from data.
    We assume the metric for analytical convenience.
    """

    def __init__(self,
                 metric_type: str = 'euclidean',
                 data: Optional[pd.DataFrame] = None,
                 weights: Optional[np.ndarray] = None):
        """
        Initialize metric tensor.

        Args:
            metric_type: 'euclidean', 'mahalanobis', 'weighted', 'correlation'
            data: Data for computing covariance-based metrics
            weights: Custom weights for weighted metric
        """
        self.metric_type = MetricType(metric_type)
        self.data = data
        self.weights = weights
        self.g = None  # Metric tensor
        self.g_inv = None  # Inverse metric tensor
        self.n_dims = None

        if data is not None:
            self._compute_metric(data)

    def _compute_metric(self, data: pd.DataFrame):
        """Compute metric tensor from data."""
        self.n_dims = data.shape[1]

        if self.metric_type == MetricType.EUCLIDEAN:
            # Identity metric
            self.g = np.eye(self.n_dims)
            self.g_inv = np.eye(self.n_dims)

        elif self.metric_type == MetricType.MAHALANOBIS:
            # Inverse covariance metric
            cov = data.cov().values
            self.g_inv = cov  # Covariance is inverse metric
            try:
                self.g = np.linalg.inv(cov)
            except np.linalg.LinAlgError:
                # Regularize
                self.g = np.linalg.inv(cov + 0.01 * np.eye(self.n_dims))

        elif self.metric_type == MetricType.WEIGHTED:
            if self.weights is None:
                self.weights = np.ones(self.n_dims)
            self.g = np.diag(self.weights)
            self.g_inv = np.diag(1 / self.weights)

        elif self.metric_type == MetricType.CORRELATION:
            # Use inverse correlation matrix
            corr = data.corr().values
            try:
                self.g = np.linalg.inv(corr)
            except np.linalg.LinAlgError:
                self.g = np.linalg.inv(corr + 0.01 * np.eye(self.n_dims))
            self.g_inv = corr

    def set_metric(self, g: np.ndarray):
        """Manually set the metric tensor."""
        self.g = g
        self.n_dims = g.shape[0]
        try:
            self.g_inv = np.linalg.inv(g)
        except:
            self.g_inv = np.linalg.pinv(g)

    def distance(self,
                 x: np.ndarray,
                 y: np.ndarray) -> float:
        """
        Compute distance between two points.

        d(x, y) = sqrt((x-y)^T g (x-y))

        Args:
            x, y: Points in indicator space

        Returns:
            Distance
        """
        if self.g is None:
            raise ValueError("Metric not initialized")

        diff = np.asarray(x) - np.asarray(y)
        squared_dist = diff @ self.g @ diff

        return np.sqrt(max(squared_dist, 0))

    def distance_matrix(self,
                        points: np.ndarray) -> np.ndarray:
        """
        Compute pairwise distance matrix.

        Args:
            points: N x D array of points

        Returns:
            N x N distance matrix
        """
        n = len(points)
        D = np.zeros((n, n))

        for i in range(n):
            for j in range(i + 1, n):
                d = self.distance(points[i], points[j])
                D[i, j] = d
                D[j, i] = d

        return D

    def report(self) -> GeometryReport:
        """Generate report on metric properties."""
        if self.g is None:
            raise ValueError("Metric not initialized")

        # Check positive definiteness
        eigenvalues = np.linalg.eigvalsh(self.g)
        is_positive_definite = np.all(eigenvalues > 0)

        # Condition number
        condition_number = np.max(np.abs(eigenvalues)) / (np.min(np.abs(eigenvalues)) + 1e-10)

        # Effective dimensions (eigenvalues > 1% of max)
        threshold = 0.01 * np.max(eigenvalues)
        effective_dims = np.sum(eigenvalues > threshold)

        notes = [
            f"Metric type: {self.metric_type.value}",
            "The metric is an imposed analytical structure.",
            "Distance measurements depend on metric choice."
        ]

        if not is_positive_definite:
            notes.append("WARNING: Metric is not positive definite. Distances may be imaginary.")

        if condition_number > 100:
            notes.append(f"WARNING: High condition number ({condition_number:.1f}). Metric nearly singular.")

        return GeometryReport(
            metric_type=self.metric_type.value,
            n_dimensions=self.n_dims,
            metric_tensor=self.g,
            is_positive_definite=is_positive_definite,
            eigenvalues=eigenvalues,
            condition_number=condition_number,
            effective_dimensions=int(effective_dims),
            notes=notes
        )


def compute_distance(x: np.ndarray,
                     y: np.ndarray,
                     metric: str = 'euclidean',
                     cov: Optional[np.ndarray] = None) -> float:
    """
    Compute distance between two points.

    Args:
        x, y: Points
        metric: 'euclidean' or 'mahalanobis'
        cov: Covariance matrix (for Mahalanobis)

    Returns:
        Distance
    """
    diff = np.asarray(x) - np.asarray(y)

    if metric == 'euclidean':
        return np.sqrt(np.sum(diff ** 2))

    elif metric == 'mahalanobis':
        if cov is None:
            raise ValueError("Covariance required for Mahalanobis distance")
        try:
            cov_inv = np.linalg.inv(cov)
        except:
            cov_inv = np.linalg.pinv(cov)
        return np.sqrt(diff @ cov_inv @ diff)

    else:
        raise ValueError(f"Unknown metric: {metric}")


def compute_geodesic(start: np.ndarray,
                     end: np.ndarray,
                     metric: MetricTensor,
                     n_points: int = 10) -> np.ndarray:
    """
    Compute geodesic path between two points.

    For flat space (Euclidean or constant metric), geodesic is straight line.
    For curved space, would need to solve geodesic equation.

    Args:
        start, end: Endpoints
        metric: MetricTensor object
        n_points: Number of points on path

    Returns:
        Array of points along geodesic
    """
    # For flat metrics, geodesic is linear interpolation
    # This is exact for Euclidean and approximate for Mahalanobis

    t = np.linspace(0, 1, n_points)
    path = np.array([
        start * (1 - ti) + end * ti
        for ti in t
    ])

    return path


def compute_volume_element(metric: MetricTensor,
                           point: np.ndarray) -> float:
    """
    Compute volume element sqrt(det(g)) at a point.

    For integration in curved space:
    ∫ f dV = ∫ f sqrt(det(g)) dx¹...dxⁿ

    Args:
        metric: MetricTensor
        point: Point (not used for constant metrics)

    Returns:
        Volume element
    """
    if metric.g is None:
        return 1.0

    det_g = np.linalg.det(metric.g)
    return np.sqrt(abs(det_g))


def ricci_scalar(metric: MetricTensor) -> float:
    """
    Compute Ricci scalar curvature.

    For flat metrics, this is zero.
    Non-zero would indicate intrinsic curvature.

    Note: Full Ricci computation requires derivatives of metric,
    which assumes metric varies with position. For constant metrics, R = 0.

    Args:
        metric: MetricTensor

    Returns:
        Ricci scalar (0 for constant metrics)
    """
    # For constant metrics, curvature is zero
    # This is a placeholder for more sophisticated analysis
    return 0.0


def print_geometry_report(report: GeometryReport):
    """Pretty-print geometry report."""
    print("=" * 60)
    print("GEOMETRY REPORT")
    print("=" * 60)
    print(f"Metric type: {report.metric_type}")
    print(f"Dimensions: {report.n_dimensions}")
    print(f"Effective dimensions: {report.effective_dimensions}")
    print(f"Positive definite: {report.is_positive_definite}")
    print(f"Condition number: {report.condition_number:.2f}")
    print()

    print("Eigenvalues:")
    for i, ev in enumerate(report.eigenvalues):
        print(f"  λ_{i+1} = {ev:.4f}")

    print()
    print("Notes:")
    for note in report.notes:
        print(f"  - {note}")

    print("=" * 60)


if __name__ == "__main__":
    print("=" * 60)
    print("Geometry Module - Demo")
    print("=" * 60)

    np.random.seed(42)

    # Create sample data
    n = 500
    data = pd.DataFrame({
        'x1': np.random.randn(n),
        'x2': np.random.randn(n) * 2,
        'x3': np.random.randn(n) * 0.5
    })

    print("\nEuclidean Metric:")
    print("-" * 40)
    metric_euc = MetricTensor('euclidean', data)
    report = metric_euc.report()
    print_geometry_report(report)

    print("\nMahalanobis Metric:")
    print("-" * 40)
    metric_mah = MetricTensor('mahalanobis', data)
    report = metric_mah.report()
    print_geometry_report(report)

    # Test distances
    x = np.array([1, 0, 0])
    y = np.array([0, 1, 0])

    print("\nDistance comparisons (x=[1,0,0], y=[0,1,0]):")
    print(f"  Euclidean: {metric_euc.distance(x, y):.4f}")
    print(f"  Mahalanobis: {metric_mah.distance(x, y):.4f}")

    # Geodesic
    geodesic = compute_geodesic(x, y, metric_euc)
    print(f"\nGeodesic (linear for flat space): {len(geodesic)} points")

    print("\nTest completed!")
