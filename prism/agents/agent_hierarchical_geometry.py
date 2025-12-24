"""
PRISM Hierarchical Geometry Engine

Multi-scale geometry with telescope and microscope views.

Core insight: Correlated indicators form clusters that act as single masses
at distance, but have internal structure visible under different lenses.

TELESCOPE VIEW (macro):
    Clusters appear as single masses
    Singletons are individual points
    Inter-cluster structure visible

MICROSCOPE VIEW (micro):
    Internal cluster structure
    Different lens → different topology
    Outliers within clusters flagged

The gravitational pull of a cluster on distant indicators depends on
cluster mass and centroid position, NOT internal structure.

But the internal structure matters for understanding what the cluster IS.

Usage:
    engine = HierarchicalGeometryEngine('prism.db')
    result = engine.compute_hierarchical_geometry(
        indicators=['TLT', 'IEF', 'SHY', 'SPY', 'VIX', ...],
        start_date=datetime(2020, 1, 1),
        end_date=datetime(2023, 12, 31)
    )

    print(result.macro_report())
    print(result.micro_report('bond_cluster'))
    print(result.divergence_report())

Author: Jason (PRISM Project)
Date: December 2024
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Set, Any
from pathlib import Path
from collections import defaultdict
import logging
import json

logger = logging.getLogger(__name__)


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class IntrinsicProfile:
    """
    Properties computed from indicator's own data only.
    Invariant - doesn't change based on what else is in the analysis.
    """
    indicator_id: str

    # Core intrinsic measures
    entropy: float
    hurst: float
    kurtosis: float
    permutation_entropy: float
    spectral_entropy: float
    recurrence_rate: float
    determinism: float
    lyapunov: float

    # Wavelet decomposition
    wavelet_energy_by_scale: Dict[str, float]  # {scale: energy}
    dominant_scale: str

    # Summary
    persistence: str  # 'trending', 'random_walk', 'mean_reverting'
    complexity: str   # 'low', 'moderate', 'high'

    def to_vector(self) -> np.ndarray:
        """Convert to numpy vector for geometry computations."""
        return np.array([
            self.entropy,
            self.hurst,
            self.kurtosis,
            self.permutation_entropy,
            self.spectral_entropy,
            self.recurrence_rate,
            self.determinism,
            self.lyapunov,
        ])

    def to_dict(self) -> Dict[str, Any]:
        return {
            'indicator_id': self.indicator_id,
            'entropy': self.entropy,
            'hurst': self.hurst,
            'kurtosis': self.kurtosis,
            'permutation_entropy': self.permutation_entropy,
            'spectral_entropy': self.spectral_entropy,
            'recurrence_rate': self.recurrence_rate,
            'determinism': self.determinism,
            'lyapunov': self.lyapunov,
            'wavelet_energy_by_scale': self.wavelet_energy_by_scale,
            'dominant_scale': self.dominant_scale,
            'persistence': self.persistence,
            'complexity': self.complexity,
        }


@dataclass
class Cluster:
    """
    A group of correlated indicators that act as single mass at distance.
    """
    cluster_id: str
    members: List[str]

    # Centroid in various spaces
    centroid_intrinsic: np.ndarray      # Intrinsic space centroid
    centroid_raw: np.ndarray            # Raw correlation space centroid

    # Cluster properties
    mass: float                          # Number of members (or weighted)
    internal_radius: float               # Max distance from centroid
    mean_internal_correlation: float     # Why they clustered
    internal_variance: float             # How tight is the cluster?

    # Internal structure per lens
    internal_structure: Dict[str, 'MicroGeometry']  # lens_name -> structure

    # Outliers within cluster
    internal_outliers: Dict[str, List[str]]  # lens_name -> [outlier_ids]

    def to_dict(self) -> Dict[str, Any]:
        return {
            'cluster_id': self.cluster_id,
            'members': self.members,
            'mass': self.mass,
            'internal_radius': self.internal_radius,
            'mean_internal_correlation': self.mean_internal_correlation,
            'internal_variance': self.internal_variance,
            'internal_outliers': self.internal_outliers,
        }


@dataclass
class MicroGeometry:
    """
    Internal structure of a cluster under one lens.
    """
    cluster_id: str
    lens_name: str

    # Positions within cluster (local coordinates)
    positions: Dict[str, np.ndarray]  # indicator_id -> position

    # Internal distances
    pairwise_distances: Dict[Tuple[str, str], float]

    # Topology
    nearest_neighbors: Dict[str, str]  # indicator_id -> nearest neighbor
    subclusters: List[List[str]]       # Groups within the cluster

    # Outliers
    outliers: List[str]                # Indicators far from others
    outlier_scores: Dict[str, float]   # indicator_id -> outlier score

    # Summary
    homogeneous: bool                  # Is structure uniform under this lens?
    description: str                   # Human-readable

    def to_dict(self) -> Dict[str, Any]:
        return {
            'cluster_id': self.cluster_id,
            'lens_name': self.lens_name,
            'positions': {k: v.tolist() for k, v in self.positions.items()},
            'nearest_neighbors': self.nearest_neighbors,
            'subclusters': self.subclusters,
            'outliers': self.outliers,
            'outlier_scores': self.outlier_scores,
            'homogeneous': self.homogeneous,
            'description': self.description,
        }


@dataclass
class MacroGeometry:
    """
    Telescope view: clusters as masses, singletons as points.
    """
    # Entities (clusters + singletons)
    entities: Dict[str, Dict[str, Any]]  # entity_id -> {type, mass, position}

    # Inter-entity distances
    distances: Dict[Tuple[str, str], float]

    # Gravitational influences
    influences: Dict[str, Dict[str, float]]  # target -> {source: pull}

    # Macro structure
    macro_clusters: List[List[str]]  # Groups of entities

    def to_dict(self) -> Dict[str, Any]:
        return {
            'entities': self.entities,
            'distances': {f"{k[0]}:{k[1]}": v for k, v in self.distances.items()},
            'macro_clusters': self.macro_clusters,
        }


@dataclass
class LensDivergence:
    """
    A case where two indicators look different in correlation but similar
    in another lens, or vice versa.
    """
    indicator_a: str
    indicator_b: str

    # Correlation view
    correlation: float
    correlated: bool  # > 0.7

    # Divergent lens
    lens_name: str
    lens_distance: float
    lens_similar: bool  # distance < threshold

    # Type
    divergence_type: str  # 'correlated_but_different' or 'uncorrelated_but_similar'

    # Interpretation
    interpretation: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            'indicator_a': self.indicator_a,
            'indicator_b': self.indicator_b,
            'correlation': self.correlation,
            'lens_name': self.lens_name,
            'lens_distance': self.lens_distance,
            'divergence_type': self.divergence_type,
            'interpretation': self.interpretation,
        }


@dataclass
class HierarchicalGeometryResult:
    """
    Complete hierarchical geometry analysis result.
    """
    # Metadata
    n_indicators: int
    n_clusters: int
    n_singletons: int
    analysis_date: datetime

    # Intrinsic profiles (invariant)
    intrinsic_profiles: Dict[str, IntrinsicProfile]

    # Cluster structure
    clusters: Dict[str, Cluster]
    singletons: List[str]
    cluster_assignments: Dict[str, str]  # indicator_id -> cluster_id or 'singleton'

    # Macro geometry (telescope)
    macro_geometry: MacroGeometry

    # Micro geometry (microscope) - per cluster, per lens
    micro_geometries: Dict[str, Dict[str, MicroGeometry]]  # cluster_id -> lens -> geometry

    # Divergences
    divergences: List[LensDivergence]

    # Cross-level findings
    findings: List[Dict[str, Any]]

    def macro_report(self) -> str:
        """Generate telescope view report."""
        lines = [
            "=" * 60,
            "MACRO GEOMETRY (Telescope View)",
            "=" * 60,
            "",
            f"Clusters: {self.n_clusters}",
            f"Singletons: {self.n_singletons}",
            "",
        ]

        # List clusters
        lines.append("CLUSTERS:")
        for cid, cluster in self.clusters.items():
            lines.append(f"  {cid}:")
            lines.append(f"    Members: {', '.join(cluster.members)}")
            lines.append(f"    Mass: {cluster.mass:.1f}")
            lines.append(f"    Mean correlation: {cluster.mean_internal_correlation:.2f}")
            lines.append(f"    Internal radius: {cluster.internal_radius:.3f}")
            lines.append("")

        # List singletons
        if self.singletons:
            lines.append(f"SINGLETONS: {', '.join(self.singletons)}")
            lines.append("")

        # Inter-cluster distances
        lines.append("INTER-ENTITY DISTANCES:")
        sorted_distances = sorted(
            self.macro_geometry.distances.items(),
            key=lambda x: x[1]
        )
        for (e1, e2), dist in sorted_distances[:10]:
            lines.append(f"  {e1} <-> {e2}: {dist:.3f}")

        return '\n'.join(lines)

    def micro_report(self, cluster_id: str) -> str:
        """Generate microscope view report for one cluster."""
        if cluster_id not in self.clusters:
            return f"Cluster '{cluster_id}' not found"

        cluster = self.clusters[cluster_id]
        micro = self.micro_geometries.get(cluster_id, {})

        lines = [
            "=" * 60,
            f"MICRO GEOMETRY: {cluster_id} (Microscope View)",
            "=" * 60,
            "",
            f"Members: {', '.join(cluster.members)}",
            f"Mean correlation: {cluster.mean_internal_correlation:.2f}",
            "",
            "STRUCTURE BY LENS:",
            "",
        ]

        for lens_name, geom in micro.items():
            lines.append(f"  {lens_name.upper()} LENS:")
            lines.append(f"    Homogeneous: {'Yes' if geom.homogeneous else 'No'}")

            if geom.outliers:
                lines.append(f"    Outliers: {', '.join(geom.outliers)}")

            if geom.subclusters and len(geom.subclusters) > 1:
                lines.append(f"    Subclusters: {len(geom.subclusters)}")
                for i, sc in enumerate(geom.subclusters):
                    lines.append(f"      Group {i+1}: {', '.join(sc)}")

            lines.append(f"    {geom.description}")
            lines.append("")

        return '\n'.join(lines)

    def divergence_report(self) -> str:
        """Generate lens divergence report."""
        lines = [
            "=" * 60,
            "LENS DIVERGENCE REPORT",
            "=" * 60,
            "",
        ]

        # Uncorrelated but similar
        uncorr_similar = [d for d in self.divergences
                        if d.divergence_type == 'uncorrelated_but_similar']

        if uncorr_similar:
            lines.append("UNCORRELATED BUT STRUCTURALLY SIMILAR:")
            for d in uncorr_similar[:10]:
                lines.append(f"  {d.indicator_a} <-> {d.indicator_b}")
                lines.append(f"    Correlation: {d.correlation:.2f}")
                lines.append(f"    {d.lens_name} distance: {d.lens_distance:.3f}")
                lines.append(f"    -> {d.interpretation}")
                lines.append("")

        # Correlated but different
        corr_different = [d for d in self.divergences
                        if d.divergence_type == 'correlated_but_different']

        if corr_different:
            lines.append("CORRELATED BUT STRUCTURALLY DIFFERENT:")
            for d in corr_different[:10]:
                lines.append(f"  {d.indicator_a} <-> {d.indicator_b}")
                lines.append(f"    Correlation: {d.correlation:.2f}")
                lines.append(f"    {d.lens_name} distance: {d.lens_distance:.3f}")
                lines.append(f"    -> {d.interpretation}")
                lines.append("")

        if not self.divergences:
            lines.append("No significant divergences detected.")

        return '\n'.join(lines)

    def findings_report(self) -> str:
        """Generate cross-level findings report."""
        lines = [
            "=" * 60,
            "CROSS-LEVEL FINDINGS",
            "=" * 60,
            "",
        ]

        for finding in self.findings:
            lines.append(f"* {finding.get('type', 'Finding')}:")
            lines.append(f"   {finding.get('description', '')}")
            if 'indicators' in finding:
                lines.append(f"   Indicators: {', '.join(finding['indicators'])}")
            lines.append("")

        if not self.findings:
            lines.append("No cross-level findings.")

        return '\n'.join(lines)

    def full_report(self) -> str:
        """Generate complete report."""
        return '\n\n'.join([
            self.macro_report(),
            self.divergence_report(),
            self.findings_report(),
        ])

    def to_dict(self) -> Dict[str, Any]:
        return {
            'n_indicators': self.n_indicators,
            'n_clusters': self.n_clusters,
            'n_singletons': self.n_singletons,
            'intrinsic_profiles': {k: v.to_dict() for k, v in self.intrinsic_profiles.items()},
            'clusters': {k: v.to_dict() for k, v in self.clusters.items()},
            'singletons': self.singletons,
            'cluster_assignments': self.cluster_assignments,
            'macro_geometry': self.macro_geometry.to_dict(),
            'divergences': [d.to_dict() for d in self.divergences],
            'findings': self.findings,
        }


# =============================================================================
# HIERARCHICAL GEOMETRY ENGINE
# =============================================================================

class HierarchicalGeometryEngine:
    """
    Multi-scale geometry engine with telescope and microscope views.
    """

    # Intrinsic lenses (don't depend on other indicators)
    INTRINSIC_LENSES = [
        'entropy', 'hurst', 'kurtosis', 'permutation_entropy',
        'spectral_entropy', 'recurrence', 'determinism', 'wavelet'
    ]

    # Thresholds
    CORRELATION_CLUSTER_THRESHOLD = 0.7  # Min correlation to cluster
    OUTLIER_THRESHOLD = 2.0              # Std devs to be outlier
    SIMILARITY_THRESHOLD = 0.3           # Max distance to be "similar"

    def __init__(self, db_path: str = None):
        """
        Initialize engine.

        Args:
            db_path: Path to PRISM DuckDB database (optional)
        """
        self.db_path = Path(db_path) if db_path else None
        self._data_cache = {}

    def compute_hierarchical_geometry(
        self,
        indicators: List[str] = None,
        start_date: datetime = None,
        end_date: datetime = None,
        data: pd.DataFrame = None,
        intrinsic_profiles: Dict[str, Dict] = None,
        correlation_matrix: pd.DataFrame = None,
    ) -> HierarchicalGeometryResult:
        """
        Compute complete hierarchical geometry.

        Args:
            indicators: List of indicator IDs
            start_date: Analysis start
            end_date: Analysis end
            data: Pre-loaded data (optional)
            intrinsic_profiles: Pre-computed profiles (optional)
            correlation_matrix: Pre-computed correlations (optional)

        Returns:
            HierarchicalGeometryResult with all views
        """
        logger.info("Computing hierarchical geometry...")

        # Step 1: Load or compute intrinsic profiles
        if intrinsic_profiles:
            profiles = self._dict_to_profiles(intrinsic_profiles)
        else:
            profiles = self._compute_intrinsic_profiles(indicators, start_date, end_date, data)

        indicator_ids = list(profiles.keys())
        logger.info(f"Computed intrinsic profiles for {len(indicator_ids)} indicators")

        # Step 2: Compute correlation matrix
        if correlation_matrix is not None:
            corr_matrix = correlation_matrix
        else:
            corr_matrix = self._compute_correlation_matrix(indicator_ids, start_date, end_date, data)

        # Step 3: Detect clusters
        clusters, singletons, assignments = self._detect_clusters(
            indicator_ids, corr_matrix, profiles
        )
        logger.info(f"Detected {len(clusters)} clusters, {len(singletons)} singletons")

        # Step 4: Compute macro geometry (telescope)
        macro_geometry = self._compute_macro_geometry(clusters, singletons, profiles)

        # Step 5: Compute micro geometry per cluster (microscope)
        micro_geometries = {}
        for cluster_id, cluster in clusters.items():
            micro_geometries[cluster_id] = self._compute_micro_geometry(
                cluster, profiles
            )

        # Step 6: Detect lens divergences
        divergences = self._detect_divergences(indicator_ids, corr_matrix, profiles)
        logger.info(f"Found {len(divergences)} lens divergences")

        # Step 7: Generate cross-level findings
        findings = self._generate_findings(clusters, singletons, profiles, divergences)

        return HierarchicalGeometryResult(
            n_indicators=len(indicator_ids),
            n_clusters=len(clusters),
            n_singletons=len(singletons),
            analysis_date=datetime.now(),
            intrinsic_profiles=profiles,
            clusters=clusters,
            singletons=singletons,
            cluster_assignments=assignments,
            macro_geometry=macro_geometry,
            micro_geometries=micro_geometries,
            divergences=divergences,
            findings=findings,
        )

    # =========================================================================
    # INTRINSIC PROFILES
    # =========================================================================

    def _compute_intrinsic_profiles(
        self,
        indicators: List[str],
        start_date: datetime,
        end_date: datetime,
        data: pd.DataFrame = None
    ) -> Dict[str, IntrinsicProfile]:
        """Compute intrinsic profile for each indicator."""

        profiles = {}

        for ind_id in indicators:
            try:
                profile = self._compute_single_profile(ind_id, start_date, end_date, data)
                profiles[ind_id] = profile
            except Exception as e:
                logger.warning(f"Failed to compute profile for {ind_id}: {e}")

        return profiles

    def _compute_single_profile(
        self,
        indicator_id: str,
        start_date: datetime,
        end_date: datetime,
        data: pd.DataFrame = None
    ) -> IntrinsicProfile:
        """Compute intrinsic profile for one indicator."""

        # Get time series
        if data is not None:
            series = data[data['indicator_id'] == indicator_id]['value'].values
        else:
            series = self._load_series(indicator_id, start_date, end_date)

        if len(series) < 50:
            raise ValueError(f"Insufficient data: {len(series)} points")

        # Compute intrinsic measures
        entropy = self._compute_entropy(series)
        hurst = self._compute_hurst(series)
        kurtosis = self._compute_kurtosis(series)
        perm_entropy = self._compute_permutation_entropy(series)
        spec_entropy = self._compute_spectral_entropy(series)
        recurrence = self._compute_recurrence_rate(series)
        determinism = self._compute_determinism(series)
        lyapunov = self._compute_lyapunov(series)
        wavelet_energy = self._compute_wavelet_energy(series)

        # Classify persistence
        if hurst > 0.6:
            persistence = 'trending'
        elif hurst < 0.4:
            persistence = 'mean_reverting'
        else:
            persistence = 'random_walk'

        # Classify complexity
        if perm_entropy > 0.8:
            complexity = 'high'
        elif perm_entropy < 0.5:
            complexity = 'low'
        else:
            complexity = 'moderate'

        return IntrinsicProfile(
            indicator_id=indicator_id,
            entropy=entropy,
            hurst=hurst,
            kurtosis=kurtosis,
            permutation_entropy=perm_entropy,
            spectral_entropy=spec_entropy,
            recurrence_rate=recurrence,
            determinism=determinism,
            lyapunov=lyapunov,
            wavelet_energy_by_scale=wavelet_energy,
            dominant_scale=max(wavelet_energy, key=wavelet_energy.get) if wavelet_energy else 'unknown',
            persistence=persistence,
            complexity=complexity,
        )

    def _dict_to_profiles(self, data: Dict[str, Dict]) -> Dict[str, IntrinsicProfile]:
        """Convert dict representation to IntrinsicProfile objects."""
        profiles = {}
        for ind_id, d in data.items():
            profiles[ind_id] = IntrinsicProfile(
                indicator_id=ind_id,
                entropy=d.get('entropy', 0),
                hurst=d.get('hurst', 0.5),
                kurtosis=d.get('kurtosis', 3),
                permutation_entropy=d.get('permutation_entropy', 0.5),
                spectral_entropy=d.get('spectral_entropy', 0.5),
                recurrence_rate=d.get('recurrence_rate', 0),
                determinism=d.get('determinism', 0),
                lyapunov=d.get('lyapunov', 0),
                wavelet_energy_by_scale=d.get('wavelet_energy_by_scale', {}),
                dominant_scale=d.get('dominant_scale', 'unknown'),
                persistence=d.get('persistence', 'random_walk'),
                complexity=d.get('complexity', 'moderate'),
            )
        return profiles

    # =========================================================================
    # INTRINSIC MEASURE COMPUTATIONS
    # =========================================================================

    def _compute_entropy(self, series: np.ndarray) -> float:
        """Shannon entropy of the distribution."""
        try:
            # Discretize into bins
            hist, _ = np.histogram(series, bins=20, density=True)
            hist = hist[hist > 0]  # Remove zeros
            return -np.sum(hist * np.log2(hist)) / np.log2(20)  # Normalized
        except:
            return 0.5

    def _compute_hurst(self, series: np.ndarray) -> float:
        """Hurst exponent estimation."""
        try:
            n = len(series)
            max_k = min(n // 2, 100)

            # R/S analysis
            rs_values = []
            ks = []

            for k in range(10, max_k, 5):
                rs = self._rs_statistic(series, k)
                if rs > 0:
                    rs_values.append(np.log(rs))
                    ks.append(np.log(k))

            if len(ks) < 3:
                return 0.5

            # Linear fit
            slope, _ = np.polyfit(ks, rs_values, 1)
            return np.clip(slope, 0, 1)
        except:
            return 0.5

    def _rs_statistic(self, series: np.ndarray, k: int) -> float:
        """R/S statistic for window size k."""
        try:
            n_windows = len(series) // k
            rs_sum = 0

            for i in range(n_windows):
                window = series[i*k:(i+1)*k]
                mean = np.mean(window)
                deviations = window - mean
                cumsum = np.cumsum(deviations)
                R = np.max(cumsum) - np.min(cumsum)
                S = np.std(window, ddof=1)
                if S > 0:
                    rs_sum += R / S

            return rs_sum / n_windows if n_windows > 0 else 0
        except:
            return 0

    def _compute_kurtosis(self, series: np.ndarray) -> float:
        """Excess kurtosis."""
        try:
            from scipy.stats import kurtosis
            return kurtosis(series, fisher=True)
        except:
            return 0

    def _compute_permutation_entropy(self, series: np.ndarray, order: int = 3) -> float:
        """Permutation entropy."""
        try:
            n = len(series)
            permutations = []

            for i in range(n - order + 1):
                window = series[i:i+order]
                perm = tuple(np.argsort(window))
                permutations.append(perm)

            # Count permutations
            from collections import Counter
            counts = Counter(permutations)
            probs = np.array(list(counts.values())) / len(permutations)

            # Entropy
            entropy = -np.sum(probs * np.log2(probs))
            max_entropy = np.log2(np.math.factorial(order))

            return entropy / max_entropy if max_entropy > 0 else 0
        except:
            return 0.5

    def _compute_spectral_entropy(self, series: np.ndarray) -> float:
        """Spectral entropy from FFT."""
        try:
            fft = np.abs(np.fft.rfft(series - np.mean(series)))
            psd = fft ** 2
            psd = psd / np.sum(psd)  # Normalize
            psd = psd[psd > 0]

            entropy = -np.sum(psd * np.log2(psd))
            max_entropy = np.log2(len(psd))

            return entropy / max_entropy if max_entropy > 0 else 0
        except:
            return 0.5

    def _compute_recurrence_rate(self, series: np.ndarray, threshold: float = None) -> float:
        """Recurrence rate from recurrence plot."""
        try:
            n = min(len(series), 500)  # Limit for speed
            series = series[:n]

            if threshold is None:
                threshold = 0.1 * np.std(series)

            # Distance matrix (simplified)
            recurrence_count = 0
            for i in range(n):
                for j in range(i+1, n):
                    if abs(series[i] - series[j]) < threshold:
                        recurrence_count += 2  # Symmetric

            total = n * (n - 1)
            return recurrence_count / total if total > 0 else 0
        except:
            return 0

    def _compute_determinism(self, series: np.ndarray) -> float:
        """Determinism from recurrence plot (simplified)."""
        try:
            # Proxy: autocorrelation strength
            acf = np.correlate(series - np.mean(series), series - np.mean(series), mode='full')
            acf = acf[len(acf)//2:]
            acf = acf / acf[0]

            # Sum of significant autocorrelations
            determinism = np.mean(np.abs(acf[1:min(20, len(acf))]))
            return np.clip(determinism, 0, 1)
        except:
            return 0

    def _compute_lyapunov(self, series: np.ndarray) -> float:
        """Largest Lyapunov exponent (simplified estimation)."""
        try:
            # Simple divergence rate estimation
            n = len(series)
            diffs = np.abs(np.diff(series))
            mean_diff = np.mean(diffs)

            # Proxy: rate of change variability
            lyap = np.log(np.std(diffs) / mean_diff) if mean_diff > 0 else 0
            return np.clip(lyap, -2, 2)
        except:
            return 0

    def _compute_wavelet_energy(self, series: np.ndarray) -> Dict[str, float]:
        """Wavelet energy by scale."""
        try:
            import pywt

            # Wavelet decomposition
            wavelet = 'db4'
            max_level = min(6, pywt.dwt_max_level(len(series), wavelet))
            coeffs = pywt.wavedec(series, wavelet, level=max_level)

            # Energy per scale
            energy = {}
            total_energy = 0

            for i, c in enumerate(coeffs):
                e = np.sum(c ** 2)
                scale_name = f'scale_{i}'
                energy[scale_name] = e
                total_energy += e

            # Normalize
            if total_energy > 0:
                energy = {k: v/total_energy for k, v in energy.items()}

            return energy
        except:
            return {'scale_0': 1.0}

    # =========================================================================
    # CLUSTER DETECTION
    # =========================================================================

    def _compute_correlation_matrix(
        self,
        indicators: List[str],
        start_date: datetime,
        end_date: datetime,
        data: pd.DataFrame = None
    ) -> pd.DataFrame:
        """Compute pairwise correlation matrix."""

        if data is not None:
            # Pivot data to wide format
            wide = data.pivot(columns='indicator_id', values='value')
            return wide[indicators].corr()

        # Load from database
        if self.db_path:
            import duckdb
            con = duckdb.connect(str(self.db_path), read_only=True)
            try:
                # Load all series
                price_data = {}
                for ind in indicators:
                    result = con.execute("""
                        SELECT date, value FROM data.indicators
                        WHERE indicator_id = ?
                        ORDER BY date
                    """, [ind]).fetchall()
                    if result:
                        price_data[ind] = {r[0]: r[1] for r in result}

                if len(price_data) >= 2:
                    # Find common dates
                    common_dates = set(list(price_data.values())[0].keys())
                    for d in price_data.values():
                        common_dates &= set(d.keys())
                    common_dates = sorted(common_dates)

                    if len(common_dates) >= 20:
                        # Build matrix
                        prices = np.array([
                            [price_data[ind][d] for d in common_dates]
                            for ind in indicators if ind in price_data
                        ])
                        valid_indicators = [ind for ind in indicators if ind in price_data]

                        # Returns
                        prices = np.maximum(prices, 1e-10)
                        returns = np.diff(np.log(prices), axis=1)

                        corr = np.corrcoef(returns)
                        return pd.DataFrame(corr, index=valid_indicators, columns=valid_indicators)
            finally:
                con.close()

        # Placeholder: identity matrix
        n = len(indicators)
        return pd.DataFrame(
            np.eye(n),
            index=indicators,
            columns=indicators
        )

    def _detect_clusters(
        self,
        indicators: List[str],
        corr_matrix: pd.DataFrame,
        profiles: Dict[str, IntrinsicProfile]
    ) -> Tuple[Dict[str, Cluster], List[str], Dict[str, str]]:
        """
        Detect clusters based on correlation.

        Returns:
            (clusters, singletons, assignments)
        """
        try:
            from scipy.cluster.hierarchy import linkage, fcluster
            from scipy.spatial.distance import squareform
        except ImportError:
            # Fallback without scipy
            logger.warning("scipy not available, skipping hierarchical clustering")
            assignments = {ind: 'singleton' for ind in indicators}
            return {}, indicators, assignments

        # Convert correlation to distance
        dist_matrix = 1 - corr_matrix.abs()

        # Hierarchical clustering
        try:
            condensed = squareform(dist_matrix.values, checks=False)
            Z = linkage(condensed, method='average')

            # Cut at threshold
            labels = fcluster(Z, t=1-self.CORRELATION_CLUSTER_THRESHOLD, criterion='distance')
        except:
            # Fallback: each indicator is its own cluster
            labels = np.arange(len(indicators)) + 1

        # Build clusters
        cluster_members = defaultdict(list)
        for ind, label in zip(indicators, labels):
            cluster_members[label].append(ind)

        clusters = {}
        singletons = []
        assignments = {}

        cluster_num = 0
        for label, members in cluster_members.items():
            if len(members) == 1:
                singletons.append(members[0])
                assignments[members[0]] = 'singleton'
            else:
                cluster_id = f'cluster_{cluster_num}'
                cluster_num += 1

                # Compute cluster properties
                cluster = self._build_cluster(
                    cluster_id, members, corr_matrix, profiles
                )
                clusters[cluster_id] = cluster

                for member in members:
                    assignments[member] = cluster_id

        return clusters, singletons, assignments

    def _build_cluster(
        self,
        cluster_id: str,
        members: List[str],
        corr_matrix: pd.DataFrame,
        profiles: Dict[str, IntrinsicProfile]
    ) -> Cluster:
        """Build Cluster object with properties."""

        # Intrinsic centroid
        vectors = [profiles[m].to_vector() for m in members if m in profiles]
        centroid_intrinsic = np.mean(vectors, axis=0) if vectors else np.zeros(8)

        # Mean internal correlation
        internal_corrs = []
        for i, m1 in enumerate(members):
            for m2 in members[i+1:]:
                if m1 in corr_matrix.index and m2 in corr_matrix.columns:
                    internal_corrs.append(abs(corr_matrix.loc[m1, m2]))

        mean_corr = np.mean(internal_corrs) if internal_corrs else 0

        # Internal radius (max distance from centroid)
        if vectors:
            distances = [np.linalg.norm(v - centroid_intrinsic) for v in vectors]
            internal_radius = max(distances)
            internal_variance = np.var(distances)
        else:
            internal_radius = 0
            internal_variance = 0

        return Cluster(
            cluster_id=cluster_id,
            members=members,
            centroid_intrinsic=centroid_intrinsic,
            centroid_raw=centroid_intrinsic,  # Simplified
            mass=float(len(members)),
            internal_radius=internal_radius,
            mean_internal_correlation=mean_corr,
            internal_variance=internal_variance,
            internal_structure={},
            internal_outliers={},
        )

    # =========================================================================
    # MACRO GEOMETRY (TELESCOPE)
    # =========================================================================

    def _compute_macro_geometry(
        self,
        clusters: Dict[str, Cluster],
        singletons: List[str],
        profiles: Dict[str, IntrinsicProfile]
    ) -> MacroGeometry:
        """Compute telescope view geometry."""

        entities = {}

        # Add clusters
        for cid, cluster in clusters.items():
            entities[cid] = {
                'type': 'cluster',
                'mass': cluster.mass,
                'position': cluster.centroid_intrinsic.tolist(),
            }

        # Add singletons
        for ind in singletons:
            if ind in profiles:
                entities[ind] = {
                    'type': 'singleton',
                    'mass': 1.0,
                    'position': profiles[ind].to_vector().tolist(),
                }

        # Compute inter-entity distances
        distances = {}
        entity_ids = list(entities.keys())

        for i, e1 in enumerate(entity_ids):
            for e2 in entity_ids[i+1:]:
                p1 = np.array(entities[e1]['position'])
                p2 = np.array(entities[e2]['position'])
                dist = np.linalg.norm(p1 - p2)
                distances[(e1, e2)] = dist

        # Compute gravitational influences
        influences = defaultdict(dict)
        for (e1, e2), dist in distances.items():
            if dist > 0:
                mass1 = entities[e1]['mass']
                mass2 = entities[e2]['mass']

                # Pull of e2 on e1
                influences[e1][e2] = mass2 / (dist ** 2)
                # Pull of e1 on e2
                influences[e2][e1] = mass1 / (dist ** 2)

        return MacroGeometry(
            entities=entities,
            distances=distances,
            influences=dict(influences),
            macro_clusters=[],  # Could add higher-level clustering
        )

    # =========================================================================
    # MICRO GEOMETRY (MICROSCOPE)
    # =========================================================================

    def _compute_micro_geometry(
        self,
        cluster: Cluster,
        profiles: Dict[str, IntrinsicProfile]
    ) -> Dict[str, MicroGeometry]:
        """
        Compute microscope view for one cluster.

        Returns structure under each lens.
        """
        micro = {}

        for lens in self.INTRINSIC_LENSES:
            try:
                geom = self._compute_lens_geometry(cluster, profiles, lens)
                micro[lens] = geom
            except Exception as e:
                logger.warning(f"Failed to compute {lens} geometry: {e}")

        return micro

    def _compute_lens_geometry(
        self,
        cluster: Cluster,
        profiles: Dict[str, IntrinsicProfile],
        lens: str
    ) -> MicroGeometry:
        """Compute geometry under one lens."""

        members = cluster.members

        # Get lens-specific values
        values = {}
        for m in members:
            if m in profiles:
                profile = profiles[m]
                if lens == 'entropy':
                    values[m] = profile.entropy
                elif lens == 'hurst':
                    values[m] = profile.hurst
                elif lens == 'kurtosis':
                    values[m] = profile.kurtosis
                elif lens == 'permutation_entropy':
                    values[m] = profile.permutation_entropy
                elif lens == 'spectral_entropy':
                    values[m] = profile.spectral_entropy
                elif lens == 'recurrence':
                    values[m] = profile.recurrence_rate
                elif lens == 'determinism':
                    values[m] = profile.determinism
                elif lens == 'wavelet':
                    # Use dominant scale as proxy
                    energy = profile.wavelet_energy_by_scale
                    values[m] = max(energy.values()) if energy else 0
                else:
                    values[m] = 0

        if not values:
            return self._empty_micro_geometry(cluster.cluster_id, lens)

        # Convert to positions (1D for single lens)
        positions = {m: np.array([v]) for m, v in values.items()}

        # Pairwise distances
        distances = {}
        member_list = list(values.keys())
        for i, m1 in enumerate(member_list):
            for m2 in member_list[i+1:]:
                dist = abs(values[m1] - values[m2])
                distances[(m1, m2)] = dist

        # Nearest neighbors
        nearest = {}
        for m in member_list:
            min_dist = float('inf')
            nearest_m = None
            for m2 in member_list:
                if m != m2 and (m, m2) in distances:
                    d = distances[(m, m2)]
                elif m != m2 and (m2, m) in distances:
                    d = distances[(m2, m)]
                else:
                    continue
                if d < min_dist:
                    min_dist = d
                    nearest_m = m2
            nearest[m] = nearest_m

        # Outlier detection
        vals = list(values.values())
        mean_val = np.mean(vals)
        std_val = np.std(vals)

        outliers = []
        outlier_scores = {}
        for m, v in values.items():
            z = abs(v - mean_val) / std_val if std_val > 0 else 0
            outlier_scores[m] = z
            if z > self.OUTLIER_THRESHOLD:
                outliers.append(m)

        # Subclusters (simplified: above/below mean)
        if std_val > 0.1:  # Meaningful variation
            above = [m for m, v in values.items() if v > mean_val]
            below = [m for m, v in values.items() if v <= mean_val]
            subclusters = [above, below] if above and below else [member_list]
        else:
            subclusters = [member_list]

        # Homogeneity
        homogeneous = std_val < 0.1 or len(outliers) == 0

        # Description
        if homogeneous:
            description = f"Uniform under {lens} lens (all values ~{mean_val:.2f})"
        elif outliers:
            description = f"Outliers in {lens}: {', '.join(outliers)} (z > {self.OUTLIER_THRESHOLD})"
        else:
            description = f"Varied {lens} structure: range {min(vals):.2f} to {max(vals):.2f}"

        return MicroGeometry(
            cluster_id=cluster.cluster_id,
            lens_name=lens,
            positions=positions,
            pairwise_distances=distances,
            nearest_neighbors=nearest,
            subclusters=subclusters,
            outliers=outliers,
            outlier_scores=outlier_scores,
            homogeneous=homogeneous,
            description=description,
        )

    def _empty_micro_geometry(self, cluster_id: str, lens: str) -> MicroGeometry:
        """Return empty MicroGeometry."""
        return MicroGeometry(
            cluster_id=cluster_id,
            lens_name=lens,
            positions={},
            pairwise_distances={},
            nearest_neighbors={},
            subclusters=[],
            outliers=[],
            outlier_scores={},
            homogeneous=True,
            description="No data",
        )

    # =========================================================================
    # LENS DIVERGENCE DETECTION
    # =========================================================================

    def _detect_divergences(
        self,
        indicators: List[str],
        corr_matrix: pd.DataFrame,
        profiles: Dict[str, IntrinsicProfile]
    ) -> List[LensDivergence]:
        """
        Find pairs that are:
        - Correlated but structurally different
        - Uncorrelated but structurally similar
        """
        divergences = []

        for i, ind1 in enumerate(indicators):
            for ind2 in indicators[i+1:]:
                # Get correlation
                try:
                    corr = corr_matrix.loc[ind1, ind2]
                except:
                    continue

                correlated = abs(corr) > self.CORRELATION_CLUSTER_THRESHOLD

                # Check each intrinsic lens
                for lens in self.INTRINSIC_LENSES:
                    try:
                        dist = self._lens_distance(ind1, ind2, lens, profiles)
                        similar = dist < self.SIMILARITY_THRESHOLD

                        if correlated and not similar:
                            divergences.append(LensDivergence(
                                indicator_a=ind1,
                                indicator_b=ind2,
                                correlation=corr,
                                correlated=True,
                                lens_name=lens,
                                lens_distance=dist,
                                lens_similar=False,
                                divergence_type='correlated_but_different',
                                interpretation=f"{ind1} and {ind2} move together (r={corr:.2f}) but have different {lens} profiles (dist={dist:.2f})"
                            ))

                        elif not correlated and similar:
                            divergences.append(LensDivergence(
                                indicator_a=ind1,
                                indicator_b=ind2,
                                correlation=corr,
                                correlated=False,
                                lens_name=lens,
                                lens_distance=dist,
                                lens_similar=True,
                                divergence_type='uncorrelated_but_similar',
                                interpretation=f"{ind1} and {ind2} move independently (r={corr:.2f}) but share similar {lens} structure (dist={dist:.2f})"
                            ))
                    except:
                        continue

        return divergences

    def _lens_distance(
        self,
        ind1: str,
        ind2: str,
        lens: str,
        profiles: Dict[str, IntrinsicProfile]
    ) -> float:
        """Compute distance between two indicators under one lens."""

        p1 = profiles.get(ind1)
        p2 = profiles.get(ind2)

        if not p1 or not p2:
            return float('inf')

        if lens == 'entropy':
            return abs(p1.entropy - p2.entropy)
        elif lens == 'hurst':
            return abs(p1.hurst - p2.hurst)
        elif lens == 'kurtosis':
            return abs(p1.kurtosis - p2.kurtosis) / 10  # Scale
        elif lens == 'permutation_entropy':
            return abs(p1.permutation_entropy - p2.permutation_entropy)
        elif lens == 'spectral_entropy':
            return abs(p1.spectral_entropy - p2.spectral_entropy)
        elif lens == 'recurrence':
            return abs(p1.recurrence_rate - p2.recurrence_rate)
        elif lens == 'determinism':
            return abs(p1.determinism - p2.determinism)
        else:
            return float('inf')

    # =========================================================================
    # FINDINGS GENERATION
    # =========================================================================

    def _generate_findings(
        self,
        clusters: Dict[str, Cluster],
        singletons: List[str],
        profiles: Dict[str, IntrinsicProfile],
        divergences: List[LensDivergence]
    ) -> List[Dict[str, Any]]:
        """Generate cross-level findings."""

        findings = []

        # Finding: Singletons that match cluster members intrinsically
        for singleton in singletons:
            if singleton not in profiles:
                continue

            s_profile = profiles[singleton]
            s_vector = s_profile.to_vector()

            for cid, cluster in clusters.items():
                for member in cluster.members:
                    if member not in profiles:
                        continue

                    m_vector = profiles[member].to_vector()
                    dist = np.linalg.norm(s_vector - m_vector)

                    if dist < 0.5:  # Threshold
                        findings.append({
                            'type': 'singleton_matches_cluster_member',
                            'description': f"Singleton {singleton} has similar intrinsic profile to {member} in {cid}, but they are uncorrelated",
                            'indicators': [singleton, member],
                            'distance': dist,
                        })

        # Finding: Cluster members that are outliers in multiple lenses
        for cid, cluster in clusters.items():
            outlier_counts = defaultdict(int)

            if hasattr(cluster, 'internal_structure'):
                for lens, micro in cluster.internal_structure.items():
                    for outlier in micro.outliers:
                        outlier_counts[outlier] += 1

            for member, count in outlier_counts.items():
                if count >= 3:  # Outlier in 3+ lenses
                    findings.append({
                        'type': 'multi_lens_outlier',
                        'description': f"{member} is an outlier in {count} lenses within {cid}",
                        'indicators': [member],
                        'cluster': cid,
                        'outlier_count': count,
                    })

        # Finding: High divergence count for a pair
        pair_counts = defaultdict(int)
        for d in divergences:
            pair = tuple(sorted([d.indicator_a, d.indicator_b]))
            pair_counts[pair] += 1

        for pair, count in pair_counts.items():
            if count >= 3:
                findings.append({
                    'type': 'high_divergence_pair',
                    'description': f"{pair[0]} and {pair[1]} diverge in {count} lenses",
                    'indicators': list(pair),
                    'divergence_count': count,
                })

        return findings

    # =========================================================================
    # DATA LOADING
    # =========================================================================

    def _load_series(
        self,
        indicator_id: str,
        start_date: datetime,
        end_date: datetime
    ) -> np.ndarray:
        """Load time series from database."""

        if self.db_path is None:
            raise ValueError("No database path configured")

        import duckdb
        con = duckdb.connect(str(self.db_path), read_only=True)

        try:
            result = con.execute("""
                SELECT value
                FROM data.indicators
                WHERE indicator_id = ?
                ORDER BY date
            """, [indicator_id]).fetchdf()

            return result['value'].values
        finally:
            con.close()


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def analyze_hierarchical_geometry(
    db_path: str,
    indicators: List[str],
    start_date: str,
    end_date: str,
    output_path: str = None
) -> HierarchicalGeometryResult:
    """
    One-liner for hierarchical geometry analysis.

    Usage:
        result = analyze_hierarchical_geometry(
            'prism.db',
            ['TLT', 'IEF', 'SPY', 'VIX'],
            '2020-01-01',
            '2023-12-31',
            'hierarchy_report.json'
        )
        print(result.full_report())
    """
    engine = HierarchicalGeometryEngine(db_path)

    result = engine.compute_hierarchical_geometry(
        indicators=indicators,
        start_date=datetime.strptime(start_date, '%Y-%m-%d'),
        end_date=datetime.strptime(end_date, '%Y-%m-%d'),
    )

    if output_path:
        with open(output_path, 'w') as f:
            json.dump(result.to_dict(), f, indent=2, default=str)
        print(f"Saved to {output_path}")

    return result


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s'
    )

    parser = argparse.ArgumentParser(description="PRISM Hierarchical Geometry Engine")

    parser.add_argument('--db', type=str, default='data/prism.duckdb', help='Database path')
    parser.add_argument('--indicators', type=str, nargs='+', help='Indicator IDs')
    parser.add_argument('--start', type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, help='End date (YYYY-MM-DD)')
    parser.add_argument('--output', '-o', type=str, help='Output JSON path')
    parser.add_argument('--report', action='store_true', help='Print full report')

    args = parser.parse_args()

    if args.indicators and args.start and args.end:
        result = analyze_hierarchical_geometry(
            args.db,
            args.indicators,
            args.start,
            args.end,
            args.output
        )

        if args.report:
            print(result.full_report())
    else:
        parser.print_help()
        print("\nExample:")
        print("  python agent_hierarchical_geometry.py --db data/prism.duckdb \\")
        print("    --indicators TLT IEF SHY SPY QQQ VIX GLD \\")
        print("    --start 2020-01-01 --end 2023-12-31 \\")
        print("    --output hierarchy.json --report")
