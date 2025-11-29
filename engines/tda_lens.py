"""
Topological Data Analysis (TDA) Lens
====================================
Analyzes the 'shape' of data to uncover recurring patterns, regime shifts,
and tipping points that traditional methods miss.

Key Applications:
- Detecting market regime transitions
- Identifying climate tipping points
- Finding recurring dynamical patterns
- Crash prediction through topological changes

Methods:
- Persistent Homology: Multi-scale topological features
- Takens Embedding: Time series to point cloud
- Persistence Diagrams/Landscapes: Topological signatures
- Betti Numbers: Counting topological features
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster
import warnings


@dataclass
class PersistencePoint:
    """A point in a persistence diagram."""
    birth: float       # Scale at which feature appears
    death: float       # Scale at which feature disappears
    dimension: int     # Homological dimension (0=components, 1=loops, 2=voids)
    persistence: float # death - birth
    
    def __post_init__(self):
        self.persistence = self.death - self.birth


@dataclass
class TDAResult:
    """Container for TDA results."""
    persistence_diagram: List[PersistencePoint]
    betti_numbers: Dict[int, List[int]]  # dimension -> betti numbers at each scale
    persistence_landscape: Optional[np.ndarray]
    total_persistence: Dict[int, float]  # dimension -> sum of persistence
    max_persistence: Dict[int, float]    # dimension -> max persistence


class TakensEmbedding:
    """
    Takens time-delay embedding for time series.
    
    Reconstructs the state space attractor from a single time series
    using the method of delays.
    """
    
    def __init__(
        self,
        dimension: int = 3,
        delay: Optional[int] = None,
        delay_method: str = 'auto'
    ):
        """
        Initialize Takens embedding.
        
        Parameters
        ----------
        dimension : int
            Embedding dimension (typically 2-10)
        delay : int, optional
            Time delay tau. If None, computed automatically.
        delay_method : str
            Method for automatic delay: 'auto', 'mutual_info', 'autocorr'
        """
        self.dimension = dimension
        self.delay = delay
        self.delay_method = delay_method
    
    def _estimate_delay_autocorr(self, data: np.ndarray) -> int:
        """Estimate delay using first zero-crossing of autocorrelation."""
        n = len(data)
        max_lag = min(n // 4, 100)
        
        # Compute autocorrelation
        data_centered = data - np.mean(data)
        autocorr = np.correlate(data_centered, data_centered, mode='full')
        autocorr = autocorr[n-1:n-1+max_lag] / autocorr[n-1]
        
        # Find first zero crossing
        for i in range(1, len(autocorr)):
            if autocorr[i] <= 0:
                return i
        
        return max_lag // 4
    
    def _estimate_delay_mutual_info(self, data: np.ndarray) -> int:
        """Estimate delay using first minimum of mutual information."""
        n = len(data)
        max_lag = min(n // 4, 100)
        n_bins = max(10, int(np.sqrt(n / 5)))
        
        mi_values = []
        
        for lag in range(1, max_lag):
            # Discretize for mutual information
            x = data[:-lag]
            y = data[lag:]
            
            # Joint histogram
            hist_2d, _, _ = np.histogram2d(x, y, bins=n_bins)
            hist_2d = hist_2d / hist_2d.sum()
            
            # Marginals
            px = hist_2d.sum(axis=1)
            py = hist_2d.sum(axis=0)
            
            # Mutual information
            mi = 0
            for i in range(n_bins):
                for j in range(n_bins):
                    if hist_2d[i, j] > 0:
                        mi += hist_2d[i, j] * np.log(
                            hist_2d[i, j] / (px[i] * py[j] + 1e-10) + 1e-10
                        )
            mi_values.append(mi)
        
        # Find first local minimum
        for i in range(1, len(mi_values) - 1):
            if mi_values[i] < mi_values[i-1] and mi_values[i] < mi_values[i+1]:
                return i + 1
        
        return np.argmin(mi_values) + 1
    
    def embed(self, data: np.ndarray) -> np.ndarray:
        """
        Embed time series into higher-dimensional space.
        
        Parameters
        ----------
        data : np.ndarray
            1D time series
            
        Returns
        -------
        np.ndarray
            Embedded point cloud (n_points × dimension)
        """
        data = np.asarray(data).flatten()
        n = len(data)
        
        # Estimate delay if not provided
        if self.delay is None:
            if self.delay_method == 'mutual_info':
                self.delay = self._estimate_delay_mutual_info(data)
            else:  # autocorr or auto
                self.delay = self._estimate_delay_autocorr(data)
        
        # Compute embedding
        n_points = n - (self.dimension - 1) * self.delay
        
        if n_points <= 0:
            raise ValueError("Time series too short for this embedding")
        
        embedded = np.zeros((n_points, self.dimension))
        
        for i in range(self.dimension):
            start = i * self.delay
            end = start + n_points
            embedded[:, i] = data[start:end]
        
        return embedded
    
    def estimate_dimension(
        self,
        data: np.ndarray,
        max_dim: int = 10
    ) -> Dict:
        """
        Estimate optimal embedding dimension using false nearest neighbors.
        
        Parameters
        ----------
        data : np.ndarray
            Time series
        max_dim : int
            Maximum dimension to test
            
        Returns
        -------
        dict
            Dimension estimates and FNN ratios
        """
        data = np.asarray(data).flatten()
        
        if self.delay is None:
            self.delay = self._estimate_delay_autocorr(data)
        
        fnn_ratios = []
        rtol = 15.0  # Distance ratio tolerance
        atol = 2.0   # Absolute tolerance (std multiplier)
        std = np.std(data)
        
        for dim in range(1, max_dim + 1):
            # Embed in dim and dim+1
            n_points = len(data) - dim * self.delay
            if n_points < 10:
                break
            
            embed_d = np.zeros((n_points, dim))
            for i in range(dim):
                start = i * self.delay
                embed_d[:, i] = data[start:start + n_points]
            
            # Find nearest neighbors in dim
            dists = squareform(pdist(embed_d))
            np.fill_diagonal(dists, np.inf)
            nn_idx = np.argmin(dists, axis=1)
            nn_dists = dists[np.arange(n_points), nn_idx]
            
            # Check if they're still neighbors in dim+1
            if n_points - self.delay <= 0:
                break
            
            # Extra dimension values
            extra = data[dim * self.delay:dim * self.delay + n_points]
            extra_nn = data[dim * self.delay:dim * self.delay + n_points][nn_idx[:len(extra)]]
            
            # False neighbor criteria
            n_valid = min(len(extra), len(nn_idx))
            extra = extra[:n_valid]
            extra_nn = extra_nn[:n_valid]
            nn_dists = nn_dists[:n_valid]
            
            # Distance in extra dimension
            extra_dist = np.abs(extra - extra_nn)
            
            # FNN if ratio too large or absolute distance too large
            is_fnn = ((extra_dist / (nn_dists + 1e-10) > rtol) | 
                     (extra_dist > atol * std))
            
            fnn_ratio = np.mean(is_fnn)
            fnn_ratios.append(fnn_ratio)
            
            # Stop if FNN ratio is small enough
            if fnn_ratio < 0.01:
                break
        
        # Optimal dimension: first where FNN < 5%
        optimal = 1
        for i, ratio in enumerate(fnn_ratios):
            if ratio < 0.05:
                optimal = i + 1
                break
        else:
            optimal = len(fnn_ratios)
        
        return {
            'optimal_dimension': optimal,
            'fnn_ratios': fnn_ratios,
            'delay': self.delay
        }


class VietorisRipsComplex:
    """
    Vietoris-Rips complex construction for persistent homology.
    
    Builds a simplicial complex from a point cloud by connecting
    points within a given distance threshold.
    """
    
    def __init__(self, max_dimension: int = 2):
        """
        Initialize Vietoris-Rips complex builder.
        
        Parameters
        ----------
        max_dimension : int
            Maximum homological dimension to compute (0, 1, or 2)
        """
        self.max_dimension = min(max_dimension, 2)  # Limit for efficiency
    
    def _compute_distance_matrix(self, points: np.ndarray) -> np.ndarray:
        """Compute pairwise distance matrix."""
        return squareform(pdist(points))
    
    def compute_persistence(
        self,
        points: np.ndarray,
        max_edge_length: Optional[float] = None,
        n_steps: int = 100
    ) -> List[PersistencePoint]:
        """
        Compute persistent homology using a filtration.
        
        Parameters
        ----------
        points : np.ndarray
            Point cloud (n_points × dimension)
        max_edge_length : float, optional
            Maximum edge length. If None, uses max distance.
        n_steps : int
            Number of filtration steps
            
        Returns
        -------
        list
            Persistence points (birth, death, dimension)
        """
        n_points = len(points)
        dist_matrix = self._compute_distance_matrix(points)
        
        if max_edge_length is None:
            max_edge_length = np.max(dist_matrix)
        
        # Filtration values
        filtration_values = np.linspace(0, max_edge_length, n_steps)
        
        persistence_points = []
        
        # Track connected components (H0)
        component_birth = {i: 0.0 for i in range(n_points)}
        
        # Union-Find for connected components
        parent = list(range(n_points))
        rank = [0] * n_points
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y, threshold):
            px, py = find(x), find(y)
            if px == py:
                return False
            
            # Merge: younger component dies
            if component_birth[px] > component_birth[py]:
                px, py = py, px
            
            # py dies (younger), px survives
            birth = component_birth[py]
            persistence_points.append(PersistencePoint(
                birth=birth,
                death=threshold,
                dimension=0,
                persistence=threshold - birth
            ))
            
            if rank[px] < rank[py]:
                px, py = py, px
            parent[py] = px
            if rank[px] == rank[py]:
                rank[px] += 1
            
            return True
        
        # Process edges in order of length
        edges = []
        for i in range(n_points):
            for j in range(i + 1, n_points):
                edges.append((dist_matrix[i, j], i, j))
        edges.sort()
        
        edge_idx = 0
        
        # H1 tracking (simplified: track triangles)
        triangles_checked = set()
        active_loops = {}  # edge tuple -> birth time
        
        for threshold in filtration_values:
            # Add edges up to this threshold
            while edge_idx < len(edges) and edges[edge_idx][0] <= threshold:
                dist, i, j = edges[edge_idx]
                
                # Check if this creates new connectivity
                union(i, j, threshold)
                
                edge_idx += 1
            
            # Simple H1 detection: check for new cycles
            if self.max_dimension >= 1:
                # Get current adjacency
                adjacency = dist_matrix <= threshold
                
                # Look for triangles (which kill loops)
                for i in range(n_points):
                    neighbors_i = np.where(adjacency[i])[0]
                    for j in neighbors_i:
                        if j <= i:
                            continue
                        neighbors_j = np.where(adjacency[j])[0]
                        common = np.intersect1d(neighbors_i, neighbors_j)
                        
                        for k in common:
                            if k <= j:
                                continue
                            
                            tri = (min(i, j, k), sorted([i, j, k])[1], max(i, j, k))
                            if tri not in triangles_checked:
                                triangles_checked.add(tri)
                                
                                # This triangle might kill a loop
                                # Simplified: record as H1 feature
                                edge_lengths = [
                                    dist_matrix[i, j],
                                    dist_matrix[j, k],
                                    dist_matrix[i, k]
                                ]
                                birth = sorted(edge_lengths)[1]  # Second edge creates loop
                                death = max(edge_lengths)  # Third edge kills loop
                                
                                if death > birth:
                                    persistence_points.append(PersistencePoint(
                                        birth=birth,
                                        death=death,
                                        dimension=1,
                                        persistence=death - birth
                                    ))
        
        # Add infinite persistence for surviving component
        roots = set(find(i) for i in range(n_points))
        for root in roots:
            persistence_points.append(PersistencePoint(
                birth=component_birth[root],
                death=max_edge_length,
                dimension=0,
                persistence=max_edge_length - component_birth[root]
            ))
        
        return persistence_points
    
    def compute_betti_numbers(
        self,
        points: np.ndarray,
        filtration_values: np.ndarray
    ) -> Dict[int, np.ndarray]:
        """
        Compute Betti numbers across filtration.
        
        Betti_0 = number of connected components
        Betti_1 = number of loops
        Betti_2 = number of voids
        
        Parameters
        ----------
        points : np.ndarray
            Point cloud
        filtration_values : np.ndarray
            Scale values at which to compute Betti numbers
            
        Returns
        -------
        dict
            dimension -> array of Betti numbers
        """
        persistence = self.compute_persistence(points)
        
        betti = {d: np.zeros(len(filtration_values)) for d in range(self.max_dimension + 1)}
        
        for i, threshold in enumerate(filtration_values):
            for point in persistence:
                if point.birth <= threshold < point.death:
                    betti[point.dimension][i] += 1
        
        return betti


class PersistenceLandscape:
    """
    Compute persistence landscapes from persistence diagrams.
    
    Persistence landscapes are stable, functional representations
    of persistence diagrams that support statistical analysis.
    """
    
    def __init__(self, n_landscapes: int = 5, resolution: int = 100):
        """
        Initialize persistence landscape calculator.
        
        Parameters
        ----------
        n_landscapes : int
            Number of landscape functions to compute
        resolution : int
            Number of points in landscape
        """
        self.n_landscapes = n_landscapes
        self.resolution = resolution
    
    def compute(
        self,
        persistence_points: List[PersistencePoint],
        dimension: int = 1
    ) -> np.ndarray:
        """
        Compute persistence landscape.
        
        Parameters
        ----------
        persistence_points : list
            Persistence diagram points
        dimension : int
            Homological dimension to use
            
        Returns
        -------
        np.ndarray
            Landscape functions (n_landscapes × resolution)
        """
        # Filter points by dimension
        points = [(p.birth, p.death) for p in persistence_points 
                  if p.dimension == dimension and p.death > p.birth]
        
        if not points:
            return np.zeros((self.n_landscapes, self.resolution))
        
        # Determine range
        all_values = [v for p in points for v in p]
        t_min = min(all_values)
        t_max = max(all_values)
        
        t_values = np.linspace(t_min, t_max, self.resolution)
        
        # Compute tent functions for each point
        tents = np.zeros((len(points), self.resolution))
        
        for i, (birth, death) in enumerate(points):
            midpoint = (birth + death) / 2
            half_life = (death - birth) / 2
            
            for j, t in enumerate(t_values):
                if birth <= t <= midpoint:
                    tents[i, j] = t - birth
                elif midpoint < t <= death:
                    tents[i, j] = death - t
                else:
                    tents[i, j] = 0
        
        # Sort at each t to get landscapes
        landscapes = np.zeros((self.n_landscapes, self.resolution))
        
        for j in range(self.resolution):
            sorted_vals = np.sort(tents[:, j])[::-1]
            for k in range(min(self.n_landscapes, len(sorted_vals))):
                landscapes[k, j] = sorted_vals[k]
        
        return landscapes


class TDALens:
    """
    Topological Data Analysis Lens for comprehensive shape-based analysis.
    
    Combines Takens embedding, persistent homology, and persistence
    landscapes to extract topological features from time series.
    """
    
    def __init__(self):
        """Initialize the TDA Lens."""
        self.embedding = None
        self.complex = None
        self.landscape = None
    
    def embed_time_series(
        self,
        data: np.ndarray,
        dimension: Optional[int] = None,
        delay: Optional[int] = None
    ) -> np.ndarray:
        """
        Embed time series using Takens embedding.
        
        Parameters
        ----------
        data : np.ndarray
            1D time series
        dimension : int, optional
            Embedding dimension
        delay : int, optional
            Time delay
            
        Returns
        -------
        np.ndarray
            Embedded point cloud
        """
        if dimension is None:
            # Estimate dimension
            temp_embed = TakensEmbedding(dimension=3, delay=delay)
            dim_result = temp_embed.estimate_dimension(data)
            dimension = dim_result['optimal_dimension']
            delay = dim_result['delay']
        
        self.embedding = TakensEmbedding(dimension=dimension, delay=delay)
        return self.embedding.embed(data)
    
    def compute_persistence(
        self,
        points: np.ndarray,
        max_dimension: int = 1,
        max_edge_length: Optional[float] = None
    ) -> TDAResult:
        """
        Compute persistent homology of point cloud.
        
        Parameters
        ----------
        points : np.ndarray
            Point cloud
        max_dimension : int
            Maximum homological dimension
        max_edge_length : float, optional
            Maximum filtration value
            
        Returns
        -------
        TDAResult
        """
        self.complex = VietorisRipsComplex(max_dimension=max_dimension)
        
        persistence_points = self.complex.compute_persistence(
            points, max_edge_length
        )
        
        # Compute filtration values for Betti numbers
        if max_edge_length is None:
            max_edge_length = max(p.death for p in persistence_points)
        
        filtration = np.linspace(0, max_edge_length, 50)
        betti = self.complex.compute_betti_numbers(points, filtration)
        
        # Compute persistence landscape
        self.landscape = PersistenceLandscape()
        landscape = self.landscape.compute(persistence_points, dimension=1)
        
        # Summary statistics
        total_persistence = {}
        max_persistence = {}
        
        for dim in range(max_dimension + 1):
            dim_points = [p for p in persistence_points if p.dimension == dim]
            if dim_points:
                total_persistence[dim] = sum(p.persistence for p in dim_points)
                max_persistence[dim] = max(p.persistence for p in dim_points)
            else:
                total_persistence[dim] = 0
                max_persistence[dim] = 0
        
        return TDAResult(
            persistence_diagram=persistence_points,
            betti_numbers=betti,
            persistence_landscape=landscape,
            total_persistence=total_persistence,
            max_persistence=max_persistence
        )
    
    def detect_regime_changes(
        self,
        data: np.ndarray,
        window_size: int = 100,
        step: int = 10
    ) -> Dict:
        """
        Detect regime changes using sliding window TDA.
        
        Parameters
        ----------
        data : np.ndarray
            Time series
        window_size : int
            Size of sliding window
        step : int
            Step size between windows
            
        Returns
        -------
        dict
            Topological features over time
        """
        data = np.asarray(data).flatten()
        n = len(data)
        
        # Compute features for each window
        window_starts = list(range(0, n - window_size, step))
        
        features = {
            'window_start': [],
            'total_persistence_h0': [],
            'total_persistence_h1': [],
            'max_persistence_h1': [],
            'n_significant_loops': []
        }
        
        for start in window_starts:
            window = data[start:start + window_size]
            
            try:
                # Embed window
                points = self.embed_time_series(window, dimension=3)
                
                # Compute persistence
                result = self.compute_persistence(points, max_dimension=1)
                
                features['window_start'].append(start)
                features['total_persistence_h0'].append(result.total_persistence.get(0, 0))
                features['total_persistence_h1'].append(result.total_persistence.get(1, 0))
                features['max_persistence_h1'].append(result.max_persistence.get(1, 0))
                
                # Count significant loops (persistence > threshold)
                threshold = result.max_persistence.get(1, 0) * 0.1
                sig_loops = sum(1 for p in result.persistence_diagram 
                               if p.dimension == 1 and p.persistence > threshold)
                features['n_significant_loops'].append(sig_loops)
                
            except Exception as e:
                # Skip problematic windows
                continue
        
        # Convert to arrays
        for key in features:
            features[key] = np.array(features[key])
        
        # Detect regime changes as large changes in topological features
        if len(features['total_persistence_h1']) > 1:
            h1_changes = np.abs(np.diff(features['total_persistence_h1']))
            threshold = np.mean(h1_changes) + 2 * np.std(h1_changes)
            regime_changes = features['window_start'][1:][h1_changes > threshold]
            features['detected_regime_changes'] = regime_changes
        else:
            features['detected_regime_changes'] = np.array([])
        
        return features
    
    def compare_attractors(
        self,
        data1: np.ndarray,
        data2: np.ndarray,
        dimension: int = 3
    ) -> Dict:
        """
        Compare topological structure of two time series.
        
        Parameters
        ----------
        data1, data2 : np.ndarray
            Time series to compare
        dimension : int
            Embedding dimension
            
        Returns
        -------
        dict
            Comparison metrics
        """
        # Embed both
        points1 = self.embed_time_series(data1, dimension=dimension)
        points2 = self.embed_time_series(data2, dimension=dimension)
        
        # Compute persistence
        result1 = self.compute_persistence(points1)
        result2 = self.compute_persistence(points2)
        
        # Compute landscape distance
        self.landscape = PersistenceLandscape()
        landscape1 = self.landscape.compute(result1.persistence_diagram, dimension=1)
        landscape2 = self.landscape.compute(result2.persistence_diagram, dimension=1)
        
        landscape_distance = np.sqrt(np.sum((landscape1 - landscape2) ** 2))
        
        # Compare summary statistics
        comparison = {
            'landscape_distance': landscape_distance,
            'total_persistence_diff_h0': abs(
                result1.total_persistence.get(0, 0) - result2.total_persistence.get(0, 0)
            ),
            'total_persistence_diff_h1': abs(
                result1.total_persistence.get(1, 0) - result2.total_persistence.get(1, 0)
            ),
            'max_persistence_diff_h1': abs(
                result1.max_persistence.get(1, 0) - result2.max_persistence.get(1, 0)
            ),
            'series1': {
                'total_h0': result1.total_persistence.get(0, 0),
                'total_h1': result1.total_persistence.get(1, 0),
                'max_h1': result1.max_persistence.get(1, 0)
            },
            'series2': {
                'total_h0': result2.total_persistence.get(0, 0),
                'total_h1': result2.total_persistence.get(1, 0),
                'max_h1': result2.max_persistence.get(1, 0)
            }
        }
        
        return comparison
    
    def analyze(
        self,
        data: Union[np.ndarray, pd.Series],
        embedding_dim: Optional[int] = None,
        max_homology_dim: int = 1,
        detect_regimes: bool = True,
        window_size: int = 100
    ) -> Dict:
        """
        Comprehensive TDA analysis of time series.
        
        Parameters
        ----------
        data : array-like
            Time series data
        embedding_dim : int, optional
            Embedding dimension (auto if None)
        max_homology_dim : int
            Maximum homological dimension
        detect_regimes : bool
            Perform regime change detection
        window_size : int
            Window size for regime detection
            
        Returns
        -------
        dict
            Complete TDA analysis
        """
        if isinstance(data, pd.Series):
            data = data.values
        data = np.asarray(data).flatten()
        
        results = {
            'n_observations': len(data)
        }
        
        # Estimate embedding parameters
        temp_embed = TakensEmbedding(dimension=3)
        dim_estimate = temp_embed.estimate_dimension(data)
        
        results['embedding'] = {
            'optimal_dimension': dim_estimate['optimal_dimension'],
            'delay': dim_estimate['delay'],
            'fnn_ratios': dim_estimate['fnn_ratios']
        }
        
        # Use estimated or provided dimension
        if embedding_dim is None:
            embedding_dim = dim_estimate['optimal_dimension']
        
        # Embed time series
        points = self.embed_time_series(
            data,
            dimension=embedding_dim,
            delay=dim_estimate['delay']
        )
        
        results['embedded_points'] = {
            'n_points': len(points),
            'dimension': embedding_dim
        }
        
        # Compute persistence
        persistence_result = self.compute_persistence(
            points,
            max_dimension=max_homology_dim
        )
        
        results['persistence'] = {
            'n_features': len(persistence_result.persistence_diagram),
            'total_persistence': persistence_result.total_persistence,
            'max_persistence': persistence_result.max_persistence,
            'betti_numbers': persistence_result.betti_numbers
        }
        
        # Persistence diagram summary
        diagram_summary = {}
        for dim in range(max_homology_dim + 1):
            dim_points = [p for p in persistence_result.persistence_diagram 
                         if p.dimension == dim]
            diagram_summary[f'H{dim}'] = {
                'n_features': len(dim_points),
                'births': [p.birth for p in dim_points],
                'deaths': [p.death for p in dim_points],
                'persistence': [p.persistence for p in dim_points]
            }
        results['diagram_summary'] = diagram_summary
        
        # Persistence landscape
        if persistence_result.persistence_landscape is not None:
            results['landscape'] = {
                'shape': persistence_result.persistence_landscape.shape,
                'max_values': np.max(persistence_result.persistence_landscape, axis=1).tolist()
            }
        
        # Regime detection
        if detect_regimes and len(data) > 2 * window_size:
            results['regime_detection'] = self.detect_regime_changes(
                data, window_size=window_size
            )
        
        return results


# Convenience function
def tda_analysis(
    data: np.ndarray,
    embedding_dim: int = 3,
    max_homology_dim: int = 1
) -> Dict:
    """
    Quick TDA analysis of time series.
    
    Parameters
    ----------
    data : np.ndarray
        Time series
    embedding_dim : int
        Embedding dimension
    max_homology_dim : int
        Maximum homological dimension
        
    Returns
    -------
    dict
        TDA analysis results
    """
    lens = TDALens()
    return lens.analyze(data, embedding_dim, max_homology_dim)


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    
    # Generate data with different dynamical regimes
    n = 1000
    t = np.linspace(0, 20 * np.pi, n)
    
    # First half: simple oscillation
    data1 = np.sin(t[:n//2]) + 0.1 * np.random.randn(n//2)
    
    # Second half: more complex dynamics (quasi-periodic)
    data2 = np.sin(t[n//2:]) + 0.5 * np.sin(2.3 * t[n//2:]) + 0.1 * np.random.randn(n//2)
    
    data = np.concatenate([data1, data2])
    
    # Analyze
    lens = TDALens()
    results = lens.analyze(data, embedding_dim=3, detect_regimes=True)
    
    print("Topological Data Analysis Complete")
    print(f"\nEmbedding Parameters:")
    print(f"  Optimal dimension: {results['embedding']['optimal_dimension']}")
    print(f"  Delay: {results['embedding']['delay']}")
    
    print(f"\nEmbedded Point Cloud:")
    print(f"  Number of points: {results['embedded_points']['n_points']}")
    
    print(f"\nPersistence Summary:")
    print(f"  Total features: {results['persistence']['n_features']}")
    for dim, total in results['persistence']['total_persistence'].items():
        print(f"  H{dim} total persistence: {total:.4f}")
    
    print(f"\nDiagram Summary:")
    for dim_name, summary in results['diagram_summary'].items():
        print(f"  {dim_name}: {summary['n_features']} features")
        if summary['persistence']:
            print(f"    Max persistence: {max(summary['persistence']):.4f}")
    
    if 'regime_detection' in results:
        changes = results['regime_detection'].get('detected_regime_changes', [])
        print(f"\nDetected Regime Changes: {len(changes)}")
        if len(changes) > 0:
            print(f"  At indices: {changes[:5]}...")  # Show first 5
    
    # Compare the two halves
    print("\n" + "="*50)
    print("Comparing First Half vs Second Half")
    
    comparison = lens.compare_attractors(data1, data2)
    print(f"\nLandscape distance: {comparison['landscape_distance']:.4f}")
    print(f"H1 persistence diff: {comparison['total_persistence_diff_h1']:.4f}")
