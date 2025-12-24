"""
PRISM Structure Extractor

Extracts dynamic geometry from indicator data using rolling windows.
Maps each time period to coordinates in geometry space.
Clusters geometry space to identify structure states.

The Core Idea:
- Run structure-defining engines on rolling windows
- Each window produces a geometry coordinate (7+ dimensions)
- Cluster the geometry coordinates to find distinct states
- Label each date with its structure state
- Then: measure indicator behavior WITHIN each state

Usage:
    from prism.structure import StructureExtractor
    
    extractor = StructureExtractor(df)
    geometry = extractor.extract_geometry(window=252)
    states = extractor.cluster_geometry(n_states=3)
    
    # Get dates for each state
    state_1_dates = extractor.get_state_dates(1)
    
    # Run bounded analysis
    bounded_metrics = extractor.run_bounded_engine("hurst", state=1)
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist

logger = logging.getLogger(__name__)


@dataclass
class GeometryPoint:
    """A single point in geometry space."""
    date: date
    coordinates: Dict[str, float]
    state: Optional[int] = None
    
    def to_vector(self, dimensions: List[str]) -> np.ndarray:
        """Convert to numpy array in specified dimension order."""
        return np.array([self.coordinates.get(d, np.nan) for d in dimensions])


@dataclass 
class StructureState:
    """A cluster in geometry space."""
    state_id: int
    centroid: Dict[str, float]
    n_points: int
    date_ranges: List[Tuple[date, date]]
    
    def __str__(self):
        return f"State {self.state_id}: {self.n_points} points, centroid={self.centroid}"


class StructureExtractor:
    """
    Extract and analyze geometric structure from indicator data.
    """
    
    # Engines that define geometry (structure-revealing)
    GEOMETRY_ENGINES = [
        "pca",
        "clustering", 
        "hmm",
        "cross_correlation",
        "lyapunov",
        "copula",
        "entropy",
    ]
    
    # Metrics that form geometry coordinates
    GEOMETRY_DIMENSIONS = [
        ("pca", "variance_pc1"),
        ("pca", "effective_dimensionality"),
        ("clustering", "silhouette_score"),
        ("cross_correlation", "avg_abs_correlation"),
        ("lyapunov", "avg_lyapunov"),
        ("copula", "tail_asymmetry"),
        ("entropy", "avg_permutation_entropy"),
    ]
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize with indicator data.
        
        Args:
            df: DataFrame with datetime index, indicators as columns
        """
        self.df = df
        self.df_zscore = (df - df.mean()) / df.std()
        self.df_returns = df.pct_change().dropna()
        
        self.geometry_points: List[GeometryPoint] = []
        self.geometry_df: Optional[pd.DataFrame] = None
        self.states: Dict[int, StructureState] = {}
        self.state_labels: Optional[pd.Series] = None
        
        # Engine instances (lazy load)
        self._engines = {}
    
    def _get_engine(self, name: str):
        """Get or create engine instance."""
        if name not in self._engines:
            from prism.engines import get_engine
            engine = get_engine(name)
            engine.store_results = lambda *args: None  # Skip DB
            self._engines[name] = engine
        return self._engines[name]
    
    def _get_data_for_engine(self, engine_name: str, window_df: pd.DataFrame) -> pd.DataFrame:
        """Get appropriately normalized data for engine."""
        if engine_name in ["hurst", "wavelet", "copula", "cointegration"]:
            return window_df
        elif engine_name in ["granger", "rolling_beta", "spectral", "garch"]:
            return window_df.pct_change().dropna()
        else:
            return (window_df - window_df.mean()) / window_df.std()
    
    def extract_geometry(
        self,
        window: int = 252,
        step: int = 21,
        min_window: int = 126,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """
        Extract geometry coordinates using rolling windows.
        
        Args:
            window: Rolling window size (default 252 = ~1 year)
            step: Step size between windows (default 21 = ~1 month)
            min_window: Minimum data points required
            verbose: Print progress
        
        Returns:
            DataFrame with geometry coordinates per date
        """
        if verbose:
            print(f"Extracting geometry: window={window}, step={step}")
            print(f"Data: {len(self.df)} rows, {len(self.df.columns)} indicators")
        
        self.geometry_points = []
        
        # Rolling window extraction
        dates = self.df.index.tolist()
        n_windows = (len(dates) - window) // step + 1
        
        for i in range(0, len(dates) - window + 1, step):
            window_end = i + window
            window_dates = dates[i:window_end]
            window_df = self.df.iloc[i:window_end]
            
            # Date label is the END of the window (what we know as of this date)
            as_of_date = window_dates[-1]
            
            if verbose and (i // step) % 10 == 0:
                print(f"  Window {i // step + 1}/{n_windows}: {window_dates[0].date()} to {as_of_date.date()}")
            
            # Extract coordinates from each geometry engine
            coordinates = {}
            
            for engine_name, metric_name in self.GEOMETRY_DIMENSIONS:
                try:
                    engine = self._get_engine(engine_name)
                    data = self._get_data_for_engine(engine_name, window_df)
                    
                    if len(data) < min_window:
                        continue
                    
                    metrics = engine.run(data, run_id=f"geo_{i}")
                    
                    if metric_name in metrics:
                        value = metrics[metric_name]
                        if isinstance(value, (int, float)) and not np.isnan(value):
                            coord_name = f"{engine_name}__{metric_name}"
                            coordinates[coord_name] = float(value)
                            
                except Exception as e:
                    logger.warning(f"Engine {engine_name} failed on window {i}: {e}")
            
            if coordinates:
                self.geometry_points.append(GeometryPoint(
                    date=as_of_date.date() if hasattr(as_of_date, 'date') else as_of_date,
                    coordinates=coordinates,
                ))
        
        # Convert to DataFrame
        if self.geometry_points:
            records = []
            for pt in self.geometry_points:
                record = {"date": pt.date}
                record.update(pt.coordinates)
                records.append(record)
            
            self.geometry_df = pd.DataFrame(records).set_index("date")
            
            if verbose:
                print(f"\nGeometry extracted: {len(self.geometry_df)} points, {len(self.geometry_df.columns)} dimensions")
        
        return self.geometry_df
    
    def cluster_geometry(
        self,
        n_states: int = 3,
        method: str = "ward",
        normalize: bool = True,
    ) -> pd.Series:
        """
        Cluster geometry space into structure states.
        
        Args:
            n_states: Number of states to identify
            method: Clustering method ('ward', 'complete', 'average')
            normalize: Normalize coordinates before clustering
        
        Returns:
            Series mapping dates to state labels
        """
        if self.geometry_df is None or len(self.geometry_df) == 0:
            raise ValueError("Must extract geometry first")
        
        # Prepare data
        geo_data = self.geometry_df.dropna()
        
        if normalize:
            geo_normalized = (geo_data - geo_data.mean()) / geo_data.std()
        else:
            geo_normalized = geo_data
        
        # Hierarchical clustering
        distances = pdist(geo_normalized.values)
        linkage_matrix = linkage(distances, method=method)
        labels = fcluster(linkage_matrix, n_states, criterion='maxclust')
        
        # Store state labels
        self.state_labels = pd.Series(labels, index=geo_data.index, name="state")
        
        # Update geometry points with state
        for pt in self.geometry_points:
            if pt.date in self.state_labels.index:
                pt.state = int(self.state_labels.loc[pt.date])
        
        # Compute state info
        self._compute_state_info(geo_data)
        
        return self.state_labels
    
    def _compute_state_info(self, geo_data: pd.DataFrame):
        """Compute centroid and date ranges for each state."""
        self.states = {}
        
        for state_id in self.state_labels.unique():
            mask = self.state_labels == state_id
            state_data = geo_data[mask]
            
            # Centroid
            centroid = state_data.mean().to_dict()
            
            # Date ranges (contiguous periods)
            state_dates = sorted(state_data.index.tolist())
            date_ranges = self._find_contiguous_ranges(state_dates)
            
            self.states[state_id] = StructureState(
                state_id=state_id,
                centroid=centroid,
                n_points=len(state_data),
                date_ranges=date_ranges,
            )
    
    def _find_contiguous_ranges(self, dates: List) -> List[Tuple]:
        """Find contiguous date ranges."""
        if not dates:
            return []
        
        ranges = []
        start = dates[0]
        prev = dates[0]
        
        for d in dates[1:]:
            # Check if gap > 60 days (roughly 2 months)
            gap = (pd.Timestamp(d) - pd.Timestamp(prev)).days
            if gap > 60:
                ranges.append((start, prev))
                start = d
            prev = d
        
        ranges.append((start, prev))
        return ranges
    
    def get_state_dates(self, state_id: int) -> pd.DatetimeIndex:
        """Get all dates belonging to a state."""
        if self.state_labels is None:
            raise ValueError("Must cluster geometry first")
        
        mask = self.state_labels == state_id
        return self.state_labels[mask].index
    
    def get_state_data(self, state_id: int) -> pd.DataFrame:
        """Get indicator data for dates in a specific state."""
        state_dates = self.get_state_dates(state_id)
        
        # Expand window dates to actual data dates
        # (geometry dates are window endpoints)
        all_dates = []
        for geo_date in state_dates:
            # Include data from surrounding period
            mask = (self.df.index >= pd.Timestamp(geo_date) - pd.Timedelta(days=30)) & \
                   (self.df.index <= pd.Timestamp(geo_date) + pd.Timedelta(days=30))
            all_dates.extend(self.df.index[mask].tolist())
        
        all_dates = sorted(set(all_dates))
        return self.df.loc[self.df.index.isin(all_dates)]
    
    def run_bounded_engine(
        self,
        engine_name: str,
        state_id: int,
    ) -> Dict[str, Any]:
        """
        Run an engine on data bounded to a specific structure state.
        
        Args:
            engine_name: Engine to run
            state_id: State to bound analysis to
        
        Returns:
            Engine metrics for that state
        """
        state_data = self.get_state_data(state_id)
        
        if len(state_data) < 50:
            raise ValueError(f"State {state_id} has insufficient data: {len(state_data)} rows")
        
        engine = self._get_engine(engine_name)
        data = self._get_data_for_engine(engine_name, state_data)
        
        metrics = engine.run(data, run_id=f"bounded_state_{state_id}")
        
        return metrics
    
    def run_all_bounded(
        self,
        engines: Optional[List[str]] = None,
    ) -> Dict[int, Dict[str, Dict[str, Any]]]:
        """
        Run engines on all states for comparison.
        
        Returns:
            {state_id: {engine_name: metrics}}
        """
        if engines is None:
            engines = ["pca", "hurst", "cross_correlation", "entropy", "lyapunov"]
        
        results = {}
        
        for state_id in self.states.keys():
            results[state_id] = {}
            
            for engine_name in engines:
                try:
                    metrics = self.run_bounded_engine(engine_name, state_id)
                    # Keep only numeric metrics
                    numeric = {k: v for k, v in metrics.items() 
                              if isinstance(v, (int, float)) and not np.isnan(v)}
                    results[state_id][engine_name] = numeric
                except Exception as e:
                    results[state_id][engine_name] = {"error": str(e)}
        
        return results
    
    def print_state_summary(self):
        """Print summary of discovered states."""
        if not self.states:
            print("No states discovered. Run cluster_geometry() first.")
            return
        
        print("=" * 60)
        print("STRUCTURE STATES")
        print("=" * 60)
        
        for state_id, state in sorted(self.states.items()):
            print(f"\nState {state_id}: {state.n_points} geometry points")
            print("-" * 40)
            
            # Centroid (key dimensions only)
            print("Centroid:")
            for dim, val in sorted(state.centroid.items()):
                print(f"  {dim}: {val:.4f}")
            
            # Date ranges
            print(f"Periods: {len(state.date_ranges)}")
            for start, end in state.date_ranges[:5]:  # Show first 5
                print(f"  {start} to {end}")
            if len(state.date_ranges) > 5:
                print(f"  ... and {len(state.date_ranges) - 5} more")
    
    def compare_states(
        self,
        metric_name: str,
        engine_name: str = "pca",
    ) -> pd.DataFrame:
        """
        Compare a specific metric across all states.
        
        Returns:
            DataFrame with metric value per state
        """
        bounded_results = self.run_all_bounded(engines=[engine_name])
        
        comparison = []
        for state_id, engines in bounded_results.items():
            if engine_name in engines and metric_name in engines[engine_name]:
                comparison.append({
                    "state": state_id,
                    "n_points": self.states[state_id].n_points,
                    metric_name: engines[engine_name][metric_name],
                })
        
        return pd.DataFrame(comparison).set_index("state")
