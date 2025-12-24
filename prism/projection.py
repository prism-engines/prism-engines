"""
PRISM Indicator Projector

Projects individual indicators into behavioral space within geometric states.
Each indicator becomes a point with coordinates derived from its behavior.
Track how indicator positions MOVE when geometry shifts.

The Core Idea:
- Within each geometric state, measure each indicator's behavior
- Behavior = (hurst, volatility, entropy, beta, tail_weight, ...)
- Each indicator is a POINT in behavioral space
- Different states = indicators move to different positions
- The MOVEMENT reveals how indicators respond to structure changes

Usage:
    from prism.structure import StructureExtractor
    from prism.projection import IndicatorProjector
    
    # First extract structure
    extractor = StructureExtractor(df)
    extractor.extract_geometry()
    extractor.cluster_geometry(n_states=3)
    
    # Then project indicators
    projector = IndicatorProjector(df, extractor)
    positions = projector.project_all_states()
    
    # Analyze movement
    movement = projector.compute_movement()
    projector.print_movement_report()
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class IndicatorPosition:
    """Position of an indicator in behavioral space."""
    indicator_id: str
    state_id: int
    coordinates: Dict[str, float]
    n_samples: int
    
    def to_vector(self, dimensions: List[str]) -> np.ndarray:
        """Convert to array in specified dimension order."""
        return np.array([self.coordinates.get(d, np.nan) for d in dimensions])
    
    def distance_to(self, other: 'IndicatorPosition', dimensions: List[str]) -> float:
        """Euclidean distance to another position."""
        v1 = self.to_vector(dimensions)
        v2 = other.to_vector(dimensions)
        mask = ~(np.isnan(v1) | np.isnan(v2))
        if mask.sum() == 0:
            return np.nan
        return np.sqrt(np.sum((v1[mask] - v2[mask]) ** 2))


@dataclass
class IndicatorMovement:
    """Movement of an indicator between states."""
    indicator_id: str
    from_state: int
    to_state: int
    distance: float
    dimension_changes: Dict[str, float]  # Change per dimension
    
    def __str__(self):
        return f"{self.indicator_id}: State {self.from_state}→{self.to_state}, distance={self.distance:.4f}"


@dataclass
class BehavioralSpace:
    """The full behavioral space across all states."""
    dimensions: List[str]
    positions: Dict[Tuple[str, int], IndicatorPosition]  # (indicator, state) -> position
    centroids: Dict[int, Dict[str, float]]  # state -> centroid coordinates
    
    def get_position(self, indicator: str, state: int) -> Optional[IndicatorPosition]:
        return self.positions.get((indicator, state))
    
    def get_state_positions(self, state: int) -> List[IndicatorPosition]:
        return [p for (ind, s), p in self.positions.items() if s == state]
    
    def get_indicator_positions(self, indicator: str) -> List[IndicatorPosition]:
        return [p for (ind, s), p in self.positions.items() if ind == indicator]


class IndicatorProjector:
    """
    Project indicators into behavioral space within geometric states.
    """
    
    # Behavioral dimensions to measure for each indicator
    BEHAVIORAL_DIMENSIONS = [
        "hurst",           # Persistence/mean-reversion
        "volatility",      # Standard deviation of returns
        "skewness",        # Asymmetry of returns
        "kurtosis",        # Tail weight of returns
        "entropy",         # Complexity/predictability
        "autocorr_1",      # First-order autocorrelation
        "autocorr_5",      # Fifth-order autocorrelation
        "trend_strength",  # Strength of directional movement
    ]
    
    def __init__(
        self, 
        df: pd.DataFrame, 
        extractor: 'StructureExtractor',
    ):
        """
        Initialize projector.
        
        Args:
            df: Indicator data (datetime index, indicators as columns)
            extractor: Fitted StructureExtractor with states identified
        """
        self.df = df
        self.extractor = extractor
        self.indicators = list(df.columns)
        
        self.behavioral_space: Optional[BehavioralSpace] = None
        self.movements: List[IndicatorMovement] = []
    
    def _compute_hurst(self, series: pd.Series) -> float:
        """Compute Hurst exponent using R/S method."""
        series = series.dropna()
        if len(series) < 100:
            return np.nan
        
        n = len(series)
        max_k = min(n // 10, 100)
        
        if max_k < 10:
            return np.nan
        
        rs_values = []
        ns = []
        
        for k in range(10, max_k + 1, 5):
            rs_list = []
            for start in range(0, n - k + 1, k):
                subseries = series.iloc[start:start + k]
                mean = subseries.mean()
                deviations = subseries - mean
                cumsum = deviations.cumsum()
                R = cumsum.max() - cumsum.min()
                S = subseries.std()
                if S > 0:
                    rs_list.append(R / S)
            
            if rs_list:
                rs_values.append(np.mean(rs_list))
                ns.append(k)
        
        if len(ns) < 3:
            return np.nan
        
        log_n = np.log(ns)
        log_rs = np.log(rs_values)
        
        slope, _, _, _, _ = stats.linregress(log_n, log_rs)
        return float(slope)
    
    def _compute_entropy(self, series: pd.Series, bins: int = 10) -> float:
        """Compute Shannon entropy of distribution."""
        series = series.dropna()
        if len(series) < 50:
            return np.nan
        
        hist, _ = np.histogram(series, bins=bins, density=True)
        hist = hist[hist > 0]
        return float(-np.sum(hist * np.log(hist + 1e-10)))
    
    def _compute_trend_strength(self, series: pd.Series) -> float:
        """Compute trend strength as R² of linear fit."""
        series = series.dropna()
        if len(series) < 20:
            return np.nan
        
        x = np.arange(len(series))
        slope, intercept, r_value, _, _ = stats.linregress(x, series.values)
        return float(r_value ** 2)
    
    def _measure_indicator_behavior(
        self, 
        series: pd.Series,
    ) -> Dict[str, float]:
        """
        Measure behavioral dimensions for a single indicator.
        
        Returns dict of dimension -> value
        """
        series = series.dropna()
        
        if len(series) < 50:
            return {}
        
        returns = series.pct_change().dropna()
        
        if len(returns) < 30:
            return {}
        
        behavior = {}
        
        # Hurst exponent
        behavior["hurst"] = self._compute_hurst(series)
        
        # Volatility (annualized)
        behavior["volatility"] = float(returns.std() * np.sqrt(252))
        
        # Skewness
        behavior["skewness"] = float(stats.skew(returns))
        
        # Kurtosis (excess)
        behavior["kurtosis"] = float(stats.kurtosis(returns))
        
        # Entropy
        behavior["entropy"] = self._compute_entropy(returns)
        
        # Autocorrelation
        if len(returns) > 5:
            behavior["autocorr_1"] = float(returns.autocorr(lag=1))
            behavior["autocorr_5"] = float(returns.autocorr(lag=5))
        
        # Trend strength
        behavior["trend_strength"] = self._compute_trend_strength(series)
        
        # Clean NaN values
        behavior = {k: v for k, v in behavior.items() 
                   if v is not None and not np.isnan(v)}
        
        return behavior
    
    def project_state(self, state_id: int) -> Dict[str, IndicatorPosition]:
        """
        Project all indicators into behavioral space for a specific state.
        
        Args:
            state_id: The geometric state to analyze
        
        Returns:
            Dict mapping indicator_id -> IndicatorPosition
        """
        # Get data for this state
        state_dates = self.extractor.get_state_dates(state_id)
        
        # Expand to actual data dates (state dates are window endpoints)
        all_dates = set()
        for geo_date in state_dates:
            # Include 30-day window around each geometry point
            mask = (self.df.index >= pd.Timestamp(geo_date) - pd.Timedelta(days=30)) & \
                   (self.df.index <= pd.Timestamp(geo_date) + pd.Timedelta(days=30))
            all_dates.update(self.df.index[mask].tolist())
        
        if not all_dates:
            return {}
        
        state_df = self.df.loc[self.df.index.isin(all_dates)].sort_index()
        
        positions = {}
        
        for indicator in self.indicators:
            if indicator not in state_df.columns:
                continue
            
            series = state_df[indicator]
            behavior = self._measure_indicator_behavior(series)
            
            if behavior:
                positions[indicator] = IndicatorPosition(
                    indicator_id=indicator,
                    state_id=state_id,
                    coordinates=behavior,
                    n_samples=len(series.dropna()),
                )
        
        return positions
    
    def project_all_states(self, verbose: bool = True) -> BehavioralSpace:
        """
        Project all indicators into behavioral space for all states.
        
        Returns:
            BehavioralSpace containing all positions
        """
        if not self.extractor.states:
            raise ValueError("Extractor has no states. Run cluster_geometry() first.")
        
        all_positions = {}
        
        for state_id in self.extractor.states.keys():
            if verbose:
                print(f"Projecting indicators for State {state_id}...")
            
            state_positions = self.project_state(state_id)
            
            for indicator, position in state_positions.items():
                all_positions[(indicator, state_id)] = position
            
            if verbose:
                print(f"  {len(state_positions)} indicators projected")
        
        # Compute centroids per state
        centroids = {}
        for state_id in self.extractor.states.keys():
            state_pos = [p for (ind, s), p in all_positions.items() if s == state_id]
            if state_pos:
                centroid = {}
                dims = set()
                for p in state_pos:
                    dims.update(p.coordinates.keys())
                
                for dim in dims:
                    values = [p.coordinates.get(dim) for p in state_pos 
                             if dim in p.coordinates]
                    values = [v for v in values if v is not None and not np.isnan(v)]
                    if values:
                        centroid[dim] = float(np.mean(values))
                
                centroids[state_id] = centroid
        
        # Get all dimensions used
        all_dims = set()
        for pos in all_positions.values():
            all_dims.update(pos.coordinates.keys())
        
        self.behavioral_space = BehavioralSpace(
            dimensions=sorted(all_dims),
            positions=all_positions,
            centroids=centroids,
        )
        
        return self.behavioral_space
    
    def compute_movement(self) -> List[IndicatorMovement]:
        """
        Compute how each indicator moves between states.
        
        Returns:
            List of IndicatorMovement objects
        """
        if self.behavioral_space is None:
            raise ValueError("Must project_all_states() first")
        
        self.movements = []
        states = sorted(self.extractor.states.keys())
        dims = self.behavioral_space.dimensions
        
        for indicator in self.indicators:
            positions = self.behavioral_space.get_indicator_positions(indicator)
            
            if len(positions) < 2:
                continue
            
            # Compare each pair of states
            for i, pos1 in enumerate(positions):
                for pos2 in positions[i+1:]:
                    distance = pos1.distance_to(pos2, dims)
                    
                    # Compute per-dimension changes
                    dim_changes = {}
                    for dim in dims:
                        v1 = pos1.coordinates.get(dim)
                        v2 = pos2.coordinates.get(dim)
                        if v1 is not None and v2 is not None:
                            dim_changes[dim] = v2 - v1
                    
                    self.movements.append(IndicatorMovement(
                        indicator_id=indicator,
                        from_state=pos1.state_id,
                        to_state=pos2.state_id,
                        distance=distance if not np.isnan(distance) else 0.0,
                        dimension_changes=dim_changes,
                    ))
        
        # Sort by distance (most movement first)
        self.movements.sort(key=lambda m: m.distance, reverse=True)
        
        return self.movements
    
    def get_position_matrix(self, state_id: int) -> pd.DataFrame:
        """
        Get indicator positions as a matrix for a specific state.
        
        Returns:
            DataFrame with indicators as rows, dimensions as columns
        """
        if self.behavioral_space is None:
            raise ValueError("Must project_all_states() first")
        
        positions = self.behavioral_space.get_state_positions(state_id)
        
        if not positions:
            return pd.DataFrame()
        
        records = []
        for pos in positions:
            record = {"indicator": pos.indicator_id}
            record.update(pos.coordinates)
            records.append(record)
        
        df = pd.DataFrame(records).set_index("indicator")
        return df.reindex(columns=sorted(df.columns))
    
    def get_movement_matrix(self) -> pd.DataFrame:
        """
        Get movement distances as indicator × state-pair matrix.
        """
        if not self.movements:
            self.compute_movement()
        
        records = []
        for m in self.movements:
            records.append({
                "indicator": m.indicator_id,
                "transition": f"{m.from_state}→{m.to_state}",
                "distance": m.distance,
            })
        
        df = pd.DataFrame(records)
        if df.empty:
            return df
        
        return df.pivot(index="indicator", columns="transition", values="distance")
    
    def print_position_report(self, state_id: Optional[int] = None):
        """Print positions for one or all states."""
        if self.behavioral_space is None:
            print("No positions computed. Run project_all_states() first.")
            return
        
        states = [state_id] if state_id else sorted(self.extractor.states.keys())
        
        for sid in states:
            print(f"\n{'='*60}")
            print(f"STATE {sid} - INDICATOR POSITIONS")
            print(f"{'='*60}")
            
            df = self.get_position_matrix(sid)
            if df.empty:
                print("No positions for this state")
                continue
            
            # Print formatted
            print(f"\n{'Indicator':<10}", end="")
            for col in df.columns:
                print(f" {col:>12}", end="")
            print()
            print("-" * (10 + 13 * len(df.columns)))
            
            for indicator in df.index:
                print(f"{indicator:<10}", end="")
                for col in df.columns:
                    val = df.loc[indicator, col]
                    if pd.isna(val):
                        print(f" {'N/A':>12}", end="")
                    else:
                        print(f" {val:>12.4f}", end="")
                print()
    
    def print_movement_report(self, top_n: int = 20):
        """Print indicators with largest movement between states."""
        if not self.movements:
            self.compute_movement()
        
        print("\n" + "=" * 60)
        print("INDICATOR MOVEMENT BETWEEN STATES")
        print("=" * 60)
        print("(Sorted by total distance moved in behavioral space)")
        print()
        
        # Aggregate by indicator
        indicator_total = {}
        for m in self.movements:
            if m.indicator_id not in indicator_total:
                indicator_total[m.indicator_id] = 0
            indicator_total[m.indicator_id] += m.distance
        
        sorted_indicators = sorted(indicator_total.items(), key=lambda x: x[1], reverse=True)
        
        print(f"{'Indicator':<12} {'Total Dist':>12} {'Transitions':>12}")
        print("-" * 40)
        
        for indicator, total_dist in sorted_indicators[:top_n]:
            n_trans = len([m for m in self.movements if m.indicator_id == indicator])
            print(f"{indicator:<12} {total_dist:>12.4f} {n_trans:>12}")
        
        # Show biggest single movements
        print("\n" + "-" * 60)
        print("LARGEST SINGLE MOVEMENTS")
        print("-" * 60)
        print(f"{'Indicator':<10} {'From':>6} {'To':>6} {'Distance':>10} {'Biggest Change':<20}")
        print("-" * 60)
        
        for m in self.movements[:top_n]:
            # Find dimension with biggest change
            if m.dimension_changes:
                biggest_dim = max(m.dimension_changes.items(), key=lambda x: abs(x[1]))
                change_str = f"{biggest_dim[0]}: {biggest_dim[1]:+.3f}"
            else:
                change_str = "N/A"
            
            print(f"{m.indicator_id:<10} {m.from_state:>6} {m.to_state:>6} {m.distance:>10.4f} {change_str:<20}")
    
    def compare_indicator_across_states(self, indicator: str) -> pd.DataFrame:
        """
        Show how a single indicator's position changes across states.
        """
        if self.behavioral_space is None:
            raise ValueError("Must project_all_states() first")
        
        positions = self.behavioral_space.get_indicator_positions(indicator)
        
        if not positions:
            return pd.DataFrame()
        
        records = []
        for pos in positions:
            record = {"state": pos.state_id, "n_samples": pos.n_samples}
            record.update(pos.coordinates)
            records.append(record)
        
        df = pd.DataFrame(records).set_index("state").sort_index()
        return df
