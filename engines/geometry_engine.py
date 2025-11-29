"""
VCF Geometric Analysis Module
==============================

Advanced geometric analysis for Vector Coherence Framework.

This module treats the normalized market state as a point in high-dimensional
geometric space and analyzes its:
- Position (magnitude, direction)
- Motion (velocity, acceleration)
- Structure (manifold, topology)
- Dynamics (rotation, divergence)

Mathematical Framework:
----------------------
State Space: R^N where N = number of normalized market signals
State Vector: x(t) = [x₁(t), x₂(t), ..., xₙ(t)]ᵀ

Key Geometric Quantities:
1. Magnitude: ||x(t)|| = market stress/distance from equilibrium
2. Direction: x̂(t) = x(t)/||x(t)|| = regime orientation
3. Velocity: v(t) = dx/dt = rate of regime change
4. Rotation: θ(t) = angle between x(t) and x(t-1)

Academic References:
-------------------
- Takens' Theorem (1981) - Phase space reconstruction
- Grassberger & Procaccia (1983) - Correlation dimension
- Ott, Sauer, Yorke (1994) - Chaos in dynamical systems
"""

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.stats import zscore
from sklearn.decomposition import PCA
from typing import Tuple, Optional, Dict, List


class GeometricAnalyzer:
    """
    Geometric analysis of market state space.
    
    Converts normalized market data into geometric quantities
    that reveal market structure and dynamics.
    """
    
    def __init__(self):
        """Initialize geometric analyzer."""
        pass
    
    def magnitude(self, state_matrix: pd.DataFrame) -> pd.Series:
        """
        Compute Euclidean magnitude of state vector at each time.
        
        This is the "distance from equilibrium" or "market stress level".
        
        Parameters:
        -----------
        state_matrix : pd.DataFrame
            Normalized market data (N×T matrix: N signals, T time points)
            
        Returns:
        --------
        pd.Series: ||x(t)|| for each time t
        
        Interpretation:
        --------------
        High magnitude: Market is far from equilibrium (stress/extremes)
        Low magnitude: Market near equilibrium (calm/balanced)
        Rising magnitude: Building stress
        Falling magnitude: Returning to normal
        
        Mathematical Formula:
        --------------------
        ||x(t)|| = √(Σᵢ xᵢ²(t))
        """
        # Compute row-wise Euclidean norm
        magnitude = np.sqrt((state_matrix ** 2).sum(axis=1))
        return magnitude
    
    def direction_vector(self, state_matrix: pd.DataFrame) -> pd.DataFrame:
        """
        Compute normalized direction of state vector (unit vectors).
        
        This captures "where" the market is in state space,
        independent of "how far" (magnitude).
        
        Parameters:
        -----------
        state_matrix : pd.DataFrame
            Normalized market data
            
        Returns:
        --------
        pd.DataFrame: Unit vectors x̂(t) = x(t)/||x(t)||
        
        Interpretation:
        --------------
        Direction represents the market "regime signature".
        Similar directions → similar market conditions
        Orthogonal directions → unrelated regimes
        """
        mag = self.magnitude(state_matrix)
        
        # Avoid division by zero
        mag_safe = mag.replace(0, 1e-10)
        
        # Normalize each row by its magnitude
        direction = state_matrix.div(mag_safe, axis=0)
        
        return direction
    
    def angular_rotation(self, state_matrix: pd.DataFrame) -> pd.Series:
        """
        Compute angle of rotation between successive state vectors.
        
        Measures how fast the market regime is changing direction.
        
        Parameters:
        -----------
        state_matrix : pd.DataFrame
            Normalized market data
            
        Returns:
        --------
        pd.Series: θ(t) = angle between x(t-1) and x(t) in radians
        
        Interpretation:
        --------------
        Small θ: Regime stable, slow evolution
        Large θ: Regime unstable, rapid change
        θ ≈ π: Complete reversal
        
        Mathematical Formula:
        --------------------
        cos(θ) = (x(t-1) · x(t)) / (||x(t-1)|| ||x(t)||)
        θ = arccos(cos(θ))
        """
        X = state_matrix.values
        angles = [np.nan]  # First point has no predecessor
        
        for t in range(1, len(X)):
            v1 = X[t-1]
            v2 = X[t]
            
            # Compute angle via dot product
            dot = np.dot(v1, v2)
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            
            if norm1 < 1e-10 or norm2 < 1e-10:
                angles.append(np.nan)
                continue
            
            cos_angle = dot / (norm1 * norm2)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Numerical safety
            angle = np.arccos(cos_angle)
            
            angles.append(angle)
        
        return pd.Series(angles, index=state_matrix.index)
    
    def velocity(self, state_matrix: pd.DataFrame, 
                dt: float = 1.0) -> pd.DataFrame:
        """
        Compute velocity in state space.
        
        v(t) = [x(t) - x(t-1)] / dt
        
        This is the rate of change of the state vector.
        
        Parameters:
        -----------
        state_matrix : pd.DataFrame
            Normalized market data
        dt : float
            Time step (default 1 = one period)
            
        Returns:
        --------
        pd.DataFrame: Velocity vectors
        
        Interpretation:
        --------------
        High velocity: Rapid market regime change
        Low velocity: Stable regime
        Velocity direction: Where market is heading
        """
        velocity = state_matrix.diff() / dt
        return velocity
    
    def acceleration(self, state_matrix: pd.DataFrame,
                    dt: float = 1.0) -> pd.DataFrame:
        """
        Compute acceleration in state space.
        
        a(t) = [v(t) - v(t-1)] / dt
        
        Second derivative of state vector.
        
        Parameters:
        -----------
        state_matrix : pd.DataFrame
            Normalized market data
        dt : float
            Time step
            
        Returns:
        --------
        pd.DataFrame: Acceleration vectors
        
        Interpretation:
        --------------
        Acceleration captures "jerk" or sudden changes in dynamics.
        High acceleration: Regime shift in progress
        """
        vel = self.velocity(state_matrix, dt)
        accel = vel.diff() / dt
        return accel
    
    def divergence_from_mean(self, state_matrix: pd.DataFrame) -> pd.Series:
        """
        Compute angle between current state and long-run mean state.
        
        This measures how far the current market regime has drifted
        from its historical "center of gravity".
        
        Parameters:
        -----------
        state_matrix : pd.DataFrame
            Normalized market data
            
        Returns:
        --------
        pd.Series: Divergence angle (radians)
        
        Interpretation:
        --------------
        Small divergence: Market near historical norms
        Large divergence: Market in extreme/unusual state
        Can signal mean-reversion opportunities or regime change
        
        Formula:
        --------
        μ = mean(x(t)) over all t
        divergence(t) = arccos((x(t) · μ) / (||x(t)|| ||μ||))
        """
        # Compute mean state vector
        mean_vector = state_matrix.mean(axis=0).values
        norm_mean = np.linalg.norm(mean_vector)
        
        if norm_mean < 1e-10:
            return pd.Series(np.nan, index=state_matrix.index)
        
        X = state_matrix.values
        divergences = []
        
        for t in range(len(X)):
            v = X[t]
            norm_v = np.linalg.norm(v)
            
            if norm_v < 1e-10:
                divergences.append(np.nan)
                continue
            
            cos_angle = np.dot(v, mean_vector) / (norm_v * norm_mean)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = np.arccos(cos_angle)
            divergences.append(angle)
        
        return pd.Series(divergences, index=state_matrix.index)
    
    def pca_projection(self, state_matrix: pd.DataFrame,
                      n_components: int = 3) -> Tuple[pd.DataFrame, PCA]:
        """
        Project state space onto principal components.
        
        Reduces dimensionality while preserving geometric structure.
        Reveals the dominant modes of variation in market dynamics.
        
        Parameters:
        -----------
        state_matrix : pd.DataFrame
            Normalized market data
        n_components : int
            Number of principal components to retain
            
        Returns:
        --------
        projected : pd.DataFrame
            State matrix in PC space
        pca_model : PCA
            Fitted PCA model (for explained variance, loadings, etc.)
            
        Use Cases:
        ----------
        1. Visualization (plot first 2-3 PCs)
        2. Dimensionality reduction
        3. Finding dominant market drivers
        4. Identifying uncorrelated factors
        """
        # Remove any NaN rows
        clean_data = state_matrix.dropna()
        
        if len(clean_data) < n_components:
            raise ValueError("Not enough data for PCA")
        
        # Fit PCA
        pca = PCA(n_components=n_components)
        projected_values = pca.fit_transform(clean_data.values)
        
        # Create DataFrame
        projected = pd.DataFrame(
            projected_values,
            index=clean_data.index,
            columns=[f'PC{i+1}' for i in range(n_components)]
        )
        
        return projected, pca
    
    def manifold_curvature(self, state_matrix: pd.DataFrame,
                          window: int = 12) -> pd.Series:
        """
        Estimate local curvature of state space manifold.
        
        Uses change in velocity direction as proxy for curvature.
        High curvature indicates nonlinear regime dynamics.
        
        Parameters:
        -----------
        state_matrix : pd.DataFrame
            Normalized market data
        window : int
            Window for computing local curvature
            
        Returns:
        --------
        pd.Series: Estimated curvature at each time
        
        Interpretation:
        --------------
        Low curvature: Linear/smooth regime evolution
        High curvature: Nonlinear/turbulent dynamics
        Peaks often precede regime changes
        """
        # Compute velocity
        vel = self.velocity(state_matrix)
        
        # Compute velocity magnitude
        vel_mag = np.sqrt((vel ** 2).sum(axis=1))
        
        # Compute rate of change of velocity direction (curvature proxy)
        angles = self.angular_rotation(vel)
        
        # Smooth with rolling mean
        curvature = angles.rolling(window=window, min_periods=1).mean()
        
        return curvature


class RegimeDetector:
    """
    Detect and classify market regimes using geometric analysis.
    
    Combines multiple geometric signals to identify:
    - Stable regimes
    - Transitions
    - Crises
    - Recoveries
    """
    
    def __init__(self, analyzer: Optional[GeometricAnalyzer] = None):
        """
        Initialize regime detector.
        
        Parameters:
        -----------
        analyzer : GeometricAnalyzer, optional
            Geometric analyzer instance
        """
        self.analyzer = analyzer if analyzer else GeometricAnalyzer()
    
    def compute_regime_signals(self, state_matrix: pd.DataFrame) -> pd.DataFrame:
        """
        Compute all geometric signals used for regime detection.
        
        Parameters:
        -----------
        state_matrix : pd.DataFrame
            Normalized market data
            
        Returns:
        --------
        pd.DataFrame with columns:
            - magnitude: Distance from equilibrium
            - rotation: Angular change rate
            - divergence: Distance from mean state
            - velocity_mag: Speed in state space
            - curvature: Manifold curvature
        """
        magnitude = self.analyzer.magnitude(state_matrix)
        rotation = self.analyzer.angular_rotation(state_matrix)
        divergence = self.analyzer.divergence_from_mean(state_matrix)
        
        vel = self.analyzer.velocity(state_matrix)
        velocity_mag = np.sqrt((vel ** 2).sum(axis=1))
        
        curvature = self.analyzer.manifold_curvature(state_matrix)
        
        signals = pd.DataFrame({
            'magnitude': magnitude,
            'rotation': rotation,
            'divergence': divergence,
            'velocity_mag': velocity_mag,
            'curvature': curvature
        })
        
        return signals
    
    def classify_regime(self, signals: pd.DataFrame,
                       mag_thresh: Tuple[float, float] = (0.33, 0.67),
                       rot_thresh: Tuple[float, float] = (0.1, 0.3),
                       div_thresh: Tuple[float, float] = (0.33, 0.67)) -> pd.Series:
        """
        Classify market regime based on geometric signals.
        
        Parameters:
        -----------
        signals : pd.DataFrame
            Output from compute_regime_signals()
        mag_thresh, rot_thresh, div_thresh : tuple
            (low, high) quantile thresholds for classification
            
        Returns:
        --------
        pd.Series: Regime labels
        
        Regime Definitions:
        ------------------
        1. "Equilibrium": Low magnitude, low rotation, low divergence
           → Market stable and near historical norms
           
        2. "Trending": High velocity, low rotation
           → Strong directional move but smooth
           
        3. "Transition": High rotation, moderate magnitude
           → Regime change in progress
           
        4. "Stress": High magnitude, high divergence
           → Market in extreme state
           
        5. "Crisis": High magnitude, high rotation, high divergence
           → Severe dislocation
           
        6. "Recovery": Falling magnitude, moderate rotation
           → Returning from extreme
        """
        # Normalize signals to quantiles
        mag_q = signals['magnitude'].rank(pct=True)
        rot_q = signals['rotation'].rank(pct=True)
        div_q = signals['divergence'].rank(pct=True)
        vel_q = signals['velocity_mag'].rank(pct=True)
        
        regimes = []
        
        for i in range(len(signals)):
            mag = mag_q.iloc[i]
            rot = rot_q.iloc[i]
            div = div_q.iloc[i]
            vel = vel_q.iloc[i]
            
            # Handle NaN
            if pd.isna(mag) or pd.isna(rot) or pd.isna(div):
                regimes.append("Unknown")
                continue
            
            # Crisis: everything high
            if mag > mag_thresh[1] and rot > rot_thresh[1] and div > div_thresh[1]:
                regimes.append("Crisis")
            
            # Stress: high magnitude and divergence
            elif mag > mag_thresh[1] and div > div_thresh[1]:
                regimes.append("Stress")
            
            # Transition: high rotation
            elif rot > rot_thresh[1]:
                regimes.append("Transition")
            
            # Trending: high velocity, low rotation
            elif vel > 0.6 and rot < rot_thresh[0]:
                regimes.append("Trending")
            
            # Recovery: high previous magnitude, now falling
            elif i > 0 and mag < mag_q.iloc[i-1] and mag_q.iloc[i-1] > mag_thresh[1]:
                regimes.append("Recovery")
            
            # Equilibrium: everything low
            elif mag < mag_thresh[0] and rot < rot_thresh[0] and div < div_thresh[0]:
                regimes.append("Equilibrium")
            
            # Normal: default
            else:
                regimes.append("Normal")
        
        return pd.Series(regimes, index=signals.index, name='regime')
    
    def detect_regime_changes(self, regimes: pd.Series) -> pd.Series:
        """
        Flag time points where regime changes occur.
        
        Parameters:
        -----------
        regimes : pd.Series
            Regime labels from classify_regime()
            
        Returns:
        --------
        pd.Series: Boolean series marking regime change points
        """
        changes = regimes != regimes.shift(1)
        changes.iloc[0] = False  # First point is not a change
        return changes


# Testing and example usage
if __name__ == "__main__":
    print("VCF Geometric Analysis - Test Suite")
    print("=" * 60)
    
    # Create synthetic state matrix
    np.random.seed(42)
    dates = pd.date_range('2000-01-01', periods=120, freq='M')
    n_signals = 8
    
    # Simulate regime-switching behavior
    regime1 = np.random.randn(60, n_signals) * 0.5  # Calm period
    regime2 = np.random.randn(30, n_signals) * 2.0  # Volatile period
    regime3 = np.random.randn(30, n_signals) * 0.8  # Recovery
    
    state_data = np.vstack([regime1, regime2, regime3])
    state_matrix = pd.DataFrame(
        state_data,
        index=dates,
        columns=[f'Signal_{i}' for i in range(n_signals)]
    )
    
    # Initialize analyzer
    analyzer = GeometricAnalyzer()
    
    print("\n1. Testing Magnitude Calculation")
    print("-" * 60)
    magnitude = analyzer.magnitude(state_matrix)
    print(f"Magnitude range: [{magnitude.min():.3f}, {magnitude.max():.3f}]")
    print(f"Magnitude mean: {magnitude.mean():.3f}")
    print(f"Expected higher in middle (volatile regime)")
    
    print("\n2. Testing Angular Rotation")
    print("-" * 60)
    rotation = analyzer.angular_rotation(state_matrix)
    print(f"Rotation mean: {rotation.mean():.3f} radians")
    print(f"Rotation max: {rotation.max():.3f} radians")
    print(f"Expected higher at regime transitions")
    
    print("\n3. Testing Divergence from Mean")
    print("-" * 60)
    divergence = analyzer.divergence_from_mean(state_matrix)
    print(f"Divergence mean: {divergence.mean():.3f}")
    print(f"Divergence std: {divergence.std():.3f}")
    
    print("\n4. Testing PCA Projection")
    print("-" * 60)
    projected, pca = analyzer.pca_projection(state_matrix, n_components=3)
    print(f"Explained variance ratios: {pca.explained_variance_ratio_}")
    print(f"Cumulative variance: {pca.explained_variance_ratio_.sum():.3f}")
    
    print("\n5. Testing Regime Detection")
    print("-" * 60)
    detector = RegimeDetector(analyzer)
    signals = detector.compute_regime_signals(state_matrix)
    regimes = detector.classify_regime(signals)
    
    print("\nRegime distribution:")
    print(regimes.value_counts())
    
    print("\n6. Testing Regime Change Detection")
    print("-" * 60)
    changes = detector.detect_regime_changes(regimes)
    print(f"Number of regime changes: {changes.sum()}")
    print(f"Regime change dates: {regimes.index[changes].tolist()[:5]}")
    
    print("\n" + "=" * 60)
    print("All geometric analysis tests completed successfully!")
