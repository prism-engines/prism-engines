"""
Transfer Entropy Lens
=====================
Quantifies directed information flow between time series,
detecting non-linear, asymmetric dependencies that correlation misses.

Key Applications:
- Detecting causal information flow in climate systems
- Identifying leading/lagging relationships in markets
- Understanding feedback loops in complex systems

Methods:
- Transfer Entropy: Information-theoretic causality measure
- Conditional Transfer Entropy: Controls for confounding variables
- Effective Transfer Entropy: Finite-sample bias correction
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from scipy.special import digamma
from scipy.spatial import cKDTree
import warnings


@dataclass
class TransferEntropyResult:
    """Container for transfer entropy results."""
    source: str
    target: str
    transfer_entropy: float
    normalized_te: float       # Normalized by target entropy
    p_value: Optional[float]   # Statistical significance
    effective_te: float        # Bias-corrected estimate
    lags_used: int


class TransferEntropyLens:
    """
    Transfer Entropy Lens for directed information flow analysis.
    
    Transfer entropy T(X→Y) measures the information about Y's future
    that X provides beyond what Y's own past provides:
    
    T(X→Y) = H(Y_t | Y_past) - H(Y_t | Y_past, X_past)
    
    Key properties:
    - Asymmetric: T(X→Y) ≠ T(Y→X)
    - Model-free: Works for nonlinear relationships
    - Directional: Indicates information flow direction
    """
    
    def __init__(
        self,
        method: str = 'ksg',
        k: int = 4,
        lag: int = 1,
        history_length: int = 1
    ):
        """
        Initialize Transfer Entropy Lens.
        
        Parameters
        ----------
        method : str
            Estimation method:
            'ksg': Kraskov-Stögbauer-Grassberger (continuous, recommended)
            'binned': Histogram-based (discrete)
            'kernel': Kernel density estimation
        k : int
            Number of nearest neighbors (for KSG)
        lag : int
            Time lag from source to target
        history_length : int
            Number of past values to use
        """
        self.method = method
        self.k = k
        self.lag = lag
        self.history_length = history_length
    
    def _embed_time_series(
        self,
        source: np.ndarray,
        target: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Create embedded vectors for transfer entropy estimation.
        
        Returns:
        - target_future: Y(t+lag)
        - target_past: [Y(t), Y(t-1), ..., Y(t-history+1)]
        - source_past: [X(t), X(t-1), ..., X(t-history+1)]
        - joint_past: concatenation of target_past and source_past
        """
        n = len(source)
        offset = self.history_length + self.lag - 1
        valid_length = n - offset
        
        if valid_length <= self.k + 1:
            raise ValueError("Time series too short for given parameters")
        
        # Target future
        target_future = target[offset:].reshape(-1, 1)
        
        # Target past
        target_past = np.zeros((valid_length, self.history_length))
        for i in range(self.history_length):
            target_past[:, i] = target[offset - 1 - i:n - 1 - i]
        
        # Source past (lagged)
        source_past = np.zeros((valid_length, self.history_length))
        for i in range(self.history_length):
            idx_start = offset - self.lag - i
            idx_end = n - self.lag - i
            source_past[:, i] = source[idx_start:idx_end]
        
        # Joint past
        joint_past = np.hstack([target_past, source_past])
        
        return target_future, target_past, source_past, joint_past
    
    def _ksg_entropy(self, X: np.ndarray) -> float:
        """
        Estimate entropy using KSG estimator.
        
        H(X) ≈ ψ(N) - ψ(k) + d*<log(2*ε)>
        
        where ε is the distance to k-th neighbor and d is dimension.
        """
        n, d = X.shape
        
        # Build KD-tree
        tree = cKDTree(X)
        
        # Find k-th neighbor distances (using k+1 because query point is included)
        distances, _ = tree.query(X, k=self.k + 1, p=float('inf'))
        
        # Use distance to k-th neighbor (excluding self)
        eps = distances[:, -1]
        eps = np.maximum(eps, 1e-10)  # Avoid log(0)
        
        # KSG estimator
        entropy = digamma(n) - digamma(self.k) + d * np.mean(np.log(2 * eps))
        
        return entropy
    
    def _ksg_conditional_entropy(
        self,
        X: np.ndarray,
        Y: np.ndarray
    ) -> float:
        """
        Estimate conditional entropy H(X|Y) using KSG.
        
        H(X|Y) = H(X,Y) - H(Y)
        """
        joint = np.hstack([X, Y])
        return self._ksg_entropy(joint) - self._ksg_entropy(Y)
    
    def _ksg_mutual_information(
        self,
        X: np.ndarray,
        Y: np.ndarray
    ) -> float:
        """
        Estimate mutual information I(X;Y) using KSG.
        
        I(X;Y) = H(X) + H(Y) - H(X,Y)
        """
        joint = np.hstack([X, Y])
        return self._ksg_entropy(X) + self._ksg_entropy(Y) - self._ksg_entropy(joint)
    
    def _ksg_conditional_mutual_information(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        Z: np.ndarray
    ) -> float:
        """
        Estimate conditional mutual information I(X;Y|Z) using KSG.
        
        I(X;Y|Z) = H(X,Z) + H(Y,Z) - H(Z) - H(X,Y,Z)
        """
        XZ = np.hstack([X, Z])
        YZ = np.hstack([Y, Z])
        XYZ = np.hstack([X, Y, Z])
        
        return (self._ksg_entropy(XZ) + self._ksg_entropy(YZ) - 
                self._ksg_entropy(Z) - self._ksg_entropy(XYZ))
    
    def _binned_entropy(self, X: np.ndarray, bins: int = 10) -> float:
        """Estimate entropy using histogram binning."""
        d = X.shape[1] if X.ndim > 1 else 1
        
        if d == 1:
            X = X.flatten()
            hist, _ = np.histogram(X, bins=bins, density=True)
            hist = hist[hist > 0]
            return -np.sum(hist * np.log(hist + 1e-10)) * (X.max() - X.min()) / bins
        else:
            # Multi-dimensional: use product of marginal bins
            n_bins = max(2, int(bins ** (1/d)))
            hist, _ = np.histogramdd(X, bins=n_bins, density=True)
            hist = hist.flatten()
            hist = hist[hist > 0]
            bin_volume = np.prod(X.ptp(axis=0)) / (n_bins ** d)
            return -np.sum(hist * np.log(hist + 1e-10)) * bin_volume
    
    def _binned_transfer_entropy(
        self,
        source: np.ndarray,
        target: np.ndarray,
        bins: int = 10
    ) -> float:
        """Estimate transfer entropy using binning."""
        target_future, target_past, source_past, joint_past = self._embed_time_series(
            source, target
        )
        
        # T(X→Y) = H(Y_future, Y_past) - H(Y_past) - H(Y_future, joint_past) + H(joint_past)
        
        YfYp = np.hstack([target_future, target_past])
        YfJp = np.hstack([target_future, joint_past])
        
        te = (self._binned_entropy(YfYp, bins) - 
              self._binned_entropy(target_past, bins) -
              self._binned_entropy(YfJp, bins) + 
              self._binned_entropy(joint_past, bins))
        
        return max(0, te)  # TE is non-negative
    
    def compute_transfer_entropy(
        self,
        source: np.ndarray,
        target: np.ndarray
    ) -> float:
        """
        Compute transfer entropy from source to target.
        
        Parameters
        ----------
        source : np.ndarray
            Source time series X
        target : np.ndarray
            Target time series Y
            
        Returns
        -------
        float
            Transfer entropy T(X→Y)
        """
        source = np.asarray(source).flatten()
        target = np.asarray(target).flatten()
        
        if len(source) != len(target):
            raise ValueError("Source and target must have same length")
        
        # Normalize data
        source = (source - np.mean(source)) / (np.std(source) + 1e-10)
        target = (target - np.mean(target)) / (np.std(target) + 1e-10)
        
        if self.method == 'ksg':
            target_future, target_past, source_past, joint_past = self._embed_time_series(
                source, target
            )
            
            # T(X→Y) = I(Y_future; X_past | Y_past)
            # = H(Y_future | Y_past) - H(Y_future | Y_past, X_past)
            
            te = self._ksg_conditional_mutual_information(
                target_future, source_past, target_past
            )
            
            return max(0, te)
        
        elif self.method == 'binned':
            return self._binned_transfer_entropy(source, target)
        
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def compute_effective_transfer_entropy(
        self,
        source: np.ndarray,
        target: np.ndarray,
        n_surrogates: int = 100
    ) -> Tuple[float, float, float]:
        """
        Compute effective transfer entropy with bias correction.
        
        Uses surrogate data to estimate and remove finite-sample bias.
        
        Parameters
        ----------
        source : np.ndarray
            Source time series
        target : np.ndarray
            Target time series
        n_surrogates : int
            Number of surrogate time series
            
        Returns
        -------
        tuple
            (raw_te, bias, effective_te)
        """
        # Raw transfer entropy
        raw_te = self.compute_transfer_entropy(source, target)
        
        # Estimate bias using shuffled surrogates
        surrogate_tes = []
        source_copy = source.copy()
        
        for _ in range(n_surrogates):
            # Shuffle source to destroy temporal structure
            np.random.shuffle(source_copy)
            surrogate_te = self.compute_transfer_entropy(source_copy, target)
            surrogate_tes.append(surrogate_te)
        
        # Bias estimate (mean of surrogate TE)
        bias = np.mean(surrogate_tes)
        
        # Effective TE
        effective_te = max(0, raw_te - bias)
        
        return raw_te, bias, effective_te
    
    def significance_test(
        self,
        source: np.ndarray,
        target: np.ndarray,
        n_surrogates: int = 1000,
        alpha: float = 0.05
    ) -> Tuple[float, float, bool]:
        """
        Test significance of transfer entropy using surrogate data.
        
        Parameters
        ----------
        source : np.ndarray
            Source time series
        target : np.ndarray
            Target time series
        n_surrogates : int
            Number of surrogates for null distribution
        alpha : float
            Significance level
            
        Returns
        -------
        tuple
            (observed_te, p_value, is_significant)
        """
        observed_te = self.compute_transfer_entropy(source, target)
        
        # Generate null distribution
        null_tes = []
        source_copy = source.copy()
        
        for _ in range(n_surrogates):
            np.random.shuffle(source_copy)
            null_te = self.compute_transfer_entropy(source_copy, target)
            null_tes.append(null_te)
        
        # P-value: fraction of null TE >= observed TE
        p_value = (np.sum(null_tes >= observed_te) + 1) / (n_surrogates + 1)
        
        return observed_te, p_value, p_value < alpha
    
    def compute_pairwise_transfer_entropy(
        self,
        data: pd.DataFrame,
        significance_test: bool = False,
        n_surrogates: int = 100
    ) -> Dict:
        """
        Compute transfer entropy for all pairs of variables.
        
        Parameters
        ----------
        data : pd.DataFrame
            Multivariate time series
        significance_test : bool
            Whether to compute p-values
        n_surrogates : int
            Surrogates for significance testing
            
        Returns
        -------
        dict
            'matrix': TE matrix (source on rows, target on columns)
            'asymmetry': T(X→Y) - T(Y→X) for each pair
        """
        columns = data.columns.tolist()
        n = len(columns)
        
        te_matrix = np.zeros((n, n))
        p_matrix = np.zeros((n, n)) if significance_test else None
        
        for i, source_col in enumerate(columns):
            for j, target_col in enumerate(columns):
                if i == j:
                    continue
                
                source = data[source_col].values
                target = data[target_col].values
                
                if significance_test:
                    te, p, _ = self.significance_test(source, target, n_surrogates)
                    te_matrix[i, j] = te
                    p_matrix[i, j] = p
                else:
                    te_matrix[i, j] = self.compute_transfer_entropy(source, target)
        
        # Compute asymmetry matrix
        asymmetry = te_matrix - te_matrix.T
        
        # Identify dominant directions
        net_flow = {}
        for i, col_i in enumerate(columns):
            outflow = te_matrix[i, :].sum()
            inflow = te_matrix[:, i].sum()
            net_flow[col_i] = outflow - inflow
        
        return {
            'matrix': pd.DataFrame(te_matrix, index=columns, columns=columns),
            'asymmetry': pd.DataFrame(asymmetry, index=columns, columns=columns),
            'p_values': pd.DataFrame(p_matrix, index=columns, columns=columns) if significance_test else None,
            'net_flow': net_flow,
            'columns': columns
        }
    
    def find_information_leaders(
        self,
        data: pd.DataFrame,
        threshold: float = 0.01
    ) -> Dict:
        """
        Identify variables that lead information flow.
        
        Parameters
        ----------
        data : pd.DataFrame
            Multivariate time series
        threshold : float
            Minimum TE to consider
            
        Returns
        -------
        dict
            Ranking of variables by information leadership
        """
        result = self.compute_pairwise_transfer_entropy(data)
        te_matrix = result['matrix'].values
        columns = result['columns']
        
        # For each variable, compute outgoing vs incoming TE
        leadership_scores = {}
        
        for i, col in enumerate(columns):
            # Outgoing: how much this variable predicts others
            outgoing = te_matrix[i, :].sum()
            
            # Incoming: how much others predict this variable
            incoming = te_matrix[:, i].sum()
            
            # Leadership: net outgoing information
            leadership_scores[col] = {
                'outgoing_te': outgoing,
                'incoming_te': incoming,
                'net_leadership': outgoing - incoming,
                'leadership_ratio': outgoing / (incoming + 1e-10)
            }
        
        # Rank by net leadership
        ranked = sorted(
            leadership_scores.items(),
            key=lambda x: x[1]['net_leadership'],
            reverse=True
        )
        
        return {
            'scores': leadership_scores,
            'ranking': [x[0] for x in ranked],
            'te_matrix': result['matrix']
        }
    
    def analyze(
        self,
        data: Union[pd.DataFrame, Tuple[np.ndarray, np.ndarray]],
        source_label: str = 'X',
        target_label: str = 'Y',
        significance: bool = True,
        effective: bool = True,
        n_surrogates: int = 100
    ) -> Dict:
        """
        Comprehensive transfer entropy analysis.
        
        Parameters
        ----------
        data : DataFrame or tuple
            Either multivariate DataFrame or (source, target) arrays
        source_label : str
            Label for source (if tuple input)
        target_label : str
            Label for target (if tuple input)
        significance : bool
            Compute significance tests
        effective : bool
            Compute effective (bias-corrected) TE
        n_surrogates : int
            Number of surrogates
            
        Returns
        -------
        dict
            Complete analysis results
        """
        results = {}
        
        if isinstance(data, pd.DataFrame):
            # Multivariate analysis
            pairwise = self.compute_pairwise_transfer_entropy(
                data, 
                significance_test=significance,
                n_surrogates=n_surrogates
            )
            
            results['type'] = 'multivariate'
            results['te_matrix'] = pairwise['matrix']
            results['asymmetry_matrix'] = pairwise['asymmetry']
            results['net_flow'] = pairwise['net_flow']
            
            if significance:
                results['p_value_matrix'] = pairwise['p_values']
            
            # Find leaders
            leadership = self.find_information_leaders(data)
            results['leadership'] = leadership
            
        else:
            # Bivariate analysis
            source, target = data
            source = np.asarray(source).flatten()
            target = np.asarray(target).flatten()
            
            results['type'] = 'bivariate'
            results['source'] = source_label
            results['target'] = target_label
            
            # Forward TE: X → Y
            forward_te = self.compute_transfer_entropy(source, target)
            results['forward_te'] = forward_te
            
            # Backward TE: Y → X
            backward_te = self.compute_transfer_entropy(target, source)
            results['backward_te'] = backward_te
            
            # Asymmetry
            results['asymmetry'] = forward_te - backward_te
            results['dominant_direction'] = (
                f"{source_label}→{target_label}" if forward_te > backward_te 
                else f"{target_label}→{source_label}"
            )
            
            if effective:
                raw, bias, eff = self.compute_effective_transfer_entropy(
                    source, target, n_surrogates
                )
                results['effective_forward_te'] = eff
                results['forward_bias'] = bias
                
                raw_b, bias_b, eff_b = self.compute_effective_transfer_entropy(
                    target, source, n_surrogates
                )
                results['effective_backward_te'] = eff_b
                results['backward_bias'] = bias_b
            
            if significance:
                _, p_forward, sig_forward = self.significance_test(
                    source, target, n_surrogates
                )
                _, p_backward, sig_backward = self.significance_test(
                    target, source, n_surrogates
                )
                
                results['p_value_forward'] = p_forward
                results['p_value_backward'] = p_backward
                results['significant_forward'] = sig_forward
                results['significant_backward'] = sig_backward
        
        return results


# Convenience function
def transfer_entropy_analysis(
    source: np.ndarray,
    target: np.ndarray,
    lag: int = 1,
    method: str = 'ksg'
) -> Dict:
    """
    Quick transfer entropy analysis between two time series.
    
    Parameters
    ----------
    source : np.ndarray
        Source time series
    target : np.ndarray
        Target time series
    lag : int
        Time lag
    method : str
        Estimation method
        
    Returns
    -------
    dict
        Analysis results
    """
    lens = TransferEntropyLens(method=method, lag=lag)
    return lens.analyze((source, target))


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    
    # Generate coupled time series with known directionality
    n = 1000
    
    # X drives Y (with lag)
    x = np.zeros(n)
    y = np.zeros(n)
    
    # AR processes with X → Y coupling
    for t in range(1, n):
        x[t] = 0.8 * x[t-1] + np.random.randn()
        y[t] = 0.5 * y[t-1] + 0.4 * x[t-1] + np.random.randn()  # X influences Y
    
    # Analyze
    lens = TransferEntropyLens(method='ksg', lag=1, history_length=1)
    results = lens.analyze((x, y), 'X', 'Y', significance=True, effective=True)
    
    print("Transfer Entropy Analysis Complete")
    print(f"\nAnalysis type: {results['type']}")
    print(f"\nForward TE (X→Y): {results['forward_te']:.4f}")
    print(f"Backward TE (Y→X): {results['backward_te']:.4f}")
    print(f"Asymmetry: {results['asymmetry']:.4f}")
    print(f"Dominant direction: {results['dominant_direction']}")
    
    if 'effective_forward_te' in results:
        print(f"\nEffective Forward TE: {results['effective_forward_te']:.4f}")
        print(f"Forward bias: {results['forward_bias']:.4f}")
    
    if 'p_value_forward' in results:
        print(f"\nP-value (X→Y): {results['p_value_forward']:.4f}")
        print(f"P-value (Y→X): {results['p_value_backward']:.4f}")
        print(f"X→Y significant: {results['significant_forward']}")
        print(f"Y→X significant: {results['significant_backward']}")
    
    # Multivariate example
    print("\n" + "="*50)
    print("Multivariate Example")
    
    # Create DataFrame with multiple variables
    z = 0.6 * x + 0.4 * np.random.randn(n)  # Z depends on X
    df = pd.DataFrame({'X': x, 'Y': y, 'Z': z})
    
    multi_results = lens.analyze(df, significance=False)
    
    print(f"\nTE Matrix:")
    print(multi_results['te_matrix'].round(4))
    
    print(f"\nNet Information Flow:")
    for var, flow in multi_results['net_flow'].items():
        direction = "outgoing" if flow > 0 else "incoming"
        print(f"  {var}: {flow:.4f} ({direction})")
    
    print(f"\nInformation Leadership Ranking:")
    for i, var in enumerate(multi_results['leadership']['ranking']):
        score = multi_results['leadership']['scores'][var]['net_leadership']
        print(f"  {i+1}. {var}: {score:.4f}")
