"""
Anomaly Detection Lens
======================
Automatically identifies unusual events, extreme values, and outliers
that indicate critical changes or regime shifts.

Key Applications:
- Extreme weather event detection (heatwaves, cold snaps)
- Market crash / flash crash identification
- Equipment failure prediction
- Data quality validation

Methods:
- Isolation Forest: Tree-based anomaly isolation
- One-Class SVM: Support vector boundary method
- Statistical Methods: Z-score, IQR, GESD
- Residual Analysis: Model-based outlier detection
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from scipy import stats
from scipy.spatial.distance import cdist
import warnings


@dataclass
class AnomalyResult:
    """Container for anomaly detection results."""
    anomaly_mask: np.ndarray      # Boolean mask of anomalies
    anomaly_indices: np.ndarray   # Indices of anomalies
    anomaly_scores: np.ndarray    # Anomaly scores (higher = more anomalous)
    threshold: float              # Score threshold used
    n_anomalies: int
    contamination: float          # Actual contamination rate


class IsolationTree:
    """Single isolation tree for the Isolation Forest algorithm."""
    
    def __init__(self, max_depth: int):
        self.max_depth = max_depth
        self.split_feature = None
        self.split_value = None
        self.left = None
        self.right = None
        self.size = 0
        self.is_leaf = False
    
    def fit(self, X: np.ndarray, depth: int = 0) -> 'IsolationTree':
        """Build the isolation tree."""
        n_samples, n_features = X.shape
        self.size = n_samples
        
        # Stopping conditions
        if depth >= self.max_depth or n_samples <= 1:
            self.is_leaf = True
            return self
        
        # Random feature and split value
        self.split_feature = np.random.randint(n_features)
        feature_min = X[:, self.split_feature].min()
        feature_max = X[:, self.split_feature].max()
        
        if feature_min == feature_max:
            self.is_leaf = True
            return self
        
        self.split_value = np.random.uniform(feature_min, feature_max)
        
        # Split data
        left_mask = X[:, self.split_feature] < self.split_value
        right_mask = ~left_mask
        
        if left_mask.sum() == 0 or right_mask.sum() == 0:
            self.is_leaf = True
            return self
        
        self.left = IsolationTree(self.max_depth).fit(X[left_mask], depth + 1)
        self.right = IsolationTree(self.max_depth).fit(X[right_mask], depth + 1)
        
        return self
    
    def path_length(self, x: np.ndarray, depth: int = 0) -> float:
        """Compute path length for a single sample."""
        if self.is_leaf:
            # Average path length for remaining samples
            return depth + self._c(self.size)
        
        if x[self.split_feature] < self.split_value:
            return self.left.path_length(x, depth + 1)
        else:
            return self.right.path_length(x, depth + 1)
    
    @staticmethod
    def _c(n: int) -> float:
        """Average path length of unsuccessful search in BST."""
        if n <= 1:
            return 0
        return 2 * (np.log(n - 1) + 0.5772156649) - 2 * (n - 1) / n


class IsolationForest:
    """
    Isolation Forest for anomaly detection.
    
    Anomalies are isolated quickly (short path length) because they
    are few and different from normal points.
    """
    
    def __init__(
        self,
        n_estimators: int = 100,
        max_samples: Union[int, float, str] = 'auto',
        contamination: float = 0.1,
        random_state: Optional[int] = None
    ):
        """
        Initialize Isolation Forest.
        
        Parameters
        ----------
        n_estimators : int
            Number of trees
        max_samples : int, float, or 'auto'
            Number of samples for each tree
        contamination : float
            Expected proportion of anomalies
        random_state : int, optional
            Random seed
        """
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.contamination = contamination
        self.random_state = random_state
        
        self.trees: List[IsolationTree] = []
        self._fitted = False
    
    def fit(self, X: np.ndarray) -> 'IsolationForest':
        """
        Fit the Isolation Forest.
        
        Parameters
        ----------
        X : np.ndarray
            Training data, shape (n_samples, n_features)
            
        Returns
        -------
        self
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        n_samples = X.shape[0]
        
        # Determine sample size
        if self.max_samples == 'auto':
            self._max_samples = min(256, n_samples)
        elif isinstance(self.max_samples, float):
            self._max_samples = int(self.max_samples * n_samples)
        else:
            self._max_samples = min(self.max_samples, n_samples)
        
        # Maximum tree depth
        max_depth = int(np.ceil(np.log2(max(self._max_samples, 2))))
        
        # Build trees
        self.trees = []
        for _ in range(self.n_estimators):
            # Sample data
            indices = np.random.choice(n_samples, self._max_samples, replace=False)
            X_sample = X[indices]
            
            # Build tree
            tree = IsolationTree(max_depth).fit(X_sample)
            self.trees.append(tree)
        
        # Compute threshold based on training data
        scores = self.score_samples(X)
        self.threshold_ = np.percentile(scores, 100 * (1 - self.contamination))
        
        self._fitted = True
        return self
    
    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """
        Compute anomaly scores for samples.
        
        Higher scores indicate more anomalous points.
        
        Parameters
        ----------
        X : np.ndarray
            Samples to score
            
        Returns
        -------
        np.ndarray
            Anomaly scores in [0, 1]
        """
        if not self.trees:
            raise ValueError("Model must be fitted first")
        
        # Average path length
        c_n = IsolationTree._c(self._max_samples)
        
        scores = np.zeros(len(X))
        for i, x in enumerate(X):
            path_lengths = [tree.path_length(x) for tree in self.trees]
            avg_path = np.mean(path_lengths)
            # Anomaly score: 2^(-avg_path / c(n))
            scores[i] = 2 ** (-avg_path / c_n) if c_n > 0 else 0.5
        
        return scores
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict anomaly labels.
        
        Parameters
        ----------
        X : np.ndarray
            Samples to classify
            
        Returns
        -------
        np.ndarray
            1 for anomaly, 0 for normal
        """
        scores = self.score_samples(X)
        return (scores > self.threshold_).astype(int)


class OneClassSVM:
    """
    Simplified One-Class SVM using RBF kernel.
    
    Learns a boundary around normal data; points outside are anomalies.
    """
    
    def __init__(
        self,
        kernel: str = 'rbf',
        gamma: Union[float, str] = 'scale',
        nu: float = 0.1,
        max_iter: int = 1000,
        tol: float = 1e-3
    ):
        """
        Initialize One-Class SVM.
        
        Parameters
        ----------
        kernel : str
            Kernel type ('rbf', 'linear', 'poly')
        gamma : float or 'scale'
            RBF kernel parameter
        nu : float
            Upper bound on fraction of outliers
        max_iter : int
            Maximum iterations for optimization
        tol : float
            Convergence tolerance
        """
        self.kernel = kernel
        self.gamma = gamma
        self.nu = nu
        self.max_iter = max_iter
        self.tol = tol
        
        self._fitted = False
    
    def _compute_kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """Compute kernel matrix."""
        if self.kernel == 'rbf':
            # RBF kernel: exp(-gamma * ||x - y||^2)
            sq_dists = cdist(X1, X2, 'sqeuclidean')
            return np.exp(-self._gamma * sq_dists)
        elif self.kernel == 'linear':
            return X1 @ X2.T
        elif self.kernel == 'poly':
            return (1 + X1 @ X2.T) ** 3
        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")
    
    def fit(self, X: np.ndarray) -> 'OneClassSVM':
        """
        Fit the One-Class SVM.
        
        Uses a simplified SMO-style algorithm.
        
        Parameters
        ----------
        X : np.ndarray
            Training data (assumed to be mostly normal)
            
        Returns
        -------
        self
        """
        n_samples, n_features = X.shape
        
        # Set gamma
        if self.gamma == 'scale':
            self._gamma = 1.0 / (n_features * X.var())
        else:
            self._gamma = self.gamma
        
        self.X_train = X.copy()
        
        # Compute kernel matrix
        K = self._compute_kernel(X, X)
        
        # Initialize alphas (Lagrange multipliers)
        alphas = np.ones(n_samples) / n_samples
        
        # Upper bound for alphas
        C = 1.0 / (n_samples * self.nu)
        
        # SMO-style optimization (simplified)
        for iteration in range(self.max_iter):
            alpha_old = alphas.copy()
            
            for i in range(n_samples):
                # Compute decision function
                f_i = (alphas * K[i, :]).sum()
                
                # Gradient
                grad = K[i, :] @ alphas - 1
                
                # Update alpha
                alphas[i] = max(0, min(C, alphas[i] - grad / (K[i, i] + 1e-8)))
            
            # Project to sum = 1
            alphas = alphas / alphas.sum()
            
            # Check convergence
            if np.linalg.norm(alphas - alpha_old) < self.tol:
                break
        
        self.alphas = alphas
        
        # Find support vectors (non-zero alphas)
        sv_mask = alphas > 1e-7
        self.support_vectors = X[sv_mask]
        self.support_alphas = alphas[sv_mask]
        
        # Compute rho (offset)
        # Use average of decision function on support vectors
        K_sv = self._compute_kernel(self.support_vectors, X)
        decision_values = K_sv @ alphas
        self.rho = np.median(decision_values)
        
        self._fitted = True
        return self
    
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Compute decision function values.
        
        Positive values indicate normal points.
        
        Parameters
        ----------
        X : np.ndarray
            Samples to evaluate
            
        Returns
        -------
        np.ndarray
            Decision values
        """
        if not self._fitted:
            raise ValueError("Model must be fitted first")
        
        K = self._compute_kernel(X, self.X_train)
        return K @ self.alphas - self.rho
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict anomaly labels.
        
        Parameters
        ----------
        X : np.ndarray
            Samples to classify
            
        Returns
        -------
        np.ndarray
            1 for anomaly, 0 for normal
        """
        decision = self.decision_function(X)
        return (decision < 0).astype(int)
    
    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """
        Compute anomaly scores (negative decision function).
        
        Higher scores indicate more anomalous points.
        """
        return -self.decision_function(X)


class StatisticalAnomalyDetector:
    """
    Statistical methods for anomaly detection.
    
    Methods:
    - Z-score: Assumes normal distribution
    - IQR: Robust to non-normality
    - GESD: Generalized ESD test for multiple outliers
    - Residual-based: Uses model residuals
    """
    
    def __init__(self, method: str = 'zscore', threshold: float = 3.0):
        """
        Initialize detector.
        
        Parameters
        ----------
        method : str
            'zscore', 'iqr', 'gesd', 'mad' (median absolute deviation)
        threshold : float
            Detection threshold (method-specific)
        """
        self.method = method
        self.threshold = threshold
        self._fitted = False
    
    def fit(self, data: np.ndarray) -> 'StatisticalAnomalyDetector':
        """
        Fit the detector (compute statistics).
        
        Parameters
        ----------
        data : np.ndarray
            Training data
            
        Returns
        -------
        self
        """
        data = np.asarray(data).flatten()
        
        if self.method == 'zscore':
            self.mean_ = np.mean(data)
            self.std_ = np.std(data)
        
        elif self.method == 'iqr':
            self.q1_ = np.percentile(data, 25)
            self.q3_ = np.percentile(data, 75)
            self.iqr_ = self.q3_ - self.q1_
        
        elif self.method == 'mad':
            self.median_ = np.median(data)
            self.mad_ = np.median(np.abs(data - self.median_))
        
        elif self.method == 'gesd':
            self.mean_ = np.mean(data)
            self.std_ = np.std(data)
        
        self._fitted = True
        return self
    
    def score_samples(self, data: np.ndarray) -> np.ndarray:
        """
        Compute anomaly scores.
        
        Parameters
        ----------
        data : np.ndarray
            Data to score
            
        Returns
        -------
        np.ndarray
            Anomaly scores (higher = more anomalous)
        """
        if not self._fitted:
            raise ValueError("Detector must be fitted first")
        
        data = np.asarray(data).flatten()
        
        if self.method == 'zscore':
            return np.abs((data - self.mean_) / (self.std_ + 1e-10))
        
        elif self.method == 'iqr':
            lower = self.q1_ - self.threshold * self.iqr_
            upper = self.q3_ + self.threshold * self.iqr_
            # Distance from acceptable range
            below = np.maximum(0, lower - data)
            above = np.maximum(0, data - upper)
            return (below + above) / (self.iqr_ + 1e-10)
        
        elif self.method == 'mad':
            # Modified Z-score using MAD
            k = 1.4826  # Scale factor for normal distribution
            return np.abs(data - self.median_) / (k * self.mad_ + 1e-10)
        
        elif self.method == 'gesd':
            return np.abs((data - self.mean_) / (self.std_ + 1e-10))
    
    def predict(self, data: np.ndarray) -> np.ndarray:
        """
        Predict anomaly labels.
        
        Parameters
        ----------
        data : np.ndarray
            Data to classify
            
        Returns
        -------
        np.ndarray
            1 for anomaly, 0 for normal
        """
        scores = self.score_samples(data)
        return (scores > self.threshold).astype(int)
    
    def detect_gesd(
        self,
        data: np.ndarray,
        max_outliers: int = 10,
        alpha: float = 0.05
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generalized Extreme Studentized Deviate test.
        
        Handles multiple outliers by iteratively removing the most extreme.
        
        Parameters
        ----------
        data : np.ndarray
            Data to test
        max_outliers : int
            Maximum outliers to detect
        alpha : float
            Significance level
            
        Returns
        -------
        tuple
            (outlier_indices, test_statistics)
        """
        data = np.asarray(data).flatten()
        n = len(data)
        
        outliers = []
        test_stats = []
        
        remaining = data.copy()
        original_indices = np.arange(n)
        
        for i in range(min(max_outliers, n - 2)):
            # Find most extreme point
            mean = np.mean(remaining)
            std = np.std(remaining, ddof=1)
            
            if std < 1e-10:
                break
            
            deviations = np.abs(remaining - mean)
            idx = np.argmax(deviations)
            test_stat = deviations[idx] / std
            
            # Critical value (t-distribution based)
            p = 1 - alpha / (2 * (n - i))
            t_crit = stats.t.ppf(p, n - i - 2)
            lambda_crit = ((n - i - 1) * t_crit / 
                          np.sqrt((n - i - 2 + t_crit**2) * (n - i)))
            
            test_stats.append((test_stat, lambda_crit))
            
            if test_stat > lambda_crit:
                outliers.append(original_indices[idx])
                remaining = np.delete(remaining, idx)
                original_indices = np.delete(original_indices, idx)
            else:
                break
        
        return np.array(outliers), test_stats


class ResidualAnomalyDetector:
    """
    Detects anomalies based on model residuals.
    
    Fits a simple model (moving average, trend) and flags
    points with large residuals.
    """
    
    def __init__(
        self,
        model: str = 'moving_average',
        window: int = 10,
        threshold: float = 3.0
    ):
        """
        Initialize residual-based detector.
        
        Parameters
        ----------
        model : str
            'moving_average', 'exponential_smooth', 'linear_trend'
        window : int
            Window size for moving average
        threshold : float
            Z-score threshold for residuals
        """
        self.model = model
        self.window = window
        self.threshold = threshold
    
    def fit_predict(self, data: np.ndarray) -> AnomalyResult:
        """
        Fit model and detect anomalies.
        
        Parameters
        ----------
        data : np.ndarray
            Time series data
            
        Returns
        -------
        AnomalyResult
            Detection results
        """
        data = np.asarray(data).flatten()
        n = len(data)
        
        # Compute fitted values based on model
        if self.model == 'moving_average':
            fitted = np.convolve(data, np.ones(self.window)/self.window, mode='same')
            # Edge correction
            for i in range(self.window // 2):
                fitted[i] = data[:i+self.window//2+1].mean()
                fitted[-(i+1)] = data[-(i+self.window//2+1):].mean()
        
        elif self.model == 'exponential_smooth':
            alpha = 2 / (self.window + 1)
            fitted = np.zeros(n)
            fitted[0] = data[0]
            for i in range(1, n):
                fitted[i] = alpha * data[i] + (1 - alpha) * fitted[i-1]
        
        elif self.model == 'linear_trend':
            x = np.arange(n)
            coeffs = np.polyfit(x, data, 1)
            fitted = np.polyval(coeffs, x)
        
        else:
            raise ValueError(f"Unknown model: {self.model}")
        
        # Compute residuals
        residuals = data - fitted
        
        # Score residuals using MAD (robust)
        median_res = np.median(residuals)
        mad = np.median(np.abs(residuals - median_res))
        
        if mad < 1e-10:
            scores = np.abs(residuals - median_res)
        else:
            scores = np.abs(residuals - median_res) / (1.4826 * mad)
        
        # Detect anomalies
        anomaly_mask = scores > self.threshold
        
        return AnomalyResult(
            anomaly_mask=anomaly_mask,
            anomaly_indices=np.where(anomaly_mask)[0],
            anomaly_scores=scores,
            threshold=self.threshold,
            n_anomalies=anomaly_mask.sum(),
            contamination=anomaly_mask.mean()
        )


class AnomalyDetectionLens:
    """
    Anomaly Detection Lens combining multiple methods.
    
    Provides comprehensive anomaly detection with method comparison
    and consensus-based detection.
    """
    
    def __init__(self):
        """Initialize the Anomaly Detection Lens."""
        self.isolation_forest = None
        self.ocsvm = None
        self.statistical = None
        self.residual = None
    
    def detect_isolation_forest(
        self,
        data: np.ndarray,
        contamination: float = 0.1,
        n_estimators: int = 100
    ) -> AnomalyResult:
        """
        Detect anomalies using Isolation Forest.
        
        Parameters
        ----------
        data : np.ndarray
            Data (1D or 2D)
        contamination : float
            Expected anomaly proportion
        n_estimators : int
            Number of trees
            
        Returns
        -------
        AnomalyResult
        """
        X = np.atleast_2d(data).T if data.ndim == 1 else data
        
        self.isolation_forest = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination
        )
        self.isolation_forest.fit(X)
        
        scores = self.isolation_forest.score_samples(X)
        predictions = self.isolation_forest.predict(X)
        
        return AnomalyResult(
            anomaly_mask=predictions.astype(bool),
            anomaly_indices=np.where(predictions)[0],
            anomaly_scores=scores,
            threshold=self.isolation_forest.threshold_,
            n_anomalies=predictions.sum(),
            contamination=predictions.mean()
        )
    
    def detect_ocsvm(
        self,
        data: np.ndarray,
        nu: float = 0.1,
        gamma: Union[float, str] = 'scale'
    ) -> AnomalyResult:
        """
        Detect anomalies using One-Class SVM.
        
        Parameters
        ----------
        data : np.ndarray
            Data (1D or 2D)
        nu : float
            Upper bound on outlier fraction
        gamma : float or 'scale'
            RBF kernel parameter
            
        Returns
        -------
        AnomalyResult
        """
        X = np.atleast_2d(data).T if data.ndim == 1 else data
        
        self.ocsvm = OneClassSVM(nu=nu, gamma=gamma)
        self.ocsvm.fit(X)
        
        scores = self.ocsvm.score_samples(X)
        predictions = self.ocsvm.predict(X)
        
        return AnomalyResult(
            anomaly_mask=predictions.astype(bool),
            anomaly_indices=np.where(predictions)[0],
            anomaly_scores=scores,
            threshold=0.0,
            n_anomalies=predictions.sum(),
            contamination=predictions.mean()
        )
    
    def detect_statistical(
        self,
        data: np.ndarray,
        method: str = 'zscore',
        threshold: float = 3.0
    ) -> AnomalyResult:
        """
        Detect anomalies using statistical methods.
        
        Parameters
        ----------
        data : np.ndarray
            1D time series
        method : str
            'zscore', 'iqr', 'mad'
        threshold : float
            Detection threshold
            
        Returns
        -------
        AnomalyResult
        """
        data = np.asarray(data).flatten()
        
        self.statistical = StatisticalAnomalyDetector(method=method, threshold=threshold)
        self.statistical.fit(data)
        
        scores = self.statistical.score_samples(data)
        predictions = self.statistical.predict(data)
        
        return AnomalyResult(
            anomaly_mask=predictions.astype(bool),
            anomaly_indices=np.where(predictions)[0],
            anomaly_scores=scores,
            threshold=threshold,
            n_anomalies=predictions.sum(),
            contamination=predictions.mean()
        )
    
    def detect_residual(
        self,
        data: np.ndarray,
        model: str = 'moving_average',
        window: int = 10,
        threshold: float = 3.0
    ) -> AnomalyResult:
        """
        Detect anomalies using residual analysis.
        
        Parameters
        ----------
        data : np.ndarray
            1D time series
        model : str
            'moving_average', 'exponential_smooth', 'linear_trend'
        window : int
            Smoothing window
        threshold : float
            Residual z-score threshold
            
        Returns
        -------
        AnomalyResult
        """
        self.residual = ResidualAnomalyDetector(
            model=model,
            window=window,
            threshold=threshold
        )
        return self.residual.fit_predict(data)
    
    def consensus_detection(
        self,
        data: np.ndarray,
        min_votes: int = 2,
        contamination: float = 0.1
    ) -> Dict:
        """
        Detect anomalies using consensus of multiple methods.
        
        Points flagged by multiple methods are more likely true anomalies.
        
        Parameters
        ----------
        data : np.ndarray
            Input data
        min_votes : int
            Minimum methods that must flag a point
        contamination : float
            Expected contamination for applicable methods
            
        Returns
        -------
        dict
            Results from each method and consensus
        """
        results = {}
        
        # Run all methods
        results['isolation_forest'] = self.detect_isolation_forest(
            data, contamination=contamination
        )
        results['ocsvm'] = self.detect_ocsvm(data, nu=contamination)
        results['zscore'] = self.detect_statistical(data, method='zscore')
        results['mad'] = self.detect_statistical(data, method='mad')
        results['residual'] = self.detect_residual(data)
        
        # Compute votes
        n = len(np.asarray(data).flatten())
        votes = np.zeros(n)
        
        for method, result in results.items():
            votes += result.anomaly_mask.flatten()
        
        # Consensus anomalies
        consensus_mask = votes >= min_votes
        
        results['consensus'] = AnomalyResult(
            anomaly_mask=consensus_mask,
            anomaly_indices=np.where(consensus_mask)[0],
            anomaly_scores=votes,
            threshold=min_votes,
            n_anomalies=consensus_mask.sum(),
            contamination=consensus_mask.mean()
        )
        
        results['vote_matrix'] = votes
        
        return results
    
    def analyze(
        self,
        data: Union[np.ndarray, pd.Series],
        contamination: float = 0.1,
        methods: Optional[List[str]] = None
    ) -> Dict:
        """
        Comprehensive anomaly analysis.
        
        Parameters
        ----------
        data : array-like
            Input data
        contamination : float
            Expected anomaly fraction
        methods : list, optional
            Methods to use. If None, uses all.
            
        Returns
        -------
        dict
            Complete analysis results
        """
        if isinstance(data, pd.Series):
            data = data.values
        data = np.asarray(data)
        
        if methods is None:
            methods = ['isolation_forest', 'ocsvm', 'zscore', 'mad', 'residual']
        
        results = {
            'n_observations': len(data.flatten()),
            'methods_used': methods
        }
        
        # Run selected methods
        if 'isolation_forest' in methods:
            results['isolation_forest'] = self.detect_isolation_forest(data, contamination)
        
        if 'ocsvm' in methods:
            results['ocsvm'] = self.detect_ocsvm(data, nu=contamination)
        
        if 'zscore' in methods:
            results['zscore'] = self.detect_statistical(data, 'zscore')
        
        if 'mad' in methods:
            results['mad'] = self.detect_statistical(data, 'mad')
        
        if 'iqr' in methods:
            results['iqr'] = self.detect_statistical(data, 'iqr')
        
        if 'residual' in methods:
            results['residual'] = self.detect_residual(data)
        
        # Compute consensus
        votes = np.zeros(len(data.flatten()))
        for method in methods:
            if method in results and hasattr(results[method], 'anomaly_mask'):
                votes += results[method].anomaly_mask.flatten()
        
        consensus_threshold = max(2, len(methods) // 2)
        consensus_mask = votes >= consensus_threshold
        
        results['consensus'] = {
            'anomaly_indices': np.where(consensus_mask)[0],
            'n_anomalies': consensus_mask.sum(),
            'votes': votes,
            'threshold': consensus_threshold
        }
        
        # Summary statistics
        results['summary'] = {
            method: results[method].n_anomalies 
            for method in methods 
            if method in results and hasattr(results[method], 'n_anomalies')
        }
        
        return results


# Convenience function
def detect_anomalies(
    data: np.ndarray,
    method: str = 'consensus',
    contamination: float = 0.1
) -> Dict:
    """
    Quick anomaly detection.
    
    Parameters
    ----------
    data : np.ndarray
        Input data
    method : str
        Detection method or 'consensus' for multi-method
    contamination : float
        Expected anomaly proportion
        
    Returns
    -------
    dict
        Detection results
    """
    lens = AnomalyDetectionLens()
    
    if method == 'consensus':
        return lens.consensus_detection(data, contamination=contamination)
    else:
        return lens.analyze(data, contamination=contamination, methods=[method])


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    
    # Generate data with anomalies
    n = 500
    data = np.random.randn(n)
    
    # Add some anomalies
    anomaly_indices = [50, 150, 250, 350, 450]
    for idx in anomaly_indices:
        data[idx] = np.random.choice([-5, 5]) + np.random.randn()
    
    # Analyze
    lens = AnomalyDetectionLens()
    results = lens.analyze(data, contamination=0.02)
    
    print("Anomaly Detection Analysis Complete")
    print(f"\nMethods used: {results['methods_used']}")
    print(f"\nDetections by method:")
    for method, count in results['summary'].items():
        print(f"  {method}: {count} anomalies")
    
    print(f"\nConsensus anomalies: {results['consensus']['n_anomalies']}")
    print(f"Consensus indices: {results['consensus']['anomaly_indices'][:10]}...")
    
    # Check how many true anomalies were found
    detected = set(results['consensus']['anomaly_indices'])
    true = set(anomaly_indices)
    recall = len(detected & true) / len(true)
    precision = len(detected & true) / len(detected) if detected else 0
    
    print(f"\nTrue anomalies: {anomaly_indices}")
    print(f"Recall: {recall*100:.1f}%")
    print(f"Precision: {precision*100:.1f}%")
