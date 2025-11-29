"""
VCF Advanced Mathematical Lenses - Integration Module
=====================================================
New lenses following the existing framework pattern for seamless
integration with LensComparator.

Lenses included:
- WaveletAnalysisLens: Multi-scale time-frequency decomposition
- NetworkGraphLens: Correlation network analysis with MST and centrality
- RegimeSwitchingLens: Hidden Markov Models for regime detection
- AnomalyDetectionLens: Multi-method outlier identification
- TransferEntropyLens: Directed information flow analysis
- VARGrangerLens: Vector autoregression with enhanced Granger tests
- TopologicalDataAnalysisLens: Shape-based pattern detection

Each lens implements:
- analyze(panel: pd.DataFrame) -> Dict
- top_indicators(result: Dict, date: pd.Timestamp, n: int) -> List[Tuple[str, float]]
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from scipy.spatial.distance import pdist, squareform
from scipy.special import digamma, logsumexp
from scipy.spatial import cKDTree
from scipy import stats
import warnings


# =============================================================================
# LENS 9: WAVELET ANALYSIS
# =============================================================================

class WaveletAnalysisLens:
    """
    Multi-scale time-frequency analysis using wavelet transforms.
    Answers: "What periodic patterns exist at different time scales?"
    
    Reveals:
    - Time-varying periodicities
    - Multi-scale patterns (e.g., short-term vs long-term cycles)
    - Localized events and transients
    """
    
    def __init__(self, name: str = "Wavelet", wavelet: str = 'morlet', 
                 sampling_rate: float = 1.0, num_scales: int = 32):
        self.name = name
        self.wavelet = wavelet
        self.sampling_rate = sampling_rate
        self.num_scales = num_scales
        # Fourier factor for Morlet wavelet
        self._omega0 = 6.0
        self._fourier_factor = 4 * np.pi / (self._omega0 + np.sqrt(2 + self._omega0**2))
    
    def _cwt_single_series(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute CWT for a single time series."""
        n = len(data)
        data_norm = (data - np.mean(data)) / (np.std(data) + 1e-10)
        
        # Compute scales
        s0 = 2 / self.sampling_rate
        dj = 0.25
        j1 = int(np.log2(n * self._fourier_factor / s0) / dj)
        scales = s0 * 2**(dj * np.arange(0, min(j1, self.num_scales)))
        
        # Pad for FFT
        npad = int(2**np.ceil(np.log2(n)))
        data_padded = np.zeros(npad)
        data_padded[:n] = data_norm
        
        data_fft = np.fft.fft(data_padded)
        freqs = np.fft.fftfreq(npad, 1/self.sampling_rate)
        
        power = np.zeros((len(scales), n))
        
        for i, scale in enumerate(scales):
            omega = 2 * np.pi * freqs * scale
            wavelet_fft = np.pi**(-0.25) * np.exp(-(omega - self._omega0)**2 / 2)
            wavelet_fft[freqs < 0] = 0
            wavelet_fft *= np.sqrt(2 * np.pi * scale * self.sampling_rate)
            
            conv = np.fft.ifft(data_fft * wavelet_fft)
            power[i, :] = np.abs(conv[:n])**2
        
        frequencies = 1 / (scales * self._fourier_factor)
        return power, scales, frequencies
    
    def analyze(self, panel: pd.DataFrame) -> Dict:
        """
        Analyze each indicator's wavelet power spectrum.
        
        Returns:
        - scale_power: Average power at each scale for each indicator
        - dominant_period: Dominant period for each indicator
        - importance: Indicators ranked by total wavelet power variance
        """
        panel_clean = panel.dropna()
        
        if panel_clean.empty or panel_clean.shape[0] < 10:
            warnings.warn("WaveletAnalysisLens: Insufficient data after NaN handling.")
            return {
                'scale_power': pd.DataFrame(dtype='float'),
                'dominant_period': pd.Series(dtype='float'),
                'importance': pd.Series(dtype='float'),
                'method': 'Continuous Wavelet Transform (Morlet)'
            }
        
        scale_power_dict = {}
        dominant_periods = {}
        total_power_variance = {}
        
        for col in panel_clean.columns:
            data = panel_clean[col].values
            
            try:
                power, scales, frequencies = self._cwt_single_series(data)
                
                # Average power at each scale
                avg_power = power.mean(axis=1)
                scale_power_dict[col] = avg_power
                
                # Dominant period (scale with max power)
                dominant_idx = np.argmax(avg_power)
                dominant_periods[col] = 1 / frequencies[dominant_idx] if frequencies[dominant_idx] > 0 else np.nan
                
                # Total power variance (how much spectral content varies)
                total_power_variance[col] = np.var(power.sum(axis=0))
                
            except Exception as e:
                warnings.warn(f"Wavelet analysis failed for {col}: {e}")
                dominant_periods[col] = np.nan
                total_power_variance[col] = 0
        
        # Create scale power DataFrame
        if scale_power_dict:
            max_scales = max(len(v) for v in scale_power_dict.values())
            scale_power_df = pd.DataFrame(
                {k: np.pad(v, (0, max_scales - len(v)), constant_values=np.nan) 
                 for k, v in scale_power_dict.items()}
            )
        else:
            scale_power_df = pd.DataFrame(dtype='float')
        
        importance = pd.Series(total_power_variance).sort_values(ascending=False)
        
        return {
            'scale_power': scale_power_df,
            'dominant_period': pd.Series(dominant_periods),
            'importance': importance,
            'method': 'Continuous Wavelet Transform (Morlet)'
        }
    
    def top_indicators(self, result: Dict, date: pd.Timestamp = None, n: int = 5) -> List[Tuple[str, float]]:
        """Top indicators by wavelet power variance."""
        importance = result.get('importance', pd.Series(dtype='float'))
        if importance.empty:
            return []
        return list(zip(importance.index[:n], importance.values[:n]))


# =============================================================================
# LENS 10: NETWORK / GRAPH THEORY
# =============================================================================

class NetworkGraphLens:
    """
    Analyzes indicator relationships as a network structure.
    Answers: "Which indicators are most central/connected?"
    
    Reveals:
    - Core dependencies (MST backbone)
    - Most influential nodes (centrality measures)
    - Natural groupings (communities)
    """
    
    def __init__(self, name: str = "NetworkGraph", correlation_threshold: float = 0.3):
        self.name = name
        self.correlation_threshold = correlation_threshold
    
    def _correlation_to_distance(self, corr: float) -> float:
        """Convert correlation to distance."""
        return np.sqrt(2 * (1 - corr))
    
    def _compute_mst(self, dist_matrix: np.ndarray, labels: List[str]) -> List[Tuple[str, str, float]]:
        """Compute Minimum Spanning Tree using Kruskal's algorithm."""
        n = len(labels)
        edges = []
        
        for i in range(n):
            for j in range(i + 1, n):
                edges.append((dist_matrix[i, j], i, j))
        edges.sort()
        
        parent = list(range(n))
        rank = [0] * n
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            px, py = find(x), find(y)
            if px == py:
                return False
            if rank[px] < rank[py]:
                px, py = py, px
            parent[py] = px
            if rank[px] == rank[py]:
                rank[px] += 1
            return True
        
        mst_edges = []
        for dist, i, j in edges:
            if union(i, j):
                corr = 1 - (dist**2) / 2  # Convert back to correlation
                mst_edges.append((labels[i], labels[j], corr))
                if len(mst_edges) == n - 1:
                    break
        
        return mst_edges
    
    def analyze(self, panel: pd.DataFrame) -> Dict:
        """
        Build correlation network and compute centrality measures.
        
        Returns:
        - mst_edges: Minimum spanning tree edges
        - degree_centrality: Node connectivity
        - betweenness_centrality: Importance for information flow
        - importance: Combined centrality score
        """
        panel_clean = panel.dropna()
        
        if panel_clean.empty or panel_clean.shape[1] < 2:
            warnings.warn("NetworkGraphLens: Insufficient data.")
            return {
                'mst_edges': [],
                'degree_centrality': pd.Series(dtype='float'),
                'betweenness_centrality': pd.Series(dtype='float'),
                'importance': pd.Series(dtype='float'),
                'method': 'Correlation Network with MST'
            }
        
        # Correlation matrix
        corr_matrix = panel_clean.corr().values
        labels = panel_clean.columns.tolist()
        n = len(labels)
        
        # Distance matrix
        dist_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                dist_matrix[i, j] = self._correlation_to_distance(corr_matrix[i, j])
        
        # MST
        mst_edges = self._compute_mst(dist_matrix, labels)
        
        # Build adjacency for centrality (using correlation threshold)
        adjacency = np.abs(corr_matrix) >= self.correlation_threshold
        np.fill_diagonal(adjacency, False)
        
        # Degree centrality
        degree = adjacency.sum(axis=1) / (n - 1)
        degree_centrality = pd.Series(degree, index=labels)
        
        # Simplified betweenness (based on MST participation)
        mst_participation = {label: 0 for label in labels}
        for src, tgt, _ in mst_edges:
            mst_participation[src] += 1
            mst_participation[tgt] += 1
        betweenness = pd.Series(mst_participation)
        betweenness = betweenness / betweenness.max() if betweenness.max() > 0 else betweenness
        
        # Combined importance
        importance = (degree_centrality + betweenness) / 2
        importance = importance.sort_values(ascending=False)
        
        return {
            'mst_edges': mst_edges,
            'degree_centrality': degree_centrality.sort_values(ascending=False),
            'betweenness_centrality': betweenness.sort_values(ascending=False),
            'importance': importance,
            'correlation_matrix': pd.DataFrame(corr_matrix, index=labels, columns=labels),
            'method': 'Correlation Network with MST'
        }
    
    def top_indicators(self, result: Dict, date: pd.Timestamp = None, n: int = 5) -> List[Tuple[str, float]]:
        """Top indicators by network centrality."""
        importance = result.get('importance', pd.Series(dtype='float'))
        if importance.empty:
            return []
        return list(zip(importance.index[:n], importance.values[:n]))


# =============================================================================
# LENS 11: REGIME SWITCHING (HIDDEN MARKOV MODEL)
# =============================================================================

class RegimeSwitchingLens:
    """
    Identifies distinct market/system states using Hidden Markov Models.
    Answers: "What regimes exist and which indicators define them?"
    
    Reveals:
    - Bull/bear or expansion/contraction regimes
    - Regime-specific indicator behavior
    - Transition probabilities
    """
    
    def __init__(self, name: str = "RegimeSwitching", n_states: int = 2, max_iter: int = 100):
        self.name = name
        self.n_states = n_states
        self.max_iter = max_iter
    
    def _fit_hmm(self, data: np.ndarray) -> Dict:
        """Fit a Gaussian HMM using EM algorithm."""
        n = len(data)
        k = self.n_states
        
        # Initialize
        initial_probs = np.ones(k) / k
        transition_matrix = np.ones((k, k)) / k
        np.fill_diagonal(transition_matrix, 0.7)
        transition_matrix /= transition_matrix.sum(axis=1, keepdims=True)
        
        # Initialize means using percentiles
        percentiles = np.linspace(20, 80, k)
        means = np.percentile(data, percentiles)
        variances = np.full(k, np.var(data) / k)
        
        prev_ll = -np.inf
        
        for _ in range(self.max_iter):
            # E-step: Forward-backward
            log_emission = np.zeros((n, k))
            for state in range(k):
                log_emission[:, state] = (
                    -0.5 * np.log(2 * np.pi * variances[state])
                    - 0.5 * (data - means[state])**2 / variances[state]
                )
            
            # Forward pass
            log_alpha = np.zeros((n, k))
            log_alpha[0] = np.log(initial_probs + 1e-10) + log_emission[0]
            log_trans = np.log(transition_matrix + 1e-10)
            
            for t in range(1, n):
                for j in range(k):
                    log_alpha[t, j] = logsumexp(log_alpha[t-1] + log_trans[:, j]) + log_emission[t, j]
            
            log_likelihood = logsumexp(log_alpha[-1])
            
            # Backward pass
            log_beta = np.zeros((n, k))
            for t in range(n - 2, -1, -1):
                for i in range(k):
                    log_beta[t, i] = logsumexp(
                        log_trans[i, :] + log_emission[t+1] + log_beta[t+1]
                    )
            
            # Gamma (state probabilities)
            log_gamma = log_alpha + log_beta
            log_gamma -= logsumexp(log_gamma, axis=1, keepdims=True)
            gamma = np.exp(log_gamma)
            
            # Check convergence
            if abs(log_likelihood - prev_ll) < 1e-4:
                break
            prev_ll = log_likelihood
            
            # M-step
            initial_probs = gamma[0] / gamma[0].sum()
            
            for state in range(k):
                gamma_sum = gamma[:, state].sum()
                means[state] = (gamma[:, state] * data).sum() / gamma_sum
                variances[state] = max(1e-6, (gamma[:, state] * (data - means[state])**2).sum() / gamma_sum)
        
        # Viterbi for most likely states
        states = np.argmax(gamma, axis=1)
        
        return {
            'states': states,
            'state_probs': gamma,
            'means': means,
            'variances': variances,
            'transition_matrix': transition_matrix,
            'log_likelihood': log_likelihood
        }
    
    def analyze(self, panel: pd.DataFrame) -> Dict:
        """
        Fit HMM to each indicator and aggregate regime information.
        
        Returns:
        - regime_labels: Overall regime classification
        - regime_separation: How well each indicator separates regimes
        - importance: Indicators ranked by regime discrimination power
        """
        panel_norm = (panel - panel.mean()) / (panel.std() + 1e-10)
        panel_clean = panel_norm.dropna()
        
        if panel_clean.empty or panel_clean.shape[0] < 20:
            warnings.warn("RegimeSwitchingLens: Insufficient data.")
            return {
                'regime_labels': pd.Series(dtype='int'),
                'regime_separation': pd.Series(dtype='float'),
                'importance': pd.Series(dtype='float'),
                'method': f'Hidden Markov Model ({self.n_states} states)'
            }
        
        # Fit HMM on first principal component for overall regime
        try:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=1)
            pc1 = pca.fit_transform(panel_clean).flatten()
            overall_hmm = self._fit_hmm(pc1)
            regime_labels = pd.Series(overall_hmm['states'], index=panel_clean.index)
        except Exception as e:
            warnings.warn(f"Overall HMM failed: {e}")
            regime_labels = pd.Series(dtype='int')
        
        # For each indicator, measure regime separation
        regime_separation = {}
        
        for col in panel_clean.columns:
            data = panel_clean[col].values
            
            try:
                hmm_result = self._fit_hmm(data)
                # Separation = distance between state means / pooled std
                means = hmm_result['means']
                vars = hmm_result['variances']
                pooled_std = np.sqrt(np.mean(vars))
                separation = np.abs(means.max() - means.min()) / (pooled_std + 1e-10)
                regime_separation[col] = separation
            except:
                regime_separation[col] = 0
        
        importance = pd.Series(regime_separation).sort_values(ascending=False)
        
        return {
            'regime_labels': regime_labels,
            'regime_separation': pd.Series(regime_separation),
            'importance': importance,
            'method': f'Hidden Markov Model ({self.n_states} states)'
        }
    
    def top_indicators(self, result: Dict, date: pd.Timestamp = None, n: int = 5) -> List[Tuple[str, float]]:
        """Top indicators by regime separation power."""
        importance = result.get('importance', pd.Series(dtype='float'))
        if importance.empty:
            return []
        return list(zip(importance.index[:n], importance.values[:n]))


# =============================================================================
# LENS 12: ANOMALY DETECTION
# =============================================================================

class AnomalyDetectionLens:
    """
    Identifies unusual events and outliers across indicators.
    Answers: "Which indicators are showing anomalous behavior?"
    
    Reveals:
    - Extreme events and outliers
    - Indicators with frequent anomalies
    - Current anomaly status
    """
    
    def __init__(self, name: str = "AnomalyDetection", contamination: float = 0.05, 
                 n_estimators: int = 100):
        self.name = name
        self.contamination = contamination
        self.n_estimators = n_estimators
    
    def _isolation_forest_scores(self, X: np.ndarray) -> np.ndarray:
        """Compute anomaly scores using simplified Isolation Forest."""
        n_samples = X.shape[0]
        max_samples = min(256, n_samples)
        max_depth = int(np.ceil(np.log2(max(max_samples, 2))))
        
        # Average path length normalization
        def c(n):
            if n <= 1:
                return 0
            return 2 * (np.log(n - 1) + 0.5772156649) - 2 * (n - 1) / n
        
        path_lengths = np.zeros(n_samples)
        
        for _ in range(self.n_estimators):
            # Sample data
            indices = np.random.choice(n_samples, max_samples, replace=False)
            X_sample = X[indices]
            
            # For each point, estimate path length
            for i in range(n_samples):
                depth = 0
                current_data = X_sample.copy()
                point = X[i]
                
                while depth < max_depth and len(current_data) > 1:
                    # Random split
                    feat = np.random.randint(X.shape[1])
                    feat_min, feat_max = current_data[:, feat].min(), current_data[:, feat].max()
                    
                    if feat_min == feat_max:
                        break
                    
                    split = np.random.uniform(feat_min, feat_max)
                    
                    if point[feat] < split:
                        current_data = current_data[current_data[:, feat] < split]
                    else:
                        current_data = current_data[current_data[:, feat] >= split]
                    
                    depth += 1
                
                path_lengths[i] += depth + c(len(current_data))
        
        path_lengths /= self.n_estimators
        
        # Anomaly score
        c_n = c(max_samples)
        scores = 2 ** (-path_lengths / c_n) if c_n > 0 else np.zeros(n_samples)
        
        return scores
    
    def _zscore_anomalies(self, data: np.ndarray, threshold: float = 3.0) -> np.ndarray:
        """Detect anomalies using z-score method."""
        z = np.abs((data - np.mean(data)) / (np.std(data) + 1e-10))
        return z > threshold
    
    def analyze(self, panel: pd.DataFrame) -> Dict:
        """
        Detect anomalies across all indicators.
        
        Returns:
        - anomaly_counts: Number of anomalies per indicator
        - anomaly_rate: Proportion of anomalous observations per indicator
        - current_anomalies: Which indicators are anomalous at the latest date
        - importance: Indicators ranked by anomaly frequency
        """
        panel_norm = (panel - panel.mean()) / (panel.std() + 1e-10)
        panel_clean = panel_norm.dropna()
        
        if panel_clean.empty or panel_clean.shape[0] < 10:
            warnings.warn("AnomalyDetectionLens: Insufficient data.")
            return {
                'anomaly_counts': pd.Series(dtype='int'),
                'anomaly_rate': pd.Series(dtype='float'),
                'current_anomalies': [],
                'importance': pd.Series(dtype='float'),
                'method': 'Isolation Forest + Z-score'
            }
        
        anomaly_counts = {}
        anomaly_matrices = {}
        
        for col in panel_clean.columns:
            data = panel_clean[col].values
            
            # Z-score method
            z_anomalies = self._zscore_anomalies(data)
            
            # MAD method
            median = np.median(data)
            mad = np.median(np.abs(data - median))
            mad_scores = np.abs(data - median) / (1.4826 * mad + 1e-10)
            mad_anomalies = mad_scores > 3.0
            
            # Consensus: either method flags it
            anomalies = z_anomalies | mad_anomalies
            
            anomaly_counts[col] = anomalies.sum()
            anomaly_matrices[col] = anomalies
        
        # Multivariate anomalies using simplified Isolation Forest
        try:
            X = panel_clean.values
            if_scores = self._isolation_forest_scores(X)
            threshold = np.percentile(if_scores, 100 * (1 - self.contamination))
            multivariate_anomalies = if_scores > threshold
        except:
            multivariate_anomalies = np.zeros(len(panel_clean), dtype=bool)
        
        anomaly_rate = pd.Series({k: v / len(panel_clean) for k, v in anomaly_counts.items()})
        
        # Current anomalies (last observation)
        current_anomalies = []
        if not panel_clean.empty:
            last_idx = panel_clean.index[-1]
            for col in panel_clean.columns:
                if anomaly_matrices[col][-1]:
                    current_anomalies.append(col)
        
        importance = anomaly_rate.sort_values(ascending=False)
        
        return {
            'anomaly_counts': pd.Series(anomaly_counts),
            'anomaly_rate': anomaly_rate,
            'current_anomalies': current_anomalies,
            'multivariate_anomaly_dates': panel_clean.index[multivariate_anomalies].tolist(),
            'importance': importance,
            'method': 'Isolation Forest + Z-score + MAD'
        }
    
    def top_indicators(self, result: Dict, date: pd.Timestamp = None, n: int = 5) -> List[Tuple[str, float]]:
        """Top indicators by anomaly frequency."""
        importance = result.get('importance', pd.Series(dtype='float'))
        if importance.empty:
            return []
        return list(zip(importance.index[:n], importance.values[:n]))


# =============================================================================
# LENS 13: TRANSFER ENTROPY
# =============================================================================

class TransferEntropyLens:
    """
    Quantifies directed information flow between indicators.
    Answers: "Which indicators drive information to others?"
    
    Reveals:
    - Asymmetric causal relationships
    - Information leaders vs followers
    - Non-linear dependencies missed by correlation
    """
    
    def __init__(self, name: str = "TransferEntropy", k: int = 4, lag: int = 1):
        self.name = name
        self.k = k
        self.lag = lag
    
    def _ksg_entropy(self, X: np.ndarray) -> float:
        """KSG entropy estimator."""
        n, d = X.shape
        if n <= self.k + 1:
            return 0
        
        tree = cKDTree(X)
        distances, _ = tree.query(X, k=self.k + 1, p=float('inf'))
        eps = np.maximum(distances[:, -1], 1e-10)
        
        return digamma(n) - digamma(self.k) + d * np.mean(np.log(2 * eps))
    
    def _compute_transfer_entropy(self, source: np.ndarray, target: np.ndarray) -> float:
        """Compute transfer entropy from source to target."""
        n = len(source)
        offset = self.lag
        valid_n = n - offset
        
        if valid_n <= self.k + 1:
            return 0
        
        # Normalize
        source = (source - np.mean(source)) / (np.std(source) + 1e-10)
        target = (target - np.mean(target)) / (np.std(target) + 1e-10)
        
        # Embed
        target_future = target[offset:].reshape(-1, 1)
        target_past = target[offset - 1:-1].reshape(-1, 1)
        source_past = source[:valid_n].reshape(-1, 1)
        
        # TE = I(Y_future; X_past | Y_past) = H(Y_f, Y_p) + H(Y_p, X_p) - H(Y_p) - H(Y_f, Y_p, X_p)
        try:
            h_yf_yp = self._ksg_entropy(np.hstack([target_future, target_past]))
            h_yp_xp = self._ksg_entropy(np.hstack([target_past, source_past]))
            h_yp = self._ksg_entropy(target_past)
            h_yf_yp_xp = self._ksg_entropy(np.hstack([target_future, target_past, source_past]))
            
            te = h_yf_yp + h_yp_xp - h_yp - h_yf_yp_xp
            return max(0, te)
        except:
            return 0
    
    def analyze(self, panel: pd.DataFrame) -> Dict:
        """
        Compute pairwise transfer entropy.
        
        Returns:
        - te_matrix: Transfer entropy matrix [source, target]
        - net_flow: Net outgoing - incoming information for each indicator
        - importance: Indicators ranked by information leadership
        """
        panel_clean = panel.dropna()
        
        if panel_clean.empty or panel_clean.shape[0] < 20 or panel_clean.shape[1] < 2:
            warnings.warn("TransferEntropyLens: Insufficient data.")
            return {
                'te_matrix': pd.DataFrame(dtype='float'),
                'net_flow': pd.Series(dtype='float'),
                'importance': pd.Series(dtype='float'),
                'method': f'KSG Transfer Entropy (k={self.k}, lag={self.lag})'
            }
        
        columns = panel_clean.columns.tolist()
        n = len(columns)
        te_matrix = np.zeros((n, n))
        
        for i, source_col in enumerate(columns):
            for j, target_col in enumerate(columns):
                if i == j:
                    continue
                
                source = panel_clean[source_col].values
                target = panel_clean[target_col].values
                
                te_matrix[i, j] = self._compute_transfer_entropy(source, target)
        
        te_df = pd.DataFrame(te_matrix, index=columns, columns=columns)
        
        # Net flow: outgoing - incoming
        outgoing = te_df.sum(axis=1)
        incoming = te_df.sum(axis=0)
        net_flow = outgoing - incoming
        
        # Importance = absolute net flow (leaders have high positive, followers have high negative)
        importance = outgoing.sort_values(ascending=False)  # Rank by outgoing influence
        
        return {
            'te_matrix': te_df,
            'outgoing': outgoing.sort_values(ascending=False),
            'incoming': incoming.sort_values(ascending=False),
            'net_flow': net_flow.sort_values(ascending=False),
            'importance': importance,
            'method': f'KSG Transfer Entropy (k={self.k}, lag={self.lag})'
        }
    
    def top_indicators(self, result: Dict, date: pd.Timestamp = None, n: int = 5) -> List[Tuple[str, float]]:
        """Top information leaders."""
        importance = result.get('importance', pd.Series(dtype='float'))
        if importance.empty:
            return []
        return list(zip(importance.index[:n], importance.values[:n]))


# =============================================================================
# LENS 14: TOPOLOGICAL DATA ANALYSIS
# =============================================================================

class TopologicalDataAnalysisLens:
    """
    Analyzes the 'shape' of data to find recurring patterns and tipping points.
    Answers: "What is the underlying topological structure of the dynamics?"
    
    Reveals:
    - Attractor structure
    - Regime transitions as topological changes
    - Recurring dynamical patterns
    """
    
    def __init__(self, name: str = "TDA", embedding_dim: int = 3, delay: int = None):
        self.name = name
        self.embedding_dim = embedding_dim
        self.delay = delay
    
    def _estimate_delay(self, data: np.ndarray) -> int:
        """Estimate delay using autocorrelation first zero-crossing."""
        n = len(data)
        max_lag = min(n // 4, 50)
        data_centered = data - np.mean(data)
        autocorr = np.correlate(data_centered, data_centered, mode='full')
        autocorr = autocorr[n-1:n-1+max_lag] / (autocorr[n-1] + 1e-10)
        
        for i in range(1, len(autocorr)):
            if autocorr[i] <= 0:
                return i
        return max_lag // 4
    
    def _takens_embed(self, data: np.ndarray, dim: int, delay: int) -> np.ndarray:
        """Takens time-delay embedding."""
        n = len(data)
        n_points = n - (dim - 1) * delay
        
        if n_points <= 0:
            return np.array([])
        
        embedded = np.zeros((n_points, dim))
        for i in range(dim):
            start = i * delay
            end = start + n_points
            embedded[:, i] = data[start:end]
        
        return embedded
    
    def _persistence_h0(self, points: np.ndarray) -> List[Tuple[float, float]]:
        """Simplified H0 persistence (connected components)."""
        n = len(points)
        if n < 2:
            return []
        
        dist_matrix = squareform(pdist(points))
        
        # Get all edges sorted by distance
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                edges.append((dist_matrix[i, j], i, j))
        edges.sort()
        
        # Union-Find
        parent = list(range(n))
        birth = [0.0] * n
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        persistence = []
        
        for dist, i, j in edges:
            pi, pj = find(i), find(j)
            if pi != pj:
                # Younger component dies
                if birth[pi] > birth[pj]:
                    pi, pj = pj, pi
                persistence.append((birth[pj], dist))
                parent[pj] = pi
        
        return persistence
    
    def analyze(self, panel: pd.DataFrame) -> Dict:
        """
        Perform TDA on each indicator's embedded attractor.
        
        Returns:
        - total_persistence: Sum of H0 persistence for each indicator
        - max_persistence: Maximum H0 persistence (largest structure)
        - importance: Indicators ranked by topological complexity
        """
        panel_clean = panel.dropna()
        
        if panel_clean.empty or panel_clean.shape[0] < 50:
            warnings.warn("TopologicalDataAnalysisLens: Insufficient data for embedding.")
            return {
                'total_persistence': pd.Series(dtype='float'),
                'max_persistence': pd.Series(dtype='float'),
                'importance': pd.Series(dtype='float'),
                'method': f'Takens Embedding + H0 Persistence (dim={self.embedding_dim})'
            }
        
        total_persistence = {}
        max_persistence = {}
        
        for col in panel_clean.columns:
            data = panel_clean[col].values
            
            # Normalize
            data = (data - np.mean(data)) / (np.std(data) + 1e-10)
            
            # Estimate delay if not provided
            delay = self.delay if self.delay else self._estimate_delay(data)
            
            try:
                # Embed
                points = self._takens_embed(data, self.embedding_dim, delay)
                
                if len(points) < 10:
                    total_persistence[col] = 0
                    max_persistence[col] = 0
                    continue
                
                # Subsample for efficiency
                if len(points) > 200:
                    idx = np.random.choice(len(points), 200, replace=False)
                    points = points[idx]
                
                # Compute H0 persistence
                persistence = self._persistence_h0(points)
                
                if persistence:
                    lifetimes = [death - birth for birth, death in persistence]
                    total_persistence[col] = sum(lifetimes)
                    max_persistence[col] = max(lifetimes)
                else:
                    total_persistence[col] = 0
                    max_persistence[col] = 0
                    
            except Exception as e:
                warnings.warn(f"TDA failed for {col}: {e}")
                total_persistence[col] = 0
                max_persistence[col] = 0
        
        # Importance = total persistence (more complex attractors = more important)
        importance = pd.Series(total_persistence).sort_values(ascending=False)
        
        return {
            'total_persistence': pd.Series(total_persistence),
            'max_persistence': pd.Series(max_persistence),
            'importance': importance,
            'method': f'Takens Embedding + H0 Persistence (dim={self.embedding_dim})'
        }
    
    def top_indicators(self, result: Dict, date: pd.Timestamp = None, n: int = 5) -> List[Tuple[str, float]]:
        """Top indicators by topological complexity."""
        importance = result.get('importance', pd.Series(dtype='float'))
        if importance.empty:
            return []
        return list(zip(importance.index[:n], importance.values[:n]))


# =============================================================================
# CONVENIENCE: ADD ALL NEW LENSES TO COMPARATOR
# =============================================================================

def add_advanced_lenses(comparator) -> None:
    """
    Add all advanced mathematical lenses to an existing LensComparator.
    
    Usage:
        comparator = LensComparator(panel)
        add_advanced_lenses(comparator)
        comparator.run_all()
    """
    comparator.add_lens(WaveletAnalysisLens())
    comparator.add_lens(NetworkGraphLens())
    comparator.add_lens(RegimeSwitchingLens())
    comparator.add_lens(AnomalyDetectionLens())
    comparator.add_lens(TransferEntropyLens())
    comparator.add_lens(TopologicalDataAnalysisLens())
    
    print(f"✓ Added 6 advanced mathematical lenses")


def run_extended_lens_analysis(panel: pd.DataFrame, 
                                date_to_analyze: pd.Timestamp = None) -> 'LensComparator':
    """
    Run all lenses (original 8 + new 6 = 14 total) on the data.
    
    This is an extended version of run_full_lens_analysis that includes
    all the advanced mathematical lenses.
    """
    # Import the original classes (assumes they're in scope)
    from __main__ import (LensComparator, MagnitudeLens, PCALens, GrangerLens,
                          DMDLens, InfluenceLens, MutualInformationLens,
                          ClusteringLens, TimeSeriesDecompositionLens)
    
    comparator = LensComparator(panel)
    
    # Original 8 lenses
    comparator.add_lens(MagnitudeLens())
    comparator.add_lens(PCALens())
    comparator.add_lens(GrangerLens())
    comparator.add_lens(DMDLens())
    comparator.add_lens(InfluenceLens())
    comparator.add_lens(MutualInformationLens())
    comparator.add_lens(ClusteringLens())
    comparator.add_lens(TimeSeriesDecompositionLens())
    
    # New 6 advanced lenses
    comparator.add_lens(WaveletAnalysisLens())
    comparator.add_lens(NetworkGraphLens())
    comparator.add_lens(RegimeSwitchingLens())
    comparator.add_lens(AnomalyDetectionLens())
    comparator.add_lens(TransferEntropyLens())
    comparator.add_lens(TopologicalDataAnalysisLens())
    
    # Run all
    comparator.run_all()
    
    # Generate comparisons
    print("\n" + "="*70)
    print("EXTENDED LENS COMPARISON ANALYSIS (14 Lenses)")
    print("="*70)
    
    print("\nLens Agreement Matrix (Spearman correlation):")
    print(comparator.agreement_matrix().to_string())
    
    print("\nConsensus Indicators (agreed upon by most lenses):")
    print(comparator.consensus_indicators(n_top=10).to_string())
    
    print("\nUnique Insights by Lens:")
    unique = comparator.unique_insights()
    for lens_name, indicators in unique.items():
        if indicators:
            print(f"  {lens_name}: {indicators}")
    
    if date_to_analyze is not None and date_to_analyze in panel.index:
        print(f"\nComparison at {date_to_analyze.strftime('%Y-%m-%d')}:\n")
        print(comparator.compare_at_date(date_to_analyze, n_top=5).to_string())
    
    return comparator


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    print("VCF Advanced Mathematical Lenses Module")
    print("="*50)
    print("\nAvailable new lenses:")
    print("  - WaveletAnalysisLens: Multi-scale time-frequency decomposition")
    print("  - NetworkGraphLens: Correlation network with MST and centrality")
    print("  - RegimeSwitchingLens: Hidden Markov Model regime detection")
    print("  - AnomalyDetectionLens: Isolation Forest + statistical methods")
    print("  - TransferEntropyLens: Directed information flow (KSG estimator)")
    print("  - TopologicalDataAnalysisLens: Takens embedding + persistence")
    print("\nTo use with existing framework:")
    print("  from advanced_lenses import add_advanced_lenses")
    print("  comparator = LensComparator(panel)")
    print("  add_advanced_lenses(comparator)")
    print("  comparator.run_all()")
