"""
VAR / Granger Causality Lens
============================
Analyzes multivariate causal relationships and dynamic feedback loops
within interconnected systems using linear models.

Key Applications:
- Economic variable interdependencies
- Climate teleconnections (ENSO, NAO, etc.)
- Market sector lead-lag relationships

Methods:
- Vector Autoregression (VAR): Multivariate AR model
- Granger Causality: Tests if X helps predict Y
- Impulse Response Functions: Shock propagation analysis
- Variance Decomposition: Contribution to forecast error
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from scipy import stats
from scipy.linalg import solve, inv, cholesky
import warnings


@dataclass
class GrangerResult:
    """Container for Granger causality test results."""
    cause: str
    effect: str
    f_statistic: float
    p_value: float
    is_significant: bool
    optimal_lag: int


@dataclass
class VARResult:
    """Container for VAR model results."""
    coefficients: Dict[int, np.ndarray]  # Lag -> coefficient matrix
    intercept: np.ndarray
    residuals: np.ndarray
    covariance: np.ndarray
    aic: float
    bic: float
    hqic: float
    log_likelihood: float


class VectorAutoregression:
    """
    Vector Autoregression (VAR) model.
    
    Models the joint dynamics of multiple time series:
    Y_t = A_1 * Y_{t-1} + A_2 * Y_{t-2} + ... + A_p * Y_{t-p} + c + ε_t
    
    where Y_t is a k-dimensional vector and A_i are k×k coefficient matrices.
    """
    
    def __init__(self, lag_order: int = 1):
        """
        Initialize VAR model.
        
        Parameters
        ----------
        lag_order : int
            Number of lags (p)
        """
        self.lag_order = lag_order
        self._fitted = False
    
    def _build_design_matrix(
        self,
        data: np.ndarray,
        include_intercept: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Build the design matrix for VAR estimation."""
        T, k = data.shape
        p = self.lag_order
        
        # Effective sample size
        n = T - p
        
        # Dependent variable matrix (n × k)
        Y = data[p:, :]
        
        # Design matrix with lagged values
        # Each row: [1, Y_{t-1}, Y_{t-2}, ..., Y_{t-p}]
        n_regressors = p * k + (1 if include_intercept else 0)
        X = np.zeros((n, n_regressors))
        
        col = 0
        if include_intercept:
            X[:, 0] = 1
            col = 1
        
        for lag in range(1, p + 1):
            X[:, col:col + k] = data[p - lag:T - lag, :]
            col += k
        
        return X, Y
    
    def fit(
        self,
        data: Union[np.ndarray, pd.DataFrame],
        include_intercept: bool = True
    ) -> 'VectorAutoregression':
        """
        Fit the VAR model using OLS.
        
        Parameters
        ----------
        data : array-like
            T × k array of time series data
        include_intercept : bool
            Include constant term
            
        Returns
        -------
        self
        """
        if isinstance(data, pd.DataFrame):
            self.names = data.columns.tolist()
            data = data.values
        else:
            self.names = [f"Y{i}" for i in range(data.shape[1])]
        
        T, k = data.shape
        p = self.lag_order
        n = T - p
        
        self.k = k
        self.n_obs = n
        self.data = data
        
        # Build design matrix
        X, Y = self._build_design_matrix(data, include_intercept)
        
        # OLS estimation: B = (X'X)^{-1} X'Y
        XtX_inv = np.linalg.inv(X.T @ X)
        B = XtX_inv @ X.T @ Y
        
        # Extract coefficients
        self.coefficients = {}
        col = 1 if include_intercept else 0
        
        if include_intercept:
            self.intercept = B[0, :]
        else:
            self.intercept = np.zeros(k)
        
        for lag in range(1, p + 1):
            self.coefficients[lag] = B[col:col + k, :].T  # k × k matrix
            col += k
        
        # Residuals and covariance
        self.fitted = X @ B
        self.residuals = Y - self.fitted
        
        # ML estimate of covariance
        self.sigma = self.residuals.T @ self.residuals / n
        
        # Log-likelihood
        sign, logdet = np.linalg.slogdet(self.sigma)
        log_likelihood = -n * k / 2 * np.log(2 * np.pi) - n / 2 * logdet - n * k / 2
        
        # Information criteria
        n_params = k * (1 + p * k) if include_intercept else k * p * k
        self.aic = -2 * log_likelihood + 2 * n_params
        self.bic = -2 * log_likelihood + n_params * np.log(n)
        self.hqic = -2 * log_likelihood + 2 * n_params * np.log(np.log(n))
        self.log_likelihood = log_likelihood
        
        self._fitted = True
        return self
    
    def predict(self, steps: int = 1, initial: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Forecast future values.
        
        Parameters
        ----------
        steps : int
            Number of steps to forecast
        initial : np.ndarray, optional
            Initial values. If None, uses last observations.
            
        Returns
        -------
        np.ndarray
            Forecasted values (steps × k)
        """
        if not self._fitted:
            raise ValueError("Model must be fitted first")
        
        if initial is None:
            initial = self.data[-self.lag_order:, :]
        
        forecasts = np.zeros((steps, self.k))
        history = initial.copy()
        
        for t in range(steps):
            # Y_t = c + A_1 * Y_{t-1} + A_2 * Y_{t-2} + ...
            y_new = self.intercept.copy()
            for lag in range(1, self.lag_order + 1):
                y_new += self.coefficients[lag] @ history[-lag, :]
            
            forecasts[t, :] = y_new
            history = np.vstack([history[1:], y_new])
        
        return forecasts
    
    def impulse_response(
        self,
        periods: int = 20,
        shock_var: Optional[int] = None,
        orthogonalized: bool = True
    ) -> np.ndarray:
        """
        Compute impulse response functions.
        
        Shows how a shock to one variable propagates through the system.
        
        Parameters
        ----------
        periods : int
            Number of periods to compute
        shock_var : int, optional
            Index of variable to shock. If None, returns all.
        orthogonalized : bool
            Use Cholesky decomposition for orthogonalized shocks
            
        Returns
        -------
        np.ndarray
            IRF array: (periods, k, k) if shock_var is None else (periods, k)
        """
        if not self._fitted:
            raise ValueError("Model must be fitted first")
        
        k = self.k
        p = self.lag_order
        
        # Build companion matrix
        companion = np.zeros((k * p, k * p))
        for lag in range(1, p + 1):
            companion[:k, (lag - 1) * k:lag * k] = self.coefficients[lag]
        if p > 1:
            companion[k:, :k * (p - 1)] = np.eye(k * (p - 1))
        
        # Shock matrix
        if orthogonalized:
            try:
                P = cholesky(self.sigma, lower=True)
            except:
                P = np.eye(k)
        else:
            P = np.eye(k)
        
        # Compute IRFs
        irf = np.zeros((periods, k, k))
        phi = np.eye(k * p)
        
        for t in range(periods):
            irf[t, :, :] = phi[:k, :k] @ P
            phi = companion @ phi
        
        if shock_var is not None:
            return irf[:, :, shock_var]
        return irf
    
    def variance_decomposition(
        self,
        periods: int = 20,
        orthogonalized: bool = True
    ) -> np.ndarray:
        """
        Compute forecast error variance decomposition.
        
        Shows what fraction of forecast error variance in each variable
        is attributable to shocks in each variable.
        
        Parameters
        ----------
        periods : int
            Number of periods
        orthogonalized : bool
            Use orthogonalized shocks
            
        Returns
        -------
        np.ndarray
            FEVD array: (periods, k, k)
            fevd[t, i, j] = contribution of shock j to variance of variable i at horizon t
        """
        irf = self.impulse_response(periods, orthogonalized=orthogonalized)
        k = self.k
        
        # Cumulative squared IRFs
        mse = np.zeros((periods, k, k))
        for t in range(periods):
            for j in range(k):
                mse[t, :, j] = np.sum(irf[:t + 1, :, j] ** 2, axis=0)
        
        # Normalize to proportions
        total_mse = mse.sum(axis=2, keepdims=True)
        fevd = mse / (total_mse + 1e-10)
        
        return fevd
    
    def get_summary(self) -> Dict:
        """Get model summary."""
        if not self._fitted:
            raise ValueError("Model must be fitted first")
        
        return {
            'lag_order': self.lag_order,
            'n_variables': self.k,
            'n_observations': self.n_obs,
            'variable_names': self.names,
            'intercept': self.intercept,
            'coefficients': self.coefficients,
            'residual_covariance': self.sigma,
            'aic': self.aic,
            'bic': self.bic,
            'hqic': self.hqic,
            'log_likelihood': self.log_likelihood
        }


class GrangerCausality:
    """
    Granger Causality tests.
    
    Tests whether past values of X help predict Y beyond Y's own past.
    "X Granger-causes Y" if knowing X's past improves Y's forecast.
    
    Note: Granger causality is predictive, not true causality.
    """
    
    def __init__(self, max_lag: int = 10):
        """
        Initialize Granger Causality tester.
        
        Parameters
        ----------
        max_lag : int
            Maximum lag to test
        """
        self.max_lag = max_lag
    
    def _fit_ar(
        self,
        y: np.ndarray,
        lags: int
    ) -> Tuple[np.ndarray, float]:
        """Fit AR model and return residuals and RSS."""
        n = len(y)
        valid_n = n - lags
        
        # Build design matrix
        X = np.ones((valid_n, lags + 1))
        for i in range(lags):
            X[:, i + 1] = y[lags - 1 - i:n - 1 - i]
        
        Y = y[lags:]
        
        # OLS
        beta = np.linalg.lstsq(X, Y, rcond=None)[0]
        residuals = Y - X @ beta
        rss = np.sum(residuals ** 2)
        
        return residuals, rss
    
    def _fit_ardl(
        self,
        y: np.ndarray,
        x: np.ndarray,
        lags: int
    ) -> Tuple[np.ndarray, float]:
        """Fit ARDL model (AR with distributed lags of X)."""
        n = len(y)
        valid_n = n - lags
        
        # Build design matrix: [1, y_lags, x_lags]
        X = np.ones((valid_n, 1 + 2 * lags))
        
        col = 1
        for i in range(lags):
            X[:, col] = y[lags - 1 - i:n - 1 - i]
            col += 1
        for i in range(lags):
            X[:, col] = x[lags - 1 - i:n - 1 - i]
            col += 1
        
        Y = y[lags:]
        
        # OLS
        beta = np.linalg.lstsq(X, Y, rcond=None)[0]
        residuals = Y - X @ beta
        rss = np.sum(residuals ** 2)
        
        return residuals, rss
    
    def test(
        self,
        x: np.ndarray,
        y: np.ndarray,
        lag: int,
        alpha: float = 0.05
    ) -> GrangerResult:
        """
        Test if X Granger-causes Y at given lag.
        
        Uses F-test comparing:
        - Restricted model: Y_t = f(Y_past)
        - Unrestricted model: Y_t = f(Y_past, X_past)
        
        Parameters
        ----------
        x : np.ndarray
            Potential cause
        y : np.ndarray
            Potential effect
        lag : int
            Number of lags
        alpha : float
            Significance level
            
        Returns
        -------
        GrangerResult
        """
        x = np.asarray(x).flatten()
        y = np.asarray(y).flatten()
        
        if len(x) != len(y):
            raise ValueError("x and y must have same length")
        
        n = len(y) - lag
        
        # Restricted model (AR)
        _, rss_r = self._fit_ar(y, lag)
        
        # Unrestricted model (ARDL)
        _, rss_u = self._fit_ardl(y, x, lag)
        
        # F-test
        # F = ((RSS_r - RSS_u) / lag) / (RSS_u / (n - 2*lag - 1))
        df1 = lag
        df2 = n - 2 * lag - 1
        
        if df2 <= 0:
            raise ValueError("Not enough observations for this lag")
        
        f_stat = ((rss_r - rss_u) / df1) / (rss_u / df2)
        p_value = 1 - stats.f.cdf(f_stat, df1, df2)
        
        return GrangerResult(
            cause='X',
            effect='Y',
            f_statistic=f_stat,
            p_value=p_value,
            is_significant=p_value < alpha,
            optimal_lag=lag
        )
    
    def find_optimal_lag(
        self,
        x: np.ndarray,
        y: np.ndarray,
        criterion: str = 'bic'
    ) -> int:
        """
        Find optimal lag using information criteria.
        
        Parameters
        ----------
        x : np.ndarray
            Potential cause
        y : np.ndarray
            Potential effect
        criterion : str
            'aic', 'bic', or 'hqic'
            
        Returns
        -------
        int
            Optimal lag order
        """
        best_lag = 1
        best_ic = np.inf
        
        for lag in range(1, self.max_lag + 1):
            try:
                _, rss = self._fit_ardl(y, x, lag)
                n = len(y) - lag
                k = 2 * lag + 1
                
                # Log-likelihood (assuming Gaussian errors)
                ll = -n / 2 * (1 + np.log(2 * np.pi) + np.log(rss / n))
                
                if criterion == 'aic':
                    ic = -2 * ll + 2 * k
                elif criterion == 'bic':
                    ic = -2 * ll + k * np.log(n)
                else:  # hqic
                    ic = -2 * ll + 2 * k * np.log(np.log(n))
                
                if ic < best_ic:
                    best_ic = ic
                    best_lag = lag
            except:
                continue
        
        return best_lag
    
    def test_all_lags(
        self,
        x: np.ndarray,
        y: np.ndarray,
        alpha: float = 0.05
    ) -> Dict:
        """
        Test Granger causality at all lags up to max_lag.
        
        Parameters
        ----------
        x : np.ndarray
            Potential cause
        y : np.ndarray
            Potential effect
        alpha : float
            Significance level
            
        Returns
        -------
        dict
            Results at each lag
        """
        results = {}
        
        for lag in range(1, self.max_lag + 1):
            try:
                result = self.test(x, y, lag, alpha)
                results[lag] = {
                    'f_statistic': result.f_statistic,
                    'p_value': result.p_value,
                    'significant': result.is_significant
                }
            except:
                results[lag] = {'error': 'Insufficient data'}
        
        return results


class VARGrangerLens:
    """
    VAR / Granger Causality Lens for comprehensive causal analysis.
    
    Combines VAR modeling with Granger causality tests to understand
    dynamic relationships in multivariate systems.
    """
    
    def __init__(self, max_lag: int = 10):
        """
        Initialize the VAR/Granger Lens.
        
        Parameters
        ----------
        max_lag : int
            Maximum lag to consider
        """
        self.max_lag = max_lag
        self.var_model = None
        self.granger = GrangerCausality(max_lag)
    
    def select_lag_order(
        self,
        data: pd.DataFrame,
        criterion: str = 'bic'
    ) -> int:
        """
        Select optimal lag order using information criteria.
        
        Parameters
        ----------
        data : pd.DataFrame
            Multivariate time series
        criterion : str
            'aic', 'bic', or 'hqic'
            
        Returns
        -------
        int
            Optimal lag order
        """
        best_lag = 1
        best_ic = np.inf
        
        for lag in range(1, self.max_lag + 1):
            try:
                var = VectorAutoregression(lag_order=lag)
                var.fit(data)
                
                if criterion == 'aic':
                    ic = var.aic
                elif criterion == 'bic':
                    ic = var.bic
                else:
                    ic = var.hqic
                
                if ic < best_ic:
                    best_ic = ic
                    best_lag = lag
            except:
                continue
        
        return best_lag
    
    def fit_var(
        self,
        data: pd.DataFrame,
        lag_order: Optional[int] = None,
        auto_select: bool = True
    ) -> Dict:
        """
        Fit VAR model.
        
        Parameters
        ----------
        data : pd.DataFrame
            Multivariate time series
        lag_order : int, optional
            Lag order. If None and auto_select, selects automatically.
        auto_select : bool
            Automatically select lag order
            
        Returns
        -------
        dict
            VAR model summary
        """
        if lag_order is None and auto_select:
            lag_order = self.select_lag_order(data)
        elif lag_order is None:
            lag_order = 1
        
        self.var_model = VectorAutoregression(lag_order=lag_order)
        self.var_model.fit(data)
        
        return self.var_model.get_summary()
    
    def pairwise_granger_causality(
        self,
        data: pd.DataFrame,
        lag: Optional[int] = None,
        alpha: float = 0.05
    ) -> pd.DataFrame:
        """
        Test Granger causality for all variable pairs.
        
        Parameters
        ----------
        data : pd.DataFrame
            Multivariate time series
        lag : int, optional
            Lag to use. If None, uses optimal for each pair.
        alpha : float
            Significance level
            
        Returns
        -------
        pd.DataFrame
            Matrix of p-values (rows=cause, cols=effect)
        """
        columns = data.columns.tolist()
        n = len(columns)
        
        p_matrix = np.ones((n, n))
        f_matrix = np.zeros((n, n))
        
        for i, cause in enumerate(columns):
            for j, effect in enumerate(columns):
                if i == j:
                    continue
                
                x = data[cause].values
                y = data[effect].values
                
                if lag is None:
                    opt_lag = self.granger.find_optimal_lag(x, y)
                else:
                    opt_lag = lag
                
                try:
                    result = self.granger.test(x, y, opt_lag, alpha)
                    p_matrix[i, j] = result.p_value
                    f_matrix[i, j] = result.f_statistic
                except:
                    p_matrix[i, j] = 1.0
                    f_matrix[i, j] = 0.0
        
        return {
            'p_values': pd.DataFrame(p_matrix, index=columns, columns=columns),
            'f_statistics': pd.DataFrame(f_matrix, index=columns, columns=columns)
        }
    
    def compute_irf(
        self,
        periods: int = 20,
        orthogonalized: bool = True
    ) -> Dict:
        """
        Compute impulse response functions.
        
        Parameters
        ----------
        periods : int
            Forecast horizon
        orthogonalized : bool
            Use orthogonalized shocks
            
        Returns
        -------
        dict
            IRF results
        """
        if self.var_model is None:
            raise ValueError("VAR model must be fitted first")
        
        irf = self.var_model.impulse_response(periods, orthogonalized=orthogonalized)
        
        return {
            'irf': irf,
            'periods': np.arange(periods),
            'variables': self.var_model.names
        }
    
    def compute_fevd(
        self,
        periods: int = 20
    ) -> Dict:
        """
        Compute forecast error variance decomposition.
        
        Parameters
        ----------
        periods : int
            Forecast horizon
            
        Returns
        -------
        dict
            FEVD results
        """
        if self.var_model is None:
            raise ValueError("VAR model must be fitted first")
        
        fevd = self.var_model.variance_decomposition(periods)
        
        return {
            'fevd': fevd,
            'periods': np.arange(periods),
            'variables': self.var_model.names
        }
    
    def identify_causal_structure(
        self,
        data: pd.DataFrame,
        alpha: float = 0.05
    ) -> Dict:
        """
        Identify causal structure from Granger tests.
        
        Parameters
        ----------
        data : pd.DataFrame
            Multivariate time series
        alpha : float
            Significance level
            
        Returns
        -------
        dict
            Causal graph structure
        """
        results = self.pairwise_granger_causality(data, alpha=alpha)
        p_values = results['p_values']
        
        columns = data.columns.tolist()
        edges = []
        
        for cause in columns:
            for effect in columns:
                if cause != effect and p_values.loc[cause, effect] < alpha:
                    edges.append({
                        'from': cause,
                        'to': effect,
                        'p_value': p_values.loc[cause, effect]
                    })
        
        # Compute node statistics
        node_stats = {}
        for col in columns:
            # Count outgoing (this variable causes others)
            outgoing = sum(1 for e in edges if e['from'] == col)
            # Count incoming (others cause this variable)
            incoming = sum(1 for e in edges if e['to'] == col)
            
            node_stats[col] = {
                'outgoing_edges': outgoing,
                'incoming_edges': incoming,
                'net_causality': outgoing - incoming
            }
        
        return {
            'edges': edges,
            'node_stats': node_stats,
            'p_value_matrix': p_values
        }
    
    def analyze(
        self,
        data: pd.DataFrame,
        lag_order: Optional[int] = None,
        irf_periods: int = 20,
        alpha: float = 0.05
    ) -> Dict:
        """
        Comprehensive VAR/Granger analysis.
        
        Parameters
        ----------
        data : pd.DataFrame
            Multivariate time series
        lag_order : int, optional
            Lag order (auto-selected if None)
        irf_periods : int
            IRF forecast horizon
        alpha : float
            Significance level for Granger tests
            
        Returns
        -------
        dict
            Complete analysis results
        """
        results = {}
        
        # Fit VAR
        results['var'] = self.fit_var(data, lag_order)
        results['optimal_lag'] = results['var']['lag_order']
        
        # Granger causality
        granger_results = self.pairwise_granger_causality(
            data, lag=results['optimal_lag'], alpha=alpha
        )
        results['granger'] = granger_results
        
        # Significant causal relationships
        causal = self.identify_causal_structure(data, alpha)
        results['causal_structure'] = causal
        
        # IRF
        results['irf'] = self.compute_irf(irf_periods)
        
        # FEVD
        results['fevd'] = self.compute_fevd(irf_periods)
        
        # Summary of causal relationships
        edges = causal['edges']
        results['summary'] = {
            'n_variables': len(data.columns),
            'n_observations': len(data),
            'lag_order': results['optimal_lag'],
            'n_significant_relationships': len(edges),
            'significant_edges': [(e['from'], e['to']) for e in edges]
        }
        
        return results


# Convenience function
def var_granger_analysis(
    data: pd.DataFrame,
    max_lag: int = 10,
    alpha: float = 0.05
) -> Dict:
    """
    Quick VAR and Granger causality analysis.
    
    Parameters
    ----------
    data : pd.DataFrame
        Multivariate time series
    max_lag : int
        Maximum lag to consider
    alpha : float
        Significance level
        
    Returns
    -------
    dict
        Analysis results
    """
    lens = VARGrangerLens(max_lag=max_lag)
    return lens.analyze(data, alpha=alpha)


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    
    # Generate VAR(1) process with known structure
    n = 500
    k = 3
    
    # Coefficient matrix (X1 -> X2 -> X3)
    A = np.array([
        [0.7, 0.0, 0.0],   # X1 only depends on itself
        [0.3, 0.5, 0.0],   # X2 depends on X1 and itself
        [0.0, 0.4, 0.6]    # X3 depends on X2 and itself
    ])
    
    # Generate data
    data = np.zeros((n, k))
    for t in range(1, n):
        data[t, :] = A @ data[t-1, :] + 0.3 * np.random.randn(k)
    
    df = pd.DataFrame(data, columns=['X1', 'X2', 'X3'])
    
    # Analyze
    lens = VARGrangerLens(max_lag=5)
    results = lens.analyze(df, alpha=0.05)
    
    print("VAR / Granger Causality Analysis Complete")
    print(f"\nOptimal lag order: {results['optimal_lag']}")
    print(f"Number of significant causal relationships: {results['summary']['n_significant_relationships']}")
    
    print(f"\nGranger Causality P-values:")
    print(results['granger']['p_values'].round(4))
    
    print(f"\nSignificant Causal Edges:")
    for edge in results['causal_structure']['edges']:
        print(f"  {edge['from']} → {edge['to']} (p={edge['p_value']:.4f})")
    
    print(f"\nNode Causality Statistics:")
    for node, stats in results['causal_structure']['node_stats'].items():
        print(f"  {node}: out={stats['outgoing_edges']}, in={stats['incoming_edges']}, "
              f"net={stats['net_causality']}")
    
    print(f"\nVAR Coefficients (lag 1):")
    print(np.array(results['var']['coefficients'][1]).round(3))
