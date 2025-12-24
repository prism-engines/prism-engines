"""
PRISM Cointegration Engine

Tests for long-run equilibrium relationships between series.

Measures:
- Engle-Granger test statistics (pairwise)
- Cointegrating vectors (hedge ratios)
- Error correction speeds

Phase: Unbound
Normalization: None (uses levels, not returns)
"""

import logging
from typing import Dict, Any, Optional, Tuple
from datetime import date

import numpy as np
import pandas as pd
from scipy import stats

from .base import BaseEngine


logger = logging.getLogger(__name__)


def _adf_test(x: np.ndarray, max_lag: int = None) -> Tuple[float, float, int]:
    """
    Augmented Dickey-Fuller test for stationarity.

    Returns (adf_stat, p_value, lags_used)
    """
    n = len(x)

    if max_lag is None:
        max_lag = int(np.floor(4 * (n / 100) ** 0.25))

    max_lag = min(max_lag, n // 4)

    # First difference
    dx = np.diff(x)

    # Lagged level
    x_lag = x[:-1]

    # Build regression matrix with lagged differences
    best_aic = np.inf
    best_result = None

    for lag in range(max_lag + 1):
        if lag == 0:
            X = np.column_stack([np.ones(len(dx)), x_lag])
            y = dx
        else:
            # Include lagged differences
            n_obs = len(dx) - lag
            dx_lags = []
            for i in range(lag):
                start = lag - i - 1
                end = start + n_obs
                dx_lags.append(dx[start:end])

            dx_lags = np.column_stack(dx_lags)

            X = np.column_stack([
                np.ones(n_obs),
                x_lag[lag:lag + n_obs],
                dx_lags
            ])
            y = dx[lag:lag + n_obs]

        if len(y) < X.shape[1] + 2:
            continue

        try:
            # OLS regression
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
            residuals = y - X @ beta

            # AIC for lag selection
            n_obs = len(y)
            k = X.shape[1]
            sse = np.sum(residuals ** 2)
            aic = n_obs * np.log(sse / n_obs) + 2 * k

            if aic < best_aic:
                best_aic = aic

                # Standard error of gamma coefficient
                mse = sse / (n_obs - k)
                var_beta = mse * np.linalg.inv(X.T @ X)
                se_gamma = np.sqrt(var_beta[1, 1])

                gamma = beta[1]
                adf_stat = gamma / se_gamma
                best_result = (adf_stat, lag)

        except (np.linalg.LinAlgError, ValueError):
            continue

    if best_result is None:
        return 0.0, 1.0, 0

    adf_stat, lags = best_result

    # Approximate p-value using MacKinnon critical values
    # For n > 500, critical values: 1%: -3.43, 5%: -2.86, 10%: -2.57
    if adf_stat < -3.43:
        p_value = 0.01
    elif adf_stat < -2.86:
        p_value = 0.05
    elif adf_stat < -2.57:
        p_value = 0.10
    else:
        p_value = 0.5  # Not significant

    return float(adf_stat), p_value, lags


def _engle_granger_test(y: np.ndarray, x: np.ndarray) -> Dict[str, float]:
    """
    Engle-Granger two-step cointegration test.

    Step 1: Regress y on x
    Step 2: Test residuals for stationarity
    """
    n = len(y)

    # Step 1: OLS regression
    X = np.column_stack([np.ones(n), x])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]

    alpha = beta[0]  # Intercept
    hedge_ratio = beta[1]  # Cointegrating coefficient

    # Residuals (spread)
    residuals = y - alpha - hedge_ratio * x

    # Step 2: ADF test on residuals
    adf_stat, p_value, lags = _adf_test(residuals)

    # Half-life of mean reversion (if cointegrated)
    if p_value < 0.10:
        # AR(1) on residuals
        resid_lag = residuals[:-1]
        resid_diff = np.diff(residuals)

        if len(resid_lag) > 2:
            rho = np.corrcoef(resid_lag, resid_diff + resid_lag)[0, 1]
            if rho > 0 and rho < 1:
                half_life = -np.log(2) / np.log(rho)
            else:
                half_life = np.nan
        else:
            half_life = np.nan
    else:
        half_life = np.nan

    return {
        "adf_stat": adf_stat,
        "p_value": p_value,
        "hedge_ratio": float(hedge_ratio),
        "intercept": float(alpha),
        "half_life": float(half_life) if not np.isnan(half_life) else None,
        "spread_std": float(np.std(residuals)),
        "is_cointegrated": p_value < 0.10,
    }


class CointegrationEngine(BaseEngine):
    """
    Cointegration engine.

    Tests for long-run equilibrium relationships using
    Engle-Granger methodology.

    Outputs:
        - derived.cointegration: Pairwise cointegration results
    """

    name = "cointegration"
    phase = "derived"
    default_normalization = None  # Uses levels

    def run(
        self,
        df: pd.DataFrame,
        run_id: str,
        significance: float = 0.10,
        **params
    ) -> Dict[str, Any]:
        """
        Run cointegration analysis.

        Args:
            df: Indicator data (levels, not returns)
            run_id: Unique run identifier
            significance: Significance level for cointegration test

        Returns:
            Dict with summary metrics
        """
        df_clean = df
        indicators = df_clean.columns.tolist()
        n_indicators = len(indicators)

        window_start = df_clean.index.min().date()
        window_end = df_clean.index.max().date()

        # Test all pairs
        records = []
        n_cointegrated = 0
        all_half_lives = []

        for i in range(n_indicators):
            for j in range(i + 1, n_indicators):
                y = df_clean.iloc[:, i].values
                x = df_clean.iloc[:, j].values

                try:
                    result = _engle_granger_test(y, x)

                    if result["is_cointegrated"]:
                        n_cointegrated += 1
                        if result["half_life"] is not None:
                            all_half_lives.append(result["half_life"])

                    records.append({
                        "indicator_1": indicators[i],
                        "indicator_2": indicators[j],
                        "window_start": window_start,
                        "window_end": window_end,
                        "adf_stat": result["adf_stat"],
                        "p_value": result["p_value"],
                        "hedge_ratio": result["hedge_ratio"],
                        "intercept": result["intercept"],
                        "half_life": result["half_life"],
                        "spread_std": result["spread_std"],
                        "is_cointegrated": result["is_cointegrated"],
                        "run_id": run_id,
                    })

                except Exception as e:
                    logger.warning(f"Cointegration test failed for {indicators[i]}-{indicators[j]}: {e}")
                    continue

        if records:
            df_results = pd.DataFrame(records)
            self.store_results("cointegration", df_results, run_id)

        # Summary metrics
        n_pairs = len(records)

        metrics = {
            "n_indicators": n_indicators,
            "n_pairs": n_pairs,
            "n_samples": len(df_clean),
            "n_cointegrated": n_cointegrated,
            "cointegration_rate": n_cointegrated / n_pairs if n_pairs > 0 else 0.0,
            "avg_half_life": float(np.mean(all_half_lives)) if all_half_lives else None,
            "significance_level": significance,
        }

        logger.info(
            f"Cointegration complete: {n_indicators} indicators, "
            f"{n_cointegrated}/{n_pairs} cointegrated pairs"
        )

        return metrics
