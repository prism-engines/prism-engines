"""
Granger Lens - Granger Causality Analysis

Tests whether one indicator helps predict another.
"""

from typing import Dict, Any, Optional, List, Tuple
import pandas as pd
import numpy as np
from scipy import stats
import time
import warnings

from .base_lens import BaseLens

warnings.filterwarnings('ignore')


class GrangerLens(BaseLens):
    """
    Granger Lens: Granger causality testing between indicators.

    Provides:
    - Pairwise Granger causality test results
    - Causal influence network
    - Key "leading" indicators
    """

    name = "granger"
    description = "Granger causality analysis for predictive relationships"
    category = "basic"

    def analyze(
        self,
        df: pd.DataFrame,
        max_lag: int = 5,
        significance: float = 0.05,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run Granger causality analysis.

        Args:
            df: Input DataFrame
            max_lag: Maximum lag to test
            significance: Significance level (default 0.05)

        Returns:
            Dictionary with causality results
        """
        start_time = time.time()

        self.validate_input(df)
        data = self.prepare_data(df)
        value_cols = self.get_value_columns(data)

        # Limit columns for computational efficiency
        max_cols = kwargs.get("max_columns", 20)
        if len(value_cols) > max_cols:
            # Select columns with highest variance
            variances = data[value_cols].var().sort_values(ascending=False)
            value_cols = variances.head(max_cols).index.tolist()

        # Run pairwise Granger tests
        causality_results = []

        for cause_col in value_cols:
            for effect_col in value_cols:
                if cause_col == effect_col:
                    continue

                result = self._granger_test(
                    data[cause_col].values,
                    data[effect_col].values,
                    max_lag
                )

                if result is not None:
                    causality_results.append({
                        "cause": cause_col,
                        "effect": effect_col,
                        "f_stat": result["f_stat"],
                        "p_value": result["p_value"],
                        "best_lag": result["best_lag"],
                        "significant": result["p_value"] < significance
                    })

        # Build causality matrix
        causality_df = pd.DataFrame(causality_results)

        # Count significant causal relationships
        if len(causality_df) > 0:
            # How many things does each indicator cause?
            causes_count = causality_df[causality_df["significant"]].groupby("cause").size()
            # How many things cause each indicator?
            effects_count = causality_df[causality_df["significant"]].groupby("effect").size()

            # Net causality (causes - effects)
            net_causality = causes_count.subtract(effects_count, fill_value=0)
        else:
            causes_count = pd.Series(dtype=int)
            effects_count = pd.Series(dtype=int)
            net_causality = pd.Series(dtype=float)

        result = {
            "n_pairs_tested": len(causality_results),
            "n_significant": int(causality_df["significant"].sum()) if len(causality_df) > 0 else 0,
            "significance_level": significance,
            "max_lag": max_lag,
            "pairwise_results": causality_results,
            "causes_count": causes_count.to_dict(),
            "effects_count": effects_count.to_dict(),
            "net_causality": net_causality.to_dict(),
            "top_leaders": net_causality.nlargest(5).to_dict() if len(net_causality) > 0 else {},
            "top_followers": net_causality.nsmallest(5).to_dict() if len(net_causality) > 0 else {},
        }

        self._computation_time = time.time() - start_time
        self._last_result = result

        return result

    def _granger_test(
        self,
        x: np.ndarray,
        y: np.ndarray,
        max_lag: int
    ) -> Optional[Dict]:
        """
        Perform Granger causality test: does x Granger-cause y?

        Uses OLS regression comparison.
        """
        try:
            from statsmodels.tsa.stattools import grangercausalitytests

            # Create DataFrame for test
            test_data = pd.DataFrame({'x': x, 'y': y}).dropna()

            if len(test_data) < max_lag + 10:
                return None

            # Run test
            results = grangercausalitytests(
                test_data[['y', 'x']],  # Note: y first, then x
                maxlag=max_lag,
                verbose=False
            )

            # Find best lag (lowest p-value)
            best_lag = 1
            best_p = 1.0
            best_f = 0.0

            for lag in range(1, max_lag + 1):
                if lag in results:
                    # Get F-test p-value
                    f_test = results[lag][0]['ssr_ftest']
                    p_value = f_test[1]
                    f_stat = f_test[0]

                    if p_value < best_p:
                        best_p = p_value
                        best_f = f_stat
                        best_lag = lag

            return {
                "f_stat": float(best_f),
                "p_value": float(best_p),
                "best_lag": best_lag
            }

        except Exception:
            return None

    def rank_indicators(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Rank indicators by their causal influence (net causality score).

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with indicator rankings
        """
        result = self.analyze(df, **kwargs)
        net_causality = result["net_causality"]

        if not net_causality:
            return pd.DataFrame(columns=["indicator", "score", "rank"])

        ranking = pd.DataFrame([
            {"indicator": k, "score": v}
            for k, v in net_causality.items()
        ])
        ranking = ranking.sort_values("score", ascending=False)
        ranking["rank"] = range(1, len(ranking) + 1)

        return ranking.reset_index(drop=True)

    def get_causality_network(self, df: pd.DataFrame, **kwargs) -> Dict:
        """
        Get data for causality network visualization.

        Args:
            df: Input DataFrame

        Returns:
            Dictionary with nodes and edges
        """
        result = self.analyze(df, **kwargs)

        nodes = list(set(
            [r["cause"] for r in result["pairwise_results"]] +
            [r["effect"] for r in result["pairwise_results"]]
        ))

        edges = [
            {
                "source": r["cause"],
                "target": r["effect"],
                "weight": -np.log10(r["p_value"] + 1e-10),  # Transform p-value to weight
                "significant": r["significant"]
            }
            for r in result["pairwise_results"]
            if r["significant"]
        ]

        return {"nodes": nodes, "edges": edges}
