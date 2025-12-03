"""
Influence Lens - Indicator influence/importance analysis

Combines multiple methods to score indicator importance.
"""

from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import time

from .base_lens import BaseLens


class InfluenceLens(BaseLens):
    """
    Influence Lens: Multi-method indicator importance scoring.

    Methods used:
    - Variance contribution
    - Correlation strength
    - Random Forest importance
    - Rolling influence over time
    """

    name = "influence"
    description = "Multi-method indicator influence analysis"
    category = "basic"

    def analyze(
        self,
        df: pd.DataFrame,
        target_col: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Analyze indicator influence.

        Args:
            df: Input DataFrame
            target_col: Target column for RF importance (default: first PC)

        Returns:
            Dictionary with influence analysis
        """
        start_time = time.time()

        self.validate_input(df)
        data = self.prepare_data(df)
        value_cols = self.get_value_columns(data)

        # 1. Variance contribution
        variances = data[value_cols].var()
        total_var = variances.sum()
        variance_contrib = (variances / total_var).to_dict()

        # 2. Mean absolute correlation
        corr_matrix = data[value_cols].corr().abs()
        mean_corr = {}
        for col in value_cols:
            other_corrs = corr_matrix[col].drop(col)
            mean_corr[col] = float(other_corrs.mean())

        # 3. Random Forest importance
        rf_importance = self._compute_rf_importance(data, value_cols, target_col)

        # 4. Composite score (weighted average)
        weights = kwargs.get("weights", {"variance": 0.3, "correlation": 0.3, "rf": 0.4})

        composite = {}
        for col in value_cols:
            score = (
                weights.get("variance", 0.33) * variance_contrib.get(col, 0) +
                weights.get("correlation", 0.33) * mean_corr.get(col, 0) +
                weights.get("rf", 0.34) * rf_importance.get(col, 0)
            )
            composite[col] = score

        # Normalize composite scores
        max_score = max(composite.values()) if composite else 1
        composite = {k: v / max_score for k, v in composite.items()}

        result = {
            "variance_contribution": variance_contrib,
            "mean_correlation": mean_corr,
            "rf_importance": rf_importance,
            "composite_score": composite,
            "top_5_influential": dict(
                sorted(composite.items(), key=lambda x: -x[1])[:5]
            ),
            "weights_used": weights,
        }

        self._computation_time = time.time() - start_time
        self._last_result = result

        return result

    def _compute_rf_importance(
        self,
        data: pd.DataFrame,
        value_cols: List[str],
        target_col: Optional[str]
    ) -> Dict[str, float]:
        """Compute Random Forest feature importance."""
        try:
            # If no target, use first PC as synthetic target
            if target_col is None or target_col not in value_cols:
                from sklearn.decomposition import PCA
                pca = PCA(n_components=1)
                target = pca.fit_transform(data[value_cols])[:, 0]
            else:
                target = data[target_col].values
                value_cols = [c for c in value_cols if c != target_col]

            X = data[value_cols].values
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            rf.fit(X_scaled, target)

            importance = dict(zip(value_cols, rf.feature_importances_))

            # If we excluded target, add it back with high importance
            if target_col and target_col not in importance:
                importance[target_col] = 1.0

            return importance

        except Exception:
            # Return equal weights on failure
            return {col: 1.0 / len(value_cols) for col in value_cols}

    def rank_indicators(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Rank indicators by composite influence score.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with indicator rankings
        """
        result = self.analyze(df, **kwargs)
        composite = result["composite_score"]

        ranking = pd.DataFrame([
            {"indicator": k, "score": v}
            for k, v in composite.items()
        ])
        ranking = ranking.sort_values("score", ascending=False)
        ranking["rank"] = range(1, len(ranking) + 1)

        return ranking.reset_index(drop=True)

    def rolling_influence(
        self,
        df: pd.DataFrame,
        window: int = 60,
        **kwargs
    ) -> pd.DataFrame:
        """
        Compute rolling influence scores over time.

        Args:
            df: Input DataFrame
            window: Rolling window size

        Returns:
            DataFrame with rolling influence per indicator
        """
        data = self.prepare_data(df)
        value_cols = self.get_value_columns(data)

        rolling_scores = []

        for i in range(window, len(data)):
            window_data = data.iloc[i-window:i]
            result = self.analyze(window_data, **kwargs)

            row = {"date": data.iloc[i]["date"]}
            row.update(result["composite_score"])
            rolling_scores.append(row)

        return pd.DataFrame(rolling_scores)
