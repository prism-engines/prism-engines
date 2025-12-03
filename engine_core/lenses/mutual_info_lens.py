"""
Mutual Information Lens - Information-theoretic analysis

Measures non-linear dependencies between indicators.
"""

from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import StandardScaler
import time

from .base_lens import BaseLens


class MutualInfoLens(BaseLens):
    """
    Mutual Information Lens: Information-theoretic dependency analysis.

    Provides:
    - Pairwise mutual information matrix
    - Total information shared by each indicator
    - Non-linear relationship detection
    """

    name = "mutual_info"
    description = "Mutual information analysis for non-linear dependencies"
    category = "basic"

    def analyze(self, df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Compute mutual information between indicators.

        Args:
            df: Input DataFrame

        Returns:
            Dictionary with MI analysis results
        """
        start_time = time.time()

        self.validate_input(df)
        data = self.prepare_data(df)
        value_cols = self.get_value_columns(data)

        n_neighbors = kwargs.get("n_neighbors", 3)

        # Limit columns for efficiency
        max_cols = kwargs.get("max_columns", 30)
        if len(value_cols) > max_cols:
            variances = data[value_cols].var().sort_values(ascending=False)
            value_cols = variances.head(max_cols).index.tolist()

        # Scale data
        scaler = StandardScaler()
        scaled_data = pd.DataFrame(
            scaler.fit_transform(data[value_cols]),
            columns=value_cols
        )

        # Compute pairwise MI
        mi_matrix = pd.DataFrame(
            np.zeros((len(value_cols), len(value_cols))),
            index=value_cols,
            columns=value_cols
        )

        for i, col_i in enumerate(value_cols):
            X = scaled_data.drop(columns=[col_i]).values
            y = scaled_data[col_i].values

            mi_scores = mutual_info_regression(X, y, n_neighbors=n_neighbors)

            other_cols = [c for c in value_cols if c != col_i]
            for j, col_j in enumerate(other_cols):
                mi_matrix.loc[col_i, col_j] = mi_scores[j]

        # Total MI per indicator
        total_mi = mi_matrix.sum(axis=1).to_dict()

        # Normalize to get relative importance
        max_mi = max(total_mi.values()) if total_mi else 1
        normalized_mi = {k: v / max_mi for k, v in total_mi.items()}

        # Find highly connected pairs
        high_mi_pairs = []
        threshold = mi_matrix.values.mean() + mi_matrix.values.std()

        for i, col_i in enumerate(value_cols):
            for j, col_j in enumerate(value_cols):
                if i < j and mi_matrix.loc[col_i, col_j] > threshold:
                    high_mi_pairs.append({
                        "indicator_1": col_i,
                        "indicator_2": col_j,
                        "mutual_info": float(mi_matrix.loc[col_i, col_j])
                    })

        high_mi_pairs.sort(key=lambda x: -x["mutual_info"])

        result = {
            "mi_matrix": mi_matrix.to_dict(),
            "total_mi": total_mi,
            "normalized_mi": normalized_mi,
            "high_mi_pairs": high_mi_pairs[:20],  # Top 20 pairs
            "mean_mi": float(mi_matrix.values.mean()),
            "threshold": float(threshold),
        }

        self._computation_time = time.time() - start_time
        self._last_result = result

        return result

    def rank_indicators(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Rank indicators by total mutual information shared.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with indicator rankings
        """
        result = self.analyze(df, **kwargs)
        normalized_mi = result["normalized_mi"]

        ranking = pd.DataFrame([
            {"indicator": k, "score": v}
            for k, v in normalized_mi.items()
        ])
        ranking = ranking.sort_values("score", ascending=False)
        ranking["rank"] = range(1, len(ranking) + 1)

        return ranking.reset_index(drop=True)
