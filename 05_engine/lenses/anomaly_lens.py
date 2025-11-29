"""
Anomaly Lens - Anomaly detection in indicator behavior

Identifies unusual patterns and outliers in the data.
"""

from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor
import time

from .base_lens import BaseLens


class AnomalyLens(BaseLens):
    """
    Anomaly Lens: Detect anomalies in indicator behavior.

    Provides:
    - Multivariate anomaly detection
    - Per-indicator anomaly scores
    - Anomaly clustering and patterns
    """

    name = "anomaly"
    description = "Anomaly detection in indicator behavior"
    category = "advanced"

    def analyze(
        self,
        df: pd.DataFrame,
        contamination: float = 0.05,
        method: str = "isolation_forest",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Detect anomalies in the data.

        Args:
            df: Input DataFrame
            contamination: Expected proportion of anomalies
            method: 'isolation_forest' or 'lof' (Local Outlier Factor)

        Returns:
            Dictionary with anomaly detection results
        """
        start_time = time.time()

        self.validate_input(df)
        data = self.prepare_data(df)
        value_cols = self.get_value_columns(data)

        # Scale data
        scaler = StandardScaler()
        scaled_values = scaler.fit_transform(data[value_cols])

        # Detect anomalies
        if method == "isolation_forest":
            detector = IsolationForest(
                contamination=contamination,
                random_state=42,
                n_jobs=-1
            )
            predictions = detector.fit_predict(scaled_values)
            scores = detector.score_samples(scaled_values)
        elif method == "lof":
            detector = LocalOutlierFactor(
                contamination=contamination,
                n_neighbors=20,
                novelty=False
            )
            predictions = detector.fit_predict(scaled_values)
            scores = detector.negative_outlier_factor_
        else:
            raise ValueError(f"Unknown method: {method}")

        # Predictions: 1 = normal, -1 = anomaly
        is_anomaly = predictions == -1

        # Get anomaly dates
        anomaly_dates = data.loc[is_anomaly, "date"].tolist()

        # Per-indicator contribution to anomalies
        indicator_anomaly_contribution = {}
        anomaly_indices = np.where(is_anomaly)[0]

        for col in value_cols:
            if len(anomaly_indices) > 0:
                # Mean absolute z-score during anomalies
                col_idx = value_cols.index(col)
                anomaly_values = scaled_values[anomaly_indices, col_idx]
                contribution = float(np.mean(np.abs(anomaly_values)))
            else:
                contribution = 0.0
            indicator_anomaly_contribution[col] = contribution

        # Recent anomalies
        recent_window = kwargs.get("recent_window", 30)
        recent_data = data.tail(recent_window)
        recent_anomalies = is_anomaly[-recent_window:].sum()

        # Anomaly clustering (are anomalies clustered in time?)
        anomaly_clustering = self._analyze_anomaly_clustering(is_anomaly)

        result = {
            "method": method,
            "contamination": contamination,
            "n_anomalies": int(is_anomaly.sum()),
            "anomaly_rate": float(is_anomaly.mean()),
            "anomaly_dates": [str(d) for d in anomaly_dates],
            "anomaly_scores": scores.tolist(),
            "indicator_contribution": indicator_anomaly_contribution,
            "recent_anomalies": int(recent_anomalies),
            "recent_window": recent_window,
            "anomaly_clustering": anomaly_clustering,
            "is_anomaly": is_anomaly.tolist(),
            "dates": [str(d) for d in data["date"].tolist()],
        }

        self._computation_time = time.time() - start_time
        self._last_result = result

        return result

    def _analyze_anomaly_clustering(self, is_anomaly: np.ndarray) -> Dict:
        """Analyze if anomalies are clustered in time."""
        if is_anomaly.sum() < 2:
            return {"clustered": False, "max_cluster_size": 0}

        # Find consecutive anomaly runs
        clusters = []
        current_run = 0

        for val in is_anomaly:
            if val:
                current_run += 1
            elif current_run > 0:
                clusters.append(current_run)
                current_run = 0

        if current_run > 0:
            clusters.append(current_run)

        if not clusters:
            return {"clustered": False, "max_cluster_size": 0}

        max_cluster = max(clusters)
        mean_cluster = np.mean(clusters)

        return {
            "clustered": max_cluster > 2,
            "max_cluster_size": int(max_cluster),
            "mean_cluster_size": float(mean_cluster),
            "n_clusters": len(clusters),
        }

    def rank_indicators(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Rank indicators by their contribution to anomalies.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with indicator rankings
        """
        result = self.analyze(df, **kwargs)
        contributions = result["indicator_contribution"]

        # Normalize
        max_contrib = max(contributions.values()) if contributions else 1
        normalized = {k: v / max_contrib for k, v in contributions.items()}

        ranking = pd.DataFrame([
            {"indicator": k, "score": v}
            for k, v in normalized.items()
        ])
        ranking = ranking.sort_values("score", ascending=False)
        ranking["rank"] = range(1, len(ranking) + 1)

        return ranking.reset_index(drop=True)

    def get_anomaly_details(
        self,
        df: pd.DataFrame,
        top_n: int = 10,
        **kwargs
    ) -> pd.DataFrame:
        """
        Get detailed information about top anomalies.

        Args:
            df: Input DataFrame
            top_n: Number of top anomalies to return

        Returns:
            DataFrame with anomaly details
        """
        result = self.analyze(df, **kwargs)

        data = self.prepare_data(df)
        scores = np.array(result["anomaly_scores"])
        is_anomaly = np.array(result["is_anomaly"])

        # Get top anomalies by score (lowest = most anomalous)
        anomaly_indices = np.where(is_anomaly)[0]
        if len(anomaly_indices) == 0:
            return pd.DataFrame()

        anomaly_scores = scores[anomaly_indices]
        top_indices = anomaly_indices[np.argsort(anomaly_scores)[:top_n]]

        details = []
        for idx in top_indices:
            row = {
                "date": data.iloc[idx]["date"],
                "anomaly_score": float(scores[idx]),
            }
            # Add indicator values
            for col in self.get_value_columns(data):
                row[col] = float(data.iloc[idx][col])
            details.append(row)

        return pd.DataFrame(details)
