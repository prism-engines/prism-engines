"""
PRISM PCA Engine

Principal Component Analysis for structure analysis.

Measures:
- Variance explained by each component
- Loading matrix (indicator weights)
- Effective dimensionality

Phase: Unbound
Normalization: Z-score required
"""

import logging
from typing import Dict, Any, Optional
from datetime import date

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from .base import BaseEngine


logger = logging.getLogger(__name__)


class PCAEngine(BaseEngine):
    """
    Principal Component Analysis engine.
    
    Outputs:
        - derived.pca_loadings: Indicator loadings per component
        - derived.pca_variance: Variance explained per component
    """
    
    name = "pca"
    phase = "derived"
    default_normalization = "zscore"
    
    def run(
        self,
        df: pd.DataFrame,
        run_id: str,
        n_components: Optional[int] = None,
        **params
    ) -> Dict[str, Any]:
        """
        Run PCA analysis.
        
        Args:
            df: Normalized indicator data (rows=dates, cols=indicators)
            run_id: Unique run identifier
            n_components: Number of components (default: all)
        
        Returns:
            Dict with summary metrics
        """
        # Clean layer guarantees no NaN
        df_clean = df
        
        if len(df_clean) < 3 * len(df_clean.columns):
            logger.warning(
                f"Limited samples ({len(df_clean)}) for {len(df_clean.columns)} indicators. "
                f"Recommended: {3 * len(df_clean.columns)}+"
            )
        
        # Fit PCA
        n_comp = n_components or min(len(df_clean), len(df_clean.columns))
        pca = PCA(n_components=n_comp)
        pca.fit(df_clean.values)
        
        # Extract results
        loadings = pd.DataFrame(
            pca.components_.T,
            index=df_clean.columns,
            columns=[f"PC{i+1}" for i in range(n_comp)]
        )
        
        variance = pd.DataFrame({
            "component": range(1, n_comp + 1),
            "variance_explained": pca.explained_variance_ratio_,
            "cumulative_variance": np.cumsum(pca.explained_variance_ratio_),
        })
        
        # Store results
        window_start = df_clean.index.min().date()
        window_end = df_clean.index.max().date()
        
        self._store_loadings(loadings, window_start, window_end, run_id)
        self._store_variance(variance, window_start, window_end, run_id)
        
        # Compute summary metrics
        n_components_90 = np.searchsorted(
            np.cumsum(pca.explained_variance_ratio_), 0.9
        ) + 1
        
        metrics = {
            "n_indicators": len(df_clean.columns),
            "n_samples": len(df_clean),
            "n_components": n_comp,
            "variance_pc1": float(pca.explained_variance_ratio_[0]),
            "variance_pc2": float(pca.explained_variance_ratio_[1]) if n_comp > 1 else 0,
            "components_for_90pct": int(n_components_90),
            "effective_dimensionality": float(
                1 / np.sum(pca.explained_variance_ratio_ ** 2)
            ),
        }
        
        logger.info(
            f"PCA complete: PC1 explains {metrics['variance_pc1']:.1%}, "
            f"{metrics['components_for_90pct']} components for 90%"
        )
        
        return metrics
    
    def _store_loadings(
        self,
        loadings: pd.DataFrame,
        window_start: date,
        window_end: date,
        run_id: str,
    ):
        """Store loadings to derived.pca_loadings."""
        # Reshape: one row per (indicator, component)
        records = []
        for indicator_id in loadings.index:
            for i, col in enumerate(loadings.columns):
                records.append({
                    "indicator_id": indicator_id,
                    "window_start": window_start,
                    "window_end": window_end,
                    "component": i + 1,
                    "loading": float(loadings.loc[indicator_id, col]),
                    "run_id": run_id,
                })
        
        df = pd.DataFrame(records)
        self.store_results("pca_loadings", df, run_id)
    
    def _store_variance(
        self,
        variance: pd.DataFrame,
        window_start: date,
        window_end: date,
        run_id: str,
    ):
        """Store variance to derived.pca_variance."""
        df = variance.copy()
        df["window_start"] = window_start
        df["window_end"] = window_end
        df["run_id"] = run_id
        
        # Reorder columns to match schema
        df = df[[
            "window_start", "window_end", "component",
            "variance_explained", "cumulative_variance", "run_id"
        ]]
        
        self.store_results("pca_variance", df, run_id)
