#!/usr/bin/env python
"""
PRISM ML Engine v1 – Unsupervised Regime Discovery

Goal:
- No "risk on / risk off" lexicon.
- Discover latent market regimes directly from market indicators.
- Use unsupervised ML (PCA + Gaussian Mixture) on engineered features
  from the master_panel.csv (market-only columns).

Outputs:
- models/prism_ml_engine.pkl          → fitted sklearn pipeline (Imputer+Scaler+PCA+GMM)
- models/prism_ml_engine_features.txt → list of feature names
- data/labels/prism_ml_states.csv     → date, state, state_id, state_prob_<k>

Notes:
- This is v1: focuses on market indicators only, no economic inputs.
- Coherence / pairwise-engine metrics will be added in later versions.
"""

import os
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Tuple

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import Pipeline
import joblib


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
PANEL_PATH = BASE_DIR / "data" / "panels" / "master_panel.csv"
MODELS_DIR = BASE_DIR / "models"
LABELS_DIR = BASE_DIR / "data" / "labels"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
LABELS_DIR.mkdir(parents=True, exist_ok=True)

# Market columns (only the ones that exist will be used)
MARKET_COLUMNS_PREFERRED = [
    # Core equity indices
    "spy", "qqq", "iwm", "dia",
    # International / regions
    "vea", "vwo", "ewj", "fez", "ewz", "fxi", "inda",
    # Sectors
    "xlk", "xlf", "xle", "xlv", "xli", "xlu", "xlp", "xly", "xlb", "xlre",
    # Factors
    "mtum", "vlue", "qual", "smlv", "usmv",
    # Bonds
    "tlt", "ief", "shy", "hyg", "lqd", "bnd",
    # Real assets / commodities / alt
    "gld", "slv", "uso", "dbc",
    # Optional: volatility / FX / crypto (will be mostly NaN on long history)
    "vix", "dxy", "btc", "eth",
]


@dataclass
class PrismMLEngineConfig:
    n_components_pca: int = 10     # size of latent space
    n_states: int = 4              # number of market states to discover
    random_state: int = 42
    min_valid_observations: int = 200  # require at least this many rows after cleaning
    feature_windows: Tuple[int, ...] = (21, 63, 252)  # ~1m, 3m, 1y
    min_date: str = "1990-01-01"   # ignore ultra-early history where most indicators are NaN


# ----------------------------------------------------------------------
# Loading and feature engineering
# ----------------------------------------------------------------------

def load_panel(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Panel file not found: {path}")
    df = pd.read_csv(path, parse_dates=["date"])
    if "date" not in df.columns:
        raise ValueError("master_panel.csv must contain a 'date' column.")
    df = df.sort_values("date").reset_index(drop=True)
    return df


def select_market_columns(df: pd.DataFrame) -> List[str]:
    cols_present = [c for c in MARKET_COLUMNS_PREFERRED if c in df.columns]
    if not cols_present:
        raise ValueError("No configured market columns found in master_panel.csv")
    return cols_present


def build_features(
    df: pd.DataFrame,
    cols: List[str],
    cfg: PrismMLEngineConfig,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Build a feature matrix from raw market indicator levels.
    - Level z-score
    - 1-day return
    - Rolling mean/std over windows (21, 63, 252)
    - Rolling slope (simple linear regression over window)
    """
    df = df.copy()
    df = df[df["date"] >= cfg.min_date].reset_index(drop=True)

    # Keep only date + selected columns
    df = df[["date"] + cols]

    # Container for all feature blocks to avoid fragmentation
    feature_blocks: List[pd.DataFrame] = []

    for col in cols:
        s = df[col].astype(float)
        block = pd.DataFrame(index=df.index)

        # Level z-score (per series)
        mean = s.mean(skipna=True)
        std = s.std(skipna=True)
        if std == 0 or np.isnan(std):
            z = pd.Series(np.nan, index=s.index)
        else:
            z = (s - mean) / std
        block[f"{col}_z"] = z

        # 1-day return
        block[f"{col}_ret_1"] = s.pct_change()

        # Rolling window features
        for w in cfg.feature_windows:
            roll = s.rolling(w, min_periods=max(5, w // 4))
            block[f"{col}_sma_{w}"] = roll.mean()
            block[f"{col}_std_{w}"] = roll.std()

            # Simple slope over window using linear regression on [0..w-1]
            # Use rolling apply
            def _slope(x: np.ndarray) -> float:
                x = np.asarray(x, dtype=float)
                if np.all(np.isnan(x)):
                    return np.nan
                n = len(x)
                t = np.arange(n, dtype=float)
                # Remove NaNs
                mask = ~np.isnan(x)
                if mask.sum() < max(5, n // 3):
                    return np.nan
                t = t[mask]
                y = x[mask]
                t_mean = t.mean()
                y_mean = y.mean()
                denom = np.sum((t - t_mean) ** 2)
                if denom == 0:
                    return 0.0
                num = np.sum((t - t_mean) * (y - y_mean))
                return num / denom

            block[f"{col}_slope_{w}"] = roll.apply(_slope, raw=True)

        feature_blocks.append(block)

    # Concatenate all blocks horizontally
    features = pd.concat(feature_blocks, axis=1)

    # Replace inf with NaN, then drop rows with all NaNs
    features = features.replace([np.inf, -np.inf], np.nan)
    # Drop rows where EVERY feature is NaN
    all_nan_mask = features.isna().all(axis=1)
    features = features.loc[~all_nan_mask].copy()
    # Also align dates with remaining rows
    dates = df.loc[features.index, "date"].reset_index(drop=True)
    features = features.reset_index(drop=True)

    # Some columns might be fully NaN (e.g., BTC/ETH on older history) → drop them
    non_all_nan_cols = features.columns[~features.isna().all(axis=0)]
    features = features[non_all_nan_cols]

    feature_names = list(features.columns)
    return pd.concat([dates.rename("date"), features], axis=1), feature_names


# ----------------------------------------------------------------------
# ML Training (PCA + GMM)
# ----------------------------------------------------------------------

def build_pipeline(cfg: PrismMLEngineConfig) -> Pipeline:
    """
    Create an sklearn pipeline:
    - Impute missing values (median)
    - Scale features (StandardScaler)
    - PCA → reduce to cfg.n_components_pca
    - Gaussian Mixture → cfg.n_states states
    """
    pca = PCA(
        n_components=cfg.n_components_pca,
        random_state=cfg.random_state,
    )
    gmm = GaussianMixture(
        n_components=cfg.n_states,
        covariance_type="full",
        random_state=cfg.random_state,
        n_init=5,
    )
    pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("pca", pca),
            ("gmm", gmm),
        ]
    )
    return pipe


def fit_prism_ml_engine(
    df_features: pd.DataFrame,
    feature_names: List[str],
    cfg: PrismMLEngineConfig,
):
    """
    Fit the unsupervised ML engine on the feature matrix.
    Returns:
      - pipeline
      - states_df (date, state, state_id, probabilities)
    """
    dates = df_features["date"].reset_index(drop=True)
    X = df_features[feature_names].values

    # Sanity: require some minimum data
    if X.shape[0] < cfg.min_valid_observations:
        raise ValueError(
            f"Not enough valid observations for ML engine: {X.shape[0]} rows "
            f"(minimum required: {cfg.min_valid_observations})."
        )

    print("============================================")
    print("  PRISM ML ENGINE – UNSUPERVISED REGIMES")
    print("============================================")
    print(f"Rows (after cleaning): {X.shape[0]}")
    print(f"Features: {X.shape[1]}")
    print(f"PCA components: {cfg.n_components_pca}")
    print(f"States to discover: {cfg.n_states}")
    print()

    pipe = build_pipeline(cfg)

    # Fit pipeline
    print("Fitting PCA + GMM pipeline...")
    pipe.fit(X)

    # Extract GMM and PCA for introspection
    gmm: GaussianMixture = pipe.named_steps["gmm"]
    pca: PCA = pipe.named_steps["pca"]

    # State assignments and probabilities
    state_ids = gmm.predict(pipe.named_steps["pca"].transform(
        pipe.named_steps["scaler"].transform(
            pipe.named_steps["imputer"].transform(X)
        )
    ))
    probs = gmm.predict_proba(
        pipe.named_steps["pca"].transform(
            pipe.named_steps["scaler"].transform(
                pipe.named_steps["imputer"].transform(X)
            )
        )
    )

    states_df = pd.DataFrame({"date": dates, "state_id": state_ids})
    # Optional: human-friendly label (for now just "state_0" etc.)
    states_df["state_label"] = states_df["state_id"].apply(lambda x: f"state_{int(x)}")

    # Add probabilities
    for k in range(cfg.n_states):
        states_df[f"state_prob_{k}"] = probs[:, k]

    # Basic summary
    print("State counts:")
    print(states_df["state_id"].value_counts().sort_index())
    print()

    # If master_panel has MRF, we can print MT summary per state
    try:
        panel = pd.read_csv(PANEL_PATH, parse_dates=["date"])
        if "MRF" in panel.columns:
            merged = states_df.merge(panel[["date", "MRF"]], on="date", how="left")
            print("Average MRF by state:")
            print(merged.groupby("state_id")["MRF"].mean())
            print()
    except Exception:
        # MRF might not exist; it's fine
        pass

    # PCA explained variance
    print("PCA explained variance ratio (first few components):")
    print(pca.explained_variance_ratio_[: min(10, cfg.n_components_pca)])
    print()

    return pipe, states_df


def save_outputs(
    pipe: Pipeline,
    feature_names: List[str],
    states_df: pd.DataFrame,
    cfg: PrismMLEngineConfig,
):
    # Model
    model_path = MODELS_DIR / "prism_ml_engine.pkl"
    joblib.dump(
        {
            "pipeline": pipe,
            "feature_names": feature_names,
            "config": asdict(cfg),
        },
        model_path,
    )

    # Feature list
    features_path = MODELS_DIR / "prism_ml_engine_features.txt"
    with features_path.open("w") as f:
        for name in feature_names:
            f.write(f"{name}\n")

    # States
    labels_path = LABELS_DIR / "prism_ml_states.csv"
    states_df.to_csv(labels_path, index=False)

    print(f"Model saved to: {model_path}")
    print(f"Feature list saved to: {features_path}")
    print(f"State labels saved to: {labels_path}")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    cfg = PrismMLEngineConfig()

    print("============================================")
    print("  PRISM ML ENGINE – BUILDING v1")
    print("============================================")
    print(f"Base dir: {BASE_DIR}")
    print(f"Panel   : {PANEL_PATH}")
    print()

    df_panel = load_panel(PANEL_PATH)
    market_cols = select_market_columns(df_panel)

    print(f"Using {len(market_cols)} market columns:")
    print(", ".join(market_cols))
    print()

    df_feats, feature_names = build_features(df_panel, market_cols, cfg)

    print(f"Feature frame shape (including date): {df_feats.shape}")
    print(f"Number of features: {len(feature_names)}")
    print()

    pipe, states_df = fit_prism_ml_engine(df_feats, feature_names, cfg)
    save_outputs(pipe, feature_names, states_df, cfg)

    print()
    print("Done.")


if __name__ == "__main__":
    main()
