"""
PRISM ML – Regime Ensemble Classifier (Random Forest + GBM + Logistic)

This script:
  - Loads the PRISM master panel and regime labels
  - Builds a rich feature matrix (returns, rolling stats, slopes)
  - Cleans NaN / inf values
  - Uses time-series cross-validation
  - Trains an ensemble VotingClassifier:
        * RandomForestClassifier
        * GradientBoostingClassifier
        * LogisticRegression
  - Evaluates on a held-out test set (last 20% of timeline)
  - Saves the trained model and feature list to disk

Run from project root:

    PYTHONPATH=. python ml/regime_classifier_ensemble.py
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
PANEL_PATH = Path("data/panels/master_panel.csv")
LABELS_PATH = Path("data/labels/prism_regimes.csv")

MODEL_PATH = Path("models/regime_ensemble.pkl")
FEATURE_LIST_PATH = Path("models/regime_ensemble_features.txt")


# ---------------------------------------------------------------------
# Data loading & feature engineering
# ---------------------------------------------------------------------
def load_panel_and_labels() -> pd.DataFrame:
    """Load master_panel + prism_regimes and merge on date."""
    print("Loading panel:", PANEL_PATH)
    panel = pd.read_csv(PANEL_PATH, parse_dates=["date"]).sort_values("date")

    print("Loading labels:", LABELS_PATH)
    labels = pd.read_csv(LABELS_PATH, parse_dates=["date"])

    df = panel.merge(labels, on="date", how="inner")
    # Drop rows with missing regime
    df = df.dropna(subset=["regime"]).reset_index(drop=True)

    print("Merged shape:", df.shape)
    print("Date range:", df["date"].min(), "→", df["date"].max())
    print("Regime counts:\n", df["regime"].value_counts())
    print()
    return df


def _add_return_features(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for col in cols:
        out[f"{col}_ret_1"] = df[col].pct_change()
    return out


def _add_rolling_features(df: pd.DataFrame, cols: List[str], windows=(21, 63, 252)) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for col in cols:
        for w in windows:
            out[f"{col}_sma_{w}"] = df[col].rolling(w).mean()
            out[f"{col}_std_{w}"] = df[col].rolling(w).std()
    return out


def _add_slope_features(df: pd.DataFrame, cols: List[str], windows=(21, 63, 252)) -> pd.DataFrame:
    """
    Approximate slope over each window using simple linear regression on
    normalized time index (0..w-1). This gives a rough trend measure.
    """
    out = pd.DataFrame(index=df.index)
    t_cache = {}

    for w in windows:
        # Precompute (t - mean)/sum((t - mean)^2)
        t = np.arange(w, dtype=float)
        t_mean = t.mean()
        denom = np.sum((t - t_mean) ** 2)
        t_cache[w] = (t, t_mean, denom)

    for col in cols:
        series = df[col].values.astype(float)
        for w in windows:
            t, t_mean, denom = t_cache[w]
            slopes = np.full_like(series, np.nan, dtype=float)

            if len(series) >= w:
                for i in range(w - 1, len(series)):
                    window = series[i - w + 1 : i + 1]
                    if np.all(np.isfinite(window)):
                        y = window
                        y_mean = y.mean()
                        num = np.sum((t - t_mean) * (y - y_mean))
                        slopes[i] = num / denom if denom != 0 else 0.0

            out[f"{col}_slope_{w}"] = slopes

    return out


def build_feature_matrix(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Build a rich feature matrix from the merged df.

    Features:
      - Raw levels for each numeric indicator
      - 1-day returns
      - Rolling SMA & STD (21, 63, 252 days)
      - Rolling slopes (21, 63, 252 days)
    """
    # Columns we won't treat as numeric features
    exclude = {"date", "regime"}
    base_cols = [c for c in df.columns if c not in exclude]

    # 1) Base numeric levels
    feats = df[base_cols].copy()

    # 2) Returns
    ret_feats = _add_return_features(df, base_cols)
    feats = pd.concat([feats, ret_feats], axis=1)

    # 3) Rolling stats
    roll_feats = _add_rolling_features(df, base_cols, windows=(21, 63, 252))
    feats = pd.concat([feats, roll_feats], axis=1)

    # 4) Slopes
    slope_feats = _add_slope_features(df, base_cols, windows=(21, 63, 252))
    feats = pd.concat([feats, slope_feats], axis=1)

    # Drop very early rows where rolling windows are not fully formed
    max_lookback = 252
    feats = feats.iloc[max_lookback:].reset_index(drop=True)
    y = df["regime"].iloc[max_lookback:].reset_index(drop=True)

    # Replace inf with NaN (for imputer)
    feats.replace([np.inf, -np.inf], np.nan, inplace=True)

    print("Final feature matrix shape: X={}, y={}".format(feats.shape, y.shape))
    print("Number of features:", feats.shape[1])
    print()
    return feats, y


# ---------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------
def build_ensemble_pipeline(n_estimators_rf: int = 400) -> Pipeline:
    """
    Build a VotingClassifier ensemble wrapped in a preprocessing pipeline.
    Preprocessing:
      - SimpleImputer(median)
      - StandardScaler
    Ensemble:
      - RandomForest (strong nonlinear workhorse)
      - GradientBoosting (smoother nonlinear learner)
      - LogisticRegression (linear baseline)
    """
    rf = RandomForestClassifier(
        n_estimators=n_estimators_rf,
        max_depth=10,
        min_samples_leaf=2,
        n_jobs=-1,
        class_weight="balanced_subsample",
        random_state=42,
    )

    gb = GradientBoostingClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.8,
        random_state=42,
    )

    logit = LogisticRegression(
        max_iter=1000,
        multi_class="multinomial",
        n_jobs=-1,
        class_weight="balanced",
        solver="lbfgs",
        random_state=42,
    )

    ensemble = VotingClassifier(
        estimators=[
            ("rf", rf),
            ("gb", gb),
            ("logit", logit),
        ],
        voting="soft",
        weights=[3.0, 2.0, 1.0],  # tilt toward RF, then GB, then Logit
        n_jobs=-1,
    )

    pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", ensemble),
        ]
    )

    return pipe


# ---------------------------------------------------------------------
# Training + evaluation
# ---------------------------------------------------------------------
def time_series_cv_scores(pipe: Pipeline, X: pd.DataFrame, y: pd.Series, n_splits: int = 5):
    """
    Run TimeSeriesSplit CV and print accuracy stats.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)

    print(f"Running time-series CV ({n_splits} splits)...")
    scores = cross_val_score(pipe, X, y, cv=tscv, scoring="accuracy", n_jobs=-1)
    print("CV scores:", scores)
    print("CV mean : {:.4f}".format(scores.mean()))
    print("CV std  : {:.4f}".format(scores.std()))
    print()
    return scores


def train_and_evaluate():
    # 1) Load data
    df = load_panel_and_labels()

    # 2) Build feature matrix
    X, y = build_feature_matrix(df)

    # 3) Train/test split: last 20% of timeline = test
    n = len(X)
    split_idx = int(n * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    print("Train size:", len(X_train), "rows")
    print("Test  size:", len(X_test), "rows")
    print()

    # 4) Build ensemble pipeline
    pipe = build_ensemble_pipeline(n_estimators_rf=400)

    # 5) Time-series CV on full dataset (for overall stability)
    time_series_cv_scores(pipe, X, y, n_splits=5)

    # 6) Fit on TRAIN set
    print("Fitting final ensemble on TRAIN set...")
    pipe.fit(X_train, y_train)
    print("Training complete.")
    print()

    # 7) Evaluate on TEST set
    print("Evaluating on TEST set (holdout)...")
    y_pred = pipe.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print("Test accuracy: {:.3f}".format(acc))
    print()

    print("Classification report (TEST):")
    print(classification_report(y_test, y_pred))
    print()

    print("Confusion matrix (TEST):")
    print(confusion_matrix(y_test, y_pred, labels=sorted(y.unique())))
    print("Labels order:", sorted(y.unique()))
    print()

    # 8) Save model and feature list
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, MODEL_PATH)
    print("Model saved to:", MODEL_PATH)

    FEATURE_LIST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with FEATURE_LIST_PATH.open("w") as f:
        for col in X.columns:
            f.write(col + "\n")
    print("Feature list saved to:", FEATURE_LIST_PATH)
    print()

    print("Done.")


# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------
if __name__ == "__main__":
    print("============================================")
    print("   PRISM ML – Regime Ensemble Classifier")
    print("============================================")
    print()
    train_and_evaluate()
