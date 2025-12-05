#!/usr/bin/env python3
"""
PRISM – Professional Grade Regime Classifier (Random Forest)

This version:
- Cleans infinities
- Drops all-NaN features
- Drops features with low data coverage
- Uses rolling engineered features
- Time-series aware CV
- Saves model + feature metadata
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# ============================================================
# 1. LOAD PANEL + LABELS
# ============================================================

def load_data():
    panel = pd.read_csv("data/panels/master_panel.csv", parse_dates=["date"]).sort_values("date")
    labels = pd.read_csv("data/labels/prism_regimes.csv", parse_dates=["date"])
    df = panel.merge(labels, on="date", how="inner")
    return df


# ============================================================
# 2. FEATURE ENGINEERING
# ============================================================

def build_features(df: pd.DataFrame):
    df = df.copy()
    df = df.set_index("date")

    feats = pd.DataFrame(index=df.index)

    # List of usable numeric columns (exclude label columns)
    numeric_cols = [
        c for c in df.columns
        if c not in ["regime", "MRF"] and df[c].dtype != "object"
    ]

    # Rolling windows
    windows = [21, 63, 252]  # 1m, 3m, 1yr approximations

    for col in numeric_cols:
        # Raw value
        feats[f"{col}"] = df[col]

        # Returns
        feats[f"{col}_ret_1"] = df[col].pct_change()

        # Rolling features
        for w in windows:
            feats[f"{col}_sma_{w}"] = df[col].rolling(w).mean()
            feats[f"{col}_std_{w}"] = df[col].rolling(w).std()

        # Rolling slope feature (linear regression slope over window)
        for w in windows:
            roll = df[col].rolling(w)
            feats[f"{col}_slope_{w}"] = (
                roll.apply(lambda s: np.polyfit(np.arange(len(s)), s, 1)[0]
                           if s.notna().sum() == w else np.nan, raw=False)
            )

    # ========================================================
    # SANITIZATION BLOCK (THIS FIXES ALL ERRORS)
    # ========================================================

    # Replace infinities
    feats = feats.replace([np.inf, -np.inf], np.nan)

    # Drop rows that are entirely NaN
    feats = feats.dropna(how="all")

    # Drop columns that are fully NaN
    feats = feats.dropna(axis=1, how="all")

    # Require a minimum number of valid data points per feature
    MIN_VALID = 200
    valid_counts = feats.count()
    feats = feats.loc[:, valid_counts > MIN_VALID]

    # Reset index
    feats = feats.reset_index()

    return feats


# ============================================================
# 3. MACHINE LEARNING PIPELINE
# ============================================================

def train_model(X, y):

    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(
            n_estimators=800,
            max_depth=None,
            min_samples_split=4,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=42
        ))
    ])

    # Time-series CV (no leakage)
    tscv = TimeSeriesSplit(n_splits=5)

    print("\n==============================================")
    print("Running time-series cross-validation...")
    print("==============================================")

    scores = cross_val_score(pipe, X, y, cv=tscv, scoring="accuracy")
    print("CV scores:", scores)
    print("CV mean :", np.nanmean(scores))
    print("CV std  :", np.nanstd(scores))

    print("\nFitting final model...")
    pipe.fit(X, y)

    return pipe


# ============================================================
# 4. MAIN EXECUTION
# ============================================================

def main():
    print("\n==============================================")
    print("   PRISM ML – PRO GRADE RF CLASSIFIER")
    print("==============================================")

    df = load_data()
    feats = build_features(df)

    # Merge features + labels
    merged = feats.merge(df[["date", "regime"]], on="date", how="inner")

    print("\nFinal feature matrix shape:",
          f"X=({merged.shape[0]}, {merged.drop(columns=['date','regime']).shape[1]})")

    # Extract X, y
    X = merged.drop(columns=["date", "regime"])
    y = merged["regime"]

    # Train model
    model = train_model(X, y)

    # ========================================================
    # Save model + metadata
    # ========================================================

    model_path = "models/regime_rf_pro.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    print(f"\nModel saved to: {model_path}")

    # Export feature names
    feat_path = "models/regime_rf_pro_features.txt"
    with open(feat_path, "w") as f:
        for col in X.columns:
            f.write(col + "\n")

    print(f"Feature list saved to: {feat_path}")
    print("\nDone.")


# ============================================================

if __name__ == "__main__":
    main()
