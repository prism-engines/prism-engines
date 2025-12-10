#!/usr/bin/env python
"""
PRISM – Market-Only Ensemble (ETF Era, Late Starts Removed)

- Uses only market indicators (no economics)
- Restricts to ETF era (post-2004 with a buffer)
- Removes late-start tickers that don't have enough history
- Cleans inf / -inf values before feeding to sklearn
- Builds a reasonably light ensemble for Chromebook, heavier later on Mac
"""

import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib


# -------------------------------------------------------------------
# Paths & constants
# -------------------------------------------------------------------
DATA_DIR = Path("data")
PANEL_PATH = DATA_DIR / "panels" / "master_panel.csv"
LABELS_PATH = DATA_DIR / "labels" / "prism_regimes.csv"

MODEL_PATH = Path("models") / "regime_ensemble_post_etf.pkl"
FEATURE_LIST_PATH = Path("models") / "regime_ensemble_post_etf_features.txt"

# Start of ETF era + small buffer to let rolling windows burn in
ETF_START = pd.Timestamp("2004-01-01")

# Require at least this many non-NaN observations per market column
MIN_HISTORY_DAYS = 252 * 3  # ~3 years

# Market-only columns we *might* use (we'll drop any that don't exist or lack history)
CANDIDATE_MARKET_COLS = [
    "spy", "qqq", "iwm", "dia",
    "vea", "vwo", "ewj", "fez", "ewz", "fxi", "inda",
    "xlk", "xlf", "xle", "xlv", "xli", "xlu", "xlp", "xly", "xlb", "xlre",
    "mtum", "vlue", "qual", "smlv", "usmv",
    "tlt", "ief", "shy", "hyg", "lqd", "bnd",
    "gld", "slv", "uso", "dbc",
    # if you later want these, you can re-enable them,
    # but for now they're late-starts / messy:
    # "vix", "dxy", "btc", "eth",
]

# Rolling windows for technical features
WINDOWS = [21, 63, 126, 252]  # 1m, 3m, 6m, 1y approx


def build_market_features(df: pd.DataFrame, market_cols):
    """
    Build return / moving-average / volatility / slope features
    for a set of market columns.

    Returns:
        feats: DataFrame of engineered features (no 'date' / 'regime')
    """
    feats = pd.DataFrame(index=df.index)

    for col in market_cols:
        s = df[col].astype(float)

        # Raw level
        feats[col] = s

        # 1-day return
        feats[f"{col}_ret_1"] = s.pct_change()

        # rolling stats
        for w in WINDOWS:
            roll = s.rolling(w)
            feats[f"{col}_sma_{w}"] = roll.mean()
            feats[f"{col}_std_{w}"] = roll.std()

            # Simple slope: (price_t - price_{t-w}) / w
            feats[f"{col}_slope_{w}"] = (s - s.shift(w)) / float(w)

    return feats


def main():
    print("============================================")
    print(" PRISM — MARKET-ONLY ENSEMBLE (POST ETF ERA)")
    print("============================================\n")

    # --------------------------------------------------------------
    # 1. Load panel + labels and merge
    # --------------------------------------------------------------
    print(f"Loading panel: {PANEL_PATH}")
    panel = pd.read_csv(PANEL_PATH, parse_dates=["date"]).sort_values("date")

    print(f"Loading labels: {LABELS_PATH}")
    labels = pd.read_csv(LABELS_PATH, parse_dates=["date"])

    df = panel.merge(labels, on="date", how="inner")
    print(f"Merged shape: {df.shape}")
    if "regime" not in df.columns:
        raise ValueError("Expected 'regime' column in labels but did not find it.")

    # --------------------------------------------------------------
    # 2. Filter to ETF era and remove duplicate dates
    # --------------------------------------------------------------
    print(f"\nApplying ETF-era filter from {ETF_START.date()} onward...")
    df = df[df["date"] >= ETF_START].copy()

    # Drop duplicate dates (keep last)
    before_dup = len(df)
    df = df.sort_values("date").drop_duplicates(subset="date", keep="last").reset_index(drop=True)
    after_dup = len(df)
    if after_dup != before_dup:
        print(f"Removed {before_dup - after_dup} duplicate date rows.")
    print(f"Post ETF filter shape: {df.shape}")

    # --------------------------------------------------------------
    # 3. Pick market-only columns that exist
    # --------------------------------------------------------------
    available_cols = [c for c in CANDIDATE_MARKET_COLS if c in df.columns]
    print(f"\nMarket columns available in panel ({len(available_cols)}): {available_cols}")

    if not available_cols:
        raise ValueError("No candidate market columns found in panel. Check column names.")

    # --------------------------------------------------------------
    # 4. Remove late-start tickers (not enough history in ETF era)
    # --------------------------------------------------------------
    market_block = df[available_cols]
    counts = market_block.notna().sum()

    good_cols = [c for c in available_cols if counts[c] >= MIN_HISTORY_DAYS]
    dropped_cols = [c for c in available_cols if c not in good_cols]

    print(f"\nHistory counts (ETF era) per column (top 10):")
    print(counts.sort_values(ascending=False).head(10))

    print(f"\nKeeping {len(good_cols)} market columns with >= {MIN_HISTORY_DAYS} non-NaN days:")
    print(good_cols)

    if dropped_cols:
        print(f"\nDropping late-start / sparse columns ({len(dropped_cols)}):")
        print(dropped_cols)

    if not good_cols:
        raise ValueError("All market columns were dropped by history filter; relax MIN_HISTORY_DAYS.")

    # Subset to good market columns + regime
    work = df[["date", "regime"] + good_cols].copy()

    # Drop rows with missing regime
    before_regime = len(work)
    work = work.dropna(subset=["regime"]).reset_index(drop=True)
    print(f"\nDropped {before_regime - len(work)} rows with NaN regime. Remaining: {len(work)}")

    # --------------------------------------------------------------
    # 5. Build features
    # --------------------------------------------------------------
    print("\nBuilding market-only technical features...")
    feats = build_market_features(work, good_cols)

    # Combine into X/y
    X = feats.copy()
    y = work["regime"].copy()

    # --------------------------------------------------------------
    # 6. Clean infinities and all-NaN rows
    # --------------------------------------------------------------
    print("\nCleaning infinities and all-NaN rows...")
    # Replace +/- inf with NaN so the imputer can handle them
    X = X.replace([np.inf, -np.inf], np.nan)

    # Drop rows where *all* features are NaN
    all_nan_mask = X.isna().all(axis=1)
    dropped_all_nan = int(all_nan_mask.sum())
    if dropped_all_nan > 0:
        print(f"Dropping {dropped_all_nan} rows where all features are NaN.")
        X = X[~all_nan_mask]
        y = y[~all_nan_mask]

    # Final shapes
    print(f"\nFinal feature matrix shape: X={X.shape}, y={y.shape}")
    print(f"Number of features: {X.shape[1]}\n")

    if len(X) < 100:
        raise ValueError("Too few rows after cleaning; ETF-era + filters left < 100 samples.")

    # --------------------------------------------------------------
    # 7. Train/test split (time-respecting)
    # --------------------------------------------------------------
    # We preserve temporal order: first 80% = train, last 20% = test
    n = len(X)
    split_idx = int(n * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    print(f"Train size: {len(X_train)} rows")
    print(f"Test  size: {len(X_test)} rows\n")

    # --------------------------------------------------------------
    # 8. Build ensemble pipeline
    # --------------------------------------------------------------
    rf = RandomForestClassifier(
        n_estimators=400,
        max_depth=8,
        min_samples_leaf=20,
        n_jobs=-1,
        random_state=42,
    )

    gb = GradientBoostingClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        random_state=42,
    )

    ensemble = VotingClassifier(
        estimators=[("rf", rf), ("gb", gb)],
        voting="soft",
        n_jobs=-1,
    )

    pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", ensemble),
        ]
    )

    print("Training ensemble model...")
    pipe.fit(X_train, y_train)

    # --------------------------------------------------------------
    # 9. Evaluate on test set
    # --------------------------------------------------------------
    print("\nEvaluating on TEST set...")
    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nTest accuracy: {acc:.3f}\n")

    print("Classification report (TEST):")
    print(classification_report(y_test, y_pred))

    print("Confusion matrix (TEST):")
    cm = confusion_matrix(y_test, y_pred, labels=["neutral", "risk_off", "risk_on"])
    print(cm)
    print("Labels order: ['neutral', 'risk_off', 'risk_on']\n")

    # --------------------------------------------------------------
    # 10. Save model + feature list
    # --------------------------------------------------------------
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(pipe, MODEL_PATH)
    print(f"Model saved to: {MODEL_PATH}")

    with open(FEATURE_LIST_PATH, "w") as f:
        for col in X.columns:
            f.write(col + "\n")
    print(f"Feature list saved to: {FEATURE_LIST_PATH}\n")

    print("Done.")


if __name__ == "__main__":
    main()
