"""
PRISM ML – Random Forest Regime Classifier (Step 1)
---------------------------------------------------

- Uses master_panel.csv as input.
- Creates simple future-return-based "regimes":
    * risk_off  : future 6m SPY return < -5%
    * neutral   : between -5% and +10%
    * risk_on   : future 6m SPY return >= +10%

- Time-based train/test split (80% / 20%)
- Trains RandomForestClassifier
- Prints:
    * class distribution
    * accuracy
    * classification report
    * confusion matrix
    * top feature importances

- Saves model to models/regime_rf.pkl
"""

from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
import joblib


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parents[1]
PANEL_PATH = ROOT_DIR / "data" / "panels" / "master_panel.csv"
MODEL_PATH = ROOT_DIR / "models" / "regime_rf.pkl"


# ---------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------

def load_panel(path: Path) -> pd.DataFrame:
    """
    Load master_panel.csv and ensure we have a proper datetime index.
    Handles the case where 'date' is:
    - a named column, or
    - the first unnamed column (index saved by to_csv).
    """
    if not path.exists():
        raise FileNotFoundError(f"Panel file not found: {path}")

    df = pd.read_csv(path)

    # Try to infer the date column
    if "date" in df.columns:
        date_col = "date"
    elif "index" in df.columns:
        df = df.rename(columns={"index": "date"})
        date_col = "date"
    else:
        # Assume first column is the date
        first_col = df.columns[0]
        df = df.rename(columns={first_col: "date"})
        date_col = "date"

    df["date"] = pd.to_datetime(df[date_col])
    df = df.sort_values("date").set_index("date")

    # Drop any accidental duplicate date column
    if date_col in df.columns:
        df = df.drop(columns=[date_col])

    return df


def select_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Select feature columns from panel.

    For now:
    - Use all columns except ultra-short series (like btc/eth)
      which can cause heavy NaN early in the history.

    You can tune this list later to:
    - focus on macro + cross-asset
    - or restrict to PRISM pillar-specific sets
    """
    drop_cols = {"btc", "eth"}  # crypto: keep out for now
    feature_cols: List[str] = [c for c in df.columns if c not in drop_cols]

    features = df[feature_cols].copy()

    # Forward fill gaps, then drop any remaining NaNs
    features = features.ffill().bfill()
    features = features.dropna()

    return features


# ---------------------------------------------------------------------
# Label engineering – simple proxy regimes
# ---------------------------------------------------------------------

def make_regime_labels(
    df: pd.DataFrame,
    horizon_days: int = 126,
    spy_col: str = "spy",
) -> pd.Series:
    """
    Build simple regime labels from future SPY returns.

    horizon_days ~ 6 trading months (~21 trading days * 6).

    Regime rules:
        fwd_ret < -5%      → risk_off
        -5% to +10%        → neutral
        >= +10%            → risk_on
    """
    if spy_col not in df.columns:
        raise KeyError(f"SPY column '{spy_col}' not found in panel columns: {df.columns.tolist()}")

    spy = df[spy_col].astype(float)
    fwd = spy.shift(-horizon_days)

    fwd_ret = (fwd / spy) - 1.0

    # Regime buckets
    y = pd.Series(index=df.index, dtype="object")
    y[fwd_ret < -0.05] = "risk_off"
    y[(fwd_ret >= -0.05) & (fwd_ret < 0.10)] = "neutral"
    y[fwd_ret >= 0.10] = "risk_on"

    # Drop last horizon_days where fwd_ret is NaN
    y = y.dropna()

    return y


def align_features_labels(
    X: pd.DataFrame,
    y: pd.Series,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Align features and labels on common index.
    """
    common_idx = X.index.intersection(y.index)
    X_aligned = X.loc[common_idx].copy()
    y_aligned = y.loc[common_idx].copy()
    return X_aligned, y_aligned


# ---------------------------------------------------------------------
# Train / test split (time-based)
# ---------------------------------------------------------------------

def time_based_split(
    X: pd.DataFrame,
    y: pd.Series,
    train_frac: float = 0.8,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split chronologically: first train_frac as train, rest as test.
    """
    if not 0.0 < train_frac < 1.0:
        raise ValueError("train_frac must be between 0 and 1")

    n = len(X)
    split_idx = int(n * train_frac)

    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_train = y.iloc[:split_idx]
    y_test = y.iloc[split_idx:]

    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------

def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> RandomForestClassifier:
    """
    Train a reasonably-sized Random Forest classifier.
    """
    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=10,
        min_samples_leaf=5,
        n_jobs=-1,
        random_state=42,
        oob_score=True,
    )
    clf.fit(X_train, y_train)
    return clf


def print_evaluation(
    clf: RandomForestClassifier,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> None:
    """
    Print accuracy, classification report, confusion matrix,
    and top feature importances.
    """
    # Train
    y_pred_train = clf.predict(X_train)
    train_acc = accuracy_score(y_train, y_pred_train)

    # Test
    y_pred_test = clf.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred_test)

    print("\n================= REGIME RF RESULTS =================\n")
    print(f"Train accuracy: {train_acc:.3f}")
    print(f"Test  accuracy: {test_acc:.3f}\n")

    print("Class distribution (train):")
    print(y_train.value_counts())
    print("\nClass distribution (test):")
    print(y_test.value_counts())
    print("\nClassification report (test):")
    print(classification_report(y_test, y_pred_test, digits=3))

    print("Confusion matrix (test):")
    print(confusion_matrix(y_test, y_pred_test, labels=["risk_off", "neutral", "risk_on"]))
    print("Labels order: ['risk_off', 'neutral', 'risk_on']\n")

    # Feature importances
    importances = clf.feature_importances_
    features = np.array(X_train.columns)
    order = np.argsort(importances)[::-1]

    print("Top 20 feature importances:")
    for rank, idx in enumerate(order[:20], start=1):
        print(f"{rank:2d}. {features[idx]:30s}  {importances[idx]:.4f}")

    if hasattr(clf, "oob_score_"):
        print(f"\nOOB score: {clf.oob_score_:.3f}")


# ---------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------

def main() -> None:
    print("============================================")
    print("   PRISM ML – Random Forest Regime Classifier")
    print("============================================\n")

    print(f"Loading panel: {PANEL_PATH}")
    panel = load_panel(PANEL_PATH)
    print(f"Panel shape (raw): {panel.shape}")

    # Build features
    X = select_features(panel)
    print(f"Feature matrix shape after cleaning: {X.shape}")

    # Build labels from future SPY return
    y = make_regime_labels(panel, horizon_days=126, spy_col="spy")
    print(f"Initial label series length: {len(y)}")

    # Align X and y
    X_aligned, y_aligned = align_features_labels(X, y)
    print(f"Aligned shapes: X={X_aligned.shape}, y={y_aligned.shape}")

    # Time split
    X_train, X_test, y_train, y_test = time_based_split(X_aligned, y_aligned, train_frac=0.8)
    print(f"Train length: {len(X_train)}, Test length: {len(X_test)}")

    # Train model
    clf = train_random_forest(X_train, y_train)

    # Evaluate
    print_evaluation(clf, X_train, y_train, X_test, y_test)

    # Save model
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, MODEL_PATH)
    print(f"\nModel saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
