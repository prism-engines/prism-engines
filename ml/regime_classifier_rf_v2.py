import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, TimeSeriesSplit

LABEL_PATH = "data/labels/prism_regimes.csv"
PANEL_PATH = "data/panels/master_panel.csv"
MODEL_OUT = "models/regime_rf_v2.pkl"

N_LAGS = 5   # Number of lagged days
WINDOW = 10  # Rolling window features


def add_lags(df, columns, n_lags):
    for col in columns:
        for i in range(1, n_lags + 1):
            df[f"{col}_lag{i}"] = df[col].shift(i)
    return df


def add_rolling(df, columns, window):
    for col in columns:
        df[f"{col}_roll_mean"] = df[col].rolling(window).mean()
        df[f"{col}_roll_std"] = df[col].rolling(window).std()
    return df


def main():
    print("==============================================")
    print("     PRISM ML – Random Forest Classifier V2")
    print("==============================================")

    print("Loading panel:", PANEL_PATH)
    df = pd.read_csv(PANEL_PATH, parse_dates=["date"]).sort_values("date")

    print("Loading labels:", LABEL_PATH)
    labels = pd.read_csv(LABEL_PATH, parse_dates=["date"])
    df = df.merge(labels, on="date", how="inner")
    print("Merged shape:", df.shape)

    feature_cols = df.drop(columns=["date", "regime"]).columns

    print("Building lag/rolling features...")
    df = add_lags(df, feature_cols, N_LAGS)
    df = add_rolling(df, feature_cols, WINDOW)

    df = df.dropna()
    print("Final cleaned shape:", df.shape)

    X = df.drop(columns=["date", "regime"])
    y = df["regime"]

    print("Feature count:", X.shape[1])
    print("Sample count:", len(X))

    print("\nTime-series split...")
    tscv = TimeSeriesSplit(n_splits=5)

    acc_scores = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        print(f"\n=== Fold {fold+1} ===")

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        clf = RandomForestClassifier(
            n_estimators=400,
            max_depth=12,
            class_weight="balanced_subsample",
            max_features="sqrt",
            n_jobs=-1,
            oob_score=False,
            random_state=42,
        )

        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        acc = (preds == y_test).mean()
        acc_scores.append(acc)

        print(f"Fold accuracy: {acc:.3f}")

    print("\n==============================================")
    print("Cross-validation accuracy:", np.mean(acc_scores))
    print("Std deviation:", np.std(acc_scores))

    # Final training on all data
    final_model = RandomForestClassifier(
        n_estimators=400,
        max_depth=12,
        class_weight="balanced_subsample",
        max_features="sqrt",
        n_jobs=-1,
        random_state=42,
    )

    final_model.fit(X, y)
    print("\nTraining complete!")

    os.makedirs("models", exist_ok=True)
    import joblib
    joblib.dump(final_model, MODEL_OUT)
    print("\nModel saved to:", MODEL_OUT)

    # Feature importances
    importances = pd.Series(final_model.feature_importances_, index=X.columns)
    top = importances.sort_values(ascending=False).head(25)

    print("\nTop 25 features:")
    for i, (name, val) in enumerate(top.items(), start=1):
        print(f"{i:2}. {name:35s} {val:.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
