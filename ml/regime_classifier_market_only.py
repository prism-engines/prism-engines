import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
import joblib
import os

print("\n==============================================")
print(" PRISM ML – MARKET-ONLY ENSEMBLE CLASSIFIER")
print("==============================================\n")

PANEL_PATH = "data/panels/master_panel.csv"
LABEL_PATH = "data/labels/prism_regimes.csv"

# ECONOMIC COLUMNS TO REMOVE
ECON_COLS = [
    'dgs10','dgs2','dgs3mo','t10y2y','t10y3m','cpi','cpi_core','ppi','unrate',
    'payrolls','industrial_production','housing_starts','permits','m2',
    'fed_balance_sheet','anfci','nfci'
]

# LOAD PANEL
df = pd.read_csv(PANEL_PATH, parse_dates=['date']).sort_values('date')
labels = pd.read_csv(LABEL_PATH, parse_dates=['date'])

# Merge regimes into panel
df = df.merge(labels[['date','regime']], on='date', how='inner')

# Drop economic indicators
market_df = df.drop(columns=ECON_COLS, errors='ignore')

print(f"Panel rows: {len(market_df)}")
print(f"Market-only columns: {len(market_df.columns)}")

# Build feature-engineered dataset
FEATURE_WINDOWS = [21, 63, 252]

feats = pd.DataFrame()
feats["date"] = market_df["date"]

market_cols = [c for c in market_df.columns if c not in ['date','regime']]

for col in market_cols:
    s = market_df[col].astype(float)

    feats[f"{col}"] = s
    feats[f"{col}_ret_1"] = s.pct_change()

    for w in FEATURE_WINDOWS:
        feats[f"{col}_sma_{w}"] = s.rolling(w).mean()
        feats[f"{col}_std_{w}"] = s.rolling(w).std()
        feats[f"{col}_slope_{w}"] = (
            s.rolling(w).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0], raw=True)
        )

print(f"\nFeature matrix shape BEFORE cleaning: {feats.shape}")

# Drop date
regimes = market_df["regime"]
feats = feats.drop(columns=["date"])

# Remove all-zero or all-NaN columns
feats = feats.dropna(axis=1, how="all")
feats = feats.loc[:, (feats != 0).any(axis=0)]

X = feats.values
y = regimes.values

print(f"Final X shape: {X.shape}, y={y.shape}")

# Train-test split
split_idx = int(len(X) * 0.80)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# Build ensemble pipeline
pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("model", VotingClassifier(
        estimators=[
            ("rf", RandomForestClassifier(n_estimators=600, max_depth=10)),
            ("gb", GradientBoostingClassifier())
        ],
        voting="soft"
    ))
])

# Time-series CV
tscv = TimeSeriesSplit(n_splits=5)
cv_scores = cross_val_score(pipe, X_train, y_train, cv=tscv)

print("\nCV scores:", cv_scores)
print("CV mean :", np.mean(cv_scores))
print("CV std  :", np.std(cv_scores))

print("\nFitting final model on train set...")
pipe.fit(X_train, y_train)

print("\nTest accuracy:", pipe.score(X_test, y_test))

from sklearn.metrics import classification_report, confusion_matrix
print("\nClassification Report:\n")
print(classification_report(y_test, pipe.predict(X_test)))

print("Confusion matrix:")
print(confusion_matrix(y_test, pipe.predict(X_test)))

# Save model and features
os.makedirs("models", exist_ok=True)

joblib.dump(pipe, "models/regime_ensemble_market.pkl")
with open("models/regime_ensemble_market_features.txt", "w") as f:
    for c in feats.columns:
        f.write(c + "\n")

print("\nModel saved to: models/regime_ensemble_market.pkl")
print("Feature list saved to: models/regime_ensemble_market_features.txt")
print("\nDone.")
