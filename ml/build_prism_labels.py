import pandas as pd
import numpy as np
import os

LABEL_OUT = "data/labels/prism_regimes.csv"


def compute_regime_from_mrf(mrf, neutral_band=0.10):
    """
    Convert MRF numeric signal into categorical regimes.
    """
    if mrf < -neutral_band:
        return "risk_off"
    elif mrf > neutral_band:
        return "risk_on"
    else:
        return "neutral"


def main():
    print("==============================================")
    print("   PRISM LABEL BUILDER – MRF → Regimes")
    print("==============================================")

    panel_path = "data/panels/master_panel.csv"
    print(f"Loading panel: {panel_path}")

    df = pd.read_csv(panel_path, parse_dates=["date"])
    df = df.sort_values("date")

    print("Panel shape:", df.shape)

    # -------- Compute synthetic MRF temporarily --------
    numeric_cols = df.drop(columns=["date"]).columns
    df_norm = (df[numeric_cols] - df[numeric_cols].mean()) / df[numeric_cols].std()
    df["MRF"] = df_norm.mean(axis=1)

    df["regime"] = df["MRF"].apply(compute_regime_from_mrf)
    df = df.dropna(subset=["MRF"])

    os.makedirs("data/labels", exist_ok=True)
    df_out = df[["date", "MRF", "regime"]]
    df_out.to_csv(LABEL_OUT, index=False)

    print("\nSaved:", LABEL_OUT)
    print("Label summary:")
    print(df_out["regime"].value_counts())
    print("\nDone.")


if __name__ == "__main__":
    main()
