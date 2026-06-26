#!/usr/bin/env python3
"""
Run the saved XGBoost Bh-noise model on new data — no retraining needed.

Input:  a prediction CSV (or any CSV with columns: time, EZIEH, ctemp)
Output: same CSV with an added column  Bh_noise_prediction  (nT)

Usage
-----
  # Run on a single day CSV:
  python step4.2_xgboost_inference.py regression/predicted/20250501.csv

  # Run on all CSVs in a folder and write results alongside originals:
  python step4.2_xgboost_inference.py regression/predicted/

  # Run on a folder, write results to a different output folder:
  python step4.2_xgboost_inference.py regression/predicted/ --out_dir regression/inferred/
"""
from __future__ import annotations

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from xgboost import XGBRegressor

MODEL_PATH = Path("regression/xgboost_noise_model.json")


# ---------------------------------------------------------------------------
# Feature engineering  (must match step4.1 exactly)
# ---------------------------------------------------------------------------

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all 47 anomaly-based features from ctemp and EZIEH.
    Input df must have columns: time, EZIEH, ctemp (any extra columns are kept).
    Features are causal (only look backward), so safe for real-time use."""
    df = df.copy().sort_values("time").reset_index(drop=True)

    e = df["EZIEH"]
    c = df["ctemp"]

    # Anomalies: deviation from causal rolling baseline at each timescale
    for w in (30, 300, 900, 1800, 3600, 7200, 14400, 21600, 28800, 43200):
        e_base = e.rolling(w, min_periods=1).mean()
        c_base = c.rolling(w, min_periods=1).mean()
        df[f"EZIEH_anom{w}"] = e - e_base
        df[f"ctemp_anom{w}"] = c - c_base

    # Local volatility (rolling std)
    df["EZIEH_std30"]    = e.rolling(30,    min_periods=2).std()
    df["EZIEH_std300"]   = e.rolling(300,   min_periods=2).std()
    df["EZIEH_std900"]   = e.rolling(900,   min_periods=2).std()
    df["EZIEH_std3600"]  = e.rolling(3600,  min_periods=2).std()
    df["EZIEH_std14400"] = e.rolling(14400, min_periods=2).std()
    df["ctemp_std30"]    = c.rolling(30,    min_periods=2).std()
    df["ctemp_std300"]   = c.rolling(300,   min_periods=2).std()
    df["ctemp_std3600"]  = c.rolling(3600,  min_periods=2).std()
    df["ctemp_std14400"] = c.rolling(14400, min_periods=2).std()

    # Cross-product interactions at matching timescales
    for w in (30, 300, 900, 1800, 3600, 7200, 14400, 21600):
        df[f"anom{w}_inter"] = df[f"EZIEH_anom{w}"] * df[f"ctemp_anom{w}"]

    # Squared non-linear terms for key long windows
    for w in (3600, 7200, 14400, 21600, 28800):
        df[f"EZIEH_anom{w}_sq"] = df[f"EZIEH_anom{w}"] ** 2
        df[f"ctemp_anom{w}_sq"] = df[f"ctemp_anom{w}"] ** 2

    return df


FEATURE_COLS = [
    "EZIEH_anom30",    "EZIEH_anom300",   "EZIEH_anom900",
    "EZIEH_anom1800",  "EZIEH_anom3600",  "EZIEH_anom7200",
    "EZIEH_anom14400", "EZIEH_anom21600", "EZIEH_anom28800", "EZIEH_anom43200",
    "ctemp_anom30",    "ctemp_anom300",   "ctemp_anom900",
    "ctemp_anom1800",  "ctemp_anom3600",  "ctemp_anom7200",
    "ctemp_anom14400", "ctemp_anom21600", "ctemp_anom28800", "ctemp_anom43200",
    "EZIEH_std30",     "EZIEH_std300",    "EZIEH_std900",
    "EZIEH_std3600",   "EZIEH_std14400",
    "ctemp_std30",     "ctemp_std300",    "ctemp_std3600",   "ctemp_std14400",
    "anom30_inter",    "anom300_inter",   "anom900_inter",
    "anom1800_inter",  "anom3600_inter",  "anom7200_inter",
    "anom14400_inter", "anom21600_inter",
    "EZIEH_anom3600_sq",  "ctemp_anom3600_sq",
    "EZIEH_anom7200_sq",  "ctemp_anom7200_sq",
    "EZIEH_anom14400_sq", "ctemp_anom14400_sq",
    "EZIEH_anom21600_sq", "ctemp_anom21600_sq",
    "EZIEH_anom28800_sq", "ctemp_anom28800_sq",
]


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_inference(csv_path: Path, model: XGBRegressor, out_dir: Path | None) -> Path:
    df = pd.read_csv(csv_path, parse_dates=["time"])

    if "EZIEH" not in df.columns or "ctemp" not in df.columns:
        raise ValueError(f"{csv_path.name}: missing required columns EZIEH and/or ctemp")

    n_before = len(df)
    df = add_features(df)

    # Rows where features are NaN (start of day, rolling std not yet computable) get NaN prediction
    valid = df[FEATURE_COLS].notna().all(axis=1)
    df["Bh_noise_prediction"] = np.nan
    df.loc[valid, "Bh_noise_prediction"] = model.predict(df.loc[valid, FEATURE_COLS].values)

    n_predicted = int(valid.sum())
    n_nan       = n_before - n_predicted

    dest_dir  = out_dir if out_dir else csv_path.parent
    dest_dir.mkdir(parents=True, exist_ok=True)
    out_path  = dest_dir / csv_path.name
    df.to_csv(out_path, index=False)

    print(f"  {csv_path.name}: {n_predicted:,} predicted  ({n_nan} NaN at start)  -> {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="XGBoost Bh-noise inference")
    parser.add_argument("input", help="CSV file or folder of CSVs")
    parser.add_argument("--model",   default=str(MODEL_PATH),
                        help=f"Path to model JSON (default: {MODEL_PATH})")
    parser.add_argument("--out_dir", default=None,
                        help="Output folder (default: same folder as input)")
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}\nTrain it first with step4.1_xgboost_noise_model.py")

    print(f"Loading model from {model_path} ...")
    model = XGBRegressor()
    model.load_model(model_path)

    input_path = Path(args.input)
    out_dir    = Path(args.out_dir) if args.out_dir else None

    if input_path.is_dir():
        csv_files = sorted(input_path.glob("*.csv"))
        print(f"Found {len(csv_files)} CSVs in {input_path}")
        for f in csv_files:
            run_inference(f, model, out_dir)
    else:
        run_inference(input_path, model, out_dir)

    print("Done.")


if __name__ == "__main__":
    main()
