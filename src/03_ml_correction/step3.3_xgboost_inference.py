"""
Run the best saved XGBoost Bh-noise model (66-feature version, 1-second
resolution) on new data.

This matches the default model trained by step3.2_xgboost_noise_model.py
(--mode filtered --resolution 1sec) with:
  - Rolling anomalies 30s → 12h  (10 timescales)
  - Cumulative (midnight-anchored) anomalies
  - EMA anomalies at 4 half-lives (1800s, 7200s, 21600s, 43200s)
  - Local volatility (rolling std at 5 timescales, both EZIEH and ctemp)
  - Cross-product interactions at 10 timescales
  - Squared non-linear terms for 8 long windows
  Total: 66 features derived from ctemp + EZIEH only.

Model file: regression/xgboost_noise_model.json

Usage
-----
  # Single day CSV:
  python step3.3_xgboost_inference.py regression/predicted/20250501.csv

  # Entire folder, output alongside originals:
  python step3.3_xgboost_inference.py regression/predicted/

  # Folder → different output folder:
  python step3.3_xgboost_inference.py regression/predicted/ --out_dir regression/inferred/
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
    """Compute all 66 anomaly-based features from ctemp and EZIEH.
    All features are causal (only look backward), safe for real-time use.
    Input df must have columns: time, EZIEH, ctemp."""
    df = df.copy().sort_values("time").reset_index(drop=True)

    e = df["EZIEH"]
    c = df["ctemp"]

    # Causal rolling baselines (30s → 12h)
    for w in (30, 300, 900, 1800, 3600, 7200, 14400, 21600, 28800, 43200):
        e_base = e.rolling(w, min_periods=1).mean()
        c_base = c.rolling(w, min_periods=1).mean()
        df[f"EZIEH_anom{w}"] = e - e_base
        df[f"ctemp_anom{w}"] = c - c_base

    # Cumulative-mean baseline (anchored to midnight)
    e_cumul = e.expanding(min_periods=1).mean()
    c_cumul = c.expanding(min_periods=1).mean()
    df["EZIEH_anom_cumul"] = e - e_cumul
    df["ctemp_anom_cumul"] = c - c_cumul

    # Exponential moving averages at key half-lives
    for hl in (1800, 7200, 21600, 43200):
        df[f"EZIEH_anom_ema{hl}"] = e - e.ewm(halflife=hl, adjust=False).mean()
        df[f"ctemp_anom_ema{hl}"] = c - c.ewm(halflife=hl, adjust=False).mean()

    # Local volatility
    df["EZIEH_std30"]    = e.rolling(30,    min_periods=2).std()
    df["EZIEH_std300"]   = e.rolling(300,   min_periods=2).std()
    df["EZIEH_std900"]   = e.rolling(900,   min_periods=2).std()
    df["EZIEH_std3600"]  = e.rolling(3600,  min_periods=2).std()
    df["EZIEH_std14400"] = e.rolling(14400, min_periods=2).std()
    df["ctemp_std30"]    = c.rolling(30,    min_periods=2).std()
    df["ctemp_std300"]   = c.rolling(300,   min_periods=2).std()
    df["ctemp_std900"]   = c.rolling(900,   min_periods=2).std()
    df["ctemp_std3600"]  = c.rolling(3600,  min_periods=2).std()
    df["ctemp_std14400"] = c.rolling(14400, min_periods=2).std()

    # Interactions at matching timescales
    for w in (30, 300, 900, 1800, 3600, 7200, 14400, 21600):
        df[f"anom{w}_inter"] = df[f"EZIEH_anom{w}"] * df[f"ctemp_anom{w}"]
    df["anom_cumul_inter"]    = df["EZIEH_anom_cumul"]    * df["ctemp_anom_cumul"]
    df["anom_ema43200_inter"] = df["EZIEH_anom_ema43200"] * df["ctemp_anom_ema43200"]

    # Squared non-linear terms
    for w in (3600, 7200, 14400, 21600, 28800, 43200):
        df[f"EZIEH_anom{w}_sq"] = df[f"EZIEH_anom{w}"] ** 2
        df[f"ctemp_anom{w}_sq"] = df[f"ctemp_anom{w}"] ** 2
    df["EZIEH_anom_cumul_sq"]    = df["EZIEH_anom_cumul"]    ** 2
    df["EZIEH_anom_ema43200_sq"] = df["EZIEH_anom_ema43200"] ** 2
    df["ctemp_anom_cumul_sq"]    = df["ctemp_anom_cumul"]    ** 2
    df["ctemp_anom_ema43200_sq"] = df["ctemp_anom_ema43200"] ** 2

    return df


FEATURE_COLS = [
    # Rolling anomalies (30s → 12h)
    "EZIEH_anom30",    "EZIEH_anom300",   "EZIEH_anom900",
    "EZIEH_anom1800",  "EZIEH_anom3600",  "EZIEH_anom7200",
    "EZIEH_anom14400", "EZIEH_anom21600", "EZIEH_anom28800", "EZIEH_anom43200",
    "ctemp_anom30",    "ctemp_anom300",   "ctemp_anom900",
    "ctemp_anom1800",  "ctemp_anom3600",  "ctemp_anom7200",
    "ctemp_anom14400", "ctemp_anom21600", "ctemp_anom28800", "ctemp_anom43200",
    # Cumulative (midnight-anchored) anomalies
    "EZIEH_anom_cumul", "ctemp_anom_cumul",
    # EMA anomalies
    "EZIEH_anom_ema1800",  "EZIEH_anom_ema7200",
    "EZIEH_anom_ema21600", "EZIEH_anom_ema43200",
    "ctemp_anom_ema1800",  "ctemp_anom_ema7200",
    "ctemp_anom_ema21600", "ctemp_anom_ema43200",
    # Volatility
    "EZIEH_std30",  "EZIEH_std300",  "EZIEH_std900",
    "EZIEH_std3600","EZIEH_std14400",
    "ctemp_std30",  "ctemp_std300",  "ctemp_std900",
    "ctemp_std3600", "ctemp_std14400",
    # Interactions
    "anom30_inter",    "anom300_inter",   "anom900_inter",
    "anom1800_inter",  "anom3600_inter",  "anom7200_inter",
    "anom14400_inter", "anom21600_inter",
    "anom_cumul_inter", "anom_ema43200_inter",
    # Non-linear
    "EZIEH_anom3600_sq",  "ctemp_anom3600_sq",
    "EZIEH_anom7200_sq",  "ctemp_anom7200_sq",
    "EZIEH_anom14400_sq", "ctemp_anom14400_sq",
    "EZIEH_anom21600_sq", "ctemp_anom21600_sq",
    "EZIEH_anom28800_sq", "ctemp_anom28800_sq",
    "EZIEH_anom43200_sq", "ctemp_anom43200_sq",
    "EZIEH_anom_cumul_sq",    "ctemp_anom_cumul_sq",
    "EZIEH_anom_ema43200_sq", "ctemp_anom_ema43200_sq",
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

    valid = df[FEATURE_COLS].notna().all(axis=1)
    df["Bh_noise_prediction"] = np.nan
    df.loc[valid, "Bh_noise_prediction"] = model.predict(df.loc[valid, FEATURE_COLS].values)

    n_predicted = int(valid.sum())
    n_nan       = n_before - n_predicted

    dest_dir = out_dir if out_dir else csv_path.parent
    dest_dir.mkdir(parents=True, exist_ok=True)
    out_path = dest_dir / csv_path.name
    df.to_csv(out_path, index=False)

    print(f"  {csv_path.name}: {n_predicted:,} predicted  ({n_nan} NaN at start)  -> {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="XGBoost Bh-noise inference (66-feature model)")
    parser.add_argument("input", help="CSV file or folder of CSVs")
    parser.add_argument("--model",   default=str(MODEL_PATH),
                        help=f"Path to model JSON (default: {MODEL_PATH})")
    parser.add_argument("--out_dir", default=None,
                        help="Output folder (default: same folder as input)")
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}\n"
            "Train it first with step3.2_xgboost_noise_model.py"
        )

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
