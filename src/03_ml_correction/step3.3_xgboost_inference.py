"""
Run the default saved XGBoost EZIEH_noise_ref model (66-feature version, 1-second
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

from _feature_engineering import add_features as _add_features_base, feature_cols

MODEL_PATH = Path("regression/xgboost_noise_model.json")


# ---------------------------------------------------------------------------
# Feature engineering (shared with step3.2, step3.4 and step3.8 via
# _feature_engineering.py; this script is fixed at the 1-second resolution)
# ---------------------------------------------------------------------------

FEATURE_COLS = feature_cols(1)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all 66 anomaly-based features from ctemp and EZIEH.
    All features are causal (only look backward), safe for real-time use.
    Input df must have columns: time, EZIEH, ctemp."""
    return _add_features_base(df, seconds_per_row=1)


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
    parser = argparse.ArgumentParser(description="XGBoost EZIEH_noise_ref inference (66-feature model)")
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
