"""
Survey plots: 6-panel per-day overview for EZIE geomagnetic data.

Panels (shared x-axis):
  1. FRDH                       - FRD ground station horizontal field
  2. EZIEH                      - EZIE satellite horizontal field
  3. ctemp                      - satellite temperature
  4. Estimated Bh noise         - EZIEH minus linear FRDH prediction
  5. ML Predicted Bh noise      - XGBoost output
  6. EZIEH denoised             - EZIEH minus predicted noise

Usage
-----
  python step3.4_survey_plots.py [--model PATH] [--data_dir PATH]
                                 [--out_dir PATH] [--days YYYYMMDD ...]

Defaults
--------
  --model    regression/xgboost_noise_model_5min.json
  --data_dir regression/predicted
  --out_dir  regression/survey_plots
  --days     all available days
"""
from __future__ import annotations

import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from xgboost import XGBRegressor

from _feature_engineering import SECONDS_PER_ROW, add_features, feature_cols

# ---------------------------------------------------------------------------
# Feature engineering (shared with step3.2_xgboost_noise_model.py and
# step3.3_xgboost_inference.py via _feature_engineering.py)
# ---------------------------------------------------------------------------

def _detect_resolution(model_path: Path):
    name = model_path.stem
    if "1min" in name:
        return "1min", "1min"
    if "5min" in name:
        return "5min", "5min"
    return "1sec", None


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

PANEL_CFG = [
    ("FRDH",            "FRDH (nT)",             "#2ca02c"),
    ("EZIEH",           "EZIEH (nT)",             "#1f77b4"),
    ("ctemp",           "ctemp (°C)",              "#8c564b"),
    ("residual",        "Estimated Bh noise (nT)",           "#7f7f7f"),
    ("noise_pred",      "ML Predicted Bh noise (nT)",    "#d62728"),
    ("EZIEH_denoised",  "EZIEH denoised (nT)",     "#9467bd"),
    
]


def plot_day(df: pd.DataFrame, date_str: str, model_label: str, out_path: Path):
    fig, axes = plt.subplots(6, 1, figsize=(16, 18), sharex=True)
    fig.suptitle(f"{date_str}  |  model: {model_label}", fontsize=14, y=0.995)

    t = df["time"]

    for ax, (col, ylabel, color) in zip(axes, PANEL_CFG):
        if col not in df.columns:
            ax.text(0.5, 0.5, f"{col} not available", transform=ax.transAxes,
                    ha="center", va="center", color="gray", fontsize=14)
        else:
            ax.plot(t, df[col], color=color, linewidth=0.8, rasterized=True)
        ax.set_ylabel(ylabel, fontsize=16)
        ax.grid(True, linewidth=0.4, alpha=0.4)
        ax.tick_params(axis="y", labelsize=13)

    day_start = pd.Timestamp(date_str)
    day_end   = day_start + pd.Timedelta(days=1)
    axes[0].set_xlim(day_start, day_end)

    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    axes[-1].xaxis.set_major_locator(mdates.HourLocator(interval=2))
    axes[-1].tick_params(axis="x", labelsize=13)
    axes[-1].set_xlabel("UTC", fontsize=16)

    plt.tight_layout(rect=[0, 0, 1, 0.997])
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",    default="regression/xgboost_noise_model_5min.json")
    ap.add_argument("--data_dir", default="regression/predicted")
    ap.add_argument("--out_dir",  default="regression/survey_plots")
    ap.add_argument("--days",     nargs="*", default=None,
                    help="Specific YYYYMMDD dates to plot (default: all)")
    args = ap.parse_args()

    model_path = Path(args.model)
    data_dir   = Path(args.data_dir)
    out_dir    = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    res_label, resample_rule = _detect_resolution(model_path)
    seconds_per_row = SECONDS_PER_ROW[res_label]
    cols = feature_cols(seconds_per_row)

    print(f"Model  : {model_path}")
    print(f"Res    : {res_label}")
    print(f"Data   : {data_dir}")
    print(f"Output : {out_dir}")

    model = XGBRegressor()
    model.load_model(model_path)
    model_label = model_path.stem

    csv_files = sorted(data_dir.glob("*.csv"))
    if args.days:
        csv_files = [f for f in csv_files if f.stem in args.days]

    print(f"Days to plot: {len(csv_files)}")

    for csv_path in csv_files:
        date_str = csv_path.stem
        try:
            df = pd.read_csv(csv_path, parse_dates=["time"])

            if resample_rule:
                df = (df.set_index("time")
                        .resample(resample_rule).median()
                        .dropna(subset=["EZIEH", "ctemp"])
                        .reset_index())

            df = df.dropna(subset=["EZIEH", "ctemp"]).reset_index(drop=True)
            if len(df) < 5:
                print(f"  {date_str}: too few rows, skipping")
                continue

            df_feat = add_features(df, seconds_per_row)
            missing = [c for c in cols if c not in df_feat.columns]
            if missing:
                print(f"  {date_str}: missing features {missing[:3]}…, skipping")
                continue

            valid = df_feat[cols].notna().all(axis=1)
            df_feat = df_feat[valid].copy()

            df_feat["noise_pred"]     = model.predict(df_feat[cols].values)
            df_feat["EZIEH_denoised"] = df_feat["EZIEH"] - df_feat["noise_pred"]

            out_path = out_dir / f"{date_str}.png"
            plot_day(df_feat, date_str, model_label, out_path)
            print(f"  {date_str}: saved -> {out_path.name}")

        except Exception as exc:
            print(f"  {date_str}: ERROR – {exc}")

    print(f"\nDone. Plots in {out_dir}/")


if __name__ == "__main__":
    main()
