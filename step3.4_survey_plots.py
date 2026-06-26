#!/usr/bin/env python3
"""
Survey plots: 6-panel per-day overview for EZIE geomagnetic data.

Panels (shared x-axis):
  1. FRDH            - FRD ground station horizontal field
  2. EZIEH           - EZIE satellite horizontal field
  3. residual        - EZIEH minus linear FRDH prediction
  4. predicted noise - XGBoost output
  5. EZIEH denoised  - EZIEH minus predicted noise
  6. ctemp           - satellite temperature

Usage
-----
  python step5_survey_plots.py [--model PATH] [--data_dir PATH]
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

# ---------------------------------------------------------------------------
# Feature engineering (one function per resolution)
# ---------------------------------------------------------------------------

def _add_features_1sec(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values("time").reset_index(drop=True)
    e = df["EZIEH"]; c = df["ctemp"]
    for w in (30, 300, 900, 1800, 3600, 7200, 14400, 21600, 28800, 43200):
        df[f"EZIEH_anom{w}"] = e - e.rolling(w, min_periods=1).mean()
        df[f"ctemp_anom{w}"] = c - c.rolling(w, min_periods=1).mean()
    df["EZIEH_anom_cumul"] = e - e.expanding(min_periods=1).mean()
    df["ctemp_anom_cumul"] = c - c.expanding(min_periods=1).mean()
    for hl in (1800, 7200, 21600, 43200):
        df[f"EZIEH_anom_ema{hl}"] = e - e.ewm(halflife=hl, adjust=False).mean()
        df[f"ctemp_anom_ema{hl}"] = c - c.ewm(halflife=hl, adjust=False).mean()
    for w in (30, 300, 900, 3600, 14400):
        df[f"EZIEH_std{w}"] = e.rolling(w, min_periods=2).std()
    for w in (30, 300, 3600, 14400):
        df[f"ctemp_std{w}"] = c.rolling(w, min_periods=2).std()
    for w in (30, 300, 900, 1800, 3600, 7200, 14400, 21600):
        df[f"anom{w}_inter"] = df[f"EZIEH_anom{w}"] * df[f"ctemp_anom{w}"]
    df["anom_cumul_inter"]    = df["EZIEH_anom_cumul"]    * df["ctemp_anom_cumul"]
    df["anom_ema43200_inter"] = df["EZIEH_anom_ema43200"] * df["ctemp_anom_ema43200"]
    for w in (3600, 7200, 14400, 21600, 28800, 43200):
        df[f"EZIEH_anom{w}_sq"] = df[f"EZIEH_anom{w}"] ** 2
        df[f"ctemp_anom{w}_sq"] = df[f"ctemp_anom{w}"] ** 2
    df["EZIEH_anom_cumul_sq"]    = df["EZIEH_anom_cumul"]    ** 2
    df["EZIEH_anom_ema43200_sq"] = df["EZIEH_anom_ema43200"] ** 2
    df["ctemp_anom_cumul_sq"]    = df["ctemp_anom_cumul"]    ** 2
    df["ctemp_anom_ema43200_sq"] = df["ctemp_anom_ema43200"] ** 2
    return df


_W1  = {s: max(1, s // 60)  for s in (300, 900, 1800, 3600, 7200, 14400, 21600, 28800, 43200)}
_HL1 = {s: max(1, s // 60)  for s in (1800, 7200, 21600, 43200)}
_SW1 = {s: max(2, s // 60)  for s in (300, 900, 3600, 14400)}

def _add_features_1min(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values("time").reset_index(drop=True)
    e = df["EZIEH"]; c = df["ctemp"]
    for w_s, w_r in _W1.items():
        df[f"EZIEH_anom{w_s}"] = e - e.rolling(w_r, min_periods=1).mean()
        df[f"ctemp_anom{w_s}"] = c - c.rolling(w_r, min_periods=1).mean()
    df["EZIEH_anom_cumul"] = e - e.expanding(min_periods=1).mean()
    df["ctemp_anom_cumul"] = c - c.expanding(min_periods=1).mean()
    for hl_s, hl_r in _HL1.items():
        df[f"EZIEH_anom_ema{hl_s}"] = e - e.ewm(halflife=hl_r, adjust=False).mean()
        df[f"ctemp_anom_ema{hl_s}"] = c - c.ewm(halflife=hl_r, adjust=False).mean()
    for w_s, w_r in _SW1.items():
        df[f"EZIEH_std{w_s}"] = e.rolling(w_r, min_periods=2).std()
        df[f"ctemp_std{w_s}"] = c.rolling(w_r, min_periods=2).std()
    for w_s in (300, 900, 1800, 3600, 7200, 14400, 21600):
        df[f"anom{w_s}_inter"] = df[f"EZIEH_anom{w_s}"] * df[f"ctemp_anom{w_s}"]
    df["anom_cumul_inter"]    = df["EZIEH_anom_cumul"]    * df["ctemp_anom_cumul"]
    df["anom_ema43200_inter"] = df["EZIEH_anom_ema43200"] * df["ctemp_anom_ema43200"]
    for w_s in (3600, 7200, 14400, 21600, 28800, 43200):
        df[f"EZIEH_anom{w_s}_sq"] = df[f"EZIEH_anom{w_s}"] ** 2
        df[f"ctemp_anom{w_s}_sq"] = df[f"ctemp_anom{w_s}"] ** 2
    df["EZIEH_anom_cumul_sq"]    = df["EZIEH_anom_cumul"]    ** 2
    df["EZIEH_anom_ema43200_sq"] = df["EZIEH_anom_ema43200"] ** 2
    df["ctemp_anom_cumul_sq"]    = df["ctemp_anom_cumul"]    ** 2
    df["ctemp_anom_ema43200_sq"] = df["ctemp_anom_ema43200"] ** 2
    return df


_W5  = {s: max(1, s // 300) for s in (900, 1800, 3600, 7200, 14400, 21600, 28800, 43200)}
_HL5 = {s: max(1, s // 300) for s in (1800, 7200, 21600, 43200)}
_SW5 = {s: max(2, s // 300) for s in (900, 3600, 14400)}

def _add_features_5min(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values("time").reset_index(drop=True)
    e = df["EZIEH"]; c = df["ctemp"]
    for w_s, w_r in _W5.items():
        df[f"EZIEH_anom{w_s}"] = e - e.rolling(w_r, min_periods=1).mean()
        df[f"ctemp_anom{w_s}"] = c - c.rolling(w_r, min_periods=1).mean()
    df["EZIEH_anom_cumul"] = e - e.expanding(min_periods=1).mean()
    df["ctemp_anom_cumul"] = c - c.expanding(min_periods=1).mean()
    for hl_s, hl_r in _HL5.items():
        df[f"EZIEH_anom_ema{hl_s}"] = e - e.ewm(halflife=hl_r, adjust=False).mean()
        df[f"ctemp_anom_ema{hl_s}"] = c - c.ewm(halflife=hl_r, adjust=False).mean()
    for w_s, w_r in _SW5.items():
        df[f"EZIEH_std{w_s}"] = e.rolling(w_r, min_periods=2).std()
        df[f"ctemp_std{w_s}"] = c.rolling(w_r, min_periods=2).std()
    for w_s in (900, 1800, 3600, 7200, 14400, 21600):
        df[f"anom{w_s}_inter"] = df[f"EZIEH_anom{w_s}"] * df[f"ctemp_anom{w_s}"]
    df["anom_cumul_inter"]    = df["EZIEH_anom_cumul"]    * df["ctemp_anom_cumul"]
    df["anom_ema43200_inter"] = df["EZIEH_anom_ema43200"] * df["ctemp_anom_ema43200"]
    for w_s in (3600, 7200, 14400, 21600, 28800, 43200):
        df[f"EZIEH_anom{w_s}_sq"] = df[f"EZIEH_anom{w_s}"] ** 2
        df[f"ctemp_anom{w_s}_sq"] = df[f"ctemp_anom{w_s}"] ** 2
    df["EZIEH_anom_cumul_sq"]    = df["EZIEH_anom_cumul"]    ** 2
    df["EZIEH_anom_ema43200_sq"] = df["EZIEH_anom_ema43200"] ** 2
    df["ctemp_anom_cumul_sq"]    = df["ctemp_anom_cumul"]    ** 2
    df["ctemp_anom_ema43200_sq"] = df["ctemp_anom_ema43200"] ** 2
    return df


# Feature lists per resolution
FEATURES_1SEC = [
    "EZIEH_anom30","EZIEH_anom300","EZIEH_anom900","EZIEH_anom1800","EZIEH_anom3600",
    "EZIEH_anom7200","EZIEH_anom14400","EZIEH_anom21600","EZIEH_anom28800","EZIEH_anom43200",
    "ctemp_anom30","ctemp_anom300","ctemp_anom900","ctemp_anom1800","ctemp_anom3600",
    "ctemp_anom7200","ctemp_anom14400","ctemp_anom21600","ctemp_anom28800","ctemp_anom43200",
    "EZIEH_anom_cumul","ctemp_anom_cumul",
    "EZIEH_anom_ema1800","EZIEH_anom_ema7200","EZIEH_anom_ema21600","EZIEH_anom_ema43200",
    "ctemp_anom_ema1800","ctemp_anom_ema7200","ctemp_anom_ema21600","ctemp_anom_ema43200",
    "EZIEH_std30","EZIEH_std300","EZIEH_std900","EZIEH_std3600","EZIEH_std14400",
    "ctemp_std30","ctemp_std300","ctemp_std3600","ctemp_std14400",
    "anom30_inter","anom300_inter","anom900_inter","anom1800_inter","anom3600_inter",
    "anom7200_inter","anom14400_inter","anom21600_inter",
    "anom_cumul_inter","anom_ema43200_inter",
    "EZIEH_anom3600_sq","ctemp_anom3600_sq","EZIEH_anom7200_sq","ctemp_anom7200_sq",
    "EZIEH_anom14400_sq","ctemp_anom14400_sq","EZIEH_anom21600_sq","ctemp_anom21600_sq",
    "EZIEH_anom28800_sq","ctemp_anom28800_sq","EZIEH_anom43200_sq","ctemp_anom43200_sq",
    "EZIEH_anom_cumul_sq","ctemp_anom_cumul_sq","EZIEH_anom_ema43200_sq","ctemp_anom_ema43200_sq",
]

FEATURES_1MIN = [
    "EZIEH_anom300","EZIEH_anom900","EZIEH_anom1800","EZIEH_anom3600","EZIEH_anom7200",
    "EZIEH_anom14400","EZIEH_anom21600","EZIEH_anom28800","EZIEH_anom43200",
    "ctemp_anom300","ctemp_anom900","ctemp_anom1800","ctemp_anom3600","ctemp_anom7200",
    "ctemp_anom14400","ctemp_anom21600","ctemp_anom28800","ctemp_anom43200",
    "EZIEH_anom_cumul","ctemp_anom_cumul",
    "EZIEH_anom_ema1800","EZIEH_anom_ema7200","EZIEH_anom_ema21600","EZIEH_anom_ema43200",
    "ctemp_anom_ema1800","ctemp_anom_ema7200","ctemp_anom_ema21600","ctemp_anom_ema43200",
    "EZIEH_std300","EZIEH_std900","EZIEH_std3600","EZIEH_std14400",
    "ctemp_std300","ctemp_std900","ctemp_std3600","ctemp_std14400",
    "anom300_inter","anom900_inter","anom1800_inter","anom3600_inter","anom7200_inter",
    "anom14400_inter","anom21600_inter",
    "anom_cumul_inter","anom_ema43200_inter",
    "EZIEH_anom3600_sq","ctemp_anom3600_sq","EZIEH_anom7200_sq","ctemp_anom7200_sq",
    "EZIEH_anom14400_sq","ctemp_anom14400_sq","EZIEH_anom21600_sq","ctemp_anom21600_sq",
    "EZIEH_anom28800_sq","ctemp_anom28800_sq","EZIEH_anom43200_sq","ctemp_anom43200_sq",
    "EZIEH_anom_cumul_sq","ctemp_anom_cumul_sq","EZIEH_anom_ema43200_sq","ctemp_anom_ema43200_sq",
]

FEATURES_5MIN = [
    "EZIEH_anom900","EZIEH_anom1800","EZIEH_anom3600","EZIEH_anom7200",
    "EZIEH_anom14400","EZIEH_anom21600","EZIEH_anom28800","EZIEH_anom43200",
    "ctemp_anom900","ctemp_anom1800","ctemp_anom3600","ctemp_anom7200",
    "ctemp_anom14400","ctemp_anom21600","ctemp_anom28800","ctemp_anom43200",
    "EZIEH_anom_cumul","ctemp_anom_cumul",
    "EZIEH_anom_ema1800","EZIEH_anom_ema7200","EZIEH_anom_ema21600","EZIEH_anom_ema43200",
    "ctemp_anom_ema1800","ctemp_anom_ema7200","ctemp_anom_ema21600","ctemp_anom_ema43200",
    "EZIEH_std900","EZIEH_std3600","EZIEH_std14400",
    "ctemp_std900","ctemp_std3600","ctemp_std14400",
    "anom900_inter","anom1800_inter","anom3600_inter","anom7200_inter",
    "anom14400_inter","anom21600_inter",
    "anom_cumul_inter","anom_ema43200_inter",
    "EZIEH_anom3600_sq","ctemp_anom3600_sq","EZIEH_anom7200_sq","ctemp_anom7200_sq",
    "EZIEH_anom14400_sq","ctemp_anom14400_sq","EZIEH_anom21600_sq","ctemp_anom21600_sq",
    "EZIEH_anom28800_sq","ctemp_anom28800_sq","EZIEH_anom43200_sq","ctemp_anom43200_sq",
    "EZIEH_anom_cumul_sq","ctemp_anom_cumul_sq","EZIEH_anom_ema43200_sq","ctemp_anom_ema43200_sq",
]


def _detect_resolution(model_path: Path):
    name = model_path.stem
    if "1min" in name:
        return "1min", "1min", _add_features_1min, FEATURES_1MIN
    if "5min" in name:
        return "5min", "5min", _add_features_5min, FEATURES_5MIN
    return "1sec", None, _add_features_1sec, FEATURES_1SEC


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

PANEL_CFG = [
    ("FRDH",            "FRDH (nT)",             "#2ca02c"),
    ("EZIEH",           "EZIEH (nT)",             "#1f77b4"),
    ("residual",        "residual (nT)",           "#7f7f7f"),
    ("noise_pred",      "predicted noise (nT)",    "#d62728"),
    ("EZIEH_denoised",  "EZIEH denoised (nT)",     "#9467bd"),
    ("ctemp",           "ctemp (°C)",              "#8c564b"),
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

    res_label, resample_rule, add_features_fn, feature_cols = _detect_resolution(model_path)

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

            df_feat = add_features_fn(df)
            missing = [c for c in feature_cols if c not in df_feat.columns]
            if missing:
                print(f"  {date_str}: missing features {missing[:3]}…, skipping")
                continue

            valid = df_feat[feature_cols].notna().all(axis=1)
            df_feat = df_feat[valid].copy()

            df_feat["noise_pred"]     = model.predict(df_feat[feature_cols].values)
            df_feat["EZIEH_denoised"] = df_feat["EZIEH"] - df_feat["noise_pred"]

            out_path = out_dir / f"{date_str}.png"
            plot_day(df_feat, date_str, model_label, out_path)
            print(f"  {date_str}: saved -> {out_path.name}")

        except Exception as exc:
            print(f"  {date_str}: ERROR – {exc}")

    print(f"\nDone. Plots in {out_dir}/")


if __name__ == "__main__":
    main()
