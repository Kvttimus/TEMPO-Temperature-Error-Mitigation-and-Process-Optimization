#!/usr/bin/env python3
"""
Single-panel survey plot spanning all of December (Dec 1 - Dec 31): just
EZIEH denoised (XGBoost-corrected) and EZIE_Bh_Predicted (linear regression),
resampled to 5-minute median for a readable month-long view.

Uses regression/xgboost_noise_model_5min.json on the unfiltered per-day
predictions in regression/predicted/.

Output
------
  regression/december_survey_plot.png
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from xgboost import XGBRegressor

DATA_DIR   = Path("regression/predicted")
MODEL_PATH = Path("regression/xgboost_noise_model_5min.json")
OUT_PATH   = Path("regression/december_survey_plot.png")
YEAR_MONTH = "202412"

_W5  = {s: max(1, s // 300) for s in (900, 1800, 3600, 7200, 14400, 21600, 28800, 43200)}
_HL5 = {s: max(1, s // 300) for s in (1800, 7200, 21600, 43200)}
_SW5 = {s: max(2, s // 300) for s in (900, 3600, 14400)}


def add_features_5min(df: pd.DataFrame) -> pd.DataFrame:
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


def main():
    days = sorted(p.stem for p in DATA_DIR.glob(f"{YEAR_MONTH}*.csv"))
    print(f"December days found: {len(days)}")

    model = XGBRegressor()
    model.load_model(MODEL_PATH)

    frames = []
    for date_str in days:
        df = pd.read_csv(DATA_DIR / f"{date_str}.csv", parse_dates=["time"])
        df = (df.set_index("time")[["EZIEH", "EZIE_Bh_Predicted", "ctemp"]]
                .resample("5min").median()
                .dropna(subset=["EZIEH", "ctemp"])
                .reset_index())
        if len(df) < 5:
            print(f"  {date_str}: too few rows, skipping")
            continue

        df_feat = add_features_5min(df)
        valid = df_feat[FEATURES_5MIN].notna().all(axis=1)
        df_feat = df_feat[valid].copy()
        if df_feat.empty:
            continue

        df_feat["noise_pred"]     = model.predict(df_feat[FEATURES_5MIN].values)
        df_feat["EZIEH_denoised"] = df_feat["EZIEH"] - df_feat["noise_pred"]
        frames.append(df_feat[["time", "EZIEH_denoised", "EZIE_Bh_Predicted"]])
        print(f"  {date_str}: {len(df_feat)} rows")

    full = pd.concat(frames, ignore_index=True).sort_values("time")
    print(f"\nTotal rows: {len(full):,}")

    fig, ax = plt.subplots(figsize=(18, 6))
    ax.plot(full["time"], full["EZIE_Bh_Predicted"], color="#2ca02c", linewidth=0.8,
             label="EZIEH regressed", rasterized=True)
    ax.plot(full["time"], full["EZIEH_denoised"], color="#9467bd", linewidth=0.8,
             label="EZIEH denoised", rasterized=True)

    day_start = pd.Timestamp("2024-12-01")
    day_end   = pd.Timestamp("2025-01-01")
    ax.set_xlim(day_start, day_end)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    ax.tick_params(axis="x", labelsize=10, rotation=45)
    ax.tick_params(axis="y", labelsize=10)
    ax.set_ylabel("Bh (nT)", fontsize=12)
    ax.set_xlabel("UTC (December 2024)", fontsize=12)
    ax.set_title("December 2024 Survey — EZIEH Denoised vs EZIEH Regressed (5-min median)",
                  fontsize=13)
    ax.grid(True, linewidth=0.4, alpha=0.4)
    ax.legend(loc="upper right", fontsize=10, frameon=False)

    plt.tight_layout()
    fig.savefig(OUT_PATH, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
