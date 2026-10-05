"""
TEMPO correction vs FRD reference, one panel per day.

Three series on a shared nT axis (EZIE-Mag units):
  Raw EZIE-Mag      EZIEH
  TEMPO corrected   tempo_h = EZIEH - EZIEH_noise_pred (XGBoost)
  FRD reference     EZIEH_ref = a*FRDH + b (column EZIE_Bh_Predicted): FRD mapped into EZIE-Mag units by that day's
                    step2.1 regression (fit on the first 12 h UTC). A perfect
                    correction would lie on this line.
Each panel title reports RMSE against the FRD reference before and after
correction.

All series are at the model's resolution (1-sec, or 1-min / 5-min medians,
as in step3.2). Features are computed per day exactly as in training.

Also writes a stacked version per day: three connected panels (raw EZIE-Mag,
TEMPO corrected, FRD as native FRDH) on a shared time axis, all with the same
nT span centred on each panel's own midpoint.

Inputs : regression/predicted/<YYYYMMDD>.csv, a step3.2 model (full features)
Outputs: <out_dir>/<YYYYMMDD>.png          overlay, one per day
         <out_dir>/<YYYYMMDD>_stacked.png  three stacked panels, one per day
         <out_dir>/tempo_vs_frd_<days>.png all requested days' overlays stacked

Usage
-----
  python src/03_ml_correction/step3.8_tempo_vs_frd_plots.py --days 20241010 20250101
  python src/03_ml_correction/step3.8_tempo_vs_frd_plots.py --model regression/xgboost_noise_model_1min.json --days 20250416
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

from _feature_engineering import add_features, feature_cols
from _training_data import RESOLUTIONS

# Raw and TEMPO keep the colors they have in the model-comparison charts
# (blue = uncorrected, aqua = XGBoost); the reference is neutral ink.
COLOR_RAW   = "#2a78d6"
COLOR_TEMPO = "#1baf7a"
COLOR_REF   = "#0b0b0b"
INK_MUTED   = "#52514e"
GRID        = "#e1e0d9"


def resolution_of(model_path: Path) -> str:
    name = model_path.stem
    for res in ("1min", "5min"):
        if f"_{res}" in name:
            return res
    return "1sec"


def correct_day(csv_path: Path, model: XGBRegressor, res: str) -> pd.DataFrame:
    cfg = RESOLUTIONS[res]
    df = pd.read_csv(csv_path, parse_dates=["time"])[["time", "EZIEH", "ctemp", "FRDH", "EZIE_Bh_Predicted"]]
    if cfg["resample"] is not None:
        df = df.set_index("time").resample(cfg["resample"]).median().reset_index()
    df = df.dropna(subset=["EZIEH", "ctemp"]).reset_index(drop=True)

    cols = feature_cols(cfg["seconds_per_row"])
    df = add_features(df, cfg["seconds_per_row"])
    df = df[df[cols].notna().all(axis=1)].copy()
    df["EZIEH_noise_pred"] = model.predict(df[cols].values)
    df["tempo_h"]          = df["EZIEH"] - df["EZIEH_noise_pred"]
    return df


def rmse(a: pd.Series, b: pd.Series) -> float:
    return float(np.sqrt(((a - b) ** 2).mean()))


def draw_day(ax, df: pd.DataFrame, date_str: str):
    day = pd.Timestamp(date_str)
    ax.plot(df["time"], df["EZIEH"],             color=COLOR_RAW,   linewidth=1.0, label="Raw EZIE-Mag")
    ax.plot(df["time"], df["tempo_h"],           color=COLOR_TEMPO, linewidth=1.0, label="TEMPO corrected")
    ax.plot(df["time"], df["EZIE_Bh_Predicted"], color=COLOR_REF,   linewidth=1.2, label="FRD reference",
            linestyle="--")

    ok = df["EZIE_Bh_Predicted"].notna()
    r_raw   = rmse(df.loc[ok, "EZIEH"],       df.loc[ok, "EZIE_Bh_Predicted"])
    r_tempo = rmse(df.loc[ok, "tempo_h"], df.loc[ok, "EZIE_Bh_Predicted"])
    ax.set_title(f"{day:%Y-%m-%d}    RMSE vs FRD reference: raw {r_raw:.1f} nT  →  TEMPO {r_tempo:.1f} nT",
                 fontsize=11, loc="left")

    ax.set_xlim(day, day + pd.Timedelta(days=1))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.set_ylabel("Bh (nT, EZIE-Mag units)", fontsize=10)
    ax.grid(True, linewidth=0.6, color=GRID)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(colors=INK_MUTED, labelsize=9)
    # Legend sits above the axes (title row, right side) so it never covers data
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), fontsize=9, frameon=False, ncol=3,
              borderaxespad=0.2)
    return r_raw, r_tempo


def draw_stacked(df: pd.DataFrame, date_str: str, label: str, out_path: Path):
    """Three connected panels: raw EZIE-Mag, TEMPO corrected, FRD (native FRDH).
    All panels share one nT span, each centred on its own midpoint, so
    fluctuation sizes are directly comparable across panels."""
    day = pd.Timestamp(date_str)
    panels = [("EZIEH",   "Raw EZIE-Mag\nBh (nT)",    COLOR_RAW),
              ("tempo_h", "TEMPO corrected\nBh (nT)", COLOR_TEMPO),
              ("FRDH",    "FRD reference\nBh (nT)",   COLOR_REF)]

    span = max(df[c].max() - df[c].min() for c, _, _ in panels) * 1.08
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    fig.subplots_adjust(hspace=0.08)
    for ax, (col, ylabel, color) in zip(axes, panels):
        ax.plot(df["time"], df[col], color=color, linewidth=1.0)
        mid = (df[col].max() + df[col].min()) / 2
        ax.set_ylim(mid - span / 2, mid + span / 2)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.grid(True, linewidth=0.6, color=GRID)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.tick_params(colors=INK_MUTED, labelsize=9)

    ok = df["EZIE_Bh_Predicted"].notna()
    r_raw   = rmse(df.loc[ok, "EZIEH"],       df.loc[ok, "EZIE_Bh_Predicted"])
    r_tempo = rmse(df.loc[ok, "tempo_h"], df.loc[ok, "EZIE_Bh_Predicted"])
    axes[0].set_title(f"{day:%Y-%m-%d}    RMSE vs FRD reference: raw {r_raw:.1f} nT  →  TEMPO {r_tempo:.1f} nT"
                      f"    (all panels share a {span:.0f} nT span)", fontsize=11, loc="left")
    axes[-1].set_xlim(day, day + pd.Timedelta(days=1))
    axes[-1].xaxis.set_major_locator(mdates.HourLocator(interval=2))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    axes[-1].set_xlabel("UTC", fontsize=10)
    fig.suptitle(label, fontsize=9, color=INK_MUTED, x=0.99, ha="right", y=0.94)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model",    default="regression/xgboost_noise_model_5min.json")
    ap.add_argument("--data_dir", default="regression/predicted")
    ap.add_argument("--out_dir",  default="regression/tempo_vs_frd")
    ap.add_argument("--days",     nargs="+", required=True, help="YYYYMMDD dates to plot")
    args = ap.parse_args()

    model_path, data_dir, out_dir = Path(args.model), Path(args.data_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    res = resolution_of(model_path)
    model = XGBRegressor()
    model.load_model(model_path)
    print(f"Model: {model_path}  (resolution {res})")

    days = {d: correct_day(data_dir / f"{d}.csv", model, res) for d in args.days}
    label = f"model: {model_path.stem}, {res} resolution"

    for d, df in days.items():
        fig, ax = plt.subplots(figsize=(14, 4.2))
        r_raw, r_tempo = draw_day(ax, df, d)
        fig.suptitle(label, fontsize=9, color=INK_MUTED, x=0.99, ha="right")
        fig.tight_layout()
        fig.savefig(out_dir / f"{d}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        draw_stacked(df, d, label, out_dir / f"{d}_stacked.png")
        print(f"  {d}: RMSE vs FRD reference raw {r_raw:.2f} nT -> TEMPO {r_tempo:.2f} nT  ({len(df):,} points)")

    fig, axes = plt.subplots(len(days), 1, figsize=(14, 3.8 * len(days)), squeeze=False)
    for ax, (d, df) in zip(axes[:, 0], days.items()):
        draw_day(ax, df, d)
    axes[-1, 0].set_xlabel("UTC", fontsize=10)
    fig.suptitle(label, fontsize=9, color=INK_MUTED, x=0.99, ha="right")
    fig.tight_layout()
    combined = out_dir / f"tempo_vs_frd_{'_'.join(days)}.png"
    fig.savefig(combined, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Combined -> {combined}")


if __name__ == "__main__":
    main()
