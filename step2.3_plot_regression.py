#!/usr/bin/env python3
"""
Generate 4-panel daily survey plots combining FRD, EZIE, and regression results.

Panels (top to bottom):
  1. FRDH  (nT)
  2. EZIEH (blue) + EZIE_Bh_Predicted (orange dashed) -- overlaid, same axis
  3. Residual: EZIEH - EZIE_Bh_Predicted (nT), with zero reference line
  4. ctemp (deg C)

Data sources:
  regression/predicted/<YYYYMMDD>.csv  ->  FRDH, EZIEH, EZIE_Bh_Predicted, residual
  humanReadable_EZIE_data/<YYYYMMDD>.csv  ->  ctemp (resampled to 1-min median)

Output: plots_regression/<YYYYMMDD>.png
"""
from __future__ import annotations

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

PRED_DIR   = Path("regression") / "predicted"
EZIE_DIR   = Path("humanReadable_EZIE_data")
OUTPUT_DIR = Path("plots_regression")


def load_ctemp_1min(date_str: str) -> pd.DataFrame | None:
    path = EZIE_DIR / f"{date_str}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, parse_dates=["timeString"])
    if "ctemp" not in df.columns:
        return None
    df = df.rename(columns={"timeString": "time"})
    df = df[["time", "ctemp"]].dropna(subset=["time"]).sort_values("time").set_index("time")
    df.index = df.index.tz_convert("UTC") if df.index.tz is not None else df.index.tz_localize("UTC")
    df = df.resample("1min").median()
    df.index = df.index.tz_localize(None)
    return df.reset_index()


def plot_day(date_str: str, i: int, total: int):
    out_path = OUTPUT_DIR / f"{date_str}.png"
    if out_path.exists():
        print(f"  [{i:>3}/{total}] {date_str}: skip (exists)")
        return

    pred_path = PRED_DIR / f"{date_str}.csv"
    if not pred_path.exists():
        print(f"  [{i:>3}/{total}] {date_str}: skip (no prediction CSV)")
        return

    pred     = pd.read_csv(pred_path, parse_dates=["time"])
    ctemp_df = load_ctemp_1min(date_str)

    t_start = pd.Timestamp(f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]} 00:00:00")
    t_end   = pd.Timestamp(f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]} 23:59:59")

    date_fmt = mdates.DateFormatter("%H:%M")
    hour_loc = mdates.HourLocator(interval=2)

    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(date_str, fontsize=13, fontweight="bold")

    # --- Panel 1: FRDH ---
    ax = axes[0]
    ax.plot(pred["time"], pred["FRDH"], color="#1f77b4", linewidth=0.6)
    ax.set_ylabel("FRDH (nT)")
    ax.set_title(f"FRDH")
    ax.grid(True, linewidth=0.3, alpha=0.5)

    # --- Panel 2: EZIEH + EZIE_Bh_Predicted overlaid ---
    ax = axes[1]
    ax.plot(pred["time"], pred["EZIEH"],             color="#1f77b4", linewidth=0.6,
            label="EZIEH")
    ax.plot(pred["time"], pred["EZIE_Bh_Predicted"], color="#ff7f0e", linewidth=0.6,
            linestyle="--", label="Predicted")
    ax.set_ylabel("Bh (nT)")
    ax.set_title("EZIEH vs EZIE_Bh_Predicted")
    ax.legend(fontsize=7, loc="upper right", framealpha=0.7)
    ax.grid(True, linewidth=0.3, alpha=0.5)

    # --- Panel 3: Residual ---
    ax = axes[2]
    ax.plot(pred["time"], pred["residual"], color="#2ca02c", linewidth=0.6)
    ax.axhline(0, color="black", linewidth=0.6, linestyle="--", alpha=0.7)
    ax.set_ylabel("Residual (nT)")
    ax.set_title("Residual (EZIEH - Predicted)")
    ax.grid(True, linewidth=0.3, alpha=0.5)

    # --- Panel 4: ctemp ---
    ax = axes[3]
    if ctemp_df is not None and not ctemp_df["ctemp"].isna().all():
        ax.plot(ctemp_df["time"], ctemp_df["ctemp"], color="#d62728", linewidth=0.6)
    ax.set_ylabel("ctemp (°C)")
    ax.set_title("EZIE coil temp")
    ax.set_xlabel("UTC time")
    ax.grid(True, linewidth=0.3, alpha=0.5)

    # x-axis shared: set limits + formatter on bottom panel only (sharex handles the rest)
    axes[3].set_xlim(t_start, t_end)
    axes[3].xaxis.set_major_formatter(date_fmt)
    axes[3].xaxis.set_major_locator(hour_loc)
    axes[3].tick_params(axis="x", labelsize=8)

    plt.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)

    print(f"  [{i:>3}/{total}] {date_str}")


def main():
    pred_dates = sorted(p.stem for p in PRED_DIR.glob("*.csv"))
    total = len(pred_dates)
    print(f"Days to plot: {total}")
    print(f"Output      : {OUTPUT_DIR.resolve()}")
    print("-" * 45)

    for i, date_str in enumerate(pred_dates, 1):
        plot_day(date_str, i, total)

    print("-" * 45)
    print("Done.")


if __name__ == "__main__":
    main()
