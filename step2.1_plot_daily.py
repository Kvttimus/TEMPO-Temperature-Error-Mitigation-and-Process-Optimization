#!/usr/bin/env python3
"""
Generate 3-panel daily plots for each overlapping EZIE + FRD day.

Panels (top to bottom):
  1. FRDH  (nT)  -- FRD horizontal component
  2. EZIEH (nT)  -- EZIE horizontal component  [sqrt(Bx^2 + By^2)]
  3. ctemp (deg C) -- EZIE coil temperature

Output: plots/<YYYYMMDD>.png
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

EZIE_DIR   = Path("humanReadable_EZIE_data")
FRD_DIR    = Path("uncompressed_FRD_data")
OUTPUT_DIR = Path("plots")

IAGA_MISSING = 99999.0


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_ezie(date_str: str) -> pd.DataFrame | None:
    path = EZIE_DIR / f"{date_str}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, parse_dates=["timeString"])
    df = df.rename(columns={"timeString": "time", "Bx": "EZIEX", "By": "EZIEY"})
    df["EZIEH"] = (df["EZIEX"] ** 2 + df["EZIEY"] ** 2) ** 0.5
    df = df[["time", "EZIEH", "ctemp"]].dropna(subset=["time"])
    df = df.sort_values("time").set_index("time")
    df.index = df.index.tz_convert("UTC") if df.index.tz is not None else df.index.tz_localize("UTC")

    # Resample ~2 Hz raw data to 1-minute median so the plot line is readable
    df = df.resample("1min").median()

    df = df.reset_index()
    df["time"] = df["time"].dt.tz_localize(None)  # strip UTC → naive, matches FRD timestamps
    return df


def load_frd(date_str: str) -> pd.DataFrame | None:
    path = FRD_DIR / f"FRD_{date_str}_1sec.sec"
    if not path.exists():
        return None
    rows = []
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if "|" in line:
                continue
            p = line.split()
            if len(p) < 7:
                continue
            try:
                frdx = float(p[3])
                frdy = float(p[4])
            except ValueError:
                continue
            if abs(frdx - IAGA_MISSING) < 0.01 or abs(frdy - IAGA_MISSING) < 0.01:
                frdx = frdy = float("nan")
            rows.append({"time": pd.Timestamp(f"{p[0]} {p[1]}"), "FRDH": (frdx**2 + frdy**2) ** 0.5})
    if not rows:
        return None
    df = pd.DataFrame(rows).sort_values("time").set_index("time")
    # Linear interpolation for short gaps only (<=600 s); multi-hour outages stay NaN
    df["FRDH"] = df["FRDH"].interpolate(method="time", limit=600)
    return df.reset_index()


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_day(date_str: str, ezie: pd.DataFrame | None, frd: pd.DataFrame | None):
    out_path = OUTPUT_DIR / f"{date_str}.png"
    if out_path.exists():
        return  # skip already done

    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=False)
    fig.suptitle(date_str, fontsize=13, fontweight="bold")

    date_fmt = mdates.DateFormatter("%H:%M")

    # x-axis bounds: exactly midnight to midnight for this date
    t_start = pd.Timestamp(f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]} 00:00:00")
    t_end   = pd.Timestamp(f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]} 23:59:59")

    # Compute shared nT range for FRDH and EZIEH panels
    frdh_vals  = frd["FRDH"].dropna()  if frd  is not None else pd.Series(dtype=float)
    ezieh_vals = ezie["EZIEH"].dropna() if ezie is not None else pd.Series(dtype=float)

    frdh_range  = float(frdh_vals.max()  - frdh_vals.min())  if len(frdh_vals)  > 1 else 1.0
    ezieh_range = float(ezieh_vals.max() - ezieh_vals.min()) if len(ezieh_vals) > 1 else 1.0
    shared_range = max(frdh_range, ezieh_range)
    padding = shared_range * 0.05  # 5% breathing room above and below

    def ylims(vals):
        mid = (vals.max() + vals.min()) / 2
        half = shared_range / 2 + padding
        return mid - half, mid + half

    # --- Panel 1: FRDH ---
    ax = axes[0]
    if len(frdh_vals) > 1:
        ax.plot(frd["time"], frd["FRDH"], color="#1f77b4", linewidth=0.6)
        ax.set_ylim(*ylims(frdh_vals))
    ax.set_xlim(t_start, t_end)
    ax.set_ylabel("FRDH (nT)")
    ax.set_title(f"FRDH  {date_str}")
    ax.xaxis.set_major_formatter(date_fmt)
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
    ax.tick_params(axis="x", labelsize=8)
    ax.grid(True, linewidth=0.3, alpha=0.5)

    # --- Panel 2: EZIEH ---
    ax = axes[1]
    if len(ezieh_vals) > 1:
        ax.plot(ezie["time"], ezie["EZIEH"], color="#1f77b4", linewidth=0.6)
        ax.set_ylim(*ylims(ezieh_vals))
    ax.set_xlim(t_start, t_end)
    ax.set_ylabel("EZIEH (nT)")
    ax.set_title(f"EZIE_Bh  {date_str}")
    ax.xaxis.set_major_formatter(date_fmt)
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
    ax.tick_params(axis="x", labelsize=8)
    ax.grid(True, linewidth=0.3, alpha=0.5)

    # --- Panel 3: ctemp ---
    ax = axes[2]
    if ezie is not None and "ctemp" in ezie.columns and not ezie["ctemp"].isna().all():
        ax.plot(ezie["time"], ezie["ctemp"], color="#d62728", linewidth=0.6)
    ax.set_xlim(t_start, t_end)
    ax.set_ylabel("ctemp (°C)")
    ax.set_title(f"EZIE coil temp  {date_str}")
    ax.set_xlabel("UTC time")
    ax.xaxis.set_major_formatter(date_fmt)
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
    ax.tick_params(axis="x", labelsize=8)
    ax.grid(True, linewidth=0.3, alpha=0.5)

    plt.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ezie_dates = {p.stem for p in EZIE_DIR.glob("*.csv")}
    frd_dates  = {p.stem[4:12] for p in FRD_DIR.glob("FRD_*_1sec.sec")}
    all_dates  = sorted(ezie_dates | frd_dates)

    print(f"Days to plot : {len(all_dates)}")
    print(f"Output       : {OUTPUT_DIR.resolve()}")
    print("-" * 45)

    for i, date_str in enumerate(all_dates, 1):
        ezie = load_ezie(date_str)
        frd  = load_frd(date_str)
        plot_day(date_str, ezie, frd)
        print(f"  [{i:>3}/{len(all_dates)}] {date_str}")

    print("-" * 45)
    print("Done.")


if __name__ == "__main__":
    main()
