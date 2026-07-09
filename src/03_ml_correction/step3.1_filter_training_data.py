"""
Filter prediction CSVs into 'training data/' folder.
Removes days where residual range > 250 nT (large disturbance / poor-fit days).

Source:  regression/daily_summary.csv  (residual_range_nT column)
Input:   regression/predicted/<YYYYMMDD>.csv
Output:  training data/<YYYYMMDD>.csv  (kept days only)
"""
from __future__ import annotations

import shutil
import pandas as pd
from pathlib import Path

SUMMARY_PATH       = Path("regression") / "daily_summary.csv"
PRED_DIR           = Path("regression") / "predicted"
OUT_DIR            = Path("training data")
MAX_RESIDUAL_RANGE = 250.0  # nT


def main():
    summary = pd.read_csv(SUMMARY_PATH, dtype={"date": str})

    total   = len(summary)
    removed = summary[summary["residual_range_nT"] > MAX_RESIDUAL_RANGE]
    kept    = summary[summary["residual_range_nT"] <= MAX_RESIDUAL_RANGE]

    print(f"Total days : {total}")
    print(f"Threshold  : residual range > {MAX_RESIDUAL_RANGE} nT -> removed")
    print(f"Removed    : {len(removed)}")
    print(f"Kept       : {len(kept)}")
    print()

    if len(removed) > 0:
        print("Removed days:")
        for _, row in removed.sort_values("residual_range_nT", ascending=False).iterrows():
            print(f"  {row['date']}  residual_range={row['residual_range_nT']:.1f} nT")
        print()

    OUT_DIR.mkdir(exist_ok=True)

    copied  = 0
    missing = 0
    for _, row in kept.iterrows():
        date_str = str(row["date"])
        src = PRED_DIR / f"{date_str}.csv"
        dst = OUT_DIR  / f"{date_str}.csv"
        if src.exists():
            shutil.copy2(src, dst)
            copied += 1
        else:
            print(f"  [WARN] {date_str}: prediction CSV not found, skipping")
            missing += 1

    print(f"Copied {copied} CSVs -> '{OUT_DIR}/'")
    if missing:
        print(f"Skipped {missing} (prediction CSV not found)")


if __name__ == "__main__":
    main()
