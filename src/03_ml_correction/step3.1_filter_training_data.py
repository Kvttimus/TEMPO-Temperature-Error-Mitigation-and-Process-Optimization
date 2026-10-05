"""
Filter prediction CSVs into 'training data/' folder.
Removes days where the residual range > 250 nT (large disturbance / poor-fit days).

The range is measured on 1-minute medians of the residual, not the raw
1-second series: a single spike sample should not remove a whole day.

Input:   regression/predicted/<YYYYMMDD>.csv
Output:  training data/<YYYYMMDD>.csv  (kept days only)
"""
from __future__ import annotations

import shutil
import pandas as pd
from pathlib import Path

PRED_DIR           = Path("regression") / "predicted"
OUT_DIR            = Path("training data")
MAX_RESIDUAL_RANGE = 250.0  # nT, on 1-minute medians
RANGE_RESAMPLE     = "1min"


def residual_range(csv_path: Path) -> float:
    df = pd.read_csv(csv_path, usecols=["time", "residual"], parse_dates=["time"])
    med = df.set_index("time")["residual"].resample(RANGE_RESAMPLE).median().dropna()
    return float(med.max() - med.min())


def main():
    pred_files = sorted(PRED_DIR.glob("*.csv"))
    ranges = pd.Series({f.stem: residual_range(f) for f in pred_files}, name="residual_range_1min_nT")

    removed = ranges[ranges > MAX_RESIDUAL_RANGE]
    kept    = ranges[ranges <= MAX_RESIDUAL_RANGE]

    print(f"Total days : {len(ranges)}")
    print(f"Threshold  : residual range (1-min medians) > {MAX_RESIDUAL_RANGE} nT -> removed")
    print(f"Removed    : {len(removed)}")
    print(f"Kept       : {len(kept)}")
    print()

    if len(removed) > 0:
        print("Removed days:")
        for date_str, r in removed.sort_values(ascending=False).items():
            print(f"  {date_str}  residual_range={r:.1f} nT")
        print()

    OUT_DIR.mkdir(exist_ok=True)
    for date_str in kept.index:
        shutil.copy2(PRED_DIR / f"{date_str}.csv", OUT_DIR / f"{date_str}.csv")
    print(f"Copied {len(kept)} CSVs -> '{OUT_DIR}/'")


if __name__ == "__main__":
    main()
