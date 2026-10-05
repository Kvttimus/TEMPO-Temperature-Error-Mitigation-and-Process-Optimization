"""
Correlate daily residual noise range against daily ranges of temperature and
inertial sensor channels, across the full mission dataset.

For each day, a channel's "range" is its 1st-to-99th percentile spread rather
than raw max-min, since a handful of glitch/spike samples (rare single-sample
outliers seen in the raw accelerometer/gyroscope channels) can otherwise
dominate a plain max-min range without reflecting real sensor behavior.

Columns tested against daily residual_range_nT (from regression/daily_summary.csv;
max-min of 1-minute medians, the same definition step3.1 filters on):
  ctemp_range_p1p99, ctemp_p1, ctemp_p99,
  Ax_range_p1p99, Ay_range_p1p99, Az_range_p1p99,
  Gy_range_p1p99, Gz_range_p1p99

Sources:
  regression/daily_summary.csv             -> date, residual_range_nT
  humanReadable_EZIE_data/<YYYYMMDD>.csv    -> ctemp, Ax, Ay, Az, Gy, Gz (raw, native resolution)

Output: regression/correlation_coefficients.csv  (covariate, pearson_r, n_days)
"""
from __future__ import annotations

import pandas as pd
from pathlib import Path

SUMMARY_CSV = Path("regression") / "daily_summary.csv"
EZIE_DIR    = Path("humanReadable_EZIE_data")
OUT_PATH    = Path("regression") / "correlation_coefficients.csv"

RAW_COLS  = ["ctemp", "Ax", "Ay", "Az", "Gy", "Gz"]
PCTILE_LO = 0.01
PCTILE_HI = 0.99


def daily_p1_p99(date_str: str) -> pd.Series | None:
    path = EZIE_DIR / f"{date_str}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, usecols=RAW_COLS)
    if df.empty:
        return None
    return df.quantile([PCTILE_LO, PCTILE_HI]).stack()


def main():
    summary = pd.read_csv(SUMMARY_CSV, dtype={"date": str})
    target = summary.set_index("date")["residual_range_nT"]

    rows = []
    for date_str in summary["date"]:
        q = daily_p1_p99(date_str)
        if q is None:
            continue
        row = {"date": date_str}
        for col in RAW_COLS:
            lo, hi = q[(PCTILE_LO, col)], q[(PCTILE_HI, col)]
            row[f"{col}_p1"]           = lo
            row[f"{col}_p99"]          = hi
            row[f"{col}_range_p1p99"]  = hi - lo
        rows.append(row)

    daily = pd.DataFrame(rows).merge(target, left_on="date", right_index=True)
    print(f"Days used: {len(daily)} / {len(summary)}")

    covariates = {
        "Ctemp Range": "ctemp_range_p1p99",
        "Ctemp Min":   "ctemp_p1",
        "Ctemp Max":   "ctemp_p99",
        "Ax Range":    "Ax_range_p1p99",
        "Ay Range":    "Ay_range_p1p99",
        "Az Range":    "Az_range_p1p99",
        "Gy Range":    "Gy_range_p1p99",
        "Gz Range":    "Gz_range_p1p99",
    }

    results = []
    print(f"\nPearson r vs daily residual_range_nT  ({PCTILE_LO*100:.0f}th-{PCTILE_HI*100:.0f}th percentile ranges)")
    for label, col in covariates.items():
        r = daily["residual_range_nT"].corr(daily[col])
        n = daily[col].notna().sum()
        results.append({"covariate": label, "pearson_r": round(float(r), 4), "n_days": int(n)})
        print(f"  {label:15s} r = {r:+.4f}   (n={n})")

    pd.DataFrame(results).to_csv(OUT_PATH, index=False)
    print(f"\nSaved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
