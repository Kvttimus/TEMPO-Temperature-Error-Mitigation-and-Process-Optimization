"""
Compute per-day summary statistics and write to a single CSV file.

Columns:
  date,
  EZIEH_mean_nT, EZIEH_range_nT,
  residual_mean_nT, residual_range_nT,
  ctemp_mean_degC, ctemp_range_degC,
  pearson_r_ctemp_residual

Sources:
  regression/predicted/<YYYYMMDD>.csv  ->  EZIEH, residual  (1-second native;
      EZIEH_mean/range and residual_mean use the full 1-second series.
      residual_range_nT is the max-min of 1-min medians — the same definition
      step3.1 uses to filter days — and the 1-min residual is also what gets
      merged with ctemp so the two series are on a common grid)
  humanReadable_EZIE_data/<YYYYMMDD>.csv  ->  ctemp (resampled to 1-min to align)

Output: regression/daily_summary.csv
"""
from __future__ import annotations

import pandas as pd
from pathlib import Path

PRED_DIR = Path("regression") / "predicted"
EZIE_DIR = Path("humanReadable_EZIE_data")
OUT_PATH = Path("regression") / "daily_summary.csv"


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


def main():
    pred_dates = sorted(p.stem for p in PRED_DIR.glob("*.csv"))
    print(f"Days found: {len(pred_dates)}")

    rows = []

    for i, date_str in enumerate(pred_dates, 1):
        pred     = pd.read_csv(PRED_DIR / f"{date_str}.csv", parse_dates=["time"])
        ctemp_df = load_ctemp_1min(date_str)

        ezieh    = pred["EZIEH"].dropna()
        residual = pred["residual"].dropna()

        # pred is 1-second native; 1-min medians are used both for the
        # residual range (same definition as the step3.1 filter, so a single
        # spike sample doesn't inflate a day's range) and for the ctemp merge
        # below, which then lands on the same grid as ctemp_df instead of
        # only matching the ~1/60 of rows that happen to fall on a minute mark.
        residual_1min = (
            pred[["time", "residual"]]
            .set_index("time")
            .resample("1min").median()
            .reset_index()
        )
        res_1min = residual_1min["residual"].dropna()

        ezieh_mean  = float(ezieh.mean())
        ezieh_range = float(ezieh.max() - ezieh.min())
        res_mean    = float(residual.mean())
        res_range   = float(res_1min.max() - res_1min.min())

        ctemp_mean  = float("nan")
        ctemp_range = float("nan")
        pearson_r   = float("nan")

        if ctemp_df is not None:
            merged = (
                residual_1min
                .merge(ctemp_df, on="time", how="inner")
                .dropna(subset=["residual", "ctemp"])
            )
            if len(merged) > 2:
                ctemp_mean  = float(merged["ctemp"].mean())
                ctemp_range = float(merged["ctemp"].max() - merged["ctemp"].min())
                pearson_r   = float(merged["ctemp"].corr(merged["residual"]))

        rows.append({
            "date":                     date_str,
            "EZIEH_mean_nT":            round(ezieh_mean,  3),
            "EZIEH_range_nT":           round(ezieh_range, 3),
            "residual_mean_nT":         round(res_mean,    3),
            "residual_range_nT":        round(res_range,   3),
            "ctemp_mean_degC":          round(ctemp_mean,  3),
            "ctemp_range_degC":         round(ctemp_range, 3),
            "pearson_r_ctemp_residual": round(pearson_r,   6),
        })

        print(f"  [{i:>3}/{len(pred_dates)}] {date_str}  "
              f"EZIEH_range={ezieh_range:.1f} nT  "
              f"res_mean={res_mean:+.2f} nT  "
              f"r(ctemp,res)={pearson_r:.3f}")

    df_out = pd.DataFrame(rows)
    df_out.to_csv(OUT_PATH, index=False)
    print(f"\nSaved -> {OUT_PATH}  ({len(df_out)} rows)")


if __name__ == "__main__":
    main()
