"""Shared data loading and train/validation/test split for the EZIEH_noise_ref models.

Single source of truth for step3.2 (XGBoost) and step3.2b (linear baseline),
so both models are always fit, early-stopped/tuned, and scored on exactly the
same rows. Anything that changes which days land in which split belongs here.

Modes differ only in which days they read:
  filtered   - 'training data/' (high-residual days removed by step3.1)
  unfiltered - 'regression/predicted/' (every day); no day is singled out in
               training, early stopping, or evaluation.

Split: chronological 60/20/20 at the day level, applied within each mode's own
set of days (so filtered and unfiltered have different test days).
  train      - earliest 60% of days; model fitting
  validation - next 20%; early stopping (XGBoost)
  test       - latest 20%; touched once, for the reported metrics. This is the
               same set of days the original 80/20 split used as its test set,
               so results stay comparable across the change.

Every call writes regression/split_days_<mode>_<resolution>.csv listing each
day's split, so reported numbers can always be traced back to the exact days
behind them.
"""
from __future__ import annotations

import pandas as pd
from pathlib import Path

from _feature_engineering import add_features, feature_cols

TRAIN_DATA_DIR      = Path("training data")
UNFILTERED_DATA_DIR = Path("regression/predicted")
SPLIT_DAYS_DIR      = Path("regression")   # split_days_<mode>_<resolution>.csv written here

TARGET     = "residual"   # EZIEH_noise_ref (column name used in the step2.1 CSVs)
TRAIN_FRAC = 0.6
VAL_FRAC   = 0.2     # test gets the remaining 0.2

# Per-resolution knobs: pandas resample rule (None = raw, no resampling),
# minimum rows to keep a day, XGBoost min_child_weight (scaled down for
# coarser resolutions' smaller per-day row counts), and scatter-plot styling.
RESOLUTIONS = {
    "1sec": dict(resample=None,   seconds_per_row=1,   min_rows=60, min_child_weight=15, scatter_size=0.3, scatter_alpha=0.15),
    "1min": dict(resample="1min", seconds_per_row=60,  min_rows=10, min_child_weight=5,   scatter_size=1.5, scatter_alpha=0.2),
    "5min": dict(resample="5min", seconds_per_row=300, min_rows=5,  min_child_weight=3,   scatter_size=3,   scatter_alpha=0.3),
}


def load_day(csv_path: Path, resolution: str, cols: list[str]) -> pd.DataFrame | None:
    cfg = RESOLUTIONS[resolution]
    df = pd.read_csv(csv_path, parse_dates=["time"])
    if "ctemp" not in df.columns or df["ctemp"].isna().all():
        return None

    if cfg["resample"] is None:
        df = df[["time", "EZIEH", "ctemp", TARGET]].dropna(subset=["EZIEH", "ctemp", TARGET])
    else:
        df = df.set_index("time")[["EZIEH", "ctemp", TARGET]].resample(cfg["resample"]).median()
        df = df.dropna(subset=["EZIEH", "ctemp", TARGET]).reset_index()

    if len(df) < cfg["min_rows"]:
        return None

    df = add_features(df, cfg["seconds_per_row"])
    df["date"] = csv_path.stem
    return df.dropna(subset=cols + [TARGET])


def load_splits(mode: str, resolution: str):
    """Load every day for the given mode/resolution and split chronologically.

    Returns (train, val, test, cols, dates). Both modes are treated identically;
    they differ only in which folder of days they read.
    """
    cols = feature_cols(RESOLUTIONS[resolution]["seconds_per_row"])
    data_dir = TRAIN_DATA_DIR if mode == "filtered" else UNFILTERED_DATA_DIR
    csv_files = sorted(data_dir.glob("*.csv"))
    dates = [f.stem for f in csv_files]
    print(f"Days available: {len(dates)}  (mode={mode}, resolution={resolution})")

    frames, skipped = [], 0
    for f in csv_files:
        df = load_day(f, resolution, cols)
        if df is not None:
            frames.append(df)
        else:
            skipped += 1
    if skipped:
        print(f"Skipped {skipped} days (missing ctemp or insufficient data)")

    full = pd.concat(frames, ignore_index=True)
    print(f"Total rows loaded: {len(full):,}  |  Features: {len(cols)}")

    n_train = int(len(dates) * TRAIN_FRAC)
    n_trval = int(len(dates) * (TRAIN_FRAC + VAL_FRAC))
    train_dates = set(dates[:n_train])
    val_dates   = set(dates[n_train:n_trval])
    test_dates  = set(dates[n_trval:])

    train = full[full["date"].isin(train_dates)].copy()
    val   = full[full["date"].isin(val_dates)].copy()
    test  = full[full["date"].isin(test_dates)].copy()

    # Record exactly which day landed in which split for this run.
    # rows_used = 0 means the day's file exists but was skipped (see load_day).
    split_of = {**{d: "train" for d in train_dates},
                **{d: "val" for d in val_dates},
                **{d: "test" for d in test_dates}}
    days = pd.DataFrame({"date": dates})
    days["split"] = days["date"].map(split_of)
    days["rows_used"] = days["date"].map(full.groupby("date").size()).fillna(0).astype(int)
    SPLIT_DAYS_DIR.mkdir(exist_ok=True)
    days.to_csv(SPLIT_DAYS_DIR / f"split_days_{mode}_{resolution}.csv", index=False)

    print("\nChronological 60/20/20 split:")
    for name, part, part_dates in (("Train", train, train_dates),
                                   ("Val",   val,   val_dates),
                                   ("Test",  test,  test_dates)):
        print(f"  {name:5s}: {len(part_dates)} days  ({len(part):,} rows)  "
              f"{min(part_dates)} -> {max(part_dates)}")

    return train, val, test, cols, dates
