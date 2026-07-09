"""Shared feature engineering for the XGBoost Bh-noise model.

Single source of truth for step3.2 (training), step3.3 (inference), and
step3.4 (survey plots) — these three previously each hand-duplicated this
logic and silently drifted out of sync (one copy was missing the
ctemp_std900 feature), so anything that changes window sizes, half-lives,
or feature order belongs here, not in the callers.
"""
from __future__ import annotations

import pandas as pd

SECONDS_PER_ROW = {"1sec": 1, "1min": 60, "5min": 300}

# Feature window definitions, always in seconds. At a given resolution, a
# window is only usable if it spans more than one sample (see feature_cols).
ROLL_WINDOWS_SEC        = (30, 300, 900, 1800, 3600, 7200, 14400, 21600, 28800, 43200)
EMA_HALFLIVES_SEC       = (1800, 7200, 21600, 43200)
STD_WINDOWS_SEC         = (30, 300, 900, 3600, 14400)
INTERACTION_WINDOWS_SEC = (30, 300, 900, 1800, 3600, 7200, 14400, 21600)
SQUARE_WINDOWS_SEC      = (3600, 7200, 14400, 21600, 28800, 43200)


def _usable_windows(windows_sec, seconds_per_row):
    return [w for w in windows_sec if w > seconds_per_row]


def add_features(df: pd.DataFrame, seconds_per_row: int) -> pd.DataFrame:
    """Anomaly-based features from ctemp and EZIEH only.
    All features are deviations from causal rolling baselines so absolute
    drift across the mission does not cause train/test distribution shift."""
    df = df.copy().sort_values("time").reset_index(drop=True)

    e = df["EZIEH"]
    c = df["ctemp"]

    def rows(seconds: int) -> int:
        return max(1, seconds // seconds_per_row)

    roll_windows = _usable_windows(ROLL_WINDOWS_SEC, seconds_per_row)
    for w in roll_windows:
        e_base = e.rolling(rows(w), min_periods=1).mean()
        c_base = c.rolling(rows(w), min_periods=1).mean()
        df[f"EZIEH_anom{w}"] = e - e_base
        df[f"ctemp_anom{w}"] = c - c_base

    # Cumulative-mean baseline (anchored to midnight; better than rolling at late hours)
    e_cumul = e.expanding(min_periods=1).mean()
    c_cumul = c.expanding(min_periods=1).mean()
    df["EZIEH_anom_cumul"] = e - e_cumul
    df["ctemp_anom_cumul"] = c - c_cumul

    # Exponential moving averages at key half-lives
    for hl in EMA_HALFLIVES_SEC:
        df[f"EZIEH_anom_ema{hl}"] = e - e.ewm(halflife=rows(hl), adjust=False).mean()
        df[f"ctemp_anom_ema{hl}"] = c - c.ewm(halflife=rows(hl), adjust=False).mean()

    # Local volatility (same windows for EZIEH and ctemp)
    std_windows = _usable_windows(STD_WINDOWS_SEC, seconds_per_row)
    for w in std_windows:
        df[f"EZIEH_std{w}"] = e.rolling(max(2, rows(w)), min_periods=2).std()
        df[f"ctemp_std{w}"] = c.rolling(max(2, rows(w)), min_periods=2).std()

    # Interactions at matching timescales
    interaction_windows = _usable_windows(INTERACTION_WINDOWS_SEC, seconds_per_row)
    for w in interaction_windows:
        df[f"anom{w}_inter"] = df[f"EZIEH_anom{w}"] * df[f"ctemp_anom{w}"]
    df["anom_cumul_inter"]    = df["EZIEH_anom_cumul"]    * df["ctemp_anom_cumul"]
    df["anom_ema43200_inter"] = df["EZIEH_anom_ema43200"] * df["ctemp_anom_ema43200"]

    # Non-linear terms for key long windows
    for w in SQUARE_WINDOWS_SEC:
        df[f"EZIEH_anom{w}_sq"] = df[f"EZIEH_anom{w}"] ** 2
        df[f"ctemp_anom{w}_sq"] = df[f"ctemp_anom{w}"] ** 2
    df["EZIEH_anom_cumul_sq"]    = df["EZIEH_anom_cumul"]    ** 2
    df["EZIEH_anom_ema43200_sq"] = df["EZIEH_anom_ema43200"] ** 2
    df["ctemp_anom_cumul_sq"]    = df["ctemp_anom_cumul"]    ** 2
    df["ctemp_anom_ema43200_sq"] = df["ctemp_anom_ema43200"] ** 2

    return df


def feature_cols(seconds_per_row: int) -> list[str]:
    roll_windows        = _usable_windows(ROLL_WINDOWS_SEC, seconds_per_row)
    std_windows          = _usable_windows(STD_WINDOWS_SEC, seconds_per_row)
    interaction_windows = _usable_windows(INTERACTION_WINDOWS_SEC, seconds_per_row)

    cols = []
    cols += [f"EZIEH_anom{w}" for w in roll_windows]
    cols += [f"ctemp_anom{w}" for w in roll_windows]
    cols += ["EZIEH_anom_cumul", "ctemp_anom_cumul"]
    cols += [f"EZIEH_anom_ema{hl}" for hl in EMA_HALFLIVES_SEC]
    cols += [f"ctemp_anom_ema{hl}" for hl in EMA_HALFLIVES_SEC]
    cols += [f"EZIEH_std{w}" for w in std_windows]
    cols += [f"ctemp_std{w}" for w in std_windows]
    cols += [f"anom{w}_inter" for w in interaction_windows]
    cols += ["anom_cumul_inter", "anom_ema43200_inter"]
    for w in SQUARE_WINDOWS_SEC:
        cols += [f"EZIEH_anom{w}_sq", f"ctemp_anom{w}_sq"]
    cols += ["EZIEH_anom_cumul_sq", "ctemp_anom_cumul_sq",
             "EZIEH_anom_ema43200_sq", "ctemp_anom_ema43200_sq"]
    return cols
