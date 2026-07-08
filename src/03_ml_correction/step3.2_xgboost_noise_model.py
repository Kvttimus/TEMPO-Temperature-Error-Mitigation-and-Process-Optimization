#!/usr/bin/env python3
"""
XGBoost model: predict Bh_noise (residual) from ctemp and EZIEH only.

During training the pre-calculated residual (EZIEH - EZIE_Bh_Predicted) is used
as the target.  At inference time ONLY ctemp and EZIEH are needed — all derived
features (lags, rolling stats, diffs) are computed from those two signals.

Features engineered from ctemp + EZIEH (window sizes are always expressed in
seconds; windows not exceeding the sample interval are dropped, e.g. the 30s
window is skipped at 1-min resolution):
  Rolling anomalies (30s -> 12h) : EZIEH_anom<w>, ctemp_anom<w>
  Cumulative (midnight-anchored) : EZIEH_anom_cumul, ctemp_anom_cumul
  EMA anomalies                  : EZIEH_anom_ema<hl>, ctemp_anom_ema<hl>
  Rolling std (volatility)       : EZIEH_std<w>, ctemp_std<w>
  Interactions                   : anom<w>_inter
  Non-linear                     : EZIEH_anom<w>_sq, ctemp_anom<w>_sq

Two axes of variation, selected via CLI flags:
  --mode filtered|unfiltered
      filtered   - trains on quiet-day-only data in 'training data/' with a
                   standard MSE objective and a plain train/test eval set.
      unfiltered - trains on all days (incl. storms) in 'regression/predicted/'
                   with a Huber loss (reg:pseudohubererror) so extreme storm
                   residuals don't dominate the gradient, and early-stops on
                   quiet-day-only test MAE (storm days excluded from eval_set
                   via regression/daily_summary.csv) for an apples-to-apples
                   comparison against the filtered model.
  --resolution 1sec|1min|5min
      Resamples input to the given interval (median) before feature
      engineering; sub-sample windows are dropped and min_child_weight is
      scaled down for the smaller per-day row counts at coarser resolutions.

Split: chronological 80/20 at the day level.

Outputs (suffix encodes resolution + mode, e.g. _5min_unfiltered; the
1sec+filtered combination — the original/default model — has no suffix):
  regression/xgboost_noise_model<suffix>.json
  regression/xgboost_test_predictions<suffix>.csv
  regression/xgboost_results<suffix>.png

Usage
-----
  python step3.2_xgboost_noise_model.py --mode filtered --resolution 1sec
  python step3.2_xgboost_noise_model.py --mode unfiltered --resolution 5min
"""
from __future__ import annotations

import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

TRAIN_DATA_DIR       = Path("training data")
UNFILTERED_DATA_DIR  = Path("regression/predicted")
SUMMARY_CSV          = Path("regression/daily_summary.csv")
OUT_DIR              = Path("regression")

TARGET                  = "residual"
EARLY_STOPPING_ROUNDS   = 100
HUBER_SLOPE              = 100.0   # unfiltered mode only; see module docstring
STORM_RESIDUAL_RANGE_NT = 250.0    # unfiltered mode only; matches step3.1 filter threshold

# Per-resolution knobs: pandas resample rule (None = raw, no resampling),
# minimum rows to keep a day, XGBoost min_child_weight (scaled down for
# coarser resolutions' smaller per-day row counts), and scatter-plot styling.
RESOLUTIONS = {
    "1sec": dict(resample=None,   seconds_per_row=1,   min_rows=60, min_child_weight=15, scatter_size=0.3, scatter_alpha=0.15),
    "1min": dict(resample="1min", seconds_per_row=60,  min_rows=10, min_child_weight=5,   scatter_size=1.5, scatter_alpha=0.2),
    "5min": dict(resample="5min", seconds_per_row=300, min_rows=5,  min_child_weight=3,   scatter_size=3,   scatter_alpha=0.3),
}

# Feature window definitions, always in seconds. At a given resolution, a
# window is only usable if it spans more than one sample (see feature_cols).
ROLL_WINDOWS_SEC        = (30, 300, 900, 1800, 3600, 7200, 14400, 21600, 28800, 43200)
EMA_HALFLIVES_SEC       = (1800, 7200, 21600, 43200)
STD_WINDOWS_SEC         = (30, 300, 900, 3600, 14400)
INTERACTION_WINDOWS_SEC = (30, 300, 900, 1800, 3600, 7200, 14400, 21600)
SQUARE_WINDOWS_SEC      = (3600, 7200, 14400, 21600, 28800, 43200)


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

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

    # Local volatility
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


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# XGBoost params
# ---------------------------------------------------------------------------

def build_xgb_params(resolution: str, mode: str) -> dict:
    params = dict(
        n_estimators      = 5000,
        learning_rate     = 0.01,
        max_depth         = 7,
        subsample         = 0.8,
        colsample_bytree  = 0.7,
        min_child_weight  = RESOLUTIONS[resolution]["min_child_weight"],
        reg_alpha         = 0.05,
        reg_lambda        = 1.5,
        gamma             = 0.02,
        random_state      = 42,
        n_jobs            = -1,
    )
    if mode == "unfiltered":
        params.update(
            objective    = "reg:pseudohubererror",
            huber_slope  = HUBER_SLOPE,
            eval_metric  = "mae",   # more robust than RMSE to storm outliers
        )
    return params


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["filtered", "unfiltered"], default="filtered",
                         help="filtered = quiet-day-only training data; unfiltered = all days, Huber loss")
    parser.add_argument("--resolution", choices=list(RESOLUTIONS), default="1sec",
                         help="resample interval for input data before feature engineering")
    args = parser.parse_args()
    mode, resolution = args.mode, args.resolution

    cfg  = RESOLUTIONS[resolution]
    cols = feature_cols(cfg["seconds_per_row"])
    suffix = ("" if resolution == "1sec" else f"_{resolution}") + ("_unfiltered" if mode == "unfiltered" else "")

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

    # 80/20 chronological day-level split
    n_train_days = int(len(dates) * 0.8)
    train_dates  = set(dates[:n_train_days])
    test_dates   = set(dates[n_train_days:])

    train = full[full["date"].isin(train_dates)].copy()
    test  = full[full["date"].isin(test_dates)].copy()

    print(f"\nChronological 80/20 split:")
    print(f"  Train: {len(train_dates)} days  ({len(train):,} rows)  "
          f"{min(train_dates)} -> {max(train_dates)}")
    print(f"  Test : {len(test_dates)} days  ({len(test):,} rows)  "
          f"{min(test_dates)} -> {max(test_dates)}")

    X_train = train[cols].values
    y_train = train[TARGET].values
    X_test  = test[cols].values
    y_test  = test[TARGET].values

    if mode == "unfiltered":
        # Identify quiet vs storm test days so early stopping (and the
        # headline comparison metric) isn't polluted by storm-day outliers.
        summary = pd.read_csv(SUMMARY_CSV)
        summary["date"] = summary["date"].astype(int).astype(str)
        storm_days = set(summary.loc[summary["residual_range_nT"] > STORM_RESIDUAL_RANGE_NT, "date"])
        test["date_str"] = test["date"].astype(str)
        test_quiet   = test[~test["date_str"].isin(storm_days)]
        n_storm_test = test["date"].nunique() - test_quiet["date"].nunique()
        print(f"  Test quiet days : {test_quiet['date'].nunique()}  |  storm days: {n_storm_test}")
        X_eval, y_eval = test_quiet[cols].values, test_quiet[TARGET].values
    else:
        X_eval, y_eval = X_test, y_test

    xgb_params = build_xgb_params(resolution, mode)
    extra = f"Huber slope={HUBER_SLOPE} nT, early-stop on quiet-day MAE, " if mode == "unfiltered" else ""
    print(f"\nTraining XGBoost  ({extra}"
          f"max_estimators={xgb_params['n_estimators']}, "
          f"max_depth={xgb_params['max_depth']}, "
          f"lr={xgb_params['learning_rate']}, "
          f"early_stopping={EARLY_STOPPING_ROUNDS}) ...")

    model = XGBRegressor(**xgb_params, early_stopping_rounds=EARLY_STOPPING_ROUNDS)
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_eval, y_eval)],
        verbose=100,
    )
    print(f"Best iteration: {model.best_iteration}")

    def metrics(yt, yp):
        return (float(np.sqrt(mean_squared_error(yt, yp))),
                float(mean_absolute_error(yt, yp)),
                float(r2_score(yt, yp)))

    y_pred_train = model.predict(X_train)
    y_pred_test  = model.predict(X_test)
    rmse_tr, mae_tr, r2_tr = metrics(y_train, y_pred_train)
    rmse_te, mae_te, r2_te = metrics(y_test, y_pred_test)

    print(f"\n--- Results ---")
    print(f"  Train  RMSE={rmse_tr:.3f} nT   MAE={mae_tr:.3f} nT   R2={r2_tr:.4f}")
    print(f"  Test   RMSE={rmse_te:.3f} nT   MAE={mae_te:.3f} nT   R2={r2_te:.4f}")

    if mode == "unfiltered":
        y_pred_eval = model.predict(X_eval)
        rmse_q, mae_q, r2_q = metrics(y_eval, y_pred_eval)
        print(f"  Test (quiet)    RMSE={rmse_q:.3f} nT   MAE={mae_q:.3f} nT   R2={r2_q:.4f}  <-- apples-to-apples vs filtered model")

    # Save model
    model_path = OUT_DIR / f"xgboost_noise_model{suffix}.json"
    model.save_model(model_path)
    print(f"\nModel saved -> {model_path}")

    # Save test predictions
    test = test.copy()
    test["Bh_noise_prediction"] = y_pred_test
    pred_out = test[["date", "time", "EZIEH", "ctemp", TARGET, "Bh_noise_prediction"]]
    pred_csv = OUT_DIR / f"xgboost_test_predictions{suffix}.csv"
    pred_out.to_csv(pred_csv, index=False)
    print(f"Test predictions saved -> {pred_csv}")

    # Feature importance
    importances = dict(zip(cols, model.feature_importances_))
    print(f"\nTop feature importances:")
    for feat, imp in sorted(importances.items(), key=lambda x: -x[1])[:10]:
        print(f"  {feat:25s}: {imp:.4f}")

    # --- Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    title = f"XGBoost Bh noise prediction ({mode}"
    if resolution != "1sec":
        title += f", {resolution} resample"
    title += f", {len(dates)} days)  |  "
    if mode == "unfiltered":
        title += f"Test quiet: RMSE={rmse_q:.1f} nT  MAE={mae_q:.1f} nT  R2={r2_q:.3f}"
    else:
        title += f"Test: RMSE={rmse_te:.1f} nT  MAE={mae_te:.1f} nT  R2={r2_te:.3f}"
    fig.suptitle(title, fontsize=11)

    # Predicted vs actual
    ax = axes[0]
    lim = max(abs(y_test).max(), abs(y_pred_test).max()) * 1.05
    clipped = ""
    if mode == "unfiltered":
        lim = min(lim, 500)
        clipped = f"  (clipped to ±{lim:.0f} nT)"
    ax.scatter(y_test, y_pred_test, s=cfg["scatter_size"], alpha=cfg["scatter_alpha"],
               color="#ff7f0e", rasterized=True)
    ax.plot([-lim, lim], [-lim, lim], "k--", linewidth=0.8)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("Actual residual (nT)")
    ax.set_ylabel("Predicted residual (nT)")
    ax.set_title(f"Test set: Predicted vs Actual{clipped}")
    ax.grid(True, linewidth=0.3, alpha=0.5)

    # Feature importance (top 15)
    ax = axes[1]
    sorted_feats = sorted(importances.items(), key=lambda x: x[1])[-15:]
    feat_names = [f for f, _ in sorted_feats]
    feat_imps  = [v for _, v in sorted_feats]
    ax.barh(feat_names, feat_imps, color="#1f77b4")
    ax.set_xlabel("Importance (gain)")
    ax.set_title("Feature Importance (top 15)")
    ax.grid(True, axis="x", linewidth=0.3, alpha=0.5)

    plt.tight_layout()
    plot_path = OUT_DIR / f"xgboost_results{suffix}.png"
    fig.savefig(plot_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Results plot saved -> {plot_path}")


if __name__ == "__main__":
    main()
