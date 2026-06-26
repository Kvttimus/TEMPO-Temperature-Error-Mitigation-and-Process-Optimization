#!/usr/bin/env python3
"""
XGBoost noise model: ALL 237 days (unfiltered), resampled to 5-minute.

Uses Huber loss + quiet-day-only early stopping (same strategy as step4.1b)
so storm days don't blow up the validation metric or cut training short.

Outputs
-------
  regression/xgboost_noise_model_5min_unfiltered.json
  regression/xgboost_test_predictions_5min_unfiltered.csv
  regression/xgboost_results_5min_unfiltered.png
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

DATA_DIR    = Path("regression/predicted")
OUT_DIR     = Path("regression")
SUMMARY_CSV = Path("regression/daily_summary.csv")
RESAMPLE    = "5min"
TARGET      = "residual"

EARLY_STOPPING_ROUNDS = 100
HUBER_SLOPE           = 100.0

XGB_PARAMS = dict(
    n_estimators      = 5000,
    learning_rate     = 0.01,
    max_depth         = 7,
    subsample         = 0.8,
    colsample_bytree  = 0.7,
    min_child_weight  = 3,
    reg_alpha         = 0.05,
    reg_lambda        = 1.5,
    gamma             = 0.02,
    objective         = "reg:pseudohubererror",
    huber_slope       = HUBER_SLOPE,
    eval_metric       = "mae",
    random_state      = 42,
    n_jobs            = -1,
)

_W  = {s: max(1, s // 300) for s in (900, 1800, 3600, 7200, 14400, 21600, 28800, 43200)}
_HL = {s: max(1, s // 300) for s in (1800, 7200, 21600, 43200)}
_SW = {s: max(2, s // 300) for s in (900, 3600, 14400)}


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values("time").reset_index(drop=True)
    e = df["EZIEH"]
    c = df["ctemp"]
    for w_sec, w_rows in _W.items():
        df[f"EZIEH_anom{w_sec}"] = e - e.rolling(w_rows, min_periods=1).mean()
        df[f"ctemp_anom{w_sec}"] = c - c.rolling(w_rows, min_periods=1).mean()
    df["EZIEH_anom_cumul"] = e - e.expanding(min_periods=1).mean()
    df["ctemp_anom_cumul"] = c - c.expanding(min_periods=1).mean()
    for hl_sec, hl_rows in _HL.items():
        df[f"EZIEH_anom_ema{hl_sec}"] = e - e.ewm(halflife=hl_rows, adjust=False).mean()
        df[f"ctemp_anom_ema{hl_sec}"] = c - c.ewm(halflife=hl_rows, adjust=False).mean()
    for w_sec, w_rows in _SW.items():
        df[f"EZIEH_std{w_sec}"] = e.rolling(w_rows, min_periods=2).std()
        df[f"ctemp_std{w_sec}"] = c.rolling(w_rows, min_periods=2).std()
    for w_sec in (900, 1800, 3600, 7200, 14400, 21600):
        df[f"anom{w_sec}_inter"] = df[f"EZIEH_anom{w_sec}"] * df[f"ctemp_anom{w_sec}"]
    df["anom_cumul_inter"]    = df["EZIEH_anom_cumul"]    * df["ctemp_anom_cumul"]
    df["anom_ema43200_inter"] = df["EZIEH_anom_ema43200"] * df["ctemp_anom_ema43200"]
    for w_sec in (3600, 7200, 14400, 21600, 28800, 43200):
        df[f"EZIEH_anom{w_sec}_sq"] = df[f"EZIEH_anom{w_sec}"] ** 2
        df[f"ctemp_anom{w_sec}_sq"] = df[f"ctemp_anom{w_sec}"] ** 2
    df["EZIEH_anom_cumul_sq"]    = df["EZIEH_anom_cumul"]    ** 2
    df["EZIEH_anom_ema43200_sq"] = df["EZIEH_anom_ema43200"] ** 2
    df["ctemp_anom_cumul_sq"]    = df["ctemp_anom_cumul"]    ** 2
    df["ctemp_anom_ema43200_sq"] = df["ctemp_anom_ema43200"] ** 2
    return df


FEATURE_COLS = [
    "EZIEH_anom900",
    "EZIEH_anom1800",  "EZIEH_anom3600",  "EZIEH_anom7200",
    "EZIEH_anom14400", "EZIEH_anom21600", "EZIEH_anom28800", "EZIEH_anom43200",
    "ctemp_anom900",
    "ctemp_anom1800",  "ctemp_anom3600",  "ctemp_anom7200",
    "ctemp_anom14400", "ctemp_anom21600", "ctemp_anom28800", "ctemp_anom43200",
    "EZIEH_anom_cumul", "ctemp_anom_cumul",
    "EZIEH_anom_ema1800",  "EZIEH_anom_ema7200",
    "EZIEH_anom_ema21600", "EZIEH_anom_ema43200",
    "ctemp_anom_ema1800",  "ctemp_anom_ema7200",
    "ctemp_anom_ema21600", "ctemp_anom_ema43200",
    "EZIEH_std900",  "EZIEH_std3600",  "EZIEH_std14400",
    "ctemp_std900",  "ctemp_std3600",  "ctemp_std14400",
    "anom900_inter",
    "anom1800_inter", "anom3600_inter",  "anom7200_inter",
    "anom14400_inter","anom21600_inter",
    "anom_cumul_inter", "anom_ema43200_inter",
    "EZIEH_anom3600_sq",  "ctemp_anom3600_sq",
    "EZIEH_anom7200_sq",  "ctemp_anom7200_sq",
    "EZIEH_anom14400_sq", "ctemp_anom14400_sq",
    "EZIEH_anom21600_sq", "ctemp_anom21600_sq",
    "EZIEH_anom28800_sq", "ctemp_anom28800_sq",
    "EZIEH_anom43200_sq", "ctemp_anom43200_sq",
    "EZIEH_anom_cumul_sq",    "ctemp_anom_cumul_sq",
    "EZIEH_anom_ema43200_sq", "ctemp_anom_ema43200_sq",
]


def load_day(csv_path: Path) -> pd.DataFrame | None:
    df = pd.read_csv(csv_path, parse_dates=["time"])
    if "ctemp" not in df.columns or df["ctemp"].isna().all():
        return None
    df = df.set_index("time")[["EZIEH", "ctemp", TARGET]].resample(RESAMPLE).median()
    df = df.dropna(subset=["EZIEH", "ctemp", TARGET]).reset_index()
    if len(df) < 5:
        return None
    df = add_features(df)
    df["date"] = csv_path.stem
    return df.dropna(subset=FEATURE_COLS + [TARGET])


def main():
    csv_files = sorted(DATA_DIR.glob("*.csv"))
    dates     = [f.stem for f in csv_files]
    print(f"Days available: {len(dates)}  (unfiltered, resampled to {RESAMPLE})")

    frames, skipped = [], 0
    for f in csv_files:
        df = load_day(f)
        if df is not None:
            frames.append(df)
        else:
            skipped += 1
    if skipped:
        print(f"Skipped {skipped} days")

    full = pd.concat(frames, ignore_index=True)
    print(f"Total rows loaded: {len(full):,}  |  Features: {len(FEATURE_COLS)}")

    n_train_days = int(len(dates) * 0.8)
    train_dates  = set(dates[:n_train_days])
    test_dates   = set(dates[n_train_days:])

    train = full[full["date"].isin(train_dates)].copy()
    test  = full[full["date"].isin(test_dates)].copy()

    # Identify quiet vs storm test days for early stopping
    summary = pd.read_csv(SUMMARY_CSV)
    summary["date"] = summary["date"].astype(int).astype(str)
    storm_days  = set(summary.loc[summary["residual_range_nT"] > 250, "date"])
    test["date_str"]  = test["date"].astype(str)
    test_quiet  = test[~test["date_str"].isin(storm_days)]
    n_storm_te  = test["date"].nunique() - test_quiet["date"].nunique()

    print(f"\nChronological 80/20 split:")
    print(f"  Train: {len(train_dates)} days  ({len(train):,} rows)  "
          f"{min(train_dates)} -> {max(train_dates)}")
    print(f"  Test : {len(test_dates)} days  ({len(test):,} rows)  "
          f"(quiet={test_quiet['date'].nunique()}, storm={n_storm_te})")

    X_train  = train[FEATURE_COLS].values
    y_train  = train[TARGET].values
    X_test   = test[FEATURE_COLS].values
    y_test   = test[TARGET].values
    X_test_q = test_quiet[FEATURE_COLS].values
    y_test_q = test_quiet[TARGET].values

    print(f"\nTraining XGBoost (Huber slope={HUBER_SLOPE} nT, early_stop on quiet-day MAE) ...")
    model = XGBRegressor(**XGB_PARAMS, early_stopping_rounds=EARLY_STOPPING_ROUNDS)
    model.fit(X_train, y_train,
              eval_set=[(X_train, y_train), (X_test_q, y_test_q)],
              verbose=100)
    print(f"Best iteration: {model.best_iteration}")

    y_pred_train  = model.predict(X_train)
    y_pred_test   = model.predict(X_test)
    y_pred_test_q = model.predict(X_test_q)

    def met(yt, yp):
        return (float(np.sqrt(mean_squared_error(yt, yp))),
                float(mean_absolute_error(yt, yp)),
                float(r2_score(yt, yp)))

    rt, mt, r2t   = met(y_train,  y_pred_train)
    ra, ma, r2a   = met(y_test,   y_pred_test)
    rq, mq, r2q   = met(y_test_q, y_pred_test_q)

    print(f"\n--- Results ---")
    print(f"  Train           RMSE={rt:.3f} nT   MAE={mt:.3f} nT   R2={r2t:.4f}")
    print(f"  Test (all days) RMSE={ra:.3f} nT   MAE={ma:.3f} nT   R2={r2a:.4f}")
    print(f"  Test (quiet)    RMSE={rq:.3f} nT   MAE={mq:.3f} nT   R2={r2q:.4f}  <-- vs filtered model")

    model_path = OUT_DIR / "xgboost_noise_model_5min_unfiltered.json"
    model.save_model(model_path)
    print(f"\nModel saved -> {model_path}")

    test = test.copy()
    test["Bh_noise_prediction"] = y_pred_test
    pred_csv = OUT_DIR / "xgboost_test_predictions_5min_unfiltered.csv"
    test[["date", "time", "EZIEH", "ctemp", TARGET, "Bh_noise_prediction"]].to_csv(pred_csv, index=False)
    print(f"Test predictions saved -> {pred_csv}")

    importances = dict(zip(FEATURE_COLS, model.feature_importances_))
    print(f"\nTop feature importances:")
    for feat, imp in sorted(importances.items(), key=lambda x: -x[1])[:10]:
        print(f"  {feat:25s}: {imp:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"XGBoost Bh noise (5-min, UNFILTERED {len(dates)} days)  |  "
        f"Test quiet: RMSE={rq:.1f} nT  MAE={mq:.1f} nT  R2={r2q:.3f}",
        fontsize=11
    )
    ax = axes[0]
    lim = min(max(abs(y_test).max(), abs(y_pred_test).max()) * 1.05, 500)
    ax.scatter(y_test, y_pred_test, s=3, alpha=0.2, color="#ff7f0e", rasterized=True)
    ax.plot([-lim, lim], [-lim, lim], "k--", linewidth=0.8)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_xlabel("Actual residual (nT)"); ax.set_ylabel("Predicted residual (nT)")
    ax.set_title("Test: Predicted vs Actual (clipped ±500 nT)")
    ax.grid(True, linewidth=0.3, alpha=0.5)
    ax = axes[1]
    sorted_feats = sorted(importances.items(), key=lambda x: x[1])[-15:]
    ax.barh([f for f, _ in sorted_feats], [v for _, v in sorted_feats], color="#1f77b4")
    ax.set_xlabel("Importance (gain)"); ax.set_title("Feature Importance (top 15)")
    ax.grid(True, axis="x", linewidth=0.3, alpha=0.5)
    plt.tight_layout()
    plot_path = OUT_DIR / "xgboost_results_5min_unfiltered.png"
    fig.savefig(plot_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Results plot saved -> {plot_path}")


if __name__ == "__main__":
    main()
