#!/usr/bin/env python3
"""
XGBoost noise model trained on filtered data (201 days) resampled to 5-minute.

Window sizes are converted from seconds to row counts at 5-min resolution.
Sub-sample windows (30s, 300s < 5-min interval) are excluded from FEATURE_COLS.

Outputs
-------
  regression/xgboost_noise_model_5min.json
  regression/xgboost_test_predictions_5min.csv
  regression/xgboost_results_5min.png
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

TRAIN_DATA_DIR = Path("training data")
OUT_DIR        = Path("regression")
RESAMPLE       = "5min"
TARGET         = "residual"

EARLY_STOPPING_ROUNDS = 100

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
    random_state      = 42,
    n_jobs            = -1,
)

# Window sizes (seconds) → row counts at 5-min (300s per row)
_W  = {s: max(1, s // 300) for s in (900, 1800, 3600, 7200, 14400, 21600, 28800, 43200)}
_HL = {s: max(1, s // 300) for s in (1800, 7200, 21600, 43200)}
_SW = {s: max(2, s // 300) for s in (900, 3600, 14400)}


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Anomaly features using row-count windows at 5-minute resolution.
    30s and 300s windows excluded (sub-sample at 5-min)."""
    df = df.copy().sort_values("time").reset_index(drop=True)

    e = df["EZIEH"]
    c = df["ctemp"]

    # Rolling anomalies (900s–43200s in rows)
    for w_sec, w_rows in _W.items():
        df[f"EZIEH_anom{w_sec}"] = e - e.rolling(w_rows, min_periods=1).mean()
        df[f"ctemp_anom{w_sec}"] = c - c.rolling(w_rows, min_periods=1).mean()

    # Cumulative mean (midnight-anchored)
    df["EZIEH_anom_cumul"] = e - e.expanding(min_periods=1).mean()
    df["ctemp_anom_cumul"] = c - c.expanding(min_periods=1).mean()

    # EMA anomalies (halflife in rows)
    for hl_sec, hl_rows in _HL.items():
        df[f"EZIEH_anom_ema{hl_sec}"] = e - e.ewm(halflife=hl_rows, adjust=False).mean()
        df[f"ctemp_anom_ema{hl_sec}"] = c - c.ewm(halflife=hl_rows, adjust=False).mean()

    # Volatility (900s+ only)
    for w_sec, w_rows in _SW.items():
        df[f"EZIEH_std{w_sec}"] = e.rolling(w_rows, min_periods=2).std()
        df[f"ctemp_std{w_sec}"] = c.rolling(w_rows, min_periods=2).std()

    # Interactions (900s+)
    for w_sec in (900, 1800, 3600, 7200, 14400, 21600):
        df[f"anom{w_sec}_inter"] = df[f"EZIEH_anom{w_sec}"] * df[f"ctemp_anom{w_sec}"]
    df["anom_cumul_inter"]    = df["EZIEH_anom_cumul"]    * df["ctemp_anom_cumul"]
    df["anom_ema43200_inter"] = df["EZIEH_anom_ema43200"] * df["ctemp_anom_ema43200"]

    # Non-linear terms
    for w_sec in (3600, 7200, 14400, 21600, 28800, 43200):
        df[f"EZIEH_anom{w_sec}_sq"] = df[f"EZIEH_anom{w_sec}"] ** 2
        df[f"ctemp_anom{w_sec}_sq"] = df[f"ctemp_anom{w_sec}"] ** 2
    df["EZIEH_anom_cumul_sq"]    = df["EZIEH_anom_cumul"]    ** 2
    df["EZIEH_anom_ema43200_sq"] = df["EZIEH_anom_ema43200"] ** 2
    df["ctemp_anom_cumul_sq"]    = df["ctemp_anom_cumul"]    ** 2
    df["ctemp_anom_ema43200_sq"] = df["ctemp_anom_ema43200"] ** 2

    return df


FEATURE_COLS = [
    # Rolling anomalies (900s–43200s; 30s and 300s excluded at 5-min)
    "EZIEH_anom900",
    "EZIEH_anom1800",  "EZIEH_anom3600",  "EZIEH_anom7200",
    "EZIEH_anom14400", "EZIEH_anom21600", "EZIEH_anom28800", "EZIEH_anom43200",
    "ctemp_anom900",
    "ctemp_anom1800",  "ctemp_anom3600",  "ctemp_anom7200",
    "ctemp_anom14400", "ctemp_anom21600", "ctemp_anom28800", "ctemp_anom43200",
    # Cumulative
    "EZIEH_anom_cumul", "ctemp_anom_cumul",
    # EMA
    "EZIEH_anom_ema1800",  "EZIEH_anom_ema7200",
    "EZIEH_anom_ema21600", "EZIEH_anom_ema43200",
    "ctemp_anom_ema1800",  "ctemp_anom_ema7200",
    "ctemp_anom_ema21600", "ctemp_anom_ema43200",
    # Volatility (900s+ only)
    "EZIEH_std900",  "EZIEH_std3600",  "EZIEH_std14400",
    "ctemp_std900",  "ctemp_std3600",  "ctemp_std14400",
    # Interactions
    "anom900_inter",
    "anom1800_inter", "anom3600_inter",  "anom7200_inter",
    "anom14400_inter","anom21600_inter",
    "anom_cumul_inter", "anom_ema43200_inter",
    # Non-linear
    "EZIEH_anom3600_sq",  "ctemp_anom3600_sq",
    "EZIEH_anom7200_sq",  "ctemp_anom7200_sq",
    "EZIEH_anom14400_sq", "ctemp_anom14400_sq",
    "EZIEH_anom21600_sq", "ctemp_anom21600_sq",
    "EZIEH_anom28800_sq", "ctemp_anom28800_sq",
    "EZIEH_anom43200_sq", "ctemp_anom43200_sq",
    "EZIEH_anom_cumul_sq",    "ctemp_anom_cumul_sq",
    "EZIEH_anom_ema43200_sq", "ctemp_anom_ema43200_sq",
]


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_day(date_str: str) -> pd.DataFrame | None:
    path = TRAIN_DATA_DIR / f"{date_str}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, parse_dates=["time"])
    if "ctemp" not in df.columns or df["ctemp"].isna().all():
        return None

    # Downsample to 5-minute
    df = df.set_index("time")[["EZIEH", "ctemp", TARGET]].resample(RESAMPLE).median()
    df = df.dropna(subset=["EZIEH", "ctemp", TARGET]).reset_index()
    if len(df) < 5:
        return None

    df = add_features(df)
    df["date"] = date_str
    return df.dropna(subset=FEATURE_COLS + [TARGET])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    dates = sorted(p.stem for p in TRAIN_DATA_DIR.glob("*.csv"))
    print(f"Days available: {len(dates)}  (filtered, resampled to {RESAMPLE})")

    frames, skipped = [], 0
    for date_str in dates:
        df = load_day(date_str)
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

    print(f"\nChronological 80/20 split:")
    print(f"  Train: {len(train_dates)} days  ({len(train):,} rows)  "
          f"{min(train_dates)} -> {max(train_dates)}")
    print(f"  Test : {len(test_dates)} days  ({len(test):,} rows)  "
          f"{min(test_dates)} -> {max(test_dates)}")

    X_train = train[FEATURE_COLS].values
    y_train = train[TARGET].values
    X_test  = test[FEATURE_COLS].values
    y_test  = test[TARGET].values

    print(f"\nTraining XGBoost (5-min data, {len(FEATURE_COLS)} features) ...")
    model = XGBRegressor(**XGB_PARAMS, early_stopping_rounds=EARLY_STOPPING_ROUNDS)
    model.fit(X_train, y_train,
              eval_set=[(X_train, y_train), (X_test, y_test)],
              verbose=100)
    print(f"Best iteration: {model.best_iteration}")

    y_pred_train = model.predict(X_train)
    y_pred_test  = model.predict(X_test)

    rmse_tr = float(np.sqrt(mean_squared_error(y_train, y_pred_train)))
    rmse_te = float(np.sqrt(mean_squared_error(y_test,  y_pred_test)))
    mae_tr  = float(mean_absolute_error(y_train, y_pred_train))
    mae_te  = float(mean_absolute_error(y_test,  y_pred_test))
    r2_tr   = float(r2_score(y_train, y_pred_train))
    r2_te   = float(r2_score(y_test,  y_pred_test))

    print(f"\n--- Results ---")
    print(f"  Train  RMSE={rmse_tr:.3f} nT   MAE={mae_tr:.3f} nT   R2={r2_tr:.4f}")
    print(f"  Test   RMSE={rmse_te:.3f} nT   MAE={mae_te:.3f} nT   R2={r2_te:.4f}")

    model_path = OUT_DIR / "xgboost_noise_model_5min.json"
    model.save_model(model_path)
    print(f"\nModel saved -> {model_path}")

    test = test.copy()
    test["Bh_noise_prediction"] = y_pred_test
    pred_csv = OUT_DIR / "xgboost_test_predictions_5min.csv"
    test[["date", "time", "EZIEH", "ctemp", TARGET, "Bh_noise_prediction"]].to_csv(pred_csv, index=False)
    print(f"Test predictions saved -> {pred_csv}")

    importances = dict(zip(FEATURE_COLS, model.feature_importances_))
    print(f"\nTop feature importances:")
    for feat, imp in sorted(importances.items(), key=lambda x: -x[1])[:10]:
        print(f"  {feat:25s}: {imp:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"XGBoost Bh noise (5-min resample, filtered)  |  "
        f"Test: RMSE={rmse_te:.1f} nT  MAE={mae_te:.1f} nT  R2={r2_te:.3f}",
        fontsize=11
    )
    ax = axes[0]
    lim = max(abs(y_test).max(), abs(y_pred_test).max()) * 1.05
    ax.scatter(y_test, y_pred_test, s=3, alpha=0.3, color="#ff7f0e", rasterized=True)
    ax.plot([-lim, lim], [-lim, lim], "k--", linewidth=0.8)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_xlabel("Actual residual (nT)"); ax.set_ylabel("Predicted residual (nT)")
    ax.set_title("Test: Predicted vs Actual"); ax.grid(True, linewidth=0.3, alpha=0.5)

    ax = axes[1]
    sorted_feats = sorted(importances.items(), key=lambda x: x[1])[-15:]
    ax.barh([f for f, _ in sorted_feats], [v for _, v in sorted_feats], color="#1f77b4")
    ax.set_xlabel("Importance (gain)"); ax.set_title("Feature Importance (top 15)")
    ax.grid(True, axis="x", linewidth=0.3, alpha=0.5)

    plt.tight_layout()
    plot_path = OUT_DIR / "xgboost_results_5min.png"
    fig.savefig(plot_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Results plot saved -> {plot_path}")


if __name__ == "__main__":
    main()
