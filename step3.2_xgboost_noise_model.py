#!/usr/bin/env python3
"""
XGBoost model: predict Bh_noise (residual) from ctemp and EZIEH only.

During training the pre-calculated residual (EZIEH - EZIE_Bh_Predicted) is used
as the target.  At inference time ONLY ctemp and EZIEH are needed — all derived
features (lags, rolling stats, diffs) are computed from those two signals.

Features engineered from ctemp + EZIEH:
  Raw           : EZIEH, ctemp
  Lags (1s,5s)  : EZIEH_lag1, EZIEH_lag5, ctemp_lag1, ctemp_lag5
  Rolling means : EZIEH_roll30, EZIEH_roll300, ctemp_roll30, ctemp_roll300
  Rolling std   : EZIEH_std30, ctemp_std30
  Diffs         : EZIEH_diff1, ctemp_diff1  (rate of change)
  Non-linear    : ctemp_sq, EZIEH_sq

Split: chronological 80/20 at the day level.

Outputs
-------
  regression/xgboost_noise_model.json
  regression/xgboost_test_predictions.csv
  regression/xgboost_results.png
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
EZIE_DIR       = Path("humanReadable_EZIE_data")
OUT_DIR        = Path("regression")

TARGET = "residual"

EARLY_STOPPING_ROUNDS = 100

XGB_PARAMS = dict(
    n_estimators      = 5000,
    learning_rate     = 0.01,
    max_depth         = 7,
    subsample         = 0.8,
    colsample_bytree  = 0.7,
    min_child_weight  = 15,
    reg_alpha         = 0.05,
    reg_lambda        = 1.5,
    gamma             = 0.02,
    random_state      = 42,
    n_jobs            = -1,
)


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Anomaly-based features from ctemp and EZIEH only.
    All features are deviations from causal rolling baselines so absolute
    drift across the mission does not cause train/test distribution shift."""
    df = df.copy().sort_values("time").reset_index(drop=True)

    e = df["EZIEH"]
    c = df["ctemp"]

    # Causal rolling baselines (30s → 12h)
    for w in (30, 300, 900, 1800, 3600, 7200, 14400, 21600, 28800, 43200):
        e_base = e.rolling(w, min_periods=1).mean()
        c_base = c.rolling(w, min_periods=1).mean()
        df[f"EZIEH_anom{w}"] = e - e_base
        df[f"ctemp_anom{w}"] = c - c_base

    # Cumulative-mean baseline (anchored to midnight; better than rolling at late hours)
    e_cumul = e.expanding(min_periods=1).mean()
    c_cumul = c.expanding(min_periods=1).mean()
    df["EZIEH_anom_cumul"] = e - e_cumul
    df["ctemp_anom_cumul"] = c - c_cumul

    # Exponential moving averages at key half-lives
    for hl in (1800, 7200, 21600, 43200):
        df[f"EZIEH_anom_ema{hl}"] = e - e.ewm(halflife=hl, adjust=False).mean()
        df[f"ctemp_anom_ema{hl}"] = c - c.ewm(halflife=hl, adjust=False).mean()

    # Local volatility
    df["EZIEH_std30"]    = e.rolling(30,    min_periods=2).std()
    df["EZIEH_std300"]   = e.rolling(300,   min_periods=2).std()
    df["EZIEH_std900"]   = e.rolling(900,   min_periods=2).std()
    df["EZIEH_std3600"]  = e.rolling(3600,  min_periods=2).std()
    df["EZIEH_std14400"] = e.rolling(14400, min_periods=2).std()
    df["ctemp_std30"]    = c.rolling(30,    min_periods=2).std()
    df["ctemp_std300"]   = c.rolling(300,   min_periods=2).std()
    df["ctemp_std3600"]  = c.rolling(3600,  min_periods=2).std()
    df["ctemp_std14400"] = c.rolling(14400, min_periods=2).std()

    # Interactions at matching timescales
    for w in (30, 300, 900, 1800, 3600, 7200, 14400, 21600):
        df[f"anom{w}_inter"] = df[f"EZIEH_anom{w}"] * df[f"ctemp_anom{w}"]
    df["anom_cumul_inter"]  = df["EZIEH_anom_cumul"]   * df["ctemp_anom_cumul"]
    df["anom_ema43200_inter"] = df["EZIEH_anom_ema43200"] * df["ctemp_anom_ema43200"]

    # Non-linear terms for key long windows
    for w in (3600, 7200, 14400, 21600, 28800, 43200):
        df[f"EZIEH_anom{w}_sq"] = df[f"EZIEH_anom{w}"] ** 2
        df[f"ctemp_anom{w}_sq"] = df[f"ctemp_anom{w}"] ** 2
    df["EZIEH_anom_cumul_sq"]    = df["EZIEH_anom_cumul"]    ** 2
    df["EZIEH_anom_ema43200_sq"] = df["EZIEH_anom_ema43200"] ** 2
    df["ctemp_anom_cumul_sq"]    = df["ctemp_anom_cumul"]    ** 2
    df["ctemp_anom_ema43200_sq"] = df["ctemp_anom_ema43200"] ** 2

    return df


FEATURE_COLS = [
    # Rolling anomalies (30s → 12h)
    "EZIEH_anom30",    "EZIEH_anom300",   "EZIEH_anom900",
    "EZIEH_anom1800",  "EZIEH_anom3600",  "EZIEH_anom7200",
    "EZIEH_anom14400", "EZIEH_anom21600", "EZIEH_anom28800", "EZIEH_anom43200",
    "ctemp_anom30",    "ctemp_anom300",   "ctemp_anom900",
    "ctemp_anom1800",  "ctemp_anom3600",  "ctemp_anom7200",
    "ctemp_anom14400", "ctemp_anom21600", "ctemp_anom28800", "ctemp_anom43200",
    # Cumulative (midnight-anchored) anomalies
    "EZIEH_anom_cumul", "ctemp_anom_cumul",
    # EMA anomalies
    "EZIEH_anom_ema1800",  "EZIEH_anom_ema7200",
    "EZIEH_anom_ema21600", "EZIEH_anom_ema43200",
    "ctemp_anom_ema1800",  "ctemp_anom_ema7200",
    "ctemp_anom_ema21600", "ctemp_anom_ema43200",
    # Volatility
    "EZIEH_std30",  "EZIEH_std300",  "EZIEH_std900",
    "EZIEH_std3600","EZIEH_std14400",
    "ctemp_std30",  "ctemp_std300",  "ctemp_std3600", "ctemp_std14400",
    # Interactions
    "anom30_inter",   "anom300_inter",   "anom900_inter",
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
    pred_path = TRAIN_DATA_DIR / f"{date_str}.csv"
    if not pred_path.exists():
        return None
    df = pd.read_csv(pred_path, parse_dates=["time"])

    if "ctemp" not in df.columns or df["ctemp"].isna().all():
        return None

    df = df[["time", "EZIEH", "ctemp", TARGET]].dropna(
        subset=["EZIEH", "ctemp", TARGET]
    )
    if len(df) < 60:
        return None

    df = add_features(df)
    df["date"] = date_str
    df = df.dropna(subset=FEATURE_COLS + [TARGET])
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    dates = sorted(p.stem for p in TRAIN_DATA_DIR.glob("*.csv"))
    print(f"Days available: {len(dates)}")

    frames = []
    skipped = 0
    for date_str in dates:
        df = load_day(date_str)
        if df is not None:
            frames.append(df)
        else:
            skipped += 1

    if skipped:
        print(f"Skipped {skipped} days (missing ctemp or insufficient data)")

    full = pd.concat(frames, ignore_index=True)
    print(f"Total rows loaded: {len(full):,}  |  Features: {len(FEATURE_COLS)}")

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

    X_train = train[FEATURE_COLS].values
    y_train = train[TARGET].values
    X_test  = test[FEATURE_COLS].values
    y_test  = test[TARGET].values

    # Train
    print(f"\nTraining XGBoost  "
          f"(max_estimators={XGB_PARAMS['n_estimators']}, "
          f"max_depth={XGB_PARAMS['max_depth']}, "
          f"lr={XGB_PARAMS['learning_rate']}, "
          f"early_stopping={EARLY_STOPPING_ROUNDS}) ...")

    model = XGBRegressor(**XGB_PARAMS, early_stopping_rounds=EARLY_STOPPING_ROUNDS)
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=100,
    )
    print(f"Best iteration: {model.best_iteration}")

    # Evaluate
    y_pred_train = model.predict(X_train)
    y_pred_test  = model.predict(X_test)

    rmse_train = float(np.sqrt(mean_squared_error(y_train, y_pred_train)))
    rmse_test  = float(np.sqrt(mean_squared_error(y_test,  y_pred_test)))
    mae_train  = float(mean_absolute_error(y_train, y_pred_train))
    mae_test   = float(mean_absolute_error(y_test,  y_pred_test))
    r2_train   = float(r2_score(y_train, y_pred_train))
    r2_test    = float(r2_score(y_test,  y_pred_test))

    print(f"\n--- Results ---")
    print(f"  Train  RMSE={rmse_train:.3f} nT   MAE={mae_train:.3f} nT   R2={r2_train:.4f}")
    print(f"  Test   RMSE={rmse_test:.3f} nT   MAE={mae_test:.3f} nT   R2={r2_test:.4f}")

    # Save model
    model_path = OUT_DIR / "xgboost_noise_model.json"
    model.save_model(model_path)
    print(f"\nModel saved -> {model_path}")

    # Save test predictions
    test = test.copy()
    test["Bh_noise_prediction"] = y_pred_test
    pred_out = test[["date", "time", "EZIEH", "ctemp", TARGET, "Bh_noise_prediction"]]
    pred_csv = OUT_DIR / "xgboost_test_predictions.csv"
    pred_out.to_csv(pred_csv, index=False)
    print(f"Test predictions saved -> {pred_csv}")

    # Feature importance
    importances = dict(zip(FEATURE_COLS, model.feature_importances_))
    print(f"\nTop feature importances:")
    for feat, imp in sorted(importances.items(), key=lambda x: -x[1])[:10]:
        print(f"  {feat:25s}: {imp:.4f}")

    # --- Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"XGBoost Bh noise prediction  |  "
        f"Test: RMSE={rmse_test:.1f} nT  MAE={mae_test:.1f} nT  R2={r2_test:.3f}",
        fontsize=11
    )

    # Predicted vs actual
    ax = axes[0]
    lim = max(abs(y_test).max(), abs(y_pred_test).max()) * 1.05
    ax.scatter(y_test, y_pred_test, s=0.3, alpha=0.15, color="#ff7f0e", rasterized=True)
    ax.plot([-lim, lim], [-lim, lim], "k--", linewidth=0.8)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("Actual residual (nT)")
    ax.set_ylabel("Predicted residual (nT)")
    ax.set_title("Test set: Predicted vs Actual")
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
    plot_path = OUT_DIR / "xgboost_results.png"
    fig.savefig(plot_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Results plot saved -> {plot_path}")


if __name__ == "__main__":
    main()
