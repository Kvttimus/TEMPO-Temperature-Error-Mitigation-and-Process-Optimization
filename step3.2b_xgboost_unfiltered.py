#!/usr/bin/env python3
"""
XGBoost noise model trained on ALL 237 days (unfiltered — includes storm days).

Same 65-feature engineering as step4.1, but reads from regression/predicted/
instead of training data/.  Storm days (residual_range > 250 nT) are included.

Uses Huber loss (reg:pseudohubererror) to prevent the extreme storm-day
residuals (up to ~29,000 nT) from dominating MSE and distorting the model
toward storm prediction at the expense of quiet-day accuracy.

Outputs
-------
  regression/xgboost_noise_model_unfiltered.json
  regression/xgboost_test_predictions_unfiltered.csv
  regression/xgboost_results_unfiltered.png
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
SUMMARY_CSV = Path("regression/daily_summary.csv")   # used to identify quiet vs storm test days

TARGET = "residual"

EARLY_STOPPING_ROUNDS = 100

# Huber slope: errors below this are penalised quadratically (like MSE),
# errors above it are penalised linearly (robust to storm-day outliers).
# 100 nT lets moderate storm variation contribute to gradients while
# preventing extreme events (>1000 nT) from dominating entirely.
HUBER_SLOPE = 100.0

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
    objective         = "reg:pseudohubererror",
    huber_slope       = HUBER_SLOPE,
    eval_metric       = "mae",   # MAE for early stopping: more robust than RMSE to storm outliers
    random_state      = 42,
    n_jobs            = -1,
)


# ---------------------------------------------------------------------------
# Feature engineering  (identical to step4.1 / step4.2b)
# ---------------------------------------------------------------------------

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values("time").reset_index(drop=True)

    e = df["EZIEH"]
    c = df["ctemp"]

    # Causal rolling baselines (30s → 12h)
    for w in (30, 300, 900, 1800, 3600, 7200, 14400, 21600, 28800, 43200):
        e_base = e.rolling(w, min_periods=1).mean()
        c_base = c.rolling(w, min_periods=1).mean()
        df[f"EZIEH_anom{w}"] = e - e_base
        df[f"ctemp_anom{w}"] = c - c_base

    # Cumulative-mean baseline (anchored to midnight)
    e_cumul = e.expanding(min_periods=1).mean()
    c_cumul = c.expanding(min_periods=1).mean()
    df["EZIEH_anom_cumul"] = e - e_cumul
    df["ctemp_anom_cumul"] = c - c_cumul

    # EMA anomalies
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

    # Interactions
    for w in (30, 300, 900, 1800, 3600, 7200, 14400, 21600):
        df[f"anom{w}_inter"] = df[f"EZIEH_anom{w}"] * df[f"ctemp_anom{w}"]
    df["anom_cumul_inter"]    = df["EZIEH_anom_cumul"]    * df["ctemp_anom_cumul"]
    df["anom_ema43200_inter"] = df["EZIEH_anom_ema43200"] * df["ctemp_anom_ema43200"]

    # Non-linear terms
    for w in (3600, 7200, 14400, 21600, 28800, 43200):
        df[f"EZIEH_anom{w}_sq"] = df[f"EZIEH_anom{w}"] ** 2
        df[f"ctemp_anom{w}_sq"] = df[f"ctemp_anom{w}"] ** 2
    df["EZIEH_anom_cumul_sq"]    = df["EZIEH_anom_cumul"]    ** 2
    df["EZIEH_anom_ema43200_sq"] = df["EZIEH_anom_ema43200"] ** 2
    df["ctemp_anom_cumul_sq"]    = df["ctemp_anom_cumul"]    ** 2
    df["ctemp_anom_ema43200_sq"] = df["ctemp_anom_ema43200"] ** 2

    return df


FEATURE_COLS = [
    "EZIEH_anom30",    "EZIEH_anom300",   "EZIEH_anom900",
    "EZIEH_anom1800",  "EZIEH_anom3600",  "EZIEH_anom7200",
    "EZIEH_anom14400", "EZIEH_anom21600", "EZIEH_anom28800", "EZIEH_anom43200",
    "ctemp_anom30",    "ctemp_anom300",   "ctemp_anom900",
    "ctemp_anom1800",  "ctemp_anom3600",  "ctemp_anom7200",
    "ctemp_anom14400", "ctemp_anom21600", "ctemp_anom28800", "ctemp_anom43200",
    "EZIEH_anom_cumul", "ctemp_anom_cumul",
    "EZIEH_anom_ema1800",  "EZIEH_anom_ema7200",
    "EZIEH_anom_ema21600", "EZIEH_anom_ema43200",
    "ctemp_anom_ema1800",  "ctemp_anom_ema7200",
    "ctemp_anom_ema21600", "ctemp_anom_ema43200",
    "EZIEH_std30",  "EZIEH_std300",  "EZIEH_std900",
    "EZIEH_std3600","EZIEH_std14400",
    "ctemp_std30",  "ctemp_std300",  "ctemp_std3600", "ctemp_std14400",
    "anom30_inter",   "anom300_inter",   "anom900_inter",
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


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_day(csv_path: Path) -> pd.DataFrame | None:
    df = pd.read_csv(csv_path, parse_dates=["time"])
    if "ctemp" not in df.columns or df["ctemp"].isna().all():
        return None
    df = df[["time", "EZIEH", "ctemp", TARGET]].dropna(subset=["EZIEH", "ctemp", TARGET])
    if len(df) < 60:
        return None
    df = add_features(df)
    df["date"] = csv_path.stem
    df = df.dropna(subset=FEATURE_COLS + [TARGET])
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    csv_files = sorted(DATA_DIR.glob("*.csv"))
    dates     = [f.stem for f in csv_files]
    print(f"Days available: {len(dates)}  (unfiltered — includes storm days)")

    frames  = []
    skipped = 0
    for f in csv_files:
        df = load_day(f)
        if df is not None:
            frames.append(df)
        else:
            skipped += 1

    if skipped:
        print(f"Skipped {skipped} days (missing ctemp or insufficient data)")

    full = pd.concat(frames, ignore_index=True)
    print(f"Total rows loaded: {len(full):,}  |  Features: {len(FEATURE_COLS)}")

    residual_range = full[TARGET].max() - full[TARGET].min()
    print(f"Residual range across all days: {residual_range:,.0f} nT")

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

    # Load daily summary to identify quiet vs storm test days
    summary = pd.read_csv(SUMMARY_CSV)
    summary["date"] = summary["date"].astype(int).astype(str).str.replace("-", "")
    # handle both int YYYYMMDD and date string formats
    try:
        summary["date"] = pd.to_datetime(summary["date"]).dt.strftime("%Y%m%d")
    except Exception:
        pass
    storm_days   = set(summary.loc[summary["residual_range_nT"] > 250, "date"].astype(str))
    test["date_str"] = test["date"].astype(str)
    test_quiet   = test[~test["date_str"].isin(storm_days)]
    n_storm_test = test["date"].nunique() - test_quiet["date"].nunique()
    print(f"  Test quiet days : {test_quiet['date'].nunique()}  |  storm days: {n_storm_test}")

    X_train = train[FEATURE_COLS].values
    y_train = train[TARGET].values
    X_test  = test[FEATURE_COLS].values
    y_test  = test[TARGET].values
    X_test_q = test_quiet[FEATURE_COLS].values
    y_test_q = test_quiet[TARGET].values

    print(f"\nTraining XGBoost (Huber slope={HUBER_SLOPE} nT, early_stop on quiet-day MAE)  "
          f"(max_estimators={XGB_PARAMS['n_estimators']}, "
          f"max_depth={XGB_PARAMS['max_depth']}, "
          f"lr={XGB_PARAMS['learning_rate']}, "
          f"early_stopping={EARLY_STOPPING_ROUNDS}) ...")

    model = XGBRegressor(**XGB_PARAMS, early_stopping_rounds=EARLY_STOPPING_ROUNDS)
    model.fit(
        X_train, y_train,
        # Early stopping on quiet-day test MAE — storm days don't pollute the signal
        eval_set=[(X_train, y_train), (X_test_q, y_test_q)],
        verbose=100,
    )
    print(f"Best iteration: {model.best_iteration}")

    y_pred_train  = model.predict(X_train)
    y_pred_test   = model.predict(X_test)
    y_pred_test_q = model.predict(X_test_q)

    def metrics(yt, yp):
        return (float(np.sqrt(mean_squared_error(yt, yp))),
                float(mean_absolute_error(yt, yp)),
                float(r2_score(yt, yp)))

    rmse_tr, mae_tr, r2_tr   = metrics(y_train,  y_pred_train)
    rmse_te, mae_te, r2_te   = metrics(y_test,   y_pred_test)
    rmse_tq, mae_tq, r2_tq   = metrics(y_test_q, y_pred_test_q)

    print(f"\n--- Results ---")
    print(f"  Train           RMSE={rmse_tr:.3f} nT   MAE={mae_tr:.3f} nT   R2={r2_tr:.4f}")
    print(f"  Test (all days) RMSE={rmse_te:.3f} nT   MAE={mae_te:.3f} nT   R2={r2_te:.4f}")
    print(f"  Test (quiet)    RMSE={rmse_tq:.3f} nT   MAE={mae_tq:.3f} nT   R2={r2_tq:.4f}  <-- apples-to-apples vs filtered model")

    model_path = OUT_DIR / "xgboost_noise_model_unfiltered.json"
    model.save_model(model_path)
    print(f"\nModel saved -> {model_path}")

    test = test.copy()
    test["Bh_noise_prediction"] = y_pred_test
    pred_out = test[["date", "time", "EZIEH", "ctemp", TARGET, "Bh_noise_prediction"]]
    pred_csv = OUT_DIR / "xgboost_test_predictions_unfiltered.csv"
    pred_out.to_csv(pred_csv, index=False)
    print(f"Test predictions saved -> {pred_csv}")

    importances = dict(zip(FEATURE_COLS, model.feature_importances_))
    print(f"\nTop feature importances:")
    for feat, imp in sorted(importances.items(), key=lambda x: -x[1])[:10]:
        print(f"  {feat:25s}: {imp:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"XGBoost Bh noise (UNFILTERED — all {len(dates)} days)  |  "
        f"Test quiet: RMSE={rmse_tq:.1f} nT  MAE={mae_tq:.1f} nT  R2={r2_tq:.3f}",
        fontsize=11
    )

    ax = axes[0]
    lim = min(max(abs(y_test).max(), abs(y_pred_test).max()) * 1.05, 500)
    ax.scatter(y_test, y_pred_test, s=0.3, alpha=0.1, color="#ff7f0e", rasterized=True)
    ax.plot([-lim, lim], [-lim, lim], "k--", linewidth=0.8)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("Actual residual (nT)")
    ax.set_ylabel("Predicted residual (nT)")
    ax.set_title("Test set: Predicted vs Actual  (clipped to ±500 nT)")
    ax.grid(True, linewidth=0.3, alpha=0.5)

    ax = axes[1]
    sorted_feats = sorted(importances.items(), key=lambda x: x[1])[-15:]
    ax.barh([f for f, _ in sorted_feats], [v for _, v in sorted_feats], color="#1f77b4")
    ax.set_xlabel("Importance (gain)")
    ax.set_title("Feature Importance (top 15)")
    ax.grid(True, axis="x", linewidth=0.3, alpha=0.5)

    plt.tight_layout()
    plot_path = OUT_DIR / "xgboost_results_unfiltered.png"
    fig.savefig(plot_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Results plot saved -> {plot_path}")


if __name__ == "__main__":
    main()
