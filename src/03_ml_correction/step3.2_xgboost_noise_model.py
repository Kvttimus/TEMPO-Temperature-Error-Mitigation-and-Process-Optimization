"""
XGBoost model: predict EZIEH_noise_ref (column `residual`) from ctemp and EZIEH only.

During training the pre-calculated residual (EZIEH - EZIEH_ref) is used
as the target.  At inference time ONLY ctemp and EZIEH are needed — all derived
features (rolling, cumulative and EMA anomalies, rolling std, interactions,
squared terms) are computed from those two signals.

Features engineered from ctemp + EZIEH (window sizes are always expressed in
seconds; windows not exceeding the sample interval are dropped, e.g. the 30s
window is skipped at 1-min resolution):
  Rolling anomalies (30s -> 12h) : EZIEH_anom<w>, ctemp_anom<w>
  Cumulative (midnight-anchored) : EZIEH_anom_cumul, ctemp_anom_cumul
  EMA anomalies                  : EZIEH_anom_ema<hl>, ctemp_anom_ema<hl>
  Rolling std (volatility)       : EZIEH_std<w>, ctemp_std<w>
  Interactions                   : anom<w>_inter
  Non-linear                     : EZIEH_anom<w>_sq, ctemp_anom<w>_sq

Three axes of variation, selected via CLI flags:
  --mode filtered|unfiltered
      filtered   - trains on the high-residual-days-removed data in
                   'training data/' with a standard MSE objective.
      unfiltered - an "all the data" model: trains, early-stops, and is
                   scored on every day in 'regression/predicted/', with no
                   day singled out. Uses a Huber loss (reg:pseudohubererror)
                   so extreme residuals don't dominate the gradient.
  --resolution 1sec|1min|5min
      Resamples input to the given interval (median) before feature
      engineering; sub-sample windows are dropped and min_child_weight is
      scaled down for the smaller per-day row counts at coarser resolutions.
  --features all|ctemp|ezieh
      Input ablation. 'all' (default) is the full TEMPO model. 'ctemp' keeps
      only features derived from ctemp, 'ezieh' only those derived from
      EZIEH; the ctemp x EZIEH interaction features use both signals and are
      dropped from both single-input variants. Everything else (params,
      split, early stopping) is unchanged.

Split: chronological 60/20/20 at the day level (see _training_data.py).
Early stopping watches the validation days only; the test days are never
seen until the final evaluation.

Outputs (suffix encodes resolution + mode + ablation, e.g. _5min_unfiltered,
_1min_ctemp_only; the 1sec+filtered+all combination — the original/default
model — has no suffix):
  regression/xgboost_noise_model<suffix>.json
  regression/xgboost_test_predictions<suffix>.csv
  regression/xgboost_results<suffix>.png

Usage
-----
  python step3.2_xgboost_noise_model.py --mode filtered --resolution 1sec
  python step3.2_xgboost_noise_model.py --mode unfiltered --resolution 5min
  python step3.2_xgboost_noise_model.py --mode filtered --resolution 1min --features ctemp
"""
from __future__ import annotations

import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from _training_data import RESOLUTIONS, TARGET, load_splits

OUT_DIR = Path("regression")

EARLY_STOPPING_ROUNDS = 100
HUBER_SLOPE           = 100.0   # unfiltered mode only; see module docstring

# --features ablation: which engineered columns each variant keeps
FEATURE_SETS = {
    "all":   lambda c: True,
    "ctemp": lambda c: c.startswith("ctemp_"),
    "ezieh": lambda c: c.startswith("EZIEH_"),
}

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
            eval_metric  = "mae",   # more robust than RMSE to extreme residuals
        )
    return params


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["filtered", "unfiltered"], default="filtered",
                         help="filtered = high-residual days removed; unfiltered = all days, Huber loss")
    parser.add_argument("--resolution", choices=list(RESOLUTIONS), default="1sec",
                         help="resample interval for input data before feature engineering")
    parser.add_argument("--features", choices=list(FEATURE_SETS), default="all",
                         help="input ablation: all = full model; ctemp / ezieh = features from that signal only")
    args = parser.parse_args()
    mode, resolution = args.mode, args.resolution

    cfg    = RESOLUTIONS[resolution]
    suffix = (("" if resolution == "1sec" else f"_{resolution}")
              + ("_unfiltered" if mode == "unfiltered" else "")
              + ("" if args.features == "all" else f"_{args.features}_only"))

    train, val, test, cols, dates = load_splits(mode, resolution)
    cols = [c for c in cols if FEATURE_SETS[args.features](c)]
    print(f"Feature set: {args.features}  ({len(cols)} features)")

    X_train, y_train = train[cols].values, train[TARGET].values
    X_test,  y_test  = test[cols].values,  test[TARGET].values
    # Early stopping sees the validation days only; test stays untouched.
    X_eval,  y_eval  = val[cols].values,   val[TARGET].values

    xgb_params = build_xgb_params(resolution, mode)
    extra = f"Huber slope={HUBER_SLOPE} nT, early-stop on val MAE, " if mode == "unfiltered" else ""
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
    y_pred_eval  = model.predict(X_eval)
    y_pred_test  = model.predict(X_test)
    rmse_tr, mae_tr, r2_tr = metrics(y_train, y_pred_train)
    rmse_va, mae_va, r2_va = metrics(y_eval,  y_pred_eval)
    rmse_te, mae_te, r2_te = metrics(y_test,  y_pred_test)

    print("\n--- Results ---")
    print(f"  Train  RMSE={rmse_tr:.3f} nT   MAE={mae_tr:.3f} nT   R2={r2_tr:.4f}")
    print(f"  Val    RMSE={rmse_va:.3f} nT   MAE={mae_va:.3f} nT   R2={r2_va:.4f}")
    print(f"  Test   RMSE={rmse_te:.3f} nT   MAE={mae_te:.3f} nT   R2={r2_te:.4f}")

    # Save model
    model_path = OUT_DIR / f"xgboost_noise_model{suffix}.json"
    model.save_model(model_path)
    print(f"\nModel saved -> {model_path}")

    # Save test predictions
    test["Bh_noise_prediction"] = y_pred_test
    pred_out = test[["date", "time", "EZIEH", "ctemp", TARGET, "Bh_noise_prediction"]]
    pred_csv = OUT_DIR / f"xgboost_test_predictions{suffix}.csv"
    pred_out.to_csv(pred_csv, index=False)
    print(f"Test predictions saved -> {pred_csv}")

    # Feature importance
    importances = dict(zip(cols, model.feature_importances_))
    print("\nTop feature importances:")
    for feat, imp in sorted(importances.items(), key=lambda x: -x[1])[:10]:
        print(f"  {feat:25s}: {imp:.4f}")

    # --- Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    title = f"XGBoost EZIEH_noise_ref prediction ({mode}"
    if args.features != "all":
        title += f", {args.features} features only"
    if resolution != "1sec":
        title += f", {resolution} resample"
    title += f", {len(dates)} days)  |  "
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
    ax.set_xlabel("EZIEH_noise_ref (nT)")
    ax.set_ylabel("EZIEH_noise_pred (nT)")
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
