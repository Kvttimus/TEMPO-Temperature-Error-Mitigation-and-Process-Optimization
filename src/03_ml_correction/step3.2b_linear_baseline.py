"""
Linear baseline for the step3.2 XGBoost EZIEH_noise_ref model.

Linear model on the same two raw inputs XGBoost uses:
    residual = m_ctemp * ctemp + m_ezieh * EZIEH + b
Ordinary least squares on the training days. XGBoost sees these same two
signals (via engineered features), so the difference between the two models
is the model itself, not the information available to it.

Same target (residual), same days and same chronological 60/20/20 split
(_training_data.py) as step3.2; only the model differs. There are no
hyperparameters, so the validation days are reported but not used for
fitting. Test days are scored once, at the end.

Outputs (same suffix convention as step3.2):
  regression/linear_baseline_test_predictions<suffix>.csv
  regression/linear_baseline_metrics<suffix>.csv

Usage
-----
  python step3.2b_linear_baseline.py --mode filtered --resolution 5min
"""
from __future__ import annotations

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from _training_data import RESOLUTIONS, TARGET, load_splits

OUT_DIR = Path("regression")


def metrics(yt, yp):
    return (float(np.sqrt(mean_squared_error(yt, yp))),
            float(mean_absolute_error(yt, yp)),
            float(r2_score(yt, yp)))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["filtered", "unfiltered"], default="filtered")
    parser.add_argument("--resolution", choices=list(RESOLUTIONS), default="1sec")
    args = parser.parse_args()
    mode, resolution = args.mode, args.resolution
    suffix = ("" if resolution == "1sec" else f"_{resolution}") + ("_unfiltered" if mode == "unfiltered" else "")

    train, val, test, _, _ = load_splits(mode, resolution)

    def design(df):
        return np.column_stack([df["ctemp"].values, df["EZIEH"].values, np.ones(len(df))])

    (m_ctemp, m_ezieh, b), *_ = np.linalg.lstsq(design(train), train[TARGET].values, rcond=None)
    print(f"\nFit: residual = {m_ctemp:.4f} * ctemp {m_ezieh:+.6f} * EZIEH {b:+.3f}")

    def predict(df):
        return design(df) @ np.array([m_ctemp, m_ezieh, b])

    rows = [("train", *metrics(train[TARGET].values, predict(train))),
            ("val",   *metrics(val[TARGET].values,   predict(val))),
            ("test",  *metrics(test[TARGET].values,  predict(test)))]

    print("\n--- Results (linear baseline) ---")
    for split, rmse, mae, r2 in rows:
        print(f"  {split:11s} RMSE={rmse:.3f} nT   MAE={mae:.3f} nT   R2={r2:.4f}")

    metrics_df = pd.DataFrame(rows, columns=["split", "rmse_nT", "mae_nT", "r2"])
    metrics_df["m_ctemp"] = m_ctemp
    metrics_df["m_ezieh"] = m_ezieh
    metrics_df["b"]       = b
    metrics_csv = OUT_DIR / f"linear_baseline_metrics{suffix}.csv"
    metrics_df.to_csv(metrics_csv, index=False)
    print(f"\nMetrics saved -> {metrics_csv}")

    test["Bh_noise_prediction"] = predict(test)
    pred_csv = OUT_DIR / f"linear_baseline_test_predictions{suffix}.csv"
    test[["date", "time", "EZIEH", "ctemp", TARGET, "Bh_noise_prediction"]].to_csv(pred_csv, index=False)
    print(f"Test predictions saved -> {pred_csv}")


if __name__ == "__main__":
    main()
