"""
95% confidence intervals and significance tests for the test-set errors of
no correction vs the linear baseline (step3.2b) vs XGBoost (step3.2),
1-second, 1-minute and 5-minute resolution, filtered and unfiltered. Each
mode is scored on all of its own test days; filtered and unfiltered have
different test days (see regression/split_days_<mode>_<res>.csv), so compare
models within a mode, not across modes.

Method (day-level bootstrap, paired tests): see _bootstrap.py.

Inputs  (regression/, written by step3.2 and step3.2b):
  xgboost_test_predictions<_res>.csv, linear_baseline_test_predictions<_res>.csv
Outputs (regression/):
  bootstrap_ci_<tag>.csv            model, metric, value, ci_low, ci_high
  bootstrap_differences_<tag>.csv   comparison, metric, difference, ci_low, ci_high, p_value, significant
  model_comparison_ci_<tag>.png     bar chart with 95% CI error bars
  (<tag> = <res> for filtered, <res>_unfiltered for unfiltered)
"""
from __future__ import annotations

import pandas as pd
from pathlib import Path

from _bootstrap import NO_CORRECTION, run_comparison

OUT_DIR = Path("regression")

# (output tag, prediction-file suffix, mode, resolution)
JOBS = [
    ("1sec",            "",                  "filtered",   "1sec"),
    ("1min",            "_1min",             "filtered",   "1min"),
    ("5min",            "_5min",             "filtered",   "5min"),
    ("1sec_unfiltered", "_unfiltered",       "unfiltered", "1sec"),
    ("1min_unfiltered", "_1min_unfiltered",  "unfiltered", "1min"),
    ("5min_unfiltered", "_5min_unfiltered",  "unfiltered", "5min"),
]

LINEAR  = "Linear (ctemp + EZIEH)"
XGBOOST = "XGBoost"
COMPARISONS = [(XGBOOST, LINEAR),
               (XGBOOST, NO_CORRECTION),
               (LINEAR, NO_CORRECTION)]

# Categorical palette colors 1-3 (checked for contrast on a light background;
# aqua is below 3:1 contrast, so every bar also carries a text label + value).
MODEL_COLORS = {NO_CORRECTION: "#2a78d6", LINEAR: "#eb6834", XGBOOST: "#1baf7a"}


def main():
    pd.set_option("display.width", 140)
    for tag, suffix, mode, res in JOBS:
        pred_files = {
            NO_CORRECTION: None,
            LINEAR:        OUT_DIR / f"linear_baseline_test_predictions{suffix}.csv",
            XGBOOST:       OUT_DIR / f"xgboost_test_predictions{suffix}.csv",
        }
        run_comparison(pred_files, COMPARISONS, MODEL_COLORS, f"{mode}, {res}",
                       OUT_DIR / f"bootstrap_ci_{tag}.csv",
                       OUT_DIR / f"bootstrap_differences_{tag}.csv",
                       OUT_DIR / f"model_comparison_ci_{tag}.png")


if __name__ == "__main__":
    main()
