"""
Input ablation for the TEMPO XGBoost model.

Compares three XGBoost models that are identical except for their inputs:
  XGBoost (full)        - every engineered feature (the TEMPO model)
  XGBoost (ctemp only)  - only features derived from ctemp
  XGBoost (EZIEH only)  - only features derived from EZIEH
The ctemp x EZIEH interaction features use both signals, so they are dropped
from both single-input variants. No correction is included as a reference.

Each variant is trained by step3.2 (--features all|ctemp|ezieh) so hyper-
parameters, the 60/20/20 split, and early stopping are shared exactly. Test
errors (RMSE / MAE / R2) get 95% CIs and paired significance tests from the
same day-level bootstrap as step3.6 (see _bootstrap.py).

Outputs (regression/; <tag> = <res> for filtered, <res>_unfiltered otherwise):
  xgboost_*_{ctemp,ezieh}_only.*   models / predictions / plots (from step3.2)
  ablation_ci_<tag>.csv            model, metric, value, ci_low, ci_high
  ablation_differences_<tag>.csv   comparison, metric, difference, ci_low, ci_high, p_value, significant
  ablation_ci_<tag>.png            bar chart with 95% CI error bars

Usage
-----
  python src/03_ml_correction/step3.7_feature_ablation.py                     # train + analyze, filtered, all resolutions
  python src/03_ml_correction/step3.7_feature_ablation.py --resolutions 5min
  python src/03_ml_correction/step3.7_feature_ablation.py --analyze-only      # reuse existing step3.2 outputs
  python src/03_ml_correction/step3.7_feature_ablation.py --mode unfiltered --analyze-only --by-residual-range

--by-residual-range additionally scores the (already trained) models on two
subsets of the test days — residual range <= 250 nT and > 250 nT (1-min
medians, the step3.1 threshold) — comparing full vs EZIEH-only vs no
correction within each. Outputs: ablation_{ci,differences}_<tag>_{normal,high_residual}.*
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import pandas as pd
from pathlib import Path

from _bootstrap import NO_CORRECTION, run_comparison
from _training_data import RESOLUTIONS

OUT_DIR  = Path("regression")
STEP3_2  = Path(__file__).with_name("step3.2_xgboost_noise_model.py")

FULL  = "XGBoost (full)"
CTEMP = "XGBoost (ctemp only)"
EZIEH = "XGBoost (EZIEH only)"
VARIANTS = {FULL: "all", CTEMP: "ctemp", EZIEH: "ezieh"}   # model name -> step3.2 --features

COMPARISONS = [(FULL, CTEMP),
               (FULL, EZIEH),
               (CTEMP, EZIEH),
               (CTEMP, NO_CORRECTION),
               (EZIEH, NO_CORRECTION)]

# Categorical palette colors (checked for contrast on a light background).
# No correction and the full model keep the colors they have in step3.6's
# charts (blue, aqua); the single-input variants take the next free slots.
# Several are below 3:1 contrast, so every bar also carries a text label + value.
MODEL_COLORS = {NO_CORRECTION: "#2a78d6", FULL: "#1baf7a", CTEMP: "#eda100", EZIEH: "#e87ba4"}

# --by-residual-range: split the test days by daily residual range (max-min of
# 1-min medians, regression/daily_summary.csv) at the step3.1 filter threshold,
# and compare full vs EZIEH-only within each subset.
SUMMARY_CSV                = Path("regression") / "daily_summary.csv"
RESIDUAL_RANGE_THRESHOLD_NT = 250.0   # matches step3.1 MAX_RESIDUAL_RANGE
SUBSET_COMPARISONS = [(FULL, EZIEH), (FULL, NO_CORRECTION), (EZIEH, NO_CORRECTION)]


def pred_suffix(mode: str, res: str, features: str) -> str:
    """Same suffix convention as step3.2."""
    return (("" if res == "1sec" else f"_{res}")
            + ("_unfiltered" if mode == "unfiltered" else "")
            + ("" if features == "all" else f"_{features}_only"))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["filtered", "unfiltered"], default="filtered")
    parser.add_argument("--resolutions", nargs="+", choices=list(RESOLUTIONS), default=list(RESOLUTIONS))
    parser.add_argument("--analyze-only", action="store_true",
                        help="skip training; bootstrap the existing step3.2 outputs")
    parser.add_argument("--by-residual-range", action="store_true",
                        help=f"also evaluate full vs EZIEH-only separately on test days with residual "
                             f"range <= / > {RESIDUAL_RANGE_THRESHOLD_NT:.0f} nT")
    args = parser.parse_args()
    pd.set_option("display.width", 140)

    summary = pd.read_csv(SUMMARY_CSV, dtype={"date": str})
    high_days = set(summary.loc[summary["residual_range_nT"] > RESIDUAL_RANGE_THRESHOLD_NT, "date"])

    for res in args.resolutions:
        if not args.analyze_only:
            for features in VARIANTS.values():
                print(f"\n##### Training {args.mode} {res} --features {features} #####", flush=True)
                subprocess.run([sys.executable, str(STEP3_2), "--mode", args.mode,
                                "--resolution", res, "--features", features], check=True)

        pred_files = {NO_CORRECTION: None}
        pred_files.update({name: OUT_DIR / f"xgboost_test_predictions{pred_suffix(args.mode, res, f)}.csv"
                           for name, f in VARIANTS.items()})
        tag = res + ("_unfiltered" if args.mode == "unfiltered" else "")
        run_comparison(pred_files, COMPARISONS, MODEL_COLORS, f"{args.mode}, {res}, input ablation",
                       OUT_DIR / f"ablation_ci_{tag}.csv",
                       OUT_DIR / f"ablation_differences_{tag}.csv",
                       OUT_DIR / f"ablation_ci_{tag}.png")

        if args.by_residual_range:
            test_days = set(pd.read_csv(pred_files[FULL], usecols=["date"])["date"].astype(str))
            subsets = {
                "normal":        (test_days - high_days, f"range <= {RESIDUAL_RANGE_THRESHOLD_NT:.0f} nT"),
                "high_residual": (test_days & high_days, f"range > {RESIDUAL_RANGE_THRESHOLD_NT:.0f} nT"),
            }
            sub_files = {k: pred_files[k] for k in (NO_CORRECTION, FULL, EZIEH)}
            for key, (days, desc) in subsets.items():
                if not days:
                    print(f"\n(no {key} test days for {args.mode} {res}; skipped)")
                    continue
                run_comparison(sub_files, SUBSET_COMPARISONS, MODEL_COLORS,
                               f"{args.mode}, {res}, test days with residual {desc}",
                               OUT_DIR / f"ablation_ci_{tag}_{key}.csv",
                               OUT_DIR / f"ablation_differences_{tag}_{key}.csv",
                               OUT_DIR / f"ablation_ci_{tag}_{key}.png",
                               days=days)


if __name__ == "__main__":
    main()
