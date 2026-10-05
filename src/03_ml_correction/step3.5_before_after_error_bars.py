"""
Before/After bar charts: raw step2 regression noise vs the XGBoost-corrected
residual, each split by unfiltered vs filtered data, for the 1-minute and
5-minute models.

"Before" = EZIEH_noise_ref, the step2 regression residual (no ML correction)
"After"  = EZIEH_noise_ref - EZIEH_noise_pred (XGBoost-corrected)
           (CSV columns: residual, Bh_noise_prediction)
Each is computed on both the UNFILTERED and FILTERED test-prediction CSVs,
giving 4 bars per metric: Before Unfiltered, Before Filtered,
After Unfiltered, After Filtered.

Outputs
-------
  regression/before_after_errors_1min.png
  regression/before_after_errors_5min.png
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

OUT_DIR = Path("regression")

# Ordinal 4-step blue ramp, light to dark (checked for contrast on a light background)
COLOR_BEFORE_UNFILTERED = "#b7d4f7"
COLOR_BEFORE_FILTERED   = "#86b6ef"
COLOR_AFTER_UNFILTERED  = "#4d8fd6"
COLOR_AFTER_FILTERED    = "#256abf"
INK          = "#0b0b0b"
INK_MUTED    = "#52514e"
GRID         = "#e1e0d9"


def compute_metrics(unfiltered_csv: Path, filtered_csv: Path) -> dict:
    unfiltered_df = pd.read_csv(unfiltered_csv)
    resid_before_unfiltered = unfiltered_df["residual"].values
    err_after_unfiltered = (unfiltered_df["residual"].values
                             - unfiltered_df["Bh_noise_prediction"].values)

    filtered_df = pd.read_csv(filtered_csv)
    resid_before_filtered = filtered_df["residual"].values
    err_after_filtered = (filtered_df["residual"].values
                           - filtered_df["Bh_noise_prediction"].values)

    return dict(
        rmse_before_unfiltered=float(np.sqrt(np.mean(resid_before_unfiltered ** 2))),
        mae_before_unfiltered=float(np.mean(np.abs(resid_before_unfiltered))),
        rmse_before_filtered=float(np.sqrt(np.mean(resid_before_filtered ** 2))),
        mae_before_filtered=float(np.mean(np.abs(resid_before_filtered))),
        rmse_after_unfiltered=float(np.sqrt(np.mean(err_after_unfiltered ** 2))),
        mae_after_unfiltered=float(np.mean(np.abs(err_after_unfiltered))),
        rmse_after_filtered=float(np.sqrt(np.mean(err_after_filtered ** 2))),
        mae_after_filtered=float(np.mean(np.abs(err_after_filtered))),
    )


BAR_LABELS = ["Before\nUnfiltered", "Before\nFiltered", "After\nUnfiltered", "After\nFiltered"]
BAR_COLORS = [COLOR_BEFORE_UNFILTERED, COLOR_BEFORE_FILTERED,
              COLOR_AFTER_UNFILTERED, COLOR_AFTER_FILTERED]


def plot_before_after(metrics: dict, label: str, out_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5.5))
    fig.suptitle(f"EZIEH_noise_ref Error — {label}", fontsize=12, fontweight="bold", color=INK, y=0.97)

    panels = [
        ("RMSE", [metrics["rmse_before_unfiltered"], metrics["rmse_before_filtered"],
                   metrics["rmse_after_unfiltered"], metrics["rmse_after_filtered"]]),
        ("MAE",  [metrics["mae_before_unfiltered"], metrics["mae_before_filtered"],
                   metrics["mae_after_unfiltered"], metrics["mae_after_filtered"]]),
    ]

    for ax, (metric_name, vals) in zip(axes, panels):
        x = [0, 0.4, 0.8, 1.2]
        bars = ax.bar(x, vals, width=0.35, color=BAR_COLORS, zorder=3)
        ax.set_xlim(-0.3, 1.5)

        for rect, val in zip(bars, vals):
            ax.text(rect.get_x() + rect.get_width() / 2, val + max(vals) * 0.02,
                     f"{val:.1f}", ha="center", va="bottom", fontsize=9, color=INK)

        ax.set_xticks(x)
        ax.set_xticklabels([])
        ax.set_title(f"{metric_name} (nT)", fontsize=11, color=INK, pad=4)
        ax.set_ylim(0, max(vals) * 1.10)

        ax.yaxis.grid(True, linewidth=0.8, color=GRID, zorder=0)
        ax.set_axisbelow(True)
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(axis="y", colors=INK_MUTED, labelsize=9)
        ax.tick_params(axis="x", length=0)

    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in BAR_COLORS]
    fig.legend(handles, BAR_LABELS, loc="upper center", ncol=4,
               bbox_to_anchor=(0.5, 0.94), frameon=False, fontsize=9)

    plt.tight_layout(rect=(0, 0, 1, 0.93))
    fig.subplots_adjust(top=0.84)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out_path}")


def main():
    jobs = [
        ("xgboost_test_predictions_1min_unfiltered.csv", "xgboost_test_predictions_1min.csv",
         "1-Minute", "before_after_errors_1min.png"),
        ("xgboost_test_predictions_5min_unfiltered.csv", "xgboost_test_predictions_5min.csv",
         "5-Minute", "before_after_errors_5min.png"),
    ]
    for unfiltered_file, filtered_file, label, out_name in jobs:
        metrics = compute_metrics(OUT_DIR / unfiltered_file, OUT_DIR / filtered_file)
        print(f"{label}: {metrics}")
        plot_before_after(metrics, label, OUT_DIR / out_name)


if __name__ == "__main__":
    main()
