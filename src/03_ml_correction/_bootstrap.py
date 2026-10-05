"""Shared day-level bootstrap and CI plotting for model comparisons.

Single source of truth for step3.6 (no correction vs linear vs XGBoost) and
step3.7 (feature ablation), so both experiments compute intervals, paired
tests, and plots identically.

Method: day-level (block) bootstrap over the test days.
  Samples within a day are strongly autocorrelated, so the independent units
  are days, not rows. Each bootstrap replicate draws the test days with
  replacement (same number of days), pools every row of the drawn days, and
  recomputes RMSE / MAE / R2. The 2.5th-97.5th percentiles of N_BOOT
  replicates give the 95% CI.

Significance: paired bootstrap on the difference between two models.
  All models are scored on the identical test rows, so each replicate uses the
  SAME drawn days for both models and records metric(A) - metric(B). If the
  95% CI of that difference excludes 0, the models differ significantly
  (two-sided, alpha = 0.05). Overlapping per-model error bars do NOT imply
  "not significant".
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

N_BOOT = 10_000
SEED   = 42
ALPHA  = 0.05

NO_CORRECTION = "No correction"   # model name whose "prediction" is zero

INK       = "#0b0b0b"
INK_MUTED = "#52514e"
GRID      = "#e1e0d9"
BASELINE  = "#c3c2b7"


def load_errors(pred_files: dict[str, Path | None], days: set[str] | None = None
                ) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Per-row error (actual - predicted) for each model.

    pred_files maps model name -> test-prediction CSV (None = no correction).
    All CSVs must hold the identical test rows, or the paired tests are invalid.
    days, if given, restricts every model to the same subset of test days.
    """
    frames = {name: pd.read_csv(path) for name, path in pred_files.items() if path is not None}
    ref_name, ref = next(iter(frames.items()))
    for name, df in frames.items():
        if not df[["date", "time", "residual"]].equals(ref[["date", "time", "residual"]]):
            raise ValueError(f"{name} and {ref_name} test rows differ — rerun both with the same split")
    if days is not None:
        keep = ref["date"].astype(str).isin(days).values
        frames = {name: df[keep].reset_index(drop=True) for name, df in frames.items()}
        ref = frames[ref_name]

    y = ref["residual"].values
    errors = {name: (y if path is None else y - frames[name]["Bh_noise_prediction"].values)
              for name, path in pred_files.items()}
    return ref[["date", "residual"]], errors


def per_day_sums(rows: pd.DataFrame, errors: dict[str, np.ndarray]) -> tuple[np.ndarray, dict]:
    """Per-day sufficient statistics, so every bootstrap replicate is one matrix product."""
    df = rows.copy()
    df["y2"] = df["residual"] ** 2
    for name, e in errors.items():
        df[f"{name}|sse"] = e ** 2
        df[f"{name}|sae"] = np.abs(e)
    g = df.groupby("date")
    base = np.column_stack([g.size(), g["residual"].sum(), g["y2"].sum()])  # n, sum y, sum y^2
    sums = {name: np.column_stack([g[f"{name}|sse"].sum(), g[f"{name}|sae"].sum()])
            for name in errors}
    return base, sums


def metrics_from_totals(base_tot: np.ndarray, err_tot: np.ndarray) -> dict[str, np.ndarray]:
    n, sy, syy = base_tot[..., 0], base_tot[..., 1], base_tot[..., 2]
    sse, sae   = err_tot[..., 0], err_tot[..., 1]
    sst = syy - sy ** 2 / n
    return {"RMSE": np.sqrt(sse / n), "MAE": sae / n, "R2": 1.0 - sse / sst}


def bootstrap(base: np.ndarray, sums: dict) -> tuple[dict, dict]:
    n_days = base.shape[0]
    rng = np.random.default_rng(SEED)
    counts = rng.multinomial(n_days, np.full(n_days, 1.0 / n_days), size=N_BOOT)  # (N_BOOT, n_days)

    base_boot = counts @ base
    point, boot = {}, {}
    for name, s in sums.items():
        point[name] = metrics_from_totals(base.sum(axis=0), s.sum(axis=0))
        boot[name]  = metrics_from_totals(base_boot, counts @ s)
    return point, boot


def summarize(point: dict, boot: dict, models: list[str],
              comparisons: list[tuple[str, str]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    lo_q, hi_q = 100 * ALPHA / 2, 100 * (1 - ALPHA / 2)

    ci_rows = []
    for name in models:
        for metric in ("RMSE", "MAE", "R2"):
            if name == NO_CORRECTION and metric == "R2":
                continue  # R2 of predicting zero is not meaningful
            b = boot[name][metric]
            ci_rows.append(dict(model=name, metric=metric, value=float(point[name][metric]),
                                ci_low=float(np.percentile(b, lo_q)), ci_high=float(np.percentile(b, hi_q))))

    diff_rows = []
    for a, b_name in comparisons:
        for metric in ("RMSE", "MAE"):
            d = boot[a][metric] - boot[b_name][metric]
            # Two-sided bootstrap p-value, floored at the resolution of N_BOOT
            p = max(2 * min((d >= 0).mean(), (d <= 0).mean()), 1 / N_BOOT)
            lo, hi = np.percentile(d, [lo_q, hi_q])
            diff_rows.append(dict(comparison=f"{a} - {b_name}", metric=metric,
                                  difference=float(point[a][metric] - point[b_name][metric]),
                                  ci_low=float(lo), ci_high=float(hi),
                                  p_value=float(min(p, 1.0)), significant=bool(lo > 0 or hi < 0)))
    return pd.DataFrame(ci_rows), pd.DataFrame(diff_rows)


def plot(ci: pd.DataFrame, models: list[str], colors: dict[str, str], title: str, out_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(10 + 1.5 * max(0, len(models) - 3), 5.2))
    fig.suptitle(title, fontsize=11, color=INK)

    x = np.arange(len(models))
    for ax, metric in zip(axes, ("RMSE", "MAE")):
        sub = ci[ci["metric"] == metric].set_index("model").loc[models]
        vals = sub["value"].values
        yerr = np.vstack([vals - sub["ci_low"].values, sub["ci_high"].values - vals])

        ax.bar(x, vals, width=0.6, color=[colors[m] for m in models], zorder=3)
        ax.errorbar(x, vals, yerr=yerr, fmt="none", ecolor=INK, elinewidth=1.2, capsize=5, zorder=4)
        top = sub["ci_high"].max()
        for xi, (v, hi, lo) in enumerate(zip(vals, sub["ci_high"].values, sub["ci_low"].values)):
            ax.text(xi, hi + top * 0.02, f"{v:.1f}\n[{lo:.1f}–{hi:.1f}]",
                    ha="center", va="bottom", fontsize=8.5, color=INK)

        ax.set_xticks(x)
        # Qualifier in parentheses on its own line so long names don't collide
        ax.set_xticklabels([m.replace(" (", "\n(") for m in models], fontsize=9, color=INK)
        ax.set_title(f"{metric} (nT)", fontsize=11, color=INK, pad=4)
        ax.set_ylim(0, top * 1.22)
        ax.yaxis.grid(True, linewidth=0.8, color=GRID, zorder=0)
        ax.set_axisbelow(True)
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
        ax.spines["bottom"].set_color(BASELINE)
        ax.tick_params(axis="y", colors=INK_MUTED, labelsize=9)
        ax.tick_params(axis="x", length=0)

    plt.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved -> {out_path}")


def run_comparison(pred_files: dict[str, Path | None], comparisons: list[tuple[str, str]],
                   colors: dict[str, str], label: str,
                   ci_csv: Path, diff_csv: Path, plot_png: Path,
                   days: set[str] | None = None):
    """Bootstrap one set of models on shared test rows; print, save CSVs, and plot.
    days, if given, restricts the evaluation to that subset of test days."""
    models = list(pred_files)
    rows, errors = load_errors(pred_files, days)
    base, sums = per_day_sums(rows, errors)
    point, boot = bootstrap(base, sums)
    ci, diffs = summarize(point, boot, models, comparisons)

    n_days = base.shape[0]
    print(f"\n=== {label}  ({n_days} test days, {len(rows):,} rows, {N_BOOT:,} bootstrap replicates) ===")
    print(ci.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print()
    print(diffs.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    ci.to_csv(ci_csv, index=False)
    diffs.to_csv(diff_csv, index=False)
    plot(ci, models, colors,
         f"Test-set EZIEH_noise_ref error, {label}  (95% CI, day-level bootstrap, {n_days} days)", plot_png)
