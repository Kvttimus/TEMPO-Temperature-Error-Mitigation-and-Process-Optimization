"""
Supplementary correlation analysis: within-day association between EZIE-Mag
temperature (ctemp) and the reference residual (EZIEH - a*FRDH - b).

Complements step2.3, which correlates daily RANGES across days. Here, for
each day separately, both series are reduced to 5-minute medians and
correlated:
  Pearson r     - linear association
  Spearman rho  - monotonic association (rank-based; robust to non-linearity
                  and outliers)
The per-day coefficients are then summarized across days for:
  filtered   - days in 'training data/' (step3.1: 1-min-median residual
               range <= 250 nT)
  unfiltered - every day in regression/predicted/
  removed    - the days in unfiltered but not filtered (high-residual days)
Filtered days are a subset of unfiltered days, so their per-day values are
identical in both; the groups differ only by the removed days.

Statistics
----------
Per-day p-values are saved but are NOT valid significance tests: 5-minute
samples within a day are strongly autocorrelated, so the effective sample
size is far below the ~288 points and those p-values are too small.
The across-day test treats each day's coefficient as one observation:
Wilcoxon signed-rank test of whether the median coefficient differs from 0.

Inputs : regression/predicted/<YYYYMMDD>.csv (time, residual, ctemp)
         training data/<YYYYMMDD>.csv          (defines the filtered days)
Outputs: regression/daily_ctemp_residual_correlation.csv          per day
         regression/daily_ctemp_residual_correlation_summary.csv  per group
         regression/daily_ctemp_residual_correlation.png          distributions

Run from the repository root.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

PRED_DIR     = Path("regression") / "predicted"
FILTERED_DIR = Path("training data")
OUT_DIR      = Path("regression")
RESAMPLE     = "5min"
MIN_POINTS   = 10     # a day needs at least this many 5-min pairs

METRICS = {"pearson_r": "Pearson r", "spearman_rho": "Spearman ρ"}

# Categorical palette colors 1-2 (checked for contrast on a light background)
GROUP_COLORS = {"filtered": "#2a78d6", "unfiltered": "#eb6834"}
INK       = "#0b0b0b"
INK_MUTED = "#52514e"
GRID      = "#e1e0d9"
BASELINE  = "#c3c2b7"


def day_correlation(csv_path: Path) -> dict | None:
    df = pd.read_csv(csv_path, usecols=["time", "residual", "ctemp"], parse_dates=["time"])
    df = df.set_index("time").resample(RESAMPLE).median().dropna()
    if len(df) < MIN_POINTS or df["ctemp"].nunique() < 2 or df["residual"].nunique() < 2:
        return None
    pr = stats.pearsonr(df["ctemp"], df["residual"])
    sr = stats.spearmanr(df["ctemp"], df["residual"])
    return dict(n_points=len(df),
                pearson_r=float(pr.statistic),    pearson_p=float(pr.pvalue),
                spearman_rho=float(sr.statistic), spearman_p=float(sr.pvalue))


def summarize(values: pd.Series) -> dict:
    v = values.dropna()
    return dict(n_days=len(v),
                mean=v.mean(), median=v.median(),
                q25=v.quantile(0.25), q75=v.quantile(0.75),
                min=v.min(), max=v.max(),
                frac_positive=(v > 0).mean(),
                frac_abs_ge_0_5=(v.abs() >= 0.5).mean(),
                wilcoxon_p_median_ne_0=float(stats.wilcoxon(v).pvalue))


def plot(daily: pd.DataFrame, out_path: Path):
    groups = {"filtered":   daily[daily["filtered"]],
              "unfiltered": daily}
    rng = np.random.default_rng(0)   # jitter only; fixed so the figure is reproducible

    fig, axes = plt.subplots(1, 2, figsize=(10, 5.2), sharey=True)
    fig.suptitle(f"Per-day correlation between ctemp and reference residual ({RESAMPLE} medians)",
                 fontsize=11, color=INK)
    for ax, (col, label) in zip(axes, METRICS.items()):
        for x, (g, df) in enumerate(groups.items()):
            v = df[col].dropna().values
            ax.boxplot(v, positions=[x], widths=0.5, showfliers=False, zorder=2,
                       medianprops=dict(color=INK, linewidth=1.5),
                       boxprops=dict(color=INK_MUTED), whiskerprops=dict(color=INK_MUTED),
                       capprops=dict(color=INK_MUTED))
            ax.scatter(x + rng.uniform(-0.18, 0.18, len(v)), v, s=10, alpha=0.55,
                       color=GROUP_COLORS[g], edgecolors="none", zorder=3)
        ax.axhline(0, color=BASELINE, linewidth=1, zorder=1)
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels([f"{g.capitalize()}\n({df[col].notna().sum()} days)\nmedian {df[col].median():+.2f}"
                            for g, df in groups.items()], fontsize=9, color=INK)
        ax.set_xlim(-0.6, len(groups) - 0.4)
        ax.set_title(label, fontsize=11, color=INK)
        ax.set_ylim(-1.05, 1.05)
        ax.yaxis.grid(True, linewidth=0.8, color=GRID, zorder=0)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.tick_params(axis="y", colors=INK_MUTED, labelsize=9)
        ax.tick_params(axis="x", length=0)
    axes[0].set_ylabel("Per-day correlation coefficient", fontsize=10)

    plt.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved -> {out_path}")


def main():
    filtered_days = {p.stem for p in FILTERED_DIR.glob("*.csv")}
    rows, skipped = [], []
    for f in sorted(PRED_DIR.glob("*.csv")):
        r = day_correlation(f)
        if r is None:
            skipped.append(f.stem)
            continue
        rows.append(dict(date=f.stem, filtered=f.stem in filtered_days, **r))
    daily = pd.DataFrame(rows)
    if skipped:
        print(f"Skipped {len(skipped)} days (too few points or constant series): {skipped}")

    groups = {"filtered":   daily[daily["filtered"]],
              "unfiltered": daily,
              "removed":    daily[~daily["filtered"]]}
    summary = pd.DataFrame([dict(group=g, metric=m, **summarize(df[m]))
                            for g, df in groups.items() for m in METRICS])

    pd.set_option("display.width", 160)
    print(f"\nPer-day ctemp vs residual correlation ({RESAMPLE} medians)")
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    daily.to_csv(OUT_DIR / "daily_ctemp_residual_correlation.csv", index=False)
    summary.to_csv(OUT_DIR / "daily_ctemp_residual_correlation_summary.csv", index=False)
    print(f"\nSaved -> {OUT_DIR / 'daily_ctemp_residual_correlation.csv'}")
    print(f"Saved -> {OUT_DIR / 'daily_ctemp_residual_correlation_summary.csv'}")
    plot(daily, OUT_DIR / "daily_ctemp_residual_correlation.png")


if __name__ == "__main__":
    main()
