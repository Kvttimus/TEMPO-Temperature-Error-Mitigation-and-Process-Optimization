#!/usr/bin/env python3
"""
Scatter plot: daily ctemp range vs daily Bh_noise range.
Storm days (residual_range > 250 nT) shown in a distinct colour.

Output: regression/scatter_ctemp_vs_noise.png
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

SUMMARY_CSV = Path("regression/daily_summary.csv")
OUT_PATH    = Path("regression/scatter_ctemp_vs_noise.png")
STORM_THRESHOLD = 250  # nT

df = pd.read_csv(SUMMARY_CSV)

quiet = df[df["residual_range_nT"] <= STORM_THRESHOLD].copy()
storm = df[df["residual_range_nT"] >  STORM_THRESHOLD].copy()

# Pearson r over all days
r_all, p_all = stats.pearsonr(df["ctemp_range_degC"], df["residual_range_nT"])
# Pearson r over quiet days only
r_quiet, p_quiet = stats.pearsonr(quiet["ctemp_range_degC"], quiet["residual_range_nT"])

print(f"All days  (n={len(df)}):   r = {r_all:.3f},  p = {p_all:.2e}")
print(f"Quiet days (n={len(quiet)}): r = {r_quiet:.3f},  p = {p_quiet:.2e}")
print(f"Storm days excluded: {len(storm)}")

fig, ax = plt.subplots(figsize=(8, 6))

ax.scatter(quiet["ctemp_range_degC"], quiet["residual_range_nT"],
           s=28, alpha=0.75, color="#1f77b4", label=f"Quiet days (n={len(quiet)})", zorder=3)
ax.scatter(storm["ctemp_range_degC"], storm["residual_range_nT"],
           s=40, alpha=0.85, color="#d62728", marker="^",
           label=f"Storm days (n={len(storm)}, residual range > {STORM_THRESHOLD} nT)", zorder=4)

# Regression line over all days
x_line = np.linspace(df["ctemp_range_degC"].min(), df["ctemp_range_degC"].max(), 200)
slope, intercept, *_ = stats.linregress(df["ctemp_range_degC"], df["residual_range_nT"])
ax.plot(x_line, slope * x_line + intercept,
        color="black", linewidth=1.2, linestyle="--", label="Linear fit (all days)", zorder=2)

ax.set_xlabel("Daily $ctemp$ range (°C)", fontsize=13)
ax.set_ylabel("Daily $Bh_{noise}$ range (nT)", fontsize=13)
ax.set_title(
    f"Daily $ctemp$ range vs $Bh_{{noise}}$ range  "
    f"(all days: $r = {r_all:.2f}$,  quiet days: $r = {r_quiet:.2f}$)",
    fontsize=12
)
ax.tick_params(labelsize=11)
ax.legend(fontsize=10, loc="upper left")
ax.grid(True, linewidth=0.4, alpha=0.4)

# Annotate r values
ax.text(0.97, 0.97,
        f"All days:   $r = {r_all:.2f}$  ($p = {p_all:.1e}$, $n = {len(df)}$)\n"
        f"Quiet days: $r = {r_quiet:.2f}$  ($p = {p_quiet:.1e}$, $n = {len(quiet)}$)",
        transform=ax.transAxes, ha="right", va="top",
        fontsize=10, family="monospace",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="gray", alpha=0.8))

plt.tight_layout()
fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nSaved -> {OUT_PATH}")
