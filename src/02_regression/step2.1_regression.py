"""
Fast version of step3.1 — saves per-day predictions as CSVs, skips Excel I/O.
Adds the Predicted sheet to Excel files as a separate final pass (step3.1c).

Outputs
-------
regression/predicted/<YYYYMMDD>.csv   — time, FRDH, EZIEH, EZIE_Bh_Predicted, residual
regression/daily_coeffs.csv           — per-day a, b, R2, RMSE, degenerate flag
regression/daily_coefficients.png     — coefficient drift over mission
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

EZIE_DIR     = Path("humanReadable_EZIE_data")
FRD_DIR      = Path("uncompressed_FRD_data")
OUT_DIR      = Path("regression")
PRED_DIR     = OUT_DIR / "predicted"
IAGA_MISSING = 99999.0
RESAMPLE     = "1s"
TRAIN_HOURS  = 12
MIN_FRDH_RANGE = 5.0  # nT — below this the training data is too flat to fit reliably


def load_ezie_1min(date_str: str) -> pd.DataFrame | None:
    path = EZIE_DIR / f"{date_str}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, parse_dates=["timeString"])
    df = df.rename(columns={"timeString": "time", "Bx": "EZIEX", "By": "EZIEY"})
    df["EZIEH"] = (df["EZIEX"] ** 2 + df["EZIEY"] ** 2) ** 0.5
    keep = ["time", "EZIEH"] + (["ctemp"] if "ctemp" in df.columns else [])
    df = df[keep].dropna(subset=["time"]).sort_values("time").set_index("time")
    df.index = df.index.tz_convert("UTC") if df.index.tz is not None else df.index.tz_localize("UTC")
    df = df.resample(RESAMPLE).median()
    df.index = df.index.tz_localize(None)
    return df.reset_index()


def load_frd_1min(date_str: str) -> pd.DataFrame | None:
    path = FRD_DIR / f"FRD_{date_str}_1sec.sec"
    if not path.exists():
        return None
    rows = []
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if "|" in line:
                continue
            p = line.split()
            if len(p) < 7:
                continue
            try:
                frdx, frdy = float(p[3]), float(p[4])
            except ValueError:
                continue
            if abs(frdx - IAGA_MISSING) < 0.01 or abs(frdy - IAGA_MISSING) < 0.01:
                continue
            rows.append({"time": pd.Timestamp(f"{p[0]} {p[1]}"), "FRDH": (frdx**2 + frdy**2)**0.5})
    if not rows:
        return None
    df = pd.DataFrame(rows).sort_values("time").set_index("time")
    df = df.resample(RESAMPLE).median()
    return df.reset_index()


def main():
    OUT_DIR.mkdir(exist_ok=True)
    PRED_DIR.mkdir(exist_ok=True)

    ezie_dates   = {p.stem for p in EZIE_DIR.glob("*.csv")}
    frd_dates    = {p.stem[4:12] for p in FRD_DIR.glob("FRD_*_1sec.sec")}
    common_dates = sorted(ezie_dates & frd_dates)

    print(f"Days to process: {len(common_dates)}")

    daily_results = []

    for i, date_str in enumerate(common_dates, 1):
        # Skip if already done
        out_csv = PRED_DIR / f"{date_str}.csv"
        if out_csv.exists():
            print(f"  [{i:>3}/{len(common_dates)}] {date_str}: already done, skipping")
            # Still load coeffs from existing CSV for summary
            continue

        ezie = load_ezie_1min(date_str)
        frd  = load_frd_1min(date_str)
        if ezie is None or frd is None:
            continue

        merged = pd.merge(frd, ezie, on="time", how="inner").dropna(subset=["FRDH", "EZIEH"])
        if len(merged) < 10:
            continue

        cutoff = pd.Timestamp(f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]} {TRAIN_HOURS:02d}:00:00")
        train  = merged[merged["time"] < cutoff]
        if len(train) < 5:
            continue

        x_tr, y_tr      = train["FRDH"].values, train["EZIEH"].values
        frdh_range_train = float(x_tr.max() - x_tr.min())
        degenerate       = frdh_range_train < MIN_FRDH_RANGE

        a, b = np.polyfit(x_tr, y_tr, 1)

        merged["EZIE_Bh_Predicted"] = a * merged["FRDH"] + b
        merged["residual"]          = merged["EZIEH"] - merged["EZIE_Bh_Predicted"]

        y_pred_tr = a * x_tr + b
        ss_res    = np.sum((y_tr - y_pred_tr) ** 2)
        ss_tot    = np.sum((y_tr - y_tr.mean()) ** 2)
        r2_train  = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
        rmse_full = float(np.sqrt((merged["residual"] ** 2).mean()))

        daily_results.append({
            "date":              date_str,
            "a":                 float(a),
            "b":                 float(b),
            "r2_train_12h":      r2_train,
            "rmse_24h_nT":       rmse_full,
            "frdh_range_train":  frdh_range_train,
            "degenerate":        degenerate,
            "n_train":           int(len(train)),
            "n_total":           int(len(merged)),
        })

        # Save prediction CSV (fast — no Excel I/O)
        save_cols = ["time", "FRDH", "EZIEH", "EZIE_Bh_Predicted", "residual"]
        if "ctemp" in merged.columns:
            save_cols.append("ctemp")
        merged[save_cols].to_csv(out_csv, index=False)

        flag = " [DEGENERATE]" if degenerate else ""
        print(f"  [{i:>3}/{len(common_dates)}] {date_str}  "
              f"a={a:.4f}  b={b:+.1f}  R2={r2_train:.3f}  RMSE_24h={rmse_full:.2f} nT{flag}")

    if not daily_results:
        print("No new results (all days already processed or no data).")
        return

    # Save daily coefficients
    coeffs_df   = pd.DataFrame(daily_results)
    coeffs_path = OUT_DIR / "daily_coeffs.csv"
    coeffs_df.to_csv(coeffs_path, index=False)

    good    = coeffs_df[~coeffs_df["degenerate"]]
    n_degen = int(coeffs_df["degenerate"].sum())
    print(f"\n--- Summary: {len(coeffs_df)} days processed  ({n_degen} flagged degenerate) ---")
    print(f"  [good days only]")
    print(f"  a        : mean={good['a'].mean():.4f},  std={good['a'].std():.4f}")
    print(f"  b        : mean={good['b'].mean():.2f},  std={good['b'].std():.2f} nT")
    print(f"  R2_train : mean={good['r2_train_12h'].mean():.4f}")
    print(f"  RMSE_24h : mean={good['rmse_24h_nT'].mean():.2f} nT")
    print(f"\nCoefficients saved -> {coeffs_path}")

    # Coefficient drift plot
    dates = pd.to_datetime(coeffs_df["date"])
    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
    fig.suptitle("Per-day regression  EZIEH = a*FRDH + b  (trained on first 12h)",
                 fontsize=12, fontweight="bold")

    # Color degenerate days red
    colors = ["#d62728" if d else "#1f77b4" for d in coeffs_df["degenerate"]]

    axes[0].scatter(dates, coeffs_df["a"], s=6, c=colors)
    axes[0].set_ylabel("slope  a")
    axes[0].set_title("Daily slope a  (red = degenerate / quiet day)")
    axes[0].grid(True, linewidth=0.3, alpha=0.5)

    axes[1].scatter(dates, coeffs_df["b"], s=6, c=colors)
    axes[1].set_ylabel("intercept  b (nT)")
    axes[1].set_title("Daily intercept b")
    axes[1].grid(True, linewidth=0.3, alpha=0.5)

    ax3 = axes[2]
    ax3.scatter(dates, coeffs_df["r2_train_12h"], s=6, c=colors, label="R2 train 12h")
    ax3b = ax3.twinx()
    ax3b.scatter(dates, coeffs_df["rmse_24h_nT"], s=6, c=["#ff7f0e"]*len(dates),
                 marker="x", label="RMSE 24h (nT)")
    ax3.set_ylabel("R2")
    ax3b.set_ylabel("RMSE (nT)", color="#ff7f0e")
    ax3.set_title("Training R2 and full-day RMSE  (red = degenerate)")
    ax3.grid(True, linewidth=0.3, alpha=0.5)

    axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    axes[2].xaxis.set_major_locator(mdates.MonthLocator())
    axes[2].tick_params(axis="x", labelsize=8)

    plt.tight_layout()
    fig.savefig(OUT_DIR / "daily_coefficients.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Coefficient plot saved -> {OUT_DIR}/daily_coefficients.png")


if __name__ == "__main__":
    main()
