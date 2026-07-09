"""
Calculate horizontal magnetic field components and write per-day Excel files.

  EZIEH = sqrt(EZIEX^2 + EZIEY^2)   (Bx renamed -> EZIEX, By -> EZIEY, Bz -> EZIEZ)
  FRDH  = sqrt(FRDX^2  + FRDY^2)

Output: excel_output/<YYYYMMDD>.xlsx
  Sheet "EZIE" — all EZIE columns with Bx/By/Bz renamed and EZIEH appended
  Sheet "FRD"  — DATE, TIME, DOY, FRDX, FRDY, FRDZ, FRDF, FRDH
"""

import math
import pandas as pd
from pathlib import Path

EZIE_DIR   = Path("humanReadable_EZIE_data")
FRD_DIR    = Path("uncompressed_FRD_data")
OUTPUT_DIR = Path("excel_output")

IAGA_MISSING = 99999.00   # IAGA-2002 fill value


# ---------------------------------------------------------------------------
# EZIE
# ---------------------------------------------------------------------------

def load_ezie(date_str: str) -> pd.DataFrame | None:
    path = EZIE_DIR / f"{date_str}.csv"
    if not path.exists():
        return None

    df = pd.read_csv(path)

    # Rename magnetic field columns
    df = df.rename(columns={"Bx": "EZIEX", "By": "EZIEY", "Bz": "EZIEZ"})

    # Insert EZIEH right after EZIEZ
    df["EZIEH"] = (df["EZIEX"] ** 2 + df["EZIEY"] ** 2) ** 0.5

    # Reorder so EZIEH sits next to the other B columns
    cols = list(df.columns)
    bz_idx = cols.index("EZIEZ")
    cols.remove("EZIEH")
    cols.insert(bz_idx + 1, "EZIEH")
    df = df[cols]

    return df


# ---------------------------------------------------------------------------
# FRD (IAGA-2002 format)
# ---------------------------------------------------------------------------

def load_frd(date_str: str) -> pd.DataFrame | None:
    path = FRD_DIR / f"FRD_{date_str}_1sec.sec"
    if not path.exists():
        return None

    rows = []
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if "|" in line:          # header / column-label lines all end with |
                continue
            parts = line.split()
            if len(parts) < 7:
                continue
            try:
                frdx = float(parts[3])
                frdy = float(parts[4])
                frdz = float(parts[5])
                frdf = float(parts[6])
            except ValueError:
                continue

            # Replace IAGA missing-value flag with NaN
            frdx = float("nan") if abs(frdx - IAGA_MISSING) < 0.01 else frdx
            frdy = float("nan") if abs(frdy - IAGA_MISSING) < 0.01 else frdy
            frdz = float("nan") if abs(frdz - IAGA_MISSING) < 0.01 else frdz
            frdf = float("nan") if abs(frdf - IAGA_MISSING) < 0.01 else frdf

            rows.append({
                "DATE": parts[0],
                "TIME": parts[1],
                "DOY":  int(parts[2]),
                "FRDX": frdx,
                "FRDY": frdy,
                "FRDZ": frdz,
                "FRDF": frdf,
            })

    if not rows:
        return None

    df = pd.DataFrame(rows)

    # Linear interpolation for short gaps only (≤600 s); multi-hour outages stay NaN
    for col in ("FRDX", "FRDY", "FRDZ", "FRDF"):
        df[col] = df[col].interpolate(method="linear", limit=600)

    # Recompute FRDH from interpolated X and Y
    df["FRDH"] = (df["FRDX"] ** 2 + df["FRDY"] ** 2) ** 0.5

    return df


# ---------------------------------------------------------------------------
# Per-day Excel writer
# ---------------------------------------------------------------------------

def process_day(date_str: str):
    out_path = OUTPUT_DIR / f"{date_str}.xlsx"

    if out_path.exists():
        print(f"  [SKIP] {date_str}  already done")
        return

    ezie_df = load_ezie(date_str)
    frd_df  = load_frd(date_str)

    if ezie_df is None and frd_df is None:
        print(f"  [NONE] {date_str}  no data in either dataset")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        if ezie_df is not None:
            ezie_df.to_excel(writer, sheet_name="EZIE", index=False)
        if frd_df is not None:
            frd_df.to_excel(writer, sheet_name="FRD", index=False)

    parts = []
    if ezie_df is not None:
        parts.append(f"EZIE {len(ezie_df):,} rows")
    if frd_df is not None:
        parts.append(f"FRD {len(frd_df):,} rows")
    size_mb = out_path.stat().st_size / 1_048_576
    print(f"  [OK]  {date_str}  {' | '.join(parts)}  {size_mb:.1f} MB")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ezie_dates = {p.stem for p in EZIE_DIR.glob("*.csv")}
    frd_dates  = {
        p.stem[4:12]          # "FRD_YYYYMMDD_1sec" -> "YYYYMMDD"
        for p in FRD_DIR.glob("FRD_*_1sec.sec")
    }
    all_dates = sorted(ezie_dates | frd_dates)

    print(f"EZIE days : {len(ezie_dates)}")
    print(f"FRD days  : {len(frd_dates)}")
    print(f"Total     : {len(all_dates)} unique days to process")
    print(f"Output    : {OUTPUT_DIR.resolve()}")
    print("-" * 55)

    for date_str in all_dates:
        process_day(date_str)

    print("-" * 55)
    print("Done.")


if __name__ == "__main__":
    main()
