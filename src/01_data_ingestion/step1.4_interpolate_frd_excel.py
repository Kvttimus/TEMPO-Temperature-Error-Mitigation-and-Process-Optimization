"""
Back-fill interpolation into the FRD sheet of existing Excel files.
Only touches the 24 days that have missing FRD data.
All other sheets (EZIE, Predicted) are preserved unchanged.
Gaps <= 600 s are filled; longer outages stay NaN.
"""
from __future__ import annotations

import pandas as pd
from pathlib import Path
from openpyxl import load_workbook

EXCEL_DIR    = Path("excel_output")
IAGA_MISSING = 99999.0

# Only the days with any missing FRD data (found by pre-scan)
AFFECTED = {
    "20241019", "20241024", "20241118", "20241120", "20241121",
    "20241130", "20250108", "20250112", "20250225", "20250302",
    "20250304", "20250305", "20250313", "20250314", "20250325",
    "20250407", "20250411", "20250415", "20250416", "20250429",
    "20250511", "20250516", "20250521", "20250527",
}


def process(date_str: str):
    excel_path = EXCEL_DIR / f"{date_str}.xlsx"
    if not excel_path.exists():
        print(f"  [SKIP] {date_str} — no Excel file")
        return

    wb = load_workbook(excel_path)
    if "FRD" not in wb.sheetnames:
        print(f"  [SKIP] {date_str} — no FRD sheet")
        return

    # Read FRD sheet into DataFrame
    ws   = wb["FRD"]
    rows = list(ws.values)
    cols = rows[0]
    df   = pd.DataFrame(rows[1:], columns=cols)

    # Build datetime index for time-aware interpolation
    df["_dt"] = pd.to_datetime(
        df["DATE"].astype(str) + " " + df["TIME"].astype(str), errors="coerce"
    )
    df = df.set_index("_dt")

    before = df[["FRDX", "FRDY", "FRDZ", "FRDF", "FRDH"]].isna().sum().sum()

    for col in ("FRDX", "FRDY", "FRDZ", "FRDF"):
        df[col] = pd.to_numeric(df[col], errors="coerce").interpolate(method="time", limit=600)

    # Recompute FRDH from interpolated X and Y
    df["FRDH"] = (df["FRDX"] ** 2 + df["FRDY"] ** 2) ** 0.5

    after = df[["FRDX", "FRDY", "FRDZ", "FRDF", "FRDH"]].isna().sum().sum()

    df = df.reset_index(drop=True)

    # Rebuild FRD sheet in place (preserves sheet order and other sheets)
    frd_idx = wb.sheetnames.index("FRD")
    del wb["FRD"]
    ws_new = wb.create_sheet("FRD", frd_idx)
    ws_new.append(list(cols))
    for row in df[list(cols)].itertuples(index=False):
        ws_new.append(list(row))

    wb.save(excel_path)
    print(f"  [OK]  {date_str} — filled {int(before - after)} NaN cells  "
          f"({int(after)} remaining = long outages)")


def main():
    dates = sorted(AFFECTED)
    print(f"Updating FRD interpolation in {len(dates)} Excel files...")
    for date_str in dates:
        process(date_str)
    print("Done.")


if __name__ == "__main__":
    main()
