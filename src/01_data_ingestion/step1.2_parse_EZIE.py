"""
Parse NASA EZIE magnetometer .rawz files (gzip-compressed binary, 160 bytes/record)
and output human-readable CSV — one file per day, combining all 24 hourly files.

Output: humanReadable_EZIE_data/<YYYYMMDD>.csv

Binary record layout (160 bytes, little-endian):
  Offset  Field      Type     Description
       0  tval       float64  Unix timestamp (seconds since 1970-01-01 UTC)
    8-35  (metadata) 28 bytes constant identifier block ("eziemag")
      36  latitude   float32  degrees
      40  longitude  float32  degrees
      44  altitude   float32  metres
      48  tres       uint32
      60  ctemp      float32  °C  (coil temperature)
      64  ccr        uint32
      72  Bx         float32  nT
      76  By         float32  nT
      80  Bz         float32  nT
     120  Ax         float32  m/s²
     124  Ay         float32  m/s²
     128  Az         float32  m/s²
     132  Gx         float32  rad/s
     136  Gy         float32  rad/s
     140  Gz         float32  rad/s
     152  imu_ctemp  float32  °C  (IMU temperature)
"""

import csv
import gzip
import struct
import io
from pathlib import Path
from datetime import datetime, timezone

INPUT_DIR  = Path("uncompressed_EZIE_data")
OUTPUT_DIR = Path("humanReadable_EZIE_data")

REC_SIZE = 160

CSV_HEADER = [
    "timeString", "tval",
    "latitude", "longitude", "altitude",
    "tres", "ctemp", "ccr",
    "Bx", "By", "Bz",
    "Ax", "Ay", "Az",
    "Gx", "Gy", "Gz",
    "imu_ctemp",
]


def parse_record(rec: bytes):
    """Parse one 160-byte binary record. Returns a list of field values."""
    if len(rec) < REC_SIZE:
        return None

    tval       = struct.unpack_from('<d', rec,   0)[0]
    latitude   = struct.unpack_from('<f', rec,  36)[0]
    longitude  = struct.unpack_from('<f', rec,  40)[0]
    altitude   = struct.unpack_from('<f', rec,  44)[0]
    tres       = struct.unpack_from('<I', rec,  48)[0]
    ctemp      = struct.unpack_from('<f', rec,  60)[0]
    ccr        = struct.unpack_from('<I', rec,  64)[0]
    Bx         = struct.unpack_from('<f', rec,  72)[0]
    By         = struct.unpack_from('<f', rec,  76)[0]
    Bz         = struct.unpack_from('<f', rec,  80)[0]
    Ax         = struct.unpack_from('<f', rec, 120)[0]
    Ay         = struct.unpack_from('<f', rec, 124)[0]
    Az         = struct.unpack_from('<f', rec, 128)[0]
    Gx         = struct.unpack_from('<f', rec, 132)[0]
    Gy         = struct.unpack_from('<f', rec, 136)[0]
    Gz         = struct.unpack_from('<f', rec, 140)[0]
    imu_ctemp  = struct.unpack_from('<f', rec, 152)[0]

    dt = datetime.fromtimestamp(tval, tz=timezone.utc)
    timeString = dt.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'

    return [
        timeString, tval,
        latitude, longitude, altitude,
        tres, ctemp, ccr,
        Bx, By, Bz,
        Ax, Ay, Az,
        Gx, Gy, Gz,
        imu_ctemp,
    ]


def parse_rawz(path: Path) -> list:
    """Decompress and parse one .rawz file. Returns list of row lists."""
    with open(path, 'rb') as f:
        raw = f.read()
    with gzip.open(io.BytesIO(raw)) as gz:
        data = gz.read()

    n_records = len(data) // REC_SIZE
    rows = []
    for i in range(n_records):
        rec = data[i * REC_SIZE:(i + 1) * REC_SIZE]
        row = parse_record(rec)
        if row is not None:
            rows.append(row)
    return rows


def process_day(day_dir: Path):
    """Parse all 24 hourly .rawz files for one day and write a single CSV."""
    out_csv = OUTPUT_DIR / (day_dir.name + ".csv")

    if out_csv.exists() and out_csv.stat().st_size > 1000:
        print(f"  [SKIP] {day_dir.name}  already done")
        return

    rawz_files = sorted(day_dir.glob("*.rawz"))
    if not rawz_files:
        print(f"  [SKIP] {day_dir.name}  no .rawz files found")
        return

    all_rows = []
    for rawz in rawz_files:
        try:
            rows = parse_rawz(rawz)
            all_rows.extend(rows)
        except Exception as exc:
            print(f"    [ERR] {rawz.name}: {exc}")

    # Ensure chronological order (sort by tval)
    all_rows.sort(key=lambda r: r[1])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_csv, 'w', newline='', encoding='utf-8') as fh:
        writer = csv.writer(fh)
        writer.writerow(CSV_HEADER)
        writer.writerows(all_rows)

    size_mb = out_csv.stat().st_size / 1_048_576
    print(f"  [OK]  {day_dir.name}  {len(all_rows):>7,} records  {size_mb:.1f} MB")


def main():
    day_dirs = sorted(d for d in INPUT_DIR.iterdir() if d.is_dir())

    print(f"Input  : {INPUT_DIR.resolve()}")
    print(f"Output : {OUTPUT_DIR.resolve()}")
    print(f"Days   : {len(day_dirs)}")
    print("-" * 55)

    for day_dir in day_dirs:
        process_day(day_dir)

    print("-" * 55)
    print("Done.")


if __name__ == "__main__":
    main()
