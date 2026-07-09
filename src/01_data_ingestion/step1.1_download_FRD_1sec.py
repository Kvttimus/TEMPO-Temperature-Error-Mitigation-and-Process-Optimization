"""
Download FRD (Fredericksburg) 1-second geomagnetic data from the BGS GIN portal.

Source: https://imag-data.bgs.ac.uk/GIN_V1/GINForms2

Data range: October 1, 2024 – June 2, 2025 (245 days)
"""

import os
import sys
import time
import requests
from datetime import datetime, timedelta

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OBSERVATORY    = "FRD"                      # IAGA code for Fredericksburg
START_DATE     = datetime(2024, 10, 1)
END_DATE       = datetime(2025, 6,  2)
OUTPUT_DIR     = "uncompressed_FRD_data"
BASE_URL       = "https://imag-data.bgs.ac.uk/GIN_V1/GINServices"

# Leave empty unless the server returns HTTP 401 for this dataset
USERNAME = ""
PASSWORD = ""

MAX_RETRIES    = 3
RETRY_DELAY    = 10   # seconds between retry attempts
REQUEST_DELAY  = 1    # seconds between successful requests (be polite)

# A complete 1-second FRD day is ~6.1 MB; anything smaller on disk is either
# a partial/interrupted download or an error page, not a finished file.
MIN_COMPLETE_SIZE_BYTES = 5_000_000
# ---------------------------------------------------------------------------


def build_params(date: datetime) -> dict:
    params = {
        "request":              "GetData",
        "observatoryIagaCode":  OBSERVATORY,
        "samplesPerDay":        "86400",       # 86400 = 1-second resolution
        "dataStartDate":        date.strftime("%Y-%m-%d"),
        "dataDuration":         "1",
        "publicationState":     "adj-or-rep",  # reported/provisional — covers recent data
        "format":               "iaga2002",
    }
    if USERNAME:
        params["username"] = USERNAME
        params["password"] = PASSWORD
    return params


def is_valid_iaga2002(text: str) -> bool:
    """IAGA-2002 files begin with a space-padded header containing 'Format'."""
    return bool(text) and len(text) > 500 and "Format" in text[:600]


def output_path(date: datetime) -> str:
    fname = f"FRD_{date.strftime('%Y%m%d')}_1sec.sec"
    return os.path.join(OUTPUT_DIR, fname)


def download_day(date: datetime, session: requests.Session) -> bool:
    """
    Download 1-second data for one day.
    Returns True on success or if file already exists, False on failure.
    """
    date_str  = date.strftime("%Y-%m-%d")
    fpath     = output_path(date)

    # Resume: skip if file already looks complete.
    if os.path.exists(fpath) and os.path.getsize(fpath) > MIN_COMPLETE_SIZE_BYTES:
        return True   # will be counted as 'skipped' by caller

    params = build_params(date)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = session.get(BASE_URL, params=params, timeout=120)

            if resp.status_code == 200:
                text = resp.text
                if is_valid_iaga2002(text):
                    with open(fpath, "w", encoding="utf-8", newline="\n") as fh:
                        fh.write(text)
                    kb = os.path.getsize(fpath) / 1024
                    print(f"  [OK]     {date_str}  {kb:,.0f} KB")
                    return True
                else:
                    snippet = text[:300].strip()
                    print(f"  [WARN]   {date_str}  unexpected content: {snippet!r}")
                    if attempt < MAX_RETRIES:
                        time.sleep(RETRY_DELAY)

            elif resp.status_code == 401:
                print(
                    f"  [AUTH]   {date_str}  HTTP 401 — set USERNAME/PASSWORD at top of script"
                )
                return False

            elif resp.status_code == 404:
                print(f"  [404]    {date_str}  no data available")
                return False

            else:
                print(
                    f"  [ERR]    {date_str}  HTTP {resp.status_code}"
                    f" (attempt {attempt}/{MAX_RETRIES})"
                )
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_DELAY * attempt)

        except requests.exceptions.Timeout:
            print(f"  [TIMEOUT]{date_str}  (attempt {attempt}/{MAX_RETRIES})")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY * attempt)

        except requests.exceptions.ConnectionError as exc:
            print(f"  [CONN]   {date_str}  {exc} (attempt {attempt}/{MAX_RETRIES})")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY * attempt)

        except requests.exceptions.RequestException as exc:
            print(f"  [ERR]    {date_str}  {exc}")
            return False

    print(f"  [FAIL]   {date_str}  gave up after {MAX_RETRIES} attempts")
    return False


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    total_days = (END_DATE - START_DATE).days + 1

    print("=" * 60)
    print("FRD 1-second geomagnetic data downloader")
    print(f"  Observatory : {OBSERVATORY} (Fredericksburg, Virginia)")
    print(f"  Start       : {START_DATE.date()}")
    print(f"  End         : {END_DATE.date()}")
    print(f"  Total days  : {total_days}")
    print(f"  Output dir  : {os.path.abspath(OUTPUT_DIR)}")
    print("=" * 60)

    downloaded   = 0
    skipped      = 0
    failed_dates = []

    with requests.Session() as session:
        session.headers.update({"User-Agent": "FRD-1sec-downloader/1.0 (research)"})

        current = START_DATE
        while current <= END_DATE:
            fpath = output_path(current)
            already = os.path.exists(fpath) and os.path.getsize(fpath) > MIN_COMPLETE_SIZE_BYTES

            if already:
                skipped += 1
                print(f"  [SKIP]   {current.strftime('%Y-%m-%d')}  already on disk")
            else:
                ok = download_day(current, session)
                if ok:
                    downloaded += 1
                    time.sleep(REQUEST_DELAY)
                else:
                    failed_dates.append(current.strftime("%Y-%m-%d"))

            current += timedelta(days=1)

    print("=" * 60)
    print("Summary")
    print(f"  Downloaded : {downloaded}")
    print(f"  Skipped    : {skipped}  (already on disk)")
    print(f"  Failed     : {len(failed_dates)}")

    if failed_dates:
        log = os.path.join(OUTPUT_DIR, "failed_downloads.txt")
        with open(log, "w") as fh:
            fh.write("\n".join(failed_dates) + "\n")
        print(f"\n  Failed dates written to: {log}")
        print("  Re-run the script to retry them (it resumes automatically).")
        sys.exit(1)
    else:
        print("\nAll days downloaded successfully.")


if __name__ == "__main__":
    main()
