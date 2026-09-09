"""
04_download_cepii.py
Downloads the CEPII Gravity database V202211 (Conte, Cotterlaz & Mayer 2022)
and extracts the main CSV. ~207 MB zip, 1.25 GB CSV; NOT redistributed in the
repository, re-download with this script.

Executed 2026-08-21 (via browser navigation, identical URL; see data/DOWNLOAD_LOG.md).
Output: data/raw/cepii/Gravity_V202211.csv
"""
import io
import zipfile
import requests
from pathlib import Path

URL = "https://www.cepii.fr/DATA_DOWNLOAD/gravity/data/Gravity_csv_V202211.zip"
RAW = Path(__file__).resolve().parents[1] / "data" / "raw"

def main():
    r = requests.get(URL, timeout=1200)
    r.raise_for_status()
    zipfile.ZipFile(io.BytesIO(r.content)).extractall(
        RAW / "cepii", members=["Gravity_V202211.csv", "Countries_V202211.csv"])
    print("extracted to", RAW / "cepii")

if __name__ == "__main__":
    main()
