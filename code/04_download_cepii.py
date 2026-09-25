"""Download the CEPII Gravity database and extract the main CSV."""
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
