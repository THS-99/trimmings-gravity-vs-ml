"""
03_download_wdi.py
Downloads GDP (current USD), population, and the Euro-area EUR/USD annual
exchange rate from the World Bank WDI API as CSV bundles, 2015-2025.

Executed 2026-08-21 (via browser session, identical URLs; see data/DOWNLOAD_LOG.md).
Outputs: data/raw/wdi_gdp/, data/raw/wdi_pop/, data/raw/wdi_fx/
"""
import io
import zipfile
import requests
from pathlib import Path

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
SERIES = [
    ("all", "NY.GDP.MKTP.CD", "wdi_gdp"),
    ("all", "SP.POP.TOTL",    "wdi_pop"),
    ("EMU", "PA.NUS.FCRF",    "wdi_fx"),   # EUR per USD, annual average
]

def main():
    for country, ind, folder in SERIES:
        url = (f"https://api.worldbank.org/v2/country/{country}/indicator/{ind}"
               f"?date=2015:2025&downloadformat=csv")
        r = requests.get(url, timeout=120)
        r.raise_for_status()
        zipfile.ZipFile(io.BytesIO(r.content)).extractall(RAW / folder)
        print(ind, "->", folder)

if __name__ == "__main__":
    main()
