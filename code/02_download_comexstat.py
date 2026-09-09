"""
02_download_comexstat.py
Downloads Brazil's exports of the six trimmings headings from the ComexStat/MDIC
API, by destination country and HS6 subheading, FOB USD and KG, 2015-2025.

Executed 2026-08-21 (via browser session with identical body; see
data/DOWNLOAD_LOG.md). Output: data/raw/comexstat_brazil_trimmings_2015_2025.csv
Rate limit: ~1 request / 10 s.
"""
import requests
import pandas as pd
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "data" / "raw"

BODY = {
    "flow": "export", "monthDetail": False,
    "period": {"from": "2015-01", "to": "2025-12"},
    "filters": [{"filter": "heading", "values": [5804, 5806, 5808, 5810, 9606, 9607]}],
    "details": ["country", "subHeading"],
    "metrics": ["metricFOB", "metricKG"],
}

def main():
    r = requests.post("https://api-comexstat.mdic.gov.br/general", json=BODY, timeout=300)
    r.raise_for_status()
    rows = r.json()["data"]["list"]
    df = pd.DataFrame([{"year": x["year"], "country_pt": x["country"],
                        "hs6": x["subHeadingCode"], "fob_usd": x["metricFOB"],
                        "kg": x["metricKG"]} for x in rows])
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "comexstat_brazil_trimmings_2015_2025.csv", index=False)
    # cross-check against the verified facts bank (ABIT/Comtrade): 27.80 / 28.72 M USD
    for y in (2024, 2025):
        print(y, round(df[df.year.astype(int)==y].fob_usd.astype(float).sum()/1e6, 2), "M USD")

if __name__ == "__main__":
    main()
