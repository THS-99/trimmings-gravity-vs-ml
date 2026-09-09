"""
01_download_comext.py
Downloads EU imports of the six trimmings headings from the Eurostat Comext API
(dataset DS-045409), at HS6 level, annual, 2015-2025, all 27 MS reporters,
top-40 extra-EU suppliers + Brazil. Also reproduces the supplier ranking.

Executed 2026-08-21 (via browser session with identical parameters; see
data/DOWNLOAD_LOG.md). Output: data/raw/comext_trimmings_hs6_2015_2025.csv
"""
import time
import requests
import pandas as pd
from pathlib import Path

BASE = ("https://ec.europa.eu/eurostat/api/comext/dissemination/statistics/"
        "1.0/data/DS-045409")
YEARS = list(range(2015, 2026))
HS6 = ["580410","580421","580429","580430",
       "580610","580620","580631","580632","580639","580640",
       "580810","580890",
       "581010","581091","581092","581099",
       "960610","960621","960622","960629","960630",
       "960711","960719","960720"]
HEADINGS = ["5804","5806","5808","5810","9606","9607"]
EU_MS = ["AT","BE","BG","HR","CY","CZ","DK","EE","FI","FR","DE","GR","HU","IE",
         "IT","LV","LT","LU","MT","NL","PL","PT","RO","SK","SI","ES","SE"]
OUT = Path(__file__).resolve().parents[1] / "data" / "raw"

def fetch(params):
    p = [("format","JSON"),("freq","A"),("flow","1"),("indicators","VALUE_IN_EUROS")]
    p += [("time",y) for y in YEARS] + params
    r = requests.get(BASE, params=p, timeout=120)
    r.raise_for_status()
    return r.json()

def decode(js, keep_reporters=None):
    """JSON-stat sparse cube -> list of (reporter, partner, year, value)."""
    ids, sizes, dim = js["id"], js["size"], js["dimension"]
    cats = []
    for d in ids:
        idx = dim[d]["category"]["index"]
        arr = [None]*len(idx)
        for k, v in idx.items(): arr[v] = k
        cats.append(arr)
    out = []
    for cell, val in js.get("value", {}).items():
        c = int(cell); coord = [0]*len(ids)
        for i in range(len(ids)-1, -1, -1):
            coord[i] = c % sizes[i]; c //= sizes[i]
        rec = {d: cats[i][coord[i]] for i, d in enumerate(ids)}
        if keep_reporters and rec["reporter"] not in keep_reporters: continue
        out.append((rec["reporter"], rec["partner"], rec["time"], val))
    return out

def rank_suppliers():
    """Cumulative EU27 imports per partner over the six headings, 2015-2025."""
    totals = {}
    for h in HEADINGS:
        js = fetch([("reporter","EU27_2020"),("product",h)])
        for _, partner, _, val in decode(js):
            totals[partner] = totals.get(partner, 0) + val
    eu_and_aggregates = set(EU_MS) | {"GB"} | {k for k in totals if "_" in k or len(k) > 2}
    ext = {k: v for k, v in totals.items() if k not in eu_and_aggregates or k == "GB"}
    ranked = sorted(ext, key=ext.get, reverse=True)[:40]
    return [p for p in ranked if p != "QW"] + (["BR"] if "BR" not in ranked else [])

def main():
    suppliers = rank_suppliers()
    rows = []
    for code in HS6:
        js = fetch([("partner", p) for p in suppliers] + [("product", code)])
        rows += [(rep, par, code, yr, val) for rep, par, yr, val
                 in decode(js, keep_reporters=set(EU_MS))]
        time.sleep(0.5)
    df = pd.DataFrame(rows, columns=["reporter","partner","hs6","year","value_eur"])
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "comext_trimmings_hs6_2015_2025.csv", index=False)
    print(len(df), "rows")

if __name__ == "__main__":
    main()
