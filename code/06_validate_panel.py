"""
06_validate_panel.py
Real-world validation of the panel (supervisor's core requirement), two checks:

  1. MIRROR: EU-reported imports from Brazil (Comext, CIF, EUR) vs
     Brazil-reported exports to the EU27 (ComexStat, FOB, USD), year by year.
     Comext EUR converted to USD with the WDI Euro-area exchange rate
     (PA.NUS.FCRF, EUR per USD, annual average). CIF > FOB is expected;
     the check is that the two series are close and co-move.

  2. KNOWN EVENT: the 2020 pandemic dip must be visible in total EU imports
     of the six headings.

Outputs: ../results/validation_report.md and validation.json
"""
import json
import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
RAW, OUT, RES = BASE/"data"/"raw", BASE/"data"/"processed", BASE/"results"
RES.mkdir(exist_ok=True)

EU27_PT = {
    "Alemanha":"DE","Áustria":"AT","Bélgica":"BE","Bulgária":"BG","Chipre":"CY",
    "Croácia":"HR","Tchéquia":"CZ","República Tcheca":"CZ","Dinamarca":"DK",
    "Eslováquia":"SK","Eslovênia":"SI","Espanha":"ES","Estônia":"EE","Finlândia":"FI",
    "França":"FR","Grécia":"GR","Hungria":"HU","Irlanda":"IE","Itália":"IT",
    "Letônia":"LV","Lituânia":"LT","Luxemburgo":"LU","Malta":"MT",
    "Países Baixos (Holanda)":"NL","Países Baixos":"NL","Polônia":"PL","Portugal":"PT",
    "Romênia":"RO","Suécia":"SE",
}

def main():
    panel = pd.read_csv(OUT/"panel_trimmings_2015_2025.csv", dtype={"hs6":str})
    bz = pd.read_csv(RAW/"comexstat_brazil_trimmings_2015_2025.csv")
    fx = pd.read_csv(next((RAW/"wdi_fx").glob("API_*.csv")), skiprows=4)
    fx_rate = {int(c): fx[c].iloc[0] for c in fx.columns if c.isdigit() and fx[c].notna().any()}

    # ---- Check 1: mirror Brazil -> EU27
    bz["dest2"] = bz.country_pt.map(EU27_PT)
    eu_names_missed = sorted(set(bz.loc[bz.dest2.isna(), "country_pt"].unique()))
    bz_eu = bz.dropna(subset=["dest2"]).groupby("year").fob_usd.sum()

    cx_br = (panel[panel.exporter=="BR"].groupby("year").value_eur.sum())
    cx_br_usd = cx_br / pd.Series(fx_rate)   # EUR / (EUR per USD) = USD

    mirror = pd.DataFrame({"comext_cif_usd": cx_br_usd, "comexstat_fob_usd": bz_eu}).dropna()
    mirror["ratio_cif_fob"] = mirror.comext_cif_usd / mirror.comexstat_fob_usd
    mirror["discrepancy_pct"] = 100*(mirror.comext_cif_usd - mirror.comexstat_fob_usd)/mirror.comexstat_fob_usd
    corr = mirror.comext_cif_usd.corr(mirror.comexstat_fob_usd)

    # ---- Check 2: 2020 dip
    tot = panel.groupby("year").value_eur.sum()/1e6
    dip_2020_vs_2019 = 100*(tot[2020]-tot[2019])/tot[2019]
    rebound_2021 = 100*(tot[2021]-tot[2020])/tot[2020]

    res = {
        "mirror": {int(y): {"comext_cif_musd": round(r.comext_cif_usd/1e6,2),
                            "comexstat_fob_musd": round(r.comexstat_fob_usd/1e6,2),
                            "discrepancy_pct": round(r.discrepancy_pct,1)}
                   for y, r in mirror.iterrows()},
        "mirror_mean_discrepancy_pct": round(mirror.discrepancy_pct.mean(),1),
        "mirror_correlation": round(corr,3),
        "unmatched_country_names": eu_names_missed[:20],
        "eu_imports_total_meur_by_year": {int(y): round(v,1) for y,v in tot.items()},
        "dip_2020_vs_2019_pct": round(dip_2020_vs_2019,1),
        "rebound_2021_vs_2020_pct": round(rebound_2021,1),
    }
    (RES/"validation.json").write_text(json.dumps(res, indent=2))

    md = ["# Panel validation report (real-world checks)", "",
          "## Check 1 — Mirror: Brazil -> EU27, Comext (CIF, converted to USD) vs ComexStat (FOB USD)", "",
          "| Year | Comext CIF (M USD) | ComexStat FOB (M USD) | Discrepancy |",
          "|---|---|---|---|"]
    for y, r in mirror.iterrows():
        md.append(f"| {y} | {r.comext_cif_usd/1e6:.2f} | {r.comexstat_fob_usd/1e6:.2f} | {r.discrepancy_pct:+.1f}% |")
    md += ["", f"Mean discrepancy: {mirror.discrepancy_pct.mean():+.1f}% | correlation of the two series: {corr:.3f}.",
           "CIF values exceeding FOB is the expected direction (freight and insurance included on the EU side).",
           "", "## Check 2 — Known event: the 2020 pandemic dip", "",
           f"Total EU27 imports of the six headings fell {dip_2020_vs_2019:+.1f}% in 2020 vs 2019 and rebounded {rebound_2021:+.1f}% in 2021.",
           "", "| Year | EU27 imports (M EUR) |", "|---|---|"]
    md += [f"| {int(y)} | {v:,.1f} |" for y, v in tot.items()]
    (RES/"validation_report.md").write_text("\n".join(md) + "\n")
    print(json.dumps(res, indent=2))

if __name__ == "__main__":
    main()
