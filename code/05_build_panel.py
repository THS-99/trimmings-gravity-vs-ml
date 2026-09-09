"""
05_build_panel.py
Builds the estimation panel: exporter x EU member state x HS6 x year, 2015-2025.

Inputs (data/raw/):
  comext_trimmings_hs6_2015_2025.csv   EU imports by CN8->HS6, EUR, CIF (Eurostat Comext DS-045409)
  wdi_gdp/API_NY.GDP.MKTP.CD_*.csv     GDP current USD (World Bank WDI)
  wdi_pop/API_SP.POP.TOTL_*.csv        Population (World Bank WDI)
  cepii/Gravity_V202211.csv            CEPII Gravity database (Conte et al. 2022)

Outputs (data/processed/):
  panel_trimmings_2015_2025.csv        the panel, zeros filled as true zeros
  hs6_scope.csv                        definitive HS6 code list
  panel_summary.json                   dimensions, zero-share (overall / by split / by year)

Decisions implemented here (see 04_Data_and_Methods/Decision_Log.md):
  (a) destination = EU member states (27 reporters)
  (d) annual frequency
  Zeros kept as true zeros: the Comext extraction returns only nonzero flows,
  so the full cross join supplier x destination x HS6 x year defines the universe
  and absent combinations are true zeros (standard for PPML panels).
  Supplier set: top 40 extra-EU suppliers by cumulative EU27 imports 2015-2025
  plus Brazil (already in top 40 at rank 16); pseudo-partner QW ("countries not
  specified") dropped -> 39 suppliers.
"""
import json
import pandas as pd
import numpy as np
from pathlib import Path

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
OUT = Path(__file__).resolve().parents[1] / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)

YEARS = list(range(2015, 2026))
TRAIN_YEARS = list(range(2015, 2023))   # reporting split; rolling origins in 09_ml_models
TEST_YEARS = [2023, 2024, 2025]

ISO2_TO_ISO3 = {
    # suppliers (Comext partner codes)
    "CN":"CHN","CH":"CHE","TR":"TUR","IN":"IND","GB":"GBR","TW":"TWN","JP":"JPN",
    "US":"USA","VN":"VNM","TH":"THA","ID":"IDN","TN":"TUN","MA":"MAR","HK":"HKG",
    "KR":"KOR","BR":"BRA","PK":"PAK","PH":"PHL","MG":"MDG","MM":"MMR","EG":"EGY",
    "XS":"SRB","EC":"ECU","BA":"BIH","LK":"LKA","SM":"SMR","IL":"ISR","CA":"CAN",
    "ZA":"ZAF","MY":"MYS","PE":"PER","NO":"NOR","MX":"MEX","KH":"KHM","MK":"MKD",
    "UA":"UKR","SG":"SGP","AL":"ALB","BY":"BLR",
    # EU member states (Comext reporter codes)
    "AT":"AUT","BE":"BEL","BG":"BGR","CY":"CYP","CZ":"CZE","DE":"DEU","DK":"DNK",
    "EE":"EST","ES":"ESP","FI":"FIN","FR":"FRA","GR":"GRC","HR":"HRV","HU":"HUN",
    "IE":"IRL","IT":"ITA","LT":"LTU","LU":"LUX","LV":"LVA","MT":"MLT","NL":"NLD",
    "PL":"POL","PT":"PRT","RO":"ROU","SE":"SWE","SI":"SVN","SK":"SVK",
}

def load_comext():
    df = pd.read_csv(RAW / "comext_trimmings_hs6_2015_2025.csv")
    df = df[df.partner != "QW"].copy()          # drop "countries not specified"
    df["hs6"] = df["hs6"].astype(str).str.zfill(6)
    # the raw extraction fetched headings 5808 twice (interrupted first batch was
    # re-requested); rows are exact duplicates, keep one of each
    df = df.drop_duplicates()
    assert not df.duplicated(["reporter","partner","hs6","year"]).any()
    return df

def load_wdi(folder, indicator_name):
    f = next((RAW / folder).glob("API_*.csv"))
    df = pd.read_csv(f, skiprows=4)
    year_cols = [c for c in df.columns if c.isdigit() and 2015 <= int(c) <= 2025]
    long = df.melt(id_vars=["Country Code"], value_vars=year_cols,
                   var_name="year", value_name=indicator_name)
    long["year"] = long["year"].astype(int)
    return long.rename(columns={"Country Code": "iso3"})

def load_cepii(suppliers_iso3, dest_iso3):
    usecols = ["year","iso3_o","iso3_d","country_exists_o","country_exists_d",
               "dist","distcap","distw_harmonic","contig",
               "comlang_off","comlang_ethno","comcol","comrelig",
               "gdp_o","pop_o","fta_wto","wto_o","eu_o"]
    chunks = []
    for ch in pd.read_csv(RAW / "cepii" / "Gravity_V202211.csv",
                          usecols=usecols, chunksize=2_000_000):
        ch = ch[(ch.year >= 2015) & ch.iso3_o.isin(suppliers_iso3) & ch.iso3_d.isin(dest_iso3)
                & (ch.country_exists_o == 1) & (ch.country_exists_d == 1)]
        chunks.append(ch)
    cep = pd.concat(chunks, ignore_index=True)
    # one row per pair-year (historical country entities dropped via country_exists)
    assert not cep.duplicated(["iso3_o","iso3_d","year"]).any()
    return cep

def main():
    comext = load_comext()
    suppliers = sorted(comext.partner.unique())
    reporters = sorted(comext.reporter.unique())
    codes = sorted(comext.hs6.unique())
    (OUT / "hs6_scope.csv").write_text(
        "hs6,heading\n" + "\n".join(f"{c},{c[:4]}" for c in codes) + "\n")

    # Full grid: absence in Comext = true zero
    grid = pd.MultiIndex.from_product(
        [suppliers, reporters, codes, YEARS],
        names=["exporter","destination","hs6","year"]).to_frame(index=False)
    panel = grid.merge(comext.rename(columns={"partner":"exporter","reporter":"destination"}),
                       on=["exporter","destination","hs6","year"], how="left")
    panel["value_eur"] = panel["value_eur"].fillna(0.0)
    panel["heading"] = panel["hs6"].str[:4]
    panel["exporter_iso3"] = panel["exporter"].map(ISO2_TO_ISO3)
    panel["destination_iso3"] = panel["destination"].map(ISO2_TO_ISO3)
    assert panel.exporter_iso3.notna().all() and panel.destination_iso3.notna().all()

    sup3 = sorted(panel.exporter_iso3.unique())
    dst3 = sorted(panel.destination_iso3.unique())

    # CEPII pair-level covariates
    cep = load_cepii(sup3, dst3)
    pair_inv = ["dist","distcap","distw_harmonic","contig","comlang_off",
                "comlang_ethno","comcol","comrelig"]
    latest = (cep.sort_values("year").groupby(["iso3_o","iso3_d"])[pair_inv].last()
                 .reset_index())
    panel = panel.merge(latest, left_on=["exporter_iso3","destination_iso3"],
                        right_on=["iso3_o","iso3_d"], how="left").drop(columns=["iso3_o","iso3_d"])

    # RTA dummy (fta_wto), observed to 2021; 2022-2025 carried forward from 2021
    rta = cep[["iso3_o","iso3_d","year","fta_wto"]].dropna()
    rta_last = rta[rta.year == 2021].drop(columns="year").rename(columns={"fta_wto":"fta_2021"})
    panel = panel.merge(rta, left_on=["exporter_iso3","destination_iso3","year"],
                        right_on=["iso3_o","iso3_d","year"], how="left").drop(columns=["iso3_o","iso3_d"])
    panel = panel.merge(rta_last, left_on=["exporter_iso3","destination_iso3"],
                        right_on=["iso3_o","iso3_d"], how="left").drop(columns=["iso3_o","iso3_d"])
    panel["rta_extended"] = (panel.year >= 2022) & panel.fta_wto.isna()
    panel["rta"] = panel["fta_wto"].fillna(panel["fta_2021"])
    panel = panel.drop(columns=["fta_wto","fta_2021"])

    # GDP and population, WDI 2015-2025 (single consistent source across all years)
    gdp = load_wdi("wdi_gdp", "gdp_usd")
    pop = load_wdi("wdi_pop", "pop")
    for side, iso_col in [("exporter","exporter_iso3"), ("destination","destination_iso3")]:
        panel = panel.merge(gdp.rename(columns={"iso3":iso_col,"gdp_usd":f"gdp_{side}"}),
                            on=[iso_col,"year"], how="left")
        panel = panel.merge(pop.rename(columns={"iso3":iso_col,"pop":f"pop_{side}"}),
                            on=[iso_col,"year"], how="left")

    # Taiwan (TWN) is absent from WDI: use CEPII gdp_o / pop_o for 2015-2021,
    # 2022-2025 stay missing and are flagged (see Decision_Log / limitations).
    tw = cep[cep.iso3_o=="TWN"][["year","gdp_o","pop_o"]].dropna().drop_duplicates("year")
    for y, g, p in tw.itertuples(index=False):
        m = (panel.exporter_iso3=="TWN") & (panel.year==y)
        panel.loc[m, "gdp_exporter"] = panel.loc[m, "gdp_exporter"].fillna(g*1e3)   # CEPII gdp: current thousands USD (verified: BRA 2020 = 1.4486e9 -> 1.45T)
        panel.loc[m, "pop_exporter"] = panel.loc[m, "pop_exporter"].fillna(p*1e3)   # CEPII pop: thousands (verified: BRA 2020 = 212559 -> 212.6M)
    panel["gdp_exporter_missing"] = panel["gdp_exporter"].isna()

    panel.to_csv(OUT / "panel_trimmings_2015_2025.csv", index=False)

    # Summary
    def zshare(df): return float((df.value_eur == 0).mean())
    summary = {
        "n_obs": int(len(panel)),
        "n_exporters": len(suppliers), "n_destinations": len(reporters),
        "n_hs6": len(codes), "years": [min(YEARS), max(YEARS)],
        "nonzero_obs": int((panel.value_eur > 0).sum()),
        "zero_share_overall": round(zshare(panel), 4),
        "zero_share_train_2015_2022": round(zshare(panel[panel.year.isin(TRAIN_YEARS)]), 4),
        "zero_share_test_2023_2025": round(zshare(panel[panel.year.isin(TEST_YEARS)]), 4),
        "zero_share_by_year": {int(y): round(zshare(g), 4) for y, g in panel.groupby("year")},
        "total_value_eur_bn": round(panel.value_eur.sum()/1e9, 3),
        "value_share_5806": round(panel[panel.heading=="5806"].value_eur.sum()/panel.value_eur.sum(), 4),
        "brazil_total_meur": round(panel[panel.exporter=="BR"].value_eur.sum()/1e6, 2),
        "brazil_share_5806": round(panel[(panel.exporter=="BR")&(panel.heading=="5806")].value_eur.sum()
                                   / max(panel[panel.exporter=="BR"].value_eur.sum(),1), 4),
        "covariate_missing_rates": {c: round(float(panel[c].isna().mean()), 4)
            for c in ["dist","contig","comlang_off","rta","gdp_exporter","gdp_destination",
                      "pop_exporter","pop_destination"]},
    }
    (OUT / "panel_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
