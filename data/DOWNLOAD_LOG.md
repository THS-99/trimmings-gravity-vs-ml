# Download log

All raw data downloaded on **2026-08-21** (Comtrade-style revisions make the date matter; Comext dataset stamp: updated 2026-08-14).

| File | Source | Query / URL | Rows / size |
|---|---|---|---|
| `comext_trimmings_hs6_2015_2025.csv` | Eurostat Comext API, dataset DS-045409 ("EU trade since 1988 by HS2-4-6 and CN8") | one GET per HS6 code: `.../statistics/1.0/data/DS-045409?format=JSON&freq=A&flow=1&indicators=VALUE_IN_EUROS&time=2015..2025&partner=<41 codes>&product=<hs6>`; all 27 MS reporters | 80,005 rows (incl. a duplicated re-fetch of heading 5808, deduplicated in `05_build_panel.py` -> 72,288 nonzero flows) |
| `comexstat_brazil_trimmings_2015_2025.csv` | ComexStat/MDIC API | POST `https://api-comexstat.mdic.gov.br/general` body: flow=export, period 2015-01..2025-12, filter heading in {5804,5806,5808,5810,9606,9607}, details [country, subHeading], metrics [metricFOB, metricKG] | 3,529 rows; totals cross-check: 2024 = US$ 27.80M, 2025 = US$ 28.72M (match the pre-verified facts bank) |
| `API_NY.GDP.MKTP.CD_*.zip` | World Bank WDI | `api.worldbank.org/v2/country/all/indicator/NY.GDP.MKTP.CD?date=2015:2025&downloadformat=csv` | 39 KB |
| `API_SP.POP.TOTL_*.zip` | World Bank WDI | same, SP.POP.TOTL | 27 KB |
| `API_EMU_PA.NUS.FCRF_*.zip` | World Bank WDI | Euro area (EMU) official exchange rate, EUR per USD, annual | 1.4 KB |
| `Gravity_csv_V202211.zip` | CEPII Gravity database V202211 (Conte et al. 2022) | `cepii.fr/DATA_DOWNLOAD/gravity/data/Gravity_csv_V202211.zip` | 207 MB zip / 1.25 GB csv — kept OUT of the repo (redistribution + size); re-downloadable via `04_download_cepii.py` |

## Execution note (honesty)

The analysis environment has no direct network access to these hosts, so the queries above were executed through the author's browser session on 2026-08-21 with byte-identical parameters, and the responses saved to CSV. Scripts `01_download_comext.py` to `04_download_cepii.py` contain the same queries as runnable `requests` code so the pipeline is reproducible in any normal environment. Everything downstream of `data/raw/` runs entirely from the scripts.

## Supplier selection (documented, reproducible)

Extra-EU suppliers ranked by cumulative EU27 imports of the six headings, 2015-2025 (Comext, reporter EU27_2020, partner=all): top 40 kept; pseudo-partner QW ("countries not specified") dropped -> 39 suppliers. Brazil ranks 16th (EUR 50.3M cumulative). Ranking query preserved in `01_download_comext.py`.

## Not collected (documented deviations)

- **WITS tariffs:** access blocked in the work environment and the per-partner preferential tariff structure is largely captured by the RTA dummy for this EU-destination panel; tariffs are listed as future work / limitation instead of a covariate. Recorded in Decision_Log.md.
- **UN Comtrade:** not needed as a third source; the mirror check uses Comext x ComexStat directly, and ComexStat totals were already cross-verified against Comtrade in the facts bank.
