# Machine Learning versus Structural Gravity: EU Import Flows of Textile Trimmings

Reproducible pipeline for the MSc dissertation *"Machine Learning versus Structural Gravity: Predicting and Explaining EU Import Flows of Textile Trimmings, with an Application to Brazilian Suppliers (2015-2025)"* (Thomas Luiz Reis, MSc Data Science, AI and Digital Business, Gisma University of Applied Sciences, Module M598, 2026).

## What this is

A benchmark of two modeling paradigms on one panel: a structural gravity model estimated by PPML against a random forest, LightGBM and a multilayer perceptron, all trained on identical inputs, for EU imports of six textile trimmings headings (HS 5804, 5806, 5808, 5810, 9606, 9607) from 39 extra-EU suppliers, 2015-2025, at HS6 level. Three research questions: does ML predict better out of sample (RQ1), do feature importances agree with gravity coefficients (RQ2), and where does Brazil trade above or below its modeled level (RQ3).

## Headline results

- Panel: 277,992 observations (39 exporters x 27 EU member states x 24 HS6 codes x 11 years), 74.0% true zeros. Validated against mirror statistics (Comext CIF vs ComexStat FOB: mean discrepancy +6.1%, correlation 0.985) and the visible 2020 pandemic dip (-20.1%).
- RQ1: random forest with lag features cuts out-of-sample RMSE by 53.5% vs the PPML prediction specification (EUR 155,657 vs 334,620; paired bootstrap p < 0.001; rolling origins 2023/2024/2025). PPML converged cleanly in all fits.
- The ablation that matters: remove the lag features and the forest loses to PPML (321,841 vs 318,125). The ML advantage is persistence, not a better use of gravity covariates.
- RQ2: permutation importance is dominated by persistence (rolling mean 0.269, first lag 0.132); GDP matters in both paradigms; distance flips sign between the PPML coefficient (-0.393) and the SHAP association (positive).
- RQ3: both benchmarks place Brazil last among its ten scale-comparable peers; observed EU trade is essentially one corridor (Romania, heading 5806), while gravity locates unrealized potential in Portugal, Italy, France, Germany and Spain.

All numbers in `results/` (Tables 5.1-5.5 of the dissertation, figures, significance tests, fit reports).

## Reproduce everything

```bash
pip install -r requirements.txt
./run_all.sh
```

`run_all.sh` runs the numbered scripts in order: download (01-04), panel construction and real-world validation (05-06), descriptives (14), PPML and ML models (08-09), evaluation, explainability, RQ3 deviations and robustness (10-13). Seeds are fixed (42); exact hyperparameters in `results/ml_tuning_report.json`.

## Data

Raw data is **not redistributed**; every source is re-downloadable by script:

| Source | Content | Script |
|---|---|---|
| Eurostat Comext (DS-045409) | EU imports by HS6, EUR, CIF | `01_download_comext.py` |
| ComexStat/MDIC | Brazil exports by NCM/HS6, USD, FOB | `02_download_comexstat.py` |
| World Bank WDI | GDP, population, EUR/USD rate | `03_download_wdi.py` |
| CEPII Gravity V202211 | distance, contiguity, language, RTA | `04_download_cepii.py` |

Downloads for the reported results were executed on 2026-08-21; `data/DOWNLOAD_LOG.md` records the exact queries, dates, row counts and the execution route. Trade statistics get revised, so a fresh download can differ marginally from the reported numbers.

## Repository structure

```
scripts/           numbered pipeline (01 download ... 13 robustness, 14 descriptives)
data/              download log + small derived artifacts (HS6 scope, panel summary)
results/           every table and figure reported in Chapter 5, plus fit/tuning reports
requirements.txt   pinned stack (Python 3.11, scikit-learn 1.8, statsmodels 0.14.6, LightGBM 4.7, SHAP 0.51)
run_all.sh         one-command reproduction
```

## Citation

Reis, T.L. (2026) *Machine Learning versus Structural Gravity: Predicting and Explaining EU Import Flows of Textile Trimmings, with an Application to Brazilian Suppliers (2015-2025)*. MSc dissertation, Gisma University of Applied Sciences.
