# Machine learning vs structural gravity on EU imports of textile trimmings

Code and results for my MSc dissertation at Gisma University of Applied Sciences (Data Science, AI and Digital Business, module M598, 2026):

*Machine Learning versus Structural Gravity: Predicting and Explaining EU Import Flows of Textile Trimmings, with an Application to Brazilian Suppliers (2015-2025)*

The idea: take EU imports of textile trimmings (lace, ribbons, braids, embroidery, buttons and zippers, HS headings 5804, 5806, 5808, 5810, 9606, 9607) at HS6 level and check whether machine learning models actually beat a gravity model estimated with PPML when both get exactly the same data. Then use both models to see where Brazil sells less to the EU than you would expect from its fundamentals.

## What I found

The random forest with lag features beats PPML clearly out of sample: RMSE around EUR 156k against 335k for the PPML prediction spec, a 53.5% reduction (paired bootstrap, p < 0.001, tested on 2023, 2024 and 2025 as held-out years). PPML converged in every fit, so it is a fair baseline.

The part I find most interesting came from an ablation. Remove the lag features and the forest loses to PPML (321,841 vs 318,125). So the ML advantage here is persistence, trade this year looks a lot like trade last year, and not some smarter use of GDP or distance. The explainability results say the same thing: permutation importance is dominated by the rolling mean and the first lag, and distance even flips sign between the PPML coefficient (-0.393) and the SHAP association.

For Brazil, both model families tell the same story. It ranks last among its ten scale-comparable peers, and almost all of its observed EU trade is a single corridor (narrow woven fabrics to Romania). The gravity model puts the unrealized potential mostly in Portugal, Italy, France, Germany and Spain.

Every number reported in the thesis (Tables 5.1 to 5.5, figures, significance tests, fit reports) is in `results/`.

## Data

The panel has 277,992 rows: 39 extra-EU exporters x 27 EU destinations x 24 HS6 codes x 11 years, with zeros kept as true zeros (74.0% of the panel). I validated it against mirror statistics (EU-reported CIF vs Brazil-reported FOB: +6.1% mean discrepancy, correlation 0.985) and the 2020 pandemic dip (-20.1%) shows up where it should.

Sources:

- Eurostat Comext (dataset DS-045409) for the import values
- ComexStat/MDIC for Brazil's own export records (mirror check)
- World Bank WDI for GDP and population
- CEPII Gravity database V202211 for distance, contiguity, language and RTA

I don't redistribute the raw files (size and licensing), but scripts 01 to 04 re-download everything with the same queries I used. `data/DOWNLOAD_LOG.md` has the exact queries, dates and row counts. Trade statistics get revised over time, so a fresh download can differ a bit from the reported numbers.

## Running it

```bash
pip install -r requirements.txt
./run_all.sh
```

Scripts are numbered in execution order: 01-04 download the data, 05-06 build and validate the panel, 14 makes the descriptives, 08 estimates the PPML specs, 09 trains the ML models, and 10-13 do evaluation, explainability, the Brazil deviations and robustness. Seed is fixed at 42. Fair warning: the ML tuning is the slow part, it took a few hours on the 2-CPU machine I ran it on (the grids are in `results/ml_tuning_report.json`).

If you just want the ML part without installing anything, there is a lean Colab notebook in `notebooks/ML_trimmings_thesis.ipynb` that loads the panel straight from this repo ([open it in Colab](https://colab.research.google.com/github/THS-99/trimmings-gravity-vs-ml/blob/main/notebooks/ML_trimmings_thesis.ipynb)). It is a simplified version of scripts 07-10, so its numbers differ a little from the thesis tables.

## What's where

```
scripts/           the numbered pipeline
data/              download log + small derived files (HS6 scope, panel summary)
results/           tables, figures, significance tests, fit and tuning reports
requirements.txt   pinned versions (Python 3.11, scikit-learn 1.8, statsmodels 0.14.6, LightGBM 4.7, SHAP 0.51)
run_all.sh         runs the whole thing in order
```

## Citation

Reis, T.L. (2026) *Machine Learning versus Structural Gravity: Predicting and Explaining EU Import Flows of Textile Trimmings, with an Application to Brazilian Suppliers (2015-2025)*. MSc dissertation, Gisma University of Applied Sciences.
