# Trimmings: Gravity vs ML

This repository contains my MSc dissertation project, implemented in Python as a set of numbered scripts and a Jupyter notebook, for module M598 of the MSc in Data Science, AI and Digital Business at Gisma University of Applied Sciences. The title of the dissertation is *Machine Learning versus Structural Gravity: Predicting and Explaining EU Import Flows of Textile Trimmings, with an Application to Brazilian Suppliers (2015-2025)*.

The dissertation compares two ways of modelling bilateral trade in a narrow product group. One is the structural gravity model, the standard tool in trade economics, which I estimate with Poisson pseudo maximum likelihood. The other is machine learning, trained on exactly the same panel so that neither side sees information the other does not have. The products are textile trimmings, meaning lace, narrow woven fabrics, braids, embroidery, buttons and zippers (HS headings 5804, 5806, 5808, 5810, 9606 and 9607), and the flows are imports of the 27 EU member states from suppliers outside the union between 2015 and 2025.

The hypothesis I tested was that whatever accuracy machine learning gains over the gravity model on a panel like this one comes from the persistence of trade over time, and not from a smarter use of the economic covariates. To separate the two, every machine learning model is trained twice, with and without lagged trade features, and there is also a hybrid version of each one, where the gravity model's own prediction enters the machine learning side as an extra feature. That last part came out of a discussion with my supervisor and turned out to be the most interesting result of the comparison.

The applied part of the work takes both model families and asks where Brazilian suppliers sell less to the EU than their fundamentals would suggest, by destination country and by product heading, using a group of scale-comparable suppliers as the reference.

Main tasks:

- Building the panel of EU imports at HS6 level for eleven years, joined with the economic and geographic covariates, keeping zero flows as true zeros instead of dropping them.
- Validating the data against Brazil's own export records as a mirror check, and against events that should be visible in the series, such as the 2020 pandemic dip.
- Estimating the gravity model in two specifications, one descriptive with fixed effects and one for prediction that only uses variables available before the flow happens.
- Training the machine learning benchmark, three model families with and without lag features, plus the hybrid variants that receive the gravity prediction.
- Comparing the gravity coefficients with the importance measures from the machine learning side, including the cases where the two disagree on the direction of an effect.
- Measuring Brazil's deviation between observed and predicted trade, and breaking it down by destination and by heading.
- Running the robustness checks on the sample, the target transformation, the validation split and the feature set.

Algorithms used: Poisson pseudo maximum likelihood with fixed effects for the gravity side, random forest, gradient boosting with LightGBM and a multilayer perceptron for the machine learning side, blocked forward validation by year for the train and test split (the panel is a time series, so a random split would leak future information into training), paired bootstrap for the significance tests between models, permutation importance and SHAP values for the explainability part.

Technologies used: the whole implementation is in Python 3.11 with the packages:

- pandas 2
- NumPy
- statsmodels 0.14.6
- scikit-learn 1.8.0
- LightGBM 4.7.0
- SHAP 0.51.0
- SciPy
- Matplotlib
- requests

Those are the versions that produced every number reported in the dissertation, with the random seed fixed at 42 in all the scripts.

Data sources: Eurostat Comext (dataset DS-045409) for the EU import values, ComexStat and MDIC for the Brazilian mirror records, the World Bank World Development Indicators for GDP and population, and the CEPII Gravity database V202211 for distance, contiguity, common language and trade agreements. I do not redistribute the raw files here, partly because of their size and partly because of the licence terms, but the first four scripts download them again with the same queries I used and `data/DOWNLOAD_LOG.md` keeps the queries, the download dates and the row counts. Trade statistics get revised over time, so a fresh download can come out slightly different from what the dissertation reports.

The scripts in `code/` are numbered in the order they run and `code/run_all.sh` runs the whole thing from the download to the last table. Every table and figure they produce lands in `results/`, which is where the numbers reported in the dissertation come from. There is also a lighter version of the modelling part in `code/ML_trimmings_thesis.ipynb`, which loads the panel straight from this repository and needs no local setup ([open it in Colab](https://colab.research.google.com/github/THS-99/trimmings-gravity-vs-ml/blob/main/code/ML_trimmings_thesis.ipynb)). The dissertation and the presentation slides go into `report/` after the submission in September 2026.
