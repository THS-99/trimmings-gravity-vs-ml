"""
08_ppml.py
PPML structural gravity baseline (Santos Silva & Tenreyro 2006; specification
following Yotov et al. 2016), two specifications per decision (c):

  Spec A ("interpretable", answers RQ2): pooled gravity with explicit GDP and
    pair covariates, year and heading fixed effects, SEs clustered by
    exporter x destination pair. Estimated on each origin's training window;
    coefficients reported from O3 (longest window).

  Spec B ("prediction", enters the RQ1 benchmark): exporter, destination and
    HS6 fixed effects plus the time-varying covariates (ln GDP o/d, RTA) and
    pair covariates. NOTE on test-year fixed effects (the pending choice the
    guide asks to describe): exporter-YEAR fixed effects cannot be extended to
    unseen years, so the prediction spec uses time-invariant country and
    product fixed effects, with time variation carried by observed GDP,
    population and RTA in the test year. Coefficients are frozen at training
    values; test-year covariates are observed. Rows with zero trade are kept
    (that is the point of PPML).

Convergence is reported explicitly (Gopinath et al. 2021's PPML reference
failed to converge; a clean convergence is worth stating).

Outputs:
  results/ppml_coefficients_specA.csv   (Table 5.3 input)
  results/predictions_ppml.csv          (origin, spec, y_true, y_pred per row)
  results/ppml_fit_report.json
"""
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from pathlib import Path
from importlib import import_module

feats = import_module("07_features") if __package__ else None
if feats is None:
    import importlib.util, sys
    spec = importlib.util.spec_from_file_location(
        "feats", Path(__file__).with_name("07_features.py"))
    feats = importlib.util.module_from_spec(spec); spec.loader.exec_module(feats)

RES = Path(__file__).resolve().parents[1] / "results"
RES.mkdir(exist_ok=True)

SPEC_A_VARS = ["ln_gdp_o","ln_gdp_d","ln_dist","contig","comlang_off","comcol","rta"]
SPEC_B_VARS = ["ln_gdp_o","ln_gdp_d","ln_dist","contig","comlang_off","comcol","rta"]

def build_X(df, spec):
    if spec == "A":
        X = pd.get_dummies(df[SPEC_A_VARS + ["year","heading"]],
                           columns=["year","heading"], drop_first=True, dtype=float)
    else:
        X = pd.get_dummies(df[SPEC_B_VARS + ["exporter","destination","hs6"]],
                           columns=["exporter","destination","hs6"],
                           drop_first=True, dtype=float)
    return sm.add_constant(X, has_constant="add")

def fit_predict(train, test, spec):
    Xtr, Xte = build_X(train, spec), build_X(test, spec)
    Xte = Xte.reindex(columns=Xtr.columns, fill_value=0.0)   # align dummies
    groups = (train.exporter + "_" + train.destination) if spec == "A" else None
    model = sm.GLM(train.value_eur.values, Xtr.values, family=sm.families.Poisson())
    res = model.fit(maxiter=300, tol=1e-8)
    if spec == "A":
        res_cl = model.fit(maxiter=300, tol=1e-8, cov_type="cluster",
                           cov_kwds={"groups": groups.values})
    else:
        res_cl = res
    pred = res.predict(Xte.values)
    return res_cl, Xtr.columns.tolist(), pred, res.converged, res.fit_history["iteration"]

def main():
    df = feats.load_panel()
    rows, report, coef_out = [], {}, None
    for origin in feats.ORIGINS:
        train, test = feats.split(df, origin)
        for spec in ["A","B"]:
            res, cols, pred, converged, iters = fit_predict(train, test, spec)
            report[f"{origin['name']}_spec{spec}"] = {
                "converged": bool(converged), "iterations": int(iters),
                "n_train": int(len(train)), "n_params": len(cols)}
            rows.append(pd.DataFrame({
                "origin": origin["name"], "model": f"PPML_{spec}",
                "exporter": test.exporter.values, "destination": test.destination.values,
                "hs6": test.hs6.values, "year": test.year.values,
                "y_true": test.value_eur.values, "y_pred": pred}))
            if spec == "A" and origin["name"] == "O3":
                names = ["const"] + [c for c in cols if c != "const"]
                tab = pd.DataFrame({"variable": cols, "coef": res.params,
                                    "se_cluster_pair": res.bse,
                                    "pvalue": res.pvalues})
                coef_out = tab[~tab.variable.str.startswith(("year_","heading_"))]
    pd.concat(rows).to_csv(RES / "predictions_ppml.csv", index=False)
    coef_out.to_csv(RES / "ppml_coefficients_specA.csv", index=False)
    (RES / "ppml_fit_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(coef_out.to_string(index=False))

if __name__ == "__main__":
    main()
