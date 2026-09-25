"""Fit the two PPML gravity specifications on each rolling origin and write the predictions, the spec A coefficients and the hybrid feature."""
import gc
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
    Xte = Xte.reindex(columns=Xtr.columns, fill_value=0.0)
    cols = Xtr.columns.tolist()
    groups = (train.exporter + "_" + train.destination) if spec == "A" else None
    Xtr, Xte = Xtr.to_numpy(), Xte.to_numpy()
    model = sm.GLM(train.value_eur.values, Xtr, family=sm.families.Poisson())
    res = model.fit(maxiter=300, tol=1e-8)
    if spec == "A":
        res_cl = model.fit(maxiter=300, tol=1e-8, cov_type="cluster",
                           cov_kwds={"groups": groups.values})
    else:
        res_cl = res
    pred = res.predict(Xte)
    converged, iters = res.converged, res.fit_history["iteration"]
    res_cl.remove_data()
    return res_cl, cols, pred, converged, iters

def main():
    df = feats.load_panel()
    rows, feature_rows, report, coef_out = [], [], {}, None
    for origin in feats.ORIGINS:
        train, test = feats.split(df, origin)
        for spec in ["A","B"]:
            res, cols, pred, converged, iters = fit_predict(train, test, spec)
            gc.collect()
            report[f"{origin['name']}_spec{spec}"] = {
                "converged": bool(converged), "iterations": int(iters),
                "n_train": int(len(train)), "n_params": len(cols)}
            print(f"{origin['name']} spec{spec} converged={converged} iters={iters}", flush=True)
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

    for t in range(2017, 2026):
        hist, cur = df[df.year < t], df[df.year == t]
        _, _, prd, converged, iters = fit_predict(hist, cur, "B")
        gc.collect()
        report[f"feature_{t}_specB"] = {"converged": bool(converged), "iterations": int(iters),
                                        "n_train": int(len(hist))}
        print(f"feature {t} specB converged={converged} iters={iters}", flush=True)
        feature_rows.append(pd.DataFrame({
            "exporter": cur.exporter.values, "destination": cur.destination.values,
            "hs6": cur.hs6.values, "year": cur.year.values, "ppml_pred_log": np.log1p(prd)}))
    pd.concat(feature_rows).to_csv(feats.PROCESSED / "ppml_feature.csv.gz", index=False)
    coef_out.to_csv(RES / "ppml_coefficients_specA.csv", index=False)
    (RES / "ppml_fit_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(coef_out.to_string(index=False))

if __name__ == "__main__":
    main()
