"""
13_robustness.py
Robustness checks (Table 5.5). To keep compute honest on limited hardware,
each check re-runs the RQ1 winner and PPML_B on origin O3 only (longest
training window, test = 2025); the main result uses all three origins.
Conclusion column: does the winner still beat PPML_B on out-of-sample RMSE
in levels? (Y/N)

Checks:
  1 without 5806          (dominant, heterogeneous heading removed)
  2 HS4 aggregation       (does granularity drive the result?)
  3 window excl. 2020-21  (pandemic years out of training)
  4 hyperparameter perturbation of the winner
  5 feature ablation      (no lags; no policy covariate rta)
  6 EU as one bloc        (destination aggregated, decision (a) robustness)

Output: results/table_5_5_robustness.csv
"""
import gc
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
import importlib.util

spec = importlib.util.spec_from_file_location("feats", Path(__file__).with_name("07_features.py"))
feats = importlib.util.module_from_spec(spec); spec.loader.exec_module(feats)
RES = Path(__file__).resolve().parents[1] / "results"
O3 = feats.ORIGINS[2]

def rmse(y, p): return float(np.sqrt(((p-y)**2).mean()))

def fit_ml(train, test, winner, params, drop_cols=None, ppml_tr=None, ppml_te=None):
    name, fs = winner.split("_", 1)
    with_lags = fs in ("lags", "hyb_lags")
    Xtr = feats.design_matrix(train, with_lags); Xte = feats.design_matrix(test, with_lags)
    if ppml_tr is not None:
        Xtr["ppml_pred_log"] = ppml_tr
        Xte["ppml_pred_log"] = ppml_te
    if drop_cols:
        Xtr = Xtr.drop(columns=[c for c in drop_cols if c in Xtr], errors="ignore")
        Xte = Xte.drop(columns=[c for c in drop_cols if c in Xte], errors="ignore")
    Xte = Xte.reindex(columns=Xtr.columns, fill_value=0.0)
    ytr = train.log1p_value.values
    m = feats.make_model(name, params)
    if name == "MLP":
        sc = StandardScaler().fit(Xtr)
        m.fit(sc.transform(Xtr), ytr)
        pl = m.predict(sc.transform(Xte))
    else:
        m.fit(Xtr, ytr)
        pl = m.predict(Xte)
    return np.clip(np.expm1(pl), 0, None)

def fit_ppml_b(train, test, drop_vars=None):
    vars_ = ["ln_gdp_o","ln_gdp_d","ln_dist","contig","comlang_off","comcol","rta"]
    if drop_vars: vars_ = [v for v in vars_ if v not in drop_vars]
    cats = [c for c in ["exporter","destination","hs6"] if train[c].nunique() > 1]
    Xtr = sm.add_constant(pd.get_dummies(train[vars_+cats], columns=cats,
                                         drop_first=True, dtype=float), has_constant="add")
    Xte = sm.add_constant(pd.get_dummies(test[vars_+cats], columns=cats,
                                         drop_first=True, dtype=float), has_constant="add")
    Xte = Xte.reindex(columns=Xtr.columns, fill_value=0.0)
    res = sm.GLM(train.value_eur.values, Xtr.values, family=sm.families.Poisson()).fit(maxiter=300)
    return res.predict(Xte.values)

def ppml_feature_expanding(train, test, drop_vars=None):
    """Hybrid feature for one check, same construction as in 08: the spec B
    prediction for year t comes from a fit on the training years before t.
    The first two training years have no fit before them and get 0, as in 08."""
    years = sorted(train.year.unique())
    tr_feat = pd.Series(0.0, index=train.index)
    for t in years[2:]:
        tr_feat[train.year == t] = fit_ppml_b(train[train.year < t], train[train.year == t], drop_vars)
    return np.log1p(tr_feat.values), np.log1p(fit_ppml_b(train, test, drop_vars))

def aggregate(df, by_bloc=False, by_hs4=False):
    keys = ["exporter", "year"]
    keys += ["heading"] if by_hs4 else ["hs6"]
    if not by_bloc: keys.insert(1, "destination")
    num_first = ["ln_gdp_o","ln_pop_o","gdp_o_missing","year_idx","rta","ln_dist",
                 "contig","comlang_off","comcol","comrelig","ln_gdp_d","ln_pop_d"]
    agg = {"value_eur": "sum", **{c: "first" for c in num_first}}
    g = df.groupby(keys, as_index=False).agg(agg)
    if by_bloc:
        # destination collapsed: pair covariates -> trade-weighted means over MS
        w = df.assign(w=df.value_eur+1).groupby(["exporter","year"] + (["heading"] if by_hs4 else ["hs6"]))
        for c in ["ln_dist","contig","comlang_off","comcol","comrelig","ln_gdp_d","ln_pop_d"]:
            g[c] = w.apply(lambda x: np.average(x[c], weights=x.w)).values
        g["destination"] = "EU27"
    if by_hs4: g = g.rename(columns={"heading":"hs6"})
    g["heading"] = g.hs6.astype(str).str[:4]
    g["log1p_value"] = np.log1p(g.value_eur)
    gg = g.groupby([c for c in ["exporter","destination","hs6"] if c in g])["log1p_value"]
    g["l1_log"], g["l2_log"] = gg.shift(1), gg.shift(2)
    g["roll3_log"] = gg.transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    g["zero_streak"] = 0.0
    return g

def main():
    sig = json.loads((RES/"significance.json").read_text())
    tuning = json.loads((RES/"ml_tuning_report.json").read_text())
    winner = sig["best_ml"]; params = eval(tuning["best_params"][winner.replace("hyb_", "")])
    df = feats.load_panel()
    tr0, te0 = feats.split(df, O3)

    # resume from a partial table if a previous run was cut short
    checks, done = [], set()
    if (RES/"table_5_5_robustness.csv").exists():
        prev = pd.read_csv(RES/"table_5_5_robustness.csv")
        checks.extend(prev.to_dict("records"))
        done = set(prev.check)
        print("resuming robustness, already have:", sorted(done), flush=True)
    def run(label, train, test, ml_kw=None, pp_kw=None, params_override=None, pp_feature=True):
        if label in done:
            return
        p_pp = fit_ppml_b(train, test, **(pp_kw or {}))
        kw = dict(ml_kw or {})
        if "hyb" in winner and pp_feature:
            # hybrid winner: the gravity feature is recomputed per check on the
            # check's own panel, with the expanding window of 08
            kw["ppml_tr"], kw["ppml_te"] = ppml_feature_expanding(train, test, **(pp_kw or {}))
        p_ml = fit_ml(train, test, winner, params_override or params, **kw)
        r_ml, r_pp = rmse(test.value_eur.values, p_ml), rmse(test.value_eur.values, p_pp)
        checks.append({"check": label, "rmse_winner_eur": round(r_ml,1),
                       "rmse_ppmlB_eur": round(r_pp,1),
                       "winner_still_beats_ppml": "Y" if r_ml < r_pp else "N",
                       "n_test": len(test)})
        pd.DataFrame(checks).to_csv(RES/"table_5_5_robustness.csv", index=False)
        gc.collect()
        print(checks[-1], flush=True)

    run("baseline_O3", tr0, te0)
    run("1_without_5806", tr0[tr0.heading!="5806"], te0[te0.heading!="5806"])
    run("3_excl_2020_21", tr0[~tr0.year.isin([2020,2021])], te0)
    # perturbation: double integer regularizers, shrink float fractions by 30%
    # (kept inside valid ranges); n_estimators unchanged
    pert = dict(params)
    for k, v in list(pert.items()):
        if k == "n_estimators" or not isinstance(v, (int, float)): continue
        pert[k] = int(v*2) if isinstance(v, int) else round(max(0.1, min(1.0, v*0.7)), 2)
    run("4_hyperparam_perturbed", tr0, te0, params_override=pert)
    if winner.endswith("_lags"):
        run("5a_ablation_no_lags", tr0, te0, ml_kw={"drop_cols": feats.LAGS})
    run("5b_ablation_no_rta", tr0, te0, ml_kw={"drop_cols": ["rta"]},
        pp_kw={"drop_vars": ["rta"]})
    if "hyb" in winner:
        run("5c_ablation_no_ppml_feature", tr0, te0, pp_feature=False)
    hs4 = aggregate(df, by_hs4=True)
    tr, te = feats.split(hs4, O3); run("2_hs4_aggregation", tr, te)
    bloc = aggregate(df, by_bloc=True)
    tr, te = feats.split(bloc, O3); run("6_eu_bloc", tr, te)

    pd.DataFrame(checks).to_csv(RES/"table_5_5_robustness.csv", index=False)

if __name__ == "__main__":
    main()
