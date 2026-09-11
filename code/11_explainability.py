"""
11_explainability.py
RQ2: puts PPML spec A coefficients side by side with ML feature importances.

  - permutation importance of the winning ML model, computed on the TEST set
    (pooled over origins would refit; we use origin O3 for the longest window)
  - SHAP values (TreeExplainer for tree models) on a test sample: give
    DIRECTION, which permutation importance alone does not
  - output feeds Table 5.3 and Fig 5.3

Outputs: results/table_5_3_rq2.csv, results/shap_summary.csv, figures/fig_5_3_shap.png
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler
import importlib.util

spec = importlib.util.spec_from_file_location("feats", Path(__file__).with_name("07_features.py"))
feats = importlib.util.module_from_spec(spec); spec.loader.exec_module(feats)
RES = Path(__file__).resolve().parents[1] / "results"
FIG = RES / "figures"; FIG.mkdir(exist_ok=True)

def refit_winner(df, winner, params, origin):
    """Refit the winning model on the given origin (params frozen from tuning)."""
    name, fs = winner.split("_", 1)
    train, test = feats.split(df, origin)
    Xtr = feats.design_matrix(train, fs in ("lags","hyb_lags"))
    Xte = feats.design_matrix(test, fs in ("lags","hyb_lags"))
    if fs.startswith("hyb_"):
        Xtr["ppml_pred_log"] = feats.ppml_feature(train)
        Xte["ppml_pred_log"] = feats.ppml_feature(test)
    Xte = Xte.reindex(columns=Xtr.columns, fill_value=0.0)
    ytr, yte = train.log1p_value.values, test.log1p_value.values
    m = feats.make_model(name, params)
    if name == "MLP":
        sc = StandardScaler().fit(Xtr)
        m.fit(sc.transform(Xtr), ytr)
        return m, Xtr, Xte, yte, sc
    m.fit(Xtr, ytr)
    return m, Xtr, Xte, yte, None

def main():
    tuning = json.loads((RES / "ml_tuning_report.json").read_text())
    sig = json.loads((RES / "significance.json").read_text())
    winner = sig["best_ml"]
    params = eval(tuning["best_params"][winner.replace("hyb_", "")])
    df = feats.load_panel()
    m, Xtr, Xte, yte, sc = refit_winner(df, winner, params, feats.ORIGINS[2])

    # memory-constrained environment: single worker, 10k-row test sample,
    # float32 design (documented in 4.7; does not change rankings materially)
    rng = np.random.default_rng(feats.SEED)
    sel = rng.choice(len(Xte), min(10_000, len(Xte)), replace=False)
    Xp = Xte.iloc[sel].astype(np.float32); yp = yte[sel]
    Xp_in = sc.transform(Xp) if sc is not None else Xp
    perm = permutation_importance(m, Xp_in, yp, n_repeats=3,
                                  random_state=feats.SEED, n_jobs=1)
    imp = (pd.DataFrame({"feature": Xte.columns, "perm_importance": perm.importances_mean})
             .sort_values("perm_importance", ascending=False))

    # group one-hot blocks so importances compare with gravity variables
    def block(f):
        for p in ["exporter_","destination_","hs6_"]:
            if f.startswith(p): return p[:-1] + " (one-hot block)"
        return f
    imp["block"] = imp.feature.map(block)
    imp_block = (imp.groupby("block").perm_importance.sum()
                   .sort_values(ascending=False).reset_index())

    imp.to_csv(RES / "perm_importance_full.csv", index=False)   # save before SHAP

    # SHAP directions. TreeExplainer on a large random forest is computationally
    # infeasible here (hours); directions are computed on LGBM with the SAME
    # feature set (the runner-up model, near-tied on log metrics), stated
    # explicitly in 4.5/5.3. Importance RANKING still comes from the winner (RF).
    shap_out = None
    if True:
        import shap
        name, fs = winner.split("_", 1)
        if name != "LGBM":
            lgb_params = eval(json.loads((RES/"ml_tuning_report.json").read_text())
                              ["best_params"][f"LGBM_{fs.replace('hyb_', '')}"])
            train, test = feats.split(df, feats.ORIGINS[2])
            Xtr2 = feats.design_matrix(train, fs in ("lags","hyb_lags"))
            if fs.startswith("hyb_"):
                Xtr2["ppml_pred_log"] = feats.ppml_feature(train)
            Xtr2 = Xtr2.reindex(columns=Xte.columns, fill_value=0.0)
            m_shap = feats.make_model("LGBM", lgb_params).fit(Xtr2, train.log1p_value.values)
        else:
            m_shap = m
        samp = Xte.sample(min(2500, len(Xte)), random_state=feats.SEED).astype(np.float32)
        sv = shap.TreeExplainer(m_shap).shap_values(samp)
        sv = pd.DataFrame(sv, columns=Xte.columns)
        shap_out = pd.DataFrame({
            "feature": Xte.columns,
            "mean_abs_shap": sv.abs().mean().values,
            "direction_corr": [sv[c].corr(samp[c].astype(float)) if samp[c].std()>0 else np.nan
                               for c in Xte.columns]})
        shap_out["block"] = shap_out.feature.map(block)
        shap_out = shap_out.sort_values("mean_abs_shap", ascending=False)
        shap_out.to_csv(RES / "shap_summary.csv", index=False)
        try:
            import matplotlib; matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            shap.summary_plot(sv.values, samp, show=False, max_display=15)
            plt.title(f"SHAP summary, LGBM_{winner.split('_',1)[1]} (direction model), origin O3 test (2025)")
            plt.tight_layout(); plt.savefig(FIG / "fig_5_3_shap.png", dpi=150); plt.close()
        except Exception as e:
            print("shap plot skipped:", e)

    # Table 5.3: coefficients vs importances
    coefs = pd.read_csv(RES / "ppml_coefficients_specA.csv")
    map_feature = {"ln_gdp_o":"ln_gdp_o","ln_gdp_d":"ln_gdp_d","ln_dist":"ln_dist",
                   "contig":"contig","comlang_off":"comlang_off","comcol":"comcol","rta":"rta"}
    rows = []
    for var, feat in map_feature.items():
        c = coefs[coefs.variable==var]
        pi = imp[imp.feature==feat].perm_importance.sum()
        sh = (float(shap_out[shap_out.feature==feat].mean_abs_shap.iloc[0]),
              float(shap_out[shap_out.feature==feat].direction_corr.iloc[0])) if shap_out is not None else (np.nan, np.nan)
        rows.append({"variable": var,
                     "ppml_coef": float(c.coef.iloc[0]), "ppml_p": float(c.pvalue.iloc[0]),
                     "ml_perm_importance": float(pi),
                     "ml_mean_abs_shap": sh[0], "ml_shap_direction": sh[1]})
    t53 = pd.DataFrame(rows)
    t53.to_csv(RES / "table_5_3_rq2.csv", index=False)
    imp_block.to_csv(RES / "perm_importance_blocks.csv", index=False)
    print(t53.to_string(index=False))
    print(imp_block.head(12).to_string(index=False))

if __name__ == "__main__":
    main()
