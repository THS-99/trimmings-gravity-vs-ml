"""
10_evaluate.py
RQ1 evaluation: builds Table 5.2 (models x metrics, pooled over the three
rolling origins and per origin) and runs the significance test (decision g:
paired bootstrap of squared-error differences, best ML vs PPML_B). The
bootstrap resamples exporter-destination pairs, not rows: the rows of one
corridor are not independent of each other (same clustering as spec A's SEs).
A naive persistence forecast (last year's value) is added to the table as a
reference point for the size of the gain over gravity.

Metrics (4.6): RMSE and MAE in levels (EUR); RMSE and MAE on log1p; WAPE
(= sum|err| / sum y, scale-free and zero-safe); R2 in levels for dialogue with
Morland/Sellami. MAPE deliberately NOT used headline (explodes on zeros,
Sellami lesson) - reported only on nonzero flows with that caveat.

Segments: by heading (with/without 5806), by flow size, Brazil vs rest,
by test year (horizon reading, echo Morland).

Outputs: results/table_5_2_metrics.csv, table_5_2_segments.csv,
         significance.json
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
import importlib.util

spec = importlib.util.spec_from_file_location("feats", Path(__file__).with_name("07_features.py"))
feats = importlib.util.module_from_spec(spec); spec.loader.exec_module(feats)
RES = Path(__file__).resolve().parents[1] / "results"
RNG = np.random.default_rng(42)
B = 2000

def metrics(g):
    err = g.y_pred - g.y_true
    out = {
        "rmse_eur": float(np.sqrt((err**2).mean())),
        "mae_eur": float(err.abs().mean()),
        "rmse_log": float(np.sqrt(((np.log1p(g.y_pred)-np.log1p(g.y_true))**2).mean())),
        "mae_log": float((np.log1p(g.y_pred)-np.log1p(g.y_true)).abs().mean()),
        "wape_pct": float(100*err.abs().sum()/g.y_true.sum()) if g.y_true.sum() > 0 else np.nan,
        "r2_levels": (float(1 - (err**2).sum()/((g.y_true-g.y_true.mean())**2).sum())
                      if g.y_true.var() > 0 else np.nan),
        "n": int(len(g)),
    }
    nz = g[g.y_true > 0]
    out["mape_nonzero_pct"] = float(100*((nz.y_pred-nz.y_true)/nz.y_true).abs().mean())
    return out

def load_predictions():
    ppml = pd.read_csv(RES / "predictions_ppml.csv")
    ppml["featureset"] = "-"
    ml = pd.read_csv(RES / "predictions_ml.csv")
    ml["model"] = ml.model + "_" + ml.featureset
    # naive persistence: predict last year's value for the same corridor
    panel = pd.read_csv(feats.PROCESSED / "panel_trimmings_2015_2025.csv")
    prev = panel[["exporter","destination","hs6","year","value_eur"]].copy()
    prev["year"] += 1
    naive = (ppml[ppml.model=="PPML_A"].drop(columns="y_pred")
             .merge(prev.rename(columns={"value_eur":"y_pred"}), on=["exporter","destination","hs6","year"]))
    naive["model"] = "Naive_lag1"
    return pd.concat([ppml, ml, naive], ignore_index=True)

def cluster_bootstrap(pair, col_a, col_b):
    """Paired bootstrap of RMSE(b) - RMSE(a) over exporter-destination clusters.
    Positive = model a better. pair is indexed by (origin, exporter, destination, hs6, year)."""
    e_a, e_b = (pair[col_a]-pair.y_true)**2, (pair[col_b]-pair.y_true)**2
    cl = pair.index.get_level_values("exporter") + "_" + pair.index.get_level_values("destination")
    g = pd.DataFrame({"cl": cl, "a": e_a.values, "b": e_b.values}).groupby("cl")
    s_a, s_b, n_cl = g.a.sum().values, g.b.sum().values, g.size().values
    k = len(s_a)
    boots = np.empty(B)
    for i in range(B):
        s = RNG.choice(k, k, replace=True)
        boots[i] = np.sqrt(s_b[s].sum()/n_cl[s].sum()) - np.sqrt(s_a[s].sum()/n_cl[s].sum())
    obs = float(np.sqrt(e_b.mean()) - np.sqrt(e_a.mean()))
    ci = np.percentile(boots, [2.5, 97.5])
    p_two = float(2*min((boots<=0).mean(), (boots>=0).mean()))
    return obs, ci, max(p_two, 1/B), k

def main():
    df = load_predictions()
    df["heading"] = df.hs6.astype(str).str.zfill(6).str[:4]

    # Table 5.2: pooled over the three origins + per origin
    rows = []
    for (model,), g in df.groupby([df.model]):
        rows.append({"model": model, "origin": "pooled", **metrics(g)})
        for o, go in g.groupby("origin"):
            rows.append({"model": model, "origin": o, **metrics(go)})
    tab = pd.DataFrame(rows).sort_values(["origin","rmse_eur"])
    tab.to_csv(RES / "table_5_2_metrics.csv", index=False)

    pooled = tab[tab.origin=="pooled"].sort_values("rmse_eur")
    best_ml = pooled[~pooled.model.str.startswith(("PPML","Naive"))].iloc[0].model
    print("Winner (pooled RMSE):", pooled.iloc[0].model, "| best ML:", best_ml)

    # Segments for the best ML and PPML_B
    seg_rows = []
    for model in [best_ml, "PPML_B", "PPML_A"]:
        g = df[df.model==model]
        for h, gh in g.groupby("heading"):
            seg_rows.append({"model": model, "segment": f"heading_{h}", **metrics(gh)})
        seg_rows.append({"model": model, "segment": "excl_5806", **metrics(g[g.heading!="5806"])})
        seg_rows.append({"model": model, "segment": "brazil", **metrics(g[g.exporter=="BR"])})
        seg_rows.append({"model": model, "segment": "not_brazil", **metrics(g[g.exporter!="BR"])})
        seg_rows.append({"model": model, "segment": "true_zero_flows", **metrics(g[g.y_true==0])})
        big = g.y_true >= 100_000
        seg_rows.append({"model": model, "segment": "flows_ge_100k", **metrics(g[big])})
        seg_rows.append({"model": model, "segment": "flows_lt_100k_nonzero",
                         **metrics(g[(~big) & (g.y_true>0)])})
        for y, gy in g.groupby("year"):
            seg_rows.append({"model": model, "segment": f"year_{y}", **metrics(gy)})
    pd.DataFrame(seg_rows).to_csv(RES / "table_5_2_segments.csv", index=False)

    # Significance: paired cluster bootstrap (B=2000) over exporter-destination
    # pairs, statistic = RMSE(PPML_B) - RMSE(best ML); positive = ML better.
    key = ["origin","exporter","destination","hs6","year"]
    a = df[df.model==best_ml].set_index(key)[["y_true","y_pred"]]
    b = df[df.model=="PPML_B"].set_index(key)["y_pred"]
    pair = a.join(b.rename("y_pred_ppml")).dropna()
    obs, ci, p_two, k = cluster_bootstrap(pair, "y_pred", "y_pred_ppml")
    rmse_pp = float(np.sqrt(((pair.y_pred_ppml-pair.y_true)**2).mean()))
    sig = {"best_ml": best_ml, "delta_rmse_ppmlB_minus_bestml_eur": round(obs,1),
           "pct_improvement_vs_ppmlB": round(100*obs/rmse_pp,2),
           "bootstrap_ci95": [round(ci[0],1), round(ci[1],1)],
           "p_two_sided": p_two, "B": B, "n_test_obs": len(pair),
           "cluster": "exporter_destination", "n_clusters": int(k)}

    # Same test against the naive persistence forecast, to size the gain
    c = df[df.model=="Naive_lag1"].set_index(key)["y_pred"]
    pair3 = a.join(c.rename("y_pred_naive")).dropna()
    obs3, ci3, p3, _ = cluster_bootstrap(pair3, "y_pred", "y_pred_naive")
    sig["vs_naive"] = {"delta_rmse_naive_minus_bestml_eur": round(obs3,1),
                       "pct_improvement_vs_naive": round(100*obs3/float(np.sqrt(((pair3.y_pred_naive-pair3.y_true)**2).mean())),2),
                       "bootstrap_ci95": [round(ci3[0],1), round(ci3[1],1)], "p_two_sided": p3}

    # Second question (supervisor): does the gravity prediction feature improve
    # the ML itself? Same cluster bootstrap, best hybrid vs best pure ML.
    hyb = pooled[pooled.model.str.contains("hyb")]
    if len(hyb):
        best_hyb = hyb.iloc[0].model
        pure = pooled[~pooled.model.str.startswith(("PPML","Naive")) & ~pooled.model.str.contains("hyb")]
        best_pure = pure.iloc[0].model
        a2 = df[df.model==best_hyb].set_index(key)[["y_true","y_pred"]]
        b2 = df[df.model==best_pure].set_index(key)["y_pred"]
        pair2 = a2.join(b2.rename("y_pred_pure")).dropna()
        obs2, ci2, p2, _ = cluster_bootstrap(pair2, "y_pred", "y_pred_pure")
        sig["hybrid_vs_pure"] = {
            "best_hybrid": best_hyb, "best_pure_ml": best_pure,
            "delta_rmse_pure_minus_hybrid_eur": round(obs2,1),
            "bootstrap_ci95": [round(ci2[0],1), round(ci2[1],1)],
            "p_two_sided": p2}
    (RES / "significance.json").write_text(json.dumps(sig, indent=2))
    print(json.dumps(sig, indent=2))
    print(pooled[["model","rmse_eur","mae_eur","rmse_log","wape_pct","r2_levels"]].to_string(index=False))

if __name__ == "__main__":
    main()
