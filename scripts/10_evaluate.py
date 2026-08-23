"""
10_evaluate.py
RQ1 evaluation: builds Table 5.2 (models x metrics, pooled over the three
rolling origins and per origin) and runs the significance test (decision g:
paired bootstrap of squared-error differences, best ML vs PPML_B).

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

RES = Path(__file__).resolve().parents[1] / "results"
RNG = np.random.default_rng(42)

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
    return pd.concat([ppml, ml], ignore_index=True)

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
    best_ml = pooled[~pooled.model.str.startswith("PPML")].iloc[0].model
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

    # Significance: paired bootstrap over test observations (B=2000),
    # statistic = RMSE(PPML_B) - RMSE(best ML); positive = ML better.
    key = ["origin","exporter","destination","hs6","year"]
    a = df[df.model==best_ml].set_index(key)[["y_true","y_pred"]]
    b = df[df.model=="PPML_B"].set_index(key)["y_pred"]
    pair = a.join(b.rename("y_pred_ppml")).dropna()
    e_ml, e_pp = (pair.y_pred-pair.y_true)**2, (pair.y_pred_ppml-pair.y_true)**2
    obs = float(np.sqrt(e_pp.mean()) - np.sqrt(e_ml.mean()))
    n = len(pair); idx = np.arange(n)
    boots = np.empty(2000)
    for i in range(2000):
        s = RNG.choice(idx, n, replace=True)
        boots[i] = np.sqrt(e_pp.values[s].mean()) - np.sqrt(e_ml.values[s].mean())
    ci = np.percentile(boots, [2.5, 97.5])
    p_two = float(2*min((boots<=0).mean(), (boots>=0).mean()))
    sig = {"best_ml": best_ml, "delta_rmse_ppmlB_minus_bestml_eur": round(obs,1),
           "pct_improvement_vs_ppmlB": round(100*obs/np.sqrt(e_pp.mean()),2),
           "bootstrap_ci95": [round(ci[0],1), round(ci[1],1)],
           "p_two_sided": max(p_two, 1/2000), "B": 2000, "n_test_obs": n}
    (RES / "significance.json").write_text(json.dumps(sig, indent=2))
    print(json.dumps(sig, indent=2))
    print(pooled[["model","rmse_eur","mae_eur","rmse_log","wape_pct","r2_levels"]].to_string(index=False))

if __name__ == "__main__":
    main()
