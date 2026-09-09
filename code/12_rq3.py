"""
12_rq3.py
RQ3: Brazil's under/over-trading. Deviation = observed - predicted, pooled over
the three test years (2023-2025), benchmark model = RQ1 winner, with PPML_B as
the robustness benchmark (agreement between the two = stronger evidence).

The honest caveat (5.4 P3): a deviation mixes unrealized potential, model error
and omitted factors. Mitigation implemented here: Brazil is compared against
suppliers of similar scale (the 10 suppliers nearest to Brazil in cumulative
EU imports), using RELATIVE deviations (share of predicted), so common model
error washes out of the comparison.

Outputs: results/table_5_4_deviations.csv, results/rq3_summary.json,
         figures/fig_5_4_brazil.png
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path

RES = Path(__file__).resolve().parents[1] / "results"
FIG = RES / "figures"; FIG.mkdir(exist_ok=True)

def main():
    sig = json.loads((RES/"significance.json").read_text())
    winner = sig["best_ml"]
    ml = pd.read_csv(RES/"predictions_ml.csv")
    ml["model"] = ml.model + "_" + ml.featureset
    pp = pd.read_csv(RES/"predictions_ppml.csv")
    preds = pd.concat([ml[ml.model==winner], pp[pp.model=="PPML_B"]], ignore_index=True)
    preds["heading"] = preds.hs6.astype(str).str.zfill(6).str[:4]

    # supplier scale (test-period observed totals)
    scale = preds[preds.model==winner].groupby("exporter").y_true.sum()
    br_scale = scale["BR"]
    similar = (scale.drop("BR") - br_scale).abs().sort_values().head(10).index.tolist()

    rows = []
    for model, g in preds.groupby("model"):
        for exp_, ge in g.groupby("exporter"):
            obs, pred = ge.y_true.sum(), ge.y_pred.sum()
            rows.append({"model": model, "exporter": exp_,
                         "group": "BR" if exp_=="BR" else ("similar" if exp_ in similar else "other"),
                         "observed_eur": obs, "predicted_eur": pred,
                         "deviation_eur": obs-pred,
                         "relative_deviation_pct": 100*(obs-pred)/pred if pred>0 else np.nan})
    dev = pd.DataFrame(rows)

    # Brazil by heading (winner)
    br_h = (preds[(preds.model==winner)&(preds.exporter=="BR")]
            .groupby("heading")[["y_true","y_pred"]].sum())
    br_h["deviation_eur"] = br_h.y_true - br_h.y_pred
    br_h["relative_deviation_pct"] = 100*br_h.deviation_eur/br_h.y_pred

    dev.to_csv(RES/"table_5_4_deviations.csv", index=False)
    br_row = dev[(dev.model==winner)&(dev.exporter=="BR")].iloc[0]
    sim = dev[(dev.model==winner)&(dev.group=="similar")]
    br_pp = dev[(dev.model=="PPML_B")&(dev.exporter=="BR")].iloc[0]
    out = {
        "winner": winner,
        "similar_suppliers": similar,
        "brazil": {"observed_meur": round(br_row.observed_eur/1e6,2),
                   "predicted_meur": round(br_row.predicted_eur/1e6,2),
                   "relative_deviation_pct": round(br_row.relative_deviation_pct,1)},
        "brazil_ppmlB_relative_deviation_pct": round(br_pp.relative_deviation_pct,1),
        "similar_median_relative_deviation_pct": round(sim.relative_deviation_pct.median(),1),
        "similar_range_pct": [round(sim.relative_deviation_pct.min(),1),
                              round(sim.relative_deviation_pct.max(),1)],
        "brazil_rank_in_similar_set": int((sim.relative_deviation_pct > br_row.relative_deviation_pct).sum()+1),
        "brazil_by_heading": {h: {"obs_meur": round(r.y_true/1e6,3),
                                  "pred_meur": round(r.y_pred/1e6,3),
                                  "rel_dev_pct": round(r.relative_deviation_pct,1)}
                              for h, r in br_h.iterrows()},
    }
    (RES/"rq3_summary.json").write_text(json.dumps(out, indent=2))

    # Fig 5.4: Brazil observed vs predicted by year (winner + PPML_B)
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    br = preds[preds.exporter=="BR"].groupby(["model","year"])[["y_true","y_pred"]].sum().reset_index()
    fig, ax = plt.subplots(figsize=(7,4.2))
    yrs = sorted(br.year.unique())
    obs = br[br.model==winner].set_index("year").y_true/1e6
    ax.plot(yrs, obs.loc[yrs], "o-", color="#1F3864", label="Observed")
    for mdl, style, col in [(winner,"s--","#C0504D"), ("PPML_B","^--","#4BACC6")]:
        p = br[br.model==mdl].set_index("year").y_pred/1e6
        ax.plot(yrs, p.loc[yrs], style, color=col, label=f"Predicted ({mdl})")
    ax.set_xlabel("Test year"); ax.set_ylabel("Brazil -> EU27 imports (M EUR)")
    ax.set_xticks(yrs); ax.legend(); ax.set_title("Brazil: observed vs predicted, test years")
    fig.tight_layout(); fig.savefig(FIG/"fig_5_4_brazil.png", dpi=150); plt.close(fig)
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
