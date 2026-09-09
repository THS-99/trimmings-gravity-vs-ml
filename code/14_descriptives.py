"""
14_descriptives.py
Setup and descriptives for section 5.1: Table 5.1 (panel dimensions and
zero-share by split) and the figures the text points to:
  Fig 5.1  EU27 imports by heading, 2015-2025 (M EUR)
  Fig 5.2  supplier shares of EU27 imports of the six headings (cumulative)

Style: single y-axis, labeled axes with units, recessive grid, fixed colors per
heading (never re-assigned), markers as secondary encoding besides color.

Outputs: results/table_5_1_panel.csv, figures/fig_5_1_headings.png,
         figures/fig_5_2_suppliers.png
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import importlib.util

spec = importlib.util.spec_from_file_location("feats", Path(__file__).with_name("07_features.py"))
feats = importlib.util.module_from_spec(spec); spec.loader.exec_module(feats)
RES = Path(__file__).resolve().parents[1] / "results"
FIG = RES / "figures"; FIG.mkdir(exist_ok=True)

HEADING_LABEL = {"5804":"5804 lace/tulle","5806":"5806 narrow fabrics",
                 "5808":"5808 braids/trimmings","5810":"5810 embroidery",
                 "9606":"9606 buttons","9607":"9607 zippers"}
COLORS = {"5804":"#4053A3","5806":"#B34A69","5808":"#2E7E64",
          "5810":"#8A6A2F","9606":"#5B7DB1","9607":"#71589E"}
MARKERS = {"5804":"o","5806":"s","5810":"^","5808":"D","9606":"v","9607":"P"}

def main():
    df = feats.load_panel()

    # Table 5.1
    rows = []
    for label, sub in [("full 2015-2025", df),
                       ("train 2015-2022", df[df.year<=2022]),
                       ("test 2023-2025", df[df.year>=2023])]:
        rows.append({"split": label, "n_obs": len(sub),
                     "nonzero": int((sub.value_eur>0).sum()),
                     "zero_share_pct": round(100*(sub.value_eur==0).mean(),1),
                     "total_meur": round(sub.value_eur.sum()/1e6,1),
                     "share_5806_pct": round(100*sub[sub.heading=="5806"].value_eur.sum()
                                             / sub.value_eur.sum(),1)})
    pd.DataFrame(rows).to_csv(RES/"table_5_1_panel.csv", index=False)

    # Fig 5.1: EU imports by heading over time
    by = df.groupby(["heading","year"]).value_eur.sum().unstack(0)/1e6
    fig, ax = plt.subplots(figsize=(7.5,4.5))
    for h in ["5806","9607","9606","5810","5804","5808"]:
        ax.plot(by.index, by[h], marker=MARKERS[h], ms=4.5, lw=1.8,
                color=COLORS[h], label=HEADING_LABEL[h])
    ax.set_xlabel("Year"); ax.set_ylabel("EU27 imports (million EUR)")
    ax.set_xticks(by.index); ax.tick_params(labelsize=8)
    ax.grid(alpha=0.25, lw=0.5); ax.spines[["top","right"]].set_visible(False)
    ax.legend(fontsize=8, ncol=2, frameon=False)
    ax.set_title("EU27 imports of the six trimmings headings from the 39 panel suppliers",
                 fontsize=10)
    fig.tight_layout(); fig.savefig(FIG/"fig_5_1_headings.png", dpi=150); plt.close(fig)

    # Fig 5.2: supplier shares (cumulative 2015-2025), top 12 + Brazil highlighted
    sup = df.groupby("exporter").value_eur.sum().sort_values(ascending=False)/1e6
    top = sup.head(12)
    if "BR" not in top.index: top = pd.concat([top, sup[["BR"]]])
    fig, ax = plt.subplots(figsize=(7.5,4.2))
    colors = ["#B34A69" if e=="BR" else "#4053A3" for e in top.index]
    ax.bar(top.index, top.values, color=colors, width=0.62)
    for i, (e, v) in enumerate(top.items()):
        ax.text(i, v, f"{v:,.0f}", ha="center", va="bottom", fontsize=7.5, color="#333")
    ax.set_ylabel("Cumulative EU27 imports 2015-2025 (million EUR)")
    ax.set_xlabel("Supplier (Brazil highlighted)")
    ax.grid(axis="y", alpha=0.25, lw=0.5); ax.spines[["top","right"]].set_visible(False)
    ax.set_title("Largest extra-EU suppliers of the six headings; Brazil ranks 16th",
                 fontsize=10)
    fig.tight_layout(); fig.savefig(FIG/"fig_5_2_suppliers.png", dpi=150); plt.close(fig)
    print(pd.DataFrame(rows).to_string(index=False))

if __name__ == "__main__":
    main()
