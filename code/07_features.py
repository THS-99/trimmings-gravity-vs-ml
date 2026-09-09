"""
07_features.py
Shared feature engineering and split definitions for the PPML baseline and the
ML models. Imported by scripts 08-13; not run directly.

Decisions implemented (Decision_Log.md):
  (b) rolling-origin validation, 3 origins:
        O1: train 2015-2022 -> test 2023
        O2: train 2015-2023 -> test 2024
        O3: train 2015-2024 -> test 2025
      One-step-ahead design: each origin predicts the next calendar year only.
  (e) lags: ML runs with AND without lag features, both reported.
      Lags of the target use actual history only (t-1, t-2 relative to the
      predicted year), which never crosses the origin's time boundary in a
      one-step-ahead design.
  (f) ML target = log1p(value_eur); predictions back-transformed with expm1 and
      ALL models are scored in levels (EUR).
Taiwan GDP 2022-25 is missing (WDI has no TWN): forward-filled within exporter
and flagged with gdp_exporter_missing, so tree models and the MLP receive a
complete matrix plus the missingness signal.
"""
import numpy as np
import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
PROCESSED = BASE / "data" / "processed"

ORIGINS = [
    {"name": "O1", "train_years": list(range(2015, 2023)), "test_year": 2023},
    {"name": "O2", "train_years": list(range(2015, 2024)), "test_year": 2024},
    {"name": "O3", "train_years": list(range(2015, 2025)), "test_year": 2025},
]
SEED = 42

NUMERIC = ["ln_gdp_o","ln_gdp_d","ln_pop_o","ln_pop_d","ln_dist",
           "contig","comlang_off","comcol","comrelig","rta",
           "gdp_o_missing","year_idx"]
LAGS = ["l1_log","l2_log","roll3_log","zero_streak"]
CATS = ["exporter","destination","hs6"]

def load_panel():
    df = pd.read_csv(PROCESSED / "panel_trimmings_2015_2025.csv",
                     dtype={"hs6": str, "heading": str})
    df = df.sort_values(["exporter","destination","hs6","year"]).reset_index(drop=True)

    # Taiwan GDP gap: forward-fill within exporter (2021 value carried), flag kept
    df["gdp_o_missing"] = df["gdp_exporter"].isna().astype(int)
    df["gdp_exporter"] = df.groupby("exporter")["gdp_exporter"].ffill()
    df["pop_exporter"] = df.groupby("exporter")["pop_exporter"].ffill()

    df["ln_gdp_o"] = np.log(df["gdp_exporter"])
    df["ln_gdp_d"] = np.log(df["gdp_destination"])
    df["ln_pop_o"] = np.log(df["pop_exporter"])
    df["ln_pop_d"] = np.log(df["pop_destination"])
    df["ln_dist"]  = np.log(df["dist"])
    df["year_idx"] = df["year"] - 2015
    df["log1p_value"] = np.log1p(df["value_eur"])

    # Lag features within each exporter x destination x hs6 series
    g = df.groupby(["exporter","destination","hs6"])["log1p_value"]
    df["l1_log"] = g.shift(1)
    df["l2_log"] = g.shift(2)
    df["roll3_log"] = g.shift(1).rolling(3, min_periods=1).mean().reset_index(drop=True)
    # consecutive zero years immediately before t (history only, no leakage)
    def streak_before(values):
        out, run = [], 0
        for v in values:
            out.append(run)
            run = run + 1 if v == 0 else 0
        return out
    df["zero_streak"] = (df.groupby(["exporter","destination","hs6"])["value_eur"]
                           .transform(lambda s: pd.Series(streak_before(s.values), index=s.index)))
    return df

def design_matrix(df, with_lags):
    """One-hot design shared by all ML models (identical inputs across models)."""
    cols = NUMERIC + (LAGS if with_lags else [])
    X = pd.get_dummies(df[CATS + cols], columns=CATS, dtype=float)
    return X.fillna(0.0)

def split(df, origin):
    tr = df[df.year.isin(origin["train_years"])]
    te = df[df.year == origin["test_year"]]
    return tr, te

def inner_split(train_df):
    """Temporal inner split for tuning: last training year is the validation year."""
    val_year = max(train_df.year)
    return train_df[train_df.year < val_year], train_df[train_df.year == val_year]

_PPML_CACHE = {}

def ppml_feature(rows, origin_name):
    """log1p of the PPML spec B prediction for these rows (hybrid feature sets).
    Train rows carry the in-sample prediction of that origin's training fit,
    test rows the frozen-coefficient prediction. Written by 08_ppml.py, so 08
    must run before the hybrid variants in 09."""
    if origin_name not in _PPML_CACHE:
        f = pd.read_csv(PROCESSED / "ppml_feature.csv.gz", dtype={"hs6": str})
        for o, g in f.groupby("origin"):
            _PPML_CACHE[o] = g.set_index(["exporter","destination","hs6","year"]).ppml_pred_log
    idx = pd.MultiIndex.from_frame(rows[["exporter","destination","hs6","year"]])
    return _PPML_CACHE[origin_name].reindex(idx).values
