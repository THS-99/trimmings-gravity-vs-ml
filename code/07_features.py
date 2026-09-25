"""Shared panel loading, feature engineering, rolling origins and model constructors for the modelling scripts."""
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

    g = df.groupby(["exporter","destination","hs6"])["log1p_value"]
    df["l1_log"] = g.shift(1)
    df["l2_log"] = g.shift(2)
    df["roll3_log"] = g.transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
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
    cols = NUMERIC + (LAGS if with_lags else [])
    X = pd.get_dummies(df[CATS + cols], columns=CATS, dtype=float)
    return X.fillna(0.0)

def split(df, origin):
    tr = df[df.year.isin(origin["train_years"])]
    te = df[df.year == origin["test_year"]]
    return tr, te

def inner_split(train_df):
    val_year = max(train_df.year)
    return train_df[train_df.year < val_year], train_df[train_df.year == val_year]

_PPML_CACHE = {}

def ppml_feature(rows):
    if "f" not in _PPML_CACHE:
        f = pd.read_csv(PROCESSED / "ppml_feature.csv.gz", dtype={"hs6": str})
        _PPML_CACHE["f"] = f.set_index(["exporter","destination","hs6","year"]).ppml_pred_log
    idx = pd.MultiIndex.from_frame(rows[["exporter","destination","hs6","year"]])
    return _PPML_CACHE["f"].reindex(idx).fillna(0.0).values

def make_model(name, params):
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.neural_network import MLPRegressor
    if name == "RF":
        return RandomForestRegressor(random_state=SEED, n_jobs=2, **params)
    if name == "LGBM":
        import lightgbm as lgb
        return lgb.LGBMRegressor(random_state=SEED, n_jobs=2, verbosity=-1, **params)
    p = dict(params); epochs = p.pop("epochs")
    return MLPRegressor(random_state=SEED, max_iter=epochs, early_stopping=False,
                        n_iter_no_change=epochs, **p)
