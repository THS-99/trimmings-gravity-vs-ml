"""
09_ml.py
ML benchmark models (decision set mirrors fichas 5-7: one per family):
  RF    random forest           (Breiman-style bagged trees)
  LGBM  gradient boosting       (LightGBM)
  MLP   feed-forward network    (mirrors Morland et al.'s FFNN, honest simple
                                 architecture over fashionable ones)

Protocol:
  - identical one-hot design for all models (07_features.design_matrix),
    two variants: without lags and with lags (decision e; both reported)
  - target log1p(value_eur), scored in levels after expm1 (decision f)
  - hyperparameters tuned ONCE on origin O1's inner temporal split
    (train 2015-2021, validate 2022), then frozen across origins; tuning
    never sees any test year (no leakage across the time boundary)
  - fixed seeds (SEED=42)

Outputs:
  results/predictions_ml.csv    (origin, model, featureset, y_true, y_pred)
  results/ml_tuning_report.json
"""
import json, time
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import importlib.util

spec = importlib.util.spec_from_file_location("feats", Path(__file__).with_name("07_features.py"))
feats = importlib.util.module_from_spec(spec); spec.loader.exec_module(feats)
SEED = feats.SEED
RES = Path(__file__).resolve().parents[1] / "results"

# Grid sized for the available hardware (2 CPUs): shallower forests via
# min_samples_leaf >= 2 and subsampled bootstrap draws; full spaces would not
# change the family comparison and are documented in the appendix.
GRIDS = {
    "RF":   [{"n_estimators":200,"min_samples_leaf":5,"max_features":0.5,"max_samples":0.7},
             {"n_estimators":200,"min_samples_leaf":2,"max_features":"sqrt","max_samples":0.7}],
    "LGBM": [{"n_estimators":1500,"learning_rate":0.05,"num_leaves":63},
             {"n_estimators":1500,"learning_rate":0.05,"num_leaves":127},
             {"n_estimators":800,"learning_rate":0.1,"num_leaves":63}],
    "MLP":  [{"hidden_layer_sizes":(128,64)},
             {"hidden_layer_sizes":(256,128)}],
}

def make_model(name, params):
    if name == "RF":
        return RandomForestRegressor(random_state=SEED, n_jobs=2, **params)
    if name == "LGBM":
        return lgb.LGBMRegressor(random_state=SEED, n_jobs=2, verbosity=-1, **params)
    if name == "MLP":
        return MLPRegressor(random_state=SEED, max_iter=80, early_stopping=True,
                            n_iter_no_change=8, **params)

def rmse_levels(y_log_true, y_log_pred):
    return float(np.sqrt(mean_squared_error(np.expm1(y_log_true),
                                            np.clip(np.expm1(y_log_pred), 0, None))))

def main():
    df = feats.load_panel()
    tuning, best_params = {}, {}
    scalers_cache = {}

    # ---- tuning on O1 inner split
    o1_train, _ = feats.split(df, feats.ORIGINS[0])
    inner_tr, inner_va = feats.inner_split(o1_train)
    for fs in ["nolags","lags"]:
        Xtr = feats.design_matrix(inner_tr, fs == "lags")
        Xva = feats.design_matrix(inner_va, fs == "lags").reindex(columns=Xtr.columns, fill_value=0.0)
        ytr, yva = inner_tr.log1p_value.values, inner_va.log1p_value.values
        sc = StandardScaler().fit(Xtr)
        for name, grid in GRIDS.items():
            scores = []
            for params in grid:
                m = make_model(name, params)
                t0 = time.time()
                if name == "MLP":
                    m.fit(sc.transform(Xtr), ytr); pred = m.predict(sc.transform(Xva))
                else:
                    m.fit(Xtr, ytr); pred = m.predict(Xva)
                scores.append((rmse_levels(yva, pred), params, round(time.time()-t0,1)))
            scores.sort(key=lambda s: s[0])
            best_params[(name, fs)] = scores[0][1]
            tuning[f"{name}_{fs}"] = [{"rmse_levels_val": round(s[0],1),
                                       "params": str(s[1]), "fit_s": s[2]} for s in scores]

    # ---- final fits per origin with frozen params
    rows = []
    for origin in feats.ORIGINS:
        train, test = feats.split(df, origin)
        for fs in ["nolags","lags"]:
            Xtr = feats.design_matrix(train, fs == "lags")
            Xte = feats.design_matrix(test, fs == "lags").reindex(columns=Xtr.columns, fill_value=0.0)
            ytr = train.log1p_value.values
            sc = StandardScaler().fit(Xtr)
            for name in GRIDS:
                m = make_model(name, best_params[(name, fs)])
                if name == "MLP":
                    m.fit(sc.transform(Xtr), ytr); pl = m.predict(sc.transform(Xte))
                else:
                    m.fit(Xtr, ytr); pl = m.predict(Xte)
                rows.append(pd.DataFrame({
                    "origin": origin["name"], "model": name, "featureset": fs,
                    "exporter": test.exporter.values, "destination": test.destination.values,
                    "hs6": test.hs6.values, "year": test.year.values,
                    "y_true": test.value_eur.values,
                    "y_pred": np.clip(np.expm1(pl), 0, None)}))
                print(origin["name"], name, fs, "done", flush=True)
    pd.concat(rows).to_csv(RES / "predictions_ml.csv", index=False)
    (RES / "ml_tuning_report.json").write_text(json.dumps(
        {"best_params": {f"{k[0]}_{k[1]}": str(v) for k, v in best_params.items()},
         "tuning": tuning}, indent=2))
    print(json.dumps({f"{k[0]}_{k[1]}": str(v) for k, v in best_params.items()}, indent=2))

if __name__ == "__main__":
    main()
