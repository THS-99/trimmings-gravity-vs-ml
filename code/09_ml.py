"""
09_ml.py
ML benchmark models (decision set mirrors fichas 5-7: one per family):
  RF    random forest           (Breiman-style bagged trees)
  LGBM  gradient boosting       (LightGBM)
  MLP   feed-forward network    (mirrors Morland et al.'s FFNN, honest simple
                                 architecture over fashionable ones)

Protocol:
  - identical one-hot design for all models (07_features.design_matrix),
    four variants: without lags, with lags, and the same two again plus the
    PPML spec B prediction as an extra feature ("hyb_", supervisor suggestion:
    raw data + gravity predictions into the ML models; 08 must run first)
  - target log1p(value_eur), scored in levels after expm1 (decision f)
  - hyperparameters tuned ONCE on origin O1's inner temporal split
    (train 2015-2021, validate 2022), then frozen across origins; tuning
    never sees any test year (no leakage across the time boundary). For the
    MLP the number of epochs is tuned on that same validation year instead of
    sklearn's early_stopping, which would hold out a random 10% of the rows.
    If results/ml_tuning_report.json already exists the frozen parameters are
    reused instead of re-tuned (delete the file to force a fresh search).
    Hybrid variants reuse the parameters of their base feature set: same
    model, one added feature, so any difference is the feature, not the tuning
  - fixed seeds (SEED=42)

Outputs:
  results/predictions_ml.csv    (origin, model, featureset, y_true, y_pred)
  results/ml_tuning_report.json
"""
import ast, gc, json, time, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.exceptions import ConvergenceWarning
import importlib.util
warnings.filterwarnings("ignore", category=ConvergenceWarning)

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

make_model = feats.make_model
MAX_EPOCHS, PATIENCE = 80, 8

def rmse_levels(y_log_true, y_log_pred):
    return float(np.sqrt(mean_squared_error(np.expm1(y_log_true),
                                            np.clip(np.expm1(y_log_pred), 0, None))))

def main():
    df = feats.load_panel()
    tuning, best_params = {}, {}
    scalers_cache = {}

    # ---- tuning on O1 inner split (skipped when the frozen report exists)
    report_path = RES / "ml_tuning_report.json"
    if report_path.exists():
        saved = json.loads(report_path.read_text())
        best_params = {tuple(k.rsplit("_", 1)): ast.literal_eval(v)
                       for k, v in saved["best_params"].items()}
        tuning = saved["tuning"]
        print("reusing frozen hyperparameters from ml_tuning_report.json", flush=True)
    o1_train, _ = feats.split(df, feats.ORIGINS[0])
    inner_tr, inner_va = feats.inner_split(o1_train)
    for fs in ([] if best_params else ["nolags","lags"]):
        Xtr = feats.design_matrix(inner_tr, fs == "lags")
        Xva = feats.design_matrix(inner_va, fs == "lags").reindex(columns=Xtr.columns, fill_value=0.0)
        ytr, yva = inner_tr.log1p_value.values, inner_va.log1p_value.values
        sc = StandardScaler().fit(Xtr)
        Xtr_s, Xva_s = sc.transform(Xtr), sc.transform(Xva)
        for name, grid in GRIDS.items():
            scores = []
            for params in grid:
                t0 = time.time()
                if name == "MLP":
                    # one epoch at a time, stop when the validation year (2022)
                    # has not improved for PATIENCE epochs; the best epoch count
                    # becomes part of the frozen parameters
                    m = MLPRegressor(random_state=SEED, **params)
                    best, best_ep, bad = np.inf, 0, 0
                    for ep in range(1, MAX_EPOCHS + 1):
                        m.partial_fit(Xtr_s, ytr)
                        r = rmse_levels(yva, m.predict(Xva_s))
                        if r < best:
                            best, best_ep, bad = r, ep, 0
                        else:
                            bad += 1
                        if bad == PATIENCE:
                            break
                    scores.append((best, {**params, "epochs": best_ep}, round(time.time()-t0,1)))
                    continue
                m = make_model(name, params)
                m.fit(Xtr, ytr); pred = m.predict(Xva)
                scores.append((rmse_levels(yva, pred), params, round(time.time()-t0,1)))
            scores.sort(key=lambda s: s[0])
            best_params[(name, fs)] = scores[0][1]
            tuning[f"{name}_{fs}"] = [{"rmse_levels_val": round(s[0],1),
                                       "params": str(s[1]), "fit_s": s[2]} for s in scores]

    # ---- final fits per origin with frozen params
    # each fit is checkpointed so the script resumes instead of restarting when
    # the small container kills a long run
    part_path = RES / "predictions_ml_partial.csv"
    rows, done_keys = [], set()
    if part_path.exists():
        prev = pd.read_csv(part_path, dtype={"hs6": str})
        rows.append(prev)
        done_keys = set(map(tuple, prev[["origin","model","featureset"]].drop_duplicates().values))
        print("resuming:", len(done_keys), "fits already saved", flush=True)
    for origin in feats.ORIGINS:
        train, test = feats.split(df, origin)
        for fs in ["nolags","lags","hyb_nolags","hyb_lags"]:
            if all((origin["name"], name, fs) in done_keys for name in GRIDS):
                continue
            with_lags = fs in ("lags", "hyb_lags")
            Xtr = feats.design_matrix(train, with_lags)
            Xte = feats.design_matrix(test, with_lags)
            if fs.startswith("hyb_"):
                Xtr["ppml_pred_log"] = feats.ppml_feature(train)
                Xte["ppml_pred_log"] = feats.ppml_feature(test)
            Xte = Xte.reindex(columns=Xtr.columns, fill_value=0.0)
            ytr = train.log1p_value.values
            sc = StandardScaler().fit(Xtr)
            for name in GRIDS:
                if (origin["name"], name, fs) in done_keys:
                    continue
                m = make_model(name, best_params[(name, fs.replace("hyb_", ""))])
                if name == "MLP":
                    Xtr_s = sc.transform(Xtr).astype(np.float32)
                    Xte_s = sc.transform(Xte).astype(np.float32)
                    m.fit(Xtr_s, ytr); pl = m.predict(Xte_s)
                    del Xtr_s, Xte_s
                else:
                    m.fit(Xtr, ytr); pl = m.predict(Xte)
                block = pd.DataFrame({
                    "origin": origin["name"], "model": name, "featureset": fs,
                    "exporter": test.exporter.values, "destination": test.destination.values,
                    "hs6": test.hs6.values, "year": test.year.values,
                    "y_true": test.value_eur.values,
                    "y_pred": np.clip(np.expm1(pl), 0, None)})
                block.to_csv(part_path, mode="a", header=not part_path.exists(), index=False)
                rows.append(block)
                del m
                gc.collect()
                print(origin["name"], name, fs, "done", flush=True)
    pd.concat(rows).to_csv(RES / "predictions_ml.csv", index=False)
    (RES / "ml_tuning_report.json").write_text(json.dumps(
        {"best_params": {f"{k[0]}_{k[1]}": str(v) for k, v in best_params.items()},
         "tuning": tuning}, indent=2))
    print(json.dumps({f"{k[0]}_{k[1]}": str(v) for k, v in best_params.items()}, indent=2))

if __name__ == "__main__":
    main()
