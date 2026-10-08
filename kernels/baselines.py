# Kernel entry: preregistered baselines on 3W-HW v1 (random forest, LightGBM, simple causal detector).
import hashlib, json, os, pickle, sys, time
import numpy as np
from w3 import features as F, pipeline as P, scorer as S

OUT = "/kaggle/working"
cfg = json.load(open("/tmp/w3code/data/config.json"))
SMOKE = cfg["mode"] == "smoke"
T0 = time.time()
bench = S.Benchmark.load("/tmp/w3code/data")
index = json.load(open("/tmp/w3code/data/index_labels.json"))
files = sorted(index["files"], key=lambda f: f["path"])
assert [f["path"] for f in files] == bench.inst_path
assert hashlib.sha256(open("/tmp/w3code/data/manifest.csv", "rb").read()).hexdigest() == cfg["manifest_sha256"], "manifest differs from the revealed one"
for k, ref in cfg["decision_array_sha256"].items():
    assert P.array_digest(getattr(bench, k if k != "inst" else "inst")) == ref, f"decision table array {k} differs"
P.log(f"benchmark loaded: {bench.n} decisions, {len(bench.wells)} wells; payload checks passed; mode={cfg['mode']}")

folds, dev, test = P.fold_plan(bench)
if SMOKE:   # code-path check on dev wells only: never touches test wells
    d0, d1, d2 = (bench.wells_with_role("dev", k) for k in (0, 1, 2))
    folds = [("dev0", d1 + d2, d0), ("smoke_final", d0 + d1, d2)]
    dev, test = d0 + d2, []
    active_wells = set(d0 + d1 + d2)
else:
    active_wells = set(bench.wells)
active_inst = [i for i in range(len(files)) if bench.wells[bench.inst_well[i]] in active_wells]
X, simple_raw, have = P.build_feature_matrix(bench, files, active_inst, workers=4)
names = F.feature_names()
json.dump(names, open(f"{OUT}/feature_names.json", "w"))
P.log(f"feature matrix {X.shape} built in {time.time() - T0:.0f}s; finite fraction {np.isfinite(X[have][::50]).mean():.3f}")

results = {"config": cfg, "n_features": len(names), "versions": {}, "timing": {}}
import sklearn, lightgbm
results["versions"] = {"numpy": np.__version__, "sklearn": sklearn.__version__, "lightgbm": lightgbm.__version__, "python": sys.version.split()[0]}
CAP = 500 if SMOKE else 20000
scored = {}

def finish(name, score, pred, extra=None):
    theta, table = S.select_threshold(bench, score, dev)
    if SMOKE:
        rep = {"dev_oof": S.report(bench, score, pred, dev, theta, classes=P.dev_classes(bench), B=50)}
    else:
        rep = P.report_all(bench, score, pred, theta, dev, test)
    results[name] = {"theta": theta, "threshold_table_head": table[:3], "reports": rep, **(extra or {})}
    json.dump(P.jsonable(results), open(f"{OUT}/results.json", "w"))
    P.log(f"{name}: theta={theta} done")

# ---- simple causal detector (no training)
t = time.time()
simple_unit = F.simple_score_to_unit(simple_raw)
sel = bench.elig & have
unusable = float(np.isnan(simple_unit[sel]).mean())
simple_score = np.where(sel, np.nan_to_num(simple_unit, nan=0.0), np.nan)
finish("simple", simple_score, None, {"unusable_fraction": unusable, "seconds": time.time() - t})
np.savez_compressed(f"{OUT}/preds_simple.npz", score=simple_score.astype(np.float32))

# ---- one-class volatility detector (fit on training-normal decisions of each fold)
t = time.time()
proba_v, _ = P.run_folds(bench, X, have, lambda: P.VolatilityDetector(names), CAP, folds, seed=0)
score_v, _ = P.to_score_pred(bench, proba_v)
finish("volatility", score_v, None, {"seconds": time.time() - t, "parameters": len([n for n in names if n.endswith("|cv_w3")])})
np.savez_compressed(f"{OUT}/preds_volatility.npz", score=score_v.astype(np.float32))
del proba_v

# ---- random forest and LightGBM with preregistered hyperparameters
def make_rf():
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(n_estimators=10 if SMOKE else 100, min_samples_leaf=10, max_features="sqrt", max_samples=0.3,
                                  class_weight="balanced_subsample", n_jobs=4, random_state=0)

def make_lgbm():
    import lightgbm as lgb
    return lgb.LGBMClassifier(objective="multiclass", n_estimators=20 if SMOKE else 200, learning_rate=0.05, num_leaves=31, min_child_samples=50,
                              subsample=0.7, subsample_freq=1, colsample_bytree=0.5, reg_lambda=1.0, class_weight="balanced", n_jobs=4,
                              random_state=0, verbose=-1)

for name, maker in (("lgbm", make_lgbm), ("rf", make_rf)):
    t = time.time()
    P.log(f"== {name}")
    proba, models = P.run_folds(bench, X, have, maker, CAP, folds, seed=0, keep_final=True)
    score, pred = P.to_score_pred(bench, proba)
    extra = {"seconds": time.time() - t}
    final = models.get("final")
    if final is not None:
        try:
            if name == "lgbm":
                final.booster_.save_model(f"{OUT}/lgbm_final.txt")
                dump = final.booster_.dump_model()
                leaves = sum(tr["num_leaves"] for tr in dump["tree_info"])
                extra.update({"n_trees": len(dump["tree_info"]), "total_leaves": leaves, "nodes_total": 2 * leaves - len(dump["tree_info"]),
                              "model_text_bytes": os.path.getsize(f"{OUT}/lgbm_final.txt")})
                imp = sorted(zip(final.booster_.feature_importance("gain"), names), reverse=True)[:40]
                extra["top_gain_features"] = [(n, float(g)) for g, n in imp]
                probe = np.random.default_rng(1).choice(np.flatnonzero(have & bench.elig), 2000, replace=False)
                np.save(f"{OUT}/probe_X.npy", X[probe]); np.save(f"{OUT}/probe_proba.npy", final.predict_proba(X[probe]).astype(np.float32))
                np.save(f"{OUT}/probe_classes.npy", np.asarray(final.classes_))
            else:
                nodes = int(sum(e.tree_.node_count for e in final.estimators_))
                extra.update({"n_trees": len(final.estimators_), "nodes_total": nodes, "pickle_bytes": len(pickle.dumps(final, protocol=4))})
        except Exception as e:  # optional artefacts must never lose the predictions
            extra["artifact_error"] = repr(e)[:300]
            P.log("artifact error: " + repr(e)[:300])
    finish(name, score, pred, extra)
    np.savez_compressed(f"{OUT}/preds_{name}.npz", score=score.astype(np.float32), pred=pred, proba=proba)
    del proba, models
results["timing"]["total_seconds"] = time.time() - T0
json.dump(P.jsonable(results), open(f"{OUT}/results.json", "w"))
P.log(f"all done in {time.time() - T0:.0f}s")
