# Kernel entry: frozen dev-selected LightGBM models of PREREGISTRATION-3W-BEST.md.
#   phase 1 (dev wells only): out-of-fold predictions for every model and seed, thresholds on dev OOF, reproduction asserts against the dev study table;
#   phase 2: per model and seed, fit on all 30 dev wells, edge verification on dev-well probes, then ONE sealed-test scoring (seed 0 is the primary result).
# cfg["smoke"]: code-path check on dev wells only (dev fold 3 plays the part of the sealed test); no sealed-test well is loaded.
import hashlib, io, json, os, sys, time
import numpy as np
from w3 import features as F, gbdt_numpy as G, pipeline as P, scorer as S

OUT = "/kaggle/working"
cfg = json.load(open("/tmp/w3code/data/config.json"))
SMOKE = bool(cfg.get("smoke"))
T0 = time.time()
bench = S.Benchmark.load("/tmp/w3code/data")
index = json.load(open("/tmp/w3code/data/index_labels.json"))
files = sorted(index["files"], key=lambda f: f["path"])
assert [f["path"] for f in files] == bench.inst_path
assert hashlib.sha256(open("/tmp/w3code/data/manifest.csv", "rb").read()).hexdigest() == cfg["manifest_sha256"], "manifest differs from the revealed one"
for k, ref in cfg["decision_array_sha256"].items():
    assert P.array_digest(getattr(bench, k)) == ref, f"decision table array {k} differs"
P.log(f"benchmark loaded: {bench.n} decisions, {len(bench.wells)} wells; payload checks passed; smoke={SMOKE}")

folds, dev, test = P.fold_plan(bench)
active_wells = set(bench.wells)
if SMOKE:
    d1, d2, d3 = (bench.wells_with_role("dev", k) for k in (1, 2, 3))
    folds = [("dev1", d2, d1), ("dev2", d1, d2), ("test", d1 + d2, d3)]
    dev, test = d1 + d2, d3
    active_wells = set(d1 + d2 + d3)
    SEEDS = cfg["seeds"][:2]
else:
    SEEDS = cfg["seeds"]
assert 0 in SEEDS
active_inst = [i for i in range(len(files)) if bench.wells[bench.inst_well[i]] in active_wells]
X, _simple, have = P.build_feature_matrix(bench, files, active_inst, workers=4, want_simple=False)
if SMOKE:
    assert not have[bench.well_mask(bench.wells_with_role("test"))[bench.well_of_decision]].any(), "a sealed-test well was loaded in smoke mode"
names = F.feature_names()
col_sets = {"all": np.arange(len(names)), "scalefree": P.scale_free_columns(names, with_availability=False)}
groups = P.channel_col_groups(names)
XS = {"all": X, "scalefree": X[:, col_sets["scalefree"]]}                      # "all" is the matrix itself (no copy)
GL = {k: P.localize_groups(groups, c) for k, c in col_sets.items()}
NAMES_USED = {k: [names[i] for i in c] for k, c in col_sets.items()}
P.log(f"features {X.shape} in {time.time() - T0:.0f}s")
import lightgbm as lgb, sklearn
classes_eval = P.dev_classes(bench)
B = 50 if SMOKE else S.BOOT_B
dev_mask = bench.well_mask(dev)[bench.well_of_decision]
results = {"config": cfg, "classes_eval": classes_eval, "smoke": SMOKE, "seeds": SEEDS,
           "versions": {"lightgbm": lgb.__version__, "numpy": np.__version__, "sklearn": sklearn.__version__, "python": sys.version.split()[0]}}


def dump():
    json.dump(P.jsonable(results), open(f"{OUT}/results.json", "w"))


def fit_model(m, seed, train_wells, tag):
    """Same recipe as the dev study (kernels/dev_study.py): seed 0 reproduces its sampling and bagging exactly."""
    key = m["alias"] + tag + ("" if seed == 0 else f"#s{seed}")
    rng = np.random.default_rng(int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) % 100000)
    idx = P.train_sample(bench, X, have, train_wells, cfg["cap"], rng, well_cap=m.get("well_cap"))
    Xtr, ytr = XS[m["cols"]][idx], bench.y[idx].astype(int)
    if m["aug"]:
        Xtr = np.vstack([Xtr, P.augment_channel_dropout(Xtr, GL[m["cols"]], rng, p=0.3)]); ytr = np.concatenate([ytr, ytr])
    model = lgb.LGBMClassifier(objective="binary" if m["binary"] else "multiclass", n_estimators=m["rounds"], learning_rate=cfg["lr"], num_leaves=cfg["leaves"],
                               min_child_samples=50, subsample=0.7, subsample_freq=1, colsample_bytree=0.5, reg_lambda=1.0, class_weight="balanced",
                               n_jobs=4, random_state=seed, verbose=-1)
    model.fit(Xtr, (ytr > 0).astype(int) if m["binary"] else ytr)
    return model


def predict_rows(model, m, rows):
    Xs = XS[m["cols"]]
    pr = np.zeros((len(rows), S.N_CLASSES), dtype=np.float32)
    if m["binary"]:
        pp = model.predict_proba(Xs[rows])[:, 1]
        pr[:, 0], pr[:, 1] = 1.0 - pp, pp
    else:
        pr[:, np.asarray(model.classes_, dtype=int)] = model.predict_proba(Xs[rows])
    return pr


def eligible_rows(wells):
    return np.flatnonzero(bench.well_mask(wells)[bench.well_of_decision] & bench.elig & have)


# ---------------------------------------------------------------- phase 1: dev out-of-fold, thresholds, reproduction asserts (no test row is predicted)
state = {}
for tag, m in cfg["models"].items():
    for seed in SEEDS:
        t = time.time()
        proba = np.full((bench.n, S.N_CLASSES), np.nan, dtype=np.float32)
        for name, train_wells, pred_wells in folds[:-1]:
            model = fit_model(m, seed, train_wells, name)
            rows = eligible_rows(pred_wells)
            proba[rows] = predict_rows(model, m, rows)
        score_dev, pred_dev = P.to_score_pred(bench, P.ema_probs(bench, proba, m["alpha"]))
        theta, table = S.select_threshold(bench, score_dev, dev)
        pt = S.report(bench, score_dev, None if m["binary"] else pred_dev, dev, None, classes=classes_eval, boot=False)["point"]
        state[(tag, seed)] = {"proba": proba, "theta": theta, "dev_point": pt}
        want = m["expect_dev"]
        got = float(pt[want["metric"]])
        P.log(f"phase 1 {tag} seed {seed}: dev OOF {want['metric']} {got:.4f} (dev study {want['value']:.4f}), theta {theta}, {time.time() - t:.0f}s")
        results.setdefault(tag, {}).setdefault("seeds", {})[str(seed)] = {"theta": theta, "dev_oof_point": {k: pt.get(k) for k in ("macro_f1", "balanced_accuracy", "auroc", "average_precision")}}
        if seed == 0:
            results[tag]["dev_reproduction"] = {"metric": want["metric"], "value": got, "dev_study": want["value"], "tolerance": want["tol"]}
            if not SMOKE:
                assert abs(got - want["value"]) < want["tol"], f"{tag}: final dev OOF {got:.4f} does not reproduce the dev study {want['value']:.4f}; stopping before any test row is predicted"
    dump()
P.log(f"phase 1 done {time.time() - T0:.0f}s: all reproduction asserts passed; the sealed-test scoring starts below")


# ---------------------------------------------------------------- phase 2: final fit, edge verification (dev probes), one test scoring per model and seed
def edge_block(model, tag, m):
    """Numpy evaluator and int8-leaf export checked against LightGBM on 2,000 DEV-well probe decisions; sizes. Needs no test row."""
    path = f"{OUT}/{tag}_final.txt"
    model.booster_.save_model(path)
    mm = G.parse_model(path)
    Xs = XS[m["cols"]]
    dev_rows = np.flatnonzero(dev_mask & bench.elig & have)
    probe = np.random.default_rng(1).choice(dev_rows, 2000, replace=False)
    Xp = Xs[probe]
    want = model.predict_proba(Xp)
    want = want[:, [1]] if m["binary"] else want
    got = G.predict_proba(mm, Xp.astype(np.float64))
    c = G.compact_export(mm)
    buf = io.BytesIO(); np.savez_compressed(buf, **c)
    np.savez_compressed(f"{OUT}/{tag}_compact.npz", **c)
    m2 = G.from_compact(c, mm["n_class"])
    got2 = G.predict_proba(m2, Xp.astype(np.float64))
    dump_m = model.booster_.dump_model()
    used = sorted({int(f) for f in mm["feat"]})
    np.save(f"{OUT}/{tag}_probe_X.npy", Xp); np.save(f"{OUT}/{tag}_probe_proba.npy", want.astype(np.float32)); np.save(f"{OUT}/{tag}_probe_classes.npy", np.asarray(model.classes_))
    return {"n_trees": int(len(mm["root"])), "n_internal_nodes": int(len(mm["feat"])), "n_leaves": int(len(mm["leaf"])), "depth": int(mm["depth"]),
            "total_leaves_lightgbm": int(sum(tr["num_leaves"] for tr in dump_m["tree_info"])), "n_features_total": int(len(NAMES_USED[m["cols"]])), "n_features_split_on": len(used),
            "model_text_bytes": os.path.getsize(path), "compact_int8_leaf_raw_bytes": int(sum(v.nbytes for v in c.values())), "compact_int8_leaf_npz_bytes": len(buf.getvalue()),
            "verify_numpy_vs_lightgbm_max_abs_diff": float(np.abs(got - want).max()), "verify_numpy_vs_lightgbm_argmax_agreement": float((got.argmax(1) == want.argmax(1)).mean()) if got.shape[1] > 1 else None,
            "compact_vs_fp_max_abs_diff": float(np.abs(got2 - got).max()), "compact_vs_fp_argmax_agreement": float((got2.argmax(1) == got.argmax(1)).mean()) if got.shape[1] > 1 else None,
            "top_gain_features": [(n, float(g)) for g, n in sorted(zip(model.booster_.feature_importance("gain"), NAMES_USED[m["cols"]]), reverse=True)[:30]],
            "probe": "2000 random eligible DEV-well decisions (seed 1)"}, mm, m2


def compact_test_report(m2, m, classes, proba, rows, theta):
    """The int8-leaf model on the same sealed-test decisions with the same frozen theta and smoothing."""
    Xs = XS[m["cols"]]
    pr_all = np.zeros((len(rows), S.N_CLASSES), dtype=np.float32)
    for a in range(0, len(rows), 4000):
        r_ = rows[a:a + 4000]
        p = G.predict_proba(m2, Xs[r_].astype(np.float64))
        if m["binary"]:
            pr_all[a:a + 4000, 0], pr_all[a:a + 4000, 1] = 1.0 - p[:, 0], p[:, 0]
        else:
            pr_all[a:a + 4000, np.asarray(classes, dtype=int)] = p
    pc_raw = proba.copy(); pc_raw[rows] = pr_all
    sc, pr = P.to_score_pred(bench, P.ema_probs(bench, pc_raw, m["alpha"]))
    return {"theta": theta, "reports": {"test": S.report(bench, sc, None if m["binary"] else pr, test, theta, B=B)}}


for tag, m in cfg["models"].items():
    for seed in SEEDS:
        t = time.time()
        st = state[(tag, seed)]
        model = fit_model(m, seed, dev, "test")                                   # fitted on dev wells only; no test row touched yet
        classes = [int(c) for c in model.classes_]
        if seed == 0:
            edge, mm, m2 = edge_block(model, tag, m)
            results[tag]["edge"] = edge
            dump()
            P.log(f"{tag}: edge block done; numpy-vs-lightgbm max diff {edge['verify_numpy_vs_lightgbm_max_abs_diff']:.2e}")
        P.log(f"=== SEALED-TEST SCORING {tag} seed {seed} starts {time.strftime('%H:%M:%S')} ===")
        rows = eligible_rows(test)
        proba = st["proba"]
        proba[rows] = predict_rows(model, m, rows)
        score, pred = P.to_score_pred(bench, P.ema_probs(bench, proba, m["alpha"]))
        pred_arg = None if m["binary"] else pred
        theta = st["theta"]
        if seed == 0:
            rep = {"dev_oof": S.report(bench, score, pred_arg, dev, theta, classes=classes_eval, B=B), "test": S.report(bench, score, pred_arg, test, theta, B=B)}
            results[tag].update({"theta": theta, "reports": rep, "seconds": time.time() - t})
            np.savez_compressed(f"{OUT}/preds_{tag}.npz", score=score.astype(np.float32), **({} if m["binary"] else {"pred": pred}))
            results[f"{tag}_compact"] = compact_test_report(m2, m, classes, proba, rows, theta)
            dump()
        else:
            rep_t = S.report(bench, score, pred_arg, test, theta, boot=False)["point"]
            results[tag]["seeds"][str(seed)]["test_point"] = {k: rep_t.get(k) for k in ("macro_f1", "balanced_accuracy", "auroc", "average_precision", "episode_coverage",
                                                                                      "onset_detection_300s", "false_alarms_per_normal_day", "normal_time_occupation")}
            dump()
        P.log(f"=== SEALED-TEST SCORING {tag} seed {seed} done {time.strftime('%H:%M:%S')}  ({time.time() - t:.0f}s) ===")
results["seconds"] = time.time() - T0
dump()
P.log(f"all done {time.time() - T0:.0f}s")
