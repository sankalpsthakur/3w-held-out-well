# Kernel entry: development study for the best-model search. DEV wells only: no sealed-test well is read, predicted or scored.
import hashlib, json, os, sys, time
import numpy as np
from w3 import features as F, pipeline as P, scorer as S

OUT = "/kaggle/working"
cfg = json.load(open("/tmp/w3code/data/config.json"))
T0 = time.time()
bench = S.Benchmark.load("/tmp/w3code/data")
index = json.load(open("/tmp/w3code/data/index_labels.json"))
files = sorted(index["files"], key=lambda f: f["path"])
assert hashlib.sha256(open("/tmp/w3code/data/manifest.csv", "rb").read()).hexdigest() == cfg["manifest_sha256"]
dev_wells = bench.wells_with_role("dev")
active_inst = [i for i in range(len(files)) if bench.wells[bench.inst_well[i]] in set(dev_wells)]
X, _simple, have = P.build_feature_matrix(bench, files, active_inst, workers=4, want_simple=False)
names = F.feature_names()
P.log(f"dev features {X.shape} in {time.time() - T0:.0f}s (test instances not loaded)")
assert not have[bench.well_mask(bench.wells_with_role("test"))[bench.well_of_decision]].any()
groups = P.channel_col_groups(names)
col_sets = {"all": np.arange(len(names)), "scalefree": P.scale_free_columns(names, with_availability=False)}
folds = [(f"dev{k}", [w for w in dev_wells if w not in bench.wells_with_role("dev", k)], bench.wells_with_role("dev", k)) for k in range(5)]
ROUNDS = cfg["rounds"]; ALPHAS = cfg["alphas"]; CAP = cfg["cap"]
import lightgbm as lgb
classes_eval = P.dev_classes(bench)
results = {"config": cfg, "classes_eval": classes_eval, "variants": []}
dev_mask = bench.well_mask(dev_wells)[bench.well_of_decision]

for cand in cfg["candidates"]:           # each: {"name", "cols": "all|scalefree", "aug": 0|1}
    t_c = time.time()
    cols = col_sets[cand["cols"]]; Xs = X[:, cols]; gl = P.localize_groups(groups, cols)
    staged = {r: np.full((bench.n, S.N_CLASSES), np.nan, dtype=np.float32) for r in ROUNDS}
    for name, train_wells, pred_wells in folds:
        rng = np.random.default_rng(int(hashlib.sha256((cand["name"] + name).encode()).hexdigest()[:8], 16) % 100000)
        idx = P.train_sample(bench, X, have, train_wells, CAP, rng, well_cap=cand.get("well_cap"))
        Xtr, ytr = Xs[idx], bench.y[idx].astype(int)
        if cand["aug"]:
            Xtr = np.vstack([Xtr, P.augment_channel_dropout(Xtr, gl, rng, p=0.3)]); ytr = np.concatenate([ytr, ytr])
        binary = bool(cand.get("binary"))
        model = lgb.LGBMClassifier(objective="binary" if binary else "multiclass", n_estimators=max(ROUNDS), learning_rate=cfg["lr"], num_leaves=cand.get("leaves", cfg["leaves"]),
                                   min_child_samples=cand.get("min_child", 50), subsample=0.7, subsample_freq=1, colsample_bytree=cand.get("colsample", 0.5),
                                   reg_lambda=1.0, class_weight="balanced", extra_trees=bool(cand.get("extra_trees", False)), n_jobs=4, random_state=0, verbose=-1)
        model.fit(Xtr, (ytr > 0).astype(int) if binary else ytr)
        rows = np.flatnonzero(bench.well_mask(pred_wells)[bench.well_of_decision] & bench.elig & have)
        for r in ROUNDS:
            pr = np.zeros((len(rows), S.N_CLASSES), dtype=np.float32)
            if binary:
                pp = model.predict_proba(Xs[rows], num_iteration=r)[:, 1]
                pr[:, 0], pr[:, 1] = 1.0 - pp, pp
            else:
                pr[:, np.asarray(model.classes_, dtype=int)] = model.predict_proba(Xs[rows], num_iteration=r)
            staged[r][rows] = pr
        P.log(f"  {cand['name']} {name}: train {len(Xtr)} rows {time.time() - t_c:.0f}s")
    for r in ROUNDS:
        for a in ALPHAS:
            pr = P.ema_probs(bench, staged[r], a)
            score, pred = P.to_score_pred(bench, pr)
            rep = S.report(bench, score, None if cand.get("binary") else pred, dev_wells, None, classes=classes_eval, boot=False)["point"]
            if cand.get("binary"):
                rep = {**{k: float("nan") for k in ("macro_f1", "balanced_accuracy", "accuracy", "per_well_macro_f1_mean")}, **rep}
            leaves = r * (1 if cand.get("binary") else S.N_CLASSES) * cand.get("leaves", cfg["leaves"])
            results["variants"].append({"cand": cand["name"], "rounds": r, "ema_alpha": a, "approx_leaves": leaves,
                                        "binary": bool(cand.get("binary")), **{k: rep.get(k, float("nan")) for k in ("macro_f1", "balanced_accuracy", "accuracy", "auroc", "average_precision", "per_well_macro_f1_mean")},
                                        **{k: rep[k] for k in rep if k.startswith("f1_c")}})
    json.dump(P.jsonable(results), open(f"{OUT}/dev_study.json", "w"))
    key = (lambda v: v["average_precision"]) if cand.get("binary") else (lambda v: v["macro_f1"])
    best = max((v for v in results["variants"] if v["cand"] == cand["name"]), key=key)
    P.log(f"== {cand['name']} done in {time.time() - t_c:.0f}s; best {'AP' if cand.get('binary') else 'macro_f1'} {key(best):.4f} (rounds {best['rounds']}, ema {best['ema_alpha']}, auroc {best['auroc']:.4f})")
    if cand.get("keep_proba"):
        np.savez_compressed(f"{OUT}/dev_proba_{cand['name']}.npz", **{f"r{r}": staged[r].astype(np.float16) for r in ROUNDS})
P.log(f"all done {time.time() - T0:.0f}s")


# ---- leakage study: same model, same dev data, three ways to split (decision-random, instance-random, well-grouped)
if cfg.get("leak_study"):
    import lightgbm as lgb
    def make_lgbm():
        return lgb.LGBMClassifier(objective="multiclass", n_estimators=200, learning_rate=0.05, num_leaves=31, min_child_samples=50, subsample=0.7,
                                  subsample_freq=1, colsample_bytree=0.5, reg_lambda=1.0, class_weight="balanced", n_jobs=4, random_state=0, verbose=-1)
    leak = {}
    inst_bucket = np.array([P.stable_bucket(p) for p in bench.inst_path])
    dec_bucket = np.random.default_rng(7).integers(0, 5, bench.n)
    schemes = {
        "decision_random": [(f"dr{k}", dev_mask & (dec_bucket != k), dev_mask & (dec_bucket == k)) for k in range(5)],
        "instance_random": [(f"ir{k}", dev_mask & (inst_bucket[bench.inst] != k), dev_mask & (inst_bucket[bench.inst] == k)) for k in range(5)],
        "well_grouped": [(f"wg{k}", dev_mask & ~bench.well_mask(bench.wells_with_role("dev", k))[bench.well_of_decision], bench.well_mask(bench.wells_with_role("dev", k))[bench.well_of_decision]) for k in range(5)],
    }
    for scheme, fl in schemes.items():
        t_s = time.time()
        proba = P.run_mask_folds(bench, X, have, make_lgbm, CAP, fl)
        score, pred = P.to_score_pred(bench, proba)
        S_, _, _ = S.well_statistics(bench, score, pred, dev_wells, None)
        for label, cls in (("classes_all_present", None), ("classes_dev_eval", classes_eval)):
            m = S.metrics_from_sums(S_.sum(0), cls)
            leak[f"{scheme}/{label}"] = {k: float(np.asarray(m[k])) for k in ("macro_f1", "balanced_accuracy", "accuracy", "auroc", "average_precision")}
            leak[f"{scheme}/{label}"].update({k: (float(np.asarray(m[k])) if np.isfinite(np.asarray(m[k])) else None) for k in m if k.startswith("f1_c")})
        leak[f"{scheme}/seconds"] = time.time() - t_s
        results["leak_study"] = leak
        json.dump(P.jsonable(results), open(f"{OUT}/dev_study.json", "w"))
        P.log(f"leak {scheme}: macro_f1(dev classes) {leak[scheme + '/classes_dev_eval']['macro_f1']:.4f}  auroc {leak[scheme + '/classes_dev_eval']['auroc']:.4f}")
    P.log("leak study done")
