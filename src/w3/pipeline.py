"""3W-HW experiment plumbing shared by kernel entry scripts: data -> features -> grouped folds -> predictions -> scorer reports.

Model objects only need ``fit(X, y)`` and ``predict_proba(X)`` (+ ``classes_``); the protocol logic lives here, not in the models.
Nothing in this module reads labels of held-out wells before predictions of that fold are final.
"""
from __future__ import annotations

import hashlib
import io
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from . import features as F
from . import scorer as S

COMMIT = "d57c058af6baf65f1b285f3e56f46e4f018a07d9"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def fetch(path, retries=5):
    url = f"https://raw.githubusercontent.com/petrobras/3W/{COMMIT}/{path}"
    last = None
    for k in range(retries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "bench-events/1"}), timeout=120) as r:
                return r.read()
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(1.5 * (k + 1))
    raise last


def load_signals(f):
    """Download + verify one instance; returns (cont [rows,18], state [rows,9]) float32 with NaN for absent columns."""
    import pyarrow.parquet as pq

    d = fetch(f["path"])
    blob = hashlib.sha1(b"blob " + str(len(d)).encode() + b"\0" + d).hexdigest()
    if blob != f["sha"]:
        raise ValueError("blob sha mismatch " + f["path"])
    pf = pq.ParquetFile(io.BytesIO(d))
    names = set(pf.schema_arrow.names)
    want = [c for c in F.CONT + F.STATE if c in names]
    tbl = pf.read(columns=want)
    rows = tbl.num_rows
    cont = np.full((rows, len(F.CONT)), np.nan, dtype=np.float32)
    state = np.full((rows, len(F.STATE)), np.nan, dtype=np.float32)
    for c in want:
        a = np.asarray(tbl[c].to_numpy(zero_copy_only=False), dtype=np.float64)
        if c in F.CONT:
            cont[:, F.CONT.index(c)] = a
        else:
            state[:, F.STATE.index(c)] = a
    return cont, state


def build_feature_matrix(bench, files, active_inst, workers=4, want_simple=True):
    """Features for every decision of the active instances. Rows of inactive instances stay NaN / have_feat=False."""
    nfeat = len(F.feature_names())
    X = np.full((bench.n, nfeat), np.nan, dtype=np.float32)
    simple = np.full(bench.n, np.nan, dtype=np.float32)
    have = np.zeros(bench.n, dtype=bool)
    t0 = time.time()

    def one(i):
        cont, state = load_signals(files[i])
        off, end = bench.inst_offset[i], bench.inst_offset[i + 1]
        fx = F.instance_features(cont, state)
        assert fx.shape[0] == end - off, (files[i]["path"], fx.shape, end - off)
        sc = F.simple_detector_score(cont) if want_simple else None
        return i, off, end, fx, sc

    done = 0
    with ThreadPoolExecutor(workers) as ex:
        for i, off, end, fx, sc in ex.map(one, active_inst):
            X[off:end] = fx
            if sc is not None:
                simple[off:end] = sc
            have[off:end] = True
            done += 1
            if done % 100 == 0:
                log(f"features {done}/{len(active_inst)} instances ({time.time() - t0:.0f}s)")
    return X, simple, have


def train_sample(bench, X, have, wells, cap_per_class, rng, every=2, well_cap=None):
    """Training decisions: eligible+labelled decisions of `wells` at every `every`-th decision, capped per class (and optionally per class x well)."""
    m = bench.well_mask(wells)[bench.well_of_decision] & bench.elig & (bench.y >= 0) & have
    idx = np.flatnonzero(m)[::every]
    out = []
    for c in range(S.N_CLASSES):
        ic = idx[bench.y[idx] == c]
        if well_cap:
            keep = []
            wc = bench.well_of_decision[ic]
            for w in np.unique(wc):
                iw = ic[wc == w]
                keep.append(iw if len(iw) <= well_cap else rng.choice(iw, well_cap, replace=False))
            ic = np.concatenate(keep) if keep else ic
        if len(ic) > cap_per_class:
            ic = rng.choice(ic, cap_per_class, replace=False)
        out.append(ic)
    return np.sort(np.concatenate(out))


def predict_full(model, X, rows):
    p = np.zeros((len(rows), S.N_CLASSES), dtype=np.float32)
    chunk = 100000
    for a in range(0, len(rows), chunk):
        pr = model.predict_proba(X[rows[a:a + chunk]])
        p[a:a + chunk][:, np.asarray(model.classes_, dtype=int)] = pr
    return p


def run_folds(bench, X, have, make_model, cap_per_class, folds, seed=0, fit_kwargs=None, keep_final=False, feature_idx=None):
    """folds: list of (name, train_wells, predict_wells). Returns proba [N,10] float16 (only rows of predicted wells), models dict."""
    proba = np.full((bench.n, S.N_CLASSES), np.nan, dtype=np.float16)
    models = {}
    Xs = X if feature_idx is None else X[:, feature_idx]
    for name, train_wells, pred_wells in folds:
        t0 = time.time()
        rng = np.random.default_rng(seed + int(hashlib.sha256(name.encode()).hexdigest()[:8], 16) % 1000)
        idx = train_sample(bench, X, have, train_wells, cap_per_class, rng)
        model = make_model()
        model.fit(Xs[idx], bench.y[idx].astype(int), **(fit_kwargs or {}))
        rows = np.flatnonzero(bench.well_mask(pred_wells)[bench.well_of_decision] & bench.elig & have)
        proba[rows] = predict_full(model, Xs, rows).astype(np.float16)
        log(f"  fold {name}: train {len(idx)} rows, predict {len(rows)} rows, {time.time() - t0:.0f}s")
        if keep_final and name == "test":
            models["final"] = model
    return proba, models


def to_score_pred(bench, proba):
    p = proba.astype(np.float32)
    score = np.where(np.isfinite(p[:, 0]), 1.0 - p[:, 0], np.nan)
    pred = np.where(np.isfinite(p[:, 0]), np.argmax(np.nan_to_num(p, nan=-1.0), axis=1), -1).astype(np.int8)
    return score.astype(np.float64), pred


def fold_plan(bench):
    dev = bench.wells_with_role("dev")
    test = bench.wells_with_role("test")
    folds = []
    for k in range(5):
        held = bench.wells_with_role("dev", k)
        folds.append((f"dev{k}", [w for w in dev if w not in held], held))
    folds.append(("test", dev, test))
    return folds, dev, test


def dev_classes(bench):
    """Event classes supported in >= 2 dev wells (dev-CV can train and evaluate them in different wells); class 0 always."""
    dev = set(bench.wells_with_role("dev"))
    cnt = {}
    for w, ev in zip(bench.ep_well, bench.ep_event):
        if bench.wells[w] in dev:
            cnt.setdefault(int(ev), set()).add(int(w))
    return [0] + sorted(c for c, ws in cnt.items() if len(ws) >= 2)


def report_all(bench, score, pred, theta, dev, test, B=S.BOOT_B, tag=""):
    out = {}
    out["dev_oof"] = S.report(bench, score, pred, dev, theta, classes=dev_classes(bench), B=B) if pred is not None else S.report(bench, score, None, dev, theta, B=B)
    out["test"] = S.report(bench, score, pred, test, theta, B=B)
    allw = dev + test
    out["cv6_all_wells"] = S.report(bench, score, pred, allw, theta, B=B)
    return out


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return jsonable(o.tolist())
    return o


def array_digest(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


class VolatilityDetector:
    """Simple one-class baseline: scale-free 15-minute variability index normalised on training-normal decisions.

    score_c = cv_w3 / q99_c(training normal) per channel; v = max_c score_c over channels with finite cv_w3; P(event) = v / (v + 1).
    Fit uses only rows with label 0 of the training wells. No class information beyond normal vs not-normal is used.
    """
    classes_ = np.array([0, 1])

    def __init__(self, feature_names):
        self.cols = [i for i, n in enumerate(feature_names) if n.endswith("|cv_w3")]

    def fit(self, X, y):
        Z = X[:, self.cols][np.asarray(y) == 0]
        self.q = np.nanpercentile(Z, 99, axis=0)
        self.q = np.where(np.isfinite(self.q) & (self.q > 0), self.q, np.nan)
        return self

    def predict_proba(self, X):
        Z = X[:, self.cols] / self.q
        with np.errstate(all="ignore"):
            v = np.nanmax(np.where(np.isfinite(Z), Z, -np.inf), axis=1)
        v = np.where(np.isfinite(v), np.maximum(v, 0.0), 0.0)
        s = v / (v + 1.0)
        return np.stack([1.0 - s, s], axis=1).astype(np.float32)


# ----------------------------------------------------------------------------- feature subsets, augmentation, smoothing
def channel_col_groups(names):
    """Column indices of each of the 27 channels (18 continuous, 9 valve states) in the feature matrix."""
    groups = []
    for ch in F.CONT + F.STATE:
        groups.append(np.array([i for i, n in enumerate(names) if n.split("|")[0] == ch], dtype=int))
    return groups


SCALE_FREE_FEATS = ["range_rel_w3", "drift_w3", "z_last_w4", "z_w2_w4", "cv_w3", "extrema_rate_w3", "rough_w3"]


def scale_free_columns(names, with_availability=False):
    keep = set(SCALE_FREE_FEATS + (["finite_w3"] if with_availability else []))
    state = set(F.STATE)
    return np.array([i for i, n in enumerate(names) if (n.split("|")[0] in state) or n.split("|")[1] in keep], dtype=int)


def augment_channel_dropout(Xs, groups_local, rng, p=0.3):
    """Copy of Xs with each channel's columns set to NaN independently with probability p per row (mimics absent instrumentation)."""
    Xa = Xs.copy()
    for cols in groups_local:
        if len(cols) == 0:
            continue
        m = rng.random(len(Xa)) < p
        Xa[np.ix_(np.flatnonzero(m), cols)] = np.nan
    return Xa


def localize_groups(groups, keep_cols):
    pos = {c: i for i, c in enumerate(keep_cols)}
    return [np.array([pos[c] for c in g if c in pos], dtype=int) for g in groups]


def ema_probs(bench, proba, alpha, rows_mask=None):
    """Causal exponential smoothing of class probabilities along each instance's decision sequence (alpha=1: no smoothing)."""
    from scipy.signal import lfilter

    out = proba.astype(np.float32).copy()
    if alpha >= 1.0:
        return out
    for i in range(len(bench.inst_rows)):
        a, b = bench.inst_offset[i], bench.inst_offset[i + 1]
        p = out[a:b]
        ok = np.isfinite(p[:, 0])
        if ok.sum() == 0:
            continue
        idx = np.where(ok, np.arange(len(p)), 0)
        idx = np.maximum.accumulate(idx)                 # forward-fill missing rows with the last finite one
        first = np.flatnonzero(ok)[0]
        idx[:first] = first
        q = p[idx]
        zi = (1 - alpha) * q[0:1]
        y, _ = lfilter([alpha], [1.0, -(1.0 - alpha)], q, axis=0, zi=zi)
        out[a:b] = np.where(ok[:, None], y, np.nan)
    return out


# ----------------------------------------------------------------------------- leakage study: random-split folds on the same data
def train_sample_mask(bench, have, base_mask, cap_per_class, rng, every=2):
    m = base_mask & bench.elig & (bench.y >= 0) & have
    idx = np.flatnonzero(m)[::every]
    out = []
    for c in range(S.N_CLASSES):
        ic = idx[bench.y[idx] == c]
        if len(ic) > cap_per_class:
            ic = rng.choice(ic, cap_per_class, replace=False)
        out.append(ic)
    return np.sort(np.concatenate(out))


def run_mask_folds(bench, X, have, make_model, cap_per_class, folds, seed=0):
    """folds: list of (name, train_mask, pred_mask) over the decision table. Returns proba [N,10] float16."""
    proba = np.full((bench.n, S.N_CLASSES), np.nan, dtype=np.float16)
    for name, train_mask, pred_mask in folds:
        t0 = time.time()
        rng = np.random.default_rng(seed + int(hashlib.sha256(name.encode()).hexdigest()[:8], 16) % 1000)
        idx = train_sample_mask(bench, have, train_mask, cap_per_class, rng)
        model = make_model()
        model.fit(X[idx], bench.y[idx].astype(int))
        rows = np.flatnonzero(pred_mask & bench.elig & have)
        proba[rows] = predict_full(model, X, rows).astype(np.float16)
        log(f"  fold {name}: train {len(idx)} rows, predict {len(rows)} rows, {time.time() - t0:.0f}s")
    return proba


def stable_bucket(key, k=5, salt="3w-hw-leak-v1"):
    return int(hashlib.sha256(f"{salt}|{key}".encode()).hexdigest()[:8], 16) % k
