"""LightGBM text model -> flat numpy arrays, batch/streaming evaluator, and compact int8-leaf export (no lightgbm needed).

Used for edge measurements on the Mac (1 thread, batch 1) and to verify exported models against the Kaggle-side predictions.
Semantics follow LightGBM's Tree::NumericalDecision (missing_type none/zero/nan, default_left, <= threshold goes left).
"""
from __future__ import annotations

import re

import numpy as np


def _arr(block, key, dtype):
    for line in block:
        if line.startswith(key + "="):
            s = line.split("=", 1)[1].strip()
            return np.array([] if s == "" else s.split(" "), dtype=dtype)
    return None


def parse_model(path_or_text):
    text = open(path_or_text).read() if "\n" not in str(path_or_text) and len(str(path_or_text)) < 4096 else path_or_text
    head, *trees = text.split("\nTree=")
    meta = {}
    for line in head.splitlines():
        if "=" in line and not line.startswith("feature_") and not line.startswith("tree_sizes"):
            k, v = line.split("=", 1)
            meta[k] = v
    n_class = int(meta.get("num_class", 1))
    per_iter = int(meta.get("num_tree_per_iteration", n_class))
    nodes = {k: [] for k in ("feat", "thr", "dt", "left", "right")}
    leaves, tree_off_node, tree_off_leaf, tree_class, tree_nleaf = [], [], [], [], []
    n_node = n_leaf = 0
    for ti, tb in enumerate(trees):
        block = tb.split("\n\n")[0].splitlines()
        nl = int(_arr(block, "num_leaves", int)[0])
        lv = _arr(block, "leaf_value", float)
        tree_off_node.append(n_node); tree_off_leaf.append(n_leaf); tree_class.append(ti % per_iter); tree_nleaf.append(nl)
        if nl == 1:                       # stump: single leaf
            nodes["feat"].append(np.array([0])); nodes["thr"].append(np.array([0.0])); nodes["dt"].append(np.array([0]))
            nodes["left"].append(np.array([~n_leaf])); nodes["right"].append(np.array([~n_leaf]))
            n_node += 1
        else:
            sf = _arr(block, "split_feature", int); th = _arr(block, "threshold", float); dt = _arr(block, "decision_type", int)
            lc = _arr(block, "left_child", int); rc = _arr(block, "right_child", int)
            nodes["feat"].append(sf); nodes["thr"].append(th); nodes["dt"].append(dt)
            # internal children stay tree-local ids offset to global; leaf children (negative) are encoded as ~global_leaf
            enc = lambda c: np.where(c >= 0, c + n_node, ~(~c + n_leaf))  # noqa: E731
            nodes["left"].append(enc(lc)); nodes["right"].append(enc(rc))
            n_node += len(sf)
        leaves.append(lv); n_leaf += nl
    out = {k: np.concatenate(v) for k, v in nodes.items()}
    out["feat"] = out["feat"].astype(np.int32); out["dt"] = out["dt"].astype(np.int32)
    out["left"] = out["left"].astype(np.int32); out["right"] = out["right"].astype(np.int32)
    out["thr"] = out["thr"].astype(np.float64)
    out["leaf"] = np.concatenate(leaves).astype(np.float64)
    out["root"] = np.array(tree_off_node, dtype=np.int32)
    out["tree_class"] = np.array(tree_class, dtype=np.int32)
    out["n_class"] = n_class
    sig = re.search(r"sigmoid:([0-9.eE+-]+)", meta.get("objective", ""))
    out["sigmoid"] = float(sig.group(1)) if sig else 1.0
    out["depth"] = _max_depth(out)
    return out


def _max_depth(m):
    """Maximum number of internal nodes on any root-to-leaf path (BFS over all trees)."""
    frontier = m["root"].astype(np.int64)
    depth = 0
    while len(frontier):
        depth += 1
        kids = np.r_[m["left"][frontier], m["right"][frontier]]
        frontier = kids[kids >= 0].astype(np.int64)
    return depth


def raw_scores(m, X):
    """X [n, F] float -> raw class scores [n, n_class] (sum of leaf values per class tree)."""
    X = np.asarray(X, dtype=np.float64)
    n, T = len(X), len(m["root"])
    out = np.zeros((n, m["n_class"]))
    cur = np.broadcast_to(m["root"], (n, T)).copy()
    rows = np.arange(n)[:, None]
    done = np.zeros((n, T), dtype=bool)
    leaf_idx = np.zeros((n, T), dtype=np.int64)
    for _ in range(m["depth"] + 1):
        node = np.where(done, 0, cur)
        f = X[rows, m["feat"][node]]
        dt = m["dt"][node]
        mt = (dt >> 2) & 3
        isnan = np.isnan(f)
        f = np.where(isnan & (mt != 2), 0.0, f)
        missing = ((mt == 1) & (np.abs(f) <= 1e-35)) | ((mt == 2) & isnan)
        go_left = np.where(missing, (dt & 2) != 0, f <= m["thr"][node])
        nxt = np.where(go_left, m["left"][node], m["right"][node])
        newly = (~done) & (nxt < 0)
        leaf_idx = np.where(newly, ~nxt, leaf_idx)
        done = done | newly
        cur = np.where(done, cur, nxt)
        if done.all():
            break
    vals = m["leaf"][leaf_idx]
    for k in range(m["n_class"]):
        out[:, k] = vals[:, m["tree_class"] == k].sum(1)
    return out


def predict_proba(m, X):
    """Multiclass: softmax over the class scores. Binary objective (n_class == 1): one column, P(positive) = sigmoid(sigmoid_param * raw)."""
    s = raw_scores(m, X)
    if m["n_class"] == 1:
        return 1.0 / (1.0 + np.exp(-m.get("sigmoid", 1.0) * s))
    s = s - s.max(1, keepdims=True)
    e = np.exp(s)
    return e / e.sum(1, keepdims=True)


def compact_export(m):
    """Edge-compact arrays: feature uint16, threshold float32, children int16 (tree-local), leaf int8 + per-tree float32 scale."""
    n_tree = len(m["root"])
    leaf_i8 = np.zeros_like(m["leaf"], dtype=np.int8)
    scale = np.zeros(n_tree, dtype=np.float32)
    bounds = np.r_[m["root"], len(m["feat"])]
    # leaf ranges per tree: derive from leaf offsets in encoded children
    leaf_off = np.zeros(n_tree + 1, dtype=np.int64)
    # number of leaves per tree = number of internal nodes + 1 (stump: 1)
    internal = np.diff(bounds)
    nleaf = np.where(internal == 1, np.maximum(1, internal), internal + 1)
    # stumps were stored with one dummy node and one leaf; trees with k internal nodes have k+1 leaves
    stump = np.array([(m["left"][b] < 0 and m["left"][b] == m["right"][b]) for b in m["root"]])
    nleaf = np.where(stump, 1, internal + 1)
    leaf_off[1:] = np.cumsum(nleaf)
    for t in range(n_tree):
        lv = m["leaf"][leaf_off[t]:leaf_off[t + 1]]
        s = float(np.abs(lv).max()) / 127.0 or 1.0
        scale[t] = s
        leaf_i8[leaf_off[t]:leaf_off[t + 1]] = np.round(lv / s).astype(np.int8)
    return {"feat": m["feat"].astype(np.uint16), "thr": m["thr"].astype(np.float32), "dt": m["dt"].astype(np.uint8),
            "left": m["left"].astype(np.int32), "right": m["right"].astype(np.int32), "leaf_i8": leaf_i8, "leaf_scale": scale,
            "root": m["root"], "tree_class": m["tree_class"].astype(np.uint8), "leaf_off": leaf_off}


def from_compact(c, n_class):
    leaf = np.empty(len(c["leaf_i8"]), dtype=np.float64)
    for t in range(len(c["root"])):
        a, b = c["leaf_off"][t], c["leaf_off"][t + 1]
        leaf[a:b] = c["leaf_i8"][a:b].astype(np.float64) * float(c["leaf_scale"][t])
    m = {"feat": c["feat"].astype(np.int32), "thr": c["thr"].astype(np.float64), "dt": c["dt"].astype(np.int32), "left": c["left"], "right": c["right"],
         "leaf": leaf, "root": c["root"], "tree_class": c["tree_class"].astype(np.int32), "n_class": n_class}
    m["depth"] = _max_depth(m)
    return m
