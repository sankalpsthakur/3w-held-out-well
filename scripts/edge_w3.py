"""Edge numbers for a LightGBM 3W-HW model without lightgbm installed: verification against kernel probe predictions, int8-leaf export,
sizes and latency at 1 thread, batch 1 (features recomputed from a 3,601-row buffer + numpy tree evaluator).

usage: python scripts/edge_w3.py <name> <model.txt> <probe_X.npy> <probe_proba.npy> <probe_classes.npy>     (verification + sizes + latency; batch-2000 probe, used for the baselines)
       python scripts/edge_w3.py latency <name> <model.txt> <probe_X.npy> [kernel_results.json tag]            (latency and compact size only: batch 1, 1 thread; run it through bin/guard.sh and only when the 1-minute load average is under 4;
                                                                                                          the verification numbers come from the Kaggle kernel and are merged from its results.json)
"""
import io
import json
import os
import sys
import time
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from w3 import features as F  # noqa: E402
from w3 import gbdt_numpy as G  # noqa: E402

LATENCY_ONLY = sys.argv[1] == "latency"
if LATENCY_ONLY:
    name, model_path, px = sys.argv[2:5]
    kernel_json = sys.argv[5:7]
    pp = pc = None
else:
    name, model_path, px, pp, pc = sys.argv[1:6]
m = G.parse_model(model_path)
n_class = m["n_class"]
out = {"name": name, "n_trees": int(len(m["root"])), "n_internal_nodes": int(len(m["feat"])), "n_leaves": int(len(m["leaf"])), "depth": int(m["depth"]),
       "model_text_bytes": os.path.getsize(model_path)}
c = G.compact_export(m)
raw_bytes = int(sum(v.nbytes for v in c.values()))
buf = io.BytesIO(); np.savez_compressed(buf, **c)
out["compact_int8_leaf_raw_bytes"] = raw_bytes
out["compact_int8_leaf_npz_bytes"] = len(buf.getvalue())
m2 = G.from_compact(c, n_class)
if LATENCY_ONLY:
    Xp = np.load(px, mmap_mode="r")[:1].astype(np.float64)               # one decision only: batch 1
    if kernel_json:
        kj = json.loads(Path(kernel_json[0]).read_text())[kernel_json[1]]["edge"]
        out["kernel_verification"] = {k: v for k, v in kj.items() if k not in ("top_gain_features",)}
        out["top_gain_features"] = kj.get("top_gain_features", [])[:10]
else:
    Xp = np.load(px).astype(np.float64)
    want = np.load(pp).astype(np.float64)
    classes = np.load(pc)
    proba = G.predict_proba(m, Xp)
    if proba.shape[1] != want.shape[1]:        # binary model: want has one column (P(event))
        proba = proba[:, [1]] if proba.shape[1] == 2 else proba
    out["verify_vs_kernel_max_abs_diff"] = float(np.abs(proba - want).max()) if proba.shape == want.shape else None
    out["verify_argmax_agreement"] = float((proba.argmax(1) == want.argmax(1)).mean()) if proba.shape == want.shape else None
    p2 = G.predict_proba(m2, Xp)
    out["compact_vs_fp_max_abs_diff_proba"] = float(np.abs(p2 - proba).max())
    out["compact_vs_fp_argmax_agreement"] = float((p2.argmax(1) == proba.argmax(1)).mean())
# latency: model only, batch 1
def bench(fn, n=300, warm=30):
    for _ in range(warm):
        fn()
    t = []
    for _ in range(n):
        t0 = time.perf_counter_ns(); fn(); t.append(time.perf_counter_ns() - t0)
    t = np.array(t) / 1e3
    return {"median_us": round(float(np.median(t)), 1), "p95_us": round(float(np.percentile(t, 95)), 1), "n": n}
x1 = Xp[:1]
out["latency_model_numpy_fp_1thread_batch1"] = bench(lambda: G.raw_scores(m, x1))
out["latency_model_numpy_int8leaf_1thread_batch1"] = bench(lambda: G.raw_scores(m2, x1))
# latency: features for one decision from a 3,601-row buffer
rng = np.random.default_rng(0)
cont = (1e6 + rng.normal(size=(3601, 18)) * 1e3).astype(np.float32); cont[:, ::3] = np.nan
state = np.zeros((3601, 9), dtype=np.float32)
out["latency_features_one_decision_from_3601_row_buffer"] = bench(lambda: F.instance_features(cont, state), n=100, warm=10)
out["streaming_buffer_bytes_fp32_27ch_x_3600"] = 27 * 3600 * 4
f_us = out["latency_features_one_decision_from_3601_row_buffer"]["median_us"]; m_us = out["latency_model_numpy_fp_1thread_batch1"]["median_us"]
out["end_to_end_median_us_numpy_reference"] = round(f_us + m_us, 1)
out["note"] = "numpy/Python reference implementations at 1 thread on the M4; features recomputed from the buffer each decision (decisions are 30 s apart); an embedded C implementation would be far faster"
(ROOT / "results" / "edge" / f"w3_{name}_edge.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
