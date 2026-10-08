"""Edge numbers for the official SKAB Conv-AE: parameters, size on disk (fp32 / int8 / TFLite), latency at 1 thread, batch 1.

The architecture is SKAB's own `Conv_AE._build_model` (imported from the pinned checkout); weights are the untrained initialisation
(size and latency do not depend on weight values). One window = 60 steps x 8 sensors -> one forward pass.
"""
import json
import os
import sys
import time
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
import tensorflow as tf

tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "external" / "SKAB"))
from core.Conv_AE import Conv_AE  # noqa: E402

m = Conv_AE()
m.shape = (341, 60, 8)
model = m._build_model()
n_params = int(model.count_params())
per_layer = {l.name: int(l.count_params()) for l in model.layers if l.count_params()}
out = {"model": "SKAB Conv_AE (core/Conv_AE.py)", "input": [60, 8], "parameters": n_params, "per_layer": per_layer,
       "size_fp32_bytes_weights_only": n_params * 4}
tmp = ROOT / "tmp"
tmp.mkdir(exist_ok=True)
model.save_weights(str(tmp / "convae.weights.h5"))
out["size_keras_weights_h5_bytes"] = os.path.getsize(tmp / "convae.weights.h5")
# int8 per-tensor symmetric quantisation of the weights (storage size only)
q = 0
for w in model.weights:
    a = w.numpy()
    q += a.size + 4
out["size_int8_weights_bytes_est"] = q
x = np.random.default_rng(0).normal(size=(1, 60, 8)).astype(np.float32)

def bench(fn, n=2000, warm=200):
    for _ in range(warm):
        fn()
    t = []
    for _ in range(n):
        t0 = time.perf_counter_ns(); fn(); t.append(time.perf_counter_ns() - t0)
    t = np.array(t) / 1e3
    return {"median_us": float(np.median(t)), "p95_us": float(np.percentile(t, 95)), "n": n}

xt = tf.constant(x)
fwd = tf.function(lambda a: model(a, training=False))
out["latency_keras_tf_function_1thread"] = bench(lambda: fwd(xt).numpy())
for name, opts in (("tflite_fp32", None), ("tflite_dynamic_int8", [tf.lite.Optimize.DEFAULT])):
    try:
        conv = tf.lite.TFLiteConverter.from_keras_model(model)
        if opts:
            conv.optimizations = opts
        blob = conv.convert()
        path = tmp / f"convae_{name}.tflite"
        path.write_bytes(blob)
        it = tf.lite.Interpreter(model_content=blob, num_threads=1)
        it.allocate_tensors()
        inp, outp = it.get_input_details()[0]["index"], it.get_output_details()[0]["index"]
        def run():
            it.set_tensor(inp, x); it.invoke(); return it.get_tensor(outp)
        out[f"size_{name}_bytes"] = len(blob)
        out[f"latency_{name}_1thread"] = bench(run)
    except Exception as e:  # keep going: report what could be measured
        out[f"{name}_error"] = repr(e)[:300]
(ROOT / "results" / "edge" / "skab_convae_edge.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
