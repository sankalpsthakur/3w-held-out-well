"""Edge numbers for SKAB's Conv-AE architecture without TensorFlow: same layer shapes in PyTorch, weights untrained (size/latency do not depend on values).

Shapes follow core/Conv_AE.py (Keras): Conv1D(32,k7,s2,same) Dropout Conv1D(16,k7,s2,same) Conv1DTranspose(16,k7,s2,same) Dropout
Conv1DTranspose(32,k7,s2,same) Conv1DTranspose(1,k7,same); input 60 steps x 8 sensors. Reports parameters, size on disk (fp32/fp16/int8 container) and
latency at 1 thread, batch 1: torch eager, TorchScript and (if the exporter works) ONNX Runtime.
"""
import io
import json
import os
import time
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
ROOT = Path(__file__).resolve().parents[1]
m = nn.Sequential(
    nn.Conv1d(8, 32, 7, stride=2, padding=3), nn.ReLU(), nn.Dropout(0.2),
    nn.Conv1d(32, 16, 7, stride=2, padding=3), nn.ReLU(),
    nn.ConvTranspose1d(16, 16, 7, stride=2, padding=3, output_padding=1), nn.ReLU(), nn.Dropout(0.2),
    nn.ConvTranspose1d(16, 32, 7, stride=2, padding=3, output_padding=1), nn.ReLU(),
    nn.ConvTranspose1d(32, 1, 7, stride=1, padding=3)).eval()
x = torch.randn(1, 8, 60)
assert m(x).shape == (1, 1, 60), m(x).shape
n_params = sum(p.numel() for p in m.parameters())
sd = {k: v.detach().numpy() for k, v in m.state_dict().items()}

def npz_size(arrs):
    b = io.BytesIO(); np.savez(b, **arrs); return len(b.getvalue())

out = {"architecture": "SKAB Conv_AE (shapes from core/Conv_AE.py), PyTorch re-implementation with untrained weights", "input": [60, 8],
       "parameters": int(n_params), "expected_keras_parameters": 11073,
       "size_fp32_raw_bytes": n_params * 4, "size_fp16_raw_bytes": n_params * 2, "size_int8_raw_bytes_with_scales": n_params + 4 * len(sd),
       "size_fp32_npz_bytes": npz_size(sd), "size_fp16_npz_bytes": npz_size({k: v.astype(np.float16) for k, v in sd.items()}),
       "size_int8_npz_bytes": npz_size({**{k: np.round(v / (np.abs(v).max() / 127 + 1e-12)).astype(np.int8) for k, v in sd.items()},
                                        **{k + "_scale": np.array([np.abs(v).max() / 127], dtype=np.float32) for k, v in sd.items()}})}

def bench(fn, n=3000, warm=300):
    for _ in range(warm):
        fn()
    t = []
    for _ in range(n):
        t0 = time.perf_counter_ns(); fn(); t.append(time.perf_counter_ns() - t0)
    t = np.array(t) / 1e3
    return {"median_us": round(float(np.median(t)), 1), "p95_us": round(float(np.percentile(t, 95)), 1), "n": n}

with torch.inference_mode():
    out["latency_torch_eager_fp32_1thread"] = bench(lambda: m(x))
    ts = torch.jit.trace(m, x); ts = torch.jit.freeze(ts)
    out["latency_torchscript_fp32_1thread"] = bench(lambda: ts(x))
try:
    import onnxruntime as ort
    p = ROOT / "tmp"; p.mkdir(exist_ok=True)
    torch.onnx.export(m, x, str(p / "convae.onnx"), input_names=["x"], output_names=["y"], opset_version=17, dynamo=False)
    so = ort.SessionOptions(); so.intra_op_num_threads = 1; so.inter_op_num_threads = 1
    sess = ort.InferenceSession(str(p / "convae.onnx"), so, providers=["CPUExecutionProvider"])
    xn = x.numpy()
    out["size_onnx_fp32_bytes"] = os.path.getsize(p / "convae.onnx")
    out["latency_onnxruntime_fp32_1thread"] = bench(lambda: sess.run(None, {"x": xn}))
    ref = m(x).detach().numpy(); got = sess.run(None, {"x": xn})[0]
    out["onnx_max_abs_diff_vs_torch"] = float(np.abs(ref - got).max())
except Exception as e:  # report what was measured
    out["onnx_error"] = repr(e)[:300]
(ROOT / "results" / "edge").mkdir(parents=True, exist_ok=True)
(ROOT / "results" / "edge" / "skab_convae_edge.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
