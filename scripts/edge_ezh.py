"""Edge numbers for EZH: parameters, size on disk, latency of one streaming update at 1 thread (numpy/Python reference implementation)."""
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
from skab import data as D  # noqa: E402
from skab import ezh  # noqa: E402

df = D.load_file("valve1/1.csv")
X = df[["Accelerometer1RMS", "Accelerometer2RMS", "Current", "Pressure", "Temperature", "Thermocouple", "Voltage", "Volume Flow RateRMS"]].values.astype(float)
det = ezh.EZH().fit(X[:400])
p = det.p
stored = {"mu": p["mu"].astype(np.float32), "sd": p["sd"].astype(np.float32), "ref": np.float32(p["ref"]), "vref": p["vref"].astype(np.float32)}
used = {"mu": p["mu"][list(ezh.STABLE)], "sd": p["sd"][list(ezh.STABLE)], "ref": p["ref"], "vref": p["vref"][list(ezh.THERMAL)]}
n_stored = int(sum(np.size(v) for v in stored.values()))
n_used = int(sum(np.size(v) for v in used.values()))
const = {"alpha": ezh.ALPHA, "on": ezh.ON, "off": ezh.OFF, "hold": ezh.HOLD, "wstd": ezh.WSTD, "vratio": ezh.VRATIO}
lat = []
rows = X[400:]
for rep in range(5):
    d = ezh.EZH().fit(X[:400])
    t0 = time.perf_counter_ns()
    for x in rows:
        d.update(x)
    lat.append((time.perf_counter_ns() - t0) / len(rows) / 1e3)
out = {"detector": "EZH (EWMA-z hysteresis, hold)", "parameters_fitted_used": n_used, "parameters_fitted_stored_in_reference_impl": n_stored, "constants": const,
       "size_fp32_bytes_used_params": n_used * 4, "size_fp16_bytes_used_params": n_used * 2, "size_int8_bytes_with_scales_estimate": n_used + 8,
       "buffer_floats_streaming_state": int(8 + ezh.WSTD * 8 + 3),
       "latency_per_update_us_python_numpy_1thread": {"median_of_5_runs": round(float(np.median(lat)), 1), "runs": [round(v, 1) for v in lat]},
       "fit_cost": "mean/std over 400 rows, no training loop (about 0.4 ms)"}
t0 = time.perf_counter_ns(); [ezh.fit_params(X[:400]) for _ in range(50)]; out["fit_ms"] = round((time.perf_counter_ns() - t0) / 50 / 1e6, 2)
(ROOT / "results" / "edge" / "skab_ezh_edge.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
