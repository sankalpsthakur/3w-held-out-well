"""Automated gate between the TSPulse smoke kernel and the full kernel: the smoke run must have completed both phases with sane numbers and the projected full runtime must fit.
usage: python scripts/gate_tspulse.py <smoke output dir>; exit 0 = pass."""
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
out = Path(sys.argv[1])
problems, notes = [], []
logs = sorted(out.glob("*.log"))
log = logs[0].read_text() if logs else ""
if not logs:
    problems.append("no kernel log downloaded")
for bad in ("Traceback", "AssertionError", "MemoryError", "could not be installed"):
    if bad in log:
        problems.append(f"'{bad}' in the kernel log")
if "all done" not in log:
    problems.append("log has no 'all done' line")
try:
    r = json.loads((out / "results.json").read_text())
except Exception as e:  # noqa: BLE001
    print("GATE: cannot read results.json:", e)
    sys.exit(1)


def fin(x):
    return isinstance(x, (int, float)) and math.isfinite(x)


if r.get("smoke") is not True:
    problems.append("results.json is not from a smoke run")
if r.get("pip_rc") != 0:
    problems.append("pip install of granite-tsfm failed")
if not r.get("versions", {}).get("n_params"):
    problems.append("model not loaded")
if "model_revision_error" in r:
    problems.append("the pinned model revision could not be loaded: " + r["model_revision_error"])
for split in ("dev_oof", "test"):
    rep = r.get("reports", {}).get(split, {})
    pt = rep.get("point", {})
    for k in ("auroc", "average_precision"):
        if not fin(pt.get(k)):
            problems.append(f"{split}.{k} not finite ({pt.get(k)})")
    if "ci95" not in rep:
        problems.append(f"{split}: no bootstrap interval")
if r.get("theta") is None:
    notes.append("theta is None on the small smoke dev set (not a code error)")
if len(r.get("dev_variants_information_only", {})) != 4:
    problems.append("dev variants missing")
q = r.get("q99_dev_normal", {})
if sum(v is not None for v in q.values()) < 8:
    problems.append(f"fewer than 8 channels have a dev-normal reference ({sum(v is not None for v in q.values())})")
if not (out / "preds_tspulse.npz").exists():
    problems.append("preds file not downloaded")
# projected cost of the full run
c = r.get("all_error_counts", {})
if c.get("windows") and c.get("calls") and c.get("model_seconds"):
    cfg_full = json.loads((ROOT / "protocol" / "tspulse_config.json").read_text())
    calls_per_window = c["calls"] / c["windows"]
    windows_full = 962002 / cfg_full["stride_dec"]
    proj = calls_per_window * windows_full * (c["model_seconds"] / c["calls"]) * 1.8      # x1.8 for loading and window building
    notes.append(f"projected full run {proj / 60:.0f} min ({calls_per_window:.1f} channels per window, {windows_full:.0f} windows, {1000 * c['model_seconds'] / c['calls']:.2f} ms per call)")
    if proj > 6000:
        problems.append(f"projected full run {proj / 60:.0f} min exceeds the 100 min budget")
else:
    problems.append("no timing counters in the smoke results")
print("GATE notes:", notes)
print("GATE problems:", problems)
sys.exit(1 if problems else 0)
