"""Automated gate between the smoke kernel and the final kernel: the smoke run must have completed all phases for both models with sane numbers.
usage: python scripts/gate_smoke.py <smoke output dir>; exit 0 = pass. Reads only kernel outputs (small json/log)."""
import json
import math
import sys
from pathlib import Path

out = Path(sys.argv[1])
problems, notes = [], []
logs = sorted(out.glob("*.log"))
log = logs[0].read_text() if logs else ""
if not logs:
    problems.append("no kernel log downloaded")
for bad in ("Traceback", "AssertionError", "MemoryError"):
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
for tag in ("best", "best_binary"):
    t = r.get(tag)
    if not t:
        problems.append(f"{tag}: missing")
        continue
    for k in ("theta", "reports", "edge", "seeds", "dev_reproduction"):
        if k not in t:
            problems.append(f"{tag}: missing {k}")
    if t.get("theta") is None:
        notes.append(f"{tag}: theta is None (no feasible threshold on the small smoke dev set); not a code error")
    rep = t.get("reports", {})
    for split in ("dev_oof", "test"):
        pt = rep.get(split, {}).get("point", {})
        for k in (("macro_f1", "auroc", "average_precision") if tag == "best" else ("auroc", "average_precision")):
            if not fin(pt.get(k)):
                problems.append(f"{tag}: {split}.{k} not finite ({pt.get(k)})")
        if "ci95" not in rep.get(split, {}):
            problems.append(f"{tag}: {split} has no bootstrap interval")
    e = t.get("edge", {})
    if not (fin(e.get("verify_numpy_vs_lightgbm_max_abs_diff")) and e["verify_numpy_vs_lightgbm_max_abs_diff"] < 1e-4):
        problems.append(f"{tag}: numpy evaluator disagrees with LightGBM ({e.get('verify_numpy_vs_lightgbm_max_abs_diff')})")
    if not (fin(e.get("compact_int8_leaf_npz_bytes")) and e["compact_int8_leaf_npz_bytes"] > 0):
        problems.append(f"{tag}: no compact size")
    seeds = t.get("seeds", {})
    if sorted(seeds) != ["0", "1"]:
        problems.append(f"{tag}: seeds {sorted(seeds)} (smoke expects 0 and 1)")
    for sd, rec in seeds.items():
        if "test_point" not in rec and sd != "0":
            problems.append(f"{tag}: seed {sd} has no test_point")
    if not r.get(f"{tag}_compact", {}).get("reports", {}).get("test"):
        problems.append(f"{tag}_compact: missing test report")
    if not (out / f"preds_{tag}.npz").exists():
        problems.append(f"{tag}: preds file not downloaded")
for f in ("best_final.txt", "best_binary_final.txt"):
    if not (out / f).exists():
        problems.append(f"{f} not downloaded")
print("GATE notes:", notes)
print("GATE problems:", problems)
sys.exit(1 if problems else 0)
