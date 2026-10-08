"""Per-well event AUROC (and counts) of saved 3W-HW predictions on the sealed test wells: how concentrated is a result in a few wells? numpy only.

usage: python scripts/perwell_3w.py <out.md> <label>=<results.json>:<tag>:<preds.npz> [...]   [--split=dev_oof]
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from w3 import scorer as S  # noqa: E402

split = "dev_oof" if "--split=dev_oof" in sys.argv else "test"
args = [a for a in sys.argv[1:] if not a.startswith("--")]
out_path = Path(args[0])
bench = S.Benchmark.load(ROOT / "protocol")
wells = bench.wells_with_role("test") if split == "test" else bench.wells_with_role("dev")
cols, names = {}, None
for a in args[1:]:
    label, rest = a.split("=", 1)
    rj, tag, npz = rest.split(":")
    res = json.loads(Path(rj).read_text())
    theta = (res if tag == "." else res[tag])["theta"]
    z = np.load(npz)
    pred = z["pred"].astype(int) if "pred" in z.files else None
    Sm, names, _ = S.well_statistics(bench, z["score"].astype(np.float64), pred, wells, theta)
    m = S.metrics_from_sums(Sm)
    cols[label] = (m["auroc"], m["episode_coverage"], m["false_alarms_per_normal_day"])
    ev, nm = m["event_decisions"], m["normal_decisions"]
hdr = "| well | event decisions | normal decisions | " + " | ".join(f"{k} AUROC" for k in cols) + " |"
lines = [f"### Per-well event AUROC on the {split} wells (a well with no events or no normal decisions shows -)", "", hdr, "|" + "---|" * (3 + len(cols))]
f = lambda v: "-" if not np.isfinite(v) else f"{v:.3f}"  # noqa: E731
for i, w in enumerate(names):
    lines.append(f"| {w} | {int(ev[i])} | {int(nm[i])} | " + " | ".join(f(cols[k][0][i]) for k in cols) + " |")
text = "\n".join(lines) + "\n"
out_path.write_text(text)
print(text)
