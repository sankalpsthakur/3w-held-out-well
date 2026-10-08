"""Confusion matrix and per-class precision/recall/F1 of a saved 3W-HW class prediction on the sealed test (or dev OOF) wells, from preds_*.npz only (numpy).

usage: python scripts/conf_3w.py <out.md> <label> <results.json>:<tag>:<preds.npz> [--split=dev_oof]
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
out_path, label = Path(args[0]), args[1]
rj, tag, npz = args[2].split(":")
theta = json.loads(Path(rj).read_text())[tag]["theta"]
bench = S.Benchmark.load(ROOT / "protocol")
wells = bench.wells_with_role("test") if split == "test" else bench.wells_with_role("dev")
z = np.load(npz)
Sm, names, _ = S.well_statistics(bench, z["score"].astype(np.float64), z["pred"].astype(int), wells, theta)
a, b = S._layout()["conf"]
conf = Sm.sum(0)[a:b].reshape(S.N_CLASSES, S.N_CLASSES)           # rows = truth, columns = prediction
tp = np.diag(conf)
rows, cols = conf.sum(1), conf.sum(0)
with np.errstate(all="ignore"):
    rec = np.where(rows > 0, tp / rows, np.nan)
    prec = np.where(cols > 0, tp / cols, np.nan)
    f1 = np.where(2 * tp + (cols - tp) + (rows - tp) > 0, 2 * tp / (2 * tp + (cols - tp) + (rows - tp)), np.nan)
lines = [f"### {label}: confusion matrix on the {split} wells ({len(names)} wells; rows = truth, columns = prediction; eligible labelled decisions)", "",
         "| truth \\ pred | " + " | ".join(str(c) for c in range(S.N_CLASSES)) + " | support | recall | precision | F1 |", "|" + "---|" * (S.N_CLASSES + 5)]
for t in range(S.N_CLASSES):
    if rows[t] == 0:
        continue
    f = lambda v: "-" if not np.isfinite(v) else f"{v:.3f}"  # noqa: E731
    lines.append(f"| {t} | " + " | ".join(str(int(conf[t, c])) for c in range(S.N_CLASSES)) + f" | {int(rows[t])} | {f(rec[t])} | {f(prec[t])} | {f(f1[t])} |")
text = "\n".join(lines) + "\n"
out_path.write_text(text)
print(text)
