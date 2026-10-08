"""Re-score saved 3W-HW predictions locally (numpy only) and compare with the kernel's results.json.

usage: python scripts/rescore_3w.py <results.json> <preds dir> [model ...]
Predictions npz: score (float32 [N]), pred (int8 [N], optional). Uses protocol/ files only; no sensor data needed.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from w3 import pipeline as P  # noqa: E402
from w3 import scorer as S  # noqa: E402

res = json.loads(Path(sys.argv[1]).read_text())
pdir = Path(sys.argv[2])
models = sys.argv[3:] or [m for m in res if (pdir / f"preds_{m}.npz").exists()]
b = S.Benchmark.load(ROOT / "protocol")
dev, test = b.wells_with_role("dev"), b.wells_with_role("test")
ok_all = True
for m in models:
    z = np.load(pdir / f"preds_{m}.npz")
    score = z["score"].astype(np.float64)
    pred = z["pred"].astype(int) if "pred" in z.files else None
    rm = res[m] if m in res else res          # single-model results files (TSPulse) keep theta and reports at the top level
    theta = rm["theta"]
    rep = {"test": S.report(b, score, pred, test, theta, boot=False), "dev_oof": S.report(b, score, pred, dev, theta, classes=P.dev_classes(b), boot=False)}
    for split in ("test", "dev_oof"):
        mine, kern = rep[split]["point"], rm["reports"][split]["point"]
        diffs = {k: abs(mine[k] - kern[k]) for k in ("macro_f1", "auroc", "average_precision", "episode_coverage", "false_alarms_per_normal_day", "normal_time_occupation", "onset_detection_300s")
                 if mine.get(k) is not None and kern.get(k) is not None and np.isfinite(mine[k]) and np.isfinite(kern[k])}
        worst = max(diffs.values()) if diffs else 0.0
        ok = worst < 1e-3
        ok_all &= ok
        print(f"{m:12s} {split:8s} max |local - kernel| over {len(diffs)} metrics = {worst:.2e}  {'OK' if ok else 'MISMATCH'}   macro-F1 {mine['macro_f1'] if mine.get('macro_f1') is not None else float('nan'):.3f}  AUROC {mine['auroc']:.3f}")
sys.exit(0 if ok_all else 1)
