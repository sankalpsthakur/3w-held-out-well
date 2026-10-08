"""Paired well-cluster bootstrap between two saved 3W-HW predictions on the sealed test wells (numpy only, chunked to stay under the guard's 512 MiB).

usage: python scripts/paired_3w.py <out.json> <A> <B> [--split test|dev_oof] [--check]
  A, B = <label>=<results.json>:<model tag>:<preds.npz>   e.g. best=kaggle_out/final/results.json:best:kaggle_out/final/preds_best.npz
The thresholds come from the results.json entries. Resamples are the same as scorer.bootstrap (seed 611, B 2000), so every interval of a model
alone and every paired interval use the same resampled well sets. Differences are A minus B. --check also recomputes A's own intervals and prints them
next to those stored in results.json (validates this chunked implementation).
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from w3 import pipeline as P  # noqa: E402
from w3 import scorer as S  # noqa: E402

KEYS = ["macro_f1", "balanced_accuracy", "auroc", "average_precision", "episode_coverage", "onset_detection_300s", "false_alarms_per_normal_day", "normal_time_occupation"]


def parse(arg):
    label, rest = arg.split("=", 1)
    rj, tag, npz = rest.split(":")
    res = json.loads(Path(rj).read_text())
    return label, (res if tag == "." else res[tag]), Path(npz), tag      # tag "." = the results.json top level (single-model kernels)


def well_stats(bench, theta, npz, wells, classes_for_macro):
    z = np.load(npz)
    score = z["score"].astype(np.float64)
    pred = z["pred"].astype(int) if "pred" in z.files else None
    Sm, names, _ = S.well_statistics(bench, score, pred, wells, theta)
    return Sm, names, pred is not None


def chunked(counts, SA, SB, classes, chunk=200):
    acc = {k: [] for k in KEYS}
    for a in range(0, len(counts), chunk):
        c = counts[a:a + chunk]
        mA = S.metrics_from_sums(c @ SA, classes)
        mB = S.metrics_from_sums(c @ SB, classes) if SB is not None else None
        for k in KEYS:
            acc[k].append(mA[k] - (mB[k] if mB is not None else 0.0))
    out = {}
    for k in KEYS:
        v = np.concatenate(acc[k])
        if np.isfinite(v).any():
            out[k] = [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5)), int(np.isfinite(v).sum())]
    return out


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    split = "dev_oof" if "--split=dev_oof" in sys.argv else "test"
    check = "--check" in sys.argv
    out_path = Path(args[0])
    A, B = parse(args[1]), parse(args[2])
    bench = S.Benchmark.load(ROOT / "protocol")
    wells = bench.wells_with_role("test") if split == "test" else bench.wells_with_role("dev")
    classes = None if split == "test" else P.dev_classes(bench)
    SA, names, hasA = well_stats(bench, A[1]["theta"], A[2], wells, classes)
    SB, _, hasB = well_stats(bench, B[1]["theta"], B[2], wells, classes)
    W = SA.shape[0]
    rng = np.random.default_rng(S.BOOT_SEED)
    counts = np.stack([np.bincount(rng.integers(0, W, W), minlength=W) for _ in range(S.BOOT_B)]).astype(np.float64)
    mA, mB = S.metrics_from_sums(SA.sum(0), classes), S.metrics_from_sums(SB.sum(0), classes)
    point = {k: [float(np.asarray(mA[k])), float(np.asarray(mB[k])), float(np.asarray(mA[k]) - np.asarray(mB[k]))] for k in KEYS}
    ci = chunked(counts, SA, SB, classes)
    res = {"split": split, "n_wells": W, "wells": names, "A": A[0], "B": B[0], "theta": {A[0]: A[1]["theta"], B[0]: B[1]["theta"]}, "bootstrap": {"B": S.BOOT_B, "seed": S.BOOT_SEED},
           "point_A_B_diff": point, "diff_ci95": ci}
    if check:
        own = chunked(counts, SA, None, classes)
        stored = A[1]["reports"][split]["ci95"]
        res["check_A_own_ci_vs_stored"] = {k: {"recomputed": own.get(k), "stored": stored.get(k)} for k in KEYS if k in stored}
    out_path.write_text(json.dumps(res, indent=1))
    print(f"{A[0]} minus {B[0]} on {W} {split} wells (same resamples as the scorer, B={S.BOOT_B})")
    for k in KEYS:
        if k in ci and np.isfinite(point[k][2]):
            lo, hi = ci[k][:2]
            sig = "excludes 0" if (lo > 0 or hi < 0) else "contains 0"
            print(f"  {k:30s} A {point[k][0]:8.4f}  B {point[k][1]:8.4f}  diff {point[k][2]:+8.4f} [{lo:+.4f}, {hi:+.4f}] {sig}")
    if check:
        for k, v in res["check_A_own_ci_vs_stored"].items():
            print(f"  check {k:28s} recomputed {v['recomputed'][:2] if v['recomputed'] else None} stored {v['stored'][:2] if v['stored'] else None}")


main()
