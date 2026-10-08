"""Render kernel results.json (baselines / final model) into markdown tables with well-bootstrap intervals."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
src = Path(sys.argv[1])
out = Path(sys.argv[2])
R = json.loads(src.read_text())
MODELS = [m for m in ("simple", "volatility", "rf", "lgbm", "best", "best_compact", "best_binary", "best_binary_compact", "tspulse") if m in R]
SPLITS = [("test", "Sealed test (10 wells)"), ("dev_oof", "Dev out-of-fold (30 wells; optimistic for tuned models)"), ("cv6_all_wells", "6-fold pooled, all 40 wells (valid for untuned models only; not computed for the dev-selected models)")]


def fmt(rep, key, pct=False, digits=3):
    p = rep["point"].get(key)
    if p is None:
        return "-"
    ci = rep.get("ci95", {}).get(key)
    sc = 100.0 if pct else 1.0
    s = f"{p * sc:.{digits}f}" if not pct else f"{p * sc:.1f}%"
    if ci:
        lo, hi = ci[0] * sc, ci[1] * sc
        s += f" [{lo:.{digits}f}, {hi:.{digits}f}]" if not pct else f" [{lo:.1f}, {hi:.1f}]"
    return s


lines = []
for split, title in SPLITS:
    lines += [f"### {title}", "", "| model | macro-F1 | bal. acc | AUROC | AP | episode coverage | onset <=300 s | FA / normal day | normal occupation | theta |", "|---|---|---|---|---|---|---|---|---|---|"]
    for m in MODELS:
        rep = R[m]["reports"].get(split)
        if rep is None:
            continue
        th = R[m].get("theta")
        lines.append(f"| {m} | {fmt(rep, 'macro_f1')} | {fmt(rep, 'balanced_accuracy')} | {fmt(rep, 'auroc')} | {fmt(rep, 'average_precision')} | {fmt(rep, 'episode_coverage', True)} | "
                     f"{fmt(rep, 'onset_detection_300s', True)} | {fmt(rep, 'false_alarms_per_normal_day', digits=2)} | {fmt(rep, 'normal_time_occupation', True)} | {'-' if th is None else f'{th:.3f}'} |")
    lines.append("")
    first = next((R[m]["reports"][split] for m in MODELS if split in R[m]["reports"]), None)
    if first:
        pt = first["point"]
        lines.append(f"Denominators: {first['n_wells']} wells, {pt['event_decisions']:.0f} event and {pt['normal_decisions']:.0f} normal eligible decisions, "
                     f"{pt['normal_days']:.1f} normal days, {pt['episodes_total']:.0f} episodes ({pt['episodes_unobservable']:.0f} unobservable).")
        lines.append("")
# per-seed table for the dev-selected models (seed 0 is the preregistered primary result)
for m in ("best", "best_binary"):
    seeds = R.get(m, {}).get("seeds")
    if not seeds:
        continue
    lines += [f"### {m}: seeds (sealed test, point estimates; seed 0 is primary)", "", "| seed | dev-OOF macro-F1 | dev-OOF AP | theta | test macro-F1 | test AUROC | test AP | episode coverage | FA / normal day |", "|---|---|---|---|---|---|---|---|---|"]
    f = lambda v, d=3: "-" if v is None else f"{v:.{d}f}"
    for sd, rec in sorted(seeds.items(), key=lambda kv: int(kv[0])):
        tp = rec.get("test_point") or R[m]["reports"]["test"]["point"]
        lines.append(f"| {sd} | {f(rec['dev_oof_point'].get('macro_f1'))} | {f(rec['dev_oof_point'].get('average_precision'))} | {f(rec.get('theta'))} | {f(tp.get('macro_f1'))} | {f(tp.get('auroc'))} | {f(tp.get('average_precision'))} | "
                     f"{f(tp.get('episode_coverage'))} | {f(tp.get('false_alarms_per_normal_day'), 2)} |")
    lines.append("")
out.write_text("\n".join(lines))
print("\n".join(lines))
