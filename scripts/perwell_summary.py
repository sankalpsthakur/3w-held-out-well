"""Summarise the per-well AUROC tables (results/3w/perwell_auroc_dev.md and perwell_auroc_test_all.md): mean, median and share above 0.5 over the wells that have both classes.
usage: python scripts/perwell_summary.py results/3w/perwell_auroc_summary.md"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load(path):
    lines = (ROOT / path).read_text().splitlines()
    hdr = [h.strip() for h in next(l for l in lines if l.startswith("| well")).strip("| ").split("|")]
    rows = [l.strip("| ").split("|") for l in lines if l.startswith("| WELL")]
    names = [h.replace(" AUROC", "") for h in hdr[3:]]
    return names, np.array([[np.nan if c.strip() == "-" else float(c) for c in r[3:]] for r in rows])


nd, Md = load("results/3w/perwell_auroc_dev.md")
nt, Mt = load("results/3w/perwell_auroc_test_all.md")
od, ot = np.isfinite(Md[:, 0]), np.isfinite(Mt[:, 0])
lines = ["### Within-well event AUROC (mean over wells that have both classes; median and number of wells above 0.5 in brackets)", "",
         f"| detector | development out-of-fold ({int(od.sum())} wells) | sealed test ({int(ot.sum())} wells) |", "|---|---|---|"]
for n in nt:
    jd, jt = nd.index(n), nt.index(n)
    d, t = Md[od, jd], Mt[ot, jt]
    lines.append(f"| {n} | {d.mean():.2f} (median {np.median(d):.2f}, {int((d > 0.5).sum())}/{len(d)} above 0.5) | {t.mean():.2f} (median {np.median(t):.2f}, {int((t > 0.5).sum())}/{len(t)} above 0.5) |")
text = "\n".join(lines) + "\n"
Path(sys.argv[1]).write_text(text)
print(text)
