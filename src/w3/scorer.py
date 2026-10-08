"""3W-HW scorer. numpy only; definitions in protocol/PROTOCOL.md.

Inputs
  Benchmark   decision table (inst, r, y, elig, state) + instance/well tables + label runs (for episodes)
  score       float array over ALL decisions (NaN outside the evaluated wells), values in [0, 1]: P(event)
  pred        int array over ALL decisions (-1 where the model gives no class), classes 0..9
Everything that depends on the evaluated wells is reduced to per-well sufficient statistics, so cluster
bootstrap over wells is one matrix product.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

try:
    from .labels import MIN_SUPPORTED_EPISODE_ROWS, episodes_from_runs, event_class
except ImportError:  # pragma: no cover
    from labels import MIN_SUPPORTED_EPISODE_ROWS, episodes_from_runs, event_class  # type: ignore

# ----- frozen protocol parameters (PROTOCOL.md, "Frozen parameters") -----
STRIDE = 30
FIRST_ROW = 3600
LOOKBACK = 3600
ELIG_WINDOW = 900
ELIG_FRAC = 0.9
ONSET_TOL = 300
ON_K = 2
OFF_K = 2
BOOT_B = 2000
BOOT_SEED = 611
N_CLASSES = 10
NBINS = 4096
LOGIT_CLIP = 12.0
PROCESS_CHANNELS = ["P-ANULAR", "P-JUS-BS", "P-JUS-CKGL", "P-JUS-CKP", "P-MON-CKGL", "P-MON-CKP", "P-MON-SDV-P", "P-PDG", "PT-P",
                    "P-TPT", "QBS", "QGL", "T-JUS-CKP", "T-MON-CKP", "T-PDG", "T-TPT", "ABER-CKGL", "ABER-CKP"]


def n_decisions(rows: int) -> int:
    return max(0, -(-(rows - FIRST_ROW) // STRIDE))


def decision_rows(rows: int) -> np.ndarray:
    return FIRST_ROW + STRIDE * np.arange(n_decisions(rows), dtype=np.int64)


def eligibility_from_any_process(any_proc: np.ndarray) -> np.ndarray:
    """any_proc[row] = at least one finite process channel. Returns eligibility at each decision row."""
    rows = len(any_proc)
    r = decision_rows(rows)
    cs = np.r_[0, np.cumsum(any_proc.astype(np.int64))]
    return (cs[r + 1] - cs[r + 1 - ELIG_WINDOW]) >= ELIG_FRAC * ELIG_WINDOW


# ------------------------------------------------------------------ benchmark container
class Benchmark:
    def __init__(self, inst, r, y, elig, state, inst_rows, inst_well, inst_path, wells, class_runs, manifest):
        self.inst = np.asarray(inst, dtype=np.int32)
        self.r = np.asarray(r, dtype=np.int32)
        self.y = np.asarray(y, dtype=np.int8)
        self.elig = np.asarray(elig, dtype=bool)
        self.state = np.asarray(state, dtype=np.int8)
        self.n = len(self.inst)
        self.inst_rows = np.asarray(inst_rows, dtype=np.int64)
        self.inst_well = np.asarray(inst_well, dtype=np.int32)
        self.inst_path = list(inst_path)
        self.wells = list(wells)
        self.manifest = manifest  # well -> (role, fold)
        n_inst = len(inst_rows)
        counts = np.bincount(self.inst, minlength=n_inst)
        self.inst_offset = np.r_[0, np.cumsum(counts)]
        self.well_of_decision = self.inst_well[self.inst]
        assert np.all(np.diff(self.inst) >= 0), "decisions must be sorted by instance"
        # episodes
        e = []
        for i, runs in enumerate(class_runs):
            for ep in episodes_from_runs(runs):
                if ep["end"] - ep["start"] >= MIN_SUPPORTED_EPISODE_ROWS:
                    e.append((i, ep["start"], ep["end"], ep["event"], ep["known_onset"]))
        e = np.array(e, dtype=np.int64).reshape(-1, 5)
        self.ep_inst, self.ep_start, self.ep_end, self.ep_event, self.ep_known = (e[:, k] for k in range(5))
        self.ep_well = self.inst_well[self.ep_inst]

    # ---- construction from the committed protocol files
    @classmethod
    def load(cls, protocol_dir):
        p = Path(protocol_dir)
        d = np.load(p / "decisions.npz")
        with open(p / "instances.csv") as f:
            inst = list(csv.DictReader(f))
        wells = sorted({x["well"] for x in inst})
        widx = {w: i for i, w in enumerate(wells)}
        index = json.loads((p / "index_labels.json").read_text())
        runs_by_path = {x["path"]: x["class_runs"] for x in index["files"]}
        man = load_manifest(p / "manifest.csv")
        return cls(d["inst"], d["r"], d["y"], d["elig"], d["state"], [int(x["rows"]) for x in inst], [widx[x["well"]] for x in inst],
                   [x["path"] for x in inst], wells, [runs_by_path[x["path"]] for x in inst], man)

    def wells_with_role(self, role, fold=None):
        out = [w for w in self.wells if self.manifest[w][0] == role and (fold is None or self.manifest[w][1] == fold)]
        return out

    def well_mask(self, wells):
        idx = [self.wells.index(w) for w in wells]
        m = np.zeros(len(self.wells), dtype=bool)
        m[idx] = True
        return m


def load_manifest(path):
    out = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            out[row["well_id"]] = (row["role"], int(row["dev_fold"]))
    return out


def make_decision_table(index_files, elig_by_path):
    """index_files: list of dicts from index_labels.json (sorted by path); elig_by_path: path -> bool array per decision."""
    inst, r, y, el, st = [], [], [], [], []
    for i, f in enumerate(index_files):
        rr = decision_rows(f["rows"])
        for name, arr, runs_key in (("y", y, "class_runs"), ("st", st, "state_runs")):
            runs = np.array(f[runs_key], dtype=np.int64)
            starts = runs[:, 1]
            lab = runs[:, 0][np.searchsorted(starts, rr, side="right") - 1]
            arr.append(event_class(lab).astype(np.int8) if name == "y" else lab.astype(np.int8))
        e = np.asarray(elig_by_path[f["path"]], dtype=bool)
        assert len(e) == len(rr), (f["path"], len(e), len(rr))
        inst.append(np.full(len(rr), i, dtype=np.int32))
        r.append(rr.astype(np.int32))
        el.append(e)
    return (np.concatenate(inst), np.concatenate(r), np.concatenate(y), np.concatenate(el), np.concatenate(st))


# ------------------------------------------------------------------ alarm state machine (2 on, 2 off), vectorised
def alarm_state(bench: Benchmark, score: np.ndarray, theta: float | None):
    n = bench.n
    ok = bench.elig & np.isfinite(score)
    above = ok & (score >= theta) if theta is not None else np.zeros(n, dtype=bool)
    same_prev = np.r_[False, bench.inst[1:] == bench.inst[:-1]]
    prev_above = np.r_[False, above[:-1]] & same_prev
    set_on = above & prev_above
    force_off = (~ok) | (~same_prev) | (ok & ~above & ~prev_above)
    val = np.where(set_on, 1, np.where(force_off, 0, -1))
    idx = np.maximum.accumulate(np.where(val >= 0, np.arange(n), 0))
    state = val[idx] == 1
    prev_state = np.r_[False, state[:-1]] & same_prev
    rise = state & ~prev_state
    return state, rise


def alarm_state_reference(inst, elig, score, theta):
    """Plain loop, used only by the tests as an oracle."""
    out = np.zeros(len(score), dtype=bool)
    state = False
    on = off = 0
    for i in range(len(score)):
        if i == 0 or inst[i] != inst[i - 1]:
            state = False
            on = off = 0
        if not elig[i] or not np.isfinite(score[i]):
            state = False
            on = off = 0
            out[i] = False
            continue
        above = theta is not None and score[i] >= theta
        on = on + 1 if above else 0
        off = 0 if above else off + 1
        if on >= ON_K:
            state = True
        if off >= OFF_K:
            state = False
        out[i] = state
    return out


# ------------------------------------------------------------------ per-well sufficient statistics
_LAYOUT = {}


def _layout():
    if not _LAYOUT:
        items = [("conf", N_CLASSES * N_CLASSES), ("hist_pos", NBINS), ("hist_neg", NBINS), ("ep_total", N_CLASSES), ("ep_obs", N_CLASSES),
                 ("ep_cov", N_CLASSES), ("ep_fresh", N_CLASSES), ("on_obs", N_CLASSES), ("on_det", N_CLASSES), ("scalars", 8)]
        o = 0
        for k, sz in items:
            _LAYOUT[k] = (o, o + sz)
            o += sz
        _LAYOUT["_dim"] = o
    return _LAYOUT


SCALARS = ["norm_dec", "norm_alarm", "false_alarms", "norm_total", "norm_elig", "event_total", "event_elig", "cls_dec"]


def _logit_bin(score):
    s = np.clip(score, 1e-12, 1 - 1e-12)
    z = np.clip(np.log(s / (1 - s)), -LOGIT_CLIP, LOGIT_CLIP - 1e-9)
    return ((z + LOGIT_CLIP) / (2 * LOGIT_CLIP) * NBINS).astype(np.int64)


def validate(bench: Benchmark, score, pred, wells, need_pred=False):
    score = np.asarray(score, dtype=np.float64)
    if score.shape != (bench.n,):
        raise ValueError(f"score must have shape ({bench.n},)")
    sel = bench.well_mask(wells)[bench.well_of_decision]
    need = sel & bench.elig
    if not np.all(np.isfinite(score[need])):
        raise ValueError("non-finite score on eligible decisions of evaluated wells")
    if np.any((score[need] < 0) | (score[need] > 1)):
        raise ValueError("scores must lie in [0, 1]")
    if need_pred:
        p = np.asarray(pred)
        if p.shape != (bench.n,) or np.any((p[need] < 0) | (p[need] >= N_CLASSES)):
            raise ValueError("pred must give a class 0..9 on every eligible decision of the evaluated wells")
    return sel


def well_statistics(bench: Benchmark, score, pred, wells, theta):
    """Return (matrix [n_wells_selected, D], well names, delays list per well)."""
    L = _layout()
    sel = validate(bench, score, pred, wells, need_pred=pred is not None)
    score = np.where(sel, np.asarray(score, dtype=np.float64), np.nan)
    nw = len(bench.wells)
    w = bench.well_of_decision
    y = bench.y.astype(np.int64)
    lab = (y >= 0) & sel
    labe = lab & bench.elig
    state, rise = alarm_state(bench, score, theta)
    S = np.zeros((nw, L["_dim"]))
    # classification
    if pred is not None:
        p = np.asarray(pred).astype(np.int64)
        m = labe & (p >= 0) & (p < N_CLASSES)
        S[:, slice(*L["conf"])] = np.bincount(w[m] * 100 + y[m] * 10 + p[m], minlength=nw * 100).reshape(nw, 100)
        sc = S[:, L["scalars"][0]:L["scalars"][1]]
        sc[:, SCALARS.index("cls_dec")] = np.bincount(w[m], minlength=nw)
    # event-score histograms (event class > 0 versus normal)
    pos = labe & (y > 0)
    neg = labe & (y == 0)
    b = _logit_bin(np.where(np.isfinite(score), score, 0.5))
    S[:, slice(*L["hist_pos"])] = np.bincount(w[pos] * NBINS + b[pos], minlength=nw * NBINS).reshape(nw, NBINS)
    S[:, slice(*L["hist_neg"])] = np.bincount(w[neg] * NBINS + b[neg], minlength=nw * NBINS).reshape(nw, NBINS)
    sc = S[:, L["scalars"][0]:L["scalars"][1]]
    sc[:, SCALARS.index("norm_dec")] = np.bincount(w[neg], minlength=nw)
    sc[:, SCALARS.index("norm_alarm")] = np.bincount(w[neg & state], minlength=nw)
    sc[:, SCALARS.index("false_alarms")] = np.bincount(w[neg & rise], minlength=nw)
    sc[:, SCALARS.index("norm_total")] = np.bincount(w[lab & (y == 0)], minlength=nw)
    sc[:, SCALARS.index("norm_elig")] = np.bincount(w[neg], minlength=nw)
    sc[:, SCALARS.index("event_total")] = np.bincount(w[lab & (y > 0)], minlength=nw)
    sc[:, SCALARS.index("event_elig")] = np.bincount(w[pos], minlength=nw)
    # episodes via prefix sums over the decision table
    cs_e = np.r_[0, np.cumsum(bench.elig & sel)]
    cs_a = np.r_[0, np.cumsum(state)]
    cs_r = np.r_[0, np.cumsum(rise)]
    off = bench.inst_offset[bench.ep_inst]
    ndec = bench.inst_offset[bench.ep_inst + 1] - off
    lo = np.maximum(0, -(-(bench.ep_start - FIRST_ROW) // STRIDE))
    hi = np.minimum(ndec - 1, -(-(bench.ep_end - FIRST_ROW) // STRIDE) - 1)
    hi_on = np.minimum(ndec - 1, (bench.ep_start + ONSET_TOL - FIRST_ROW) // STRIDE)
    ok_ep = bench.well_mask(wells)[bench.ep_well] & (hi >= lo)
    gl, gh, gho = off + lo, off + hi, off + hi_on
    n_el = cs_e[gh + 1] - cs_e[gl]
    n_al = cs_a[gh + 1] - cs_a[gl]
    n_ri = cs_r[gh + 1] - cs_r[gl]
    obs = ok_ep & (n_el >= 1)
    cov = obs & (n_al >= 1)
    fresh = cov & (n_ri >= 1)
    n_el_on = cs_e[gho + 1] - cs_e[gl]
    n_ri_on = cs_r[gho + 1] - cs_r[gl]
    onset_obs = ok_ep & (bench.ep_known == 1) & (hi_on >= lo) & (n_el_on >= 2)
    onset_det = onset_obs & (n_ri_on >= 1)
    ev, ew = bench.ep_event, bench.ep_well
    for key, mask in (("ep_total", ok_ep), ("ep_obs", obs), ("ep_cov", cov), ("ep_fresh", fresh), ("on_obs", onset_obs), ("on_det", onset_det)):
        S[:, slice(*L[key])] = np.bincount(ew[mask] * N_CLASSES + ev[mask], minlength=nw * N_CLASSES).reshape(nw, N_CLASSES)
    # detection delays of detected onsets
    delays = [[] for _ in range(nw)]
    rise_idx = np.flatnonzero(rise)
    for k in np.flatnonzero(onset_det):
        pos0 = np.searchsorted(rise_idx, gl[k])
        d = bench.r[rise_idx[pos0]] - bench.ep_start[k]
        delays[ew[k]].append(int(d))
    keep = [i for i, wn in enumerate(bench.wells) if wn in set(wells)]
    return S[keep], [bench.wells[i] for i in keep], [delays[i] for i in keep]


# ------------------------------------------------------------------ metrics from (batched) sums
def _sl(S, key):
    a, b = _layout()[key]
    return S[..., a:b]


def metrics_from_sums(S, classes=None):
    """S: [..., D] summed statistics. Returns dict name -> array [...]. classes: restrict macro to these classes."""
    L = _layout()
    S = np.asarray(S, dtype=np.float64)
    out = {}
    conf = _sl(S, "conf").reshape(S.shape[:-1] + (N_CLASSES, N_CLASSES))
    tp = np.einsum("...ii->...i", conf)
    rowsum, colsum = conf.sum(-1), conf.sum(-2)
    with np.errstate(all="ignore"):
        f1 = np.where(2 * tp + (colsum - tp) + (rowsum - tp) > 0, 2 * tp / (2 * tp + (colsum - tp) + (rowsum - tp)), 0.0)
        rec = np.where(rowsum > 0, tp / rowsum, 0.0)
        prec = np.where(colsum > 0, tp / colsum, 0.0)
    present = rowsum > 0
    if classes is not None:
        present = present & np.isin(np.arange(N_CLASSES), classes)
    npres = present.sum(-1)
    out["macro_f1"] = np.where(npres > 0, (f1 * present).sum(-1) / np.maximum(npres, 1), np.nan)
    out["balanced_accuracy"] = np.where(npres > 0, (rec * present).sum(-1) / np.maximum(npres, 1), np.nan)
    out["accuracy"] = np.where(conf.sum((-1, -2)) > 0, tp.sum(-1) / np.maximum(conf.sum((-1, -2)), 1), np.nan)
    for c in range(N_CLASSES):
        out[f"f1_c{c}"] = np.where(rowsum[..., c] > 0, f1[..., c], np.nan)
    # event metrics
    ev = slice(1, N_CLASSES)
    ep_total = _sl(S, "ep_total")[..., ev].sum(-1)
    ep_obs = _sl(S, "ep_obs")[..., ev].sum(-1)
    ep_cov = _sl(S, "ep_cov")[..., ev].sum(-1)
    ep_fresh = _sl(S, "ep_fresh")[..., ev].sum(-1)
    on_obs = _sl(S, "on_obs")[..., ev].sum(-1)
    on_det = _sl(S, "on_det")[..., ev].sum(-1)
    sc = _sl(S, "scalars")
    g = lambda name: sc[..., SCALARS.index(name)]  # noqa: E731
    with np.errstate(all="ignore"):
        out["episode_coverage"] = np.where(ep_obs > 0, ep_cov / ep_obs, np.nan)
        out["carried_in_share_of_covered"] = np.where(ep_cov > 0, (ep_cov - ep_fresh) / ep_cov, np.nan)
        out["onset_detection_300s"] = np.where(on_obs > 0, on_det / on_obs, np.nan)
        days = g("norm_dec") * STRIDE / 86400.0
        out["false_alarms_per_normal_day"] = np.where(days > 0, g("false_alarms") / days, np.nan)
        out["normal_time_occupation"] = np.where(g("norm_dec") > 0, g("norm_alarm") / g("norm_dec"), np.nan)
        out["normal_eligibility"] = np.where(g("norm_total") > 0, g("norm_elig") / g("norm_total"), np.nan)
        out["event_eligibility"] = np.where(g("event_total") > 0, g("event_elig") / g("event_total"), np.nan)
        out["episodes_total"] = ep_total
        out["episodes_unobservable"] = ep_total - ep_obs
        out["normal_days"] = days
        for c in range(1, N_CLASSES):
            o = _sl(S, "ep_obs")[..., c]
            out[f"coverage_c{c}"] = np.where(o > 0, _sl(S, "ep_cov")[..., c] / o, np.nan)
        # threshold-free: AUROC and average precision from logit-binned histograms
        hp, hn = _sl(S, "hist_pos"), _sl(S, "hist_neg")
        P, N = hp.sum(-1), hn.sum(-1)
        cn = np.cumsum(hn, -1) - hn  # negatives strictly below the bin
        auc_num = (hp * (cn + 0.5 * hn)).sum(-1)
        out["auroc"] = np.where((P > 0) & (N > 0), auc_num / (P * N), np.nan)
        rp = np.cumsum(hp[..., ::-1], -1)
        rn = np.cumsum(hn[..., ::-1], -1)
        prec_curve = np.where(rp + rn > 0, rp / (rp + rn), 1.0)
        dr = hp[..., ::-1] / np.maximum(P[..., None], 1)
        out["average_precision"] = np.where(P > 0, (prec_curve * dr).sum(-1), np.nan)
        out["event_decisions"] = P
        out["normal_decisions"] = N
    return out


def report(bench, score, pred, wells, theta, classes=None, boot=True, B=BOOT_B, seed=BOOT_SEED):
    """Point estimates and wells-cluster bootstrap 95 % intervals for the evaluated wells."""
    S, names, delays = well_statistics(bench, score, pred, wells, theta)
    pt = {k: float(np.asarray(v)) if np.ndim(v) == 0 else v for k, v in metrics_from_sums(S.sum(0), classes).items()}
    flat = [d for ds in delays for d in ds]
    pt["onset_delay_median_s"] = float(np.median(flat)) if flat else float("nan")
    # per-well macro-F1 mean and sd (classes present in that well)
    if pred is not None:
        per_well = metrics_from_sums(S, classes)["macro_f1"]
        pt["per_well_macro_f1_mean"] = float(np.nanmean(per_well))
        pt["per_well_macro_f1_sd"] = float(np.nanstd(per_well, ddof=1)) if np.isfinite(per_well).sum() > 1 else float("nan")
    res = {"n_wells": len(names), "wells": names, "theta": theta, "point": pt}
    if boot:
        res["ci95"] = bootstrap(S, classes, B, seed)
    return res


def bootstrap(S, classes=None, B=BOOT_B, seed=BOOT_SEED, S2=None):
    """Percentile intervals; with S2 returns the paired difference (S minus S2) of every metric."""
    W = S.shape[0]
    rng = np.random.default_rng(seed)
    counts = np.stack([np.bincount(rng.integers(0, W, W), minlength=W) for _ in range(B)]).astype(np.float64)
    m1 = metrics_from_sums(counts @ S, classes)
    if S2 is not None:
        m2 = metrics_from_sums(counts @ S2, classes)
        m1 = {k: m1[k] - m2[k] for k in m1}
    out = {}
    for k, v in m1.items():
        v = np.asarray(v)
        if v.shape == (B,) and np.isfinite(v).any():
            out[k] = [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5)), int(np.isfinite(v).sum())]
    return out


def select_threshold(bench, score, wells, fa_per_day_max=1.0, occupation_max=0.05, min_normal_days=0.5):
    """Frozen-on-dev threshold: lowest threshold meeting both budgets on the given (dev, out-of-fold) predictions.
    Ties on coverage go to the lower false-alarm rate. Returns (theta or None, table of candidates)."""
    sel = bench.well_mask(wells)[bench.well_of_decision]
    cand_scores = score[sel & bench.elig & (bench.y == 0) & np.isfinite(score)]
    if len(cand_scores) == 0:
        return None, []
    qs = np.unique(np.r_[np.quantile(cand_scores, np.linspace(0.5, 1.0, 101)), np.linspace(0.05, 0.99, 48)])
    table, best = [], None
    for th in qs:
        S, _, _ = well_statistics(bench, score, None, wells, float(th))
        m = metrics_from_sums(S.sum(0))
        fa, occ, cov, days = float(m["false_alarms_per_normal_day"]), float(m["normal_time_occupation"]), float(m["episode_coverage"]), float(m["normal_days"])
        feasible = days >= min_normal_days and fa <= fa_per_day_max and occ <= occupation_max
        table.append({"theta": float(th), "fa_per_day": fa, "occupation": occ, "coverage": cov, "feasible": bool(feasible)})
        if feasible and (best is None or (cov, -fa) > (best["coverage"], -best["fa_per_day"])):
            best = table[-1]
    return (best["theta"] if best else None), table
