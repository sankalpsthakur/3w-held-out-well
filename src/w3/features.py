"""Causal window features for 3W decisions (numpy only).

For a decision at row r only rows r-3599 .. r of the same instance are used (lookback cap of the protocol).
Everything is computed from prefix sums (O(rows)) except min/max, which use strided windows at decision rows only.
"""
from __future__ import annotations

import numpy as np

try:
    from .scorer import FIRST_ROW, STRIDE, decision_rows
except ImportError:  # pragma: no cover
    from scorer import FIRST_ROW, STRIDE, decision_rows  # type: ignore

CONT = ["P-ANULAR", "P-JUS-BS", "P-JUS-CKGL", "P-JUS-CKP", "P-MON-CKGL", "P-MON-CKP", "P-MON-SDV-P", "P-PDG", "PT-P", "P-TPT",
        "QBS", "QGL", "T-JUS-CKP", "T-MON-CKP", "T-PDG", "T-TPT", "ABER-CKGL", "ABER-CKP"]
STATE = ["ESTADO-DHSV", "ESTADO-M1", "ESTADO-M2", "ESTADO-PXO", "ESTADO-SDV-GL", "ESTADO-SDV-P", "ESTADO-W1", "ESTADO-W2", "ESTADO-XO"]
W2, W3, W4 = 300, 900, 3600
EPS_REL = 1e-9

PER_CHANNEL = ["last", "mean_w2", "mean_w3", "mean_w4", "std_w2", "std_w3", "std_w4", "min_w3", "max_w3", "range_rel_w3",
               "drift_w3", "z_last_w4", "z_w2_w4", "cv_w3", "rough_w3", "extrema_rate_w3", "finite_w3"]
PER_STATE = ["last", "changes_w3", "changes_w4"]
# scale-free subset (no absolute level): usable across wells with different units / operating points
SCALE_FREE = ["range_rel_w3", "drift_w3", "z_last_w4", "z_w2_w4", "cv_w3", "extrema_rate_w3", "finite_w3"]


def feature_names():
    return [f"{c}|{f}" for c in CONT for f in PER_CHANNEL] + [f"{c}|{f}" for c in STATE for f in PER_STATE]


def _prefix(a):
    return np.r_[0.0, np.cumsum(a, dtype=np.float64)]


def _window_sums(cs, r, W):
    lo = r - W + 1
    return cs[r + 1] - cs[lo]


def channel_features(x: np.ndarray, r: np.ndarray) -> np.ndarray:
    """x: [rows] float64 with NaN; r: decision rows. Returns [n_dec, len(PER_CHANNEL)] float32."""
    n = len(r)
    out = np.full((n, len(PER_CHANNEL)), np.nan, dtype=np.float64)
    fin = np.isfinite(x)
    if not fin.any():
        out[:, PER_CHANNEL.index("finite_w3")] = 0.0
        return out.astype(np.float32)
    shift = float(np.median(x[fin][:7200]))            # numerical conditioning only; added back for levels
    xs = np.where(fin, x - shift, 0.0)
    c0, c1, c2 = _prefix(fin.astype(np.float64)), _prefix(xs), _prefix(xs * xs)

    def stats(W):
        cnt = _window_sums(c0, r, W)
        s1 = _window_sums(c1, r, W)
        s2 = _window_sums(c2, r, W)
        with np.errstate(all="ignore"):
            m = np.where(cnt > 0, s1 / cnt, np.nan)
            v = np.where(cnt > 1, np.maximum(s2 / cnt - m * m, 0.0), np.nan)
        return cnt, m, np.sqrt(v)

    cnt2, m2, s2 = stats(W2)
    cnt3, m3, s3 = stats(W3)
    cnt4, m4, s4 = stats(W4)
    last = np.where(fin[r], x[r], np.nan)
    ix = {k: i for i, k in enumerate(PER_CHANNEL)}
    out[:, ix["last"]] = last
    out[:, ix["mean_w2"]] = m2 + shift
    out[:, ix["mean_w3"]] = m3 + shift
    out[:, ix["mean_w4"]] = m4 + shift
    out[:, ix["std_w2"]], out[:, ix["std_w3"]], out[:, ix["std_w4"]] = s2, s3, s4
    # min / max over W3 using strided windows at decision rows only
    xw = np.where(fin, x, np.nan)
    win = np.lib.stride_tricks.sliding_window_view(xw, W3)[FIRST_ROW - W3 + 1::STRIDE][:n]
    with np.errstate(all="ignore"):
        mn = np.fmin.reduce(win, axis=1)
        mx = np.fmax.reduce(win, axis=1)
    out[:, ix["min_w3"]], out[:, ix["max_w3"]] = mn, mx
    with np.errstate(all="ignore"):
        out[:, ix["range_rel_w3"]] = (mx - mn) / (np.abs(m3 + shift) + EPS_REL + s4)
        # drift: mean of last quarter minus mean of first quarter of W3, in units of std_w4
        q = W3 // 4
        first = (_window_sums(c1, r - W3 + q, q)) / np.maximum(_window_sums(c0, r - W3 + q, q), 1)
        lastq = (_window_sums(c1, r, q)) / np.maximum(_window_sums(c0, r, q), 1)
        ok_q = (_window_sums(c0, r - W3 + q, q) > 0) & (_window_sums(c0, r, q) > 0)
        out[:, ix["drift_w3"]] = np.where(ok_q, (lastq - first) / (s4 + EPS_REL * (1 + np.abs(m4 + shift))), np.nan)
        denom = s4 + EPS_REL * (1 + np.abs(m4 + shift))
        out[:, ix["z_last_w4"]] = (last - (m4 + shift)) / denom
        out[:, ix["z_w2_w4"]] = (m2 - m4) / denom
        out[:, ix["cv_w3"]] = s3 / (np.abs(m3 + shift) + EPS_REL + s4)
    # roughness and local-extrema rate over W3
    dx = np.diff(xw)
    dfin = np.isfinite(dx)
    ad = np.where(dfin, np.abs(dx), 0.0)
    cd0, cd1 = _prefix(dfin.astype(np.float64)), _prefix(ad)
    rr = r - 1  # differences index: dx[i] = x[i+1]-x[i]; window rows lo..r use dx[lo..r-1]
    cntd = cd0[rr + 1] - cd0[r - W3 + 1]
    with np.errstate(all="ignore"):
        out[:, ix["rough_w3"]] = np.where(cntd > 0, (cd1[rr + 1] - cd1[r - W3 + 1]) / cntd, np.nan) / (s4 + EPS_REL * (1 + np.abs(m4 + shift)))
    ext = np.zeros(len(dx), dtype=np.float64)
    prod = dx[1:] * dx[:-1]
    ext[1:] = np.where(np.isfinite(prod) & (prod < 0), 1.0, 0.0)
    ce = _prefix(ext)
    cnte = cntd
    with np.errstate(all="ignore"):
        out[:, ix["extrema_rate_w3"]] = np.where(cnte > 2, (ce[rr + 1] - ce[r - W3 + 2]) / cnte, np.nan)
    out[:, ix["finite_w3"]] = cnt3 / W3
    return out.astype(np.float32)


def state_features(e: np.ndarray, r: np.ndarray) -> np.ndarray:
    n = len(r)
    out = np.full((n, len(PER_STATE)), np.nan, dtype=np.float64)
    fin = np.isfinite(e)
    if not fin.any():
        return out.astype(np.float32)
    out[:, 0] = np.where(fin[r], e[r], np.nan)
    d = np.diff(e)
    ch = np.where(np.isfinite(d) & (d != 0), 1.0, 0.0)
    cc = _prefix(ch)
    rr = r - 1
    out[:, 1] = cc[rr + 1] - cc[r - W3 + 1]
    out[:, 2] = cc[rr + 1] - cc[r - W4 + 1]
    return out.astype(np.float32)


def instance_features(cont: np.ndarray, state: np.ndarray) -> np.ndarray:
    """cont: [rows, 18] process channels, state: [rows, 9] valve states (NaN where absent). Returns [n_dec, F] float32."""
    rows = cont.shape[0]
    r = decision_rows(rows)
    blocks = [channel_features(cont[:, j].astype(np.float64), r) for j in range(len(CONT))]
    blocks += [state_features(state[:, j].astype(np.float64), r) for j in range(len(STATE))]
    return np.concatenate(blocks, axis=1)


def reference_channel_features(x, r):
    """Straightforward slice implementation used only by the tests."""
    out = np.full((len(r), len(PER_CHANNEL)), np.nan)
    ix = {k: i for i, k in enumerate(PER_CHANNEL)}
    for k, rr in enumerate(r):
        def win(W):
            w = x[rr - W + 1: rr + 1]
            return w[np.isfinite(w)], w
        a2, _ = win(W2); a3, w3 = win(W3); a4, _ = win(W4)
        out[k, ix["last"]] = x[rr] if np.isfinite(x[rr]) else np.nan
        for name, a in (("w2", a2), ("w3", a3), ("w4", a4)):
            out[k, ix["mean_" + name]] = a.mean() if len(a) else np.nan
            out[k, ix["std_" + name]] = a.std() if len(a) > 1 else np.nan
        out[k, ix["min_w3"]] = a3.min() if len(a3) else np.nan
        out[k, ix["max_w3"]] = a3.max() if len(a3) else np.nan
        out[k, ix["finite_w3"]] = len(a3) / W3
        s4 = a4.std() if len(a4) > 1 else np.nan
        m4 = a4.mean() if len(a4) else np.nan
        m3 = a3.mean() if len(a3) else np.nan
        m2 = a2.mean() if len(a2) else np.nan
        out[k, ix["range_rel_w3"]] = (a3.max() - a3.min()) / (abs(m3) + EPS_REL + s4) if len(a3) else np.nan
        denom = s4 + EPS_REL * (1 + abs(m4))
        out[k, ix["z_last_w4"]] = (out[k, ix["last"]] - m4) / denom
        out[k, ix["z_w2_w4"]] = (m2 - m4) / denom
        out[k, ix["cv_w3"]] = (a3.std() / (abs(m3) + EPS_REL + s4)) if len(a3) > 1 else np.nan
        q = W3 // 4
        wq1, wq2 = w3[:q], w3[-q:]
        f1, f2 = wq1[np.isfinite(wq1)], wq2[np.isfinite(wq2)]
        out[k, ix["drift_w3"]] = (f2.mean() - f1.mean()) / denom if len(f1) and len(f2) else np.nan
        d = np.diff(w3)
        dd = np.abs(d)[np.isfinite(d)]
        out[k, ix["rough_w3"]] = (dd.mean() / denom) if len(dd) else np.nan
        prod = d[1:] * d[:-1]
        pf = np.isfinite(prod)
        cnt = np.isfinite(d).sum()
        out[k, ix["extrema_rate_w3"]] = ((pf & (prod < 0)).sum() / cnt) if cnt > 2 else np.nan
    return out


# ----------------------------------------------------------------------------- simple causal reference detector
SIMPLE_CHANNELS = CONT[:16]          # pressures, flows, temperatures (no choke apertures)
SHORT, REF = 300, 3300               # short window [r-299, r]; reference window [r-3599, r-300]


def simple_detector_score(cont: np.ndarray) -> np.ndarray:
    """Windowed z-test against the instance's own recent past, max over channels. Returns raw score >= 0 [n_dec] (NaN if no channel usable).

    level_z = |mean_short - mean_ref| / (std_ref + 1e-3 |mean_ref| + 1e-12);  vol_z = 2 max(0, std_short / (std_ref + eps) - 1).
    A channel is usable at a decision when >= 80 % of both windows are finite. Score = max_c max(level_z, vol_z). No training, no labels.
    """
    rows = cont.shape[0]
    r = decision_rows(rows)
    best = np.full(len(r), -np.inf)
    usable = np.zeros(len(r), dtype=bool)
    for j in range(len(SIMPLE_CHANNELS)):
        x = cont[:, j].astype(np.float64)
        fin = np.isfinite(x)
        if not fin.any():
            continue
        shift = float(np.median(x[fin][:7200]))
        xs = np.where(fin, x - shift, 0.0)
        c0, c1, c2 = _prefix(fin.astype(np.float64)), _prefix(xs), _prefix(xs * xs)

        def stats(lo_off, hi_off):   # rows r-lo_off+1 .. r-hi_off
            cnt = c0[r - hi_off + 1] - c0[r - lo_off + 1]
            s1 = c1[r - hi_off + 1] - c1[r - lo_off + 1]
            s2 = c2[r - hi_off + 1] - c2[r - lo_off + 1]
            with np.errstate(all="ignore"):
                m = np.where(cnt > 0, s1 / cnt, np.nan)
                sd = np.sqrt(np.where(cnt > 1, np.maximum(s2 / cnt - m * m, 0.0), np.nan))
            return cnt, m, sd

        cnt_s, m_s, sd_s = stats(SHORT, 0)
        cnt_r, m_r, sd_r = stats(SHORT + REF, SHORT)
        ok = (cnt_s >= 0.8 * SHORT) & (cnt_r >= 0.8 * REF)
        with np.errstate(all="ignore"):
            mean_ref = m_r + shift
            level = np.abs(m_s - m_r) / (sd_r + 1e-3 * np.abs(mean_ref) + 1e-12)
            vol = 2.0 * np.maximum(0.0, sd_s / (sd_r + 1e-3 * np.abs(mean_ref) + 1e-12) - 1.0)
            z = np.where(ok, np.maximum(level, vol), -np.inf)
        best = np.maximum(best, z)
        usable |= ok
    out = np.where(usable & np.isfinite(best), best, np.nan)
    return out


def simple_score_to_unit(z):
    """Monotone map of the raw score to [0, 1] (the scorer expects P(event)-like values)."""
    z = np.asarray(z, dtype=np.float64)
    return np.where(np.isfinite(z), z / (z + 3.0), np.nan)
