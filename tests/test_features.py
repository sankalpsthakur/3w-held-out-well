import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from w3 import features as F  # noqa: E402
from w3.scorer import decision_rows  # noqa: E402


def synth(rows=9000, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(rows)
    x = 1e7 + 3e5 * np.sin(t / 200.0) + 2e4 * rng.normal(size=rows) + 50.0 * t / 1000
    x[2000:2100] = np.nan
    x[6000:6003] = np.nan
    return x


def test_channel_features_match_reference():
    x = synth()
    r = decision_rows(len(x))
    got = F.channel_features(x, r).astype(np.float64)
    ref = F.reference_channel_features(x, r)
    for j, name in enumerate(F.PER_CHANNEL):
        a, b = got[:, j], ref[:, j]
        both_nan = np.isnan(a) & np.isnan(b)
        assert np.array_equal(np.isnan(a), np.isnan(b)), name
        scale = max(1.0, float(np.nanmax(np.abs(b[~both_nan]))))
        assert np.allclose(a[~both_nan], b[~both_nan], rtol=2e-4, atol=1e-5 * scale), (name, np.nanmax(np.abs(a - b)))


def test_causality_future_rows_do_not_change_past_features():
    rng = np.random.default_rng(1)
    cont = np.stack([synth(9000, s) for s in range(3)] + [np.full(9000, np.nan)], axis=1)
    state = np.zeros((9000, 2)); state[4000:, 0] = 1
    r = decision_rows(9000)
    base = np.concatenate([F.channel_features(cont[:, j], r) for j in range(cont.shape[1])], axis=1)
    cut = r[40]
    cont2 = cont.copy()
    cont2[cut + 1:] = rng.normal(size=cont2[cut + 1:].shape) * 1e9
    alt = np.concatenate([F.channel_features(cont2[:, j], r) for j in range(cont.shape[1])], axis=1)
    k = np.flatnonzero(r <= cut)
    assert np.array_equal(np.nan_to_num(base[k], nan=-7), np.nan_to_num(alt[k], nan=-7))
    s1, s2 = F.state_features(state[:, 0], r), F.state_features(np.where(np.arange(9000) <= cut, state[:, 0], 5.0), r)
    assert np.array_equal(s1[k], s2[k])


def test_all_nan_channel_and_shapes():
    x = np.full(5000, np.nan)
    r = decision_rows(5000)
    o = F.channel_features(x, r)
    assert o.shape == (len(r), len(F.PER_CHANNEL)) and (o[:, F.PER_CHANNEL.index("finite_w3")] == 0).all()
    cont = np.full((5000, len(F.CONT)), np.nan); st = np.full((5000, len(F.STATE)), np.nan)
    out = F.instance_features(cont, st)
    assert out.shape == (len(r), len(F.feature_names()))


def test_simple_detector_matches_naive_and_detects_shift():
    rng = np.random.default_rng(0)
    rows = 9000
    cont = np.full((rows, 18), np.nan)
    cont[:, 0] = 1e7 + rng.normal(size=rows) * 1e3
    cont[5000:, 0] += 8e3
    cont[:, 3] = 50 + rng.normal(size=rows) * 0.1
    cont[7000:, 3] = 50 + rng.normal(size=rows - 7000) * 0.6
    z = F.simple_detector_score(cont)
    r = decision_rows(rows)

    def naive(k):
        rr = r[k]; best = -np.inf
        for j in range(16):
            xs = cont[rr - 299:rr + 1, j]; xr = cont[rr - 3599:rr - 299, j]
            if np.isfinite(xs).mean() < 0.8 or np.isfinite(xr).mean() < 0.8:
                continue
            xs = xs[np.isfinite(xs)]; xr = xr[np.isfinite(xr)]
            mr, sr, ss = xr.mean(), xr.std(), xs.std()
            d = sr + 1e-3 * abs(mr) + 1e-12
            best = max(best, abs(xs.mean() - mr) / d, 2 * max(0, ss / d - 1))
        return best if np.isfinite(best) else np.nan
    for k in (0, 10, 40, 60, 100, 120, 150):
        assert np.isclose(z[k], naive(k), rtol=1e-6, equal_nan=True)
    assert np.nanmean(z[(r > 5100) & (r < 5500)]) > 10 * np.nanmean(z[(r > 4000) & (r < 4800)])
    assert np.isnan(F.simple_detector_score(np.full((5000, 18), np.nan))).all()
    u = F.simple_score_to_unit(np.array([0.0, 3.0, np.nan, 1e9]))
    assert u[0] == 0.0 and u[1] == 0.5 and np.isnan(u[2]) and u[3] < 1.0 + 1e-6


def test_pipeline_helpers_scale_free_and_augmentation_and_smoothing():
    from w3 import pipeline as P
    from w3 import scorer as S
    names = F.feature_names()
    cols = P.scale_free_columns(names)
    kept = [names[i] for i in cols]
    assert all(("mean_w" not in n and "|last" not in n.replace("ESTADO", "")) or n.startswith("ESTADO") for n in kept)
    assert len(cols) == 18 * len(P.SCALE_FREE_FEATS) + 9 * 3
    groups = P.channel_col_groups(names)
    assert len(groups) == 27 and sum(len(g) for g in groups) == len(names)
    X = np.ones((2000, len(cols)), dtype=np.float32)
    gl = P.localize_groups(groups, cols)
    Xa = P.augment_channel_dropout(X, gl, np.random.default_rng(0), p=0.3)
    frac = np.isnan(Xa).mean()
    assert 0.25 < frac < 0.35 and np.isfinite(X).all()
    # a dropped channel is dropped for all of its columns in that row
    g = gl[0]
    assert (np.isnan(Xa[:, g]).sum(1) % len(g) == 0).all()

    class B:  # minimal benchmark stand-in
        inst_rows = np.array([10, 10]); inst_offset = np.array([0, 6, 12])
    p = np.zeros((12, 10), dtype=np.float32); p[:, 0] = 1.0
    p[3:, 0] = 0.0; p[3:, 2] = 1.0         # a step from class 0 to class 2 in instance 0
    p[6:, :] = p[6:, :]                      # instance 1 identical
    p[5, :] = np.nan                          # a missing decision
    sm = P.ema_probs(B, p, 0.5)
    assert np.allclose(sm[0, 0], 1.0) and 0.4 < sm[3, 2] < 0.6 and sm[4, 2] > sm[3, 2]
    assert np.isnan(sm[5]).all() and np.isfinite(sm[4]).all()
    assert np.allclose(P.ema_probs(B, p, 1.0)[:5], p[:5])
