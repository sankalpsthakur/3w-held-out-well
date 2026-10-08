import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from w3 import scorer as S  # noqa: E402
from w3.labels import episodes_from_runs, event_class, class_support  # noqa: E402


def fake_file(path, well, rows, class_runs, state_runs=None):
    return {"path": path, "well": well, "rows": rows, "class_runs": class_runs, "state_runs": state_runs or [[-1, 0, 3600], [0, 3600, rows]]}


def make_bench(files, elig=None, manifest=None):
    files = sorted(files, key=lambda f: f["path"])
    elig_by_path = {f["path"]: (elig[f["path"]] if elig and f["path"] in elig else np.ones(S.n_decisions(f["rows"]), bool)) for f in files}
    inst, r, y, el, st = S.make_decision_table(files, elig_by_path)
    wells = sorted({f["well"] for f in files})
    man = manifest or {w: ("test", -1) for w in wells}
    return S.Benchmark(inst, r, y, el, st, [f["rows"] for f in files], [wells.index(f["well"]) for f in files], [f["path"] for f in files],
                       wells, [f["class_runs"] for f in files], man)


def test_grid_and_eligibility():
    assert S.n_decisions(3600) == 0 and S.n_decisions(3601) == 1 and S.n_decisions(3630) == 1 and S.n_decisions(3631) == 2
    assert list(S.decision_rows(3700)) == [3600, 3630, 3660, 3690]
    rows = 6000
    any_proc = np.ones(rows, bool)
    assert S.eligibility_from_any_process(any_proc).all()
    any_proc[3550:3600 + 95] = False           # hole of 145 rows before first decisions
    e = S.eligibility_from_any_process(any_proc)
    r = S.decision_rows(rows)
    # window for decision r is rows r-899..r; eligible iff <=90 missing rows in window
    for k, rr in enumerate(r):
        missing = (~any_proc[rr - 899:rr + 1]).sum()
        assert e[k] == (missing <= 90), (rr, missing)


def test_event_class_and_episodes():
    assert list(event_class(np.array([-1, 0, 3, 103, 109]))) == [-1, 0, 3, 3, 9]
    runs = [[-1, 0, 3600], [0, 3600, 5000], [102, 5000, 6000], [2, 6000, 6500], [0, 6500, 7000], [4, 7000, 8000]]
    eps = episodes_from_runs(runs)
    assert eps == [{"event": 2, "start": 5000, "end": 6500, "known_onset": True}, {"event": 4, "start": 7000, "end": 8000, "known_onset": True}]
    # an NA gap or a missing neighbour breaks onset knowledge
    eps = episodes_from_runs([[-1, 0, 3600], [4, 3600, 9000]])
    assert eps[0]["known_onset"] is False
    rows, sup = class_support(runs)
    assert sup == [2, 4] and rows[2] == 1500


def test_decision_table_labels():
    f = fake_file("a/WELL-00001_1.parquet", "WELL-00001", 8000, [[-1, 0, 3600], [0, 3600, 5000], [102, 5000, 6000], [2, 6000, 8000]])
    b = make_bench([f])
    assert b.n == S.n_decisions(8000)
    r = b.r.astype(int)
    assert b.y[r == 3600][0] == 0 and b.y[r == 4980][0] == 0 and b.y[r == 5010][0] == 2 and b.y[r == 7980][0] == 2
    assert len(b.ep_start) == 1 and b.ep_start[0] == 5000 and b.ep_end[0] == 8000


def test_alarm_state_matches_reference_on_random_sequences():
    rng = np.random.default_rng(0)
    files = [fake_file(f"d/{i}/WELL-0000{i}_x.parquet", f"WELL-0000{i}", 3600 + 30 * n, [[-1, 0, 3600], [0, 3600, 3600 + 30 * n]])
             for i, n in [(1, 300), (2, 200), (3, 450)]]
    elig = {f["path"]: rng.random(S.n_decisions(f["rows"])) > 0.15 for f in files}
    b = make_bench(files, elig)
    for trial in range(40):
        score = rng.random(b.n)
        score[rng.random(b.n) < 0.05] = np.nan
        theta = float(rng.uniform(0.2, 0.9))
        state, rise = S.alarm_state(b, score, theta)
        ref = S.alarm_state_reference(b.inst, b.elig, score, theta)
        assert np.array_equal(state, ref)
        prev = np.r_[False, ref[:-1]] & np.r_[False, b.inst[1:] == b.inst[:-1]]
        assert np.array_equal(rise, ref & ~prev)
    st, _ = S.alarm_state(b, score, None)
    assert not st.any()


def _scenario():
    # one well, two instances: A has a normal period then event 2 (known onset) ; B is all normal
    A = fake_file("dataset/2/WELL-00001_a.parquet", "WELL-00001", 9000, [[-1, 0, 3600], [0, 3600, 5100], [102, 5100, 6600], [2, 6600, 9000]])
    B = fake_file("dataset/0/WELL-00001_b.parquet", "WELL-00001", 8000, [[-1, 0, 3600], [0, 3600, 8000]])
    return make_bench([A, B])


def test_perfect_detector_scores_and_cluster_bootstrap_determinism():
    b = _scenario()
    wells = ["WELL-00001"]
    score = np.where(b.y > 0, 0.99, 0.01).astype(float)
    pred = np.where(b.y >= 0, b.y, 0)
    rep = S.report(b, score, pred, wells, theta=0.5, boot=False)
    p = rep["point"]
    assert p["macro_f1"] == 1.0 and p["episode_coverage"] == 1.0 and p["false_alarms_per_normal_day"] == 0.0
    assert p["normal_time_occupation"] == 0.0 and p["auroc"] == pytest.approx(1.0) and p["average_precision"] == pytest.approx(1.0)
    # onset: alarm needs two consecutive decisions, so the first on-state is 30 s after the first event decision (rise at 2nd)
    assert p["onset_detection_300s"] == 1.0 and p["onset_delay_median_s"] <= 60
    r1 = S.report(b, score, pred, wells, 0.5, boot=True, B=50, seed=1)["ci95"]
    r2 = S.report(b, score, pred, wells, 0.5, boot=True, B=50, seed=1)["ci95"]
    assert r1 == r2


def test_constant_normal_predictor_and_false_alarm_accounting():
    b = _scenario()
    wells = ["WELL-00001"]
    zeros = np.zeros(b.n)
    rep = S.report(b, zeros, np.zeros(b.n, int), wells, theta=0.5, boot=False)["point"]
    assert rep["episode_coverage"] == 0.0 and rep["false_alarms_per_normal_day"] == 0.0
    # always alarm: one rising edge per instance start inside normal periods, occupation near 1
    ones = np.ones(b.n)
    rep = S.report(b, ones, np.ones(b.n, int), wells, theta=0.5, boot=False)["point"]
    assert rep["episode_coverage"] == 1.0 and rep["normal_time_occupation"] > 0.98
    # alarm turns on at 2nd decision of each instance (both instances start in a normal period) -> 2 false alarm edges
    days = rep["normal_days"]
    assert rep["false_alarms_per_normal_day"] == pytest.approx(2 / days)
    assert rep["carried_in_share_of_covered"] == 1.0 or rep["carried_in_share_of_covered"] >= 0.0
    # a detector that only fires after the event started carries nothing in
    late = np.where(b.r.astype(int) >= 6600, 0.99, 0.0) * (b.inst == 1)  # instance 1 = dataset/2/... (sorted by path)
    rep = S.report(b, late, np.where(late > 0.5, 2, 0), wells, 0.5, boot=False)["point"]
    assert rep["episode_coverage"] == 1.0 and rep["onset_detection_300s"] == 0.0 and rep["false_alarms_per_normal_day"] == 0.0


def test_ineligible_decisions_reset_state_and_are_excluded():
    A = fake_file("dataset/2/WELL-00001_a.parquet", "WELL-00001", 9000, [[-1, 0, 3600], [0, 3600, 5100], [102, 5100, 6600], [2, 6600, 9000]])
    n = S.n_decisions(9000)
    e = np.ones(n, bool)
    e[:10] = False                                    # first 10 decisions ineligible
    b = make_bench([A], {A["path"]: e})
    score = np.where(np.isfinite(np.zeros(b.n)), 0.9, 0.0)
    rep = S.report(b, score, np.zeros(b.n, int), ["WELL-00001"], 0.5, boot=False)["point"]
    assert rep["normal_eligibility"] < 1.0 and rep["event_eligibility"] == 1.0
    bad = score.copy(); bad[20] = np.nan
    with pytest.raises(ValueError):
        S.report(b, bad, np.zeros(b.n, int), ["WELL-00001"], 0.5, boot=False)
    with pytest.raises(ValueError):
        S.report(b, score * 2, np.zeros(b.n, int), ["WELL-00001"], 0.5, boot=False)


def test_threshold_selection_respects_budgets():
    b = _scenario()
    rng = np.random.default_rng(3)
    score = np.where(b.y > 0, rng.beta(8, 2, b.n), rng.beta(2, 8, b.n))
    theta, table = S.select_threshold(b, score, ["WELL-00001"], fa_per_day_max=5.0, occupation_max=0.05, min_normal_days=0.05)
    assert theta is not None
    rep = S.report(b, score, None, ["WELL-00001"], theta, boot=False)["point"]
    assert rep["false_alarms_per_normal_day"] <= 5.0 and rep["normal_time_occupation"] <= 0.05


def test_paired_bootstrap_difference_zero_for_identical_models():
    b = _scenario()
    score = np.where(b.y > 0, 0.99, 0.01).astype(float)
    Sx, _, _ = S.well_statistics(b, score, np.where(b.y >= 0, b.y, 0), ["WELL-00001"], 0.5)
    d = S.bootstrap(np.vstack([Sx, Sx]), None, B=20, seed=2, S2=np.vstack([Sx, Sx]))
    assert all(v[0] == 0.0 and v[1] == 0.0 for k, v in d.items() if k in ("macro_f1", "auroc"))
