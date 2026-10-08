import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from w3 import gbdt_numpy as G  # noqa: E402

# Hand-written LightGBM-format text: 2 classes, 1 iteration = 2 trees. Tree 0: x0 <= 0.5 -> leaf0, else x1 <= 2 -> leaf1 else leaf2 (NaN follows default).
MODEL = """tree
version=v4
num_class=2
num_tree_per_iteration=2
objective=multiclass num_class:2

Tree=0
num_leaves=3
num_cat=0
split_feature=0 1
split_gain=1 1
threshold=0.5 2
decision_type=10 2
left_child=-1 -2
right_child=1 -3
leaf_value=0.5 -0.25 1.0
leaf_weight=1 1 1
leaf_count=1 1 1
internal_value=0 0
internal_weight=0 0
internal_count=3 2
is_linear=0
shrinkage=1


Tree=1
num_leaves=2
num_cat=0
split_feature=1
split_gain=1
threshold=1.0
decision_type=2
left_child=-1
right_child=-2
leaf_value=-0.5 0.5
leaf_weight=1 1
leaf_count=1 1
internal_value=0
internal_weight=0
internal_count=2
is_linear=0
shrinkage=1


end of trees
"""


def manual(x):
    # tree 0
    x0, x1 = x
    if np.isnan(x0):
        left = (10 & 2) != 0                       # decision_type 10: missing_type NaN (2<<2=8) | default_left (2)
        node0 = left
    else:
        node0 = x0 <= 0.5
    if node0:
        v0 = 0.5
    else:
        # node 1: decision_type 2 -> missing_type none, default_left bit set; NaN is turned into 0.0 and compared
        f = 0.0 if np.isnan(x1) else x1
        v0 = -0.25 if f <= 2 else 1.0
    f1 = 0.0 if np.isnan(x1) else x1                 # tree 1: decision_type 2, missing type none -> NaN becomes 0.0
    v1 = -0.5 if f1 <= 1.0 else 0.5
    return np.array([v0, v1])


def test_parse_and_predict_matches_manual_semantics():
    m = G.parse_model(MODEL)
    X = np.array([[0.0, 0.0], [0.7, 1.0], [0.7, 3.0], [np.nan, 5.0], [0.7, np.nan], [1.0, 2.0], [0.5, 9.0]])
    got = G.raw_scores(m, X)
    want = np.stack([manual(x) for x in X])
    assert np.allclose(got, want), (got, want)
    p = G.predict_proba(m, X)
    assert np.allclose(p.sum(1), 1.0)


def test_compact_roundtrip_preserves_argmax_and_is_small():
    m = G.parse_model(MODEL)
    c = G.compact_export(m)
    m2 = G.from_compact(c, 2)
    X = np.random.default_rng(0).normal(size=(200, 2)) * 2
    a, b = G.raw_scores(m, X), G.raw_scores(m2, X)
    assert np.abs(a - b).max() < 0.02 and (a.argmax(1) == b.argmax(1)).mean() > 0.97


def test_depth_is_max_over_all_paths():
    # tree 0 above has a right-hand chain of depth 2 and a left leaf: the max depth must be 2 (not the left-path depth 1)
    m = G.parse_model(MODEL)
    assert m["depth"] == 2


BINARY = """tree
version=v4
num_class=1
num_tree_per_iteration=1
objective=binary sigmoid:1

Tree=0
num_leaves=2
num_cat=0
split_feature=0
split_gain=1
threshold=0.5
decision_type=2
left_child=-1
right_child=-2
leaf_value=-1.0 2.0
leaf_weight=1 1
leaf_count=1 1
internal_value=0
internal_weight=0
internal_count=2
is_linear=0
shrinkage=1


Tree=1
num_leaves=2
num_cat=0
split_feature=1
split_gain=1
threshold=1.0
decision_type=2
left_child=-1
right_child=-2
leaf_value=0.25 -0.75
leaf_weight=1 1
leaf_count=1 1
internal_value=0
internal_weight=0
internal_count=2
is_linear=0
shrinkage=1


end of trees
"""


def test_binary_objective_gives_one_sigmoid_column_not_a_constant():
    m = G.parse_model(BINARY)
    assert m["n_class"] == 1 and m["sigmoid"] == 1.0
    X = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 2.0], [1.0, 2.0]])
    raw = np.array([-1.0 + 0.25, 2.0 + 0.25, -1.0 - 0.75, 2.0 - 0.75])
    p = G.predict_proba(m, X)
    assert p.shape == (4, 1)
    assert np.allclose(p[:, 0], 1.0 / (1.0 + np.exp(-raw)))
    assert p[:, 0].std() > 0.1                     # the old softmax over one column returned all ones
    m2 = G.from_compact(G.compact_export(m), 1)
    assert np.abs(G.predict_proba(m2, X) - p).max() < 0.02
