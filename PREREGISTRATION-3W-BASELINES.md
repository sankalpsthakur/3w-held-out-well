# Preregistration: 3W-HW v1 baselines (written before the first run that touches the sealed test wells)

Frozen by the git commit that adds this file. Protocol: protocol/PROTOCOL.md. Split: protocol/manifest.csv (sha256 a28c1ebd...).
Kernel entry: kernels/baselines.py (+ src/w3). Compute: private Kaggle CPU kernel `bench-events-3w-baselines`, internet on only to fetch the
pinned 3W commit; no GPU.

Features (333): per process channel (16 pressure/flow/temperature + 2 choke apertures) 17 causal window statistics over 300/900/3600 s; per valve state 3.
Absent channels are NaN. Forbidden inputs: timestamps, instance/well ids, class/state columns, rows after the decision.

Training sample per fold: eligible labelled decisions of the training wells at every 2nd decision, then at most 20,000 per class (seeded uniform).
Folds: dev0..dev4 (train on the other four dev folds, predict the held-out dev fold) and `test` (train on all 30 dev wells, predict the 10 test wells).

Models, fixed hyperparameters, no tuning on any fold:
- `rf`: sklearn RandomForestClassifier(n_estimators=100, min_samples_leaf=10, max_features="sqrt", max_samples=0.3, class_weight="balanced_subsample", random_state=0)
- `lgbm`: LightGBM multiclass (n_estimators=200, learning_rate=0.05, num_leaves=31, min_child_samples=50, subsample=0.7 x1, colsample_bytree=0.5, reg_lambda=1, class_weight="balanced", random_state=0)
- `simple`: windowed z-test of the last 300 s against rows r-3599..r-300 of the same instance, max over 16 channels (src/w3/features.py::simple_detector_score); no training
- `volatility`: max over channels of cv_w3 / q99 of training-normal cv_w3; fit on label-0 decisions of the training wells
Event score = 1 - P(class 0) (rf, lgbm), mapped score (simple, volatility). Class prediction = argmax (rf, lgbm only).

Threshold rule (one per model): lowest score threshold on the dev out-of-fold predictions with <= 1 false alarm per eligible normal day and <= 5 % normal-time
occupation (hysteresis 2/2), ties to max episode coverage (scorer.select_threshold). Applied unchanged to the test wells.

Reported: dev OOF (30 wells), sealed test (10 wells), 6-fold pooled over all 40 wells (valid only because nothing is tuned), each with 2,000-resample
well-cluster bootstrap intervals. Every test scoring is logged in TEST-LOG.md; a rerun after a bug fix is logged as a new row.
