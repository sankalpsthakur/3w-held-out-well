# Preregistration: 3W-HW v1 dev-selected LightGBM ("best")

Written 2026-10-07 (about 10:20 UTC), before any model of this family is fitted on all 30 dev wells and before any sealed-test well is scored by it.
Frozen by the git commit that adds this file. Protocol: `protocol/PROTOCOL.md`; manifest sha256 `a28c1ebd...`; baselines: `PREREGISTRATION-3W-BASELINES.md`.
Kernel entry: `kernels/final_model.py` (+ `src/w3`) at this commit; configuration: `protocol/best_config.json` (sha256 in the commit message and in `TEST-LOG.md`).
Compute: private Kaggle CPU kernel `bench-events-3w-final`, internet on only to fetch the pinned 3W commit; no GPU.

## 1. What had been seen before this file (disclosure)
- The baselines (rf, lgbm, simple, volatility) were scored once on the sealed test wells (`TEST-LOG.md`, 03:52Z). Their sealed-test numbers (lgbm macro-F1 0.292, rf 0.266, ...)
  were known when the candidate families below were designed (scale-free features, channel-dropout augmentation, a binary event detector), and the low numbers motivated them.
  The choice among the candidates uses dev data only.
- Dev study: kernel `bench-events-3w-devstudy` (dev wells only, no sealed-test well loaded; payload sha256 `1e025e99...`; finished 04:56 UTC). Its output is committed as
  `results/3w/dev_study.json` (sha256 `3c10518b...`): 6 candidates x rounds {20, 50, 100, 200} x EMA alpha {1, 0.5, 0.25} = 72 rows, plus the leakage study.
- The multiclass selection rule was written in `STATUS.md` (commit `9f8ba54`) before the dev results were read. The binary rule below was written after the dev table had been seen
  (dev data only); it picks the same row under the obvious variants (AP or AUROC as the criterion, any alpha).
- All 40 real wells were used in the earlier private study and Hugging Face card (`sankalpsthakur/well-production-event-causal`). A new split seed does not remove that exposure for
  our own models. Our results here are organizer baselines.

## 2. Sanity check and leakage study (dev wells, the same LightGBM, three ways of splitting)
| split of the dev data | macro-F1 (8 classes with >= 2 dev wells) | AUROC |
|---|---|---|
| random decisions | 0.991 | 1.000 |
| random instances | 0.739 | 0.962 |
| grouped by well (the protocol) | 0.295 | 0.583 |

The sanity check passes (a random-decision split gives about 0.99, so features and labels are aligned). The gap is the leakage the protocol exists to remove.

## 3. Selection rule and its result (on `results/3w/dev_study.json` only)
Multiclass model `best`: the row with the highest dev out-of-fold macro-F1 over the classes with >= 2 dev wells (0, 1, 2, 4, 5, 7, 8, 9), ties by AP; then, inside the winning
candidate, the smallest round count whose macro-F1 is within 0.01 of that best; then the EMA alpha with the highest macro-F1 at that round count.
Result: candidate `all_aug` (all 333 features, channel-dropout augmentation p = 0.3), **20 rounds, no smoothing (alpha 1.0)**: dev macro-F1 0.3603, AUROC 0.680, AP 0.747.
The next candidate is `all` at 20 rounds (0.3366); no other (candidate, rounds) pair is within 0.01 of the winner. Note that 20 is the smallest round count in the grid and macro-F1
falls with more rounds (all_aug: 0.360, 0.305, 0.284, 0.287 at 20, 50, 100, 200), so the optimum may lie below the grid. The grid is not extended after the fact.

Binary event detector `best_binary`: the same rule with average precision as the criterion (macro-F1 is undefined for a binary model).
Result: candidate `bin_all` (all features, no augmentation), **50 rounds, EMA alpha 0.25**: dev AP 0.8241, AUROC 0.818. Round 20 (AP 0.809) is more than 0.01 below the best, so 50 stands.

Everything else is fixed and was not tuned: learning rate 0.1, 15 leaves, min_child_samples 50, subsample 0.7 every iteration, colsample_bytree 0.5, reg_lambda 1, class_weight balanced,
training sample = every 2nd eligible labelled decision capped at 20,000 per class (seeded uniform), n_jobs 4. Event score of `best` = 1 - P(class 0); of `best_binary` = P(event).

## 4. Procedure (`kernels/final_model.py`)
Phase 1, dev wells only. For each model and each seed in {0, 1, 2}: five well-grouped folds give out-of-fold predictions; the alarm threshold theta is `scorer.select_threshold` on that dev OOF
(<= 1 false alarm per eligible normal day, <= 5 % normal-time occupation, hysteresis 2/2), then frozen. Seed 0 uses the dev study's own sampling seeds (`"all_aug"+fold`, `"bin_all"+fold`) and
LightGBM `random_state` 0, so its dev OOF must reproduce the table: the kernel asserts |macro-F1 - 0.3603| < 0.01 (`best`) and |AP - 0.8241| < 0.01 (`best_binary`) **before any test row is
predicted**. If an assertion trips, nothing sealed has been scored; the cause is investigated, the configuration is not changed, and the rerun is logged. Seeds 1 and 2 change the sampling and the bagging seed.

Phase 2. Per model and seed: fit on all 30 dev wells with the same recipe; (seed 0 only) check the numpy tree evaluator and the int8-leaf export against LightGBM on 2,000 probe decisions from
**dev** wells and measure sizes; then predict the 10 sealed test wells once and score with `scorer.report` at the frozen theta (2,000-resample well-cluster bootstrap, seed 611).
The int8-leaf export of each seed-0 model is scored on the same test decisions with the same theta. **Seed 0 is the primary, preregistered result**; seeds 1 and 2 are a robustness check
(point estimates) and never replace it.

Logging: the sealed-test scoring of `best` and `best_binary` (3 seeds each, plus their two int8 variants) is one run of one frozen configuration and is logged as one row in `TEST-LOG.md`.
A rerun is allowed only after a crash and is a new row. The kernel prints `SEALED-TEST SCORING ... starts/done` lines with timestamps.

## 5. What is reported and compared
Per model, on dev OOF and on the sealed test: macro-F1 and balanced accuracy (`best`), AUROC, AP, episode coverage, onset detection within 300 s, false alarms per normal day, normal-time occupation,
with well-cluster bootstrap intervals; the seed table; the confusion matrix of `best`; per-class F1. Edge: parameters, bytes (LightGBM text, int8-leaf export), numpy-vs-LightGBM agreement, latency at 1 thread,
batch 1 on this M4 when the machine is quiet.
Paired comparison on the same 10 test wells (same 2,000 resamples, seed 611): `best` minus baseline lgbm and minus baseline rf on macro-F1, AUROC, AP and episode coverage; `best_binary` minus the
baselines on the event metrics. An interval that contains 0 is reported as no significant difference; a point difference smaller than its interval is not a win.
Claim wording, if any: "dev-selected LightGBM on 3W-HW v1, sealed test of 10 wells (one draw of wells), organizer baseline, self-evaluated, 2026-10-07". Never "best", "#1" or "SOTA".

## 6. Limits stated in advance
Ten test wells and one draw of wells: classes 1, 3, 5, 6 and 8 each have a single test well, so per-class numbers are close to anecdotes and the intervals are wide. The dev table has 72 rows and its best row
is subject to winner's curse (the dev macro-F1 of the selected model is optimistic; the sealed test is the unbiased number). Hyper-parameters outside the grid were not explored.
