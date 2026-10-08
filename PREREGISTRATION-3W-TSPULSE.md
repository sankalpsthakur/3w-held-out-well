# Preregistration: zero-shot TSPulse event detector on 3W-HW v1

Written 2026-10-07 (10:57 UTC; first committed as a09de46 with the time given as 11:05, corrected here before any kernel was pushed), before any TSPulse score of a 3W well exists. Frozen by the git commit that adds this file. Protocol: `protocol/PROTOCOL.md`; manifest sha256 `a28c1ebd...`.
Kernel entry: `kernels/tspulse_eval.py` (+ `src/w3`) at this commit; configuration `protocol/tspulse_config.json` (sha256 `db42d03b...`). Compute: private Kaggle CPU kernels (`bench-events-3w-tspulse-eval-smoke`, `bench-events-3w-tspulse-eval`),
internet on to fetch the pinned 3W commit, `granite-tsfm` and the public checkpoint; no GPU. Nothing is trained.

## 1. What had been seen before this file (disclosure)
- The sealed-test results of the supervised models are known: baselines (`TEST-LOG.md`, 03:52Z) and the dev-selected `best` / `best_binary` (10:45Z; `best_binary` event AUROC 0.449, below chance). This detector is tried because those supervised event detectors did not transfer across wells.
- No TSPulse output on any 3W well has been seen. The only TSPulse run so far is `bench-events-3w-tspulse-smoke` (synthetic random input, no 3W data): it showed that `granite-tsfm` 0.3.10 installs on the Kaggle image (torch 2.11.0, transformers 5.16.1), that the checkpoint loads
  (1,084,330 parameters, context 512, patch 8, RevIN scaling) and that a batch of 256 univariate windows takes 0.51 s on 4 threads.
- Every design constant below comes from the checkpoint's native shape (context 512, patch 8) and from that timing, not from looking at 3W scores. Nothing is tuned.
- As for all our entries: all 40 wells were used in the earlier private study and Hugging Face card; our results are organizer baselines.

## 2. The detector (event detection only: class > 0 against normal; no class output, so no macro-F1)
Checkpoint `ibm-granite/granite-timeseries-tspulse-r1` (Apache-2.0) at revision `2e64fcdc2a06d3565dfadaf0065c0ab5055f80f2`, `TSPulseForReconstruction`, one shared univariate model (`num_input_channels=1`, `mask_type="user"`), CPU, no fine-tuning.
For every 20th decision of every instance (every 600 s; decision k = 0, 20, 40, ...) and for each of the 18 continuous process channels separately:
1. Window: rows r-3583 .. r (3,584 rows, inside the protocol's 3,600-row lookback), block-averaged over 512 blocks of 7 rows (NaN-aware mean). The channel is valid at this decision if at least 90 % of the 512 block means are finite.
2. Normalisation with the past only: mean and standard deviation of the first 504 block means (the last 8 are excluded); standard deviation floored at 1e-6 (|mean| + 1); z = clip((x - mean) / sd, -20, 20); a missing block is set to 0.
3. Mask the last patch (the last 8 points, 56 s) as unobserved (`past_observed_mask`), let TSPulse reconstruct it, and take e = the mean squared error between the reconstruction and z over those 8 points. This is the channel's forecast-style error: how badly the model "imputes" the latest minute from the preceding hour.
4. Calibration (development wells only): q_c = the 99th percentile of e_c over the scored, eligible, label-0 decisions of the 30 development wells. No sealed well enters it.
5. Event value v = max over valid channels of e_c / q_c (0 if no channel is valid); score s = v / (v + 1), in [0, 1). The score of a scored decision is held for the following 19 decisions (decision k uses the latest scored decision <= k, so it is causal).
Threshold: `scorer.select_threshold` on the development scores (at most 1 false alarm per eligible normal day and 5 % normal-time occupation, hysteresis 2/2), frozen. Applied unchanged to the test wells.

Information only (logged on development wells, never used for the test): three alternative aggregations (mean of the top 3 channels, mean over channels, max without the q_c scaling). The primary is the max with q_c scaling above, whatever they show.

## 3. Procedure
1. Dev-only smoke kernel (`bench-events-3w-tspulse-eval-smoke`): 4 development wells as development, 2 development wells (fold 3) as the pseudo-test, stride 100; checks the code path, the timing and that no sealed well is loaded (asserted). An automated gate
   (`scripts/gate_tspulse.py`) must pass, including a projected full runtime of at most 100 minutes, before the full kernel is pushed under the same Kaggle slot.
2. Full kernel (`bench-events-3w-tspulse-eval`): phase 1 computes the errors of the 30 development wells, q_c, the development variants and theta; phase 2 computes the errors of the 10 sealed wells and scores them once (2,000-resample well-cluster bootstrap, seed 611).
   The launch and the outcome are logged in `TEST-LOG.md`; a rerun is allowed only after a crash and is a new row.

## 4. Reported and compared
Event AUROC, AP, episode coverage, onset detection within 300 s (resolution is limited by the 600 s stride), false alarms per normal day, normal-time occupation, each with well-cluster intervals, on development and on the sealed wells; per-well AUROC.
Paired well-bootstrap differences (same 2,000 resamples) against the two training-free baselines (`simple`, `volatility`), the baseline LightGBM and random forest, and `best` / `best_binary`; an interval containing 0 is no significant difference.
Compute cost: parameters, windows scored, seconds per 1,000 windows on Kaggle's CPU (the Mac is not used for model inference).
Claim wording, if any: "zero-shot TSPulse event detector on 3W-HW v1, sealed test of 10 wells, organizer baseline, self-evaluated, 2026-10-07".

## 5. Limits stated in advance
Univariate channels (no cross-channel structure), 7-second blocks (fast dynamics are averaged away), per-window normalisation (a slow level change that fills the whole window is invisible), 600-second scoring stride (coarse for onsets), no use of the valve states.
The expectation, written down now: event AUROC between 0.5 and 0.65 on the sealed wells, i.e. no improvement over the baselines; an honest negative is an acceptable outcome. Ten test wells make any single comparison noisy.
