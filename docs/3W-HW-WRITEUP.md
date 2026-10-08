# 3W-HW v1: a strict held-out-well protocol for Petrobras 3W Dataset 2.0, with a scorer and baselines

Organizer baselines, self-evaluated, 2026-10-07. Not an official 3W benchmark. Nothing here has been published or submitted.
Data: Petrobras 3W Dataset 2.0.0 (CC BY 4.0, doi 10.1038/s41597-026-07225-z) at `petrobras/3W` commit `d57c058af6baf65f1b285f3e56f46e4f018a07d9`. Code and protocol files: written for this study; the licence is the owner's choice (upstream `petrobras/3W` code is Apache-2.0).

## 1. Why a held-out-well protocol

The 3W repository removed its benchmark folds (`dataset/folds`, `problems/`) in PR #185, and the maintainers wrote in issues #70 and #112 that new benchmarks need to say which instances go to training and
testing and need baselines. The folds that were removed split by instance. Instances of one well share instrumentation, units, operating regime and often overlap in time, so an instance-level split leaks.

- The removed `folds_clf_02.csv` (commit `b9c92ae`, 166 rows, 150 real instances, 17 real wells): **13 of the 17 real wells appear in more than one fold**, four of them (00002, 00006, 00011, 00014) in all five.
  `results/3w/old_dhsv_folds_leakage.json`.
- Carvalho (UFES 2021, Table 6; flow instability, 3W v1, 5 wells): random forest macro-F1 0.9928 with equal folds against 0.6434 +- 0.2326 with folds by well.
- This study, same data and same model, three ways of splitting the 30 development wells (LightGBM, 5 folds; macro-F1 over the 8 classes that have at least 2 development wells; `results/3w/dev_study.json`):

| split of the development data | macro-F1 | event AUROC |
|---|---|---|
| random decisions | 0.991 | 1.000 |
| random instances | 0.739 | 0.962 |
| grouped by well (this protocol) | 0.295 | 0.583 |

The random-decision split doubles as the sanity check that features and labels are aligned (about 0.99 is what an aligned pipeline gives). The gap between the first and the last row is what a split that is not grouped by well adds through leakage.

## 2. The protocol in short (`protocol/PROTOCOL.md` is authoritative)

- Scope: the 1,119 real instances (`WELL-*`), 40 wells, 32,872,250 rows at 1 Hz. Simulated and hand-drawn instances are not used. Labels decide, not folder names (43 of 57 instances in folder 9 contain no event label).
- Decisions on a fixed grid, every 30 s from row 3,600 (962,002 decisions; 961,578 eligible, i.e. at least 90 % of the previous 900 rows have a finite process channel). A model may use only the previous 3,600 rows of the same
  instance: causal, no timestamps, no well or instance ids, no state/class columns, nothing from other instances of a test well.
- Split: 10 sealed test wells and 30 development wells in 5 grouped folds, drawn by a public rule (`src/w3/manifest.py`, constraints on class coverage) from a drand quicknet round that was committed in git before the manifest existed
  (commit A: frozen parameters; commit B: round 32846926, manifest sha256 `a28c1ebd...`). Test wells: 00001, 00004, 00011, 00013, 00015, 00023, 00026, 00035, 00037, 00041.
- Metrics (`src/w3/scorer.py`, checked against a loop oracle in `tests/`): window classification (confusion matrix, macro-F1, balanced accuracy, per-well macro-F1) and event detection with one frozen threshold chosen on development
  out-of-fold predictions (episode coverage, onset detection within 300 s, false alarms per normal day, normal-time occupation, AUROC, AP), all with 2,000-resample **well-cluster** bootstrap intervals and paired bootstrap for differences.
- Development rules: models, hyper-parameters and thresholds use development wells only; the sealed wells are scored once per frozen model and every scoring is logged (`TEST-LOG.md`).

Dataset facts (`results/3w/dataset_facts.json`; "wells" counts wells with at least one episode of 300 rows or more, written as development + test):

| class | name | wells (dev + test) | episodes | decisions | hours |
|---|---|---|---|---|---|
| 0 | normal | 40 (30 + 10) | | 486,026 | 4,050 |
| 1 | abrupt increase of BSW | 3 (2 + 1) | 4 | 2,505 | 20.9 |
| 2 | spurious closure of DHSV | 7 (5 + 2) | 22 | 3,680 | 30.7 |
| 3 | severe slugging | 2 (1 + 1) | 32 | 18,987 | 158.2 |
| 4 | flow instability | 7 (5 + 2) | 343 | 81,994 | 683.3 |
| 5 | rapid productivity loss | 3 (2 + 1) | 11 | 11,131 | 92.8 |
| 6 | quick restriction in PCK | 2 (1 + 1) | 6 | 661 | 5.5 |
| 7 | scaling in PCK | 6 (4 + 2) | 36 | 223,368 | 1,861.4 |
| 8 | hydrate in production line | 9 (8 + 1) | 14 | 126,544 | 1,054.5 |
| 9 | hydrate in service line | 6 (4 + 2) | 13 | 7,106 | 59.2 |

Five of the nine event classes have a single well in the test set (classes 1, 3, 5, 6, 8), so per-class numbers on the sealed test are anecdotes and the intervals are wide. Development-side macro-F1 uses the 8 classes
with at least 2 development wells (0, 1, 2, 4, 5, 7, 8, 9).

## 3. Baselines (preregistered, one run on the sealed test; `PREREGISTRATION-3W-BASELINES.md`, `TEST-LOG.md`)

Fixed hyper-parameters, nothing tuned on any fold: LightGBM (200 trees x 10 classes), random forest (100 trees), a windowed z-test detector (no training) and a volatility detector (fit on normal decisions only).
Features: 333 causal window statistics per decision. Thresholds: lowest score meeting at most 1 false alarm per normal day and 5 % normal-time occupation on development out-of-fold predictions.

Sealed test, 10 wells (95 % well-cluster bootstrap intervals):

| model | macro-F1 | AUROC | AP | episode coverage | onset <= 300 s | false alarms / normal day |
|---|---|---|---|---|---|---|
| simple z-test | - | 0.472 [0.369, 0.765] | 0.447 [0.217, 0.758] | 20.9 % | 50.0 % | 1.06 |
| volatility | - | 0.537 [0.328, 0.833] | 0.563 [0.237, 0.817] | 19.1 % | 16.7 % | 0.41 |
| random forest | 0.266 [0.145, 0.499] | 0.740 [0.536, 0.933] | 0.733 [0.632, 0.833] | 30.9 % | 33.3 % | 2.94 |
| LightGBM | 0.292 [0.202, 0.493] | 0.581 [0.308, 0.968] | 0.626 [0.356, 0.918] | 60.0 % | 40.0 % | 0.39 |

LightGBM minus random forest on the same wells: macro-F1 +0.026 [-0.037, +0.139], AUROC -0.159 [-0.334, +0.031]; every interval contains 0 (`scripts/paired_3w.py`). With ten wells the two baselines cannot be told apart.
Development out-of-fold (30 wells, optimistic for anything tuned): LightGBM macro-F1 0.343 [0.232, 0.421], random forest 0.278 [0.207, 0.411]. Full tables: `results/3w/baselines_tables.md`.
Edge numbers of the baseline LightGBM (`results/edge/w3_lgbm_baseline_edge.json`): 2,000 trees, 41,068 leaves, 4.9 MB as text, 359 KB as int8-leaf export (argmax agreement with the float model 99.9 %), 4.3 ms per decision in the numpy reference
evaluator at 1 thread, batch 1, plus 15 ms to compute the features for one decision from a 3,601-row buffer (numpy/Python reference implementations on this M4; embedded C would be far faster).

## 4. A dev-selected small LightGBM (preregistered in `PREREGISTRATION-3W-BEST.md`; one scoring of the sealed test)

Question: does a small model whose configuration was picked on the 30 development wells only beat the fixed baselines on the 10 sealed wells? Answer: no.

Selection (grouped 5-fold out-of-fold predictions on the development wells; 6 candidates x 4 round counts x 3 smoothing factors = 72 rows; best row per candidate, `results/3w/dev_study.json`):

| candidate | what it is | best dev macro-F1 or AP (rounds, EMA alpha) | dev AUROC |
|---|---|---|---|
| `all` | 333 features | macro-F1 0.337 (20, 0.25) | 0.614 |
| `sf` | scale-free subset (no absolute levels) | 0.242 (50, 0.25) | 0.694 |
| `all_aug` | 333 features, channel-dropout augmentation | **0.360 (20, 1.0)** | 0.680 |
| `sf_aug` | scale-free + augmentation | 0.298 (100, 0.25) | 0.488 |
| `bin_all` | binary event detector, 333 features | **AP 0.824 (50, 0.25)** | 0.818 |
| `bin_sf_aug` | binary, scale-free + augmentation | AP 0.766 (20, 0.25) | 0.746 |

Frozen before the test was touched: `best` = `all_aug`, 20 rounds x 10 classes x 15 leaves, no smoothing; `best_binary` = `bin_all`, 50 rounds x 15 leaves, EMA 0.25. The kernel reproduced the development
table exactly before it predicted any test row (dev macro-F1 0.3603 and AP 0.8241, as asserted). Seeds 0, 1, 2; seed 0 is the preregistered primary result.

Sealed test, 10 wells (seed 0; intervals are 95 % well-cluster bootstrap; baselines from section 3):

| model | macro-F1 | event AUROC | AP | episode coverage | onset <= 300 s | false alarms / normal day |
|---|---|---|---|---|---|---|
| `best` (200 trees, 3,000 leaves) | 0.258 [0.164, 0.483] | 0.751 [0.504, 0.959] | 0.726 [0.570, 0.905] | 65.5 % [29.4, 87.8] | 40.0 % [0.0, 67.6] | 0.55 [0.00, 3.13] |
| `best`, int8-leaf export (28 KB) | 0.258 [0.165, 0.484] | 0.752 [0.504, 0.959] | 0.726 | 65.5 % | 40.0 % | 0.53 |
| `best_binary` (50 trees, 750 leaves) | - | 0.449 [0.236, 0.913] | 0.423 [0.249, 0.796] | 5.5 % [1.6, 32.3] | 0.0 % | 0.18 [0.00, 1.16] |
| baseline LightGBM (2,000 trees) | 0.292 [0.202, 0.493] | 0.581 [0.308, 0.968] | 0.626 [0.356, 0.918] | 60.0 % | 40.0 % | 0.39 |
| baseline random forest | 0.266 [0.145, 0.499] | 0.740 [0.536, 0.933] | 0.733 [0.632, 0.833] | 30.9 % | 33.3 % | 2.94 |

Paired differences on the same 10 wells (same 2,000 resamples; model minus baseline; `results/3w/paired_*.json`):

| | vs baseline LightGBM | vs baseline random forest |
|---|---|---|
| `best` macro-F1 | -0.034 [-0.139, +0.071] | -0.008 [-0.107, +0.116] |
| `best` event AUROC | +0.170 [-0.006, +0.325] | +0.011 [-0.083, +0.203] |
| `best_binary` event AUROC | -0.133 [-0.335, -0.001] | -0.292 [-0.433, -0.004] |

What this shows:
- **No improvement.** For `best` every interval contains 0 and the macro-F1 point estimate is below both baselines. The dedicated event detector, the best candidate on development by a wide margin (AUROC 0.818 against 0.58 for the baseline LightGBM and 0.64 for the random forest on development), scores below chance on the sealed wells in the pooled metric (0.449; the within-well view below is milder), and its paired AUROC difference against both baselines just excludes 0. The training-free volatility detector shows the same kind of drop (dev AUROC 0.799, sealed 0.537).
  (`results/3w/paired_summary.txt` holds 8 metrics x 4 comparisons; read single intervals that just exclude 0 with that in mind.)
- **The development advantage did not transfer.** `best`: dev macro-F1 0.360 -> sealed 0.258. `best_binary`: dev AUROC 0.818 -> sealed 0.449. Three seeds, same pipeline: dev macro-F1 0.360 / 0.398 / 0.411 (sd 0.026), sealed 0.258 / 0.221 / 0.233 (sd 0.019);
  `best_binary` dev AUROC 0.818 / 0.747 / 0.826, sealed 0.449 / 0.532 / 0.511. The seed-to-seed spread on development is as large as the 0.024 by which `all_aug` beat the runner-up `all`, so the ranking of candidates on 30 wells is within noise (a retrospective observation, not a selection).
- **Pooled versus within-well event AUROC (post hoc, not preregistered).** The pooled AUROC mixes separation inside a well with the offset between wells' score levels. A descriptive look taken after the sealed results were known (`scripts/perwell_summary.py`, `results/3w/perwell_auroc_summary.md`: unweighted mean of the per-well AUROC over the wells that have both classes, 24 development and 9 sealed wells, no intervals;
  well 00013 with 219 event decisions counts as much as well 00023 with 45,419) gives these point estimates, development -> sealed: `best` 0.74 -> 0.76, baseline LightGBM 0.75 -> 0.76, random forest 0.75 -> 0.72, volatility 0.74 -> 0.76 (z-test 0.66 -> 0.69), `best_binary` 0.77 -> 0.66 (median 0.56, above 0.5 on 6 of 9 wells) and, in section 5, TSPulse 0.46 -> 0.38.
  A large part of the pooled drop of `best_binary` (0.818 -> 0.449) is not visible within wells. These numbers are exploratory: with 9 wells and no intervals they support neither a ranking nor a parity claim. Version 1 reports the pooled metric (it is frozen); a within-well AUROC with a well-cluster interval is a candidate secondary metric for a later version.
- **Thresholds do not transfer either.** Chosen for at most 1 false alarm per normal day on development, the alarm threshold gave 0.55, 6.24 and 3.07 false alarms per normal day on the sealed wells for seeds 0, 1, 2.
- **A few wells decide the pooled numbers.** Per-well event AUROC on the sealed wells (`results/3w/perwell_auroc_test.md`): well 00023 (45,419 event decisions of class 7, 45 % of all test event decisions) is 0.41 for `best`, 0.39 for `best_binary`, 0.43 for the baseline LightGBM and
  0.39 for the random forest, i.e. every model ranks that well's scaling events below its own normal operation; well 00026 (29,963 event decisions, class 8) is 0.52 to 0.67. Wells 00011 and 00013 are 0.91 to 1.00 for every model. `best_binary` falls to 0.56 on well 00001 (`best`: 0.93) and 0.49 on 00041 (`best`: 0.83).
- **Classes.** Per-class F1 on the sealed wells, `best` / baseline LightGBM (`results/3w/best_confusion_test.md`, `lgbm_confusion_test.md`): normal 0.70 / 0.74, class 1 0.12 / 0.17, class 2 0.88 / 0.93, class 4 0.00 / 0.64, class 8 0.63 / 0.42, class 9 0.25 / 0.03,
  and 0 / 0 for classes 3, 5, 6 and 7. The macro-F1 gap of -0.034 is one class: flow instability (class 4, two test wells, 19,009 decisions) is called class 5 or 7 by `best` and is found by the baseline, while `best` finds more of classes 8 and 9.
  Class 7 (46,729 decisions, 97 % of them predicted as normal by `best`) is missed by every model.
- Context: the leakage study's by-well development macro-F1 for the baseline model (0.295) and the sealed-test macro-F1 of every model here (0.22 to 0.29) are of the same size; what is left of the 0.99 that a random split reports is about 0.3.

## 5. A zero-shot foundation-model baseline: TSPulse (preregistered in `PREREGISTRATION-3W-TSPULSE.md`; one scoring)

Question: do the supervised event detectors fail across wells because they overfit, and would a pretrained time-series model that never saw 3W do better? Answer: no, it is at chance.

What was run (event detection only, no class output, no training): IBM Granite TSPulse r1 (1.08 M parameters, Apache-2.0, revision `2e64fcdc2a06d3565dfadaf0065c0ab5055f80f2`). At every 20th decision (every 600 s) and for each process channel separately, the last 3,584 rows are
block-averaged to 512 points, z-normalised with the window's own past, the last patch (56 s) is masked and TSPulse reconstructs it; the reconstruction error of that minute, scaled by the development-normal 99th percentile of the channel and combined with a max over channels, is the event
score (held between scored decisions, so causal). The alarm threshold comes from the 30 development wells like for every other model. Cost: 360,379 model calls, 444 s of model time in a private Kaggle CPU kernel (551 s in all). Nothing ran on the Mac.

Sealed test, 10 wells (95 % well-cluster bootstrap intervals; development in brackets):

| | event AUROC | AP | episode coverage | onset <= 300 s | false alarms / normal day |
|---|---|---|---|---|---|
| TSPulse zero-shot | 0.493 [0.346, 0.565] (dev 0.690 [0.484, 0.792]) | 0.441 [0.127, 0.751] | 14.5 % [4.0, 56.4] | 26.7 % [0.0, 45.3] | 0.96 [0.23, 1.51] |

Paired AUROC differences, TSPulse minus the other detector on the same wells (`results/3w/tspulse_paired_summary.txt`): random forest -0.248 [-0.535, -0.057], `best` -0.259 [-0.590, -0.041] (both exclude 0); baseline LightGBM -0.089 [-0.485, +0.162], volatility -0.045 [-0.357, +0.182],
z-test +0.021 [-0.399, +0.174], `best_binary` +0.044 [-0.415, +0.227] (all contain 0).
Per well (`results/3w/perwell_auroc_test_all.md`), TSPulse's AUROC is below 0.5 on 6 of the 9 test wells that have events (00013 0.04, 00015 0.24, 00001 0.27, 00011 0.34, 00026 0.46, 00004 0.47): on those wells the model reconstructs the last minute of normal operation worse than the last minute of an event.
Within wells it is no better (post hoc, unweighted, no intervals: mean per-well AUROC 0.46 on 24 development wells, 0.38 on 9 sealed wells; section 4). Reading: a reconstruction error of the latest minute, computed per channel without supervision, does not separate 3W events from normal operation. The preregistered expectation was 0.5 to 0.65; the result is slightly below it. Development AUROC 0.69 falling to 0.49 on the sealed wells is the same
development-to-test drop as for the supervised detectors. Three alternative aggregations were logged on development only (mean of the top 3 channels 0.701, mean over channels 0.677, max without scaling 0.689) and not used.

## 6. Edge numbers (this M4; 1 thread, batch 1 where measured)

| model | trees | leaves | features split on | LightGBM text | int8-leaf export (raw / compressed) | numpy evaluator vs LightGBM | int8 vs float argmax | latency |
|---|---|---|---|---|---|---|---|---|
| baseline LightGBM | 2,000 | 41,068 | - | 4,884,154 B | 661,096 B / 359,310 B | 3e-8 (float32 probe) | 99.9 % | 4.3 ms model + 15.4 ms features, numpy reference |
| `best` | 200 | 3,000 | 232 | 368,577 B | 48,408 B / 28,245 B | 4e-16 (2,000 probes) | 100 % | latency deferred (machine busy: load average 7 to 8) |
| `best_binary` | 50 | 750 | 129 | 103,643 B | 12,108 B / 8,952 B | 0 (2,000 probes) | - | latency deferred (machine busy) |

`best` is about 13 times smaller than the baseline LightGBM (leaves, text and int8 export). With 10 sealed wells it cannot be told apart from that baseline, but that is not parity: the paired macro-F1 interval is [-0.139, +0.071], so a deficit of up to 0.14 is still allowed. This is neither a win nor a non-inferiority result; the size reduction is the only firm statement.
Probes are 2,000 decisions from development wells; the int8 export (per-tree scaled int8 leaf values) changes event AUROC by 0.0003 and macro-F1 by 0.0007 on the sealed wells.
A full decision also needs the 333 window features from a 3,601-row buffer (15 ms in the numpy reference; an embedded C implementation would be far faster).

## 7. Where the numbers come from

| claim | file |
|---|---|
| dev table, leak study | `results/3w/dev_study.json`, `dev_study_log.txt` |
| preregistration, frozen config | `PREREGISTRATION-3W-BEST.md`, `protocol/best_config.json` (sha256 `131ce334...`) |
| kernel run (code commit `5634314`, payload sha256 `3965a6f0...`, 705 s) | `kernels/final_model.py`, `results/3w/best_results.json`, `results/3w/best_kernel_log.json`, `TEST-LOG.md` |
| tables, seeds | `results/3w/best_tables.md` |
| paired intervals, confusion matrices, per-well AUROC | `results/3w/paired_*.json`, `paired_summary.txt`, `best_confusion_test.md`, `lgbm_confusion_test.md`, `perwell_auroc_test.md` |
| TSPulse | `PREREGISTRATION-3W-TSPULSE.md`, `protocol/tspulse_config.json` (sha256 `db42d03b...`), `kernels/tspulse_eval.py` (code commit `44490ce`, payload sha256 `4e03cf96...`), `results/3w/tspulse_results.json`, `tspulse_kernel_log.json`, `tspulse_paired_summary.txt`, `paired_tspulse_*.json`, `perwell_auroc_test_all.md` |
| model files (not in git) | the private Kaggle kernel `sankalpsthakur/bench-events-3w-final` output: `best_final.txt` (sha256 `57f2b325...`), `best_binary_final.txt` (`338ad8d8...`), int8 exports `best_compact.npz` (`7b8c1cac...`), `best_binary_compact.npz` (`9267b31e...`) |

## 8. What this does not show

- Ten test wells and one draw of wells. Intervals are wide and a margin inside them is not a win.
- Event AUROC and AP are pooled over wells in v1, so they depend on cross-well score calibration and on the longest wells (well 00023 holds 45 % of the sealed event decisions); see the post-hoc within-well numbers in section 4.
- All 40 real wells were used in an earlier private study and Hugging Face card (`sankalpsthakur/well-production-event-causal`). A new split seed does not remove that exposure for our own models, so our entries are organizer baselines,
  not independent results.
- Real wells only; simulated and hand-drawn instances, the transient/steady distinction and other 3W tasks (for example fault localisation) are not covered. One foundation-model baseline (zero-shot TSPulse, event detection only, section 5) was run; fine-tuned foundation models were not.

## 9. Reproduce

```
scripts/verify_local.sh                                                        # one command: tests, split regeneration, local re-score of every saved prediction (numpy only)
python3 -m pytest -q tests                                                     # scorer vs loop oracle, causal features, numpy LightGBM evaluator, SKAB detector
python3 src/w3/manifest.py --index protocol/index_labels.json --beacon-signature-hex $(python3 -c "import json;print(json.load(open('protocol/drand_round.json'))['signature'])") --out /tmp/manifest.csv
cmp /tmp/manifest.csv protocol/manifest.csv                                    # byte-identical
python3 scripts/rescore_3w.py results/3w/baselines_results.json results/3w/preds   # local re-score of saved predictions (numpy only)
scripts/post_final.sh                                                          # tables, paired well-bootstrap, confusion matrices from kaggle_out/final (numpy only, under bin/guard.sh)
python3 scripts/build_kernel.py bench-events-3w-final kernels/final_model.py --files protocol/decisions.npz protocol/instances.csv protocol/index_labels.json protocol/manifest.csv --config "$(cat protocol/best_config.json)"
kaggle kernels push -p kaggle/bench-events-3w-final                            # private CPU kernel, internet on to fetch the pinned 3W commit
```
Everything that trains a model ran in private Kaggle CPU kernels (4 cores); nothing heavy ran locally.

## 10. Files and licences

`protocol/` (frozen parameters, manifest, commits A and B, label index, decision grid), `src/w3/` (scorer, features, split, evaluator), `tests/`, `kernels/`, `results/3w/`, `PREREGISTRATION-3W-*.md`, `TEST-LOG.md`.
`protocol/index_labels.json` and `protocol/decisions.npz` are derived from 3W labels and carry its CC BY 4.0 attribution. No data and no model weights are in the repository; saved predictions (24 MB) are not tracked.
