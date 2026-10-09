# 3W held-out-well benchmark protocol

A benchmark protocol for event detection and classification on the real instances of the
Petrobras [3W dataset](https://github.com/petrobras/3W) 2.0.0, where the split is made by
**well**: 10 sealed test wells, 30 development wells in 5 grouped folds, a committed draw
seeded by a public [drand](https://drand.love) beacon round, a numpy scorer, and baselines
scored once on the sealed wells. The full write-up is [docs/3W-HW-WRITEUP.md](docs/3W-HW-WRITEUP.md).

## Why split by well

Same development data, same LightGBM model, three ways of making folds:

| folds made by | macro-F1 | event AUROC |
|---|---|---|
| random decisions | 0.991 | 1.000 |
| random instances | 0.739 | 0.962 |
| whole wells held out | 0.295 | 0.583 |

In the folds that 3W shipped until PR #185 (`dataset/folds/folds_clf_02.csv`), 13 of the 17
real wells appeared in more than one fold, four of them in all five. Carvalho (UFES, 2021,
Table 6) saw the same effect on 3W v1: 0.9928 with equal folds, 0.6434 +- 0.2326 with folds by well.

## Baselines on the sealed wells (scored once; 95 % well-bootstrap intervals)

| model | macro-F1 | event AUROC (pooled) |
|---|---|---|
| LightGBM (41,068 leaves) | 0.292 [0.202, 0.493] | 0.581 [0.308, 0.968] |
| random forest | 0.266 [0.145, 0.499] | 0.740 [0.536, 0.933] |
| simple change detector (training-free) | - | 0.47 |
| volatility detector (training-free) | - | 0.54 |
| TSPulse reconstruction, zero-shot | - | 0.49 |
| compact LightGBM selected on development wells (3,000 leaves, 28 KB int8) | 0.258 [0.164, 0.483] | 0.449 |

With ten test wells the intervals are wide (five of the nine event classes have a single test
well) and the baselines cannot be told apart. The protocol is meant to make that visible, not
to rank models. Nothing here beats the baselines; see the write-up for the within-well view
(post hoc, exploratory) and the full tables in `results/3w/`.

Disclosure: all 40 real wells were used in the author's earlier private experiments, so these
entries are organizer baselines, not independent results.

## What is here

| path | contents |
|---|---|
| `protocol/` | protocol text, commitments (`COMMIT-A.md`, `COMMIT-B.md`), the drand round, the well manifest, the label-derived index, instance table and decision grid, frozen configs |
| `src/w3/` | scorer, split, features, numpy LightGBM evaluator, pipeline |
| `tests/` | unit tests, including the scorer against a loop oracle |
| `kernels/` | the Kaggle kernel entry scripts that produced every result (payloads are rebuilt with `scripts/build_kernel.py`) |
| `scripts/` | tables, paired bootstraps, re-scoring, verification |
| `results/3w/` | baselines, dev study, final run, TSPulse run; `results/edge/` edge numbers |
| `PREREGISTRATION-3W-*.md`, `TEST-LOG.md` | what was fixed before each test scoring, and every scoring of the sealed wells |

The sensor data is not included: the kernels fetch the pinned 3W commit themselves.

## Verify

```
scripts/verify.sh
```

runs the unit tests, regenerates `protocol/manifest.csv` byte for byte from the committed rule and
the drand round (sha256 `a28c1ebd...`), and checks the frozen config hashes. With the saved
predictions (`results/3w/preds/`, not in this repository, available on request) it also re-scores
every result table locally.

## Licence

Code and text: Apache-2.0 (see `LICENSE`). The label-derived files under `protocol/` carry the
3W dataset's CC BY 4.0 licence with attribution (see `NOTICE`).

Sankalp Thakur, October 2026. Self-evaluated; not peer-reviewed.

## Edge numbers (Apple M4, one thread, batch 1, measured 2026-10-09 at a 1-minute load average under 4)

| model | leaves | int8 leaf export | model-only per decision (median) | features per decision from the 3,601-row buffer | end to end |
|---|---|---|---|---|---|
| compact LightGBM selected on development wells | 3,000 | 28 KB | 0.19 ms | 2.40 ms | 2.6 ms |
| binary event detector | 750 | 9 KB | 0.14 ms | 2.44 ms | 2.6 ms |
| baseline LightGBM | 41,068 | 359 KB | 0.61 ms | 2.47 ms | 3.1 ms |

Streaming buffer: 27 channels x 3,600 rows of float32 = 389 KB. Files: `results/edge/w3_*_edge.json` (the `_remeasured` baseline file replaces an earlier measurement taken on a loaded machine).
