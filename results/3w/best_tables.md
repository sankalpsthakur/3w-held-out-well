### Sealed test (10 wells)

| model | macro-F1 | bal. acc | AUROC | AP | episode coverage | onset <=300 s | FA / normal day | normal occupation | theta |
|---|---|---|---|---|---|---|---|---|---|
| best | 0.258 [0.164, 0.483] | 0.349 [0.246, 0.546] | 0.751 [0.504, 0.959] | 0.726 [0.570, 0.905] | 65.5% [29.4, 87.8] | 40.0% [0.0, 67.6] | 0.55 [0.00, 3.13] | 0.3% [0.0, 2.0] | 0.963 |
| best_compact | 0.258 [0.165, 0.484] | 0.350 [0.246, 0.548] | 0.752 [0.504, 0.959] | 0.726 [0.570, 0.905] | 65.5% [29.4, 87.8] | 40.0% [0.0, 67.6] | 0.53 [0.00, 2.97] | 0.3% [0.0, 1.7] | 0.963 |
| best_binary | - | - | 0.449 [0.236, 0.913] | 0.423 [0.249, 0.796] | 5.5% [1.6, 32.3] | 0.0% [0.0, 0.0] | 0.18 [0.00, 1.16] | 0.5% [0.0, 2.9] | 0.749 |
| best_binary_compact | - | - | 0.448 [0.236, 0.912] | 0.423 [0.249, 0.796] | 5.5% [1.6, 32.3] | 0.0% [0.0, 0.0] | 0.18 [0.00, 1.16] | 0.5% [0.0, 2.9] | 0.749 |

Denominators: 10 wells, 100612 event and 125525 normal eligible decisions, 43.6 normal days, 110 episodes (0 unobservable).

### Dev out-of-fold (30 wells; optimistic for tuned models)

| model | macro-F1 | bal. acc | AUROC | AP | episode coverage | onset <=300 s | FA / normal day | normal occupation | theta |
|---|---|---|---|---|---|---|---|---|---|
| best | 0.360 [0.240, 0.521] | 0.539 [0.370, 0.654] | 0.680 [0.464, 0.858] | 0.747 [0.537, 0.886] | 7.8% [2.1, 21.1] | 11.8% [2.8, 27.8] | 0.39 [0.03, 1.50] | 1.0% [0.0, 4.0] | 0.963 |
| best_binary | - | - | 0.818 [0.659, 0.893] | 0.824 [0.680, 0.920] | 22.6% [9.8, 55.9] | 3.9% [0.0, 11.1] | 0.85 [0.28, 2.43] | 2.5% [0.4, 8.4] | 0.749 |

Denominators: 30 wells, 375070 event and 360371 normal eligible decisions, 125.1 normal days, 371 episodes (0 unobservable).

### 6-fold pooled, all 40 wells (valid for untuned models only; not computed for the dev-selected models)

| model | macro-F1 | bal. acc | AUROC | AP | episode coverage | onset <=300 s | FA / normal day | normal occupation | theta |
|---|---|---|---|---|---|---|---|---|---|

### best: seeds (sealed test, point estimates; seed 0 is primary)

| seed | dev-OOF macro-F1 | dev-OOF AP | theta | test macro-F1 | test AUROC | test AP | episode coverage | FA / normal day |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.360 | 0.747 | 0.963 | 0.258 | 0.751 | 0.726 | 0.655 | 0.55 |
| 1 | 0.398 | 0.823 | 0.906 | 0.221 | 0.659 | 0.662 | 0.709 | 6.24 |
| 2 | 0.411 | 0.801 | 0.947 | 0.233 | 0.655 | 0.664 | 0.700 | 3.07 |

### best_binary: seeds (sealed test, point estimates; seed 0 is primary)

| seed | dev-OOF macro-F1 | dev-OOF AP | theta | test macro-F1 | test AUROC | test AP | episode coverage | FA / normal day |
|---|---|---|---|---|---|---|---|---|
| 0 | - | 0.824 | 0.749 | - | 0.449 | 0.423 | 0.055 | 0.18 |
| 1 | - | 0.789 | 0.836 | - | 0.532 | 0.508 | 0.036 | 0.09 |
| 2 | - | 0.841 | 0.679 | - | 0.511 | 0.505 | 0.155 | 0.11 |
