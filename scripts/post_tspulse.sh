#!/bin/zsh
ROOT="$(cd "$(dirname "$0")/.." && pwd)"   # repository root
# Local post-processing of the TSPulse kernel output (numpy only, every python call under the guard): archive, paired well-bootstrap against the other event detectors, per-well AUROC.
# usage: scripts/post_tspulse.sh [kaggle_out/tspulse-eval]
cd $ROOT || exit 1
G=../bin/guard.sh; D="${1:-kaggle_out/tspulse-eval}"; R=$D/results.json
[ -f "$R" ] || { echo "no $R"; exit 1; }
cp "$R" results/3w/tspulse_results.json
cp "$D"/preds_tspulse.npz results/3w/preds/ 2>/dev/null
Bs=results/3w/baselines_results.json; Fb=results/3w/best_results.json; Tr=results/3w/tspulse_results.json; Pp=results/3w/preds
: > results/3w/tspulse_paired_summary.txt
for b in simple volatility rf lgbm; do
  $G -- python3 scripts/paired_3w.py results/3w/paired_tspulse_vs_${b}.json tspulse=${Tr}:.:${Pp}/preds_tspulse.npz ${b}=${Bs}:${b}:${Pp}/preds_${b}.npz | tee -a results/3w/tspulse_paired_summary.txt
done
for b in best best_binary; do
  $G -- python3 scripts/paired_3w.py results/3w/paired_tspulse_vs_${b}.json tspulse=${Tr}:.:${Pp}/preds_tspulse.npz ${b}=${Fb}:${b}:${Pp}/preds_${b}.npz | tee -a results/3w/tspulse_paired_summary.txt
done
$G -- python3 scripts/perwell_3w.py results/3w/perwell_auroc_test_all.md tspulse=${Tr}:.:${Pp}/preds_tspulse.npz best=${Fb}:best:${Pp}/preds_best.npz best_binary=${Fb}:best_binary:${Pp}/preds_best_binary.npz lgbm=${Bs}:lgbm:${Pp}/preds_lgbm.npz rf=${Bs}:rf:${Pp}/preds_rf.npz volatility=${Bs}:volatility:${Pp}/preds_volatility.npz simple=${Bs}:simple:${Pp}/preds_simple.npz
