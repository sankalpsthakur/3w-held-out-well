#!/bin/zsh
ROOT="$(cd "$(dirname "$0")/.." && pwd)"   # repository root
# Local post-processing of the final kernel's output (numpy only, every python call under the guard, one at a time):
# archive results, tables, paired well-bootstrap against the baselines, confusion matrix. usage: scripts/post_final.sh [kaggle_out/final]
cd $ROOT || exit 1
G=../bin/guard.sh; D="${1:-kaggle_out/final}"; R=$D/results.json
[ -f "$R" ] || { echo "no $R"; exit 1; }
cp "$R" results/3w/best_results.json
cp "$D"/preds_best.npz "$D"/preds_best_binary.npz results/3w/preds/ 2>/dev/null
$G -- python3 scripts/make_tables.py results/3w/best_results.json results/3w/best_tables.md > /dev/null && echo "tables ok"
B=results/3w/baselines_results.json
for a in best best_binary; do
  for b in lgbm rf; do
    $G -- python3 scripts/paired_3w.py results/3w/paired_${a}_vs_${b}.json ${a}=results/3w/best_results.json:${a}:results/3w/preds/preds_${a}.npz ${b}=${B}:${b}:results/3w/preds/preds_${b}.npz | tee -a results/3w/paired_summary.txt
  done
done
$G -- python3 scripts/conf_3w.py results/3w/best_confusion_test.md best results/3w/best_results.json:best:results/3w/preds/preds_best.npz | tail -14
$G -- python3 scripts/conf_3w.py results/3w/lgbm_confusion_test.md lgbm ${B}:lgbm:results/3w/preds/preds_lgbm.npz > /dev/null && echo "lgbm confusion ok"
