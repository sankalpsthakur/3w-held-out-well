#!/bin/zsh
ROOT="$(cd "$(dirname "$0")/.." && pwd)"   # repository root
# Waits (up to MAX_MIN minutes, default 150) until the 1-minute load average is under 4, then measures the numpy reference latency (1 thread, batch 1; models under 50 MB) of the baseline LightGBM and the two
# dev-selected models through bin/guard.sh -t 60 -m 512, one at a time. Writes results/edge/w3_*_edge.json. If the load never drops, nothing is run ("latency deferred (machine busy)").
cd $ROOT || exit 1
G=../bin/guard.sh; MAX=${MAX_MIN:-150}; t0=$SECONDS
while true; do
  load=$(sysctl -n vm.loadavg | awk '{print $2}')
  if [ "$(echo "$load < 4" | bc -l)" = 1 ]; then break; fi
  [ $(( (SECONDS - t0) / 60 )) -ge "$MAX" ] && { echo "$(date -u +%T) load still $load after $MAX min: latency deferred (machine busy)"; exit 3; }
  sleep 60
done
echo "$(date -u +%T) load $load: measuring"
$G -t 60 -m 512 -- python3 scripts/edge_w3.py latency lgbm_baseline_remeasured kaggle_out/baselines/lgbm_final.txt kaggle_out/baselines/probe_X.npy > /dev/null && echo "baseline ok"
$G -t 60 -m 512 -- python3 scripts/edge_w3.py latency best kaggle_out/final/best_final.txt kaggle_out/final/best_probe_X.npy results/3w/best_results.json best > /dev/null && echo "best ok"
$G -t 60 -m 512 -- python3 scripts/edge_w3.py latency best_binary kaggle_out/final/best_binary_final.txt kaggle_out/final/best_binary_probe_X.npy results/3w/best_results.json best_binary > /dev/null && echo "best_binary ok"
echo "$(date -u +%T) load after: $(sysctl -n vm.loadavg)"
