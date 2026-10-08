#!/bin/zsh
# usage: push_when_free.sh <kernel dir> ; retries while the account is at its concurrent batch-CPU-session cap
dir="$1"; n=${2:-90}
for i in $(seq 1 $n); do
  out=$(kaggle kernels push -p "$dir" 2>&1 | tail -2)
  echo "$(date -u +%T) try $i: $out"
  if echo "$out" | grep -q "successfully pushed"; then exit 0; fi
  if echo "$out" | grep -q "Maximum batch CPU session count"; then sleep 60; else sleep 20; fi
done
exit 1
