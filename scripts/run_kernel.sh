#!/bin/zsh
ROOT="$(cd "$(dirname "$0")/.." && pwd)"   # repository root
# usage: run_kernel.sh <kernel dir> <owner/slug> <outdir> [file-pattern]      env: POLL=<seconds, default 300> EXPECT_MIN=<minutes, default 30>
# Takes a free locks/kaggle-cpu-N slot (waits if none), pushes (retries ONLY on the session-cap message; any other push failure
# releases the slot and exits 1), polls the kernel status, downloads the output (optionally only files matching the regex), releases the slot.
# The slot's `owner` file always reads "events <ref> <UTC start> <state>" so the lead can see what holds it.
dir="$1"; slug="$2"; out="$3"; pat="${4:-}"; poll="${POLL:-300}"; exp="${EXPECT_MIN:-30}"
L=${LOCKS:-$ROOT/../locks}
slot=""
while [ -z "$slot" ]; do
  for n in 1 2 3 4; do
    if mkdir "$L/kaggle-cpu-$n" 2>/dev/null; then slot="$L/kaggle-cpu-$n"; break; fi
  done
  [ -z "$slot" ] && sleep "${SLOT_POLL:-60}"
done
t0=$(date -u +%FT%TZ)
echo "events $slug $t0 pushing" > "$slot/owner"
trap 'rm -rf "$slot"; echo "released $slot"' EXIT
echo "$(date -u +%T) took $slot"
tries=0
while true; do
  res=$(kaggle kernels push -p "$dir" 2>&1); echo "$res"
  if echo "$res" | grep -q "successfully pushed"; then break; fi
  if echo "$res" | grep -q "Maximum batch CPU session count"; then
    tries=$((tries + 1)); [ $tries -gt 120 ] && { echo "gave up waiting for a session"; exit 1; }
    sleep 60; continue
  fi
  echo "push failed (not the session cap): releasing the slot"; exit 1
done
echo "events $slug $t0 running, expected end about $(date -u -v+${exp}M +%FT%TZ)" > "$slot/owner"
$ROOT/scripts/watch_kernel.sh "$slug" "$out" - "$poll" "$pat"
