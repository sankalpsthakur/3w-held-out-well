#!/bin/zsh
ROOT="$(cd "$(dirname "$0")/.." && pwd)"   # repository root
# usage: run_chain.sh <smoke dir> <smoke owner/slug> <smoke outdir> <final dir> <final owner/slug> <final outdir>      env: SLOT_POLL (default 10 s) POLL (default 120 s) EXPECT_SMOKE_MIN EXPECT_FINAL_MIN
# Holds ONE locks/kaggle-cpu-N slot across smoke -> automated gate -> final, so the slot cannot be taken by another track in between.
# The gate (scripts/gate_smoke.py) must pass before the final kernel is pushed; otherwise the slot is released and the exit code is 2.
# The slot's `owner` file always reads "events <ref> <UTC> <state>".
sd="$1"; ss="$2"; so="$3"; fd="$4"; fs="$5"; fo="$6"
poll="${POLL:-120}"; L=${LOCKS:-$ROOT/../locks}; E=$ROOT
slot=""
while [ -z "$slot" ]; do
  for n in 1 2 3 4; do
    if mkdir "$L/kaggle-cpu-$n" 2>/dev/null; then slot="$L/kaggle-cpu-$n"; break; fi
  done
  [ -z "$slot" ] && sleep "${SLOT_POLL:-10}"
done
trap 'rm -rf "$slot"; echo "released $slot $(date -u +%T)"' EXIT
echo "$(date -u +%T) took $slot"

push() {   # dir slug expect_min : retries only on the session cap
  local tries=0 res
  echo "events $2 $(date -u +%FT%TZ) pushing" > "$slot/owner"
  while true; do
    res=$(kaggle kernels push -p "$1" 2>&1); echo "$res"
    if echo "$res" | grep -q "successfully pushed"; then break; fi
    if echo "$res" | grep -q "Maximum batch CPU session count"; then tries=$((tries + 1)); [ $tries -gt 120 ] && return 1; sleep 60; continue; fi
    return 1
  done
  echo "events $2 $(date -u +%FT%TZ) running, expected end about $(date -u -v+${3}M +%FT%TZ)" > "$slot/owner"
}

push "$sd" "$ss" "${EXPECT_SMOKE_MIN:-15}" || { echo "smoke push failed"; exit 1; }
$E/scripts/watch_kernel.sh "$ss" "$so" - "$poll"
$E/../bin/guard.sh -t 60 -- python3 $E/${GATE:-scripts/gate_smoke.py} "$so" || { echo "GATE FAILED: not pushing the final kernel"; exit 2; }
echo "gate passed $(date -u +%T): pushing the final kernel under the same slot"
push "$fd" "$fs" "${EXPECT_FINAL_MIN:-45}" || { echo "final push failed"; exit 1; }
POLL_FINAL="${POLL_FINAL:-300}"
$E/scripts/watch_kernel.sh "$fs" "$fo" - "$POLL_FINAL"
