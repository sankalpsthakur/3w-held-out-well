#!/bin/zsh
# Verification of this kit from a clean checkout. Numpy only, no model is run; every python call goes through bin/guard.sh, one at a time.
# usage: scripts/verify_local.sh      (about 30 s)   Prints PASS or FAIL per check and exits 1 if any fails.
cd "$(dirname "$0")/.." || exit 1
G=${GUARD:-}; fail=0; T=$(mktemp -d)
chk() { if [ "$1" = 0 ]; then echo "PASS  $2"; else echo "FAIL  $2"; fail=1; fi; }
$G -t 120 -- python3 -m pytest -q tests > $T/pytest.txt 2>&1; chk $? "unit tests ($(tail -1 $T/pytest.txt))"
SIG=$($G -- python3 -c "import json;print(json.load(open('protocol/drand_round.json'))['signature'])")
$G -- python3 src/w3/manifest.py --index protocol/index_labels.json --beacon-signature-hex $SIG --out $T/manifest.csv > /dev/null 2>&1 && cmp -s $T/manifest.csv protocol/manifest.csv; chk $? "well split regenerates byte for byte from the drand round"
[ "$(shasum -a 256 protocol/manifest.csv | cut -c1-64)" = "a28c1ebddd698806a0f293b3307b4709f251dec086c618513254892a2dcfc206" ]; chk $? "manifest sha256 a28c1ebd..."
[ "$(shasum -a 256 protocol/best_config.json | cut -c1-8)" = "131ce334" ] && [ "$(shasum -a 256 protocol/tspulse_config.json | cut -c1-8)" = "db42d03b" ]; chk $? "frozen configs unchanged (131ce334..., db42d03b...)"
if [ -f results/3w/preds/preds_lgbm.npz ]; then
  (cd results/3w/preds && shasum -a 256 -c SHA256SUMS > $T/sums.txt 2>&1); chk $? "saved predictions match SHA256SUMS ($(grep -c OK $T/sums.txt) files)"
  $G -- python3 scripts/rescore_3w.py results/3w/baselines_results.json results/3w/preds > $T/r1.txt 2>&1; chk $? "baselines: local re-score equals the kernel numbers ($(grep -c OK $T/r1.txt) checks)"
  $G -- python3 scripts/rescore_3w.py results/3w/best_results.json results/3w/preds best best_binary > $T/r2.txt 2>&1; chk $? "dev-selected models: local re-score equals the kernel numbers ($(grep -c OK $T/r2.txt) checks)"
  $G -- python3 scripts/rescore_3w.py results/3w/tspulse_results.json results/3w/preds tspulse > $T/r3.txt 2>&1; chk $? "TSPulse: local re-score equals the kernel numbers ($(grep -c OK $T/r3.txt) checks)"
else echo "SKIP  saved predictions are not in this checkout (results/3w/preds is gitignored): re-score checks skipped"; fi
