#!/bin/zsh
# usage: watch_kernel.sh <owner/slug> <outdir> <lockdir or -> [poll seconds, default 300] [file-pattern regex]
# Polls until the kernel ends (COMPLETE / ERROR / CANCEL), downloads the output (log always), releases the lock if one is given.
slug="$1"; out="$2"; lock="$3"; poll="${4:-300}"; pat="${5:-}"
mkdir -p "$out"
while true; do
  st=$(kaggle kernels status "$slug" 2>&1 | tail -1)
  echo "$(date -u +%T) $st"
  if echo "$st" | grep -q -E "COMPLETE|ERROR|CANCEL"; then break; fi
  sleep "$poll"
done
if [ -n "$pat" ]; then
  kaggle kernels output "$slug" -p "$out" --file-pattern "$pat" 2>&1 | tail -5
else
  kaggle kernels output "$slug" -p "$out" 2>&1 | tail -5
fi
[ "$lock" != "-" ] && rm -rf "$lock" && echo "released $lock"
echo "$(date -u +%T) finished: $st"
