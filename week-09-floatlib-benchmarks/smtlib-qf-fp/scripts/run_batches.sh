#!/bin/bash
# run_batches.sh DIR [PROCS]: elaborate every DIR/*.lean that has no .rc file yet, PROCS at a time (8 at most),
# under `lake env` of this week's Lake project, where `lean` sees FloatLib. Nothing is written outside DIR.
set -u
dir=$(realpath "$1"); procs=${2:-8}
(( procs > 8 )) && procs=8
here=$(cd "$(dirname "$0")" && pwd); suite=$(dirname "$here"); week=$(dirname "$suite")
cd "$week" || exit 2
start=$(date +%s.%N)
for f in "$dir"/*.lean; do [ -f "$f.rc" ] || printf '%s\0' "$f"; done |
  lake env xargs -0 -r -P "$procs" -n 1 "$here/run_one.sh"
end=$(date +%s.%N)
echo "run_batches $dir procs=$procs wall=$(echo "$end - $start" | bc) files=$(ls "$dir"/*.lean | wc -l) rc0=$(grep -lx 0 "$dir"/*.rc 2>/dev/null | wc -l)"
