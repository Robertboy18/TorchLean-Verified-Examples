#!/bin/bash
# final.sh OUT WORKERS TIMEOUT: the passes in order. Finished cases are not run again.
# Run it under `lake env` of this week's Lake project, after `gen.py`, `lifted.py` and `loops.py`.
set -u
here=$(cd "$(dirname "$0")" && pwd); suite=$(dirname "$here"); week=$(dirname "$suite")
work=$(realpath -m "${WORK:-$suite/work}")
floatlib=$(realpath "${FLOATLIB_DIR:-$week/.lake/packages/floatlib}")
out=$1; workers=$2; to=$3
"$here/passes.sh" "$out" "$workers" "$to" plan.json plan-lifted.json
python3 -I "$here/closure.py" "$work" first
"$here/passes.sh" "$out" "$workers" "$to" plan-closure-first.json:ProofsNF.lean
python3 -I "$here/closure.py" "$work" rest
"$here/passes.sh" "$out" "$workers" "$to" plan-closure-rest.json:ProofsNF.lean plan-loops.json:ProofsNF.lean
