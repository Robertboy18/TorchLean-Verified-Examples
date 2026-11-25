#!/bin/bash
# passes.sh OUT WORKERS TIMEOUT PLAN[:RUNNER]...: run plans of `WORK` one after another.
# The runner is FloatLib's own `benchmarks/suites/Proofs.lean` unless a plan names `ProofsNF.lean`.
# Run it under `lake env` of this week's Lake project: `(cd WEEK && lake env numfuzz/scripts/passes.sh ...)`.
set -u
here=$(cd "$(dirname "$0")" && pwd); suite=$(dirname "$here"); week=$(dirname "$suite")
work=$(realpath -m "${WORK:-$suite/work}")
floatlib=$(realpath "${FLOATLIB_DIR:-$week/.lake/packages/floatlib}")
out=$(realpath -m "$1"); workers=$2; to=$3; shift 3
for item in "$@"; do
  plan=${item%%:*}; runner=$floatlib/benchmarks/suites/Proofs.lean
  [[ "$item" == *:* ]] && runner=$here/${item##*:}
  echo "== $(date -u +%FT%TZ) plan $plan runner $(basename "$runner")"
  python3 -I "$here/prove.py" "$work/$plan" "$out" --runner "$runner" --timeout "$to" --workers "$workers"
done
echo "== $(date -u +%FT%TZ) end"
