#!/bin/bash
# lcbuild.sh TARGET...: build LeanCert modules in the ported checkout (at most 8 Lean threads by default).
set -u
here=$(cd "$(dirname "$0")" && pwd); suite=$(dirname "$here"); week=$(dirname "$suite")
cd "${LEANCERT_DIR:-$suite/work/leancert}" || exit 2
export LEAN_NUM_THREADS=${LC_JOBS:-8}
exec lake build "$@"
