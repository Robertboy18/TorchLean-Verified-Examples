#!/bin/bash
# harness_cases.sh OUT CASE...: run single cases of LeanCert's benchmark harness (interpreter), each with
# its own timeout, in the ported checkout. The JSON lines give the enclosure that each case computes.
set -u
here=$(cd "$(dirname "$0")" && pwd); suite=$(dirname "$here"); week=$(dirname "$suite")
out=$(realpath -m "$1"); shift
cd "${LEANCERT_DIR:-$suite/work/leancert}" || exit 2
: > "$out"
for c in "$@"; do
  timeout 420 lake env lean --run LeanCert/Benchmark/Main.lean --suite all --case "$c" --samples 1 --warmups 0 --format jsonl >> "$out" 2>> "$out.err" || echo "CASE_FAILED $c rc=$?" >> "$out"
done
