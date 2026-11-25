#!/bin/bash
# run_selftest_fpexp.sh: cvc5 with --fp-exp on the self-test instances whose format is neither binary32 nor
# binary64 (cvc5 1.4.2 refuses those formats without the option). `RUN/meta/selftest-other-formats.txt`
# lists them.
set -u
here=$(cd "$(dirname "$0")" && pwd); suite=$(dirname "$here"); week=$(dirname "$suite")
run=$(realpath -m "${RUN:-$suite/run}")          # generated files and raw results
z3=${Z3:-$run/tools/z3/z3}; cvc5=${CVC5:-$run/tools/cvc5/cvc5}
exec python3 -I "$here/time_solvers.py" "$run/selftest/instances" "$run/meta/selftest-other-formats.txt" \
  "$run/results/selftest-cvc5-fpexp.csv" --reps 1 --timeout 120 --tool "cvc5-fp-exp=$cvc5 --fp-exp"
