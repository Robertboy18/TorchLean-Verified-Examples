#!/bin/bash
# run_timing.sh NAME LIST REPS [extra time_solvers.py arguments]: time Z3, cvc5 (and Lean when asked) on the
# instances of LIST, one process at a time, under `lake env` of this week's Lake project so that `lean`
# sees the library. Results in RUN/results/timing-NAME.csv. QF_FP is the extracted benchmark directory
# (`fetch.sh` puts it at RUN/dl/non-incremental/QF_FP).
set -u
here=$(cd "$(dirname "$0")" && pwd); suite=$(dirname "$here"); week=$(dirname "$suite")
run=$(realpath -m "${RUN:-$suite/run}")          # generated files and raw results
z3=${Z3:-$run/tools/z3/z3}; cvc5=${CVC5:-$run/tools/cvc5/cvc5}
name=$1; list=$(realpath "$2"); reps=$3; shift 3
bench=$(realpath "${QF_FP:-$run/dl/non-incremental/QF_FP}")
mkdir -p "$run/results"
cd "$week" || exit 2
exec lake env python3 -I "$here/time_solvers.py" "$bench" "$list" \
  "$run/results/timing-$name.csv" --reps "$reps" --timeout 60 \
  --tool "z3=$z3 -smt2" --tool "cvc5=$cvc5" "$@"
