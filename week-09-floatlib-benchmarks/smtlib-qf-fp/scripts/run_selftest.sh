#!/bin/bash
# run_selftest.sh: the translator's self-test (selftest_gen.py): generate the instances with Z3, classify
# and translate them, prove them in Lean (8 processes at most), and ask Z3 and cvc5 for each instance once.
# Everything goes under RUN (default `run/` of this suite); Z3 and CVC5 name the two solver binaries.
set -u
here=$(cd "$(dirname "$0")" && pwd); suite=$(dirname "$here"); week=$(dirname "$suite")
run=$(realpath -m "${RUN:-$suite/run}")          # generated files and raw results
z3=${Z3:-$run/tools/z3/z3}; cvc5=${CVC5:-$run/tools/cvc5/cvc5}
out=$run/selftest
rm -rf "$out" "$run/lean/selftest"
mkdir -p "$out" "$run/meta" "$run/results"
python3 -I "$here/selftest_gen.py" "$z3" "$out" || exit 1
python3 -I "$here/qffp2lean.py" classify "$out/instances" "$run/meta/selftest-classify.jsonl" || exit 1
python3 -I "$here/qffp2lean.py" emit "$out/instances" "$run/meta/selftest-classify.jsonl" "$run/lean/selftest" \
  --batch 250 --prefix Self || exit 1
"$here/run_batches.sh" "$run/lean/selftest" 8
python3 -I "$here/collect.py" "$run/lean/selftest" "$run/results/selftest-instances.csv" \
  "$run/results/selftest-summary.json" | grep -v "^NOT PROVED" | tail -4
python3 -I -c "
import json, sys
for line in open(sys.argv[1]):
    print(json.loads(line)['instance'])" "$run/meta/selftest-classify.jsonl" > "$run/meta/selftest-list.txt"
cd "$week" || exit 2
lake env python3 -I "$here/time_solvers.py" "$out/instances" "$run/meta/selftest-list.txt" \
  "$run/results/selftest-solvers.csv" --reps 1 --timeout 120 \
  --tool "z3=$z3 -smt2" --tool "cvc5=$cvc5"
