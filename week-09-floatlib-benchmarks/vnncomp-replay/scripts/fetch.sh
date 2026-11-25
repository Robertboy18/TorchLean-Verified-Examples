#!/bin/bash
# fetch.sh: the inputs of this suite, each pinned, into RUN/dl (default `run/dl` of this suite).
#   the VNN-COMP 2025 benchmarks (github.com/VNN-COMP/vnncomp2025_benchmarks at 8b7b811b): the 14 families
#   with small files, as a sparse checkout (927 MB)
#   the VNN-COMP 2025 results (github.com/VNN-COMP/vnncomp2025_results at ea89fbc2): `results.csv` and the
#   counterexample files of the participating tools, and the logs of the competition's two scorings (81 MB)
# The families have licenses of their own, so nothing of this is in the repository.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd); suite=$(dirname "$here")
run=$(realpath -m "${RUN:-$suite/run}"); dl=$run/dl
FAMILIES="acasxu_2023 soundnessbench safenlp_2024 sat_relu collins_rul_cnn_2022 cora_2024 malbeware relusplitter
          tllverifybench_2023 test cersyve dist_shift_2023 linearizenn_2024 lsnc_relu"
mkdir -p "$dl"; cd "$dl"
git clone --filter=blob:none --no-checkout https://github.com/VNN-COMP/vnncomp2025_benchmarks
(cd vnncomp2025_benchmarks && git sparse-checkout init --cone &&
  git sparse-checkout set $(for f in $FAMILIES; do echo "benchmarks/$f"; done) && git checkout 8b7b811b)
git clone --filter=blob:none --no-checkout https://github.com/VNN-COMP/vnncomp2025_results
(cd vnncomp2025_results && git checkout ea89fbc2)
echo "benchmarks: $(cd vnncomp2025_benchmarks && git rev-parse HEAD)"
echo "results:    $(cd vnncomp2025_results && git rev-parse HEAD)"
