#!/bin/bash
# pipeline.sh NPROC FAMILY... : for every family, the compiled Lean replay (run_chunks.sh, NPROC
# processes, at most 8), then the numpy cross-check and the onnxruntime comparison.
# Extra options for the replay (for example --no-real) come from the environment: REPLAY_OPTS.
set -u
here=$(cd "$(dirname "$0")" && pwd); suite=$(dirname "$here"); week=$(dirname "$suite")
run=$(realpath -m "${RUN:-$suite/run}")          # downloads (dl/), translated inputs (work/), raw results (results/)
replay=${REPLAY:-$week/.lake/build/bin/replay}       # `lake build replay` in the week's Lake project
cd "$run" || exit 2
nproc=$1; shift
for family in "$@"; do
  start=$(date +%s)
  "$here/run_chunks.sh" "$family" "$nproc" ${REPLAY_OPTS:-} || echo "$family: replay failed"
  mid=$(date +%s)
  python3 -I "$here/crosscheck.py" work/cases/$family.cases results/replay-$family.tsv results/crosscheck-$family.tsv \
    || echo "$family: crosscheck failed"
  mid2=$(date +%s)
  python3 -I "$here/ort_compare.py" "$family" dl/vnncomp2025_benchmarks/benchmarks work \
    results/replay-$family.tsv results/ort-$family.tsv || echo "$family: onnxruntime comparison failed"
  end=$(date +%s)
  echo "$family: replay $((mid - start)) s with $nproc processes, numpy $((mid2 - mid)) s, onnxruntime $((end - mid2)) s"
done
