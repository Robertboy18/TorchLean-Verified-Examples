#!/bin/bash
# run_chunks.sh FAMILY NPROC [replay options]: replay RUN/work/cases/FAMILY.cases with NPROC processes
# (at most 8), each on a contiguous range of cases, and join the outputs in case order.
set -u
here=$(cd "$(dirname "$0")" && pwd); suite=$(dirname "$here"); week=$(dirname "$suite")
run=$(realpath -m "${RUN:-$suite/run}")          # downloads (dl/), translated inputs (work/), raw results (results/)
replay=${REPLAY:-$week/.lake/build/bin/replay}       # `lake build replay` in the week's Lake project
family=$1; nproc=$2; shift 2
(( nproc > 8 )) && nproc=8
cases=$run/work/cases/$family.cases
total=$(grep -c '^end$' "$cases")
out=$run/results/replay-$family.tsv
mkdir -p "$run/results/parts"
size=$(( (total + nproc - 1) / nproc ))
pids=()
for (( k = 0; k < nproc; k++ )); do
  from=$(( k * size )); to=$(( from + size ))
  "$replay" "$cases" --from $from --to $to "$@" \
    > "$run/results/parts/$family.$k.tsv" 2> "$run/results/parts/$family.$k.err" &
  pids+=($!)
done
rc=0
for pid in "${pids[@]}"; do wait $pid || rc=1; done
head -1 "$run/results/parts/$family.0.tsv" > "$out"
for (( k = 0; k < nproc; k++ )); do tail -n +2 "$run/results/parts/$family.$k.tsv" >> "$out"; done
echo "$family: $total cases, $(( $(wc -l < "$out") - 1 )) lines, rc=$rc"
cat "$run"/results/parts/$family.*.err | head -20
exit $rc
