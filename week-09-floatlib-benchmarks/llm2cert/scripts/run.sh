#!/usr/bin/env bash
# run.sh NAME...: `by interval` for each of the 291 obligations, with Lean's own limits (NAME `defaults`)
# or with the raised limits (NAME `raised`). Six Lean processes at a time and 600 seconds per obligation.
# Generates the files (LLM2Cert's statements verbatim, one generated proof line), runs Lean on them under
# `lake env` of this week's Lake project, and prints the table against LLM2Cert's records.
#   RUN      where the files and results go (default `run/` of this suite)
#   RECORDS  LLM2Cert's `lean_records.json` (default: where `fetch.sh` leaves it)
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd); suite=$(dirname "$here"); week=$(dirname "$suite")
run=$(realpath -m "${RUN:-$suite/run}")
records=${RECORDS:-$run/dl/llm2cert/results/rq1/data/lean_records.json}
status=0
for name in "$@"; do
  out="$run/$name"
  if [ "$name" = defaults ]; then flag=--defaults; else flag=; fi
  python3 -I "$here/llm2cert.py" gen "$suite/inputs/obligations.json" "$out" $flag
  {
    echo "start $(date -u +%FT%TZ)"
    echo "loadavg $(cat /proc/loadavg)"
    echo "idle $(mpstat 5 1 | awk '/Average/ {print $NF}')"
    echo "lean $(cd "$week" && lake env lean --version)"
    echo "workers 6, timeout 600 s, LEAN_NUM_THREADS ${LEAN_NUM_THREADS:-unset}"
  } > "$out/run.txt"
  start=$(date +%s)
  (cd "$week" && lake env python3 -I "$here/llm2cert.py" run "$out" --jobs 6 --timeout 600 --lean lean) || status=1
  {
    echo "wall_seconds $(( $(date +%s) - start ))"
    echo "end $(date -u +%FT%TZ)"
    echo "loadavg_end $(cat /proc/loadavg)"
    echo "idle_end $(mpstat 5 1 | awk '/Average/ {print $NF}')"
  } >> "$out/run.txt"
  python3 -I "$here/llm2cert.py" table "$out" "$records" > "$out/table.txt" || status=1
  cat "$out/run.txt" "$out/table.txt"
done
exit $status
