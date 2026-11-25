#!/bin/bash
# run_kernel_dirs.sh NPROC DIR...: run_kernel.sh on several directories of generated kernel files,
# one after the other (under `lake env` of this week's Lake project, as run_kernel.sh).
set -u
nproc=$1; shift
here=$(dirname "$(readlink -f "$0")")
for dir in "$@"; do
  start=$(date +%s)
  bash "$here/run_kernel.sh" "$dir" "$nproc" | tail -3
  echo "$dir: $(( $(date +%s) - start )) s with $nproc Lean processes"
done
