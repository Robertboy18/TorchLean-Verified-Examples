#!/bin/bash
# run.sh RAW_DIR...: the measurements of this figure are the `exact-native` section of the kernel timing
# runs, which `../kernel-vs-coq/scripts/run_kernel_sections.py` makes together with the `matched` section.
# One directory per run; then the collector writes `results/kernel-native.csv` and `results/run.csv` here.
set -eu
here=$(cd "$(dirname "$0")" && pwd); suite=$(dirname "$here"); week=$(dirname "$suite")
for raw in "$@"; do python3 -I "$week/kernel-vs-coq/scripts/run_kernel_sections.py" "$raw"; done
python3 -I "$week/kernel-vs-coq/scripts/collect.py" "$@"
python3 "$here/plot.py"
