#!/bin/bash
# fetch.sh: the NumFuzz artifact (github.com/ak-2485/NumFuzz, GPL-3.0) at the commit of the runs, into
# RUN/dl/NumFuzz (default `run/dl` of this suite). Its 110 `.fz` programs, 35 FPTaylor inputs and 30 Gappa
# scripts are read as data; nothing of the artifact is in the repository and none of its code is run.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd); suite=$(dirname "$here")
run=$(realpath -m "${RUN:-$suite/run}"); mkdir -p "$run/dl"; cd "$run/dl"
git clone https://github.com/ak-2485/NumFuzz
(cd NumFuzz && git checkout 80c54f35348c60b8a1e95f1881adda8d2e9c1dfb && git rev-parse HEAD)
