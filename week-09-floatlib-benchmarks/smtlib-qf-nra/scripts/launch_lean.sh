#!/bin/bash
# launch_lean.sh <lean dir> <names list> <out dir> <limit seconds> [extra run_lean.py options]
# Runs run_lean.py under `lake env` of this week's Lake project, so that `lean` finds FloatLib.
set -eu
here=$(cd "$(dirname "$0")" && pwd); suite=$(dirname "$here"); week=$(dirname "$suite")
lean_dir=$(realpath "$1"); names=$(realpath "$2"); out=$(realpath -m "$3"); limit=$4; shift 4
mkdir -p "$out"
cd "$week"
exec lake env python3 -I "$here/run_lean.py" "$lean_dir" "$names" "$out" --limit "$limit" "$@"
