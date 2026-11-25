#!/bin/bash
# fetch.sh: LeanCert (github.com/alerad/leancert, Apache-2.0) at the commit of the runs, ported to Lean
# v4.36.0 and Mathlib v4.36.0 with `leancert-lean-4.36.diff`, into LEANCERT_DIR (default `work/leancert` of
# this suite). `dl/leancert-src` under it keeps the untouched clone, which `harvest_examples.py` reads.
# Then `lcbuild.sh LeanCert.Tactic LeanCert.Benchmark.Main` builds what the runs need.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd); suite=$(dirname "$here")
target=$(realpath -m "${LEANCERT_DIR:-$suite/work/leancert}"); work=$(dirname "$target")
mkdir -p "$work/dl"
git clone https://github.com/alerad/leancert "$work/dl/leancert-src"
(cd "$work/dl/leancert-src" && git checkout 44b323a0d6af00211d5013f6e021432db032087c)
cp -a "$work/dl/leancert-src" "$target"
(cd "$target" && patch -p2 < "$here/leancert-lean-4.36.diff" && cat lean-toolchain)
