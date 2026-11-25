#!/bin/bash
# fetch.sh: LLM2Cert's repository (github.com/FormalCert-AI/llm2cert, MIT) at the commit of the comparison,
# into RUN/dl/llm2cert (default `run/dl` of this suite), with its recorded experiments unpacked. The
# comparison reads `results/rq1/data/lean_records.json` there. The 291 locked statements themselves are in
# `inputs/obligations.json` of this suite.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd); suite=$(dirname "$here")
run=$(realpath -m "${RUN:-$suite/run}"); mkdir -p "$run/dl"; cd "$run/dl"
git clone https://github.com/FormalCert-AI/llm2cert
cd llm2cert && git checkout 37b6a0484fc8b532bb29588b56f5993193c42fcc
python3 scripts/unpack_records.py rq1
ls -la results/rq1/data/lean_records.json
