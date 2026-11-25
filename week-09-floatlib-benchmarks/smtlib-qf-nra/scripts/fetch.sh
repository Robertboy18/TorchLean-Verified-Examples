#!/bin/bash
# fetch.sh: the inputs of this suite, each pinned, into RUN/dl (default `run/dl` of this suite).
#   SMT-LIB release 2025.08.04, logic QF_NRA (Zenodo record 16740866, CC BY 4.0), without `meti-tarski`
#   the SMT-COMP 2025 result file (github.com/SMT-COMP/smt-comp.github.io, `data/`)
#   Z3 5.3.3 (the binary of the PyPI wheel), cvc5 1.4.2 and Yices 2.8.0 (static release binaries), into RUN/tools
# Nothing of this is in the repository. Read a downloaded file before you run it.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd); suite=$(dirname "$here")
run=$(realpath -m "${RUN:-$suite/run}"); dl=$run/dl; tools=$run/tools
SMTCOMP=cd15cbedd4ef312d615448100797a80e922cf575
mkdir -p "$dl" "$tools/z3" "$tools/cvc5" "$tools/yices"
cd "$dl"
curl -fL -o QF_NRA.tar.zst "https://zenodo.org/records/16740866/files/QF_NRA.tar.zst?download=1"
echo "85a04fcea6dd10fb4c505703cac46e2c  QF_NRA.tar.zst" | md5sum -c -
tar --zstd -xf QF_NRA.tar.zst --exclude='non-incremental/QF_NRA/meti-tarski'       # 5,148 benchmarks, 3.3 GB
curl -fL -o results-sq-2025.json.gz \
  "https://raw.githubusercontent.com/SMT-COMP/smt-comp.github.io/$SMTCOMP/data/results-sq-2025.json.gz"
python3 -m pip download --no-deps --only-binary=:all: --platform manylinux_2_27_x86_64 -d z3-wheel z3-solver==5.3.3.0
echo "0431023a82278dab706c02a320f63178ee0755fb8f10a5c980f0bfe3c84f6f96  z3-wheel/z3_solver-5.3.3.0-py3-none-manylinux_2_27_x86_64.whl" | sha256sum -c -
unzip -o -q z3-wheel/*.whl -d z3-wheel
find z3-wheel -type f -name z3 -exec install -m 755 {} "$tools/z3/z3" \;
curl -fL -o cvc5-Linux-x86_64-static.zip \
  "https://github.com/cvc5/cvc5/releases/download/cvc5-1.4.2/cvc5-Linux-x86_64-static.zip"
unzip -o -q cvc5-Linux-x86_64-static.zip -d cvc5-release
find cvc5-release -type f -name cvc5 -exec install -m 755 {} "$tools/cvc5/cvc5" \;
curl -fL -o yices-2.8.0-x86_64-pc-linux-gnu-static-gmp.tar.gz \
  "https://github.com/SRI-CSL/yices2/releases/download/yices-2.8.0/yices-2.8.0-x86_64-pc-linux-gnu-static-gmp.tar.gz"
mkdir -p yices-release && tar -xzf yices-2.8.0-x86_64-pc-linux-gnu-static-gmp.tar.gz -C yices-release
find yices-release -type f -name yices-smt2 -exec install -m 755 {} "$tools/yices/yices-smt2" \;
"$tools/z3/z3" --version; "$tools/cvc5/cvc5" --version | head -1; "$tools/yices/yices-smt2" --version | head -1
