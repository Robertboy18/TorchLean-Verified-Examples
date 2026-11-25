#!/bin/bash
# fetch.sh: the inputs of this suite, each pinned, into RUN/dl (default `run/dl` of this suite).
#   SMT-LIB release 2025.08.04, logic QF_FP (Zenodo record 16740866, CC BY 4.0), 40,406 benchmarks
#   the SMT-COMP result files of 2025 and 2018 (github.com/SMT-COMP/smt-comp.github.io, `data/`)
#   Z3 5.4.1 (the binary of the PyPI wheel) and cvc5 1.4.2 (the static release binary), into RUN/tools
# Nothing of this is in the repository. Read a downloaded file before you run it.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd); suite=$(dirname "$here")
run=$(realpath -m "${RUN:-$suite/run}"); dl=$run/dl; tools=$run/tools
SMTCOMP=cd15cbedd4ef312d615448100797a80e922cf575
mkdir -p "$dl" "$tools/z3" "$tools/cvc5"
cd "$dl"
curl -fL -o QF_FP.tar.zst "https://zenodo.org/records/16740866/files/QF_FP.tar.zst?download=1"
echo "b7f9786b5317d7be8a185c861088f504  QF_FP.tar.zst" | md5sum -c -
tar --zstd -xf QF_FP.tar.zst                     # non-incremental/QF_FP/...
for name in results-sq-2025.json.gz benchmarks-2025.json.gz results-sq-2018.json.gz; do
  curl -fL -o "$name" "https://raw.githubusercontent.com/SMT-COMP/smt-comp.github.io/$SMTCOMP/data/$name"
done
python3 -m pip download --no-deps --only-binary=:all: --platform manylinux_2_27_x86_64 -d z3-wheel z3-solver==5.4.1.0
python3 -c "import sys, zipfile; z = zipfile.ZipFile(sys.argv[1]); z.extract('z3_solver-5.4.1.0.data/data/bin/z3', sys.argv[2])" \
  z3-wheel/z3_solver-5.4.1.0-py3-none-manylinux_2_27_x86_64.whl z3-wheel || unzip -o -q z3-wheel/*.whl -d z3-wheel
find z3-wheel -type f -name z3 -exec install -m 755 {} "$tools/z3/z3" \;
curl -fL -o cvc5-Linux-x86_64-static.zip \
  "https://github.com/cvc5/cvc5/releases/download/cvc5-1.4.2/cvc5-Linux-x86_64-static.zip"
unzip -o -q cvc5-Linux-x86_64-static.zip -d cvc5-release
find cvc5-release -type f -name cvc5 -exec install -m 755 {} "$tools/cvc5/cvc5" \;
"$tools/z3/z3" --version; "$tools/cvc5/cvc5" --version | head -1
sha256sum QF_FP.tar.zst ./*.json.gz cvc5-Linux-x86_64-static.zip z3-wheel/*.whl > SHA256SUMS
