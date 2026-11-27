#!/usr/bin/env bash
# Build the public-operation benchmark against an existing FloatLib build.
# One Lean module is compiled and one executable is linked; nothing in the library is rebuilt and
# nothing is written in the FloatLib checkout or in its build directory.
#   build_public.sh [OUT_DIR [--low-word-128]]      (OUT_DIR is `work/public` of this suite by default)
# FLOATLIB_DIR is a FloatLib checkout whose benchmark executables are built (its
# `benchmarks/scripts/format-comparison.sh` builds them); by default the copy that Lake keeps for this
# week, `.lake/packages/floatlib`. FLOATLIB_BUILD_DIR is its build directory.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd); suite=$(dirname "$here"); week=$(dirname "$suite")
checkout=$(realpath "${FLOATLIB_DIR:-$week/.lake/packages/floatlib}")
build="${FLOATLIB_BUILD_DIR:-$checkout/.lake/build}"
out=$(realpath -m "${1:-$suite/work/public}")
flag="${2:-}"
toolchain="${LEAN_TOOLCHAIN_DIR:-$(cd "$checkout" && lean --print-prefix)}"

python3 -I "$here/make_public.py" "$checkout" "$build" "$out" $flag

# The search path that Lake gave `lean` for FormatComparison (from its trace file).
lean_path="$(python3 -I - "$build/lib/lean/FloatLibBenchmarks/Public/FormatComparison.trace" <<'PY'
import json, re, sys
trace = json.load(open(sys.argv[1]))
print(re.search(r"LEAN_PATH=(\S+)", trace["log"][0]["message"]).group(1))
PY
)"
cd "$out"
echo "lean start $(date -u +%FT%TZ)"
LEAN_PATH="$lean_path" "$toolchain/bin/lean" "$out/PublicOperations.lean" \
  -o "$out/PublicOperations.olean" -i "$out/PublicOperations.ilean" -c "$out/PublicOperations.c" \
  --setup "$out/PublicOperations.setup.json"
echo "lean done $(date -u +%FT%TZ); C bytes $(stat -c %s "$out/PublicOperations.c")"
"$toolchain/bin/clang" -c -o "$out/PublicOperations.c.o.export" "$out/PublicOperations.c" \
  -I "$toolchain/include" -fstack-clash-protection -ffp-contract=off -fdata-sections -ffunction-sections \
  -fPIC -fvisibility=hidden -Wno-unused-command-line-argument --sysroot "$toolchain" -nostdinc \
  -isystem "$toolchain/include/clang" -O3 -DNDEBUG -DLEAN_EXPORTING
echo "clang -c done $(date -u +%FT%TZ)"
"$toolchain/bin/clang" -o "$out/publicOperations" "@$out/publicOperations.rsp"
echo "link done $(date -u +%FT%TZ)"
sha256sum "$out/publicOperations" "$out/PublicOperations.lean" "$out/PublicOperations.c"
# Uncalibrated calls of every row (1,000,000 iterations at 32 and 64 bits): the backend that the
# default policy selects, and a first look at the three execution classes.
for width in 32 64; do
  FORMAT_COMPARE_WIDTH=$width FORMAT_COMPARE_ITERATIONS=1000000 "$out/publicOperations"
done
FORMAT_COMPARE_WIDTH=128 FORMAT_COMPARE_ITERATIONS=100000 "$out/publicOperations"
