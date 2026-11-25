#!/bin/bash
# run_kernel.sh DIR NPROC [FILE...]: check the generated kernel files of DIR with Lean, NPROC at a
# time (at most 8). Run it under `lake env` of this week's Lake project:
#   (cd WEEK && lake env bash vnncomp-replay/scripts/run_kernel.sh DIR NPROC)
# Every file gets DIR/logs/NAME.log (Lean's output, with the profiler lines) and NAME.time.
set -u
dir=$1; nproc=$2; shift 2
(( nproc > 8 )) && nproc=8
mkdir -p "$dir/logs"
if [ ! -f "$dir/VnnPrelude.olean" ] || [ "$dir/VnnPrelude.lean" -nt "$dir/VnnPrelude.olean" ]; then
  lean --root="$dir" -o "$dir/VnnPrelude.olean" "$dir/VnnPrelude.lean" > "$dir/logs/VnnPrelude.log" 2>&1 || { echo "prelude failed"; cat "$dir/logs/VnnPrelude.log"; exit 1; }
fi
if [ $# -gt 0 ]; then files=("$@"); else files=("$dir"/K_*.lean); fi
export LEAN_PATH="$dir:$LEAN_PATH"
printf '%s\n' "${files[@]}" | xargs -P "$nproc" -I{} bash -c '
  f={}; name=$(basename "$f" .lean); d=$(dirname "$f")
  /usr/bin/time -v -o "$d/logs/$name.time" timeout --kill-after=10 3000 lean -Dprofiler=true -Dprofiler.threshold=1000 "$f" > "$d/logs/$name.log" 2>&1
  echo "$name exit $?"'
