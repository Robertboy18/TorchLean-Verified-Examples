#!/bin/bash
# run_one.sh FILE: elaborate one Lean file (inside `lake env`), with its output in FILE.out, the wall, user and
# system seconds and the peak resident kilobytes in FILE.time, and Lean's exit code in FILE.rc.
f=$1
# Lean stops a file after 100 errors unless told otherwise; a control batch has an error in every theorem.
/usr/bin/time -f "%e %U %S %M" -o "$f.time" lean -DmaxErrors=1000000 "$f" > "$f.out" 2>&1
echo $? > "$f.rc"
