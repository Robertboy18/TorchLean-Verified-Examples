"""Copies of the goal files that Lean proved, with the trace of the route switched on (a diagnostic).

usage: python3 -I make_trace_goals.py <out dir> <lean dir>=<run dir> ...
For every goal that `results.jsonl` of a run dir records as proved, the goal file of the lean dir is copied to
<out dir>/lean with `set_option trace.interval.constraints true in` before the theorem; <out dir>/names.list
lists the copies.  The statement and the call of `interval` are unchanged.
"""
import json
import os
import sys

out = sys.argv[1]
os.makedirs(os.path.join(out, "lean"), exist_ok=True)
names = []
for pair in sys.argv[2:]:
    lean_dir, run = pair.split("=", 1)
    for line in open(os.path.join(run, "results.jsonl")):
        r = json.loads(line)
        if r["outcome"] != "proved":
            continue
        text = open(os.path.join(lean_dir, r["lean"]), encoding="utf-8").read()
        assert text.count("\ntheorem goal : ") == 1
        text = text.replace("\ntheorem goal : ", "\nset_option trace.interval.constraints true in\ntheorem goal : ")
        open(os.path.join(out, "lean", r["lean"]), "w", encoding="utf-8").write(text)
        names.append(r["lean"])
open(os.path.join(out, "names.list"), "w").write("\n".join(sorted(names)) + "\n")
print(len(names), "goals")
