"""How the route closed each proved goal, from the logs of the trace run (a diagnostic).

usage: python3 -I trace_summary.py <trace run dir> <out.csv>
Reads the lines of the trace class `interval.constraints` in each log and writes one row per goal: the steps that
closed the goal or its cases, the number of cases, and the boxes of the certificates.
"""
import collections
import csv
import json
import os
import re
import sys

run, out = sys.argv[1:3]
rows = []
for line in open(os.path.join(run, "results.jsonl")):
    r = json.loads(line)
    log = open(os.path.join(run, "logs", r["lean"][:-5] + ".log"), encoding="utf-8", errors="replace").read()
    closed = collections.Counter()
    boxes = 0
    for m in re.finditer(r"closed (on the ranges of the atoms, without a box|by a form that is definite without a box|"
                         r"by linarith, without a box|by linarith|by a certificate of (\d+) boxes|by ([^\n]*))", log):
        text = m.group(1)
        if m.group(2):
            boxes += int(m.group(2))
            text = "by a certificate"
        closed[text.strip()] += 1
    cases = sum(int(n) for n in re.findall(r"(\d+) cases of the disjunctions", log))
    family = r["lean"].split("__")[0]
    steps = set(closed)
    if "by a certificate" in steps:
        how = "a certificate of one box" if boxes == 1 else "certificates of more than one box"
    elif any("linarith" in s for s in steps):
        how = "linarith or nlinarith, without a box"
    elif "by a form that is definite without a box" in steps:
        how = "a definite form, without a box"
    elif steps:
        how = "the ranges of the atoms alone, without a box"
    else:
        how = "no closing line in the trace"
    rows.append({"lean": r["lean"], "family": family, "outcome": r["outcome"], "wall_s": r["wall_s"],
                 "how": how, "cases": cases, "certificate_boxes": boxes,
                 "closed": "; ".join(f"{k} x{v}" for k, v in sorted(closed.items())),
                 "derived_bounds": log.count("derived the bounds"), "pinned": log.count("is pinned to"),
                 "trace_lines": log.count("[interval.constraints]")})
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print(len(rows), "goals;", collections.Counter(r["outcome"] for r in rows))
by = collections.defaultdict(collections.Counter)
for r in rows:
    steps = tuple(sorted({s.split(" x")[0] for s in r["closed"].split("; ") if s}))
    by[r["family"]][steps] += 1
for family in sorted(by):
    print(family)
    for steps, n in by[family].most_common():
        print(f"   {n:4d}  {' + '.join(steps) or 'no closing line in the trace'}")
allb = [r["certificate_boxes"] for r in rows if r["certificate_boxes"]]
if allb:
    allb.sort()
    print("certificate boxes per goal: min", allb[0], "median", allb[len(allb) // 2], "max", allb[-1], "goals", len(allb))
