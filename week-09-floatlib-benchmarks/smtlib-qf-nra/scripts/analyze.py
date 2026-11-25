"""Select and translate: parse each candidate benchmark, decide boundedness, write the Lean goal.

usage: python3 -I analyze.py <QF_NRA root> <results-sq-2025.json.gz> <out dir> [--workers N] [--cap NODES]

Candidates are the benchmarks outside meti-tarski whose `:status` is `unsat` (tier "status"), and those whose
`:status` is `unknown` and that at least one SMT-COMP 2025 solver answered `unsat` and none `sat` (tier
"solvers").  One JSON line per candidate goes to <out dir>/candidates.jsonl.  A Lean file is written for every
candidate whose let-expansion has at most `--cap` nodes: to <out dir>/lean when every real variable is bounded by
the constraints (the selection), to <out dir>/lean-unbounded otherwise (run only for information).
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import signal
import sys
import time
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import smt2lean as S

HEADER = """module

public import FloatLib.Tactic.Interval.Constraints
meta import FloatLib.Tactic.Interval.Constraints

/-! {comment} -/

theorem goal : {goal} := by
  interval

#print axioms goal
"""


class Alarm(Exception):
    pass


def _alarm(signum, frame):
    raise Alarm


def work(job):
    root, rel, tier, outdir, cap, limit, literal, vector_above = job
    row = {"path": rel, "family": rel.split("/")[0], "tier": tier}
    start = time.monotonic()
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(limit)
    try:
        text = open(os.path.join(root, rel), encoding="utf-8", errors="replace").read()
        row["bytes"] = len(text)
        p = S.Problem().load(text)
        row["status"] = p.status
        reals, bools = p.used()
        row.update(reals=len(reals), bools=len(bools), declared_reals=len(p.real_names),
                   declared_bools=len(p.bool_names), asserts=len(p.asserts))
        size = sum(p.size(a) for a in p.asserts)
        row["nodes"] = size
        feats = p.features()
        row["real_ite"] = feats.get("ite", 0)
        row["div_nonconst"] = feats.get("div_nonconst", 0)
        row["parse_s"] = round(time.monotonic() - start, 3)
        if size > 10 * cap:
            row.update(outcome="not translated", selected=None,
                       reason=f"formula has {size} nodes after let-expansion, over ten times the cap of {cap}")
            return row
        bounder = S.Bounder(p, reals)
        verdict, box = bounder.run()
        row["verdict"] = verdict
        row["skipped_atoms"] = bounder.skipped_atoms
        row["unbounded_vars"] = sum(1 for lo, hi in box.values() if lo == -S.INF or hi == S.INF)
        row["analysis_s"] = round(time.monotonic() - start, 3)
        row["selected"] = verdict.startswith("bounded")
        if size > cap:
            row.update(outcome="not translated",
                       reason=f"formula has {size} nodes after let-expansion, over the cap of {cap}")
            return row
        row["vector"] = vector_above is not None and len(reals) + len(bools) > vector_above
        emitter = S.Emitter(p, reals, bools, bool_literal=literal, vector=row["vector"])
        goal = emitter.goal()
        ident = re.sub(r"[^A-Za-z0-9_.-]", "_", rel[:-5].replace("/", "__"))
        names = ", ".join(f"{emitter.rname[v]} = {p.real_names[v]}" for v in reals)
        names += "".join(f", {emitter.bname[v]} = {p.bool_names[v]} (Bool as 0/1)" for v in bools)
        comment = (f"SMT-LIB 2025 QF_NRA/{rel} (status {p.status}). Variables: {names}.").replace("-/", "- /")
        sub = "lean" if row["selected"] else "lean-unbounded"
        lean = os.path.join(outdir, sub, ident + ".lean")
        with open(lean, "w", encoding="utf-8") as f:
            f.write(HEADER.format(comment=comment, goal=goal))
        row.update(outcome="translated", lean=ident + ".lean", lean_dir=sub, lean_bytes=os.path.getsize(lean))
        return row
    except S.Unsupported as e:
        row.update(outcome="not translated", reason=f"unsupported: {e}")
        return row
    except Alarm:
        row.update(outcome="not translated", reason=f"parsing and analysis took more than {limit} s")
        return row
    except RecursionError:
        row.update(outcome="not translated", reason="recursion limit of the translator")
        return row
    except MemoryError:
        row.update(outcome="not translated", reason="memory limit of the translator")
        return row
    finally:
        signal.alarm(0)
        row["total_s"] = round(time.monotonic() - start, 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root"); ap.add_argument("results"); ap.add_argument("out")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--cap", type=int, default=200000)
    ap.add_argument("--limit", type=int, default=600)
    ap.add_argument("--only", default="")
    ap.add_argument("--vector-above", type=int, default=None,
                    help="state a goal with more than this many variables over one vector x : Fin n → ℝ")
    ap.add_argument("--list", default="", help="take only the benchmarks whose paths this file lists")
    ap.add_argument("--control", action="store_true",
                    help="take the benchmarks with `:status sat` instead: none of their goals may be proved")
    ap.add_argument("--bool-literal", default="eq1", choices=["eq1", "half"],
                    help="the literal of a Boolean variable on its 0/1 real: b = 1 (eq1) or 1/2 < b (half)")
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, "lean"), exist_ok=True)
    os.makedirs(os.path.join(a.out, "lean-unbounded"), exist_ok=True)
    answers = {}
    for r in json.load(gzip.open(a.results, "rt"))["results"]:
        f = r["file"]
        if f["logic"] == "QF_NRA":
            answers.setdefault("/".join(f["family"]) + "/" + f["name"], set()).add(r["result"])
    status_re = re.compile(rb"\(set-info\s+:status\s+(\w+)\s*\)")
    listed = {line.strip() for line in open(a.list) if line.strip()} if a.list else None
    jobs = []
    for dirpath, dirnames, filenames in os.walk(a.root):
        dirnames.sort()
        for name in sorted(filenames):
            rel = os.path.relpath(os.path.join(dirpath, name), a.root)
            if not name.endswith(".smt2") or rel.startswith("meti-tarski/"):
                continue
            if a.only and not re.search(a.only, rel):
                continue
            if listed is not None and rel not in listed:
                continue
            m = status_re.search(open(os.path.join(a.root, rel), "rb").read(20000))
            status = m.group(1).decode() if m else "none"
            got = answers.get(rel, set())
            if a.control:
                if status == "sat":
                    jobs.append((a.root, rel, "sat-control", a.out, a.cap, a.limit, a.bool_literal, a.vector_above))
            elif status == "unsat":
                jobs.append((a.root, rel, "status", a.out, a.cap, a.limit, a.bool_literal, a.vector_above))
            elif status == "unknown" and "unsat" in got and "sat" not in got:
                jobs.append((a.root, rel, "solvers", a.out, a.cap, a.limit, a.bool_literal, a.vector_above))
    # big files last, so that the progress is visible early
    jobs.sort(key=lambda j: os.path.getsize(os.path.join(j[0], j[1])))
    print(f"{len(jobs)} candidates", flush=True)
    done = 0
    with Pool(a.workers, maxtasksperchild=50) as pool, open(os.path.join(a.out, "candidates.jsonl"), "w") as out:
        for row in pool.imap_unordered(work, jobs, chunksize=1):
            out.write(json.dumps(row) + "\n"); out.flush()
            done += 1
            if done % 200 == 0:
                print(f"{done}/{len(jobs)}", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
