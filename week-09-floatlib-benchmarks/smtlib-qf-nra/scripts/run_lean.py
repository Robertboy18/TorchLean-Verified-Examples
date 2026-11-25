"""Run one Lean process per goal, at most `--workers` at once, with a fixed wall-clock limit per goal.

usage (from the FloatLib clone, so that `lake env` supplies LEAN_PATH):
  lake env python3 -I run_lean.py <dir with .lean files> <list of file names> <out dir> --limit SECONDS

Each goal file ends with `#print axioms goal`.  A goal is `proved` only when Lean exits with status 0, prints no
error, and the axioms line lists nothing beyond propext, Classical.choice and Quot.sound.  Otherwise it is
`timeout` (killed at the wall limit), `memory` (Lean stopped at its `-M` guard or the kernel killed it), or
`refused` (Lean reported an error: the tactic's own failure, or one of Lean's default limits).
One JSON line per goal goes to <out dir>/results.jsonl; Lean's output to <out dir>/logs/<name>.log.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

STANDARD = {"propext", "Classical.choice", "Quot.sound"}
AXIOMS = re.compile(r"depends on axioms: \[([^\]]*)\]")


def classify(code, killed, text):
    """(outcome, detail) from the exit status and Lean's output."""
    if killed:
        return "timeout", "wall-clock limit"
    errors = [line for line in text.splitlines() if ": error:" in line or line.startswith("error:")]
    m = AXIOMS.search(text)
    axioms = {a.strip() for a in m.group(1).split(",") if a.strip()} if m else None
    if code == 0 and not errors:
        if "does not depend on any axioms" in text:
            return "proved", "no axioms"
        if axioms is not None and axioms <= STANDARD:
            return "proved", ", ".join(sorted(axioms))
        return "refused", "no axioms line or a non-standard axiom: " + (m.group(0) if m else "none")
    low = text.lower()
    if code in (-9, 137) or "memory limit" in low or "out of memory" in low or "std::bad_alloc" in low:
        return "memory", "memory guard"
    first = errors[0] if errors else text.strip().splitlines()[0] if text.strip() else f"exit {code}"
    first = re.sub(r"^.*?: error: ", "", first)
    if "maximum recursion depth" in low:
        kind = "Lean limit: maxRecDepth"
    elif "(deterministic) timeout" in low or "deterministic timeout" in low or "thousand heartbeats of the declaration" in low:
        kind = "Lean limit: maxHeartbeats"
    elif "stack overflow" in low:
        kind = "Lean limit: stack"
    else:
        kind = "tactic"
    return "refused", kind + " | " + first[:300]


ACTIVE: set = set()          # process groups of the Lean processes that are running
DEADLINE = [None]            # epoch seconds after which no new goal starts


def stop(signum, frame):
    """On SIGTERM or SIGINT: kill the running Lean processes (each leads a process group of its own) and leave."""
    for pid in list(ACTIVE):
        try:
            os.killpg(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    os._exit(143)


def run_one(job):
    lean_dir, name, out, limit, memory_mb, threads = job
    if DEADLINE[0] is not None and time.time() > DEADLINE[0]:
        return None
    if os.path.exists(os.path.join(out, "STOP")):       # a STOP file ends the run after the goals in flight
        return None
    path = os.path.join(lean_dir, name)
    log = os.path.join(out, "logs", name[:-5] + ".log")
    env = dict(os.environ, LEAN_NUM_THREADS=str(threads))
    start = time.monotonic()
    utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    killed = False
    with open(log, "wb") as stream:
        proc = subprocess.Popen(["lean", "-M", str(memory_mb), path], stdout=stream, stderr=subprocess.STDOUT,
                                env=env, start_new_session=True, cwd=os.getcwd())
        ACTIVE.add(proc.pid)
        usage = None
        while True:
            pid, status, usage = os.wait4(proc.pid, os.WNOHANG)
            if pid:
                break
            if time.monotonic() - start > limit:
                killed = True
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                pid, status, usage = os.wait4(proc.pid, 0)
                break
            time.sleep(0.02)
        ACTIVE.discard(proc.pid)
        proc.returncode = os.waitstatus_to_exitcode(status)
    wall = time.monotonic() - start
    text = open(log, encoding="utf-8", errors="replace").read()
    outcome, detail = classify(proc.returncode, killed, text)
    return {"lean": name, "outcome": outcome, "detail": detail, "wall_s": round(wall, 3),
            "exit": proc.returncode, "maxrss_kib": usage.ru_maxrss if usage else None,
            "cpu_s": round(usage.ru_utime + usage.ru_stime, 3) if usage else None,
            "utc": utc, "threads": threads, "limit_s": limit, "memory_mb": memory_mb,
            "load1": os.getloadavg()[0]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("lean_dir"); ap.add_argument("names"); ap.add_argument("out")
    ap.add_argument("--limit", type=float, required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--memory-mb", type=int, default=8192)
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--deadline", default="", help="UTC time HH:MM today after which no new goal starts")
    a = ap.parse_args()
    if a.workers > 8:
        sys.exit("at most 8 Lean processes at once")
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    if a.deadline:
        hh, mm = map(int, a.deadline.split(":"))
        now = time.gmtime()
        DEADLINE[0] = time.time() + ((hh - now.tm_hour) * 3600 + (mm - now.tm_min) * 60 - now.tm_sec)
    os.makedirs(os.path.join(a.out, "logs"), exist_ok=True)
    names = [n.strip() for n in open(a.names) if n.strip()]
    results = os.path.join(a.out, "results.jsonl")
    done = set()
    if os.path.exists(results):
        done = {json.loads(line)["lean"] for line in open(results) if line.strip()}
    todo = [n for n in names if n not in done]
    print(f"{len(names)} goals, {len(done)} already measured, {len(todo)} to run; limit {a.limit} s, "
          f"{a.workers} processes, {a.threads} threads each, -M {a.memory_mb}", flush=True)
    jobs = [(a.lean_dir, n, a.out, a.limit, a.memory_mb, a.threads) for n in todo]
    count = 0
    tally: dict = {}
    with ThreadPoolExecutor(a.workers) as pool, open(results, "a") as out:
        for row in pool.map(run_one, jobs):
            if row is None:
                tally["not started (deadline or STOP)"] = tally.get("not started (deadline or STOP)", 0) + 1
                continue
            out.write(json.dumps(row) + "\n"); out.flush()
            count += 1
            tally[row["outcome"]] = tally.get(row["outcome"], 0) + 1
            if count % 20 == 0 or count == len(jobs):
                print(f"{count}/{len(jobs)} {tally}", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
