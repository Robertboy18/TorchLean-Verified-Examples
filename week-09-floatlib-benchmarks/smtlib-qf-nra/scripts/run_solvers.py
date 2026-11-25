"""Run SMT solvers on benchmark files, timed the same way as `run_lean.py` times Lean.

usage: python3 -I run_solvers.py <QF_NRA root> <list of benchmark paths> <out dir> --limit SECONDS
           --solver NAME=COMMAND ... [--workers N] [--memory-mb M]

Each COMMAND is one executable with its options, split at spaces; the benchmark path is appended.  One process
per (solver, benchmark), at most `--workers` at once, each killed at the wall-clock limit; the wall time is that
of the whole process (start to exit), as for Lean.  The answer is the first line of the output that is `sat`,
`unsat` or `unknown`; a killed process is `timeout`, anything else is `error`.  One JSON line per run goes to
<out dir>/results.jsonl, the output of each run to <out dir>/logs/<solver>/<name>.log.
The benchmark files are data: they are passed as paths and never executed.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import resource
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

ACTIVE: set = set()


def stop(signum, frame):
    for pid in list(ACTIVE):
        try:
            os.killpg(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    os._exit(143)


def run_one(job):
    root, rel, solver, command, out, limit, memory_mb = job
    ident = re.sub(r"[^A-Za-z0-9_.-]", "_", rel[:-5].replace("/", "__"))
    log = os.path.join(out, "logs", solver, ident + ".log")
    start = time.monotonic()
    utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    killed = False

    def guard():
        resource.setrlimit(resource.RLIMIT_AS, (memory_mb * 1024 * 1024, memory_mb * 1024 * 1024))

    with open(log, "wb") as stream:
        proc = subprocess.Popen(command + [os.path.join(root, rel)], stdout=stream, stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL, start_new_session=True, preexec_fn=guard)
        ACTIVE.add(proc.pid)
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
            time.sleep(0.005)
        ACTIVE.discard(proc.pid)
        code = os.waitstatus_to_exitcode(status)
    wall = time.monotonic() - start
    text = open(log, encoding="utf-8", errors="replace").read()
    answer = next((line.strip() for line in text.splitlines() if line.strip() in ("sat", "unsat", "unknown")), None)
    if killed:
        result = "timeout"
    elif answer is None:
        result = "error"
    else:
        result = answer
    return {"path": rel, "solver": solver, "result": result, "wall_s": round(wall, 3), "exit": code,
            "cpu_s": round(usage.ru_utime + usage.ru_stime, 3), "maxrss_kib": usage.ru_maxrss, "utc": utc,
            "limit_s": limit, "load1": os.getloadavg()[0],
            "detail": "" if result in ("sat", "unsat", "unknown", "timeout") else text.strip()[:300]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root"); ap.add_argument("paths"); ap.add_argument("out")
    ap.add_argument("--limit", type=float, required=True)
    ap.add_argument("--solver", action="append", required=True, help="NAME=COMMAND")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--memory-mb", type=int, default=8192)
    a = ap.parse_args()
    if a.workers > 8:
        sys.exit("at most 8 processes at once")
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    solvers = [(s.split("=", 1)[0], s.split("=", 1)[1].split(" ")) for s in a.solver]
    for name, _ in solvers:
        os.makedirs(os.path.join(a.out, "logs", name), exist_ok=True)
    paths = [p.strip() for p in open(a.paths) if p.strip()]
    results = os.path.join(a.out, "results.jsonl")
    done = set()
    if os.path.exists(results):
        done = {(r["solver"], r["path"]) for r in map(json.loads, open(results))}
    # the solvers take turns on each benchmark, so that a change of the load of the machine reaches all of them
    jobs = [(a.root, p, name, command, a.out, a.limit, a.memory_mb)
            for p in paths for name, command in solvers if (name, p) not in done]
    print(f"{len(paths)} benchmarks x {len(solvers)} solvers, {len(jobs)} runs to do; limit {a.limit} s, "
          f"{a.workers} processes", flush=True)
    tally: dict = {}
    with ThreadPoolExecutor(a.workers) as pool, open(results, "a") as out:
        for count, row in enumerate(pool.map(run_one, jobs), start=1):
            out.write(json.dumps(row) + "\n"); out.flush()
            key = (row["solver"], row["result"])
            tally[key] = tally.get(key, 0) + 1
            if count % 50 == 0 or count == len(jobs):
                print(f"{count}/{len(jobs)} " + ", ".join(f"{s} {r} {n}" for (s, r), n in sorted(tally.items())),
                      flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
