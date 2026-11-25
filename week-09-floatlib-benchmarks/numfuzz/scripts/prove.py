"""Run a plan of Lean cases, one Lean process per case, at most `--workers` at a time.

Usage (under `lake env` of this week's Lake project, so that `lean` sees the built library):
  lake env python3 -I numfuzz/scripts/prove.py PLAN.json OUTDIR [--workers 8]
      [--timeout 300] [--memory 8192] [--only SUBSTRING]

The runner is FloatLib's `benchmarks/suites/Proofs.lean`, or `scripts/ProofsNF.lean` (same protocol:
`FLOATLIB_SUITE_PLAN`, `FLOATLIB_SUITE_OUTPUT`, `FLOATLIB_SUITE_ACTIVE`). A case counts as proved
only when Lean elaborates its commands without error and the named theorem depends on no axiom
other than propext, Classical.choice and Quot.sound. A finished case is not run again, unless it
stopped at a wall-clock limit smaller than the one now given. The key of a case is its name, the
declaration, the commands and the hash of the runner file.
"""
import sys
sys.dont_write_bytecode = True
import argparse, hashlib, json, os, signal, subprocess, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent


def case_key(case, runner_hash):
    """The cache key of a case: what Lean reads (the commands, the declaration it must find, the
    runner file) and the case's name. Planning notes of the scripts are not part of it."""
    text = json.dumps([case["id"], case["declaration"], case["commands"], runner_hash])
    return hashlib.sha256(text.encode()).hexdigest()[:20]


def run_one(case, out, runner, timeout, memory, threads, runner_hash):
    key = case_key(case, runner_hash)
    base = out / ("case-" + key)
    done = base.with_suffix(".result.json")
    if done.exists():
        row = json.loads(done.read_text())
        # A case that ran out of wall-clock time is run again when the limit is now larger.
        if not (row["status"] == "timeout" and row.get("timeout_seconds", 0) < timeout):
            return row
    plan, raw, log, active = (base.with_suffix(s) for s in (".plan.json", ".jsonl", ".log", ".active"))
    plan.write_text(json.dumps([{"id": case["id"], "declaration": case["declaration"],
                                  "commands": case["commands"]}], ensure_ascii=False))
    raw.unlink(missing_ok=True)
    env = {**os.environ, "FLOATLIB_SUITE_PLAN": str(plan), "FLOATLIB_SUITE_OUTPUT": str(raw),
           "FLOATLIB_SUITE_ACTIVE": str(active), "LEAN_NUM_THREADS": str(threads)}
    start = time.monotonic()
    utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    peak, termination = 0, ""
    with log.open("w") as stream:
        process = subprocess.Popen(["lean", "-M", str(memory), str(runner)], env=env, stdout=stream,
                                   stderr=stream, start_new_session=True)
        while process.poll() is None:
            try:
                text = Path(f"/proc/{process.pid}/status").read_text()
                rss = next((int(line.split()[1]) for line in text.splitlines()
                            if line.startswith("VmRSS:")), 0)
                peak = max(peak, rss)
            except (FileNotFoundError, ProcessLookupError):
                pass
            if time.monotonic() - start > timeout:
                termination = "timeout"
                os.killpg(process.pid, signal.SIGKILL)
                break
            time.sleep(0.05)
        code = process.wait()
    elapsed = time.monotonic() - start
    row = None
    if raw.exists():
        for line in raw.read_text().splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                pass
    if row is None:
        tail = log.read_text()[-600:] if log.exists() else ""
        row = {"id": case["id"], "status": termination or "runner_error",
               "reason": f"Lean process exit {code}; {tail}", "elapsed_ns": "", "statement": "",
               "axioms": []}
    row.update(process_seconds=round(elapsed, 3), peak_rss_kib=peak, utc=utc, log=log.name,
               threads=threads, exit_code=code, runner=runner.name, runner_sha256=runner_hash,
               timeout_seconds=timeout)
    done.write_text(json.dumps(row, ensure_ascii=False))
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("plan")
    parser.add_argument("out")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--memory", type=int, default=8192)
    parser.add_argument("--only", default="")
    parser.add_argument("--runner", default=str(HERE / "Proofs.lean"))
    args = parser.parse_args()
    cases = json.loads(Path(args.plan).read_text())
    if args.only:
        cases = [case for case in cases if args.only in case["id"]]
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    workers = max(1, min(8, args.workers))
    budget = int(os.environ.get("LEAN_NUM_THREADS", "8"))
    threads = max(1, min(4, budget // workers))
    runner = Path(args.runner).resolve()
    runner_hash = hashlib.sha256(runner.read_bytes()).hexdigest()
    rows = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(run_one, case, out, runner, args.timeout, args.memory, threads,
                               runner_hash)
                   for case in cases]
        for index, future in enumerate(futures):
            row = future.result()
            rows.append(row)
            seconds = (row["elapsed_ns"] / 1e9) if row.get("elapsed_ns") else float("nan")
            print(f"{index + 1}/{len(cases)} {row['status']:8s} {seconds:8.2f}s "
                  f"{row.get('process_seconds', 0):8.2f}s {row['id']}  "
                  f"{row['reason'][:160].replace(chr(10), ' ')}", flush=True)
    name = "results-" + Path(args.plan).stem + ".json"
    (out / name).write_text(json.dumps(rows, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
