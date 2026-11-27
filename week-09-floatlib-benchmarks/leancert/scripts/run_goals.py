#!/usr/bin/env python3
"""run_goals.py: prove each benchmark goal with LeanCert's tactic and with FloatLib's `interval`.

One Lean process per (goal, tool, configuration, repetition); the same `lean` binary for both
tools (each tool's `lake env` supplies only its LEAN_PATH).  Every file has the same shape:

    import <tool>                       -- `LeanCert.Tactic` or `FloatLib`
    <a five-line `#timed` command>      -- wall time of one command, inside the process
    set_option Elab.async false         -- the kernel check runs inside the timed command
    <file options of the configuration>
    #timed theorem bench_goal : <statement> := by <one tactic call>
    #print axioms bench_goal

A goal counts as proved only if Lean reports no error and `#print axioms` lists nothing beyond
propext, Classical.choice and Quot.sound.

Configurations form a ladder; a rung runs only when the one before did not prove the goal:

  LeanCert, kernel route   kernel  -> kernel-limits -> kernel-strong
  LeanCert, shipped route  native  -> native-limits -> native-strong   (`native_decide`: extra axioms)
  FloatLib                 default -> limits        -> strong

`*-limits` adds `set_option maxRecDepth 100000` and `set_option maxHeartbeats 0` to the file and
keeps the tactic call; `*-strong` keeps those and passes one fixed set of larger tactic options.
Every run has 300 s of wall time; a rung that times out ends its ladder.
"""
import argparse
import concurrent.futures
import json
import os
import pathlib
import re
import signal
import subprocess
import sys
import threading
import time

SUITE = pathlib.Path(__file__).resolve().parents[1]
WORK = pathlib.Path(os.environ.get("WORK", SUITE / "work")).resolve()       # goals, Lean files, run records
# Each tool runs under `lake env` of its own Lake project: the ported LeanCert checkout that `fetch.sh`
# makes, and this week's project, which requires FloatLib.
PROJECT = {"lc": pathlib.Path(os.environ.get("LEANCERT_DIR", WORK / "leancert")).resolve(),
           "fl": pathlib.Path(os.environ.get("FLOATLIB_PROJECT", SUITE.parent)).resolve()}
IMPORT = {"lc": "import LeanCert.Tactic", "fl": "import FloatLib"}
STD = {"propext", "Classical.choice", "Quot.sound"}
LIMITS = "set_option maxRecDepth 100000\nset_option maxHeartbeats 0\n"
TIMED = """open Lean Elab Command in
elab "#timed " c:command : command => do
  let t0 ← IO.monoNanosNow
  elabCommand c
  let t1 ← IO.monoNanosNow
  logInfo m!"TIMED_NS {t1 - t0}"
set_option Elab.async false
"""
LC_STRONG = " (budget := 24) (taylorDepth := 20) (subdivisions := 12) (maxIterations := 10000)"
FL_STRONG = "interval (precision := 128) (degree := 40) (depth := 12) (maxHeartbeats := 4000000)"

LADDERS = {
    ("lc", "kernel"): ["kernel", "kernel-limits", "kernel-strong"],
    ("lc", "native"): ["native", "native-limits", "native-strong"],
    ("fl", "default"): ["default", "limits", "strong"],
}
TIMEOUT = {"kernel": 300, "native": 300, "default": 300, "kernel-limits": 300, "native-limits": 300,
           "limits": 300, "kernel-strong": 300, "native-strong": 300, "strong": 300}


def lc_tactic(goal, config):
    trust = "kernel" if config.startswith("kernel") else "native"
    strong = config.endswith("strong")
    if "lc_native_call" in goal:  # a harvested example: LeanCert's own call, as written
        return None if strong else goal[f"lc_{trust}_call"]
    base = goal["lc_tactic"]
    if base == "decide +kernel":
        return None if strong else ("decide +kernel" if trust == "kernel" else "native_decide")
    if base == "leancert":
        call = "leancert" + (LC_STRONG if strong else "")
    elif base == "interval_decide":
        call = "interval_decide 30" if strong else "interval_decide"
    else:
        if strong:
            return None
        call = base
    return call + (" (trust := kernel)" if trust == "kernel" else "")


def tactic(goal, tool, config):
    if tool == "lc":
        return lc_tactic(goal, config)
    return FL_STRONG if config == "strong" else "interval"


def source(goal, tool, config):
    call = tactic(goal, tool, config)
    if call is None:
        return None
    header = goal.get(f"{tool}_header", "")
    imports = [line for line in header.split("\n") if line.startswith("import ")]
    rest = "\n".join(line for line in header.split("\n") if not line.startswith("import "))
    options = "" if config in ("kernel", "native", "default") else LIMITS
    return (IMPORT[tool] + "\n" + "".join(i + "\n" for i in imports) + TIMED + options + rest +
            f"\n#timed theorem bench_goal : {goal['statement']} := by {call}\n"
            "#print axioms bench_goal\n")


def safe(name):
    return re.sub(r"[^A-Za-z0-9_.+-]", "_", name)


_env_cache = {}
_env_lock = threading.Lock()


def tool_env(tool):
    with _env_lock:
        if tool not in _env_cache:
            out = subprocess.run(["lake", "env", "env", "-0"], cwd=PROJECT[tool],
                                 capture_output=True, check=True).stdout.decode()
            env = dict(item.split("=", 1) for item in out.split("\0") if "=" in item)
            env["LEAN_NUM_THREADS"] = "4"
            _env_cache[tool] = env
        return _env_cache[tool]


def run_file(path, tool, timeout):
    env = tool_env(tool)
    timefile = str(path) + ".time"
    command = ["/usr/bin/time", "-f", "%e %M", "-o", timefile, env["LEAN"], str(path)]
    start = time.monotonic()
    process = subprocess.Popen(command, cwd=PROJECT[tool], env=env, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, start_new_session=True)
    timed_out = False
    try:
        output, _ = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        os.killpg(process.pid, signal.SIGKILL)
        output, _ = process.communicate()
    wall = time.monotonic() - start
    text = output.decode("utf-8", "replace")
    rss = None
    try:
        parts = open(timefile).read().split()
        rss = int(parts[-1])
        os.remove(timefile)
    except (OSError, ValueError, IndexError):
        pass
    return text, wall, rss, process.returncode, timed_out


def classify(text, code, timed_out):
    timed = re.findall(r"TIMED_NS (\d+)", text)
    axioms = None
    match = re.search(r"'bench_goal' depends on axioms: \[(.*?)\]", text, flags=re.S)
    if match:
        axioms = [a.strip() for a in match.group(1).replace("\n", " ").split(",") if a.strip()]
    elif "'bench_goal' does not depend on any axioms" in text:
        axioms = []
    errors = [line for line in text.split("\n") if ": error:" in line or line.startswith("error:")]
    if timed_out:
        status = "timeout"
    elif axioms is None or errors or code != 0:
        status = "failed" if (errors or axioms is not None) else "crash"
    elif "sorryAx" in axioms:
        status = "failed"
    elif set(axioms) <= STD:
        status = "proved"
    else:
        status = "proved-extra-axioms"
    return status, (int(timed[0]) if timed else None), axioms, errors


def record_path(out, goal, tool, config, rep):
    return out / f"{safe(goal['id'])}__{tool}__{config}__r{rep}.json"


def run_one(goal, tool, config, rep, out, files):
    path = record_path(out, goal, tool, config, rep)
    if path.exists():
        return json.load(open(path))
    text = source(goal, tool, config)
    if text is None:
        return None
    lean = files / f"{safe(goal['id'])}__{tool}__{config}.lean"
    lean.write_text(text, encoding="utf-8")
    output, wall, rss, code, timed_out = run_file(lean, tool, TIMEOUT[config])
    status, timed_ns, axioms, errors = classify(output, code, timed_out)
    message = ""
    if status not in ("proved", "proved-extra-axioms"):
        body = "\n".join(line for line in output.split("\n") if "TIMED_NS" not in line)
        message = body.replace(str(lean), "FILE")[:1500]
    record = {"goal": goal["id"], "suite": goal["suite"], "tool": tool, "config": config, "rep": rep,
              "tactic": tactic(goal, tool, config), "status": status, "wall_s": round(wall, 3),
              "timed_s": None if timed_ns is None else timed_ns / 1e9, "max_rss_kb": rss,
              "exit": code, "axioms": axioms, "message": message,
              "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    json.dump(record, open(path, "w"), ensure_ascii=False)
    return record


def run_ladder(goal, tool, ladder, out, files, reps, rep_max_wall):
    last = None
    configs = LADDERS[(tool, ladder)]
    if tool == "fl" and not goal["in_scope"]:
        configs = configs[:1]  # outside `interval`: one plain call, to record what it says
    for config in configs:
        record = run_one(goal, tool, config, 0, out, files)
        if record is None:
            break
        last = record
        if record["status"] in ("proved", "proved-extra-axioms"):
            if record["wall_s"] <= rep_max_wall:
                for rep in range(1, reps):
                    run_one(goal, tool, config, rep, out, files)
            break
        if record["status"] == "timeout":
            break  # a rung that ran out of time is not escalated
    return last


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--goals", default=str(WORK / "goals.json"))
    parser.add_argument("--out", default=str(WORK / "runs"))
    parser.add_argument("--files", default=str(WORK / "files"))
    parser.add_argument("--select", default=".*", help="regex on the goal id")
    parser.add_argument("--ladders", default="lc:kernel,lc:native,fl:default")
    parser.add_argument("--jobs", type=int, default=6)
    parser.add_argument("--reps", type=int, default=1)
    parser.add_argument("--rep-max-wall", type=float, default=60.0)
    parser.add_argument("--baseline", type=int, default=0, help="import-only runs per tool")
    parser.add_argument("--tag", default="", help="suffix for baseline records")
    args = parser.parse_args()
    out, files = pathlib.Path(args.out), pathlib.Path(args.files)
    out.mkdir(parents=True, exist_ok=True)
    files.mkdir(parents=True, exist_ok=True)
    if args.baseline:
        for tool in ("lc", "fl"):
            lean = files / f"baseline__{tool}.lean"
            lean.write_text(IMPORT[tool] + "\n" + TIMED, encoding="utf-8")
            walls = []
            for index in range(args.baseline):
                _, wall, rss, code, _ = run_file(lean, tool, 300)
                walls.append(round(wall, 3))
                print(f"baseline {tool} run {index}: {wall:.2f}s rss={rss}kB exit={code}", flush=True)
            path = out / f"baseline__{tool}{args.tag}.json"
            json.dump({"tool": tool, "walls_s": walls, "max_rss_kb": rss,
                       "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, open(path, "w"))
        return
    goals = [g for g in json.load(open(args.goals)) if re.search(args.select, g["id"])]
    tasks = []
    for goal in goals:
        for spec in args.ladders.split(","):
            tool, ladder = spec.split(":")
            if tool == "fl" and goal.get("lc_only"):
                continue
            tasks.append((goal, tool, ladder))
    print(f"{len(goals)} goals, {len(tasks)} ladders, {args.jobs} at once", flush=True)
    done = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(run_ladder, goal, tool, ladder, out, files, args.reps,
                               args.rep_max_wall): (goal, tool, ladder)
                   for goal, tool, ladder in tasks}
        for future in concurrent.futures.as_completed(futures):
            goal, tool, ladder = futures[future]
            done += 1
            try:
                record = future.result()
            except Exception as error:  # keep going; the missing record shows up in the summary
                print(f"[{done}/{len(tasks)}] {goal['id']} {tool}:{ladder} RUNNER ERROR {error}",
                      flush=True)
                continue
            if record is None:
                continue
            timed = "-" if record["timed_s"] is None else f"{record['timed_s']:.2f}s"
            print(f"[{done}/{len(tasks)}] {goal['id']} {tool}:{record['config']} "
                  f"{record['status']} timed={timed} wall={record['wall_s']:.1f}s", flush=True)


if __name__ == "__main__":
    sys.exit(main())
