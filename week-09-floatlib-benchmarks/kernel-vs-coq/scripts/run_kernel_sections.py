"""Measure the two kernel sections of the figures with FloatLib's timing mode.

    python3 -I run_kernel_sections.py RAW_DIR

`matched` (the Lean side of the loops that Coq with Flocq also runs: `benchmarks/suite/run.sh matched`)
and `exact-native` (`decide +kernel` against `native_decide`: `benchmarks/exact/run.sh native`) run one at
a time through `measure` of `benchmarks/plots/timing.py`, each with a core-scheduling cookie. Their logs
go to RAW_DIR, outside the checkout; no result file of the checkout is written (that is what
`figures.py time` would do). `facts.json` and `sections.log` in RAW_DIR record the day, the toolchain,
the load and the exit code and seconds of each section.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

# A FloatLib checkout with its benchmark workspace; by default the copy Lake keeps for this week.
CHECKOUT = Path(os.environ.get("FLOATLIB_DIR", Path(__file__).resolve().parents[2] / ".lake/packages/floatlib")).resolve()
sys.dont_write_bytecode = True
sys.path.insert(0, str(CHECKOUT / "benchmarks/plots"))
import timing  # noqa: E402

raw = Path(sys.argv[1]).resolve()
raw.mkdir(parents=True, exist_ok=True)
lines = []


def log(message):
    lines.append(message)
    print(message, flush=True)


def mpstat():
    out = subprocess.run(["mpstat", "5", "1"], capture_output=True, text=True).stdout
    return float(out.strip().splitlines()[-1].split()[-1])


started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
before, idle_before = Path("/proc/loadavg").read_text().split()[:3], mpstat()
done = timing.measure(raw, 1, int(os.environ.get("LEAN_NUM_THREADS", "8")), ["matched", "exact-native"], log)
after, idle_after = Path("/proc/loadavg").read_text().split()[:3], mpstat()
version = subprocess.run(["lean", "--version"], capture_output=True, text=True, cwd=CHECKOUT).stdout
facts = {"date": started[:10], "started_utc": started,
         "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
         "lean": version.split("version ")[1].split(",")[0], "core_scheduling_cookie": timing.own_core(),
         "loadavg_before": " ".join(before), "loadavg_after": " ".join(after),
         "machine_idle_percent_before": idle_before, "machine_idle_percent_after": idle_after,
         "floatlib_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                                           cwd=CHECKOUT).stdout.strip(),
         "sections": "; ".join(f"{name}: exit {code}, {seconds:.0f} s" for name, code, seconds in done)}
(raw / "facts.json").write_text(json.dumps(facts, indent=1) + "\n")
(raw / "sections.log").write_text("\n".join(lines) + "\n")
print(json.dumps(facts), flush=True)
sys.exit(0 if all(code == 0 for _, code, _ in done) else 1)
