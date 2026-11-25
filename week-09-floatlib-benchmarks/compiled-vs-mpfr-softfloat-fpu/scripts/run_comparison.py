"""Start the comparison harness on one quiet core and record the machine's load.

    python3 -I run_comparison.py TRIALS OUT_DIR [--widths "32 64 ..."] [--seconds N]

It picks the physical core whose two hardware threads were the least busy over a five-second sample
(niced and not niced work together, then the work that is not niced), pins every measured process to its first thread (`FORMAT_COMPARE_CPU`), and
starts the harness through `shell(..., cookie=True)` of FloatLib's `benchmarks/plots/timing.py`:
the harness and its children then have a core-scheduling cookie of their own, so no other task runs on
the second thread of the core while a measurement runs on the first. While the harness runs, a
thread samples `/proc/stat` every ten seconds (the whole machine, the pinned thread and its sibling)
into `OUT_DIR-load.csv`, and the harness script as generated is kept as `OUT_DIR-harness.sh`. Nothing is
measured here: the times come from the benchmark executables.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

SUITE = Path(__file__).resolve().parents[1]
# A FloatLib 4.1.0 checkout with its benchmark workspace; by default the copy Lake keeps for this week.
CHECKOUT = Path(os.environ.get("FLOATLIB_DIR", SUITE.parent / ".lake/packages/floatlib")).resolve()
WORK = Path(os.environ.get("WORK", SUITE / "work")).resolve()      # generated harness and executable
sys.dont_write_bytecode = True
sys.path.insert(0, str(CHECKOUT / "benchmarks/plots"))
import timing  # noqa: E402  (FloatLib's timing mode: `shell`, `own_core`, the cookie)


def cpu_times():
    """Per-CPU jiffies from /proc/stat: {cpu: (user, nice, system, idle, iowait, irq, softirq, steal)}."""
    table = {}
    for line in Path("/proc/stat").read_text().splitlines():
        fields = line.split()
        if fields[0].startswith("cpu"):
            key = -1 if fields[0] == "cpu" else int(fields[0][3:])
            table[key] = tuple(int(value) for value in fields[1:9])
    return table


def shares(before, after):
    """(not niced, niced, idle) shares of a CPU between two samples."""
    delta = [b - a for a, b in zip(before, after)]
    total = sum(delta) or 1
    user, nice, system, idle, iowait, irq, softirq, steal = delta
    return (user + system + irq + softirq + steal) / total, nice / total, (idle + iowait) / total


def siblings(cpu):
    text = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list").read_text().strip()
    out = []
    for part in text.split(","):
        low, _, high = part.partition("-")
        out.extend(range(int(low), int(high or low) + 1))
    return out


def pick_core(seconds=5.0):
    allowed = os.sched_getaffinity(0)
    before = cpu_times()
    time.sleep(seconds)
    after = cpu_times()
    best = None
    for cpu in sorted(allowed):
        group = siblings(cpu)
        if cpu != group[0] or not all(member in allowed for member in group):
            continue
        work = sum(shares(before[member], after[member])[0] for member in group)
        niced = sum(shares(before[member], after[member])[1] for member in group)
        key = (round(work + niced, 2), work)
        if best is None or key < best[0]:
            best = (key, work, niced, cpu, group)
    return best[1:], shares(before[-1], after[-1])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trials", type=int)
    parser.add_argument("out", type=Path)
    parser.add_argument("--widths", default="32 64 128 256 512 1024 2048 4096")
    parser.add_argument("--operations", default="")
    parser.add_argument("--series", default="binary-public binary-kernel binary-software binary-native-c "
                                             "mpfr-reference softfloat-reference")
    parser.add_argument("--public-widths", default="32 64")
    parser.add_argument("--public-binary", default=str(WORK / "public/publicOperations"))
    parser.add_argument("--harness", default=str(WORK / "format-comparison-public.sh"))
    parser.add_argument("--seconds", type=int, default=7200)
    args = parser.parse_args()
    out = args.out.resolve()
    (work, niced, cpu, group), machine = pick_core()
    cookie = timing.own_core()
    harness = Path(args.harness)
    harness.parent.mkdir(parents=True, exist_ok=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, "-I", str(SUITE / "scripts/make_harness.py"), str(CHECKOUT), str(harness)],
                   check=True)
    shutil.copyfile(harness, f"{out}-harness.sh")      # the exact script of this run, for the record
    env = {**os.environ,
           "FORMAT_COMPARE_ROOT": str(CHECKOUT),
           "FORMAT_COMPARE_PUBLIC_BINARY": args.public_binary,
           "FORMAT_COMPARE_PUBLIC_WIDTHS": args.public_widths,
           "FORMAT_COMPARE_DEVELOPMENT_ONLY": "1",
           "FORMAT_COMPARE_INCLUDE_UNIVERSAL": "0", "FORMAT_COMPARE_INCLUDE_PYTHON": "0",
           "FORMAT_COMPARE_INCLUDE_SOFTFLOAT": "1", "FORMAT_COMPARE_INCLUDE_FLOCQ": "0",
           "FORMAT_COMPARE_LANES": args.series,
           "FORMAT_COMPARE_WIDTHS": args.widths,
           "FORMAT_COMPARE_CPU": str(cpu)}      # SOFTFLOAT_SOURCE and TMPDIR pass through when set
    if args.operations:
        env["FORMAT_COMPARE_OPERATIONS"] = args.operations
    launch = {"started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "trials": args.trials,
              "cpu": cpu, "core_threads": group, "core_scheduling_cookie": cookie,
              "core_not_niced_share_before": round(work, 4), "core_niced_share_before": round(niced, 4),
              "machine_shares_before": dict(zip(("not_niced", "niced", "idle"), (round(v, 4) for v in machine))),
              "loadavg_before": Path("/proc/loadavg").read_text().split()[:3],
              "nice": os.nice(0), "widths": args.widths, "series": env["FORMAT_COMPARE_LANES"]}
    Path(f"{out}-launch.json").write_text(json.dumps(launch, indent=1) + "\n")
    print(json.dumps(launch), flush=True)

    stop = threading.Event()

    def sample():
        with open(f"{out}-load.csv", "w") as stream:
            stream.write("utc,machine_not_niced,machine_niced,machine_idle,cpu_not_niced,cpu_niced,cpu_idle,"
                         "sibling_not_niced,sibling_niced,sibling_idle,load1\n")
            before = cpu_times()
            while not stop.wait(10):
                after = cpu_times()
                row = [time.strftime("%H:%M:%S", time.gmtime())]
                for key in (-1, cpu, group[-1]):
                    row += [f"{value:.4f}" for value in shares(before[key], after[key])]
                row.append(Path("/proc/loadavg").read_text().split()[0])
                stream.write(",".join(row) + "\n")
                stream.flush()
                before = after

    sampler = threading.Thread(target=sample, daemon=True)
    sampler.start()
    command = f"exec bash {harness} {args.trials} {out}"
    code, seconds = timing.shell(command, env=env, log=Path(f"{out}.log"), seconds=args.seconds,
                                 cwd=CHECKOUT, cookie=cookie)
    stop.set()
    sampler.join(timeout=15)
    launch.update(exit=code, seconds=round(seconds, 1),
                  finished_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                  loadavg_after=Path("/proc/loadavg").read_text().split()[:3])
    Path(f"{out}-launch.json").write_text(json.dumps(launch, indent=1) + "\n")
    print(json.dumps(launch), flush=True)
    sys.exit(code)


if __name__ == "__main__":
    main()
